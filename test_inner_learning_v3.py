"""
Experiment 1c: Learned Inner Learning Rule

Instead of pre-defined Hebbian, let the model LEARN how to update itself.

The idea:
- Outer loop (backprop): Learns the model AND the inner learning rule
- Inner loop (learned rule): Applies that rule at inference

This is closer to how evolution found brains with their own learning rules.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import math

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

print("=" * 70)
print("LEARNED INNER LEARNING RULE")
print("=" * 70)


# =============================================================================
# TASK: Pattern Completion (same task)
# =============================================================================

def generate_patterns(n_samples, pattern_len=8, mask_ratio=0.5):
    data = []
    for _ in range(n_samples):
        pattern = []
        val = random.randint(0, 1)
        for i in range(pattern_len):
            if random.random() < 0.7:
                val = 1 - val
            pattern.append(val)

        masked = pattern.copy()
        n_mask = int(pattern_len * mask_ratio)
        mask_positions = random.sample(range(pattern_len), n_mask)
        for pos in mask_positions:
            masked[pos] = -1

        data.append({
            'input': masked,
            'target': pattern,
            'mask_positions': mask_positions
        })
    return data


# =============================================================================
# MODEL 1: Transformer Baseline
# =============================================================================

class TransformerBaseline(nn.Module):
    def __init__(self, pattern_len, hidden_dim=64, n_heads=4, n_layers=2):
        super().__init__()
        self.embed = nn.Embedding(3, hidden_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, pattern_len, hidden_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads,
            dim_feedforward=hidden_dim*4, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.output = nn.Linear(hidden_dim, 2)

    def forward(self, x):
        x_embed = x.clone()
        x_embed[x_embed == -1] = 2
        h = self.embed(x_embed) + self.pos_embed
        h = self.transformer(h)
        return self.output(h)


# =============================================================================
# MODEL 2: Learned Update Rule
# =============================================================================

class LearnedUpdateRule(nn.Module):
    """
    A small network that takes current state and input,
    and outputs how to update the state.

    This is the "inner learning rule" - learned by backprop,
    but applied at inference time.
    """
    def __init__(self, hidden_dim):
        super().__init__()
        # Input: current state (h) + original input embedding (x)
        # Output: delta to apply to state
        self.rule = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),  # Bounded output
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )
        # Learnable step size
        self.step_size = nn.Parameter(torch.tensor(0.1))

    def forward(self, h, x_embed):
        # h: current state (batch, seq, hidden)
        # x_embed: original input embedding (batch, seq, hidden)
        combined = torch.cat([h, x_embed], dim=-1)
        delta = self.rule(combined)
        return self.step_size * delta


class TransformerLearnedRule(nn.Module):
    """
    Transformer that uses a LEARNED update rule at inference time.

    The update rule is trained by backprop (outer loop)
    but applied iteratively at inference (inner loop).
    """
    def __init__(self, pattern_len, hidden_dim=64, n_heads=4, n_layers=2, n_ticks=4):
        super().__init__()
        self.embed = nn.Embedding(3, hidden_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, pattern_len, hidden_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads,
            dim_feedforward=hidden_dim*4, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # The learned inner update rule
        self.update_rule = LearnedUpdateRule(hidden_dim)

        self.output = nn.Linear(hidden_dim, 2)
        self.n_ticks = n_ticks

        # For analysis
        self.state_deltas = None

    def forward(self, x, return_analysis=False):
        x_embed_raw = x.clone()
        x_embed_raw[x_embed_raw == -1] = 2
        x_embed = self.embed(x_embed_raw) + self.pos_embed

        # Initial state from transformer
        h = self.transformer(x_embed)

        deltas = []

        # Apply learned update rule for n_ticks
        for tick in range(self.n_ticks):
            # Get update from learned rule
            delta = self.update_rule(h, x_embed)
            h = h + delta

            deltas.append(delta.norm(dim=-1).mean().item())

        self.state_deltas = deltas
        out = self.output(h)

        if return_analysis:
            return out, deltas
        return out


# =============================================================================
# MODEL 3: Learned Rule with Convergence Signal
# =============================================================================

class LearnedUpdateRuleWithConvergence(nn.Module):
    """
    Learned rule that also outputs a "done" signal.
    When done > 0.5, stop updating.
    """
    def __init__(self, hidden_dim):
        super().__init__()
        self.rule = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim + 1),  # +1 for done signal
        )
        self.step_size = nn.Parameter(torch.tensor(0.1))

    def forward(self, h, x_embed):
        combined = torch.cat([h, x_embed], dim=-1)
        out = self.rule(combined)
        delta = torch.tanh(out[..., :-1]) * self.step_size
        done = torch.sigmoid(out[..., -1:])  # (batch, seq, 1)
        return delta, done


class TransformerLearnedRuleConverge(nn.Module):
    """
    Like above but with adaptive halting -
    the model learns when to stop iterating.
    """
    def __init__(self, pattern_len, hidden_dim=64, n_heads=4, n_layers=2, max_ticks=8):
        super().__init__()
        self.embed = nn.Embedding(3, hidden_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, pattern_len, hidden_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads,
            dim_feedforward=hidden_dim*4, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.update_rule = LearnedUpdateRuleWithConvergence(hidden_dim)
        self.output = nn.Linear(hidden_dim, 2)
        self.max_ticks = max_ticks

        self.halting_stats = None

    def forward(self, x, return_analysis=False):
        x_embed_raw = x.clone()
        x_embed_raw[x_embed_raw == -1] = 2
        x_embed = self.embed(x_embed_raw) + self.pos_embed

        h = self.transformer(x_embed)

        # Track when each position halts
        halted = torch.zeros_like(h[..., 0])  # (batch, seq)
        final_h = torch.zeros_like(h)

        deltas = []
        halt_counts = []

        for tick in range(self.max_ticks):
            delta, done = self.update_rule(h, x_embed)
            done = done.squeeze(-1)  # (batch, seq)

            # Update only non-halted positions
            still_running = 1 - halted
            h = h + delta * still_running.unsqueeze(-1)

            # Accumulate final state weighted by halt probability
            new_halts = (done > 0.5).float() * still_running
            final_h = final_h + h * new_halts.unsqueeze(-1)
            halted = halted + new_halts

            deltas.append(delta.norm(dim=-1).mean().item())
            halt_counts.append(halted.mean().item())

            # Early exit if all halted
            if halted.all():
                break

        # For any position that never halted, use final state
        never_halted = 1 - halted
        final_h = final_h + h * never_halted.unsqueeze(-1)

        self.halting_stats = {'deltas': deltas, 'halt_progress': halt_counts}
        out = self.output(final_h)

        if return_analysis:
            return out, self.halting_stats
        return out


# =============================================================================
# TRAINING & EVALUATION
# =============================================================================

def train_model(model, train_data, n_epochs=100, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(n_epochs):
        model.train()
        inputs = torch.tensor([d['input'] for d in train_data])
        targets = torch.tensor([d['target'] for d in train_data])

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = F.cross_entropy(outputs.view(-1, 2), targets.view(-1))
        loss.backward()
        optimizer.step()

    return model


def evaluate(model, data):
    model.eval()
    with torch.no_grad():
        inputs = torch.tensor([d['input'] for d in data])
        targets = torch.tensor([d['target'] for d in data])
        mask_positions = [d['mask_positions'] for d in data]

        outputs = model(inputs)
        preds = outputs.argmax(dim=-1)

        masked_correct = 0
        masked_total = 0
        for i, positions in enumerate(mask_positions):
            for pos in positions:
                if preds[i, pos] == targets[i, pos]:
                    masked_correct += 1
                masked_total += 1

        return masked_correct / masked_total if masked_total > 0 else 0


def analyze_model(model, data, name):
    model.eval()
    print(f"\n--- Analysis: {name} ---")

    inputs = torch.tensor([d['input'] for d in data[:10]])

    with torch.no_grad():
        if hasattr(model, 'state_deltas'):
            outputs, deltas = model(inputs, return_analysis=True)
            print("Update magnitudes per tick:")
            for i, d in enumerate(deltas):
                print(f"  Tick {i+1}: Δ = {d:.4f}")

        if hasattr(model, 'halting_stats') and model.halting_stats:
            stats = model.halting_stats
            print("Halting progress:")
            for i, (d, h) in enumerate(zip(stats['deltas'], stats['halt_progress'])):
                print(f"  Tick {i+1}: Δ = {d:.4f}, halted = {h:.1%}")


# =============================================================================
# EXPERIMENT
# =============================================================================

PATTERN_LEN = 8
HIDDEN_DIM = 64

print("\n1. Generating data...")
train_data = generate_patterns(500, PATTERN_LEN, mask_ratio=0.5)
test_data = generate_patterns(100, PATTERN_LEN, mask_ratio=0.5)
test_hard = generate_patterns(100, PATTERN_LEN, mask_ratio=0.75)

print(f"   Train: {len(train_data)}, Test: {len(test_data)}, Hard: {len(test_hard)}")

print("\n2. Creating models...")
models = {
    "Transformer": TransformerBaseline(PATTERN_LEN, HIDDEN_DIM),
    "Learned Rule": TransformerLearnedRule(PATTERN_LEN, HIDDEN_DIM, n_ticks=4),
    "Learned+Converge": TransformerLearnedRuleConverge(PATTERN_LEN, HIDDEN_DIM, max_ticks=8),
}

for name, model in models.items():
    n_params = sum(p.numel() for p in model.parameters())
    print(f"   {name}: {n_params:,} params")

print("\n3. Training...")
results = {}
for name, model in models.items():
    print(f"   Training {name}...")
    model = train_model(model, train_data, n_epochs=200)

    test_acc = evaluate(model, test_data)
    hard_acc = evaluate(model, test_hard)

    results[name] = {'test': test_acc, 'hard': hard_acc}
    print(f"   → Test: {test_acc:.1%}, Hard: {hard_acc:.1%}")

print("\n" + "=" * 70)
print("RESULTS")
print("=" * 70)
print(f"\n{'Model':<25} {'Test (50%)':<15} {'Hard (75%)':<15}")
print("-" * 55)
for name in models:
    r = results[name]
    print(f"{name:<25} {r['test']:<15.1%} {r['hard']:<15.1%}")

print("\n" + "=" * 70)
print("HOW EACH MODEL SOLVES THE TASK")
print("=" * 70)

for name, model in models.items():
    analyze_model(model, test_data, name)

print("\n" + "=" * 70)
print("INTERPRETATION")
print("=" * 70)

base = results["Transformer"]['hard']
learned = results["Learned Rule"]['hard']
converge = results["Learned+Converge"]['hard']

print(f"""
On hard test (75% masked):
- Transformer:       {base:.1%}
- Learned Rule:      {learned:.1%} ({'+' if learned > base else ''}{(learned-base)*100:.1f}%)
- Learned+Converge:  {converge:.1%} ({'+' if converge > base else ''}{(converge-base)*100:.1f}%)

Key question: Does a LEARNED inner rule help vs pre-defined Hebbian?
(Hebbian from v2 was: 49.2% - worse than baseline)
""")
