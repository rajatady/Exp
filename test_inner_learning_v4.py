"""
Experiment 1d: Data-Driven Inner Rule

Not meta-learning (black box).
Not hand-crafted (Hebbian).

Instead: Look at the DATA and derive what rule makes sense.

The data: alternating patterns (70% flip, 30% repeat)
The insight: predictions should be CONSISTENT with neighbors

The rule:
1. Make initial prediction
2. Check consistency with neighbors
3. Update toward consistency
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
print("DATA-DRIVEN INNER RULE")
print("=" * 70)


# =============================================================================
# TASK: Pattern Completion
# =============================================================================

def generate_patterns(n_samples, pattern_len=8, mask_ratio=0.5):
    data = []
    for _ in range(n_samples):
        pattern = []
        val = random.randint(0, 1)
        for i in range(pattern_len):
            if random.random() < 0.7:  # 70% flip
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
# MODEL 2: Consistency-Based Update (Data-Driven Rule)
# =============================================================================

class ConsistencyRule(nn.Module):
    """
    Data-driven inner rule based on the structure of alternating patterns.
    VECTORIZED for speed.
    """
    def __init__(self, hidden_dim, pattern_len):
        super().__init__()
        self.pattern_len = pattern_len
        self.neighbor_weight = nn.Parameter(torch.tensor(0.3))

    def forward(self, logits, mask_positions_batch=None):
        """Vectorized consistency update"""
        probs = F.softmax(logits, dim=-1)  # (batch, seq, 2)

        # Flip probs: if neighbor is [p0, p1], suggestion is [p1, p0]
        flipped = probs.flip(-1)  # (batch, seq, 2)

        # Shift left and right to get neighbor suggestions
        left_suggest = F.pad(flipped[:, :-1, :], (0, 0, 1, 0))  # shift right
        right_suggest = F.pad(flipped[:, 1:, :], (0, 0, 0, 1))  # shift left

        # Average neighbor suggestions
        neighbor_suggest = (left_suggest + right_suggest) / 2

        # Blend with current
        updated_probs = (1 - self.neighbor_weight) * probs + self.neighbor_weight * neighbor_suggest

        # Back to logits
        return torch.log(updated_probs + 1e-8)


class TransformerConsistency(nn.Module):
    """
    Transformer + data-driven consistency rule
    """
    def __init__(self, pattern_len, hidden_dim=64, n_heads=4, n_layers=2, n_ticks=4):
        super().__init__()
        self.embed = nn.Embedding(3, hidden_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, pattern_len, hidden_dim) * 0.02)
        self.pattern_len = pattern_len

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads,
            dim_feedforward=hidden_dim*4, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.output = nn.Linear(hidden_dim, 2)

        self.consistency_rule = ConsistencyRule(hidden_dim, pattern_len)
        self.n_ticks = n_ticks

    def forward(self, x, mask_positions_batch=None):
        x_embed = x.clone()
        x_embed[x_embed == -1] = 2
        h = self.embed(x_embed) + self.pos_embed
        h = self.transformer(h)
        logits = self.output(h)

        # If no mask positions provided, infer from input
        if mask_positions_batch is None:
            mask_positions_batch = []
            for i in range(x.size(0)):
                mask_positions_batch.append((x[i] == -1).nonzero().squeeze(-1).tolist())
                if isinstance(mask_positions_batch[-1], int):
                    mask_positions_batch[-1] = [mask_positions_batch[-1]]

        # Apply consistency rule for n_ticks
        for tick in range(self.n_ticks):
            logits = self.consistency_rule(logits, mask_positions_batch)

        return logits


# =============================================================================
# MODEL 3: Prediction-Refinement (Another Data-Driven Approach)
# =============================================================================

class PredictionRefinement(nn.Module):
    """
    Another data-driven approach:

    1. Make initial prediction
    2. Treat prediction AS IF it were true input
    3. Re-run the model to refine

    This is like asking: "Given my current guess, what would I predict?"
    Iterate until stable.
    """
    def __init__(self, pattern_len, hidden_dim=64, n_heads=4, n_layers=2, n_ticks=4):
        super().__init__()
        self.embed = nn.Embedding(3, hidden_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, pattern_len, hidden_dim) * 0.02)
        self.pattern_len = pattern_len

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads,
            dim_feedforward=hidden_dim*4, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.output = nn.Linear(hidden_dim, 2)
        self.n_ticks = n_ticks

        self.prediction_changes = None

    def forward(self, x, return_analysis=False):
        original_input = x.clone()
        current_input = x.clone()

        changes = []

        for tick in range(self.n_ticks):
            # Embed current input (with predictions filled in)
            x_embed = current_input.clone()
            x_embed[x_embed == -1] = 2
            h = self.embed(x_embed) + self.pos_embed
            h = self.transformer(h)
            logits = self.output(h)

            # Get predictions
            preds = logits.argmax(dim=-1)

            # Track how much predictions changed
            if tick > 0:
                change = (preds != prev_preds).float().mean().item()
                changes.append(change)
            prev_preds = preds.clone()

            # Fill in masked positions with predictions for next iteration
            mask = (original_input == -1)
            current_input = original_input.clone()
            current_input[mask] = preds[mask]

        self.prediction_changes = changes

        if return_analysis:
            return logits, changes
        return logits


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

    with torch.no_grad():
        if hasattr(model, 'consistency_rule'):
            print(f"Learned neighbor_weight: {model.consistency_rule.neighbor_weight.item():.3f}")

        if hasattr(model, 'prediction_changes'):
            inputs = torch.tensor([d['input'] for d in data[:10]])
            outputs, changes = model(inputs, return_analysis=True)
            if changes:
                print("Prediction changes per tick:")
                for i, c in enumerate(changes):
                    print(f"  Tick {i+2}: {c:.1%} changed")


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
print(f"   Data structure: ~70% alternating, ~30% repeating")

print("\n2. Creating models...")
models = {
    "Transformer": TransformerBaseline(PATTERN_LEN, HIDDEN_DIM),
    "Consistency Rule": TransformerConsistency(PATTERN_LEN, HIDDEN_DIM, n_ticks=4),
    "Self-Refinement": PredictionRefinement(PATTERN_LEN, HIDDEN_DIM, n_ticks=4),
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
print("WHAT THE MODEL LEARNED")
print("=" * 70)

for name, model in models.items():
    analyze_model(model, test_data, name)

print("\n" + "=" * 70)
print("INTERPRETATION")
print("=" * 70)

base = results["Transformer"]['hard']
consist = results["Consistency Rule"]['hard']
refine = results["Self-Refinement"]['hard']

print(f"""
On hard test (75% masked):
- Transformer:       {base:.1%}
- Consistency Rule:  {consist:.1%} ({'+' if consist > base else ''}{(consist-base)*100:.1f}%)
- Self-Refinement:   {refine:.1%} ({'+' if refine > base else ''}{(refine-base)*100:.1f}%)

The rules are DATA-DRIVEN:
- Consistency: Uses the alternating pattern structure
- Self-Refinement: Uses its own predictions as input

Previous results for comparison:
- Hebbian (hand-crafted): 49.2%
- Learned rule (meta): 53.3%
""")
