"""
Experiment: Inner Loop Learning (Not Backprop)

The hypothesis:
- Outer loop (training): Backprop finds the structure
- Inner loop (inference): Something ELSE operates

This test compares:
1. Standard: Forward pass only (no inner dynamics)
2. Dynamics: State evolves over ticks (CTM-style, no weight changes)
3. Hebbian: Weights change at inference using local rule (no backprop)

Task: Simple pattern completion
- Input: partial pattern
- Output: complete pattern

This is exploratory. Expect confounds.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

print("=" * 70)
print("INNER LOOP LEARNING: Backprop vs Dynamics vs Hebbian")
print("=" * 70)


# =============================================================================
# TASK: Pattern Completion
# =============================================================================

def generate_patterns(n_samples, pattern_len=8, mask_ratio=0.5):
    """
    Generate pattern completion task.
    Pattern: sequence of 0s and 1s with structure
    Input: pattern with some positions masked (-1)
    Target: full pattern
    """
    data = []
    for _ in range(n_samples):
        # Create structured pattern (not random)
        # Rule: alternating with occasional repeats
        pattern = []
        val = random.randint(0, 1)
        for i in range(pattern_len):
            if random.random() < 0.7:  # 70% alternate
                val = 1 - val
            pattern.append(val)

        # Create masked version
        masked = pattern.copy()
        n_mask = int(pattern_len * mask_ratio)
        mask_positions = random.sample(range(pattern_len), n_mask)
        for pos in mask_positions:
            masked[pos] = -1  # -1 = masked

        data.append({
            'input': masked,
            'target': pattern,
            'mask_positions': mask_positions
        })

    return data


# =============================================================================
# MODEL 1: Standard (No inner dynamics)
# =============================================================================

class StandardModel(nn.Module):
    def __init__(self, pattern_len, hidden_dim=32):
        super().__init__()
        # +1 for the mask token (-1 -> 2)
        self.embed = nn.Embedding(3, hidden_dim)
        self.layers = nn.Sequential(
            nn.Linear(hidden_dim * pattern_len, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, pattern_len * 2)  # 2 classes per position
        )
        self.pattern_len = pattern_len

    def forward(self, x):
        # x: (batch, pattern_len) with values in {-1, 0, 1}
        # Map -1 -> 2 for embedding
        x_embed = x.clone()
        x_embed[x_embed == -1] = 2
        h = self.embed(x_embed)  # (batch, pattern_len, hidden_dim)
        h = h.view(h.size(0), -1)  # flatten
        out = self.layers(h)  # (batch, pattern_len * 2)
        return out.view(-1, self.pattern_len, 2)


# =============================================================================
# MODEL 2: Dynamics (State evolves, weights fixed)
# =============================================================================

class DynamicsModel(nn.Module):
    def __init__(self, pattern_len, hidden_dim=32, n_ticks=4):
        super().__init__()
        self.embed = nn.Embedding(3, hidden_dim)
        self.pattern_len = pattern_len
        self.n_ticks = n_ticks
        self.hidden_dim = hidden_dim

        # Dynamics: state update at each tick
        self.dynamics = nn.Sequential(
            nn.Linear(hidden_dim * pattern_len, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim * pattern_len)
        )

        self.output = nn.Linear(hidden_dim, 2)

    def forward(self, x):
        x_embed = x.clone()
        x_embed[x_embed == -1] = 2
        h = self.embed(x_embed)  # (batch, pattern_len, hidden_dim)

        # Run dynamics for n_ticks
        state = h.view(h.size(0), -1)  # (batch, pattern_len * hidden_dim)
        for tick in range(self.n_ticks):
            delta = self.dynamics(state)
            state = state + 0.1 * delta  # Accumulation

        # Output
        state = state.view(-1, self.pattern_len, self.hidden_dim)
        return self.output(state)


# =============================================================================
# MODEL 3: Hebbian (Weights change at inference)
# =============================================================================

class HebbianModel(nn.Module):
    """
    At inference time, uses Hebbian learning to adapt weights.

    Hebbian rule: Δw = η * pre * post
    This is LOCAL - no backprop needed at inference.
    """
    def __init__(self, pattern_len, hidden_dim=32, n_ticks=4, hebbian_lr=0.01):
        super().__init__()
        self.embed = nn.Embedding(3, hidden_dim)
        self.pattern_len = pattern_len
        self.n_ticks = n_ticks
        self.hidden_dim = hidden_dim
        self.hebbian_lr = hebbian_lr

        # These weights will be temporarily modified at inference
        self.W1 = nn.Linear(hidden_dim * pattern_len, hidden_dim * 2)
        self.W2 = nn.Linear(hidden_dim * 2, hidden_dim * pattern_len)

        self.output = nn.Linear(hidden_dim, 2)

    def forward(self, x):
        x_embed = x.clone()
        x_embed[x_embed == -1] = 2
        h = self.embed(x_embed)

        state = h.view(h.size(0), -1)

        # Clone weights for this inference (don't modify originals)
        W1_weight = self.W1.weight.clone()
        W1_bias = self.W1.bias.clone()
        W2_weight = self.W2.weight.clone()
        W2_bias = self.W2.bias.clone()

        for tick in range(self.n_ticks):
            # Forward with current weights
            pre1 = state
            post1 = F.relu(F.linear(pre1, W1_weight, W1_bias))
            pre2 = post1
            post2 = F.linear(pre2, W2_weight, W2_bias)

            # Hebbian update: Δw = η * post ⊗ pre (outer product, averaged over batch)
            # This is the key: LOCAL learning, no backprop
            with torch.no_grad():
                # Update W1: connects state -> hidden
                delta_W1 = torch.einsum('bi,bj->ij', post1, pre1) / pre1.size(0)
                W1_weight = W1_weight + self.hebbian_lr * delta_W1

                # Update W2: connects hidden -> state
                delta_W2 = torch.einsum('bi,bj->ij', post2, pre2) / pre2.size(0)
                W2_weight = W2_weight + self.hebbian_lr * delta_W2

            # Update state with new weights
            state = state + 0.1 * post2

        state = state.view(-1, self.pattern_len, self.hidden_dim)
        return self.output(state)


# =============================================================================
# TRAINING (Outer loop uses backprop for all models)
# =============================================================================

def train_model(model, train_data, n_epochs=100, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(n_epochs):
        model.train()
        total_loss = 0

        # Batch all data
        inputs = torch.tensor([d['input'] for d in train_data])
        targets = torch.tensor([d['target'] for d in train_data])

        optimizer.zero_grad()
        outputs = model(inputs)  # (batch, pattern_len, 2)

        loss = F.cross_entropy(outputs.view(-1, 2), targets.view(-1))
        loss.backward()
        optimizer.step()

        total_loss = loss.item()

    return model


def evaluate(model, data):
    model.eval()
    with torch.no_grad():
        inputs = torch.tensor([d['input'] for d in data])
        targets = torch.tensor([d['target'] for d in data])
        mask_positions = [d['mask_positions'] for d in data]

        outputs = model(inputs)
        preds = outputs.argmax(dim=-1)

        # Overall accuracy
        overall_acc = (preds == targets).float().mean().item()

        # Accuracy only on masked positions (the hard part)
        masked_correct = 0
        masked_total = 0
        for i, positions in enumerate(mask_positions):
            for pos in positions:
                if preds[i, pos] == targets[i, pos]:
                    masked_correct += 1
                masked_total += 1

        masked_acc = masked_correct / masked_total if masked_total > 0 else 0

    return {'overall': overall_acc, 'masked': masked_acc}


# =============================================================================
# EXPERIMENT
# =============================================================================

PATTERN_LEN = 8
HIDDEN_DIM = 32
N_TICKS = 4

print("\n1. Generating data...")
train_data = generate_patterns(500, PATTERN_LEN, mask_ratio=0.5)
test_data = generate_patterns(100, PATTERN_LEN, mask_ratio=0.5)

# Harder test: more masking
test_hard = generate_patterns(100, PATTERN_LEN, mask_ratio=0.75)

print(f"   Train: {len(train_data)}, Test: {len(test_data)}, Test Hard: {len(test_hard)}")

print("\n2. Creating models...")
models = {
    "Standard (no dynamics)": StandardModel(PATTERN_LEN, HIDDEN_DIM),
    "Dynamics (state evolves)": DynamicsModel(PATTERN_LEN, HIDDEN_DIM, N_TICKS),
    "Hebbian (inner learning)": HebbianModel(PATTERN_LEN, HIDDEN_DIM, N_TICKS, hebbian_lr=0.01),
}

for name, model in models.items():
    n_params = sum(p.numel() for p in model.parameters())
    print(f"   {name}: {n_params:,} params")

print("\n3. Training (outer loop: backprop for all)...")
print("-" * 60)

results = {}
for name, model in models.items():
    print(f"\n   Training {name}...")
    model = train_model(model, train_data, n_epochs=200)

    test_result = evaluate(model, test_data)
    hard_result = evaluate(model, test_hard)

    results[name] = {
        'test_overall': test_result['overall'],
        'test_masked': test_result['masked'],
        'hard_overall': hard_result['overall'],
        'hard_masked': hard_result['masked'],
    }

    print(f"   Test: overall={test_result['overall']:.1%}, masked={test_result['masked']:.1%}")
    print(f"   Hard: overall={hard_result['overall']:.1%}, masked={hard_result['masked']:.1%}")

# =============================================================================
# RESULTS
# =============================================================================

print("\n" + "=" * 70)
print("RESULTS")
print("=" * 70)

print(f"\n{'Model':<30} {'Test (masked)':<15} {'Hard (masked)':<15}")
print("-" * 60)
for name in models:
    r = results[name]
    print(f"{name:<30} {r['test_masked']:<15.1%} {r['hard_masked']:<15.1%}")

print("\n" + "=" * 70)
print("INTERPRETATION")
print("=" * 70)

# Compare
std = results["Standard (no dynamics)"]['hard_masked']
dyn = results["Dynamics (state evolves)"]['hard_masked']
heb = results["Hebbian (inner learning)"]['hard_masked']

print(f"""
On hard test (75% masked):
- Standard:  {std:.1%}
- Dynamics:  {dyn:.1%} ({'+' if dyn > std else ''}{(dyn-std)*100:.1f}% vs standard)
- Hebbian:   {heb:.1%} ({'+' if heb > std else ''}{(heb-std)*100:.1f}% vs standard)

Key question: Does inner-loop learning (Hebbian) help beyond just dynamics?
- Dynamics vs Standard: {'+' if dyn > std else ''}{(dyn-std)*100:.1f}%
- Hebbian vs Dynamics:  {'+' if heb > dyn else ''}{(heb-dyn)*100:.1f}%
""")

if heb > dyn:
    print("→ Hebbian inner learning shows improvement over pure dynamics!")
    print("  This suggests inference-time learning (not backprop) might help.")
elif dyn > std:
    print("→ Dynamics help, but Hebbian doesn't add more.")
    print("  Maybe wrong task, or wrong Hebbian rule, or confounded.")
else:
    print("→ Neither dynamics nor Hebbian help on this task.")
    print("  Either task is too simple, or implementation has issues.")

print("\n" + "=" * 70)
print("CAVEATS (Known confounds)")
print("=" * 70)
print("""
1. Task might be too simple (pattern completion can be memorized)
2. Hebbian rule is simplistic (just pre*post, no normalization)
3. Hebbian learning rate is arbitrary
4. Dynamics and Hebbian have more "operations" than Standard
5. Not controlled for compute
""")
