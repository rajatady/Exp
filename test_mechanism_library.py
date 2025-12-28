"""
Experiment 2: General-Purpose Inner Loop Mechanisms

The question: How does a data-driven inner rule generalize to arbitrary tasks?

Hypothesis: We need a LIBRARY of general mechanisms, not one rule per task.
The outer loop learns which mechanisms to use.
The inner loop applies them.

This is like how evolution gave brains general mechanisms (attention, memory, prediction)
that can be configured for specific tasks.

We test:
1. Library of mechanisms (local consistency, global consistency, energy)
2. Outer loop learns weights for each mechanism
3. Test on MULTIPLE DIFFERENT tasks
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
print("GENERAL-PURPOSE INNER LOOP: Mechanism Library")
print("=" * 70)


# =============================================================================
# TASKS: Multiple different structures
# =============================================================================

def generate_alternating_patterns(n_samples, pattern_len=8, mask_ratio=0.5):
    """Task 1: Alternating patterns (70% flip) - LOCAL structure"""
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
            'mask_positions': mask_positions,
            'task': 'alternating'
        })
    return data


def generate_repeat_patterns(n_samples, pattern_len=8, mask_ratio=0.5):
    """Task 2: Repeating patterns (70% repeat) - LOCAL structure but OPPOSITE"""
    data = []
    for _ in range(n_samples):
        pattern = []
        val = random.randint(0, 1)
        for i in range(pattern_len):
            if random.random() < 0.3:  # Only 30% flip = 70% repeat
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
            'mask_positions': mask_positions,
            'task': 'repeating'
        })
    return data


def generate_parity_patterns(n_samples, pattern_len=8, mask_ratio=0.5):
    """Task 3: Parity constraint - GLOBAL structure (all must sum to even)"""
    data = []
    for _ in range(n_samples):
        # Generate random pattern with even parity
        pattern = [random.randint(0, 1) for _ in range(pattern_len - 1)]
        # Last bit makes parity even
        pattern.append(sum(pattern) % 2)

        masked = pattern.copy()
        n_mask = int(pattern_len * mask_ratio)
        mask_positions = random.sample(range(pattern_len), n_mask)
        for pos in mask_positions:
            masked[pos] = -1

        data.append({
            'input': masked,
            'target': pattern,
            'mask_positions': mask_positions,
            'task': 'parity'
        })
    return data


def generate_majority_patterns(n_samples, pattern_len=8, mask_ratio=0.5):
    """Task 4: Majority voting - GLOBAL structure (more 1s or more 0s)"""
    data = []
    for _ in range(n_samples):
        # Decide majority first
        majority = random.randint(0, 1)
        n_majority = random.randint(pattern_len // 2 + 1, pattern_len)

        pattern = [majority] * n_majority + [1 - majority] * (pattern_len - n_majority)
        random.shuffle(pattern)

        masked = pattern.copy()
        n_mask = int(pattern_len * mask_ratio)
        mask_positions = random.sample(range(pattern_len), n_mask)
        for pos in mask_positions:
            masked[pos] = -1

        data.append({
            'input': masked,
            'target': pattern,
            'mask_positions': mask_positions,
            'task': 'majority'
        })
    return data


# =============================================================================
# MECHANISM LIBRARY
# =============================================================================

class LocalConsistencyMechanism(nn.Module):
    """
    Mechanism 1: Neighbors suggest values.
    Good for: spatial/temporal patterns with local structure.
    """
    def __init__(self, flip_neighbors=True):
        super().__init__()
        self.flip = flip_neighbors  # True = alternating, False = repeating
        self.weight = nn.Parameter(torch.tensor(0.3))

    def forward(self, logits):
        probs = F.softmax(logits, dim=-1)  # (batch, seq, 2)

        if self.flip:
            neighbor_probs = probs.flip(-1)  # Suggest opposite
        else:
            neighbor_probs = probs  # Suggest same

        # Shift to get neighbor suggestions
        left = F.pad(neighbor_probs[:, :-1, :], (0, 0, 1, 0))
        right = F.pad(neighbor_probs[:, 1:, :], (0, 0, 0, 1))
        neighbor_suggest = (left + right) / 2

        # Blend
        updated = (1 - self.weight) * probs + self.weight * neighbor_suggest
        return torch.log(updated + 1e-8)


class GlobalConsistencyMechanism(nn.Module):
    """
    Mechanism 2: All positions should agree on some global property.
    Good for: parity, majority voting, global constraints.
    """
    def __init__(self, pattern_len):
        super().__init__()
        self.pattern_len = pattern_len
        self.weight = nn.Parameter(torch.tensor(0.2))
        # Learn what global property to track
        self.global_proj = nn.Linear(pattern_len * 2, pattern_len * 2)

    def forward(self, logits):
        probs = F.softmax(logits, dim=-1)  # (batch, seq, 2)
        batch_size = probs.size(0)

        # Compute global signal
        flat_probs = probs.view(batch_size, -1)  # (batch, seq*2)
        global_signal = self.global_proj(flat_probs)  # (batch, seq*2)
        global_signal = global_signal.view(batch_size, -1, 2)  # (batch, seq, 2)
        global_probs = F.softmax(global_signal, dim=-1)

        # Blend with global
        updated = (1 - self.weight) * probs + self.weight * global_probs
        return torch.log(updated + 1e-8)


class EnergyMinimizationMechanism(nn.Module):
    """
    Mechanism 3: Settle to low-energy state.
    Good for: associative memory, constraint satisfaction.
    """
    def __init__(self, pattern_len, hidden_dim=32):
        super().__init__()
        self.pattern_len = pattern_len
        # Hopfield-like energy: E = -0.5 * h^T W h
        self.W = nn.Parameter(torch.randn(pattern_len, pattern_len) * 0.1)
        self.step_size = nn.Parameter(torch.tensor(0.1))

    def forward(self, logits):
        probs = F.softmax(logits, dim=-1)  # (batch, seq, 2)

        # Use prob of class 1 as "activation"
        h = probs[:, :, 1]  # (batch, seq)

        # Energy gradient: dE/dh = -W @ h
        # Update: h_new = h - step * dE/dh = h + step * W @ h
        W_sym = (self.W + self.W.T) / 2  # Symmetric for Hopfield
        delta = torch.matmul(h, W_sym)  # (batch, seq)
        h_new = h + self.step_size * torch.tanh(delta)
        h_new = torch.clamp(h_new, 0, 1)

        # Convert back to logits
        new_probs = torch.stack([1 - h_new, h_new], dim=-1)
        return torch.log(new_probs + 1e-8)


# =============================================================================
# MODEL: Transformer + Mechanism Library
# =============================================================================

class TransformerWithMechanisms(nn.Module):
    """
    Transformer that learns to use a library of inner-loop mechanisms.

    The outer loop (backprop) learns:
    - Which mechanisms to engage (mechanism_weights)
    - How to parameterize each mechanism

    The inner loop (inference) applies:
    - Weighted combination of mechanisms
    - For multiple ticks
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
        self.output = nn.Linear(hidden_dim, 2)

        # Mechanism library
        self.mechanisms = nn.ModuleDict({
            'local_flip': LocalConsistencyMechanism(flip_neighbors=True),
            'local_same': LocalConsistencyMechanism(flip_neighbors=False),
            'global': GlobalConsistencyMechanism(pattern_len),
            'energy': EnergyMinimizationMechanism(pattern_len),
        })

        # Learn which mechanisms to use (outer loop configures this)
        self.mechanism_weights = nn.Parameter(torch.zeros(len(self.mechanisms)))

        self.n_ticks = n_ticks
        self.pattern_len = pattern_len

    def forward(self, x, return_weights=False):
        x_embed = x.clone()
        x_embed[x_embed == -1] = 2
        h = self.embed(x_embed) + self.pos_embed
        h = self.transformer(h)
        logits = self.output(h)

        # Get mechanism weights (softmax to sum to 1)
        weights = F.softmax(self.mechanism_weights, dim=0)

        # Apply mechanisms for n_ticks
        for tick in range(self.n_ticks):
            # Compute each mechanism's output
            mechanism_outputs = []
            for name, mechanism in self.mechanisms.items():
                out = mechanism(logits)
                mechanism_outputs.append(out)

            # Weighted combination
            stacked = torch.stack(mechanism_outputs, dim=0)  # (n_mechanisms, batch, seq, 2)
            logits = (weights.view(-1, 1, 1, 1) * stacked).sum(dim=0)

        if return_weights:
            return logits, weights
        return logits


class TransformerBaseline(nn.Module):
    """Baseline: no inner loop mechanisms"""
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


def get_mechanism_weights(model):
    if hasattr(model, 'mechanism_weights'):
        weights = F.softmax(model.mechanism_weights, dim=0)
        return {name: weights[i].item()
                for i, name in enumerate(model.mechanisms.keys())}
    return None


# =============================================================================
# EXPERIMENT
# =============================================================================

PATTERN_LEN = 8
HIDDEN_DIM = 64

print("\n1. Generating data for DIFFERENT tasks...")

tasks = {
    'Alternating (local flip)': generate_alternating_patterns,
    'Repeating (local same)': generate_repeat_patterns,
    'Parity (global)': generate_parity_patterns,
    'Majority (global)': generate_majority_patterns,
}

task_data = {}
for name, gen_fn in tasks.items():
    train = gen_fn(500, PATTERN_LEN, mask_ratio=0.5)
    test = gen_fn(100, PATTERN_LEN, mask_ratio=0.75)
    task_data[name] = {'train': train, 'test': test}
    print(f"   {name}: {len(train)} train, {len(test)} test")

print("\n2. Training models on each task...")
print("=" * 70)

all_results = {}

for task_name in tasks:
    print(f"\n--- Task: {task_name} ---")

    train = task_data[task_name]['train']
    test = task_data[task_name]['test']

    # Create fresh models for each task
    baseline = TransformerBaseline(PATTERN_LEN, HIDDEN_DIM)
    mechanism_model = TransformerWithMechanisms(PATTERN_LEN, HIDDEN_DIM, n_ticks=4)

    # Train baseline
    baseline = train_model(baseline, train, n_epochs=200)
    base_acc = evaluate(baseline, test)

    # Train mechanism model
    mechanism_model = train_model(mechanism_model, train, n_epochs=200)
    mech_acc = evaluate(mechanism_model, test)

    # Get learned mechanism weights
    weights = get_mechanism_weights(mechanism_model)

    all_results[task_name] = {
        'baseline': base_acc,
        'mechanisms': mech_acc,
        'weights': weights
    }

    print(f"   Baseline: {base_acc:.1%}")
    print(f"   Mechanisms: {mech_acc:.1%} ({'+' if mech_acc > base_acc else ''}{(mech_acc-base_acc)*100:.1f}%)")
    if weights:
        print(f"   Learned weights: {', '.join(f'{k}={v:.2f}' for k,v in weights.items())}")

print("\n" + "=" * 70)
print("SUMMARY: Does the library learn to select appropriate mechanisms?")
print("=" * 70)

print(f"\n{'Task':<25} {'Baseline':<12} {'Mechanisms':<12} {'Best Mechanism':<20}")
print("-" * 70)

for task_name, results in all_results.items():
    best_mech = max(results['weights'].items(), key=lambda x: x[1]) if results['weights'] else ('N/A', 0)
    print(f"{task_name:<25} {results['baseline']:<12.1%} {results['mechanisms']:<12.1%} {best_mech[0]} ({best_mech[1]:.2f})")

print("\n" + "=" * 70)
print("INTERPRETATION")
print("=" * 70)

# Check if mechanism selection matches task structure
expected = {
    'Alternating (local flip)': 'local_flip',
    'Repeating (local same)': 'local_same',
    'Parity (global)': 'global',
    'Majority (global)': 'global',
}

matches = 0
for task_name, exp_mech in expected.items():
    if task_name in all_results and all_results[task_name]['weights']:
        actual_best = max(all_results[task_name]['weights'].items(), key=lambda x: x[1])[0]
        if actual_best == exp_mech:
            matches += 1
            print(f"✓ {task_name}: Correctly selected {exp_mech}")
        else:
            print(f"✗ {task_name}: Expected {exp_mech}, got {actual_best}")

print(f"\nMechanism selection accuracy: {matches}/{len(expected)} ({matches/len(expected):.0%})")

print("""

KEY INSIGHT:
-----------
If the model learns to select local_flip for alternating patterns and
local_same for repeating patterns, this shows the outer loop can learn
to configure which inner loop mechanism to use.

This is how a general-purpose system might work:
1. Library of mechanisms (attention, consistency, energy, etc.)
2. Outer loop learns which to engage for which task
3. Inner loop applies the selected mechanisms

Like how evolution gave brains general mechanisms that experience configures.
""")
