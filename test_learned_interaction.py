"""
Experiment 2b: Learned Interaction (Continuous Inner Loop)

The question: Can we have ONE mechanism that smoothly handles all cases?

Instead of discrete mechanism selection, we have:
- Learned pairwise interaction weights W[i,j]
- Learned transform (how one position's state influences another)
- Backprop configures both

This subsumes:
- Local flip: W is tridiagonal, transform flips
- Local same: W is tridiagonal, transform is identity
- Global: W has uniform rows
- Hopfield: W is arbitrary symmetric

NO DISCRETE SELECTION. Just learned interactions.
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
print("LEARNED INTERACTION: One Continuous Mechanism")
print("=" * 70)


# =============================================================================
# TASKS (same as before)
# =============================================================================

def generate_alternating_patterns(n_samples, pattern_len=8, mask_ratio=0.5):
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
        data.append({'input': masked, 'target': pattern, 'mask_positions': mask_positions, 'task': 'alternating'})
    return data


def generate_repeat_patterns(n_samples, pattern_len=8, mask_ratio=0.5):
    data = []
    for _ in range(n_samples):
        pattern = []
        val = random.randint(0, 1)
        for i in range(pattern_len):
            if random.random() < 0.3:  # 70% repeat
                val = 1 - val
            pattern.append(val)
        masked = pattern.copy()
        n_mask = int(pattern_len * mask_ratio)
        mask_positions = random.sample(range(pattern_len), n_mask)
        for pos in mask_positions:
            masked[pos] = -1
        data.append({'input': masked, 'target': pattern, 'mask_positions': mask_positions, 'task': 'repeating'})
    return data


def generate_parity_patterns(n_samples, pattern_len=8, mask_ratio=0.5):
    data = []
    for _ in range(n_samples):
        pattern = [random.randint(0, 1) for _ in range(pattern_len - 1)]
        pattern.append(sum(pattern) % 2)
        masked = pattern.copy()
        n_mask = int(pattern_len * mask_ratio)
        mask_positions = random.sample(range(pattern_len), n_mask)
        for pos in mask_positions:
            masked[pos] = -1
        data.append({'input': masked, 'target': pattern, 'mask_positions': mask_positions, 'task': 'parity'})
    return data


# =============================================================================
# LEARNED INTERACTION (One Continuous Mechanism)
# =============================================================================

class LearnedInteraction(nn.Module):
    """
    One continuous mechanism that can learn ANY interaction pattern.

    Instead of discrete selection between mechanisms, we learn:
    1. W[i,j]: how much position j influences position i
    2. transform: how to convert state[j] to influence on i

    Special cases:
    - W tridiagonal + transform flips = alternating
    - W tridiagonal + transform identity = repeating
    - W uniform + transform identity = global averaging
    - W arbitrary = any pattern
    """
    def __init__(self, pattern_len, transform_hidden=16):
        super().__init__()
        self.pattern_len = pattern_len

        # Learned pairwise interactions
        # Initialize near-identity (each position influences itself mostly)
        self.W = nn.Parameter(torch.eye(pattern_len) + 0.1 * torch.randn(pattern_len, pattern_len))

        # Learned transform: how position j's state becomes influence on i
        # This can learn: identity (same), flip, or anything else
        self.transform = nn.Sequential(
            nn.Linear(2, transform_hidden),
            nn.Tanh(),
            nn.Linear(transform_hidden, 2)
        )

        # How much to blend with original
        self.blend = nn.Parameter(torch.tensor(0.3))

    def forward(self, logits):
        probs = F.softmax(logits, dim=-1)  # (batch, seq, 2)
        batch_size, seq_len, _ = probs.shape

        # Transform each position's state
        transformed = self.transform(probs)  # (batch, seq, 2)

        # Weighted influence from all positions
        # influence[i] = sum_j W[i,j] * transformed[j]
        # Using einsum for clarity
        influence = torch.einsum('ij,bj...->bi...', F.softmax(self.W, dim=1), transformed)

        # Normalize to probabilities
        influence_probs = F.softmax(influence, dim=-1)

        # Blend with current
        blend_weight = torch.sigmoid(self.blend)
        updated = (1 - blend_weight) * probs + blend_weight * influence_probs

        return torch.log(updated + 1e-8)

    def analyze(self):
        """Analyze what the model learned"""
        W = F.softmax(self.W, dim=1).detach()

        # Check if W is local (tridiagonal-ish)
        diagonal_mass = torch.diag(W).sum() / W.sum()
        neighbor_mask = torch.zeros_like(W)
        for i in range(self.pattern_len):
            if i > 0:
                neighbor_mask[i, i-1] = 1
            if i < self.pattern_len - 1:
                neighbor_mask[i, i+1] = 1
        neighbor_mass = (W * neighbor_mask).sum() / W.sum()

        # Check transform: does it flip or preserve?
        test_0 = torch.tensor([[1.0, 0.0]])  # prob of 0
        test_1 = torch.tensor([[0.0, 1.0]])  # prob of 1
        with torch.no_grad():
            out_0 = F.softmax(self.transform(test_0), dim=-1)
            out_1 = F.softmax(self.transform(test_1), dim=-1)

        # If transform(0) -> high prob of 1, it's flipping
        # If transform(0) -> high prob of 0, it's preserving
        flip_score = (out_0[0, 1] + out_1[0, 0]) / 2  # Average flip probability

        return {
            'diagonal_mass': diagonal_mass.item(),
            'neighbor_mass': neighbor_mass.item(),
            'flip_score': flip_score.item(),  # >0.5 = flipping, <0.5 = preserving
            'blend': torch.sigmoid(self.blend).item()
        }


class TransformerWithLearnedInteraction(nn.Module):
    """Transformer + learned interaction inner loop"""
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

        self.interaction = LearnedInteraction(pattern_len)
        self.n_ticks = n_ticks

    def forward(self, x):
        x_embed = x.clone()
        x_embed[x_embed == -1] = 2
        h = self.embed(x_embed) + self.pos_embed
        h = self.transformer(h)
        logits = self.output(h)

        # Apply learned interaction for n_ticks
        for tick in range(self.n_ticks):
            logits = self.interaction(logits)

        return logits


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
# TRAINING & EVALUATION
# =============================================================================

def train_model(model, train_data, n_epochs=200, lr=1e-3):
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
        correct = sum(preds[i, pos] == targets[i, pos]
                     for i, positions in enumerate(mask_positions)
                     for pos in positions)
        total = sum(len(positions) for positions in mask_positions)
        return correct / total if total > 0 else 0


# =============================================================================
# EXPERIMENT
# =============================================================================

PATTERN_LEN = 8

tasks = {
    'Alternating': (generate_alternating_patterns, 'Should learn: flip + local'),
    'Repeating': (generate_repeat_patterns, 'Should learn: same + local'),
    'Parity': (generate_parity_patterns, 'Should learn: global'),
}

print("\n1. Testing on different tasks...")
print("=" * 70)

for task_name, (gen_fn, expected) in tasks.items():
    print(f"\n--- Task: {task_name} ---")
    print(f"    Expected: {expected}")

    train = gen_fn(500, PATTERN_LEN, mask_ratio=0.5)
    test = gen_fn(100, PATTERN_LEN, mask_ratio=0.75)

    # Baseline
    baseline = TransformerBaseline(PATTERN_LEN)
    baseline = train_model(baseline, train)
    base_acc = evaluate(baseline, test)

    # Learned interaction
    model = TransformerWithLearnedInteraction(PATTERN_LEN, n_ticks=4)
    model = train_model(model, train)
    model_acc = evaluate(model, test)

    # Analyze what it learned
    analysis = model.interaction.analyze()

    print(f"    Baseline:    {base_acc:.1%}")
    print(f"    Learned:     {model_acc:.1%} ({'+' if model_acc > base_acc else ''}{(model_acc-base_acc)*100:.1f}%)")
    print(f"    Analysis:")
    print(f"      - Diagonal mass: {analysis['diagonal_mass']:.2f} (self-influence)")
    print(f"      - Neighbor mass: {analysis['neighbor_mass']:.2f} (local = high)")
    print(f"      - Flip score: {analysis['flip_score']:.2f} (>0.5 = flip, <0.5 = same)")
    print(f"      - Blend: {analysis['blend']:.2f}")

    # Interpret
    is_local = analysis['neighbor_mass'] > 0.2
    is_flip = analysis['flip_score'] > 0.5

    if is_local and is_flip:
        learned_type = "local + flip (like alternating)"
    elif is_local and not is_flip:
        learned_type = "local + same (like repeating)"
    else:
        learned_type = "global or other"

    print(f"    Interpretation: {learned_type}")

print("\n" + "=" * 70)
print("CONCLUSION")
print("=" * 70)

print("""
The key insight:

1. NO DISCRETE SELECTION needed
2. One continuous mechanism with learned parameters
3. Backprop configures the interaction pattern (W) and transform
4. The SAME mechanism handles different structures

This is like:
- ONE type of neuron with different connection patterns
- Not discrete "mechanism types"
- The structure emerges from learned weights

Evolution analogy:
- Evolution didn't give brains discrete "mechanism modules"
- It gave neurons with learnable connections
- The "mechanisms" emerge from connection patterns

For AI:
- Don't build a library of mechanisms to select from
- Build ONE general interaction mechanism
- Let backprop learn what interaction pattern works
""")
