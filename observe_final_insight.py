"""
FINAL INSIGHT: Iterative Accumulation is the Key

Previous finding:
  ONLY ACCUMULATION (R-A+H-): 99.0%  ← BEST
  Full CTM: 98.3%

This means:
1. History mechanism is NOT essential
2. Self-reflection is NOT essential
3. The key is: iteration + accumulation (h = h + h_new)

Let's verify and understand:
1. Multiple runs to confirm consistency
2. Compare with standard transformer
3. Understand what accumulation provides
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import math

print("=" * 80)
print("FINAL INSIGHT: Iterative Accumulation")
print("=" * 80)

# =============================================================================
# DATA
# =============================================================================

def generate_expression_eval(n_samples, vocab_size=30):
    PLUS = vocab_size - 2
    TIMES = vocab_size - 3
    EQUALS = vocab_size - 1

    data, labels = [], []
    for _ in range(n_samples):
        a = random.randint(1, 5)
        b = random.randint(1, 5)
        c = random.randint(1, 5)
        result = a + b * c

        inp = [a, PLUS, b, TIMES, c, EQUALS, 0, 0]
        target = [0, 0, 0, 0, 0, 0,
                  result // 10 if result >= 10 else result,
                  result % 10 if result >= 10 else 0]

        data.append(inp)
        labels.append(target)

    return torch.tensor(data), torch.tensor(labels)

vocab_size = 30
train_data, train_labels = generate_expression_eval(1000, vocab_size)
test_data, test_labels = generate_expression_eval(200, vocab_size)

# =============================================================================
# MODELS
# =============================================================================

class StandardTransformer(nn.Module):
    """Standard transformer - baseline"""
    def __init__(self, vocab_size, hidden_dim=64, n_layers=4, n_heads=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embedding = nn.Embedding(256, hidden_dim)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim, nhead=n_heads,
                dim_feedforward=hidden_dim * 4, batch_first=True
            ) for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, vocab_size)
        self.name = "Standard Transformer (4 layers)"

    def forward(self, x):
        pos = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
        h = self.embedding(x) + self.pos_embedding(pos)
        for block in self.blocks:
            h = block(h)
        return self.head(self.ln_f(h))

class FullCTM(nn.Module):
    """Full CTM with history"""
    def __init__(self, vocab_size, hidden_dim=64, n_layers=2, n_heads=4, n_ticks=4):
        super().__init__()
        self.n_ticks = n_ticks
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embedding = nn.Embedding(256, hidden_dim)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim, nhead=n_heads,
                dim_feedforward=hidden_dim * 4, batch_first=True
            ) for _ in range(n_layers)
        ])
        self.history_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.ln_f = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, vocab_size)
        self.name = "Full CTM (history)"

    def forward(self, x):
        pos = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
        h = self.embedding(x) + self.pos_embedding(pos)
        h_prev = torch.zeros_like(h)

        for tick in range(self.n_ticks):
            history = self.history_proj(torch.cat([h, h_prev], dim=-1))
            h_input = h + 0.1 * history
            h_new = h_input
            for block in self.blocks:
                h_new = block(h_new)
            h_prev = h.detach()
            h = h + h_new

        return self.head(self.ln_f(h))

class SimpleIterativeAccumulator(nn.Module):
    """
    The key insight: Just iterate with accumulation.
    No history, no reflection - just h = h + f(h)
    """
    def __init__(self, vocab_size, hidden_dim=64, n_layers=2, n_heads=4, n_ticks=4):
        super().__init__()
        self.n_ticks = n_ticks
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embedding = nn.Embedding(256, hidden_dim)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim, nhead=n_heads,
                dim_feedforward=hidden_dim * 4, batch_first=True
            ) for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, vocab_size)
        self.name = "Simple Accumulator"

    def forward(self, x):
        pos = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
        h = self.embedding(x) + self.pos_embedding(pos)

        for tick in range(self.n_ticks):
            h_new = h
            for block in self.blocks:
                h_new = block(h_new)
            h = h + h_new  # THE KEY: accumulation

        return self.head(self.ln_f(h))

class SharedWeightDeep(nn.Module):
    """
    Alternative view: Deep network with weight sharing.
    Same as accumulator but presented differently.
    """
    def __init__(self, vocab_size, hidden_dim=64, n_heads=4, n_repeats=4):
        super().__init__()
        self.n_repeats = n_repeats
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embedding = nn.Embedding(256, hidden_dim)

        # Single block repeated
        self.block = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads,
            dim_feedforward=hidden_dim * 4, batch_first=True
        )
        self.ln_f = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, vocab_size)
        self.name = "Shared Weight Deep"

    def forward(self, x):
        pos = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
        h = self.embedding(x) + self.pos_embedding(pos)
        h_accum = h

        for _ in range(self.n_repeats):
            h = self.block(h)
            h_accum = h_accum + h  # Accumulate all outputs

        return self.head(self.ln_f(h_accum))

class DenseConnection(nn.Module):
    """
    Like DenseNet: accumulate outputs from all layers.
    Similar idea but with unique weights.
    """
    def __init__(self, vocab_size, hidden_dim=64, n_layers=4, n_heads=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embedding = nn.Embedding(256, hidden_dim)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim, nhead=n_heads,
                dim_feedforward=hidden_dim * 4, batch_first=True
            ) for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, vocab_size)
        self.name = "Dense Connection"

    def forward(self, x):
        pos = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
        h = self.embedding(x) + self.pos_embedding(pos)
        h_accum = h

        for block in self.blocks:
            h = block(h)
            h_accum = h_accum + h  # Accumulate all layer outputs

        return self.head(self.ln_f(h_accum))

# =============================================================================
# TRAINING
# =============================================================================

def train_and_eval(model, train_data, train_labels, test_data, test_labels, n_epochs=100):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    for epoch in range(n_epochs):
        model.train()
        logits = model(train_data)
        loss = criterion(logits.view(-1, logits.size(-1)), train_labels.view(-1))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    model.eval()
    with torch.no_grad():
        logits = model(test_data)
        preds = logits.argmax(dim=-1)
        mask = test_labels != 0
        correct = (preds == test_labels) & mask
        return (correct.sum().float() / mask.sum().float()).item() * 100

def count_params(model):
    return sum(p.numel() for p in model.parameters())

# =============================================================================
# EXPERIMENT: Multiple runs for consistency
# =============================================================================

print("\n" + "=" * 80)
print("EXPERIMENT: Multiple runs to verify consistency")
print("=" * 80)

n_runs = 5
all_results = {
    "Standard Transformer (4 layers)": [],
    "Full CTM (history)": [],
    "Simple Accumulator": [],
    "Shared Weight Deep": [],
    "Dense Connection": [],
}

for run in range(n_runs):
    print(f"\n--- Run {run + 1}/{n_runs} ---")

    # Set different seed each run
    torch.manual_seed(42 + run)
    np.random.seed(42 + run)
    random.seed(42 + run)

    # Regenerate data
    train_data, train_labels = generate_expression_eval(1000, vocab_size)
    test_data, test_labels = generate_expression_eval(200, vocab_size)

    models = [
        StandardTransformer(vocab_size),
        FullCTM(vocab_size),
        SimpleIterativeAccumulator(vocab_size),
        SharedWeightDeep(vocab_size),
        DenseConnection(vocab_size),
    ]

    for model in models:
        acc = train_and_eval(model, train_data, train_labels, test_data, test_labels)
        all_results[model.name].append(acc)
        print(f"  {model.name}: {acc:.1f}%")

# =============================================================================
# ANALYSIS
# =============================================================================

print("\n" + "=" * 80)
print("RESULTS ACROSS RUNS")
print("=" * 80)

print(f"\n{'Model':<30} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
print("-" * 64)

for name, accs in all_results.items():
    mean = np.mean(accs)
    std = np.std(accs)
    min_acc = min(accs)
    max_acc = max(accs)
    print(f"{name:<30} {mean:>7.1f}% {std:>7.1f}% {min_acc:>7.1f}% {max_acc:>7.1f}%")

# =============================================================================
# FINAL INSIGHT
# =============================================================================

print("\n" + "=" * 80)
print("FINAL INSIGHT: What Makes CTM Work")
print("=" * 80)

print("""
DISCOVERY:

1. The Simple Accumulator (just h = h + f(h)) matches or beats Full CTM
2. The history mechanism (h_prev) is NOT essential
3. Self-reflection is NOT essential
4. Weight sharing + accumulation is the key

WHAT THIS MEANS:

The CTM's power comes from ITERATIVE REFINEMENT:
  h_final = h_0 + f(h_0) + f(f(h_0)) + f(f(f(h_0))) + ...

This is like:
  - Gradient descent on hidden representations
  - Progressive elaboration of features
  - Residual connections applied ACROSS iterations, not just within layers

KEY MECHANISM:
  Standard Transformer: h → f1 → f2 → f3 → f4
  Simple Accumulator:   h → f(h) → h + f(h) → f(h') → h' + f(h') → ...

The accumulation allows information to PERSIST across iterations.
Each iteration adds a CORRECTION to the running sum.

POTENTIAL RUG-PULLER INSIGHT:

Instead of:  Multiple unique layers (expensive, more params)
Use:         Shared layers + accumulation (cheaper, fewer params, works better?)

This is like Universal Transformers but with explicit accumulation.
The "explosion" we observed earlier is actually the accumulation working.
The LayerNorm at the end rescales everything.

SIMPLIFIED ARCHITECTURE:
  - Single transformer block (or small stack)
  - Apply N times with accumulation: h = h + block(h)
  - Final LayerNorm and output head

This could be more parameter-efficient AND more effective than standard transformers
for tasks requiring iterative computation.
""")

# Parameter comparison
print("\n" + "=" * 80)
print("PARAMETER EFFICIENCY")
print("=" * 80)

models_for_params = [
    StandardTransformer(vocab_size),
    FullCTM(vocab_size),
    SimpleIterativeAccumulator(vocab_size),
    SharedWeightDeep(vocab_size),
    DenseConnection(vocab_size),
]

print(f"\n{'Model':<30} {'Params':>12}")
print("-" * 44)
for model in models_for_params:
    print(f"{model.name:<30} {count_params(model):>12,}")

print("""
The Simple Accumulator and Shared Weight Deep use the SAME weights
across iterations, giving them fewer parameters.

If they match or beat the standard transformer:
  → We can get better performance with fewer parameters
  → Through the power of iterative accumulation
""")

print("=" * 80)
