"""
Testing Variant A: Fuller History CTM

Previous ablation was on Variant B (simplified):
- history_proj = Linear(hidden_dim * 2, hidden_dim)
- Just one previous state (h_prev)

Now testing Variant A (original):
- history_processor = MLP with GELU
- history_len = 2 (maintains multiple past states)

Question: Does the fuller history mechanism matter?
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

print("=" * 80)
print("VARIANT A vs VARIANT B: Does Fuller History Matter?")
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
# VARIANT A: Fuller History (original design)
# =============================================================================

class CTMVariantA(nn.Module):
    """
    Original CTM design:
    - history_len = 2 (maintains 2 past states)
    - 2-layer MLP with GELU for history processing
    """
    def __init__(self, vocab_size, hidden_dim=64, n_layers=2, n_heads=4,
                 n_ticks=4, history_len=2):
        super().__init__()
        self.n_ticks = n_ticks
        self.history_len = history_len
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embedding = nn.Embedding(256, hidden_dim)

        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim, nhead=n_heads,
                dim_feedforward=hidden_dim * 4, batch_first=True
            ) for _ in range(n_layers)
        ])

        # Fuller history processor (2-layer MLP)
        self.history_processor = nn.Sequential(
            nn.Linear(hidden_dim * history_len, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.ln_f = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, vocab_size)
        self.name = "Variant A (full history)"

    def forward(self, x):
        pos = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
        h = self.embedding(x) + self.pos_embedding(pos)

        # Initialize history with zeros
        history = [torch.zeros_like(h) for _ in range(self.history_len)]

        for tick in range(self.n_ticks):
            # Concatenate history states
            history_cat = torch.cat(history, dim=-1)
            history_features = self.history_processor(history_cat)
            h_input = h + 0.1 * history_features

            h_new = h_input
            for block in self.blocks:
                h_new = block(h_new)

            # Update history (shift and add new)
            history = history[1:] + [h.clone()]

            # Accumulation
            h = h + h_new

        return self.head(self.ln_f(h))


class CTMVariantANoAccum(nn.Module):
    """Variant A without accumulation"""
    def __init__(self, vocab_size, hidden_dim=64, n_layers=2, n_heads=4,
                 n_ticks=4, history_len=2):
        super().__init__()
        self.n_ticks = n_ticks
        self.history_len = history_len
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embedding = nn.Embedding(256, hidden_dim)

        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim, nhead=n_heads,
                dim_feedforward=hidden_dim * 4, batch_first=True
            ) for _ in range(n_layers)
        ])

        self.history_processor = nn.Sequential(
            nn.Linear(hidden_dim * history_len, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.ln_f = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, vocab_size)
        self.name = "Variant A NO accumulation"

    def forward(self, x):
        pos = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
        h = self.embedding(x) + self.pos_embedding(pos)

        history = [torch.zeros_like(h) for _ in range(self.history_len)]

        for tick in range(self.n_ticks):
            history_cat = torch.cat(history, dim=-1)
            history_features = self.history_processor(history_cat)
            h_input = h + 0.1 * history_features

            h_new = h_input
            for block in self.blocks:
                h_new = block(h_new)

            history = history[1:] + [h.clone()]
            h = h_new  # NO accumulation

        return self.head(self.ln_f(h))


# =============================================================================
# VARIANT B: Simplified History (what we tested before)
# =============================================================================

class CTMVariantB(nn.Module):
    """
    Simplified CTM:
    - Just h_prev (one previous state)
    - Single linear projection
    """
    def __init__(self, vocab_size, hidden_dim=64, n_layers=2, n_heads=4, n_ticks=4):
        super().__init__()
        self.n_ticks = n_ticks
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embedding = nn.Embedding(256, hidden_dim)

        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim, nhead=n_heads,
                dim_feedforward=hidden_dim * 4, batch_first=True
            ) for _ in range(n_layers)
        ])

        # Simple history projection
        self.history_proj = nn.Linear(hidden_dim * 2, hidden_dim)

        self.ln_f = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, vocab_size)
        self.name = "Variant B (simple history)"

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


# =============================================================================
# SIMPLE ACCUMULATOR (no history at all)
# =============================================================================

class SimpleAccumulator(nn.Module):
    """Just accumulation, no history mechanism"""
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
        self.name = "Simple Accumulator (no history)"

    def forward(self, x):
        pos = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
        h = self.embedding(x) + self.pos_embedding(pos)

        for tick in range(self.n_ticks):
            h_new = h
            for block in self.blocks:
                h_new = block(h_new)
            h = h + h_new

        return self.head(self.ln_f(h))


# =============================================================================
# STANDARD TRANSFORMER (baseline)
# =============================================================================

class StandardTransformer(nn.Module):
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
        self.name = "Standard Transformer"

    def forward(self, x):
        pos = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
        h = self.embedding(x) + self.pos_embedding(pos)
        for block in self.blocks:
            h = block(h)
        return self.head(self.ln_f(h))


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
# EXPERIMENT
# =============================================================================

print("\n" + "=" * 80)
print("EXPERIMENT: Variant A vs Variant B vs Simple Accumulator")
print("=" * 80)

n_runs = 5
all_results = {}

models_to_test = [
    ("Standard Transformer", StandardTransformer),
    ("Variant A (full history)", CTMVariantA),
    ("Variant A NO accumulation", CTMVariantANoAccum),
    ("Variant B (simple history)", CTMVariantB),
    ("Simple Accumulator", SimpleAccumulator),
]

for name, ModelClass in models_to_test:
    all_results[name] = []

for run in range(n_runs):
    print(f"\n--- Run {run + 1}/{n_runs} ---")

    torch.manual_seed(42 + run)
    np.random.seed(42 + run)
    random.seed(42 + run)

    train_data, train_labels = generate_expression_eval(1000, vocab_size)
    test_data, test_labels = generate_expression_eval(200, vocab_size)

    for name, ModelClass in models_to_test:
        model = ModelClass(vocab_size)
        acc = train_and_eval(model, train_data, train_labels, test_data, test_labels)
        all_results[name].append(acc)
        print(f"  {name}: {acc:.1f}%")

# =============================================================================
# RESULTS
# =============================================================================

print("\n" + "=" * 80)
print("RESULTS ACROSS RUNS")
print("=" * 80)

print(f"\n{'Model':<35} {'Mean':>8} {'Std':>8} {'Params':>10}")
print("-" * 65)

for name, ModelClass in models_to_test:
    accs = all_results[name]
    mean = np.mean(accs)
    std = np.std(accs)
    model = ModelClass(vocab_size)
    params = count_params(model)
    print(f"{name:<35} {mean:>7.1f}% {std:>7.1f}% {params:>10,}")

# =============================================================================
# KEY COMPARISONS
# =============================================================================

print("\n" + "=" * 80)
print("KEY COMPARISONS")
print("=" * 80)

var_a = np.mean(all_results["Variant A (full history)"])
var_a_no_accum = np.mean(all_results["Variant A NO accumulation"])
var_b = np.mean(all_results["Variant B (simple history)"])
simple = np.mean(all_results["Simple Accumulator"])
transformer = np.mean(all_results["Standard Transformer"])

print(f"""
1. Does fuller history (Variant A) beat simpler history (Variant B)?
   Variant A: {var_a:.1f}%
   Variant B: {var_b:.1f}%
   Gap: {var_a - var_b:+.1f}%

2. Is accumulation still essential for Variant A?
   Variant A with accumulation:    {var_a:.1f}%
   Variant A without accumulation: {var_a_no_accum:.1f}%
   Gap: {var_a - var_a_no_accum:+.1f}%

3. Does any history beat no history?
   Variant A (full history):   {var_a:.1f}%
   Variant B (simple history): {var_b:.1f}%
   Simple Accumulator:         {simple:.1f}%
   Gap (A vs Simple): {var_a - simple:+.1f}%
   Gap (B vs Simple): {var_b - simple:+.1f}%

4. Do iterative models beat standard transformer?
   Best iterative:       {max(var_a, var_b, simple):.1f}%
   Standard Transformer: {transformer:.1f}%
   Gap: {max(var_a, var_b, simple) - transformer:+.1f}%
""")

# =============================================================================
# CONCLUSION
# =============================================================================

print("=" * 80)
print("CONCLUSION")
print("=" * 80)

if var_a > var_b + 1:
    print("\n→ Fuller history (Variant A) DOES help over simplified (Variant B)")
    print("  Our previous conclusion was WRONG - we tested on a simplified version")
elif var_b > var_a + 1:
    print("\n→ Simpler history (Variant B) actually works BETTER")
    print("  The extra complexity in Variant A is harmful")
else:
    print("\n→ Fuller history (Variant A) and simpler (Variant B) are EQUIVALENT")
    print("  The extra MLP and history_len don't provide meaningful benefit")

if var_a - var_a_no_accum > 5:
    print("\n→ Accumulation is ESSENTIAL for Variant A too")
    print("  This confirms our finding - accumulation is the key mechanism")
else:
    print("\n→ Accumulation is NOT essential for Variant A")
    print("  This would contradict our previous finding")

if simple > var_a - 1 and simple > var_b - 1:
    print("\n→ Simple Accumulator matches both variants")
    print("  History mechanism (in any form) is NOT essential")
else:
    print("\n→ History mechanism DOES provide benefit over simple accumulation")
    print("  Our previous finding needs revision")

print("\n" + "=" * 80)
