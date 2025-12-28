"""
CRITICAL OBSERVATION: Accumulation vs Self-Reflection

Previous findings conflict:
1. observe_tick1.py: Big jump at tick 1 when h_prev becomes non-zero (self-reflection)
2. observe_self_reflection.py: Removing self-reflection doesn't hurt (accumulation enough)

Resolution: Maybe both work, but through different mechanisms?

Let's look at:
1. Per-tick accuracy with AND without self-reflection
2. What exactly does accumulation (h = h + h_new) provide?
3. What does reflection (seeing h_prev) provide?
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
print("ACCUMULATION VS SELF-REFLECTION: Which mechanism matters?")
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
# MODELS with different combinations
# =============================================================================

class CTMVariant(nn.Module):
    def __init__(self, vocab_size, hidden_dim=64, n_layers=2, n_heads=4, n_ticks=4,
                 use_reflection=True, use_accumulation=True, use_history=True):
        super().__init__()
        self.n_ticks = n_ticks
        self.hidden_dim = hidden_dim
        self.use_reflection = use_reflection  # h_prev updated or zeros
        self.use_accumulation = use_accumulation  # h = h + h_new or h = h_new
        self.use_history = use_history  # history mechanism used or not

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

        name_parts = ["CTM"]
        name_parts.append("R+" if use_reflection else "R-")
        name_parts.append("A+" if use_accumulation else "A-")
        name_parts.append("H+" if use_history else "H-")
        self.name = "".join(name_parts)

    def forward_with_intermediates(self, x):
        """Forward with per-tick accuracy tracking"""
        pos = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
        h = self.embedding(x) + self.pos_embedding(pos)
        h_prev = torch.zeros_like(h)

        tick_logits = []

        for tick in range(self.n_ticks):
            if self.use_history:
                history = self.history_proj(torch.cat([h, h_prev], dim=-1))
                h_input = h + 0.1 * history
            else:
                h_input = h

            h_new = h_input
            for block in self.blocks:
                h_new = block(h_new)

            if self.use_reflection:
                h_prev = h.detach()

            if self.use_accumulation:
                h = h + h_new
            else:
                h = h_new

            # Record logits at this tick
            tick_logits.append(self.head(self.ln_f(h)))

        return tick_logits

    def forward(self, x):
        return self.forward_with_intermediates(x)[-1]

# =============================================================================
# TRAINING
# =============================================================================

def train_model(model, train_data, train_labels, n_epochs=100):
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

def evaluate_per_tick(model, test_data, test_labels):
    model.eval()
    with torch.no_grad():
        tick_logits = model.forward_with_intermediates(test_data)

        results = []
        for tick, logits in enumerate(tick_logits):
            preds = logits.argmax(dim=-1)
            mask = test_labels != 0
            correct = (preds == test_labels) & mask
            acc = (correct.sum().float() / mask.sum().float()).item() * 100
            results.append(acc)

        return results

# =============================================================================
# EXPERIMENT: All combinations
# =============================================================================

print("\n" + "=" * 80)
print("EXPERIMENT: Testing all combinations of R(eflection), A(ccumulation), H(istory)")
print("=" * 80)

configs = [
    (True, True, True),    # Full CTM: R+ A+ H+
    (False, True, True),   # No reflection: R- A+ H+
    (True, False, True),   # No accumulation: R+ A- H+
    (True, True, False),   # No history: R+ A+ H-
    (False, False, True),  # Only history: R- A- H+
    (False, True, False),  # Only accumulation: R- A+ H-
    (True, False, False),  # Only reflection: R+ A- H-
    (False, False, False), # Nothing: R- A- H-
]

all_results = {}

for reflection, accumulation, history in configs:
    model = CTMVariant(vocab_size,
                       use_reflection=reflection,
                       use_accumulation=accumulation,
                       use_history=history)

    print(f"\nTraining {model.name}...")
    train_model(model, train_data, train_labels)

    per_tick = evaluate_per_tick(model, test_data, test_labels)
    all_results[model.name] = per_tick

    print(f"  Per-tick accuracy: {[f'{x:.1f}' for x in per_tick]}")
    print(f"  Final accuracy: {per_tick[-1]:.1f}%")

# =============================================================================
# ANALYSIS
# =============================================================================

print("\n" + "=" * 80)
print("DETAILED COMPARISON: Per-tick accuracy")
print("=" * 80)

print(f"\n{'Model':<20} {'Tick 0':>8} {'Tick 1':>8} {'Tick 2':>8} {'Tick 3':>8}")
print("-" * 54)

for name, accs in all_results.items():
    print(f"{name:<20}", end="")
    for acc in accs:
        print(f"{acc:>8.1f}", end="")
    print()

# =============================================================================
# KEY COMPARISONS
# =============================================================================

print("\n" + "=" * 80)
print("KEY COMPARISONS")
print("=" * 80)

full_ctm = all_results["CTMR+A+H+"]
no_reflect = all_results["CTMR-A+H+"]
no_accum = all_results["CTMR+A-H+"]
no_history = all_results["CTMR+A+H-"]

print(f"""
Effect of Reflection (A+H+ constant):
  With reflection (R+):    {[f'{x:.1f}' for x in full_ctm]}
  Without reflection (R-): {[f'{x:.1f}' for x in no_reflect]}
  Tick 0 gap: {full_ctm[0] - no_reflect[0]:+.1f}%
  Tick 1 gap: {full_ctm[1] - no_reflect[1]:+.1f}%
  Final gap: {full_ctm[-1] - no_reflect[-1]:+.1f}%

Effect of Accumulation (R+H+ constant):
  With accumulation (A+): {[f'{x:.1f}' for x in full_ctm]}
  Without accumulation (A-): {[f'{x:.1f}' for x in no_accum]}
  Tick 0 gap: {full_ctm[0] - no_accum[0]:+.1f}%
  Final gap: {full_ctm[-1] - no_accum[-1]:+.1f}%

Effect of History (R+A+ constant):
  With history (H+): {[f'{x:.1f}' for x in full_ctm]}
  Without history (H-): {[f'{x:.1f}' for x in no_history]}
  Tick 0 gap: {full_ctm[0] - no_history[0]:+.1f}%
  Final gap: {full_ctm[-1] - no_history[-1]:+.1f}%
""")

# =============================================================================
# THE BIG QUESTION
# =============================================================================

print("=" * 80)
print("THE BIG QUESTION: What makes CTM work?")
print("=" * 80)

only_accum = all_results["CTMR-A+H-"]
only_history = all_results["CTMR-A-H+"]
nothing = all_results["CTMR-A-H-"]

print(f"""
Minimal models:
  Nothing (R-A-H-):      Final = {nothing[-1]:.1f}% (baseline)
  Only Accumulation:     Final = {only_accum[-1]:.1f}% (gain: {only_accum[-1] - nothing[-1]:+.1f}%)
  Only History:          Final = {only_history[-1]:.1f}% (gain: {only_history[-1] - nothing[-1]:+.1f}%)
  Full CTM:              Final = {full_ctm[-1]:.1f}%

If accumulation alone gets close to full CTM:
  → The residual connection (h = h + h_new) is the key
  → Iteration allows repeated refinement
  → History/reflection are just helpers, not essential

The pattern:
  Iteration × Accumulation = Progressive refinement
  Each tick adds a delta to h
  Over ticks: h = h0 + delta0 + delta1 + delta2 + delta3

  This is like an iterative solver: guess + correction + correction + ...
""")

print("=" * 80)
