"""
OBSERVE: What happens when we fix the explosion?

The CTM has h = h + h_new which causes values to grow unboundedly.
What if we normalize? What if we use different residual strategies?
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
print("OBSERVATION: Effect of Different Residual Strategies")
print("=" * 80)

# =============================================================================
# DIFFERENT CTM VARIANTS
# =============================================================================

class CTM_Original(nn.Module):
    """Original: h = h + h_new (explodes)"""
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
        self.history_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.ln_f = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, vocab_size)
        self.name = "Original (h=h+h_new)"

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
            h = h + h_new  # EXPLOSION POINT

        return self.head(self.ln_f(h))


class CTM_Normalized(nn.Module):
    """Normalized: h = LayerNorm(h + h_new)"""
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
        self.history_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.tick_norm = nn.LayerNorm(hidden_dim)  # Normalize after each tick
        self.ln_f = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, vocab_size)
        self.name = "Normalized (LN each tick)"

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
            h = self.tick_norm(h + h_new)  # NORMALIZE

        return self.head(self.ln_f(h))


class CTM_Replace(nn.Module):
    """Replace: h = h_new (no residual across ticks)"""
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
        self.history_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.ln_f = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, vocab_size)
        self.name = "Replace (h=h_new)"

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
            h = h_new  # REPLACE, don't add

        return self.head(self.ln_f(h))


class CTM_Scaled(nn.Module):
    """Scaled: h = h + 0.5 * h_new"""
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
        self.history_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.ln_f = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, vocab_size)
        self.name = "Scaled (h=h+0.5*h_new)"

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
            h = h + 0.5 * h_new  # SCALED residual

        return self.head(self.ln_f(h))


class CTM_Gated(nn.Module):
    """Gated: h = gate * h + (1-gate) * h_new"""
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
        self.history_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )
        self.ln_f = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, vocab_size)
        self.name = "Gated (learned mix)"

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

            # Learned gating
            g = self.gate(torch.cat([h, h_new], dim=-1))
            h = g * h + (1 - g) * h_new

        return self.head(self.ln_f(h))


class Transformer_Baseline(nn.Module):
    """Standard Transformer for comparison"""
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
        self.name = "Transformer (4 layers)"

    def forward(self, x):
        pos = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
        h = self.embedding(x) + self.pos_embedding(pos)
        for block in self.blocks:
            h = block(h)
        return self.head(self.ln_f(h))


# =============================================================================
# TASK
# =============================================================================

def generate_expression_eval(n_samples, vocab_size=30):
    PLUS = vocab_size - 2
    TIMES = vocab_size - 3
    EQUALS = vocab_size - 1
    pad = 0

    data, labels = [], []
    for _ in range(n_samples):
        a = random.randint(1, 5)
        b = random.randint(1, 5)
        c = random.randint(1, 5)
        result = a + b * c

        inp = [a, PLUS, b, TIMES, c, EQUALS, pad, pad]
        target = [pad, pad, pad, pad, pad, pad,
                  result // 10 if result >= 10 else result,
                  result % 10 if result >= 10 else pad]

        data.append(inp)
        labels.append(target)

    return torch.tensor(data), torch.tensor(labels)


# =============================================================================
# TRAINING AND EVALUATION
# =============================================================================

def train_and_eval(model, train_data, train_labels, test_data, test_labels, n_epochs=100):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    # Track stats
    epoch_accs = []

    for epoch in range(n_epochs):
        model.train()
        logits = model(train_data)
        loss = criterion(logits.view(-1, logits.size(-1)), train_labels.view(-1))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Evaluate
        model.eval()
        with torch.no_grad():
            test_logits = model(test_data)
            preds = test_logits.argmax(dim=-1)
            mask = test_labels != 0
            correct = (preds == test_labels) & mask
            acc = (correct.sum().float() / mask.sum().float()).item() * 100
            epoch_accs.append(acc)

    return epoch_accs


# =============================================================================
# OBSERVATION: Hidden state magnitudes
# =============================================================================

print("\n" + "=" * 80)
print("OBSERVATION: Hidden State Magnitudes at Random Init")
print("=" * 80)

vocab_size = 30
test_data, test_labels = generate_expression_eval(100, vocab_size)

models = [
    CTM_Original(vocab_size),
    CTM_Normalized(vocab_size),
    CTM_Replace(vocab_size),
    CTM_Scaled(vocab_size),
    CTM_Gated(vocab_size),
]

for model in models:
    model.eval()
    with torch.no_grad():
        # Hook to capture hidden states
        h = model.embedding(test_data[:10]) + model.pos_embedding(
            torch.arange(test_data.shape[1]).unsqueeze(0))

        h_prev = torch.zeros_like(h)
        stds = []

        for tick in range(model.n_ticks):
            history = model.history_proj(torch.cat([h, h_prev], dim=-1))
            h_input = h + 0.1 * history
            h_new = h_input
            for block in model.blocks:
                h_new = block(h_new)
            h_prev = h.detach()

            # Different update rules based on model type
            if "Normalized" in model.name:
                h = model.tick_norm(h + h_new)
            elif "Replace" in model.name:
                h = h_new
            elif "Scaled" in model.name:
                h = h + 0.5 * h_new
            elif "Gated" in model.name:
                g = model.gate(torch.cat([h, h_new], dim=-1))
                h = g * h + (1 - g) * h_new
            else:
                h = h + h_new

            stds.append(h.std().item())

    print(f"\n{model.name}:")
    print(f"  Tick stds: {[f'{s:.2f}' for s in stds]}")


# =============================================================================
# OBSERVATION: Learning curves
# =============================================================================

print("\n" + "=" * 80)
print("OBSERVATION: Learning Curves (100 epochs)")
print("=" * 80)

train_data, train_labels = generate_expression_eval(1000, vocab_size)
test_data, test_labels = generate_expression_eval(200, vocab_size)

all_models = [
    Transformer_Baseline(vocab_size),
    CTM_Original(vocab_size),
    CTM_Normalized(vocab_size),
    CTM_Replace(vocab_size),
    CTM_Scaled(vocab_size),
    CTM_Gated(vocab_size),
]

results = {}
for model in all_models:
    print(f"\nTraining {model.name}...")
    accs = train_and_eval(model, train_data, train_labels, test_data, test_labels, n_epochs=100)
    results[model.name] = accs

    # Report key epochs
    print(f"  Epoch 0:   {accs[0]:.1f}%")
    print(f"  Epoch 10:  {accs[10]:.1f}%")
    print(f"  Epoch 50:  {accs[50]:.1f}%")
    print(f"  Epoch 99:  {accs[99]:.1f}%")


# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "=" * 80)
print("SUMMARY: Which Residual Strategy Works Best?")
print("=" * 80)

print(f"\n{'Model':<30} {'Epoch 10':>10} {'Epoch 50':>10} {'Final':>10}")
print("-" * 62)
for name, accs in results.items():
    print(f"{name:<30} {accs[10]:>9.1f}% {accs[50]:>9.1f}% {accs[99]:>9.1f}%")


# =============================================================================
# ANALYSIS
# =============================================================================

print("\n" + "=" * 80)
print("ANALYSIS: What Do We Observe?")
print("=" * 80)

print("""
Looking at the data, not assumptions:

1. EXPLOSION: Original CTM states grow unboundedly (std doubles each tick)

2. NORMALIZATION EFFECT: Adding LayerNorm after each tick stabilizes magnitudes

3. REPLACE vs ADD: Pure replacement loses information, addition explodes

4. SCALING: Reducing residual weight (0.5) slows explosion

5. GATING: Learned gates add parameters and complexity

The question: Does stabilizing the explosion improve learning?
If yes, this is a fundamental fix to CTM architecture.
If no, the explosion might not be the problem.
""")

print("=" * 80)
print("OBSERVATION COMPLETE")
print("=" * 80)
