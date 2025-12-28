"""
CRITICAL OBSERVATION: Is self-reflection the key?

CTM: Tick 1 sees tick 0's output (SAME function, different time)
Compositional: Stage 2 sees stage 1's output (DIFFERENT function)

Hypothesis: The key isn't multi-phase, it's seeing YOUR OWN computation.

Test: Can we get CTM-level performance with explicit self-reflection
      in a non-iterative architecture?
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
print("SELF-REFLECTION: Is seeing your own output the key?")
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
# MODEL 1: CTM (baseline, has self-reflection)
# =============================================================================

class CTM(nn.Module):
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
        self.name = "CTM (self-reflection)"

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
# MODEL 2: CTM without self-reflection (h_prev always zero)
# =============================================================================

class CTMNoReflection(nn.Module):
    """CTM but h_prev is always zeros (no self-reflection)"""
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
        self.name = "CTM NO self-reflection"

    def forward(self, x):
        pos = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
        h = self.embedding(x) + self.pos_embedding(pos)
        h_prev = torch.zeros_like(h)  # Never updated!

        for tick in range(self.n_ticks):
            history = self.history_proj(torch.cat([h, h_prev], dim=-1))
            h_input = h + 0.1 * history
            h_new = h_input
            for block in self.blocks:
                h_new = block(h_new)
            # h_prev = h.detach()  # DISABLED
            h = h + h_new

        return self.head(self.ln_f(h))

# =============================================================================
# MODEL 3: Self-Reflective but non-iterative (unrolled with shared weights)
# =============================================================================

class UnrolledSelfReflection(nn.Module):
    """
    Like CTM but fully unrolled - no iteration loop.
    Same weights for each "tick" but explicit passes.
    """
    def __init__(self, vocab_size, hidden_dim=64, n_heads=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embedding = nn.Embedding(256, hidden_dim)

        # Shared weights (like CTM)
        self.block1 = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads,
            dim_feedforward=hidden_dim * 4, batch_first=True
        )
        self.block2 = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads,
            dim_feedforward=hidden_dim * 4, batch_first=True
        )
        self.history_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.ln_f = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, vocab_size)
        self.name = "Unrolled Self-Reflection"

    def forward(self, x):
        pos = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
        h = self.embedding(x) + self.pos_embedding(pos)

        # Tick 0
        h_prev = torch.zeros_like(h)
        history = self.history_proj(torch.cat([h, h_prev], dim=-1))
        h_input = h + 0.1 * history
        h_new = self.block1(self.block2(h_input))
        h_prev = h
        h = h + h_new

        # Tick 1 (sees tick 0)
        history = self.history_proj(torch.cat([h, h_prev], dim=-1))
        h_input = h + 0.1 * history
        h_new = self.block1(self.block2(h_input))
        h_prev = h
        h = h + h_new

        # Tick 2 (sees tick 1)
        history = self.history_proj(torch.cat([h, h_prev], dim=-1))
        h_input = h + 0.1 * history
        h_new = self.block1(self.block2(h_input))
        h_prev = h
        h = h + h_new

        # Tick 3 (sees tick 2)
        history = self.history_proj(torch.cat([h, h_prev], dim=-1))
        h_input = h + 0.1 * history
        h_new = self.block1(self.block2(h_input))
        h = h + h_new

        return self.head(self.ln_f(h))

# =============================================================================
# MODEL 4: Non-iterative but with explicit self-attention feedback
# =============================================================================

class SelfFeedbackTransformer(nn.Module):
    """
    Non-iterative transformer where later layers explicitly see
    the output of earlier layers (like skip connections but for attention)
    """
    def __init__(self, vocab_size, hidden_dim=64, n_heads=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads

        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embedding = nn.Embedding(256, hidden_dim)

        # Layer 1-2: Initial processing
        self.block1 = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads,
            dim_feedforward=hidden_dim * 4, batch_first=True
        )
        self.block2 = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads,
            dim_feedforward=hidden_dim * 4, batch_first=True
        )

        # Self-reflection projector (sees own output like CTM's history_proj)
        self.reflect_proj = nn.Linear(hidden_dim * 2, hidden_dim)

        # Layer 3-4: Process with self-reflection
        self.block3 = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads,
            dim_feedforward=hidden_dim * 4, batch_first=True
        )
        self.block4 = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads,
            dim_feedforward=hidden_dim * 4, batch_first=True
        )

        self.ln_f = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, vocab_size)
        self.name = "Self-Feedback Transformer"

    def forward(self, x):
        pos = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
        h = self.embedding(x) + self.pos_embedding(pos)

        # First pass (like tick 0)
        h_before = h
        h = self.block1(h)
        h = self.block2(h)
        h_after_first = h

        # Self-reflection: see what we just computed (like CTM's history mechanism)
        reflection = self.reflect_proj(torch.cat([h, h_before], dim=-1))
        h = h + 0.1 * reflection

        # Second pass (like tick 1, seeing our own computation)
        h = self.block3(h)
        h = self.block4(h)

        # Accumulation (like CTM's h = h + h_new)
        h = h + h_after_first

        return self.head(self.ln_f(h))

# =============================================================================
# MODEL 5: Minimal self-reflection (just the history mechanism)
# =============================================================================

class MinimalReflection(nn.Module):
    """
    Just add the history projection to a standard transformer.
    Is the reflection mechanism the key, independent of iteration?
    """
    def __init__(self, vocab_size, hidden_dim=64, n_heads=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embedding = nn.Embedding(256, hidden_dim)

        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim, nhead=n_heads,
                dim_feedforward=hidden_dim * 4, batch_first=True
            ) for _ in range(4)
        ])

        # Reflection at middle layer
        self.reflect_proj = nn.Linear(hidden_dim * 2, hidden_dim)

        self.ln_f = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, vocab_size)
        self.name = "Minimal Reflection"

    def forward(self, x):
        pos = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
        h = self.embedding(x) + self.pos_embedding(pos)
        h_input = h

        # First half
        h = self.blocks[0](h)
        h = self.blocks[1](h)

        # Reflect on what we've computed
        reflection = self.reflect_proj(torch.cat([h, h_input], dim=-1))
        h = h + 0.1 * reflection

        # Second half
        h = self.blocks[2](h)
        h = self.blocks[3](h)

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
print("EXPERIMENT: What role does self-reflection play?")
print("=" * 80)

models = [
    CTM(vocab_size),
    CTMNoReflection(vocab_size),
    UnrolledSelfReflection(vocab_size),
    SelfFeedbackTransformer(vocab_size),
    MinimalReflection(vocab_size),
]

results = {}
for model in models:
    print(f"\nTraining {model.name}...")
    print(f"  Parameters: {count_params(model):,}")
    acc = train_and_eval(model, train_data, train_labels, test_data, test_labels)
    results[model.name] = acc
    print(f"  Final accuracy: {acc:.1f}%")

# =============================================================================
# ANALYSIS
# =============================================================================

print("\n" + "=" * 80)
print("RESULTS SUMMARY")
print("=" * 80)

print(f"\n{'Model':<35} {'Params':>10} {'Accuracy':>10}")
print("-" * 57)
for model in models:
    name = model.name
    params = count_params(model)
    acc = results[name]
    print(f"{name:<35} {params:>10,} {acc:>9.1f}%")

# =============================================================================
# KEY INSIGHT
# =============================================================================

print("\n" + "=" * 80)
print("KEY INSIGHT")
print("=" * 80)

ctm_acc = results["CTM (self-reflection)"]
no_reflect = results["CTM NO self-reflection"]

print(f"""
Self-reflection contribution:
  CTM with self-reflection:    {ctm_acc:.1f}%
  CTM WITHOUT self-reflection: {no_reflect:.1f}%
  Gap: {ctm_acc - no_reflect:+.1f}%

If the gap is large → Self-reflection is the key mechanism
If the gap is small → Iteration/accumulation is more important

Non-iterative alternatives:
""")

for name, acc in results.items():
    if name not in ["CTM (self-reflection)", "CTM NO self-reflection"]:
        gap = acc - ctm_acc
        print(f"  {name}: {acc:.1f}% (gap to CTM: {gap:+.1f}%)")

print("""
If non-iterative alternatives match CTM:
  → Iteration is not essential
  → Can build "CTM-like" models without the speed tax

If they don't match:
  → Something about iteration is fundamentally necessary
  → The "self-reflection" needs temporal dimension
""")

print("=" * 80)
