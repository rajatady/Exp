"""
Diagnose Inner Loop Divergence

The inner loop error is INCREASING, not decreasing.
Let's understand why and fix it.

Hypotheses:
1. Learning rate too high/low
2. Re-running encoder after update is wrong
3. The prediction target (embedding) isn't learnable
4. Gradient through encoder causes instability
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
print("DIAGNOSING INNER LOOP DIVERGENCE")
print("=" * 80)

# Simple setup
VOCAB_SIZE = 14
HIDDEN_DIM = 64

if torch.backends.mps.is_available():
    device = torch.device('mps')
elif torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print(f"Device: {device}")


class InnerLoopVariant(nn.Module):
    """Test different inner loop configurations."""

    def __init__(self, vocab_size, hidden_dim=64, n_layers=2, n_inner_steps=8,
                 inner_lr=0.1, rerun_encoder=True, use_momentum=False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_inner_steps = n_inner_steps
        self.inner_lr = inner_lr
        self.rerun_encoder = rerun_encoder
        self.use_momentum = use_momentum

        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.pos_enc = nn.Parameter(torch.randn(1, 128, hidden_dim) * 0.02)
        self.encoder = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim, nhead=4,
                dim_feedforward=hidden_dim*4,
                batch_first=True, dropout=0.0  # No dropout for stability
            ) for _ in range(n_layers)
        ])
        self.embed_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.ln = nn.LayerNorm(hidden_dim)
        self.output = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, return_diagnostics=False):
        B, T = x.shape
        input_embed = self.embed(x) + self.pos_enc[:, :T, :]

        # Initial encoding
        h = input_embed.clone()
        for layer in self.encoder:
            h = layer(h)

        diagnostics = {'errors': [], 'state_norms': []}
        velocity = torch.zeros_like(h) if self.use_momentum else None

        for step in range(self.n_inner_steps):
            pred_embed = self.embed_predictor(h)
            error = pred_embed - input_embed
            error_norm = error.norm(dim=-1).mean().item()
            diagnostics['errors'].append(error_norm)
            diagnostics['state_norms'].append(h.norm(dim=-1).mean().item())

            # Update
            if self.use_momentum:
                velocity = 0.9 * velocity - self.inner_lr * error
                h = h + velocity
            else:
                h = h - self.inner_lr * error

            # Optionally re-run encoder
            if self.rerun_encoder:
                for layer in self.encoder:
                    h = layer(h)

        logits = self.output(self.ln(h))

        if return_diagnostics:
            return logits, diagnostics
        return logits


# =============================================================================
# TEST DIFFERENT CONFIGURATIONS
# =============================================================================

print("\n" + "=" * 80)
print("TESTING DIFFERENT INNER LOOP CONFIGURATIONS")
print("=" * 80)

# Create dummy input
x = torch.randint(0, VOCAB_SIZE, (4, 10), device=device)

configs = [
    {"name": "Original (lr=0.1, rerun)", "inner_lr": 0.1, "rerun_encoder": True, "use_momentum": False},
    {"name": "No rerun encoder", "inner_lr": 0.1, "rerun_encoder": False, "use_momentum": False},
    {"name": "Lower LR (0.01)", "inner_lr": 0.01, "rerun_encoder": True, "use_momentum": False},
    {"name": "Lower LR + no rerun", "inner_lr": 0.01, "rerun_encoder": False, "use_momentum": False},
    {"name": "Tiny LR (0.001)", "inner_lr": 0.001, "rerun_encoder": False, "use_momentum": False},
    {"name": "With momentum", "inner_lr": 0.1, "rerun_encoder": False, "use_momentum": True},
]

print(f"\n{'Config':<25} {'Initial Err':>12} {'Final Err':>12} {'Change':>10}")
print("-" * 65)

for config in configs:
    torch.manual_seed(42)
    model = InnerLoopVariant(
        VOCAB_SIZE, HIDDEN_DIM,
        inner_lr=config["inner_lr"],
        rerun_encoder=config["rerun_encoder"],
        use_momentum=config["use_momentum"]
    ).to(device)

    model.eval()
    with torch.no_grad():
        _, diag = model(x, return_diagnostics=True)

    initial = diag['errors'][0]
    final = diag['errors'][-1]
    change = (final - initial) / initial * 100

    status = "↓" if change < 0 else "↑"
    print(f"{config['name']:<25} {initial:>11.3f} {final:>11.3f} {change:>+9.1f}% {status}")


print("\n" + "=" * 80)
print("DETAILED VIEW: Best Configuration")
print("=" * 80)

# Test the best one in detail
torch.manual_seed(42)
model = InnerLoopVariant(
    VOCAB_SIZE, HIDDEN_DIM,
    inner_lr=0.01,
    rerun_encoder=False,
    use_momentum=False
).to(device)

model.eval()
with torch.no_grad():
    _, diag = model(x, return_diagnostics=True)

print("\nError over steps (Lower LR + no rerun):")
for i, err in enumerate(diag['errors']):
    bar = "█" * int(err * 2)
    print(f"  Step {i}: {err:.4f} {bar}")

change = (diag['errors'][-1] - diag['errors'][0]) / diag['errors'][0] * 100
print(f"\nTotal change: {change:+.2f}%")


print("\n" + "=" * 80)
print("THE PROBLEM IDENTIFIED")
print("=" * 80)

print("""
The issue: Re-running the encoder after each state update!

When we do:
  1. h = h - lr * error
  2. h = encoder(h)  ← This undoes the update!

The encoder transforms h in complex ways, potentially moving it
AWAY from the direction we just pushed it.

Solution: Don't re-run encoder during inner loop, OR use a
separate "refinement" pathway that doesn't overwrite the update.
""")


print("\n" + "=" * 80)
print("TESTING FIX: No encoder rerun")
print("=" * 80)

# Now test with actual training
from collections import defaultdict

# Simple data
def generate_simple_data(n=100):
    data = []
    for _ in range(n):
        length = random.randint(3, 6)
        seq = [1] + [random.randint(4, 13) for _ in range(length)] + [3] + [random.randint(4, 13) for _ in range(length)] + [2]
        data.append(seq)
    return data

def pad_sequences(seqs, pad_id=0):
    max_len = max(len(s) for s in seqs)
    return [s + [pad_id] * (max_len - len(s)) for s in seqs]

train_data = generate_simple_data(200)
test_data = generate_simple_data(50)

def train_and_eval(model, train_data, test_data, n_epochs=100, device='cpu'):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    for epoch in range(n_epochs):
        model.train()
        batch = pad_sequences(train_data)
        x = torch.tensor(batch, device=device)
        inputs, targets = x[:, :-1], x[:, 1:]

        optimizer.zero_grad()
        logits = model(inputs)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=0)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    # Eval
    model.eval()
    with torch.no_grad():
        batch = pad_sequences(test_data)
        x = torch.tensor(batch, device=device)
        inputs, targets = x[:, :-1], x[:, 1:]
        logits = model(inputs)
        preds = logits.argmax(-1)
        mask = targets != 0
        acc = ((preds == targets) & mask).sum().item() / mask.sum().item()

    return acc

print("\nTraining models with different configs...")

results = {}
for config in [
    {"name": "Original (rerun)", "rerun_encoder": True, "inner_lr": 0.1},
    {"name": "Fixed (no rerun)", "rerun_encoder": False, "inner_lr": 0.1},
    {"name": "Fixed + lower LR", "rerun_encoder": False, "inner_lr": 0.01},
]:
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    model = InnerLoopVariant(
        VOCAB_SIZE, HIDDEN_DIM,
        inner_lr=config["inner_lr"],
        rerun_encoder=config["rerun_encoder"],
    ).to(device)

    acc = train_and_eval(model, train_data, test_data, n_epochs=100, device=device)
    results[config["name"]] = acc
    print(f"  {config['name']}: {acc:.1%}")

print("\n" + "=" * 80)
