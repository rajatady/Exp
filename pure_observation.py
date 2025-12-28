"""
PURE OBSERVATION: What is CTM actually doing?

No assumptions. Just measure:
1. What happens at each tick?
2. What is different between CTM and Transformer?
3. What does the history mechanism contribute?
4. Why does CTM start with 30% at epoch 0 (before training)?
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
print("PURE OBSERVATION: What is CTM actually doing?")
print("=" * 80)

# =============================================================================
# MODELS WITH INSTRUMENTATION
# =============================================================================

class TransformerInstrumented(nn.Module):
    def __init__(self, vocab_size, hidden_dim=64, n_layers=4, n_heads=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
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

        # Instrumentation
        self.layer_outputs = []

    def forward(self, x, record=False):
        self.layer_outputs = []

        pos = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
        h = self.embedding(x) + self.pos_embedding(pos)

        if record:
            self.layer_outputs.append(h.detach().clone())

        for block in self.blocks:
            h = block(h)
            if record:
                self.layer_outputs.append(h.detach().clone())

        return self.head(self.ln_f(h))


class CTMInstrumented(nn.Module):
    def __init__(self, vocab_size, hidden_dim=64, n_layers=2, n_heads=4, n_ticks=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
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

        # Instrumentation
        self.tick_outputs = []
        self.history_contributions = []
        self.pre_history_states = []

    def forward(self, x, record=False):
        self.tick_outputs = []
        self.history_contributions = []
        self.pre_history_states = []

        pos = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
        h = self.embedding(x) + self.pos_embedding(pos)
        h_prev = torch.zeros_like(h)

        for tick in range(self.n_ticks):
            if record:
                self.pre_history_states.append(h.detach().clone())

            # History contribution
            history = self.history_proj(torch.cat([h, h_prev], dim=-1))

            if record:
                self.history_contributions.append(history.detach().clone())

            h_input = h + 0.1 * history

            h_new = h_input
            for block in self.blocks:
                h_new = block(h_new)

            h_prev = h.detach()
            h = h + h_new

            if record:
                self.tick_outputs.append(h.detach().clone())

        return self.head(self.ln_f(h))


# =============================================================================
# TASK
# =============================================================================

def generate_expression_eval(n_samples, vocab_size=30):
    """a + b * c with operator precedence"""
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
# OBSERVATION 1: What happens at epoch 0 (random init)?
# =============================================================================

print("\n" + "=" * 80)
print("OBSERVATION 1: Behavior at Random Initialization (Epoch 0)")
print("=" * 80)

vocab_size = 30
train_data, train_labels = generate_expression_eval(500, vocab_size)
test_data, test_labels = generate_expression_eval(200, vocab_size)

# Create models (random init)
transformer = TransformerInstrumented(vocab_size, hidden_dim=64, n_layers=4)
ctm = CTMInstrumented(vocab_size, hidden_dim=64, n_layers=2, n_ticks=4)

# Evaluate at random init
def evaluate(model, data, labels):
    model.eval()
    with torch.no_grad():
        logits = model(data)
        preds = logits.argmax(dim=-1)
        mask = labels != 0
        correct = (preds == labels) & mask
        return (correct.sum().float() / mask.sum().float()).item() * 100

print(f"\nAccuracy at random init:")
print(f"  Transformer: {evaluate(transformer, test_data, test_labels):.1f}%")
print(f"  CTM:         {evaluate(ctm, test_data, test_labels):.1f}%")

# =============================================================================
# OBSERVATION 2: What do the hidden states look like?
# =============================================================================

print("\n" + "=" * 80)
print("OBSERVATION 2: Hidden State Statistics")
print("=" * 80)

# Run with recording
sample = test_data[:10]
with torch.no_grad():
    _ = transformer(sample, record=True)
    _ = ctm(sample, record=True)

print("\nTransformer layer outputs (mean, std, max):")
for i, out in enumerate(transformer.layer_outputs):
    mean = out.mean().item()
    std = out.std().item()
    max_val = out.abs().max().item()
    print(f"  Layer {i}: mean={mean:+.4f}, std={std:.4f}, max={max_val:.4f}")

print("\nCTM tick outputs (mean, std, max):")
for i, out in enumerate(ctm.tick_outputs):
    mean = out.mean().item()
    std = out.std().item()
    max_val = out.abs().max().item()
    print(f"  Tick {i}: mean={mean:+.4f}, std={std:.4f}, max={max_val:.4f}")

# =============================================================================
# OBSERVATION 3: How much does history contribute?
# =============================================================================

print("\n" + "=" * 80)
print("OBSERVATION 3: History Contribution Analysis")
print("=" * 80)

print("\nHistory contribution magnitude (before 0.1 scaling):")
for i, hist in enumerate(ctm.history_contributions):
    magnitude = hist.norm(dim=-1).mean().item()
    print(f"  Tick {i}: history_norm = {magnitude:.4f}")

print("\nPre-history hidden state magnitude:")
for i, state in enumerate(ctm.pre_history_states):
    magnitude = state.norm(dim=-1).mean().item()
    print(f"  Tick {i}: hidden_norm = {magnitude:.4f}")

print("\nRatio (history / hidden):")
for i in range(len(ctm.history_contributions)):
    hist_mag = ctm.history_contributions[i].norm(dim=-1).mean().item()
    hidden_mag = ctm.pre_history_states[i].norm(dim=-1).mean().item()
    ratio = hist_mag / hidden_mag if hidden_mag > 0 else 0
    effective = ratio * 0.1  # After 0.1 scaling
    print(f"  Tick {i}: ratio={ratio:.4f}, effective_contribution={effective:.4f}")

# =============================================================================
# OBSERVATION 4: How do states evolve across ticks?
# =============================================================================

print("\n" + "=" * 80)
print("OBSERVATION 4: State Evolution Across Ticks")
print("=" * 80)

print("\nCosine similarity between consecutive tick outputs:")
for i in range(1, len(ctm.tick_outputs)):
    prev = ctm.tick_outputs[i-1].flatten()
    curr = ctm.tick_outputs[i].flatten()
    cos_sim = F.cosine_similarity(prev.unsqueeze(0), curr.unsqueeze(0)).item()
    print(f"  Tick {i-1} → Tick {i}: cos_sim = {cos_sim:.4f}")

print("\nL2 distance between consecutive tick outputs:")
for i in range(1, len(ctm.tick_outputs)):
    dist = (ctm.tick_outputs[i] - ctm.tick_outputs[i-1]).norm().item()
    print(f"  Tick {i-1} → Tick {i}: L2_dist = {dist:.4f}")

# =============================================================================
# OBSERVATION 5: Output logit entropy
# =============================================================================

print("\n" + "=" * 80)
print("OBSERVATION 5: Output Entropy (Confidence)")
print("=" * 80)

def compute_entropy(logits):
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    entropy = -(probs * log_probs).sum(dim=-1)
    return entropy.mean().item()

with torch.no_grad():
    trans_logits = transformer(sample)
    ctm_logits = ctm(sample)

print(f"\nOutput entropy (lower = more confident):")
print(f"  Transformer: {compute_entropy(trans_logits):.4f}")
print(f"  CTM:         {compute_entropy(ctm_logits):.4f}")

# Max entropy for reference
max_entropy = math.log(vocab_size)
print(f"  Max possible: {max_entropy:.4f}")

# =============================================================================
# OBSERVATION 6: Train for 1 epoch and observe change
# =============================================================================

print("\n" + "=" * 80)
print("OBSERVATION 6: Change After 1 Epoch of Training")
print("=" * 80)

# Store initial predictions
with torch.no_grad():
    trans_preds_before = transformer(test_data).argmax(dim=-1)
    ctm_preds_before = ctm(test_data).argmax(dim=-1)

# Train for 1 epoch
criterion = nn.CrossEntropyLoss(ignore_index=0)

for model, name in [(transformer, "Transformer"), (ctm, "CTM")]:
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.train()
    logits = model(train_data)
    loss = criterion(logits.view(-1, logits.size(-1)), train_labels.view(-1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Check predictions after
with torch.no_grad():
    trans_preds_after = transformer(test_data).argmax(dim=-1)
    ctm_preds_after = ctm(test_data).argmax(dim=-1)

# How many predictions changed?
trans_changed = (trans_preds_before != trans_preds_after).float().mean().item() * 100
ctm_changed = (ctm_preds_before != ctm_preds_after).float().mean().item() * 100

print(f"\nPredictions changed after 1 epoch:")
print(f"  Transformer: {trans_changed:.1f}%")
print(f"  CTM:         {ctm_changed:.1f}%")

print(f"\nAccuracy after 1 epoch:")
print(f"  Transformer: {evaluate(transformer, test_data, test_labels):.1f}%")
print(f"  CTM:         {evaluate(ctm, test_data, test_labels):.1f}%")

# =============================================================================
# OBSERVATION 7: Gradient magnitude
# =============================================================================

print("\n" + "=" * 80)
print("OBSERVATION 7: Gradient Flow")
print("=" * 80)

# Reset models
transformer = TransformerInstrumented(vocab_size, hidden_dim=64, n_layers=4)
ctm = CTMInstrumented(vocab_size, hidden_dim=64, n_layers=2, n_ticks=4)

for model, name in [(transformer, "Transformer"), (ctm, "CTM")]:
    model.train()
    logits = model(train_data[:100])
    loss = criterion(logits.view(-1, logits.size(-1)), train_labels[:100].view(-1))
    loss.backward()

    total_grad = 0
    n_params = 0
    for p in model.parameters():
        if p.grad is not None:
            total_grad += p.grad.abs().sum().item()
            n_params += p.grad.numel()

    avg_grad = total_grad / n_params if n_params > 0 else 0
    print(f"\n{name}:")
    print(f"  Total gradient magnitude: {total_grad:.4f}")
    print(f"  Avg gradient per param:   {avg_grad:.6f}")
    print(f"  Number of parameters:     {n_params}")

# =============================================================================
# OBSERVATION 8: What is the model actually outputting at epoch 0?
# =============================================================================

print("\n" + "=" * 80)
print("OBSERVATION 8: What Predictions at Epoch 0?")
print("=" * 80)

transformer = TransformerInstrumented(vocab_size, hidden_dim=64, n_layers=4)
ctm = CTMInstrumented(vocab_size, hidden_dim=64, n_layers=2, n_ticks=4)

with torch.no_grad():
    trans_logits = transformer(test_data[:20])
    ctm_logits = ctm(test_data[:20])

    trans_preds = trans_logits.argmax(dim=-1)
    ctm_preds = ctm_logits.argmax(dim=-1)

print("\nFirst 5 examples:")
for i in range(5):
    inp = test_data[i].tolist()
    true = test_labels[i].tolist()
    t_pred = trans_preds[i].tolist()
    c_pred = ctm_preds[i].tolist()

    # Only show output positions (last 2)
    print(f"\n  Example {i}:")
    print(f"    Input:       {inp}")
    print(f"    True output: {true[-2:]}")
    print(f"    Trans pred:  {t_pred[-2:]}")
    print(f"    CTM pred:    {c_pred[-2:]}")

# What tokens are most commonly predicted?
print("\nMost common predictions at output positions:")
trans_pred_counts = {}
ctm_pred_counts = {}

for i in range(len(test_data)):
    for pos in [-2, -1]:
        if test_labels[i][pos] != 0:  # Only count non-padding
            t_tok = trans_preds[i][pos].item() if i < 20 else 0
            c_tok = ctm_preds[i][pos].item() if i < 20 else 0
            trans_pred_counts[t_tok] = trans_pred_counts.get(t_tok, 0) + 1
            ctm_pred_counts[c_tok] = ctm_pred_counts.get(c_tok, 0) + 1

print(f"\n  Transformer top preds: {sorted(trans_pred_counts.items(), key=lambda x: -x[1])[:5]}")
print(f"  CTM top preds:         {sorted(ctm_pred_counts.items(), key=lambda x: -x[1])[:5]}")

# =============================================================================
# OBSERVATION 9: Parameter count comparison
# =============================================================================

print("\n" + "=" * 80)
print("OBSERVATION 9: Parameter Counts")
print("=" * 80)

def count_params(model):
    return sum(p.numel() for p in model.parameters())

print(f"\nTotal parameters:")
print(f"  Transformer (4 layers): {count_params(transformer):,}")
print(f"  CTM (2 layers × 4 ticks): {count_params(ctm):,}")

# =============================================================================
# OBSERVATION 10: Effective depth analysis
# =============================================================================

print("\n" + "=" * 80)
print("OBSERVATION 10: Effective Computation Depth")
print("=" * 80)

print("""
Transformer: 4 unique layers, 4 forward passes through different weights
CTM: 2 unique layers, but 4 ticks = 8 total passes through same 2 layers

Key difference:
- Transformer: 4 different transformations
- CTM: 2 transformations repeated 4 times with history

The CTM is applying the SAME transformation repeatedly, but with:
1. Residual connections (h = h + h_new)
2. History information (0.1 * history_proj([h, h_prev]))

This is like an iterative refinement process vs a sequential processing pipeline.
""")

print("=" * 80)
print("OBSERVATIONS COMPLETE")
print("=" * 80)
