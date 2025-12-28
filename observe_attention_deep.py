"""
DEEP OBSERVATION: What exactly changes in attention between ticks?

Looking at:
1. Actual attention patterns (not just Q/K vectors)
2. Which positions change attention most
3. What pattern is being corrected
4. Orthogonal components: FFN, per-layer, per-head
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
print("DEEP OBSERVATION: Attention Pattern Changes")
print("=" * 80)

# =============================================================================
# CTM with attention capture
# =============================================================================

class CTMWithAttentionCapture(nn.Module):
    def __init__(self, vocab_size, hidden_dim=64, n_heads=4, n_ticks=4):
        super().__init__()
        self.n_ticks = n_ticks
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads

        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embedding = nn.Embedding(256, hidden_dim)

        # Manual attention for capture
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )

        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)

        # History
        self.history_proj = nn.Linear(hidden_dim * 2, hidden_dim)

        self.ln_f = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, vocab_size)

        # Storage for analysis
        self.attention_patterns = []
        self.q_vectors = []
        self.k_vectors = []
        self.ffn_activations = []

    def forward(self, x, capture=False):
        if capture:
            self.attention_patterns = []
            self.q_vectors = []
            self.k_vectors = []
            self.ffn_activations = []

        batch_size, seq_len = x.shape
        pos = torch.arange(seq_len, device=x.device).unsqueeze(0)
        h = self.embedding(x) + self.pos_embedding(pos)
        h_prev = torch.zeros_like(h)

        for tick in range(self.n_ticks):
            # History
            history = self.history_proj(torch.cat([h, h_prev], dim=-1))
            h_input = h + 0.1 * history

            # Attention
            h_normed = self.ln1(h_input)
            q = self.q_proj(h_normed)
            k = self.k_proj(h_normed)
            v = self.v_proj(h_normed)

            if capture:
                self.q_vectors.append(q.detach().clone())
                self.k_vectors.append(k.detach().clone())

            # Multi-head attention
            q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
            k = k.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
            v = v.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

            attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            attn_weights = F.softmax(attn_scores, dim=-1)

            if capture:
                self.attention_patterns.append(attn_weights.detach().clone())

            attn_out = torch.matmul(attn_weights, v)
            attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
            attn_out = self.o_proj(attn_out)

            h_input = h_input + attn_out

            # FFN
            h_normed = self.ln2(h_input)
            ffn_out = self.ffn(h_normed)

            if capture:
                self.ffn_activations.append(ffn_out.detach().clone())

            h_new = h_input + ffn_out

            h_prev = h.detach()
            h = h + h_new

        return self.head(self.ln_f(h))


# =============================================================================
# Task
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

# Train model
model = CTMWithAttentionCapture(vocab_size)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss(ignore_index=0)

print("\nTraining model...")
for epoch in range(100):
    model.train()
    logits = model(train_data)
    loss = criterion(logits.view(-1, logits.size(-1)), train_labels.view(-1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print("Model trained.")

# =============================================================================
# OBSERVATION 1: Attention pattern changes
# =============================================================================

print("\n" + "=" * 80)
print("OBSERVATION 1: How do attention patterns change across ticks?")
print("=" * 80)

model.eval()
with torch.no_grad():
    _ = model(test_data[:20], capture=True)

# Analyze attention patterns
# attn_patterns[tick] has shape [batch, heads, seq, seq]

print("\nAttention pattern statistics per tick:")
for tick, attn in enumerate(model.attention_patterns):
    entropy = -(attn * torch.log(attn + 1e-10)).sum(dim=-1).mean()
    max_attn = attn.max(dim=-1)[0].mean()
    print(f"  Tick {tick}: entropy={entropy:.4f}, max_weight={max_attn:.4f}")

print("\nAttention pattern change (L1 distance between consecutive ticks):")
for i in range(1, len(model.attention_patterns)):
    prev = model.attention_patterns[i-1]
    curr = model.attention_patterns[i]
    l1_dist = (curr - prev).abs().mean()
    print(f"  Tick {i-1} → Tick {i}: L1={l1_dist:.4f}")

# =============================================================================
# OBSERVATION 2: Which positions change attention most?
# =============================================================================

print("\n" + "=" * 80)
print("OBSERVATION 2: Which positions change attention most?")
print("=" * 80)

# Position meanings: 0=a, 1=+, 2=b, 3=*, 4=c, 5==, 6=out1, 7=out2
pos_names = ['a', '+', 'b', '*', 'c', '=', 'out1', 'out2']

# Change from tick 0 to tick 1 (where the big jump happens)
attn_0 = model.attention_patterns[0]  # [batch, heads, seq, seq]
attn_1 = model.attention_patterns[1]

change = (attn_1 - attn_0).abs().mean(dim=(0, 1))  # [seq, seq] - avg over batch and heads

print("\nAttention change matrix (tick 0 → tick 1):")
print("Query\\Key", end="")
for name in pos_names:
    print(f"{name:>6}", end="")
print()

for i, name in enumerate(pos_names):
    print(f"{name:>6}   ", end="")
    for j in range(len(pos_names)):
        print(f"{change[i, j]:.3f} ", end="")
    print()

# Which query position changes most?
query_change = change.sum(dim=1)
print(f"\nTotal change per query position:")
for i, name in enumerate(pos_names):
    print(f"  {name}: {query_change[i]:.4f}")

# =============================================================================
# OBSERVATION 3: What do output positions attend to?
# =============================================================================

print("\n" + "=" * 80)
print("OBSERVATION 3: What does OUTPUT position attend to at each tick?")
print("=" * 80)

# Position 6 is the main output
for tick, attn in enumerate(model.attention_patterns):
    # Average over batch and heads
    attn_avg = attn.mean(dim=(0, 1))  # [seq, seq]
    out_attn = attn_avg[6]  # What pos 6 attends to

    print(f"\nTick {tick} - Position 'out1' attends to:")
    for i, name in enumerate(pos_names):
        print(f"  {name}: {out_attn[i]:.4f}")

# =============================================================================
# OBSERVATION 4: Per-head analysis
# =============================================================================

print("\n" + "=" * 80)
print("OBSERVATION 4: Do different heads specialize?")
print("=" * 80)

# Compare heads at tick 0 vs tick 1
attn_0 = model.attention_patterns[0]  # [batch, heads, seq, seq]
attn_1 = model.attention_patterns[1]

print("\nPer-head attention change (tick 0 → tick 1):")
for head in range(model.n_heads):
    head_change = (attn_1[:, head] - attn_0[:, head]).abs().mean()
    print(f"  Head {head}: {head_change:.4f}")

# What does each head focus on at tick 1?
print("\nWhat each head focuses on at tick 1 (from output position):")
attn_1_avg = attn_1.mean(dim=0)  # [heads, seq, seq]
for head in range(model.n_heads):
    out_attn = attn_1_avg[head, 6]  # Head's attention from output pos
    max_pos = out_attn.argmax().item()
    print(f"  Head {head}: max attention to '{pos_names[max_pos]}' ({out_attn[max_pos]:.3f})")

# =============================================================================
# OBSERVATION 5: FFN activation changes
# =============================================================================

print("\n" + "=" * 80)
print("OBSERVATION 5: How do FFN activations change?")
print("=" * 80)

print("\nFFN activation magnitude per tick:")
for tick, ffn in enumerate(model.ffn_activations):
    mag = ffn.norm(dim=-1).mean()
    print(f"  Tick {tick}: {mag:.4f}")

print("\nFFN activation change (cosine similarity between ticks):")
for i in range(1, len(model.ffn_activations)):
    prev = model.ffn_activations[i-1].flatten()
    curr = model.ffn_activations[i].flatten()
    cos_sim = F.cosine_similarity(prev.unsqueeze(0), curr.unsqueeze(0))
    print(f"  Tick {i-1} → Tick {i}: cos_sim={cos_sim.item():.4f}")

# =============================================================================
# OBSERVATION 6: Which examples improve at tick 1?
# =============================================================================

print("\n" + "=" * 80)
print("OBSERVATION 6: Which examples improve at tick 1?")
print("=" * 80)

# Get predictions at each tick
model.eval()
with torch.no_grad():
    _ = model(test_data, capture=True)

    # For each tick, what would the prediction be?
    tick_preds = []
    pos = torch.arange(test_data.shape[1]).unsqueeze(0)
    h = model.embedding(test_data) + model.pos_embedding(pos)
    h_prev = torch.zeros_like(h)

    for tick in range(model.n_ticks):
        history = model.history_proj(torch.cat([h, h_prev], dim=-1))
        h_input = h + 0.1 * history

        h_normed = model.ln1(h_input)
        q = model.q_proj(h_normed)
        k = model.k_proj(h_normed)
        v = model.v_proj(h_normed)

        batch_size, seq_len = test_data.shape
        q = q.view(batch_size, seq_len, model.n_heads, model.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, model.n_heads, model.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, model.n_heads, model.head_dim).transpose(1, 2)

        attn = F.softmax(torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(model.head_dim), dim=-1)
        attn_out = torch.matmul(attn, v)
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, seq_len, model.hidden_dim)
        attn_out = model.o_proj(attn_out)
        h_input = h_input + attn_out

        h_normed = model.ln2(h_input)
        ffn_out = model.ffn(h_normed)
        h_new = h_input + ffn_out

        h_prev = h.detach()
        h = h + h_new

        # Prediction at this tick
        logits = model.head(model.ln_f(h))
        preds = logits.argmax(dim=-1)
        tick_preds.append(preds)

# Compare tick 0 and tick 1 predictions
correct_0 = (tick_preds[0] == test_labels) & (test_labels != 0)
correct_1 = (tick_preds[1] == test_labels) & (test_labels != 0)

improved = (~correct_0 & correct_1)  # Wrong at 0, right at 1
regressed = (correct_0 & ~correct_1)  # Right at 0, wrong at 1
stayed_wrong = (~correct_0 & ~correct_1)
stayed_right = (correct_0 & correct_1)

print(f"\nFrom tick 0 to tick 1:")
print(f"  Improved (wrong→right): {improved.sum().item()}")
print(f"  Regressed (right→wrong): {regressed.sum().item()}")
print(f"  Stayed wrong: {stayed_wrong.sum().item()}")
print(f"  Stayed right: {stayed_right.sum().item()}")

# =============================================================================
# KEY INSIGHT
# =============================================================================

print("\n" + "=" * 80)
print("SYNTHESIS: What is the pattern?")
print("=" * 80)

print("""
Looking for patterns in:
1. Which positions change attention most
2. What the output position attends to differently
3. Whether specific heads specialize
4. What types of examples improve

The rug-puller insight would be:
- If attention correction follows a SYSTEMATIC pattern
- That pattern could potentially be achieved WITHOUT iteration
- By building that correction into the architecture itself
""")

print("=" * 80)
