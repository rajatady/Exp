"""
CTM with Structured Accumulator Memory

Hypothesis: CTM fails on some tasks because it has TIME (iteration) but not SPACE (memory).
The history mechanism is too weak - just echoes of past states.

This experiment tests: What if CTM has a SEPARATE memory bank that:
1. Persists across ticks (not overwritten)
2. Can be written to (accumulate results)
3. Can be read from (use accumulated results)
4. Is separate from input (data vs computation)
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
print("CTM WITH STRUCTURED ACCUMULATOR MEMORY")
print("=" * 80)

# =============================================================================
# ARCHITECTURE 1: Standard Transformer (baseline)
# =============================================================================

class StandardTransformer(nn.Module):
    """Single pass, no iteration, no memory."""
    def __init__(self, vocab_size, hidden_dim=64, n_layers=2, n_heads=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embedding = nn.Embedding(128, hidden_dim)

        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim, nhead=n_heads,
                dim_feedforward=hidden_dim * 4, batch_first=True
            ) for _ in range(n_layers)
        ])

        self.ln_f = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, vocab_size)
        self.name = "Transformer"

    def forward(self, x):
        pos = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
        h = self.embedding(x) + self.pos_embedding(pos)
        for block in self.blocks:
            h = block(h)
        return self.head(self.ln_f(h))


# =============================================================================
# ARCHITECTURE 2: CTM with History (iteration only)
# =============================================================================

class CTMHistory(nn.Module):
    """Iteration with weak history - current approach."""
    def __init__(self, vocab_size, hidden_dim=64, n_layers=2, n_heads=4, n_ticks=4):
        super().__init__()
        self.n_ticks = n_ticks
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embedding = nn.Embedding(128, hidden_dim)

        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim, nhead=n_heads,
                dim_feedforward=hidden_dim * 4, batch_first=True
            ) for _ in range(n_layers)
        ])

        # History: just concat previous states
        self.history_proj = nn.Linear(hidden_dim * 2, hidden_dim)

        self.ln_f = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, vocab_size)
        self.name = "CTM-History"

    def forward(self, x):
        pos = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
        h = self.embedding(x) + self.pos_embedding(pos)

        h_prev = torch.zeros_like(h)

        for tick in range(self.n_ticks):
            # Weak history: just previous state
            history = self.history_proj(torch.cat([h, h_prev], dim=-1))
            h_input = h + 0.1 * history

            h_new = h_input
            for block in self.blocks:
                h_new = block(h_new)

            h_prev = h.detach()
            h = h + h_new  # Residual

        return self.head(self.ln_f(h))


# =============================================================================
# ARCHITECTURE 3: CTM with Accumulator (iteration + structured memory)
# =============================================================================

class CTMAccumulator(nn.Module):
    """
    Iteration with STRUCTURED memory.

    Key differences from CTM-History:
    1. Memory is SEPARATE from hidden state
    2. Memory ACCUMULATES (write operations add, don't overwrite)
    3. Memory is ADDRESSABLE (read via attention, not just concat)
    4. Input is READ-ONLY (separation of data and computation)
    """
    def __init__(self, vocab_size, hidden_dim=64, n_layers=2, n_heads=4,
                 n_ticks=4, memory_slots=16):
        super().__init__()
        self.n_ticks = n_ticks
        self.hidden_dim = hidden_dim
        self.memory_slots = memory_slots

        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embedding = nn.Embedding(128, hidden_dim)

        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim, nhead=n_heads,
                dim_feedforward=hidden_dim * 4, batch_first=True
            ) for _ in range(n_layers)
        ])

        # Memory operations
        self.memory_query = nn.Linear(hidden_dim, hidden_dim)  # For reading
        self.memory_key = nn.Linear(hidden_dim, hidden_dim)    # Memory addressing
        self.memory_value = nn.Linear(hidden_dim, hidden_dim)  # What to write
        self.write_gate = nn.Sequential(                        # How much to write
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        # Combine input attention + memory attention
        self.combine = nn.Linear(hidden_dim * 2, hidden_dim)

        self.ln_f = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, vocab_size)
        self.name = "CTM-Accumulator"

    def forward(self, x):
        batch_size, seq_len = x.shape
        device = x.device

        # Input embedding (READ-ONLY - we never modify this)
        pos = torch.arange(seq_len, device=device).unsqueeze(0)
        input_embed = self.embedding(x) + self.pos_embedding(pos)

        # Initialize memory bank (empty)
        memory = torch.zeros(batch_size, self.memory_slots, self.hidden_dim, device=device)
        memory_mask = torch.zeros(batch_size, self.memory_slots, device=device)  # Track written slots

        # Working state (separate from input)
        h = input_embed.clone()

        for tick in range(self.n_ticks):
            # === READ from memory ===
            # Query: what does each position want to read?
            q = self.memory_query(h)  # [batch, seq, dim]
            # Keys: what's in memory?
            k = self.memory_key(memory)  # [batch, slots, dim]

            # Attention over memory
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.hidden_dim)
            # Mask unwritten slots
            attn_scores = attn_scores.masked_fill(memory_mask.unsqueeze(1) == 0, float('-inf'))

            # If memory is empty, just use zeros
            if memory_mask.sum() > 0:
                attn_weights = F.softmax(attn_scores, dim=-1)
                attn_weights = torch.nan_to_num(attn_weights, 0.0)
                memory_read = torch.matmul(attn_weights, memory)
            else:
                memory_read = torch.zeros_like(h)

            # Combine input + memory read
            combined = self.combine(torch.cat([h, memory_read], dim=-1))

            # === PROCESS ===
            h_new = combined
            for block in self.blocks:
                h_new = block(h_new)
            h = h + h_new

            # === WRITE to memory ===
            # Decide what to write
            write_content = self.memory_value(h)  # [batch, seq, dim]
            write_strength = self.write_gate(h)   # [batch, seq, 1]

            # Write to memory slots via soft attention (no in-place ops)
            # Compute write addresses for all positions
            write_logits = torch.matmul(h, memory.transpose(-2, -1)) / math.sqrt(self.hidden_dim)
            write_weights = F.softmax(write_logits, dim=-1)  # [batch, seq, slots]

            # Aggregate what to write to each slot
            # [batch, seq, 1] * [batch, seq, dim] -> weighted content per position
            weighted_content = write_strength * write_content  # [batch, seq, dim]
            # [batch, slots, seq] @ [batch, seq, dim] -> [batch, slots, dim]
            memory_update = torch.matmul(write_weights.transpose(-2, -1), weighted_content)

            # Accumulate (not in-place)
            memory = memory + memory_update
            memory_mask = torch.ones(batch_size, self.memory_slots, device=device)

        return self.head(self.ln_f(h))


# =============================================================================
# TASKS THAT REQUIRE ACCUMULATION
# =============================================================================

def generate_reversal(n_samples, vocab_size=20, length=8):
    """Reversal requires accumulating the reversed sequence."""
    data, labels = [], []
    max_len = length * 2
    for _ in range(n_samples):
        seq = [random.randint(1, vocab_size-2) for _ in range(length)]
        sep, pad = vocab_size - 1, 0
        inp = seq + [sep] + [pad] * (max_len - length - 1)
        target = seq[::-1] + [pad] * (max_len - length)
        data.append(inp)
        labels.append(target)
    return torch.tensor(data), torch.tensor(labels)


def generate_multi_step_reverse(n_samples, vocab_size=20, length=6):
    """Reverse THEN sort - requires two sequential operations."""
    data, labels = [], []
    max_len = length * 2
    for _ in range(n_samples):
        seq = [random.randint(1, min(9, vocab_size-2)) for _ in range(length)]
        sep, pad = vocab_size - 1, 0
        inp = seq + [sep] + [pad] * (max_len - length - 1)
        # First reverse, then sort
        result = sorted(seq[::-1])
        target = result + [pad] * (max_len - length)
        data.append(inp)
        labels.append(target)
    return torch.tensor(data), torch.tensor(labels)


def generate_parity_accumulation(n_samples, vocab_size=20, length=8):
    """
    Output running parity at each position.
    Requires accumulating state across sequence.
    """
    data, labels = [], []
    for _ in range(n_samples):
        # Binary sequence
        seq = [random.randint(0, 1) for _ in range(length)]
        # Use tokens 1 and 2 for 0 and 1
        inp = [s + 1 for s in seq]
        # Running parity: XOR of all elements so far
        parity = 0
        target = []
        for s in seq:
            parity ^= s
            target.append(parity + 1)  # 1 or 2
        data.append(inp)
        labels.append(target)
    return torch.tensor(data), torch.tensor(labels)


# =============================================================================
# TRAINING
# =============================================================================

def train_and_eval(model, train_data, train_labels, test_data, test_labels,
                   n_epochs=100, lr=0.001):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    for epoch in range(n_epochs):
        model.train()
        logits = model(train_data)
        loss = criterion(logits.view(-1, logits.size(-1)), train_labels.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Evaluate
    model.eval()
    with torch.no_grad():
        logits = model(test_data)
        preds = logits.argmax(dim=-1)
        mask = test_labels != 0
        correct = (preds == test_labels) & mask
        acc = correct.sum().float() / mask.sum().float()

    return acc.item() * 100


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

print("\n" + "=" * 80)
print("EXPERIMENT: Does Structured Memory Help?")
print("=" * 80)

vocab_size = 20
n_train = 500
n_test = 200

tasks = {
    'Reversal-8': lambda n, v: generate_reversal(n, v, length=8),
    'MultiStep': lambda n, v: generate_multi_step_reverse(n, v, length=6),
    'Parity': lambda n, v: generate_parity_accumulation(n, v, length=8),
}

results = {name: {} for name in tasks}

for task_name, gen_fn in tasks.items():
    print(f"\n--- {task_name} ---")

    train_data, train_labels = gen_fn(n_train, vocab_size)
    test_data, test_labels = gen_fn(n_test, vocab_size)

    for ModelClass in [StandardTransformer, CTMHistory, CTMAccumulator]:
        model = ModelClass(vocab_size, hidden_dim=64, n_layers=2, n_heads=4)
        if hasattr(model, 'n_ticks'):
            model.n_ticks = 4

        acc = train_and_eval(model, train_data, train_labels,
                            test_data, test_labels, n_epochs=100)
        results[task_name][model.name] = acc
        print(f"  {model.name:<18}: {acc:5.1f}%")

# =============================================================================
# ANALYSIS
# =============================================================================

print("\n" + "=" * 80)
print("RESULTS SUMMARY")
print("=" * 80)

print(f"\n{'Task':<15} {'Transformer':>12} {'CTM-History':>14} {'CTM-Accum':>12} {'Δ Accum':>10}")
print("-" * 65)

for task_name in tasks:
    t = results[task_name].get('Transformer', 0)
    h = results[task_name].get('CTM-History', 0)
    a = results[task_name].get('CTM-Accumulator', 0)
    delta = a - max(t, h)
    delta_str = f"+{delta:.1f}" if delta > 0 else f"{delta:.1f}"
    print(f"{task_name:<15} {t:>11.1f}% {h:>13.1f}% {a:>11.1f}% {delta_str:>10}")

print("\n" + "=" * 80)
print("INTERPRETATION")
print("=" * 80)

print("""
If CTM-Accumulator >> CTM-History:
  → Structured memory IS the missing piece
  → Iteration alone isn't enough; need SPACE to accumulate

If CTM-Accumulator ≈ CTM-History:
  → Memory isn't the bottleneck
  → Need to look elsewhere (attention patterns, training dynamics)

If CTM-History >> Transformer but CTM-Accumulator doesn't improve further:
  → Iteration is valuable, memory is neutral
  → Current history mechanism is already sufficient
""")

# =============================================================================
# OOD GENERALIZATION TEST
# =============================================================================

print("\n" + "=" * 80)
print("OOD GENERALIZATION: Longer Parity Sequences")
print("=" * 80)

# Train on length 8, test on length 12
train_data, train_labels = generate_parity_accumulation(n_train, vocab_size, length=8)

# Retrain models
models = {}
for ModelClass in [StandardTransformer, CTMHistory, CTMAccumulator]:
    model = ModelClass(vocab_size, hidden_dim=64, n_layers=2, n_heads=4)
    if hasattr(model, 'n_ticks'):
        model.n_ticks = 4

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    for epoch in range(100):
        model.train()
        logits = model(train_data)
        loss = criterion(logits.view(-1, logits.size(-1)), train_labels.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    models[model.name] = model

# Test on length 12 (OOD)
test_ood, labels_ood = generate_parity_accumulation(200, vocab_size, length=12)

print(f"\nTrained on length 8, tested on length 12:")
for name, model in models.items():
    model.eval()
    with torch.no_grad():
        logits = model(test_ood)
        preds = logits.argmax(dim=-1)
        # For parity, all positions matter
        correct = (preds == labels_ood)
        acc = correct.float().mean()
    print(f"  {name:<18}: {acc.item()*100:5.1f}%")

print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)
