"""
DEFINITIVE TEST: Stack Operations

A task that CANNOT be solved without structured memory:
- Push values onto a stack
- Pop them in LIFO order

This requires:
1. REMEMBERING what was pushed (not just attending to input)
2. ORDERING the output correctly (LIFO, not input order)
3. MAINTAINING state across the sequence

If accumulator memory helps here, it's real.
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
print("DEFINITIVE TEST: Stack Operations (LIFO)")
print("=" * 80)

# =============================================================================
# MODELS (same as before)
# =============================================================================

class StandardTransformer(nn.Module):
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


class CTMHistory(nn.Module):
    def __init__(self, vocab_size, hidden_dim=64, n_layers=2, n_heads=4, n_ticks=6):
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
        self.history_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.ln_f = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, vocab_size)
        self.name = "CTM-History"

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


class CTMAccumulator(nn.Module):
    def __init__(self, vocab_size, hidden_dim=64, n_layers=2, n_heads=4,
                 n_ticks=6, memory_slots=16):
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

        self.memory_query = nn.Linear(hidden_dim, hidden_dim)
        self.memory_key = nn.Linear(hidden_dim, hidden_dim)
        self.memory_value = nn.Linear(hidden_dim, hidden_dim)
        self.write_gate = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())
        self.combine = nn.Linear(hidden_dim * 2, hidden_dim)
        self.ln_f = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, vocab_size)
        self.name = "CTM-Accumulator"

    def forward(self, x):
        batch_size, seq_len = x.shape
        device = x.device

        pos = torch.arange(seq_len, device=device).unsqueeze(0)
        input_embed = self.embedding(x) + self.pos_embedding(pos)
        memory = torch.zeros(batch_size, self.memory_slots, self.hidden_dim, device=device)
        h = input_embed.clone()

        for tick in range(self.n_ticks):
            # Read from memory
            q = self.memory_query(h)
            k = self.memory_key(memory)
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.hidden_dim)
            attn_weights = F.softmax(attn_scores, dim=-1)
            memory_read = torch.matmul(attn_weights, memory)

            combined = self.combine(torch.cat([h, memory_read], dim=-1))
            h_new = combined
            for block in self.blocks:
                h_new = block(h_new)
            h = h + h_new

            # Write to memory
            write_content = self.memory_value(h)
            write_strength = self.write_gate(h)
            write_logits = torch.matmul(h, memory.transpose(-2, -1)) / math.sqrt(self.hidden_dim)
            write_weights = F.softmax(write_logits, dim=-1)
            weighted_content = write_strength * write_content
            memory_update = torch.matmul(write_weights.transpose(-2, -1), weighted_content)
            memory = memory + memory_update

        return self.head(self.ln_f(h))


# =============================================================================
# STACK TASK
# =============================================================================

def generate_stack_data(n_samples, vocab_size=20, n_pushes=4):
    """
    Stack operations:
    Input:  PUSH v1 PUSH v2 PUSH v3 PUSH v4 POP POP POP POP
    Output: PAD PAD PAD PAD PAD PAD PAD PAD v4 v3 v2 v1

    This REQUIRES memory - you can't solve this with just attention to input
    because the OUTPUT ORDER is REVERSED from input order.
    """
    PUSH = vocab_size - 1
    POP = vocab_size - 2
    PAD = 0

    data, labels = [], []

    for _ in range(n_samples):
        # Random values to push (1 to vocab_size-3)
        values = [random.randint(1, vocab_size-3) for _ in range(n_pushes)]

        # Input: PUSH v1 PUSH v2 ... POP POP ...
        inp = []
        for v in values:
            inp.extend([PUSH, v])
        for _ in range(n_pushes):
            inp.append(POP)

        # Output: PAD for pushes, then values in REVERSE order
        target = [PAD] * (n_pushes * 2)  # Padding during push phase
        target.extend(values[::-1])       # Pop in LIFO order

        data.append(inp)
        labels.append(target)

    return torch.tensor(data), torch.tensor(labels)


def generate_nested_stack(n_samples, vocab_size=20):
    """
    Nested push-pop:
    PUSH a PUSH b POP PUSH c POP POP

    Output: b c a

    Even harder - interleaved operations.
    """
    PUSH = vocab_size - 1
    POP = vocab_size - 2
    PAD = 0

    data, labels = [], []

    for _ in range(n_samples):
        # Generate random sequence of push/pop operations
        stack = []
        inp = []
        output = []

        n_ops = random.randint(4, 8)
        for _ in range(n_ops):
            if len(stack) == 0 or (len(stack) < 4 and random.random() < 0.6):
                # Push
                v = random.randint(1, vocab_size-3)
                stack.append(v)
                inp.extend([PUSH, v])
                output.extend([PAD, PAD])
            else:
                # Pop
                v = stack.pop()
                inp.append(POP)
                output.append(v)

        # Pop remaining
        while stack:
            v = stack.pop()
            inp.append(POP)
            output.append(v)

        # Pad to fixed length
        max_len = 20
        inp = inp[:max_len] + [PAD] * (max_len - len(inp))
        output = output[:max_len] + [PAD] * (max_len - len(output))

        data.append(inp[:max_len])
        labels.append(output[:max_len])

    return torch.tensor(data), torch.tensor(labels)


# =============================================================================
# TRAINING
# =============================================================================

def train_and_eval(model, train_data, train_labels, test_data, test_labels,
                   n_epochs=150, lr=0.001):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    for epoch in range(n_epochs):
        model.train()
        logits = model(train_data)
        loss = criterion(logits.view(-1, logits.size(-1)), train_labels.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 50 == 0:
            model.eval()
            with torch.no_grad():
                test_logits = model(test_data)
                preds = test_logits.argmax(dim=-1)
                mask = test_labels != 0
                correct = (preds == test_labels) & mask
                acc = correct.sum().float() / mask.sum().float()
            print(f"  {model.name} Epoch {epoch}: {acc.item()*100:.1f}%")

    model.eval()
    with torch.no_grad():
        logits = model(test_data)
        preds = logits.argmax(dim=-1)
        mask = test_labels != 0
        correct = (preds == test_labels) & mask
        acc = correct.sum().float() / mask.sum().float()

    return acc.item() * 100


# =============================================================================
# EXPERIMENT
# =============================================================================

print("\n" + "=" * 80)
print("TEST 1: Simple Stack (PUSH PUSH PUSH PUSH POP POP POP POP)")
print("=" * 80)

vocab_size = 20
n_train = 1000
n_test = 200

train_data, train_labels = generate_stack_data(n_train, vocab_size, n_pushes=4)
test_data, test_labels = generate_stack_data(n_test, vocab_size, n_pushes=4)

print(f"\nExample:")
print(f"  Input:  {train_data[0].tolist()}")
print(f"  Output: {train_labels[0].tolist()}")
print(f"  (PUSH=19, POP=18, values are 1-17)")

for ModelClass in [StandardTransformer, CTMHistory, CTMAccumulator]:
    print(f"\nTraining {ModelClass.__name__}...")
    model = ModelClass(vocab_size, hidden_dim=64, n_layers=2, n_heads=4)
    acc = train_and_eval(model, train_data, train_labels, test_data, test_labels,
                        n_epochs=150)
    print(f"  Final: {acc:.1f}%")


print("\n" + "=" * 80)
print("TEST 2: Nested Stack (interleaved PUSH/POP)")
print("=" * 80)

train_data, train_labels = generate_nested_stack(n_train, vocab_size)
test_data, test_labels = generate_nested_stack(n_test, vocab_size)

print(f"\nExample:")
print(f"  Input:  {train_data[0].tolist()}")
print(f"  Output: {train_labels[0].tolist()}")

for ModelClass in [StandardTransformer, CTMHistory, CTMAccumulator]:
    print(f"\nTraining {ModelClass.__name__}...")
    model = ModelClass(vocab_size, hidden_dim=64, n_layers=2, n_heads=4)
    acc = train_and_eval(model, train_data, train_labels, test_data, test_labels,
                        n_epochs=150)
    print(f"  Final: {acc:.1f}%")


print("\n" + "=" * 80)
print("TEST 3: OOD - Train on 4 pushes, test on 6")
print("=" * 80)

train_data, train_labels = generate_stack_data(n_train, vocab_size, n_pushes=4)
test_ood, labels_ood = generate_stack_data(n_test, vocab_size, n_pushes=6)

results_ood = {}
for ModelClass in [StandardTransformer, CTMHistory, CTMAccumulator]:
    model = ModelClass(vocab_size, hidden_dim=64, n_layers=2, n_heads=4)
    # Train
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    for epoch in range(150):
        model.train()
        logits = model(train_data)
        loss = criterion(logits.view(-1, logits.size(-1)), train_labels.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Test OOD
    model.eval()
    with torch.no_grad():
        logits = model(test_ood)
        preds = logits.argmax(dim=-1)
        mask = labels_ood != 0
        correct = (preds == labels_ood) & mask
        acc = correct.sum().float() / mask.sum().float()
    results_ood[model.name] = acc.item() * 100
    print(f"  {model.name}: {acc.item()*100:.1f}%")


print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)

print("""
Stack operations are a DEFINITIVE test for structured memory because:
1. Output order is REVERSED from input order
2. Can't be solved by simple pattern matching
3. Requires REMEMBERING pushed values
4. Requires RETRIEVING in correct (LIFO) order

If CTM-Accumulator significantly outperforms:
  → Structured memory IS the key
  → We've found the architectural improvement needed

If all models are similar:
  → Either memory isn't being learned effectively
  → Or the task can still be solved positionally somehow
""")
