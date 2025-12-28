"""
CTM with Automatic Log (No Decisions)

Key insight: Don't add decisions (read/write). Remove them.
- Every tick AUTOMATICALLY writes to a log (structural, not learned)
- Nothing is forgotten (log grows)
- Reading is standard attention (existing mechanism)

Like residual connections: don't decide whether to pass info → always pass.
Like this: don't decide whether to write → always write.
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
print("CTM WITH AUTOMATIC LOG (NO DECISIONS)")
print("=" * 80)

# =============================================================================
# ARCHITECTURE 1: Standard Transformer
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
        self.name = "Transformer"

    def forward(self, x):
        pos = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
        h = self.embedding(x) + self.pos_embedding(pos)
        for block in self.blocks:
            h = block(h)
        return self.head(self.ln_f(h))


# =============================================================================
# ARCHITECTURE 2: CTM with History (baseline CTM)
# =============================================================================

class CTMHistory(nn.Module):
    def __init__(self, vocab_size, hidden_dim=64, n_layers=2, n_heads=4, n_ticks=8):
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


# =============================================================================
# ARCHITECTURE 3: CTM with Automatic Log (THE NEW IDEA)
# =============================================================================

class CTMAutoLog(nn.Module):
    """
    Every tick AUTOMATICALLY writes a summary to the log.
    No decision about whether to write. Always write.

    Structure:
    - Each tick produces a small summary (bottleneck forces diversity)
    - Summary is appended to log
    - Attention operates over [input tokens] + [log entries]

    This is like residual connections across ticks:
    - Residual: always pass information through layers
    - AutoLog: always pass information through ticks
    """
    def __init__(self, vocab_size, hidden_dim=64, n_layers=2, n_heads=4,
                 n_ticks=8, summary_dim=16):
        super().__init__()
        self.n_ticks = n_ticks
        self.hidden_dim = hidden_dim
        self.summary_dim = summary_dim

        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embedding = nn.Embedding(256, hidden_dim)

        # Main transformer blocks
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim, nhead=n_heads,
                dim_feedforward=hidden_dim * 4, batch_first=True
            ) for _ in range(n_layers)
        ])

        # Summary extractor: bottleneck forces each tick to contribute unique info
        self.summary_proj = nn.Linear(hidden_dim, summary_dim)

        # Expand summary back to hidden dim for attention
        self.summary_expand = nn.Linear(summary_dim, hidden_dim)

        # Cross-attention to read from log
        self.log_query = nn.Linear(hidden_dim, hidden_dim)
        self.log_key = nn.Linear(hidden_dim, hidden_dim)
        self.log_value = nn.Linear(hidden_dim, hidden_dim)

        self.ln_f = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, vocab_size)
        self.name = "CTM-AutoLog"

    def forward(self, x):
        batch_size, seq_len = x.shape
        device = x.device

        pos = torch.arange(seq_len, device=device).unsqueeze(0)
        h = self.embedding(x) + self.pos_embedding(pos)

        # Log starts empty - will accumulate summaries from each tick
        # Shape: [batch, num_entries, hidden_dim]
        log = None

        for tick in range(self.n_ticks):
            # Read from log if it exists
            if log is not None:
                # Cross-attention: h queries the log
                q = self.log_query(h)  # [batch, seq, hidden]
                k = self.log_key(log)   # [batch, entries, hidden]
                v = self.log_value(log) # [batch, entries, hidden]

                attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.hidden_dim)
                attn_weights = F.softmax(attn_scores, dim=-1)
                log_read = torch.matmul(attn_weights, v)  # [batch, seq, hidden]

                h = h + 0.1 * log_read  # Add log info to hidden state

            # Process through transformer blocks
            h_new = h
            for block in self.blocks:
                h_new = block(h_new)
            h = h + h_new

            # AUTOMATIC WRITE: Extract summary and append to log
            # No decision about whether to write - always write
            summary = self.summary_proj(h)  # [batch, seq, summary_dim] - bottleneck
            summary_expanded = self.summary_expand(summary)  # [batch, seq, hidden]

            if log is None:
                log = summary_expanded
            else:
                log = torch.cat([log, summary_expanded], dim=1)  # Grow the log

        return self.head(self.ln_f(h))


# =============================================================================
# ARCHITECTURE 4: CTM-AutoLog with Tick Embeddings (even better)
# =============================================================================

class CTMAutoLogV2(nn.Module):
    """
    V2: Add tick embeddings so the model knows WHICH tick's summary it's reading.
    This helps distinguish "what I thought at tick 2" from "what I thought at tick 5".
    """
    def __init__(self, vocab_size, hidden_dim=64, n_layers=2, n_heads=4,
                 n_ticks=8, summary_dim=16):
        super().__init__()
        self.n_ticks = n_ticks
        self.hidden_dim = hidden_dim
        self.summary_dim = summary_dim

        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embedding = nn.Embedding(256, hidden_dim)
        self.tick_embedding = nn.Embedding(n_ticks, hidden_dim)  # NEW: tick identity

        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim, nhead=n_heads,
                dim_feedforward=hidden_dim * 4, batch_first=True
            ) for _ in range(n_layers)
        ])

        self.summary_proj = nn.Linear(hidden_dim, summary_dim)
        self.summary_expand = nn.Linear(summary_dim, hidden_dim)

        self.log_query = nn.Linear(hidden_dim, hidden_dim)
        self.log_key = nn.Linear(hidden_dim, hidden_dim)
        self.log_value = nn.Linear(hidden_dim, hidden_dim)

        self.ln_f = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, vocab_size)
        self.name = "CTM-AutoLog-V2"

    def forward(self, x):
        batch_size, seq_len = x.shape
        device = x.device

        pos = torch.arange(seq_len, device=device).unsqueeze(0)
        h = self.embedding(x) + self.pos_embedding(pos)

        log = None
        log_tick_ids = []

        for tick in range(self.n_ticks):
            # Add tick embedding so model knows "what time it is"
            tick_emb = self.tick_embedding(torch.tensor([tick], device=device))
            h_with_tick = h + 0.1 * tick_emb

            # Read from log
            if log is not None:
                q = self.log_query(h_with_tick)
                k = self.log_key(log)
                v = self.log_value(log)

                attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.hidden_dim)
                attn_weights = F.softmax(attn_scores, dim=-1)
                log_read = torch.matmul(attn_weights, v)

                h = h + 0.1 * log_read

            # Process
            h_new = h_with_tick
            for block in self.blocks:
                h_new = block(h_new)
            h = h + h_new

            # AUTOMATIC WRITE with tick embedding
            summary = self.summary_proj(h)
            summary_expanded = self.summary_expand(summary)
            # Add tick info to summary so reader knows when it was written
            summary_with_tick = summary_expanded + 0.1 * tick_emb

            if log is None:
                log = summary_with_tick
            else:
                log = torch.cat([log, summary_with_tick], dim=1)

        return self.head(self.ln_f(h))


# =============================================================================
# HARD TASKS - Need step-by-step reasoning
# =============================================================================

def generate_nested_reversal(n_samples, vocab_size=20, depth=3, seg_len=3):
    """
    Nested reversal: reverse at multiple levels.
    Example (depth=2, seg_len=2):
        [1,2,3,4] -> [[1,2],[3,4]] -> [[2,1],[4,3]] -> [4,3,2,1]

    This requires HIERARCHICAL thinking - can't be solved positionally.
    """
    data, labels = [], []
    total_len = seg_len ** depth

    for _ in range(n_samples):
        # Generate sequence
        seq = [random.randint(1, vocab_size-2) for _ in range(total_len)]

        # Apply nested reversal
        result = seq.copy()
        chunk_size = seg_len
        for d in range(depth):
            new_result = []
            for i in range(0, len(result), chunk_size):
                chunk = result[i:i+chunk_size]
                new_result.extend(chunk[::-1])
            result = new_result
            chunk_size *= seg_len

        sep, pad = vocab_size - 1, 0
        max_len = total_len * 2 + 2
        inp = seq + [sep] + [pad] * (max_len - total_len - 1)
        target = result + [pad] * (max_len - total_len)

        data.append(inp[:max_len])
        labels.append(target[:max_len])

    return torch.tensor(data), torch.tensor(labels)


def generate_multi_operation(n_samples, vocab_size=20, length=8):
    """
    Multiple operations in sequence:
    1. Reverse
    2. Rotate left by 2
    3. Swap pairs

    Requires tracking state through multiple transformations.
    """
    data, labels = [], []

    for _ in range(n_samples):
        seq = [random.randint(1, vocab_size-3) for _ in range(length)]

        # Op 1: Reverse
        result = seq[::-1]
        # Op 2: Rotate left by 2
        result = result[2:] + result[:2]
        # Op 3: Swap adjacent pairs
        swapped = []
        for i in range(0, len(result)-1, 2):
            swapped.extend([result[i+1], result[i]])
        if len(result) % 2 == 1:
            swapped.append(result[-1])
        result = swapped

        sep, pad = vocab_size - 1, 0
        max_len = length * 2 + 2
        inp = seq + [sep] + [pad] * (max_len - length - 1)
        target = result + [pad] * (max_len - length)

        data.append(inp[:max_len])
        labels.append(target[:max_len])

    return torch.tensor(data), torch.tensor(labels)


def generate_longest_increasing(n_samples, vocab_size=20, length=12):
    """
    Find the longest increasing subsequence.
    Output: the LIS itself (padded).

    This is a classic DP problem requiring state tracking.
    """
    data, labels = [], []

    for _ in range(n_samples):
        seq = [random.randint(1, vocab_size-3) for _ in range(length)]

        # Find LIS using DP
        n = len(seq)
        dp = [1] * n
        parent = [-1] * n

        for i in range(1, n):
            for j in range(i):
                if seq[j] < seq[i] and dp[j] + 1 > dp[i]:
                    dp[i] = dp[j] + 1
                    parent[i] = j

        # Reconstruct LIS
        max_len = max(dp)
        max_idx = dp.index(max_len)

        lis = []
        idx = max_idx
        while idx != -1:
            lis.append(seq[idx])
            idx = parent[idx]
        lis = lis[::-1]

        sep, pad = vocab_size - 1, 0
        max_out_len = length * 2 + 2
        inp = seq + [sep] + [pad] * (max_out_len - length - 1)
        target = lis + [pad] * (max_out_len - len(lis))

        data.append(inp[:max_out_len])
        labels.append(target[:max_out_len])

    return torch.tensor(data), torch.tensor(labels)


def generate_expression_eval(n_samples, vocab_size=20):
    """
    Evaluate simple expression: a + b * c (respecting precedence).
    Output: single number result.

    Requires holding intermediate results.
    """
    data, labels = [], []
    PLUS = vocab_size - 2
    TIMES = vocab_size - 3
    EQUALS = vocab_size - 1
    pad = 0

    for _ in range(n_samples):
        a = random.randint(1, 5)
        b = random.randint(1, 5)
        c = random.randint(1, 5)

        # a + b * c
        result = a + b * c

        inp = [a, PLUS, b, TIMES, c, EQUALS, pad, pad]
        # Result can be up to 5 + 5*5 = 30
        target = [pad, pad, pad, pad, pad, pad, result // 10 if result >= 10 else result,
                  result % 10 if result >= 10 else pad]

        data.append(inp)
        labels.append(target)

    return torch.tensor(data), torch.tensor(labels)


# =============================================================================
# TRAINING
# =============================================================================

def train_and_eval(model, train_data, train_labels, test_data, test_labels,
                   n_epochs=200, lr=0.001):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    best_acc = 0
    for epoch in range(n_epochs):
        model.train()
        logits = model(train_data)
        loss = criterion(logits.view(-1, logits.size(-1)), train_labels.view(-1))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if epoch % 50 == 0 or epoch == n_epochs - 1:
            model.eval()
            with torch.no_grad():
                test_logits = model(test_data)
                preds = test_logits.argmax(dim=-1)
                mask = test_labels != 0
                correct = (preds == test_labels) & mask
                acc = correct.sum().float() / mask.sum().float()
                best_acc = max(best_acc, acc.item())
            print(f"  {model.name:<16} Epoch {epoch:3d}: {acc.item()*100:5.1f}%")

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
print("HARD TASK 1: Multi-Operation (Reverse → Rotate → Swap)")
print("=" * 80)

vocab_size = 30
n_train = 2000
n_test = 500

train_data, train_labels = generate_multi_operation(n_train, vocab_size, length=8)
test_data, test_labels = generate_multi_operation(n_test, vocab_size, length=8)

print(f"\nExample:")
print(f"  Input:  {train_data[0].tolist()}")
print(f"  Output: {train_labels[0].tolist()}")

results = {}
for ModelClass in [StandardTransformer, CTMHistory, CTMAutoLog, CTMAutoLogV2]:
    model = ModelClass(vocab_size, hidden_dim=64, n_layers=2, n_heads=4)
    acc = train_and_eval(model, train_data, train_labels, test_data, test_labels, n_epochs=200)
    results[model.name] = acc
    print(f"  → Final: {acc:.1f}%\n")


print("\n" + "=" * 80)
print("HARD TASK 2: Nested Reversal (depth=2, segment=3)")
print("=" * 80)

train_data, train_labels = generate_nested_reversal(n_train, vocab_size, depth=2, seg_len=3)
test_data, test_labels = generate_nested_reversal(n_test, vocab_size, depth=2, seg_len=3)

print(f"\nExample (9 elements, nested 2-level reversal):")
print(f"  Input:  {train_data[0].tolist()}")
print(f"  Output: {train_labels[0].tolist()}")

results2 = {}
for ModelClass in [StandardTransformer, CTMHistory, CTMAutoLog, CTMAutoLogV2]:
    model = ModelClass(vocab_size, hidden_dim=64, n_layers=2, n_heads=4)
    acc = train_and_eval(model, train_data, train_labels, test_data, test_labels, n_epochs=200)
    results2[model.name] = acc
    print(f"  → Final: {acc:.1f}%\n")


print("\n" + "=" * 80)
print("HARD TASK 3: Expression Evaluation (a + b * c)")
print("=" * 80)

train_data, train_labels = generate_expression_eval(n_train, vocab_size)
test_data, test_labels = generate_expression_eval(n_test, vocab_size)

print(f"\nExample:")
print(f"  Input:  {train_data[0].tolist()}")
print(f"  Output: {train_labels[0].tolist()}")

results3 = {}
for ModelClass in [StandardTransformer, CTMHistory, CTMAutoLog, CTMAutoLogV2]:
    model = ModelClass(vocab_size, hidden_dim=64, n_layers=2, n_heads=4)
    acc = train_and_eval(model, train_data, train_labels, test_data, test_labels, n_epochs=200)
    results3[model.name] = acc
    print(f"  → Final: {acc:.1f}%\n")


# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

print(f"\n{'Task':<25} {'Transformer':>12} {'CTM-History':>12} {'AutoLog':>12} {'AutoLog-V2':>12}")
print("-" * 75)
print(f"{'Multi-Operation':<25} {results.get('Transformer', 0):>11.1f}% {results.get('CTM-History', 0):>11.1f}% "
      f"{results.get('CTM-AutoLog', 0):>11.1f}% {results.get('CTM-AutoLog-V2', 0):>11.1f}%")
print(f"{'Nested Reversal':<25} {results2.get('Transformer', 0):>11.1f}% {results2.get('CTM-History', 0):>11.1f}% "
      f"{results2.get('CTM-AutoLog', 0):>11.1f}% {results2.get('CTM-AutoLog-V2', 0):>11.1f}%")
print(f"{'Expression Eval':<25} {results3.get('Transformer', 0):>11.1f}% {results3.get('CTM-History', 0):>11.1f}% "
      f"{results3.get('CTM-AutoLog', 0):>11.1f}% {results3.get('CTM-AutoLog-V2', 0):>11.1f}%")

print("\n" + "=" * 80)
print("THE PRINCIPLE TESTED")
print("=" * 80)

print("""
CTM-AutoLog implements the insight:
  "Don't add decisions (learned read/write). Remove them."

Every tick AUTOMATICALLY writes to the log:
  - No "should I write?" decision
  - Writing is structural, like residual connections
  - The model learns WHAT to write (projection), not WHETHER

This parallels successful innovations:
  - Residual: Always pass info through → enabled very deep networks
  - Dropout: Random, no decision → enabled regularization
  - AutoLog: Always write to log → enables accumulation without learning control flow

If AutoLog >> CTM-History:
  → The principle is validated
  → Automatic accumulation > learned memory operations

If AutoLog ≈ others:
  → Need even stronger structural constraint
  → Or the bottleneck isn't working as intended
""")
