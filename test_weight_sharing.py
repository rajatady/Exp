"""
Testing: Is weight sharing essential to CTM, or just limiting?

Comparing:
1. Standard Transformer (4 layers, different weights)
2. CTM Shared (2 layers × 4 ticks, SAME weights)
3. CTM Unrolled (2 layers × 4 ticks, DIFFERENT weights per tick)
4. Unrolled + Accumulation (different weights + h = h + h_new)
5. Unrolled + Accumulation + History (full unrolled CTM)

Testing on all 6 task classes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# =============================================================================
# ARCHITECTURES
# =============================================================================

class StandardTransformer(nn.Module):
    """4 different layers"""
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
        self.name = "Standard (4 diff)"

    def forward(self, x):
        pos = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
        h = self.embedding(x) + self.pos_embedding(pos)
        for block in self.blocks:
            h = block(h)
        return self.head(self.ln_f(h))


class CTMShared(nn.Module):
    """2 layers × 4 ticks, SAME weights across ticks"""
    def __init__(self, vocab_size, hidden_dim=64, n_layers=2, n_heads=4, n_ticks=4):
        super().__init__()
        self.n_ticks = n_ticks
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embedding = nn.Embedding(256, hidden_dim)
        # Only 2 layers, shared across ticks
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim, nhead=n_heads,
                dim_feedforward=hidden_dim * 4, batch_first=True
            ) for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, vocab_size)
        self.name = "CTM Shared"

    def forward(self, x):
        pos = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
        h = self.embedding(x) + self.pos_embedding(pos)

        for tick in range(self.n_ticks):
            h_new = h
            for block in self.blocks:
                h_new = block(h_new)
            h = h + h_new  # Accumulation

        return self.head(self.ln_f(h))


class CTMUnrolled(nn.Module):
    """2 layers × 4 ticks, DIFFERENT weights per tick"""
    def __init__(self, vocab_size, hidden_dim=64, n_layers_per_tick=2, n_heads=4, n_ticks=4):
        super().__init__()
        self.n_ticks = n_ticks
        self.n_layers_per_tick = n_layers_per_tick
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embedding = nn.Embedding(256, hidden_dim)

        # Different layers for each tick
        self.tick_blocks = nn.ModuleList([
            nn.ModuleList([
                nn.TransformerEncoderLayer(
                    d_model=hidden_dim, nhead=n_heads,
                    dim_feedforward=hidden_dim * 4, batch_first=True
                ) for _ in range(n_layers_per_tick)
            ]) for _ in range(n_ticks)
        ])

        self.ln_f = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, vocab_size)
        self.name = "CTM Unrolled (diff)"

    def forward(self, x):
        pos = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
        h = self.embedding(x) + self.pos_embedding(pos)

        for tick in range(self.n_ticks):
            h_new = h
            for block in self.tick_blocks[tick]:
                h_new = block(h_new)
            h = h_new  # No accumulation, just pass through

        return self.head(self.ln_f(h))


class CTMUnrolledAccum(nn.Module):
    """2 layers × 4 ticks, DIFFERENT weights per tick, WITH accumulation"""
    def __init__(self, vocab_size, hidden_dim=64, n_layers_per_tick=2, n_heads=4, n_ticks=4):
        super().__init__()
        self.n_ticks = n_ticks
        self.n_layers_per_tick = n_layers_per_tick
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embedding = nn.Embedding(256, hidden_dim)

        # Different layers for each tick
        self.tick_blocks = nn.ModuleList([
            nn.ModuleList([
                nn.TransformerEncoderLayer(
                    d_model=hidden_dim, nhead=n_heads,
                    dim_feedforward=hidden_dim * 4, batch_first=True
                ) for _ in range(n_layers_per_tick)
            ]) for _ in range(n_ticks)
        ])

        self.ln_f = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, vocab_size)
        self.name = "Unrolled + Accum"

    def forward(self, x):
        pos = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
        h = self.embedding(x) + self.pos_embedding(pos)

        for tick in range(self.n_ticks):
            h_new = h
            for block in self.tick_blocks[tick]:
                h_new = block(h_new)
            h = h + h_new  # WITH accumulation

        return self.head(self.ln_f(h))


class CTMUnrolledFull(nn.Module):
    """2 layers × 4 ticks, DIFFERENT weights, accumulation, AND history"""
    def __init__(self, vocab_size, hidden_dim=64, n_layers_per_tick=2, n_heads=4, n_ticks=4):
        super().__init__()
        self.n_ticks = n_ticks
        self.n_layers_per_tick = n_layers_per_tick
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embedding = nn.Embedding(256, hidden_dim)

        # Different layers for each tick
        self.tick_blocks = nn.ModuleList([
            nn.ModuleList([
                nn.TransformerEncoderLayer(
                    d_model=hidden_dim, nhead=n_heads,
                    dim_feedforward=hidden_dim * 4, batch_first=True
                ) for _ in range(n_layers_per_tick)
            ]) for _ in range(n_ticks)
        ])

        # History projector per tick
        self.history_projs = nn.ModuleList([
            nn.Linear(hidden_dim * 2, hidden_dim) for _ in range(n_ticks)
        ])

        self.ln_f = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, vocab_size)
        self.name = "Unrolled + Full"

    def forward(self, x):
        pos = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
        h = self.embedding(x) + self.pos_embedding(pos)
        h_prev = torch.zeros_like(h)

        for tick in range(self.n_ticks):
            # History
            history = self.history_projs[tick](torch.cat([h, h_prev], dim=-1))
            h_input = h + 0.1 * history

            h_new = h_input
            for block in self.tick_blocks[tick]:
                h_new = block(h_new)

            h_prev = h.detach()
            h = h + h_new

        return self.head(self.ln_f(h))


# =============================================================================
# TASK GENERATORS (same as before)
# =============================================================================

def generate_expression_eval(n_samples, vocab_size=50):
    PLUS, TIMES, EQUALS = vocab_size - 3, vocab_size - 2, vocab_size - 1
    data, labels = [], []
    for _ in range(n_samples):
        a, b, c = random.randint(1, 5), random.randint(1, 5), random.randint(1, 5)
        result = a + b * c
        inp = [a, PLUS, b, TIMES, c, EQUALS, 0, 0]
        target = [0, 0, 0, 0, 0, 0,
                  result // 10 if result >= 10 else result,
                  result % 10 if result >= 10 else 0]
        data.append(inp)
        labels.append(target)
    return torch.tensor(data), torch.tensor(labels)


def generate_counting(n_samples, vocab_size=50):
    SEP, QUERY = vocab_size - 2, vocab_size - 1
    data, labels = [], []
    for _ in range(n_samples):
        target = random.randint(1, 5)
        seq = [random.randint(1, 5) for _ in range(5)]
        count = seq.count(target)
        inp = [target, SEP] + seq + [QUERY, 0]
        target_out = [0] * (len(inp) - 1) + [count]
        data.append(inp)
        labels.append(target_out)
    return torch.tensor(data), torch.tensor(labels)


def generate_bracket_matching(n_samples, vocab_size=50):
    OPEN_PAREN, CLOSE_PAREN = 1, 2
    OPEN_BRACKET, CLOSE_BRACKET = 3, 4
    QUERY = vocab_size - 3
    VALID, INVALID = vocab_size - 2, vocab_size - 1

    data, labels = [], []
    for _ in range(n_samples):
        length = random.randint(2, 6)
        seq = []
        stack = []
        valid = True

        for _ in range(length):
            if random.random() < 0.5 and len(stack) > 0:
                expected = stack.pop()
                if random.random() < 0.8:
                    seq.append(expected)
                else:
                    seq.append(CLOSE_PAREN if expected == CLOSE_BRACKET else CLOSE_BRACKET)
                    valid = False
            else:
                if random.random() < 0.5:
                    seq.append(OPEN_PAREN)
                    stack.append(CLOSE_PAREN)
                else:
                    seq.append(OPEN_BRACKET)
                    stack.append(CLOSE_BRACKET)

        if len(stack) > 0:
            valid = False

        while len(seq) < 6:
            seq.append(0)
        seq = seq[:6]

        inp = seq + [QUERY, 0]
        target = [0] * 7 + [VALID if valid else INVALID]
        data.append(inp)
        labels.append(target)

    return torch.tensor(data), torch.tensor(labels)


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
        if mask.sum() == 0:
            return 0.0
        return (correct.sum().float() / mask.sum().float()).item() * 100


def count_params(model):
    return sum(p.numel() for p in model.parameters())


# =============================================================================
# RUN TESTS
# =============================================================================

def run_task(task_name, generate_fn, vocab_size=50, n_runs=3):
    print(f"\n{'='*70}")
    print(f"TASK: {task_name}")
    print('='*70)

    model_classes = [
        StandardTransformer,
        CTMShared,
        CTMUnrolled,
        CTMUnrolledAccum,
        CTMUnrolledFull,
    ]

    results = {cls(vocab_size).name: [] for cls in model_classes}

    for run in range(n_runs):
        torch.manual_seed(42 + run)
        np.random.seed(42 + run)
        random.seed(42 + run)

        train_data, train_labels = generate_fn(1000, vocab_size)
        test_data, test_labels = generate_fn(200, vocab_size)

        for ModelClass in model_classes:
            model = ModelClass(vocab_size)
            acc = train_and_eval(model, train_data, train_labels, test_data, test_labels)
            results[model.name].append(acc)

    # Print results
    print(f"\n{'Model':<25} {'Mean':>8} {'Std':>8} {'Params':>12}")
    print("-" * 55)
    for ModelClass in model_classes:
        model = ModelClass(vocab_size)
        name = model.name
        accs = results[name]
        mean, std = np.mean(accs), np.std(accs)
        params = count_params(model)
        print(f"{name:<25} {mean:>7.1f}% {std:>7.1f}% {params:>12,}")

    return results


if __name__ == "__main__":
    print("=" * 70)
    print("WEIGHT SHARING TEST: Is sharing weights essential or limiting?")
    print("=" * 70)

    vocab_size = 50

    # Print parameter counts first
    print("\nParameter counts:")
    for ModelClass in [StandardTransformer, CTMShared, CTMUnrolled, CTMUnrolledAccum, CTMUnrolledFull]:
        model = ModelClass(vocab_size)
        print(f"  {model.name}: {count_params(model):,}")

    all_results = {}

    tasks = [
        ("Expression Evaluation", generate_expression_eval),
        ("Counting", generate_counting),
        ("Bracket Matching", generate_bracket_matching),
    ]

    for task_name, generate_fn in tasks:
        results = run_task(task_name, generate_fn)
        all_results[task_name] = results

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    model_names = ["Standard (4 diff)", "CTM Shared", "CTM Unrolled (diff)",
                   "Unrolled + Accum", "Unrolled + Full"]

    print(f"\n{'Task':<25}", end="")
    for name in model_names:
        short = name[:8]
        print(f"{short:>12}", end="")
    print()
    print("-" * 85)

    for task_name, results in all_results.items():
        print(f"{task_name:<25}", end="")
        for name in model_names:
            mean = np.mean(results[name])
            print(f"{mean:>11.1f}%", end="")
        print()

    print("\n" + "=" * 70)
    print("KEY QUESTION: Does unrolling (different weights per tick) help?")
    print("=" * 70)

    for task_name, results in all_results.items():
        shared = np.mean(results["CTM Shared"])
        unrolled = np.mean(results["CTM Unrolled (diff)"])
        unrolled_accum = np.mean(results["Unrolled + Accum"])
        standard = np.mean(results["Standard (4 diff)"])

        print(f"\n{task_name}:")
        print(f"  CTM Shared:        {shared:.1f}%")
        print(f"  CTM Unrolled:      {unrolled:.1f}% (diff from shared: {unrolled - shared:+.1f}%)")
        print(f"  Unrolled + Accum:  {unrolled_accum:.1f}% (diff from shared: {unrolled_accum - shared:+.1f}%)")
        print(f"  Standard:          {standard:.1f}%")

        if unrolled_accum > shared + 2:
            print(f"  → Unrolling HELPS on this task")
        elif shared > unrolled_accum + 2:
            print(f"  → Sharing HELPS on this task (unrolling hurts)")
        else:
            print(f"  → No significant difference")

    print("\n" + "=" * 70)
