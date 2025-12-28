"""
COMPREHENSIVE TASK TESTING: 6 Different Task Classes

Testing whether our findings generalize across fundamentally different task types:
1. Expression evaluation (iterative refinement) - BASELINE
2. Variable binding (memory retrieval)
3. Conditional branching (ordered computation)
4. Bracket matching (long-range dependency)
5. Counting (discrete state)
6. Reasoning chains (conclusion chaining)

Same architectures tested on each:
- Standard Transformer
- Variant A (full history)
- Variant B (simple history)
- Simple Accumulator (no history)
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
# ARCHITECTURES (same as before)
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
        self.name = "Standard Transformer"

    def forward(self, x):
        pos = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
        h = self.embedding(x) + self.pos_embedding(pos)
        for block in self.blocks:
            h = block(h)
        return self.head(self.ln_f(h))


class CTMVariantA(nn.Module):
    """Full history: 2-layer MLP, history_len=2"""
    def __init__(self, vocab_size, hidden_dim=64, n_layers=2, n_heads=4,
                 n_ticks=4, history_len=2):
        super().__init__()
        self.n_ticks = n_ticks
        self.history_len = history_len
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embedding = nn.Embedding(256, hidden_dim)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim, nhead=n_heads,
                dim_feedforward=hidden_dim * 4, batch_first=True
            ) for _ in range(n_layers)
        ])
        self.history_processor = nn.Sequential(
            nn.Linear(hidden_dim * history_len, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.ln_f = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, vocab_size)
        self.name = "Variant A (full)"

    def forward(self, x):
        pos = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
        h = self.embedding(x) + self.pos_embedding(pos)
        history = [torch.zeros_like(h) for _ in range(self.history_len)]

        for tick in range(self.n_ticks):
            history_cat = torch.cat(history, dim=-1)
            history_features = self.history_processor(history_cat)
            h_input = h + 0.1 * history_features

            h_new = h_input
            for block in self.blocks:
                h_new = block(h_new)

            history = history[1:] + [h.clone()]
            h = h + h_new

        return self.head(self.ln_f(h))


class CTMVariantB(nn.Module):
    """Simple history: single linear, h_prev only"""
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
        self.name = "Variant B (simple)"

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


class SimpleAccumulator(nn.Module):
    """No history, just accumulation"""
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
        self.ln_f = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, vocab_size)
        self.name = "Simple Accumulator"

    def forward(self, x):
        pos = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
        h = self.embedding(x) + self.pos_embedding(pos)

        for tick in range(self.n_ticks):
            h_new = h
            for block in self.blocks:
                h_new = block(h_new)
            h = h + h_new

        return self.head(self.ln_f(h))


# =============================================================================
# TASK 1: Expression Evaluation (baseline)
# =============================================================================

def generate_expression_eval(n_samples, vocab_size=50):
    """a + b * c = ?"""
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


# =============================================================================
# TASK 2: Variable Binding
# =============================================================================

def generate_variable_binding(n_samples, vocab_size=50):
    """x = 5, y = 3, x + y = ?
    Format: [x_val, ASSIGN, y_val, ASSIGN, QUERY, 0, 0]
    Target: [0, 0, 0, 0, 0, sum_tens, sum_ones]
    """
    ASSIGN, QUERY = vocab_size - 2, vocab_size - 1
    data, labels = [], []
    for _ in range(n_samples):
        x_val = random.randint(1, 9)
        y_val = random.randint(1, 9)
        result = x_val + y_val
        inp = [x_val, ASSIGN, y_val, ASSIGN, QUERY, 0, 0]
        target = [0, 0, 0, 0, 0,
                  result // 10 if result >= 10 else result,
                  result % 10 if result >= 10 else 0]
        data.append(inp)
        labels.append(target)
    return torch.tensor(data), torch.tensor(labels)


# =============================================================================
# TASK 3: Conditional Branching
# =============================================================================

def generate_conditional(n_samples, vocab_size=50):
    """If a > b, output a - b, else output b - a
    Format: [a, CMP, b, THEN, 0, 0]
    Target: [0, 0, 0, 0, abs_diff_tens, abs_diff_ones]
    """
    CMP, THEN = vocab_size - 2, vocab_size - 1
    data, labels = [], []
    for _ in range(n_samples):
        a = random.randint(1, 9)
        b = random.randint(1, 9)
        result = abs(a - b)
        inp = [a, CMP, b, THEN, 0, 0]
        target = [0, 0, 0, 0, result // 10, result % 10]
        data.append(inp)
        labels.append(target)
    return torch.tensor(data), torch.tensor(labels)


# =============================================================================
# TASK 4: Bracket Matching (Long-Range Dependency)
# =============================================================================

def generate_bracket_matching(n_samples, vocab_size=50):
    """Check if brackets are balanced
    Tokens: 0=pad, 1=(, 2=), 3=[, 4=], VALID=vocab-2, INVALID=vocab-1
    Format: [brackets..., QUERY, 0]
    Target: [0, ..., 0, VALID or INVALID]
    """
    OPEN_PAREN, CLOSE_PAREN = 1, 2
    OPEN_BRACKET, CLOSE_BRACKET = 3, 4
    QUERY = vocab_size - 3
    VALID, INVALID = vocab_size - 2, vocab_size - 1

    data, labels = [], []
    for _ in range(n_samples):
        # Generate random bracket sequence
        length = random.randint(2, 6)
        seq = []
        stack = []
        valid = True

        for _ in range(length):
            if random.random() < 0.5 and len(stack) > 0:
                # Close a bracket
                expected = stack.pop()
                if random.random() < 0.8:  # Usually match correctly
                    seq.append(expected)
                else:  # Sometimes mismatch
                    seq.append(CLOSE_PAREN if expected == CLOSE_BRACKET else CLOSE_BRACKET)
                    valid = False
            else:
                # Open a bracket
                if random.random() < 0.5:
                    seq.append(OPEN_PAREN)
                    stack.append(CLOSE_PAREN)
                else:
                    seq.append(OPEN_BRACKET)
                    stack.append(CLOSE_BRACKET)

        if len(stack) > 0:
            valid = False

        # Pad to fixed length
        while len(seq) < 6:
            seq.append(0)
        seq = seq[:6]

        inp = seq + [QUERY, 0]
        target = [0] * 7 + [VALID if valid else INVALID]

        data.append(inp)
        labels.append(target)

    return torch.tensor(data), torch.tensor(labels)


# =============================================================================
# TASK 5: Counting
# =============================================================================

def generate_counting(n_samples, vocab_size=50):
    """Count occurrences of target symbol in sequence
    Format: [target, SEP, seq..., QUERY, 0]
    Target: [0, ..., 0, count]
    """
    SEP, QUERY = vocab_size - 2, vocab_size - 1
    data, labels = [], []
    for _ in range(n_samples):
        target = random.randint(1, 5)
        seq_len = 5
        seq = [random.randint(1, 5) for _ in range(seq_len)]
        count = seq.count(target)

        inp = [target, SEP] + seq + [QUERY, 0]
        target_out = [0] * (len(inp) - 1) + [count]

        data.append(inp)
        labels.append(target_out)

    return torch.tensor(data), torch.tensor(labels)


# =============================================================================
# TASK 6: Reasoning Chains
# =============================================================================

def generate_reasoning_chain(n_samples, vocab_size=50):
    """A→B, B→C, A is true. Is C true?
    Simple version: transitive relation
    Format: [a, IMPLIES, b, IMPLIES, c, START, a, QUERY, 0]
    If chain is valid and starts with a: output c
    """
    IMPLIES, START, QUERY = vocab_size - 3, vocab_size - 2, vocab_size - 1
    data, labels = [], []
    for _ in range(n_samples):
        a, b, c = random.randint(1, 5), random.randint(1, 5), random.randint(1, 5)
        # a → b → c, starting with a, should output c
        inp = [a, IMPLIES, b, IMPLIES, c, START, a, QUERY, 0]
        target = [0] * 8 + [c]
        data.append(inp)
        labels.append(target)

    return torch.tensor(data), torch.tensor(labels)


# =============================================================================
# TRAINING AND EVALUATION
# =============================================================================

def train_and_eval(model, train_data, train_labels, test_data, test_labels,
                   n_epochs=100, ignore_index=0):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)

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
        mask = test_labels != ignore_index
        correct = (preds == test_labels) & mask
        if mask.sum() == 0:
            return 0.0
        return (correct.sum().float() / mask.sum().float()).item() * 100


# =============================================================================
# RUN ALL TASKS
# =============================================================================

def run_task(task_name, generate_fn, vocab_size=50, n_runs=3):
    print(f"\n{'='*60}")
    print(f"TASK: {task_name}")
    print('='*60)

    results = {
        "Standard Transformer": [],
        "Variant A (full)": [],
        "Variant B (simple)": [],
        "Simple Accumulator": [],
    }

    for run in range(n_runs):
        torch.manual_seed(42 + run)
        np.random.seed(42 + run)
        random.seed(42 + run)

        train_data, train_labels = generate_fn(1000, vocab_size)
        test_data, test_labels = generate_fn(200, vocab_size)

        models = [
            StandardTransformer(vocab_size),
            CTMVariantA(vocab_size),
            CTMVariantB(vocab_size),
            SimpleAccumulator(vocab_size),
        ]

        for model in models:
            acc = train_and_eval(model, train_data, train_labels, test_data, test_labels)
            results[model.name].append(acc)

    # Print results
    print(f"\n{'Model':<25} {'Mean':>8} {'Std':>8}")
    print("-" * 43)
    for name, accs in results.items():
        mean, std = np.mean(accs), np.std(accs)
        print(f"{name:<25} {mean:>7.1f}% {std:>7.1f}%")

    return results


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("COMPREHENSIVE TASK TESTING")
    print("Testing 6 fundamentally different task classes")
    print("=" * 60)

    all_results = {}

    tasks = [
        ("1. Expression Evaluation", generate_expression_eval),
        ("2. Variable Binding", generate_variable_binding),
        ("3. Conditional Branching", generate_conditional),
        ("4. Bracket Matching", generate_bracket_matching),
        ("5. Counting", generate_counting),
        ("6. Reasoning Chains", generate_reasoning_chain),
    ]

    for task_name, generate_fn in tasks:
        results = run_task(task_name, generate_fn)
        all_results[task_name] = results

    # =============================================================================
    # SUMMARY
    # =============================================================================

    print("\n" + "=" * 60)
    print("SUMMARY: Mean accuracy across all tasks")
    print("=" * 60)

    model_names = ["Standard Transformer", "Variant A (full)", "Variant B (simple)", "Simple Accumulator"]

    print(f"\n{'Task':<30}", end="")
    for name in model_names:
        short = name.split()[0][:6]
        print(f"{short:>10}", end="")
    print()
    print("-" * 70)

    model_totals = {name: [] for name in model_names}

    for task_name, results in all_results.items():
        print(f"{task_name:<30}", end="")
        for name in model_names:
            mean = np.mean(results[name])
            model_totals[name].append(mean)
            print(f"{mean:>9.1f}%", end="")
        print()

    print("-" * 70)
    print(f"{'AVERAGE':<30}", end="")
    for name in model_names:
        avg = np.mean(model_totals[name])
        print(f"{avg:>9.1f}%", end="")
    print()

    # =============================================================================
    # ANALYSIS
    # =============================================================================

    print("\n" + "=" * 60)
    print("ANALYSIS: Where does each architecture excel/struggle?")
    print("=" * 60)

    for task_name, results in all_results.items():
        means = {name: np.mean(results[name]) for name in model_names}
        best = max(means, key=means.get)
        worst = min(means, key=means.get)
        gap = means[best] - means[worst]

        print(f"\n{task_name}:")
        print(f"  Best:  {best} ({means[best]:.1f}%)")
        print(f"  Worst: {worst} ({means[worst]:.1f}%)")
        print(f"  Gap:   {gap:.1f}%")

        # Check if history helps
        history_avg = (means["Variant A (full)"] + means["Variant B (simple)"]) / 2
        no_history = means["Simple Accumulator"]
        if history_avg > no_history + 2:
            print(f"  → History HELPS on this task (+{history_avg - no_history:.1f}%)")
        elif no_history > history_avg + 2:
            print(f"  → History HURTS on this task ({history_avg - no_history:.1f}%)")
        else:
            print(f"  → History neutral on this task")

    print("\n" + "=" * 60)
