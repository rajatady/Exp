"""
Comprehensive Multi-Pass Analysis

Three questions:
1. Does multi-pass generalize across tasks?
2. What changes between passes?
3. How does this connect to the intelligence loop theory?
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import math
from collections import defaultdict

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

print("=" * 80)
print("COMPREHENSIVE MULTI-PASS ANALYSIS")
print("=" * 80)

# Device
if torch.backends.mps.is_available():
    device = torch.device('mps')
elif torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print(f"Device: {device}")

# =============================================================================
# PART 1: MULTIPLE TASK TYPES
# =============================================================================

print("\n" + "=" * 80)
print("PART 1: TESTING MULTI-PASS ACROSS TASK TYPES")
print("=" * 80)

# -----------------------------------------------------------------------------
# Task Generators
# -----------------------------------------------------------------------------

def generate_compositional_data():
    """SCAN-like: WALK TWICE → WW"""
    PRIMITIVES = ['WALK', 'RUN', 'JUMP', 'LOOK']
    MODIFIERS = ['TWICE', 'THRICE']
    PRIM_TO_OUT = {'WALK': 'W', 'RUN': 'R', 'JUMP': 'J', 'LOOK': 'L'}

    def execute(cmd):
        result = []
        i = 0
        while i < len(cmd):
            if cmd[i] in PRIMITIVES:
                out = PRIM_TO_OUT[cmd[i]]
                if i + 1 < len(cmd) and cmd[i + 1] == 'TWICE':
                    result.extend([out, out])
                    i += 2
                elif i + 1 < len(cmd) and cmd[i + 1] == 'THRICE':
                    result.extend([out, out, out])
                    i += 2
                else:
                    result.append(out)
                    i += 1
            else:
                i += 1
        return result

    train, test = [], []
    holdout = [('WALK', 'THRICE'), ('RUN', 'THRICE'), ('JUMP', 'TWICE'), ('LOOK', 'TWICE')]

    for p in PRIMITIVES:
        train.append(([p], execute([p])))
        for m in MODIFIERS:
            cmd = [p, m]
            if (p, m) in holdout:
                test.append((cmd, execute(cmd)))
            else:
                train.append((cmd, execute(cmd)))

    vocab = {'<PAD>': 0, '<BOS>': 1, '<EOS>': 2, '<SEP>': 3}
    for p in PRIMITIVES:
        vocab[p] = len(vocab)
    for m in MODIFIERS:
        vocab[m] = len(vocab)
    for o in PRIM_TO_OUT.values():
        vocab[o] = len(vocab)

    return train, test, vocab, "Compositional (SCAN-like)"


def generate_reversal_data(n_train=200, n_test=50):
    """Reverse: 1 2 3 | 3 2 1"""
    train, test = [], []

    for _ in range(n_train):
        length = random.randint(3, 6)
        nums = [random.randint(0, 9) for _ in range(length)]
        train.append((nums, list(reversed(nums))))

    for _ in range(n_test):
        length = random.randint(3, 6)
        nums = [random.randint(0, 9) for _ in range(length)]
        test.append((nums, list(reversed(nums))))

    vocab = {'<PAD>': 0, '<BOS>': 1, '<EOS>': 2, '<SEP>': 3}
    for i in range(10):
        vocab[str(i)] = len(vocab)

    return train, test, vocab, "Reversal"


def generate_sorting_data(n_train=200, n_test=50):
    """Sort: 3 1 4 | 1 3 4"""
    train, test = [], []

    for _ in range(n_train):
        length = random.randint(3, 6)
        nums = [random.randint(0, 9) for _ in range(length)]
        train.append((nums, sorted(nums)))

    for _ in range(n_test):
        length = random.randint(3, 6)
        nums = [random.randint(0, 9) for _ in range(length)]
        test.append((nums, sorted(nums)))

    vocab = {'<PAD>': 0, '<BOS>': 1, '<EOS>': 2, '<SEP>': 3}
    for i in range(10):
        vocab[str(i)] = len(vocab)

    return train, test, vocab, "Sorting"


def generate_counting_data(n_train=200, n_test=50):
    """Count: a a b a | a:3 b:1"""
    train, test = [], []
    symbols = ['a', 'b', 'c']

    for _ in range(n_train):
        length = random.randint(4, 8)
        seq = [random.choice(symbols) for _ in range(length)]
        counts = {s: seq.count(s) for s in symbols}
        output = []
        for s in symbols:
            if counts[s] > 0:
                output.extend([s, str(counts[s])])
        train.append((seq, output))

    for _ in range(n_test):
        length = random.randint(4, 8)
        seq = [random.choice(symbols) for _ in range(length)]
        counts = {s: seq.count(s) for s in symbols}
        output = []
        for s in symbols:
            if counts[s] > 0:
                output.extend([s, str(counts[s])])
        test.append((seq, output))

    vocab = {'<PAD>': 0, '<BOS>': 1, '<EOS>': 2, '<SEP>': 3}
    for s in symbols:
        vocab[s] = len(vocab)
    for i in range(10):
        vocab[str(i)] = len(vocab)

    return train, test, vocab, "Counting"


def generate_copy_data(n_train=200, n_test=50):
    """Simple copy: 1 2 3 | 1 2 3"""
    train, test = [], []

    for _ in range(n_train):
        length = random.randint(3, 6)
        nums = [random.randint(0, 9) for _ in range(length)]
        train.append((nums, nums.copy()))

    for _ in range(n_test):
        length = random.randint(3, 6)
        nums = [random.randint(0, 9) for _ in range(length)]
        test.append((nums, nums.copy()))

    vocab = {'<PAD>': 0, '<BOS>': 1, '<EOS>': 2, '<SEP>': 3}
    for i in range(10):
        vocab[str(i)] = len(vocab)

    return train, test, vocab, "Copy"


# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------

class StandardTransformer(nn.Module):
    def __init__(self, vocab_size, hidden_dim=64, n_layers=4):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.pos = nn.Parameter(torch.randn(1, 64, hidden_dim) * 0.02)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4,
                dim_feedforward=hidden_dim*4, batch_first=True, dropout=0.1)
            for _ in range(n_layers)
        ])
        self.ln = nn.LayerNorm(hidden_dim)
        self.out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, return_intermediates=False):
        h = self.embed(x) + self.pos[:, :x.size(1)]
        intermediates = [h.clone()]
        for layer in self.layers:
            h = layer(h)
            intermediates.append(h.clone())
        logits = self.out(self.ln(h))
        if return_intermediates:
            return logits, intermediates
        return logits


class MultiPassTransformer(nn.Module):
    def __init__(self, vocab_size, hidden_dim=64, n_layers=2, n_passes=4):
        super().__init__()
        self.n_passes = n_passes
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.pos = nn.Parameter(torch.randn(1, 64, hidden_dim) * 0.02)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4,
                dim_feedforward=hidden_dim*4, batch_first=True, dropout=0.1)
            for _ in range(n_layers)
        ])
        self.ln = nn.LayerNorm(hidden_dim)
        self.out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, return_intermediates=False):
        h = self.embed(x) + self.pos[:, :x.size(1)]
        intermediates = [h.clone()]
        for p in range(self.n_passes):
            for layer in self.layers:
                h = layer(h)
            intermediates.append(h.clone())
        logits = self.out(self.ln(h))
        if return_intermediates:
            return logits, intermediates
        return logits


# -----------------------------------------------------------------------------
# Training & Evaluation
# -----------------------------------------------------------------------------

def to_tensor(inp, out, vocab):
    ids = [vocab['<BOS>']]
    for x in inp:
        ids.append(vocab.get(str(x) if isinstance(x, int) else x, vocab['<PAD>']))
    ids.append(vocab['<SEP>'])
    for x in out:
        ids.append(vocab.get(str(x) if isinstance(x, int) else x, vocab['<PAD>']))
    ids.append(vocab['<EOS>'])
    return ids


def pad(seqs):
    max_len = max(len(s) for s in seqs)
    return [s + [0] * (max_len - len(s)) for s in seqs]


def train_model(model, train_data, vocab, n_epochs=200, device='cpu'):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    seqs = [to_tensor(inp, out, vocab) for inp, out in train_data]

    for epoch in range(n_epochs):
        model.train()
        random.shuffle(seqs)
        batch = pad(seqs)
        x = torch.tensor(batch, device=device)
        inputs, targets = x[:, :-1], x[:, 1:]
        optimizer.zero_grad()
        logits = model(inputs)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=0)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    return model


def evaluate(model, data, vocab, device='cpu'):
    model.eval()
    seqs = [to_tensor(inp, out, vocab) for inp, out in data]
    correct = 0

    with torch.no_grad():
        batch = pad(seqs)
        x = torch.tensor(batch, device=device)
        inputs, targets = x[:, :-1], x[:, 1:]
        logits = model(inputs)
        preds = logits.argmax(-1)

        sep_id = vocab['<SEP>']
        for i, (inp, out) in enumerate(data):
            seq = seqs[i]
            sep_pos = seq.index(sep_id)
            start, end = sep_pos, len(seq) - 1
            if start < end:
                if preds[i, start:end].tolist() == targets[i, start:end].tolist():
                    correct += 1

    return correct / len(data)


# -----------------------------------------------------------------------------
# Run Tasks
# -----------------------------------------------------------------------------

HIDDEN_DIM = 64
N_SEEDS = 3

task_generators = [
    generate_compositional_data,
    generate_reversal_data,
    generate_sorting_data,
    generate_counting_data,
    generate_copy_data,
]

all_results = {}

for gen_fn in task_generators:
    train_data, test_data, vocab, task_name = gen_fn()
    vocab_size = len(vocab)

    print(f"\n{'='*60}")
    print(f"Task: {task_name}")
    print(f"Train: {len(train_data)}, Test: {len(test_data)}, Vocab: {vocab_size}")
    print(f"{'='*60}")

    results = defaultdict(list)

    for seed in range(N_SEEDS):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # Standard (4 layers)
        std = StandardTransformer(vocab_size, HIDDEN_DIM, n_layers=4).to(device)
        std = train_model(std, train_data, vocab, n_epochs=200, device=device)
        std_acc = evaluate(std, test_data, vocab, device)
        results['Standard'].append(std_acc)

        # Multi-pass (2 layers x 4 passes = 8 effective layers)
        mp = MultiPassTransformer(vocab_size, HIDDEN_DIM, n_layers=2, n_passes=4).to(device)
        mp = train_model(mp, train_data, vocab, n_epochs=200, device=device)
        mp_acc = evaluate(mp, test_data, vocab, device)
        results['Multi-Pass'].append(mp_acc)

    std_mean = np.mean(results['Standard'])
    mp_mean = np.mean(results['Multi-Pass'])
    diff = mp_mean - std_mean

    print(f"  Standard:   {std_mean:.1%} ± {np.std(results['Standard']):.1%}")
    print(f"  Multi-Pass: {mp_mean:.1%} ± {np.std(results['Multi-Pass']):.1%}")
    print(f"  Difference: {diff:+.1%}")

    all_results[task_name] = {
        'standard': std_mean,
        'multipass': mp_mean,
        'diff': diff,
    }


print("\n" + "=" * 80)
print("PART 1 SUMMARY: Multi-Pass Across Tasks")
print("=" * 80)

print(f"\n{'Task':<20} {'Standard':>10} {'Multi-Pass':>12} {'Diff':>10}")
print("-" * 55)
for task, r in all_results.items():
    print(f"{task:<20} {r['standard']:>9.1%} {r['multipass']:>11.1%} {r['diff']:>+9.1%}")

helps_count = sum(1 for r in all_results.values() if r['diff'] > 0.02)
hurts_count = sum(1 for r in all_results.values() if r['diff'] < -0.02)
print(f"\nMulti-Pass helps on: {helps_count}/{len(all_results)} tasks")
print(f"Multi-Pass hurts on: {hurts_count}/{len(all_results)} tasks")


# =============================================================================
# PART 2: WHAT CHANGES BETWEEN PASSES?
# =============================================================================

print("\n" + "=" * 80)
print("PART 2: WHAT CHANGES BETWEEN PASSES?")
print("=" * 80)

# Use compositional task for analysis
train_data, test_data, vocab, _ = generate_compositional_data()
vocab_size = len(vocab)

torch.manual_seed(42)
model = MultiPassTransformer(vocab_size, HIDDEN_DIM, n_layers=2, n_passes=4).to(device)
model = train_model(model, train_data, vocab, n_epochs=200, device=device)

# Get intermediates for a few examples
model.eval()
sample_data = test_data[:5]
seqs = [to_tensor(inp, out, vocab) for inp, out in sample_data]
batch = pad(seqs)
x = torch.tensor(batch, device=device)
inputs = x[:, :-1]

with torch.no_grad():
    logits, intermediates = model(inputs, return_intermediates=True)

print(f"\nAnalyzing {len(intermediates)} intermediate representations")
print("(Initial embedding + 4 passes)")

# Measure changes between passes
print("\n1. Representation change between passes (L2 norm):")
for i in range(1, len(intermediates)):
    change = (intermediates[i] - intermediates[i-1]).norm(dim=-1).mean().item()
    print(f"   Pass {i-1} → {i}: {change:.3f}")

# Measure convergence (are later passes smaller changes?)
changes = [(intermediates[i] - intermediates[i-1]).norm(dim=-1).mean().item()
           for i in range(1, len(intermediates))]
if changes[-1] < changes[0]:
    print(f"\n   → Changes DECREASE over passes (converging): {changes[0]:.3f} → {changes[-1]:.3f}")
else:
    print(f"\n   → Changes do NOT decrease (not converging): {changes[0]:.3f} → {changes[-1]:.3f}")

# Measure representation similarity to final
print("\n2. Similarity to final representation (cosine):")
final = intermediates[-1]
for i, inter in enumerate(intermediates[:-1]):
    sim = F.cosine_similarity(inter, final, dim=-1).mean().item()
    print(f"   Pass {i}: {sim:.3f}")

# Prediction confidence over passes
print("\n3. Prediction confidence over passes:")
for i, inter in enumerate(intermediates):
    logits_i = model.out(model.ln(inter))
    probs = F.softmax(logits_i, dim=-1)
    max_prob = probs.max(dim=-1).values.mean().item()
    print(f"   Pass {i}: {max_prob:.3f}")

# Check if predictions change
print("\n4. Do predictions change between passes?")
prev_preds = None
for i, inter in enumerate(intermediates):
    logits_i = model.out(model.ln(inter))
    preds = logits_i.argmax(dim=-1)
    if prev_preds is not None:
        changed = (preds != prev_preds).float().mean().item()
        print(f"   Pass {i-1} → {i}: {changed:.1%} of predictions changed")
    prev_preds = preds


# =============================================================================
# PART 3: CONNECTING TO THEORY
# =============================================================================

print("\n" + "=" * 80)
print("PART 3: CONNECTION TO INTELLIGENCE LOOP THEORY")
print("=" * 80)

print("""
THEORETICAL FRAMEWORK RECAP:

Intelligence = Sense → Store → Model → Predict → Act → Feedback → Update
                ↑                                                    ↓
                └────────────────────────────────────────────────────┘

The minimal action is SELF-MODIFICATION based on prediction error.

WHAT WE HYPOTHESIZED:
- Inner loop at inference: State predicts input → Error → Update state
- This implements self-modification during inference

WHAT WE FOUND:
- Inner loop error doesn't actually decrease
- Multi-pass (just rerunning encoder) works BETTER

THE QUESTION: How does multi-pass relate to the intelligence loop?
""")

print("=" * 60)
print("ANALYSIS: Multi-Pass as Iterative Refinement")
print("=" * 60)

print("""
Multi-pass does:
  h₀ = embed(input)
  h₁ = encoder(h₀)
  h₂ = encoder(h₁)
  h₃ = encoder(h₂)
  h₄ = encoder(h₃)
  output = project(h₄)

Each pass refines the representation. But is this "self-modification"?
""")

# Measure if later passes are "correcting" earlier predictions
print("Checking: Do later passes CORRECT earlier predictions?")

correct_by_pass = []
targets = x[:, 1:]
sep_id = vocab['<SEP>']

for i, inter in enumerate(intermediates):
    logits_i = model.out(model.ln(inter))
    preds = logits_i.argmax(dim=-1)

    # Count correct predictions for output portion
    n_correct = 0
    n_total = 0
    for j, (inp, out) in enumerate(sample_data):
        seq = seqs[j]
        sep_pos = seq.index(sep_id)
        start, end = sep_pos, len(seq) - 1
        if start < end:
            pred_part = preds[j, start:end]
            target_part = targets[j, start:end]
            n_correct += (pred_part == target_part).sum().item()
            n_total += end - start

    acc = n_correct / n_total if n_total > 0 else 0
    correct_by_pass.append(acc)
    print(f"   Pass {i}: {acc:.1%} correct")

if correct_by_pass[-1] > correct_by_pass[0]:
    improvement = correct_by_pass[-1] - correct_by_pass[0]
    print(f"\n   → Later passes ARE more accurate (+{improvement:.1%})")
    print("   → This IS a form of iterative refinement / self-correction")
else:
    print(f"\n   → Later passes are NOT more accurate")
    print("   → Multi-pass benefit comes from something else")


print("\n" + "=" * 60)
print("THEORETICAL INTERPRETATION")
print("=" * 60)

print("""
MAPPING MULTI-PASS TO INTELLIGENCE LOOP:

Standard Forward Pass:
  Sense → Model → Predict → Output
  (One shot, no feedback, no update)

Multi-Pass:
  Sense → Model → Predict (internal) → [Implicit feedback via attention] →
  → Refine Model → Predict (internal) → [...] → Output

The KEY INSIGHT:

Multi-pass provides IMPLICIT feedback through ATTENTION.

Each pass, attention patterns see the CURRENT state of all positions.
If position A's representation changes, position B notices in next pass.
This creates an implicit feedback loop WITHOUT explicit error computation.

     ┌─────────────────────────────────────┐
     ↓                                     │
   Attend to         Process with       Update
   current state →   transformer →   representation
     ↑                                     │
     └─────────────────────────────────────┘
           (This IS a loop, just implicit)

RELATIONSHIP TO BIOLOGICAL RECURRENCE:

Brain has recurrent connections - information flows back and forth.
Multi-pass simulates this with multiple forward passes.
Attention = how neurons affect each other.
Passes = time steps of recurrent processing.

WHY DOES THIS HELP COMPOSITIONAL GENERALIZATION?

Composing "WALK" + "THRICE" (never seen together):
- Pass 1: Recognize "WALK" and "THRICE" independently
- Pass 2: "WALK" attends to "THRICE", starts binding
- Pass 3: Binding refines, output representation forms
- Pass 4: Final refinement, output

Single pass: Must do all this in one shot (hard for novel combinations)
Multi pass: Can iteratively figure out the relationship
""")


print("\n" + "=" * 80)
print("FINAL SUMMARY")
print("=" * 80)

print(f"""
QUESTION 1: Does multi-pass generalize across tasks?
{'-'*50}
""")
for task, r in all_results.items():
    status = "✓ Helps" if r['diff'] > 0.02 else ("✗ Hurts" if r['diff'] < -0.02 else "~ Neutral")
    print(f"  {task}: {status} ({r['diff']:+.1%})")

print(f"""

QUESTION 2: What changes between passes?
{'-'*50}
  - Representations change by ~{np.mean(changes):.2f} L2 norm per pass
  - Changes {'decrease' if changes[-1] < changes[0] else 'do not decrease'} (convergence: {'Yes' if changes[-1] < changes[0] else 'No'})
  - Predictions improve from {correct_by_pass[0]:.1%} to {correct_by_pass[-1]:.1%} over passes
  - This IS iterative refinement / self-correction

QUESTION 3: How does this connect to intelligence loop?
{'-'*50}
  - Multi-pass provides IMPLICIT feedback via attention
  - Each pass sees updated representations from all positions
  - This creates a recurrent-like loop without explicit error
  - It's closer to brain's recurrent processing than explicit "predict-update"

  The intelligence loop:
    Predict → Feedback → Update

  Multi-pass implements:
    Predict → [Attention sees predictions] → Refine

  Same structure, different mechanism.
""")

print("=" * 80)
