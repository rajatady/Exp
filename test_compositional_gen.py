"""
Compositional Generalization Test

The key test: Can the model combine known parts in NEW ways?

Task: SCAN-inspired command execution
- Primitives: WALK, RUN, JUMP, LOOK
- Modifiers: TWICE, THRICE, AND
- Train on SOME combinations
- Test on HELD-OUT combinations

Example:
  "WALK" → "W"
  "RUN TWICE" → "RR"
  "JUMP AND LOOK" → "JL"

  Train: WALK with TWICE, RUN with THRICE
  Test: WALK with THRICE (never seen this combo)

This tests TRUE compositional generalization:
- Can the model understand "THRICE" means repeat 3x
- And apply it to WALK even though it only saw WALK+TWICE?
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
print("COMPOSITIONAL GENERALIZATION TEST")
print("=" * 80)


# =============================================================================
# DATA GENERATION
# =============================================================================

# Vocabulary
PRIMITIVES = ['WALK', 'RUN', 'JUMP', 'LOOK']
MODIFIERS = ['TWICE', 'THRICE']
CONNECTORS = ['AND', 'AFTER']

# Output tokens
PRIM_TO_OUT = {'WALK': 'W', 'RUN': 'R', 'JUMP': 'J', 'LOOK': 'L'}

# Build vocabulary
SPECIAL = ['<PAD>', '<BOS>', '<EOS>', '<SEP>']
INPUT_VOCAB = SPECIAL + PRIMITIVES + MODIFIERS + CONNECTORS
OUTPUT_VOCAB = SPECIAL + list(PRIM_TO_OUT.values())

INPUT_TO_ID = {w: i for i, w in enumerate(INPUT_VOCAB)}
OUTPUT_TO_ID = {w: i for i, w in enumerate(OUTPUT_VOCAB)}
ID_TO_OUTPUT = {i: w for w, i in OUTPUT_TO_ID.items()}

VOCAB_SIZE = max(len(INPUT_VOCAB), len(OUTPUT_VOCAB))


def execute_command(cmd_tokens):
    """Execute a command and return output tokens."""
    result = []
    i = 0
    while i < len(cmd_tokens):
        token = cmd_tokens[i]

        if token in PRIMITIVES:
            out = PRIM_TO_OUT[token]

            # Check for modifier
            if i + 1 < len(cmd_tokens):
                next_token = cmd_tokens[i + 1]
                if next_token == 'TWICE':
                    result.extend([out, out])
                    i += 2
                    continue
                elif next_token == 'THRICE':
                    result.extend([out, out, out])
                    i += 2
                    continue

            result.append(out)
            i += 1

        elif token in CONNECTORS:
            # AND and AFTER just sequence the outputs
            i += 1
        else:
            i += 1

    return result


def generate_all_combinations():
    """Generate all possible command combinations."""
    all_data = []

    # Single primitives
    for p in PRIMITIVES:
        cmd = [p]
        out = execute_command(cmd)
        all_data.append((cmd, out))

    # Primitive + modifier
    for p in PRIMITIVES:
        for m in MODIFIERS:
            cmd = [p, m]
            out = execute_command(cmd)
            all_data.append((cmd, out))

    # Two primitives with connector
    for p1 in PRIMITIVES:
        for conn in CONNECTORS:
            for p2 in PRIMITIVES:
                cmd = [p1, conn, p2]
                out = execute_command(cmd)
                all_data.append((cmd, out))

    # Primitive + modifier + connector + primitive
    for p1 in PRIMITIVES:
        for m in MODIFIERS:
            for conn in CONNECTORS:
                for p2 in PRIMITIVES:
                    cmd = [p1, m, conn, p2]
                    out = execute_command(cmd)
                    all_data.append((cmd, out))

    # Primitive + connector + primitive + modifier
    for p1 in PRIMITIVES:
        for conn in CONNECTORS:
            for p2 in PRIMITIVES:
                for m in MODIFIERS:
                    cmd = [p1, conn, p2, m]
                    out = execute_command(cmd)
                    all_data.append((cmd, out))

    return all_data


def create_compositional_split(all_data, holdout_combos):
    """
    Split data into train/test where test has NOVEL combinations.

    holdout_combos: list of (primitive, modifier) pairs to hold out
    """
    train_data = []
    test_data = []

    for cmd, out in all_data:
        is_holdout = False

        # Check if this command contains a held-out combination
        for i, token in enumerate(cmd):
            if token in PRIMITIVES:
                if i + 1 < len(cmd) and cmd[i + 1] in MODIFIERS:
                    combo = (token, cmd[i + 1])
                    if combo in holdout_combos:
                        is_holdout = True
                        break

        if is_holdout:
            test_data.append((cmd, out))
        else:
            train_data.append((cmd, out))

    return train_data, test_data


def to_tensor(cmd, out):
    """Convert command and output to tensor sequences."""
    # Input: <BOS> cmd... <SEP> out... <EOS>
    input_ids = [INPUT_TO_ID['<BOS>']]
    input_ids.extend([INPUT_TO_ID[t] for t in cmd])
    input_ids.append(INPUT_TO_ID['<SEP>'])
    input_ids.extend([OUTPUT_TO_ID[t] for t in out])
    input_ids.append(OUTPUT_TO_ID['<EOS>'])

    return input_ids


def pad_sequences(seqs, pad_id=0):
    max_len = max(len(s) for s in seqs)
    return [s + [pad_id] * (max_len - len(s)) for s in seqs]


# =============================================================================
# MODELS
# =============================================================================

class StandardNetwork(nn.Module):
    def __init__(self, vocab_size, hidden_dim=64, n_layers=4):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.pos_enc = nn.Parameter(torch.randn(1, 128, hidden_dim) * 0.02)

        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim, nhead=4,
                dim_feedforward=hidden_dim*4,
                batch_first=True, dropout=0.1
            ) for _ in range(n_layers)
        ])

        self.ln = nn.LayerNorm(hidden_dim)
        self.output = nn.Linear(hidden_dim, vocab_size)
        self.name = "Standard"

    def forward(self, x):
        B, T = x.shape
        h = self.embed(x) + self.pos_enc[:, :T, :]
        for layer in self.layers:
            h = layer(h)
        return self.output(self.ln(h))


class InnerLoopNetwork(nn.Module):
    """Inner loop with settling toward input consistency."""

    def __init__(self, vocab_size, hidden_dim=64, n_layers=2, n_inner_steps=8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_inner_steps = n_inner_steps

        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.pos_enc = nn.Parameter(torch.randn(1, 128, hidden_dim) * 0.02)

        self.encoder = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim, nhead=4,
                dim_feedforward=hidden_dim*4,
                batch_first=True, dropout=0.1
            ) for _ in range(n_layers)
        ])

        # Predictor: state → predicted embedding
        self.embed_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.ln = nn.LayerNorm(hidden_dim)
        self.output = nn.Linear(hidden_dim, vocab_size)
        self.name = f"Inner Loop (steps={n_inner_steps})"

    def forward(self, x):
        B, T = x.shape

        # Input embedding (ground truth)
        input_embed = self.embed(x) + self.pos_enc[:, :T, :]

        # Initial state
        h = input_embed.clone()
        for layer in self.encoder:
            h = layer(h)

        # Inner loop: settle toward consistency with input
        for step in range(self.n_inner_steps):
            # State predicts what embedding should be
            pred_embed = self.embed_predictor(h)

            # Error: difference from actual input embedding
            error = pred_embed - input_embed

            # Update: move state to reduce error
            h = h - 0.1 * error

            # Re-process (integrate information)
            for layer in self.encoder:
                h = layer(h)

        return self.output(self.ln(h))


class AccumulatorNetwork(nn.Module):
    """Simple accumulator (what we found works)."""

    def __init__(self, vocab_size, hidden_dim=64, n_layers=2, n_ticks=4):
        super().__init__()
        self.n_ticks = n_ticks

        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.pos_enc = nn.Parameter(torch.randn(1, 128, hidden_dim) * 0.02)

        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim, nhead=4,
                dim_feedforward=hidden_dim*4,
                batch_first=True, dropout=0.1
            ) for _ in range(n_layers)
        ])

        self.ln = nn.LayerNorm(hidden_dim)
        self.output = nn.Linear(hidden_dim, vocab_size)
        self.name = f"Accumulator (ticks={n_ticks})"

    def forward(self, x):
        B, T = x.shape
        h = self.embed(x) + self.pos_enc[:, :T, :]

        for tick in range(self.n_ticks):
            h_new = h
            for layer in self.layers:
                h_new = layer(h_new)
            h = h + h_new  # Accumulate

        return self.output(self.ln(h))


# =============================================================================
# TRAINING & EVALUATION
# =============================================================================

def train_model(model, train_data, n_epochs=200, lr=1e-3, device='cpu'):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # Convert to tensors
    seqs = [to_tensor(cmd, out) for cmd, out in train_data]

    for epoch in range(n_epochs):
        model.train()
        random.shuffle(seqs)

        batch = pad_sequences(seqs)
        x = torch.tensor(batch, device=device)

        inputs = x[:, :-1]
        targets = x[:, 1:]

        optimizer.zero_grad()
        logits = model(inputs)

        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            ignore_index=0
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    return model


def evaluate(model, data, device='cpu'):
    """Evaluate accuracy on output portion only."""
    model.eval()

    seqs = [to_tensor(cmd, out) for cmd, out in data]

    correct = 0
    total = 0
    seq_correct = 0

    with torch.no_grad():
        batch = pad_sequences(seqs)
        x = torch.tensor(batch, device=device)

        inputs = x[:, :-1]
        targets = x[:, 1:]

        logits = model(inputs)
        preds = logits.argmax(dim=-1)

        SEP_ID = INPUT_TO_ID['<SEP>']

        for i, (cmd, out) in enumerate(data):
            seq = seqs[i]

            # Find SEP position
            sep_pos = seq.index(SEP_ID)

            # Output starts after SEP
            start = sep_pos
            end = len(seq) - 1  # Before final token

            if start < end:
                pred_part = preds[i, start:end].tolist()
                target_part = targets[i, start:end].tolist()

                matches = sum(1 for p, t in zip(pred_part, target_part) if p == t)
                correct += matches
                total += len(target_part)

                if pred_part == target_part:
                    seq_correct += 1

    return {
        'token_acc': correct / total if total > 0 else 0,
        'seq_acc': seq_correct / len(data) if data else 0
    }


def count_params(model):
    return sum(p.numel() for p in model.parameters())


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

print("\n1. Setting up compositional split...")

# Generate all data
all_data = generate_all_combinations()
print(f"   Total combinations: {len(all_data)}")

# Hold out specific primitive+modifier combinations
# Train: WALK+TWICE, RUN+TWICE, JUMP+THRICE, LOOK+THRICE
# Test:  WALK+THRICE, RUN+THRICE, JUMP+TWICE, LOOK+TWICE
holdout_combos = [
    ('WALK', 'THRICE'),
    ('RUN', 'THRICE'),
    ('JUMP', 'TWICE'),
    ('LOOK', 'TWICE'),
]

train_data, test_data = create_compositional_split(all_data, holdout_combos)

print(f"   Train: {len(train_data)} examples")
print(f"   Test (novel combos): {len(test_data)} examples")

# Show examples
print("\n   Train examples:")
for cmd, out in train_data[:5]:
    print(f"      {' '.join(cmd)} → {''.join(out)}")

print("\n   Test examples (NOVEL combinations):")
for cmd, out in test_data[:5]:
    print(f"      {' '.join(cmd)} → {''.join(out)}")

# Device
if torch.backends.mps.is_available():
    device = torch.device('mps')
elif torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print(f"\n   Device: {device}")

HIDDEN_DIM = 64

print("\n2. Creating models...")

models = {
    "Standard": StandardNetwork(VOCAB_SIZE, HIDDEN_DIM, n_layers=4),
    "Inner Loop": InnerLoopNetwork(VOCAB_SIZE, HIDDEN_DIM, n_layers=2, n_inner_steps=8),
    "Accumulator": AccumulatorNetwork(VOCAB_SIZE, HIDDEN_DIM, n_layers=2, n_ticks=4),
}

for name, model in models.items():
    model = model.to(device)
    print(f"   {model.name}: {count_params(model):,} params")


print("\n" + "=" * 80)
print("TRAINING AND EVALUATION")
print("=" * 80)

results = {}
for name, model in models.items():
    print(f"\n   Training {model.name}...")
    model = train_model(model, train_data, n_epochs=300, device=device)

    train_acc = evaluate(model, train_data, device)
    test_acc = evaluate(model, test_data, device)

    results[name] = {
        'train_token': train_acc['token_acc'],
        'train_seq': train_acc['seq_acc'],
        'test_token': test_acc['token_acc'],
        'test_seq': test_acc['seq_acc'],
    }

    print(f"   → Train: {train_acc['seq_acc']:.1%} seq, {train_acc['token_acc']:.1%} token")
    print(f"   → Test (novel): {test_acc['seq_acc']:.1%} seq, {test_acc['token_acc']:.1%} token")


print("\n" + "=" * 80)
print("RESULTS: COMPOSITIONAL GENERALIZATION")
print("=" * 80)

print(f"\n{'Model':<20} {'Train Seq':>10} {'Train Tok':>10} {'Test Seq':>10} {'Test Tok':>10}")
print("-" * 65)
for name, r in results.items():
    print(f"{name:<20} {r['train_seq']:>9.1%} {r['train_token']:>9.1%} {r['test_seq']:>9.1%} {r['test_token']:>9.1%}")


print("\n" + "=" * 80)
print("ANALYSIS")
print("=" * 80)

# Compositional gap = train - test
print("\nCompositional Generalization Gap (Train - Test on seq accuracy):")
for name, r in results.items():
    gap = r['train_seq'] - r['test_seq']
    print(f"   {name}: {gap:+.1%} gap")

best_test = max(results.items(), key=lambda x: x[1]['test_seq'])
print(f"\nBest on novel combinations: {best_test[0]} ({best_test[1]['test_seq']:.1%})")

# Check if inner loop helps
inner_test = results['Inner Loop']['test_seq']
standard_test = results['Standard']['test_seq']
accum_test = results['Accumulator']['test_seq']

print(f"""
COMPARISON:
  Standard:    {standard_test:.1%} on novel combos
  Inner Loop:  {inner_test:.1%} on novel combos  ({inner_test - standard_test:+.1%} vs Standard)
  Accumulator: {accum_test:.1%} on novel combos  ({accum_test - standard_test:+.1%} vs Standard)
""")

if inner_test > standard_test + 0.05:
    print("→ Inner Loop helps with compositional generalization!")
    print("  Self-consistency with input enables novel combinations")
elif accum_test > standard_test + 0.05:
    print("→ Accumulator helps with compositional generalization!")
    print("  Iterative refinement enables novel combinations")
else:
    print("→ No clear winner on compositional generalization")
    print("  All models struggle equally with novel combinations")


print("\n" + "=" * 80)
print("DETAILED BREAKDOWN: Which combinations fail?")
print("=" * 80)

# Analyze which specific holdout combinations fail
model = models['Standard'].to(device)
model.eval()

print("\nPer-combination accuracy (Standard model):")
for combo in holdout_combos:
    prim, mod = combo

    # Find test examples with this combo
    combo_examples = [(cmd, out) for cmd, out in test_data
                      if prim in cmd and mod in cmd]

    if combo_examples:
        acc = evaluate(model, combo_examples, device)
        print(f"   {prim} + {mod}: {acc['seq_acc']:.1%} ({len(combo_examples)} examples)")


print("\n" + "=" * 80)
