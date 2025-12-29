"""
Compositional Generalization Test - Multi-Seed Version

Testing across multiple seeds to account for variance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import math
from collections import defaultdict

print("=" * 80)
print("COMPOSITIONAL GENERALIZATION - MULTI-SEED TEST")
print("=" * 80)


# =============================================================================
# DATA GENERATION (same as before)
# =============================================================================

PRIMITIVES = ['WALK', 'RUN', 'JUMP', 'LOOK']
MODIFIERS = ['TWICE', 'THRICE']
CONNECTORS = ['AND', 'AFTER']
PRIM_TO_OUT = {'WALK': 'W', 'RUN': 'R', 'JUMP': 'J', 'LOOK': 'L'}

SPECIAL = ['<PAD>', '<BOS>', '<EOS>', '<SEP>']
INPUT_VOCAB = SPECIAL + PRIMITIVES + MODIFIERS + CONNECTORS
OUTPUT_VOCAB = SPECIAL + list(PRIM_TO_OUT.values())

INPUT_TO_ID = {w: i for i, w in enumerate(INPUT_VOCAB)}
OUTPUT_TO_ID = {w: i for i, w in enumerate(OUTPUT_VOCAB)}
VOCAB_SIZE = max(len(INPUT_VOCAB), len(OUTPUT_VOCAB))


def execute_command(cmd_tokens):
    result = []
    i = 0
    while i < len(cmd_tokens):
        token = cmd_tokens[i]
        if token in PRIMITIVES:
            out = PRIM_TO_OUT[token]
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
            i += 1
        else:
            i += 1
    return result


def generate_all_combinations():
    all_data = []
    for p in PRIMITIVES:
        all_data.append(([p], execute_command([p])))
    for p in PRIMITIVES:
        for m in MODIFIERS:
            cmd = [p, m]
            all_data.append((cmd, execute_command(cmd)))
    for p1 in PRIMITIVES:
        for conn in CONNECTORS:
            for p2 in PRIMITIVES:
                cmd = [p1, conn, p2]
                all_data.append((cmd, execute_command(cmd)))
    for p1 in PRIMITIVES:
        for m in MODIFIERS:
            for conn in CONNECTORS:
                for p2 in PRIMITIVES:
                    cmd = [p1, m, conn, p2]
                    all_data.append((cmd, execute_command(cmd)))
    for p1 in PRIMITIVES:
        for conn in CONNECTORS:
            for p2 in PRIMITIVES:
                for m in MODIFIERS:
                    cmd = [p1, conn, p2, m]
                    all_data.append((cmd, execute_command(cmd)))
    return all_data


def create_compositional_split(all_data, holdout_combos):
    train_data, test_data = [], []
    for cmd, out in all_data:
        is_holdout = False
        for i, token in enumerate(cmd):
            if token in PRIMITIVES:
                if i + 1 < len(cmd) and cmd[i + 1] in MODIFIERS:
                    if (token, cmd[i + 1]) in holdout_combos:
                        is_holdout = True
                        break
        if is_holdout:
            test_data.append((cmd, out))
        else:
            train_data.append((cmd, out))
    return train_data, test_data


def to_tensor(cmd, out):
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
        self.embed_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.ln = nn.LayerNorm(hidden_dim)
        self.output = nn.Linear(hidden_dim, vocab_size)
        self.name = "Inner Loop"

    def forward(self, x):
        B, T = x.shape
        input_embed = self.embed(x) + self.pos_enc[:, :T, :]
        h = input_embed.clone()
        for layer in self.encoder:
            h = layer(h)
        for step in range(self.n_inner_steps):
            pred_embed = self.embed_predictor(h)
            error = pred_embed - input_embed
            h = h - 0.1 * error
            for layer in self.encoder:
                h = layer(h)
        return self.output(self.ln(h))


class AccumulatorNetwork(nn.Module):
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
        self.name = "Accumulator"

    def forward(self, x):
        B, T = x.shape
        h = self.embed(x) + self.pos_enc[:, :T, :]
        for tick in range(self.n_ticks):
            h_new = h
            for layer in self.layers:
                h_new = layer(h_new)
            h = h + h_new
        return self.output(self.ln(h))


# =============================================================================
# TRAINING & EVALUATION
# =============================================================================

def train_model(model, train_data, n_epochs=300, lr=1e-3, device='cpu'):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
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
    model.eval()
    seqs = [to_tensor(cmd, out) for cmd, out in data]
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
            sep_pos = seq.index(SEP_ID)
            start = sep_pos
            end = len(seq) - 1
            if start < end:
                pred_part = preds[i, start:end].tolist()
                target_part = targets[i, start:end].tolist()
                if pred_part == target_part:
                    seq_correct += 1
    return seq_correct / len(data) if data else 0


# =============================================================================
# MULTI-SEED EXPERIMENT
# =============================================================================

# Setup
if torch.backends.mps.is_available():
    device = torch.device('mps')
elif torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print(f"\nDevice: {device}")

# Data
all_data = generate_all_combinations()
holdout_combos = [
    ('WALK', 'THRICE'),
    ('RUN', 'THRICE'),
    ('JUMP', 'TWICE'),
    ('LOOK', 'TWICE'),
]
train_data, test_data = create_compositional_split(all_data, holdout_combos)

print(f"Train: {len(train_data)}, Test: {len(test_data)}")

# Seeds
SEEDS = [42, 123, 456, 789, 1024]
HIDDEN_DIM = 64

results = defaultdict(list)

print(f"\nRunning {len(SEEDS)} seeds...")
print("-" * 60)

for seed in SEEDS:
    print(f"\nSeed {seed}:")

    # Set all seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Create fresh models
    models = {
        "Standard": StandardNetwork(VOCAB_SIZE, HIDDEN_DIM, n_layers=4).to(device),
        "Inner Loop": InnerLoopNetwork(VOCAB_SIZE, HIDDEN_DIM, n_layers=2, n_inner_steps=8).to(device),
        "Accumulator": AccumulatorNetwork(VOCAB_SIZE, HIDDEN_DIM, n_layers=2, n_ticks=4).to(device),
    }

    for name, model in models.items():
        model = train_model(model, train_data, n_epochs=300, device=device)
        train_acc = evaluate(model, train_data, device)
        test_acc = evaluate(model, test_data, device)
        results[name].append({
            'seed': seed,
            'train': train_acc,
            'test': test_acc
        })
        print(f"   {name}: train={train_acc:.1%}, test={test_acc:.1%}")


# =============================================================================
# AGGREGATE RESULTS
# =============================================================================

print("\n" + "=" * 80)
print("AGGREGATE RESULTS ACROSS SEEDS")
print("=" * 80)

print(f"\n{'Model':<15} {'Train (mean±std)':>20} {'Test (mean±std)':>20}")
print("-" * 60)

summary = {}
for name in ["Standard", "Inner Loop", "Accumulator"]:
    train_accs = [r['train'] for r in results[name]]
    test_accs = [r['test'] for r in results[name]]

    train_mean = np.mean(train_accs)
    train_std = np.std(train_accs)
    test_mean = np.mean(test_accs)
    test_std = np.std(test_accs)

    summary[name] = {
        'train_mean': train_mean,
        'train_std': train_std,
        'test_mean': test_mean,
        'test_std': test_std,
        'test_accs': test_accs
    }

    print(f"{name:<15} {train_mean:>7.1%} ± {train_std:>5.1%}     {test_mean:>7.1%} ± {test_std:>5.1%}")


print("\n" + "=" * 80)
print("STATISTICAL COMPARISON")
print("=" * 80)

# Compare Inner Loop vs Standard
inner_tests = summary['Inner Loop']['test_accs']
standard_tests = summary['Standard']['test_accs']
accum_tests = summary['Accumulator']['test_accs']

# Paired differences
inner_vs_standard = [i - s for i, s in zip(inner_tests, standard_tests)]
accum_vs_standard = [a - s for a, s in zip(accum_tests, standard_tests)]

print(f"\nInner Loop vs Standard (per-seed differences):")
for i, (seed, diff) in enumerate(zip(SEEDS, inner_vs_standard)):
    print(f"   Seed {seed}: {diff:+.1%}")
print(f"   Mean difference: {np.mean(inner_vs_standard):+.1%} ± {np.std(inner_vs_standard):.1%}")

# Check if improvement is consistent
wins = sum(1 for d in inner_vs_standard if d > 0)
losses = sum(1 for d in inner_vs_standard if d < 0)
ties = sum(1 for d in inner_vs_standard if d == 0)

print(f"\n   Wins: {wins}, Losses: {losses}, Ties: {ties}")

if wins > losses and np.mean(inner_vs_standard) > 0.05:
    print(f"\n✓ INNER LOOP CONSISTENTLY HELPS (+{np.mean(inner_vs_standard):.1%} on average)")
elif wins > losses:
    print(f"\n~ Inner Loop helps but effect is small (+{np.mean(inner_vs_standard):.1%})")
else:
    print(f"\n✗ Inner Loop does NOT consistently help")


print(f"\nAccumulator vs Standard (per-seed differences):")
for i, (seed, diff) in enumerate(zip(SEEDS, accum_vs_standard)):
    print(f"   Seed {seed}: {diff:+.1%}")
print(f"   Mean difference: {np.mean(accum_vs_standard):+.1%} ± {np.std(accum_vs_standard):.1%}")


print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

best_model = max(summary.items(), key=lambda x: x[1]['test_mean'])
print(f"""
Best model on compositional generalization: {best_model[0]}
  Test accuracy: {best_model[1]['test_mean']:.1%} ± {best_model[1]['test_std']:.1%}

Inner Loop improvement over Standard:
  Mean: {np.mean(inner_vs_standard):+.1%}
  Consistent across seeds: {wins}/{len(SEEDS)} wins

Accumulator vs Standard:
  Mean: {np.mean(accum_vs_standard):+.1%}
""")

print("=" * 80)
