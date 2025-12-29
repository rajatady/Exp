"""
Alternative Hypothesis Testing

If the inner loop's error doesn't actually decrease, what provides the benefit?

Hypotheses:
1. The ARCHITECTURE (predicting input) forces better representations during TRAINING
2. Multiple encoder passes help integration, even without error reduction
3. The state updates provide implicit regularization
4. It's just the extra computation (more parameters effectively)

Tests:
A) Input prediction LOSS during training (no inner loop at inference)
B) Multiple encoder passes without state update
C) Inner loop only at training, not inference
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
print("ALTERNATIVE HYPOTHESIS TESTING")
print("=" * 80)

# =============================================================================
# DATA (compositional task)
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


def create_split(all_data, holdout_combos):
    train, test = [], []
    for cmd, out in all_data:
        is_holdout = False
        for i, token in enumerate(cmd):
            if token in PRIMITIVES:
                if i + 1 < len(cmd) and cmd[i + 1] in MODIFIERS:
                    if (token, cmd[i + 1]) in holdout_combos:
                        is_holdout = True
                        break
        (test if is_holdout else train).append((cmd, out))
    return train, test


def to_tensor(cmd, out):
    ids = [INPUT_TO_ID['<BOS>']]
    ids.extend([INPUT_TO_ID[t] for t in cmd])
    ids.append(INPUT_TO_ID['<SEP>'])
    ids.extend([OUTPUT_TO_ID[t] for t in out])
    ids.append(OUTPUT_TO_ID['<EOS>'])
    return ids


def pad_sequences(seqs, pad_id=0):
    max_len = max(len(s) for s in seqs)
    return [s + [pad_id] * (max_len - len(s)) for s in seqs]


# =============================================================================
# MODELS
# =============================================================================

class StandardNetwork(nn.Module):
    """Baseline: Standard transformer."""
    def __init__(self, vocab_size, hidden_dim=64, n_layers=4):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.pos_enc = nn.Parameter(torch.randn(1, 128, hidden_dim) * 0.02)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4,
                dim_feedforward=hidden_dim*4, batch_first=True, dropout=0.1)
            for _ in range(n_layers)
        ])
        self.ln = nn.LayerNorm(hidden_dim)
        self.output = nn.Linear(hidden_dim, vocab_size)
        self.name = "Standard"

    def forward(self, x):
        h = self.embed(x) + self.pos_enc[:, :x.size(1), :]
        for layer in self.layers:
            h = layer(h)
        return self.output(self.ln(h))


class InputPredictionLoss(nn.Module):
    """
    Hypothesis A: Add input prediction as auxiliary loss during TRAINING.
    No inner loop at inference.
    """
    def __init__(self, vocab_size, hidden_dim=64, n_layers=4):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.pos_enc = nn.Parameter(torch.randn(1, 128, hidden_dim) * 0.02)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4,
                dim_feedforward=hidden_dim*4, batch_first=True, dropout=0.1)
            for _ in range(n_layers)
        ])
        # Auxiliary: predict input embedding from hidden state
        self.embed_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.ln = nn.LayerNorm(hidden_dim)
        self.output = nn.Linear(hidden_dim, vocab_size)
        self.name = "Input Pred Loss"

    def forward(self, x, return_pred_loss=False):
        input_embed = self.embed(x) + self.pos_enc[:, :x.size(1), :]
        h = input_embed.clone()
        for layer in self.layers:
            h = layer(h)
        logits = self.output(self.ln(h))

        if return_pred_loss:
            pred_embed = self.embed_predictor(h)
            pred_loss = F.mse_loss(pred_embed, input_embed)
            return logits, pred_loss
        return logits


class MultiPassEncoder(nn.Module):
    """
    Hypothesis B: Multiple encoder passes without state update.
    Just run encoder multiple times.
    """
    def __init__(self, vocab_size, hidden_dim=64, n_layers=2, n_passes=4):
        super().__init__()
        self.n_passes = n_passes
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.pos_enc = nn.Parameter(torch.randn(1, 128, hidden_dim) * 0.02)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4,
                dim_feedforward=hidden_dim*4, batch_first=True, dropout=0.1)
            for _ in range(n_layers)
        ])
        self.ln = nn.LayerNorm(hidden_dim)
        self.output = nn.Linear(hidden_dim, vocab_size)
        self.name = f"Multi-Pass ({n_passes}x)"

    def forward(self, x):
        h = self.embed(x) + self.pos_enc[:, :x.size(1), :]
        for _ in range(self.n_passes):
            for layer in self.layers:
                h = layer(h)
        return self.output(self.ln(h))


class InnerLoopFixed(nn.Module):
    """
    Fixed inner loop: No encoder rerun, proper settling.
    """
    def __init__(self, vocab_size, hidden_dim=64, n_layers=2, n_inner_steps=8):
        super().__init__()
        self.n_inner_steps = n_inner_steps
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.pos_enc = nn.Parameter(torch.randn(1, 128, hidden_dim) * 0.02)
        self.encoder = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4,
                dim_feedforward=hidden_dim*4, batch_first=True, dropout=0.1)
            for _ in range(n_layers)
        ])
        self.embed_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.ln = nn.LayerNorm(hidden_dim)
        self.output = nn.Linear(hidden_dim, vocab_size)
        self.name = "Inner Loop (Fixed)"

    def forward(self, x):
        input_embed = self.embed(x) + self.pos_enc[:, :x.size(1), :]
        h = input_embed.clone()
        for layer in self.encoder:
            h = layer(h)

        # Inner loop WITHOUT re-running encoder
        for step in range(self.n_inner_steps):
            pred_embed = self.embed_predictor(h)
            error = pred_embed - input_embed
            h = h - 0.1 * error
            # NO encoder rerun here

        return self.output(self.ln(h))


class InnerLoopTrainOnly(nn.Module):
    """
    Hypothesis C: Inner loop during training, standard forward at inference.
    """
    def __init__(self, vocab_size, hidden_dim=64, n_layers=2, n_inner_steps=4):
        super().__init__()
        self.n_inner_steps = n_inner_steps
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.pos_enc = nn.Parameter(torch.randn(1, 128, hidden_dim) * 0.02)
        self.encoder = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4,
                dim_feedforward=hidden_dim*4, batch_first=True, dropout=0.1)
            for _ in range(n_layers)
        ])
        self.embed_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.ln = nn.LayerNorm(hidden_dim)
        self.output = nn.Linear(hidden_dim, vocab_size)
        self.name = "Inner Loop (Train Only)"

    def forward(self, x):
        input_embed = self.embed(x) + self.pos_enc[:, :x.size(1), :]
        h = input_embed.clone()
        for layer in self.encoder:
            h = layer(h)

        # Only do inner loop during training
        if self.training:
            for step in range(self.n_inner_steps):
                pred_embed = self.embed_predictor(h)
                error = pred_embed - input_embed
                h = h - 0.1 * error

        return self.output(self.ln(h))


# =============================================================================
# TRAINING
# =============================================================================

def train_model(model, train_data, n_epochs=300, lr=1e-3, device='cpu',
                use_pred_loss=False, pred_loss_weight=0.1):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    seqs = [to_tensor(cmd, out) for cmd, out in train_data]

    for epoch in range(n_epochs):
        model.train()
        random.shuffle(seqs)
        batch = pad_sequences(seqs)
        x = torch.tensor(batch, device=device)
        inputs, targets = x[:, :-1], x[:, 1:]

        optimizer.zero_grad()

        if use_pred_loss and hasattr(model, 'forward') and 'return_pred_loss' in model.forward.__code__.co_varnames:
            logits, pred_loss = model(inputs, return_pred_loss=True)
            main_loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=0)
            loss = main_loss + pred_loss_weight * pred_loss
        else:
            logits = model(inputs)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=0)

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
        inputs, targets = x[:, :-1], x[:, 1:]
        logits = model(inputs)
        preds = logits.argmax(dim=-1)
        SEP_ID = INPUT_TO_ID['<SEP>']
        for i, (cmd, out) in enumerate(data):
            seq = seqs[i]
            sep_pos = seq.index(SEP_ID)
            start, end = sep_pos, len(seq) - 1
            if start < end:
                if preds[i, start:end].tolist() == targets[i, start:end].tolist():
                    seq_correct += 1
    return seq_correct / len(data) if data else 0


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

if torch.backends.mps.is_available():
    device = torch.device('mps')
elif torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print(f"\nDevice: {device}")

# Data
all_data = generate_all_combinations()
holdout = [('WALK', 'THRICE'), ('RUN', 'THRICE'), ('JUMP', 'TWICE'), ('LOOK', 'TWICE')]
train_data, test_data = create_split(all_data, holdout)

print(f"Train: {len(train_data)}, Test: {len(test_data)}")

HIDDEN_DIM = 64
N_SEEDS = 3

results = defaultdict(list)

models_to_test = [
    ("Standard", lambda: StandardNetwork(VOCAB_SIZE, HIDDEN_DIM, n_layers=4)),
    ("Input Pred Loss", lambda: InputPredictionLoss(VOCAB_SIZE, HIDDEN_DIM, n_layers=4)),
    ("Multi-Pass (4x)", lambda: MultiPassEncoder(VOCAB_SIZE, HIDDEN_DIM, n_layers=2, n_passes=4)),
    ("Inner Loop (Fixed)", lambda: InnerLoopFixed(VOCAB_SIZE, HIDDEN_DIM, n_layers=2, n_inner_steps=8)),
    ("Inner Loop (Train Only)", lambda: InnerLoopTrainOnly(VOCAB_SIZE, HIDDEN_DIM, n_layers=2, n_inner_steps=4)),
]

print(f"\nTesting {len(models_to_test)} models across {N_SEEDS} seeds...")
print("-" * 60)

for seed in range(N_SEEDS):
    print(f"\nSeed {seed}:")

    for name, model_fn in models_to_test:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        model = model_fn().to(device)

        use_pred_loss = (name == "Input Pred Loss")
        model = train_model(model, train_data, n_epochs=300, device=device,
                           use_pred_loss=use_pred_loss)

        test_acc = evaluate(model, test_data, device)
        results[name].append(test_acc)
        print(f"   {name}: {test_acc:.1%}")


print("\n" + "=" * 80)
print("RESULTS SUMMARY")
print("=" * 80)

print(f"\n{'Model':<25} {'Mean':>10} {'Std':>10} {'Individual Runs'}")
print("-" * 70)

for name, _ in models_to_test:
    accs = results[name]
    mean = np.mean(accs)
    std = np.std(accs)
    runs = ", ".join([f"{a:.1%}" for a in accs])
    print(f"{name:<25} {mean:>9.1%} {std:>9.1%}   [{runs}]")


print("\n" + "=" * 80)
print("ANALYSIS")
print("=" * 80)

baseline = np.mean(results["Standard"])
for name, _ in models_to_test[1:]:
    mean = np.mean(results[name])
    diff = mean - baseline
    print(f"{name}: {diff:+.1%} vs Standard")


print("\n" + "=" * 80)
print("CONCLUSIONS")
print("=" * 80)

best = max([(name, np.mean(accs)) for name, accs in results.items()], key=lambda x: x[1])
print(f"""
Best model: {best[0]} ({best[1]:.1%})

What this tells us about why the inner loop helps (or doesn't):
""")

if np.mean(results["Input Pred Loss"]) > baseline + 0.02:
    print("✓ Input prediction LOSS helps → forcing input preservation during training matters")
else:
    print("✗ Input prediction loss doesn't help alone → it's not just about training signal")

if np.mean(results["Multi-Pass (4x)"]) > baseline + 0.02:
    print("✓ Multi-pass helps → extra computation/integration matters")
else:
    print("✗ Multi-pass doesn't help alone → it's not just about more forward passes")

if np.mean(results["Inner Loop (Train Only)"]) > baseline + 0.02:
    print("✓ Inner loop at train-only helps → it changes what the model learns")
else:
    print("✗ Inner loop at train-only doesn't help → inference-time dynamics matter")

if np.mean(results["Inner Loop (Fixed)"]) > baseline + 0.02:
    print("✓ Fixed inner loop helps → the settling mechanism itself matters")
else:
    print("✗ Fixed inner loop doesn't help → the mechanism needs rethinking")

print("\n" + "=" * 80)
