"""
Analyze Seed Variance in Inner Loop

Why does seed 789 fail (-14.7%) while seed 1024 succeeds (+26.5%)?

Hypotheses:
1. Different weight initializations lead to different inner loop dynamics
2. Some initializations make the inner loop "settle" better
3. The prediction error landscape is different
4. Training dynamics differ (loss curves, convergence)

Let's investigate.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import math
from collections import defaultdict

print("=" * 80)
print("ANALYZING SEED VARIANCE IN INNER LOOP")
print("=" * 80)

# =============================================================================
# DATA (same as before)
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
# MODELS WITH DIAGNOSTICS
# =============================================================================

class InnerLoopNetworkDiagnostic(nn.Module):
    """Inner Loop with diagnostic outputs."""

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

    def forward(self, x, return_diagnostics=False):
        B, T = x.shape
        input_embed = self.embed(x) + self.pos_enc[:, :T, :]
        h = input_embed.clone()
        for layer in self.encoder:
            h = layer(h)

        diagnostics = {
            'errors': [],
            'state_norms': [],
            'state_changes': [],
        }

        h_prev = h.clone()

        for step in range(self.n_inner_steps):
            pred_embed = self.embed_predictor(h)
            error = pred_embed - input_embed

            # Diagnostics
            error_norm = error.norm(dim=-1).mean().item()
            state_norm = h.norm(dim=-1).mean().item()
            state_change = (h - h_prev).norm(dim=-1).mean().item()

            diagnostics['errors'].append(error_norm)
            diagnostics['state_norms'].append(state_norm)
            diagnostics['state_changes'].append(state_change)

            h_prev = h.clone()
            h = h - 0.1 * error

            for layer in self.encoder:
                h = layer(h)

        logits = self.output(self.ln(h))

        if return_diagnostics:
            return logits, diagnostics
        return logits


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

    def forward(self, x):
        B, T = x.shape
        h = self.embed(x) + self.pos_enc[:, :T, :]
        for layer in self.layers:
            h = layer(h)
        return self.output(self.ln(h))


# =============================================================================
# TRAINING WITH LOSS TRACKING
# =============================================================================

def train_model_with_tracking(model, train_data, n_epochs=300, lr=1e-3, device='cpu'):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    seqs = [to_tensor(cmd, out) for cmd, out in train_data]

    loss_history = []

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

        if epoch % 50 == 0 or epoch == n_epochs - 1:
            loss_history.append((epoch, loss.item()))

    return model, loss_history


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
# ANALYZE SEEDS
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
holdout_combos = [
    ('WALK', 'THRICE'),
    ('RUN', 'THRICE'),
    ('JUMP', 'TWICE'),
    ('LOOK', 'TWICE'),
]
train_data, test_data = create_compositional_split(all_data, holdout_combos)

# Focus on the extreme seeds
SEEDS_TO_ANALYZE = {
    789: "FAILURE seed (-14.7%)",
    1024: "SUCCESS seed (+26.5%)",
    42: "BASELINE seed (+1.5%)",
}

HIDDEN_DIM = 64

print("\n" + "=" * 80)
print("COMPARING EXTREME SEEDS")
print("=" * 80)

seed_analysis = {}

for seed, label in SEEDS_TO_ANALYZE.items():
    print(f"\n{'='*60}")
    print(f"SEED {seed}: {label}")
    print(f"{'='*60}")

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Train Inner Loop
    inner_model = InnerLoopNetworkDiagnostic(VOCAB_SIZE, HIDDEN_DIM, n_layers=2, n_inner_steps=8).to(device)
    inner_model, inner_loss_history = train_model_with_tracking(inner_model, train_data, n_epochs=300, device=device)

    # Train Standard
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    std_model = StandardNetwork(VOCAB_SIZE, HIDDEN_DIM, n_layers=4).to(device)
    std_model, std_loss_history = train_model_with_tracking(std_model, train_data, n_epochs=300, device=device)

    # Evaluate
    inner_train = evaluate(inner_model, train_data, device)
    inner_test = evaluate(inner_model, test_data, device)
    std_train = evaluate(std_model, train_data, device)
    std_test = evaluate(std_model, test_data, device)

    print(f"\n  Accuracy:")
    print(f"    Standard:   train={std_train:.1%}, test={std_test:.1%}")
    print(f"    Inner Loop: train={inner_train:.1%}, test={inner_test:.1%}")
    print(f"    Difference: {inner_test - std_test:+.1%}")

    # Get inner loop diagnostics on test data
    inner_model.eval()
    test_seqs = [to_tensor(cmd, out) for cmd, out in test_data]
    batch = pad_sequences(test_seqs)
    x = torch.tensor(batch, device=device)
    inputs = x[:, :-1]

    with torch.no_grad():
        _, diagnostics = inner_model(inputs, return_diagnostics=True)

    print(f"\n  Inner Loop Dynamics (on test data):")
    print(f"    Error norm over steps: {[f'{e:.3f}' for e in diagnostics['errors']]}")
    print(f"    State norm over steps: {[f'{n:.2f}' for n in diagnostics['state_norms']]}")

    # Error reduction
    error_reduction = (diagnostics['errors'][0] - diagnostics['errors'][-1]) / diagnostics['errors'][0]
    print(f"    Error reduction: {error_reduction:.1%}")

    # State stability
    state_change_trend = diagnostics['state_changes'][-1] / (diagnostics['state_changes'][0] + 1e-8)
    print(f"    State change trend: {state_change_trend:.2f}x (final/initial)")

    # Training loss convergence
    print(f"\n  Training Loss:")
    for epoch, loss in inner_loss_history:
        print(f"    Epoch {epoch}: {loss:.4f}")

    seed_analysis[seed] = {
        'inner_test': inner_test,
        'std_test': std_test,
        'diff': inner_test - std_test,
        'error_reduction': error_reduction,
        'state_change_trend': state_change_trend,
        'final_loss': inner_loss_history[-1][1],
        'diagnostics': diagnostics,
    }


print("\n" + "=" * 80)
print("COMPARISON SUMMARY")
print("=" * 80)

print(f"\n{'Seed':<10} {'Diff':>10} {'Err Reduce':>12} {'State Trend':>12} {'Final Loss':>12}")
print("-" * 60)

for seed in SEEDS_TO_ANALYZE.keys():
    a = seed_analysis[seed]
    print(f"{seed:<10} {a['diff']:>+9.1%} {a['error_reduction']:>11.1%} {a['state_change_trend']:>11.2f}x {a['final_loss']:>11.4f}")


print("\n" + "=" * 80)
print("ANALYSIS")
print("=" * 80)

success = seed_analysis[1024]
failure = seed_analysis[789]
baseline = seed_analysis[42]

print(f"""
SUCCESS (seed 1024) vs FAILURE (seed 789):

1. Error Reduction:
   - Success: {success['error_reduction']:.1%} reduction
   - Failure: {failure['error_reduction']:.1%} reduction
   - Baseline: {baseline['error_reduction']:.1%} reduction

2. State Change Trend:
   - Success: {success['state_change_trend']:.2f}x
   - Failure: {failure['state_change_trend']:.2f}x
   - Baseline: {baseline['state_change_trend']:.2f}x

3. Final Training Loss:
   - Success: {success['final_loss']:.4f}
   - Failure: {failure['final_loss']:.4f}
   - Baseline: {baseline['final_loss']:.4f}
""")

# Look for patterns
if success['error_reduction'] > failure['error_reduction']:
    print("→ Success seed has BETTER error reduction")
else:
    print("→ Failure seed has better error reduction (unexpected!)")

if success['state_change_trend'] < failure['state_change_trend']:
    print("→ Success seed has MORE STABLE state (smaller change trend)")
else:
    print("→ Failure seed has more stable state (unexpected!)")

print("\n" + "=" * 80)
