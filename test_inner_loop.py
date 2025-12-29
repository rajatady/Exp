"""
Testing the Inner Loop Hypothesis

Core idea:
- Training time: Ground truth = labels (external supervision)
- Inference time: Ground truth = input itself (self-supervision)

The inner loop at inference:
1. Receive input
2. Internal state makes prediction about input
3. Compare prediction to actual input (prediction error)
4. Update internal state to reduce error
5. Repeat until converged
6. Then produce output

This is essentially: Predictive coding / Free energy minimization at inference time.
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
print("INNER LOOP HYPOTHESIS: Input as Ground Truth at Inference")
print("=" * 80)


# =============================================================================
# STANDARD NETWORK (No inner loop at inference)
# =============================================================================

class StandardNetwork(nn.Module):
    """Standard: Input → Forward → Output. No inner loop."""

    def __init__(self, vocab_size, hidden_dim=64, n_layers=4):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.pos_enc = nn.Parameter(torch.randn(1, 512, hidden_dim) * 0.02)

        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim, nhead=4,
                dim_feedforward=hidden_dim*4,
                batch_first=True, dropout=0.1
            ) for _ in range(n_layers)
        ])

        self.ln = nn.LayerNorm(hidden_dim)
        self.output = nn.Linear(hidden_dim, vocab_size)
        self.name = "Standard (no inner loop)"

    def forward(self, x):
        B, T = x.shape
        h = self.embed(x) + self.pos_enc[:, :T, :]
        for layer in self.layers:
            h = layer(h)
        return self.output(self.ln(h))


# =============================================================================
# INNER LOOP NETWORK (Predict input → Update state → Repeat)
# =============================================================================

class InnerLoopNetwork(nn.Module):
    """
    Inner loop at inference:
    1. Encode input to state
    2. State predicts input (reconstruction)
    3. Prediction error updates state
    4. Repeat
    5. State produces output

    Ground truth at inference = the input itself
    """

    def __init__(self, vocab_size, hidden_dim=64, n_layers=2, n_inner_steps=4,
                 inner_lr=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_inner_steps = n_inner_steps
        self.inner_lr = inner_lr

        # Encoder: input → state
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.pos_enc = nn.Parameter(torch.randn(1, 512, hidden_dim) * 0.02)

        self.encoder = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim, nhead=4,
                dim_feedforward=hidden_dim*4,
                batch_first=True, dropout=0.1
            ) for _ in range(n_layers)
        ])

        # Predictor: state → predicted input (reconstruction)
        self.input_predictor = nn.Linear(hidden_dim, vocab_size)

        # Output head: state → task output
        self.ln = nn.LayerNorm(hidden_dim)
        self.output = nn.Linear(hidden_dim, vocab_size)

        self.name = f"Inner Loop (steps={n_inner_steps}, lr={inner_lr})"

    def forward(self, x, return_errors=False):
        B, T = x.shape

        # Initial encoding
        h = self.embed(x) + self.pos_enc[:, :T, :]
        for layer in self.encoder:
            h = layer(h)

        # Inner loop: predict input → error → update state
        errors = []
        for step in range(self.n_inner_steps):
            # State predicts input
            pred_logits = self.input_predictor(h)  # (B, T, vocab)

            # Prediction error (cross-entropy with actual input)
            pred_error = F.cross_entropy(
                pred_logits.reshape(-1, pred_logits.size(-1)),
                x.reshape(-1),
                reduction='none'
            ).reshape(B, T)

            errors.append(pred_error.mean().item())

            # Gradient of error w.r.t. state (not weights!)
            # This is the key: update activations, not weights
            h_for_grad = h.detach().requires_grad_(True)
            pred_logits_grad = self.input_predictor(h_for_grad)
            error_scalar = F.cross_entropy(
                pred_logits_grad.reshape(-1, pred_logits_grad.size(-1)),
                x.reshape(-1)
            )

            # Compute gradient w.r.t. state
            grad_h = torch.autograd.grad(error_scalar, h_for_grad)[0]

            # Update state to reduce prediction error
            h = h - self.inner_lr * grad_h

        # Final output from refined state
        logits = self.output(self.ln(h))

        if return_errors:
            return logits, errors
        return logits


# =============================================================================
# INNER LOOP V2: Simpler, no gradient (Hopfield-like settling)
# =============================================================================

class InnerLoopNetworkV2(nn.Module):
    """
    Alternative: State settles based on consistency, not gradient.

    The state "reconstructs" the input embedding.
    Update: move state toward reconstruction target.
    """

    def __init__(self, vocab_size, hidden_dim=64, n_layers=2, n_inner_steps=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_inner_steps = n_inner_steps

        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.pos_enc = nn.Parameter(torch.randn(1, 512, hidden_dim) * 0.02)

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

        self.name = f"Inner Loop V2 (settling, steps={n_inner_steps})"

    def forward(self, x):
        B, T = x.shape

        # Input embedding (this is ground truth)
        input_embed = self.embed(x) + self.pos_enc[:, :T, :]

        # Initial state
        h = input_embed.clone()
        for layer in self.encoder:
            h = layer(h)

        # Inner loop: settle toward consistency
        for step in range(self.n_inner_steps):
            # State predicts what input embedding should be
            pred_embed = self.embed_predictor(h)

            # Error: difference from actual input embedding
            error = pred_embed - input_embed

            # Update: move state to reduce error
            # (This is like Hopfield dynamics - move toward attractor)
            h = h - 0.1 * error

            # Re-process through encoder (optional, helps integration)
            for layer in self.encoder:
                h = layer(h)

        return self.output(self.ln(h))


# =============================================================================
# DATA: Sequence task that might benefit from iterative refinement
# =============================================================================

def generate_sorting_data(n_samples, seq_len=8, vocab_size=14):
    """
    Task: Sort a sequence of numbers.
    Input: [3, 1, 4, 1, 5, |]
    Output: [1, 1, 3, 4, 5]

    This requires iterative comparison - can't be done in one parallel pass.
    """
    PAD, BOS, EOS, SEP = 0, 1, 2, 3
    NUM_OFFSET = 4
    max_num = vocab_size - NUM_OFFSET

    data = []
    for _ in range(n_samples):
        nums = [random.randint(0, max_num-1) for _ in range(seq_len)]
        sorted_nums = sorted(nums)

        seq = [BOS]
        seq.extend([n + NUM_OFFSET for n in nums])
        seq.append(SEP)
        seq.extend([n + NUM_OFFSET for n in sorted_nums])
        seq.append(EOS)

        data.append(seq)

    return data


def generate_reversal_data(n_samples, min_len=3, max_len=8):
    """Same reversal task as before for comparison."""
    PAD, BOS, EOS, SEP = 0, 1, 2, 3
    DIGIT_OFFSET = 4

    data = []
    for _ in range(n_samples):
        length = random.randint(min_len, max_len)
        digits = [random.randint(0, 9) for _ in range(length)]

        seq = [BOS]
        seq.extend([d + DIGIT_OFFSET for d in digits])
        seq.append(SEP)
        seq.extend([d + DIGIT_OFFSET for d in reversed(digits)])
        seq.append(EOS)

        data.append(seq)

    return data


def pad_sequences(seqs, pad_id=0):
    max_len = max(len(s) for s in seqs)
    return [s + [pad_id] * (max_len - len(s)) for s in seqs]


# =============================================================================
# TRAINING & EVALUATION
# =============================================================================

def train_model(model, train_data, n_epochs=100, lr=1e-3, device='cpu'):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(n_epochs):
        model.train()
        random.shuffle(train_data)

        batch = pad_sequences(train_data)
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
    SEP = 3

    # Don't use no_grad for inner loop models (they need gradients internally)
    batch = pad_sequences(data)
    x = torch.tensor(batch, device=device)

    inputs = x[:, :-1]
    targets = x[:, 1:]

    with torch.set_grad_enabled(isinstance(model, InnerLoopNetwork)):
        logits = model(inputs)

    preds = logits.argmax(dim=-1)

    # Accuracy on output part (after SEP)
    correct = 0
    total = 0

    for i in range(len(data)):
        seq = data[i]
        sep_pos = seq.index(SEP)
        start = sep_pos
        end = len(seq) - 1
        if start < end:
            pred_part = preds[i, start:end]
            target_part = targets[i, start:end]
            correct += (pred_part == target_part).sum().item()
            total += (end - start)

    return correct / total if total > 0 else 0


def count_params(model):
    return sum(p.numel() for p in model.parameters())


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

print("\n1. Generating data...")

# Test on sorting (requires iterative comparison)
train_sort = generate_sorting_data(500, seq_len=6)
test_sort = generate_sorting_data(100, seq_len=6)
test_sort_long = generate_sorting_data(50, seq_len=10)  # OOD

# Test on reversal (for comparison)
train_rev = generate_reversal_data(500, min_len=3, max_len=6)
test_rev = generate_reversal_data(100, min_len=3, max_len=6)

print(f"   Sorting: train={len(train_sort)}, test={len(test_sort)}, test_long={len(test_sort_long)}")
print(f"   Reversal: train={len(train_rev)}, test={len(test_rev)}")

if torch.backends.mps.is_available():
    device = torch.device('mps')
elif torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print(f"   Device: {device}")

VOCAB_SIZE = 14
HIDDEN_DIM = 64

print("\n2. Creating models...")

models = {
    "Standard": StandardNetwork(VOCAB_SIZE, HIDDEN_DIM, n_layers=4),
    "Inner Loop (grad)": InnerLoopNetwork(VOCAB_SIZE, HIDDEN_DIM, n_layers=2, n_inner_steps=4),
    "Inner Loop V2 (settle)": InnerLoopNetworkV2(VOCAB_SIZE, HIDDEN_DIM, n_layers=2, n_inner_steps=4),
}

for name, model in models.items():
    model = model.to(device)
    print(f"   {name}: {count_params(model):,} params")


print("\n" + "=" * 80)
print("EXPERIMENT 1: SORTING TASK")
print("=" * 80)

print("\n3. Training on sorting...")

sort_results = {}
for name, model in models.items():
    print(f"\n   Training {model.name}...")
    model = train_model(model, train_sort, n_epochs=200, device=device)

    test_acc = evaluate(model, test_sort, device)
    long_acc = evaluate(model, test_sort_long, device)

    sort_results[name] = {'test': test_acc, 'long': long_acc}
    print(f"   → Test: {test_acc:.1%}, Long (OOD): {long_acc:.1%}")

print("\n" + "-" * 60)
print("SORTING RESULTS:")
print(f"{'Model':<30} {'Test':>10} {'Long (OOD)':>12}")
print("-" * 60)
for name, r in sort_results.items():
    print(f"{name:<30} {r['test']:>9.1%} {r['long']:>11.1%}")


print("\n" + "=" * 80)
print("EXPERIMENT 2: REVERSAL TASK (for comparison)")
print("=" * 80)

# Reset models
models = {
    "Standard": StandardNetwork(VOCAB_SIZE, HIDDEN_DIM, n_layers=4),
    "Inner Loop (grad)": InnerLoopNetwork(VOCAB_SIZE, HIDDEN_DIM, n_layers=2, n_inner_steps=4),
    "Inner Loop V2 (settle)": InnerLoopNetworkV2(VOCAB_SIZE, HIDDEN_DIM, n_layers=2, n_inner_steps=4),
}

for name, model in models.items():
    model = model.to(device)

print("\n3. Training on reversal...")

rev_results = {}
for name, model in models.items():
    print(f"\n   Training {model.name}...")
    model = train_model(model, train_rev, n_epochs=200, device=device)

    test_acc = evaluate(model, test_rev, device)

    rev_results[name] = {'test': test_acc}
    print(f"   → Test: {test_acc:.1%}")

print("\n" + "-" * 60)
print("REVERSAL RESULTS:")
print(f"{'Model':<30} {'Test':>10}")
print("-" * 60)
for name, r in rev_results.items():
    print(f"{name:<30} {r['test']:>9.1%}")


print("\n" + "=" * 80)
print("ANALYSIS: Does the inner loop help?")
print("=" * 80)

print(f"""
SORTING (requires iterative comparison):
  Standard:           {sort_results['Standard']['test']:.1%}
  Inner Loop (grad):  {sort_results['Inner Loop (grad)']['test']:.1%}
  Inner Loop V2:      {sort_results['Inner Loop V2 (settle)']['test']:.1%}

REVERSAL (single attention pattern):
  Standard:           {rev_results['Standard']['test']:.1%}
  Inner Loop (grad):  {rev_results['Inner Loop (grad)']['test']:.1%}
  Inner Loop V2:      {rev_results['Inner Loop V2 (settle)']['test']:.1%}
""")

# Check if inner loop helps on sorting more than reversal
sort_improvement = sort_results['Inner Loop (grad)']['test'] - sort_results['Standard']['test']
rev_improvement = rev_results['Inner Loop (grad)']['test'] - rev_results['Standard']['test']

print(f"Improvement from inner loop:")
print(f"  Sorting:  {sort_improvement:+.1%}")
print(f"  Reversal: {rev_improvement:+.1%}")

if sort_improvement > rev_improvement + 0.05:
    print("\n→ Inner loop helps MORE on iterative task (sorting)")
    print("  This supports the hypothesis!")
elif sort_improvement > 0:
    print("\n→ Inner loop helps, but not specifically for iterative tasks")
else:
    print("\n→ Inner loop doesn't help")
    print("  Hypothesis not supported (or wrong task/implementation)")


print("\n" + "=" * 80)
print("EXAMINING INNER LOOP DYNAMICS")
print("=" * 80)

# Look at how prediction error evolves over inner steps
model = InnerLoopNetwork(VOCAB_SIZE, HIDDEN_DIM, n_layers=2, n_inner_steps=8).to(device)
model = train_model(model, train_sort, n_epochs=200, device=device)

model.eval()
# Need gradients enabled for inner loop
with torch.set_grad_enabled(True):
    batch = pad_sequences(test_sort[:10])
    x = torch.tensor(batch, device=device)
    inputs = x[:, :-1]

    _, errors = model(inputs, return_errors=True)

print("\nPrediction error over inner steps:")
for i, err in enumerate(errors):
    bar = "█" * int(err * 10)
    print(f"  Step {i}: {err:.4f} {bar}")

if errors[-1] < errors[0]:
    print(f"\n→ Error decreased: {errors[0]:.4f} → {errors[-1]:.4f}")
    print("  The inner loop IS reducing prediction error on input")
else:
    print(f"\n→ Error did not decrease")
    print("  Inner loop not working as intended")

print("\n" + "=" * 80)
