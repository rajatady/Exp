"""
BDH-Style Architecture vs Our Previous Attempts

Based on the Dragon Hatchling paper, the key differences:
1. Linear attention (no softmax) - state accumulates over sequence
2. High-dimensional sparse representations (ReLU)
3. Element-wise product x * y (Hebbian-like)
4. No explicit tick loops

This tests whether the BDH architectural choices help on our tasks.
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
print("BDH-Style vs Standard Transformer vs Our CTM")
print("=" * 80)


# =============================================================================
# BDH-STYLE ARCHITECTURE
# =============================================================================

class LinearAttention(nn.Module):
    """
    Linear attention (no softmax) - allows state to accumulate.
    Based on original BDH implementation.
    """
    def __init__(self, n_heads, N, theta=2**16):
        super().__init__()
        self.n_heads = n_heads
        self.N = N
        # Precompute frequencies for RoPE
        freqs = 1.0 / (theta ** (torch.arange(0, N, 1).float() / N)) / (2 * math.pi)
        self.register_buffer('freqs', freqs.view(1, 1, 1, N))

    def rope(self, phases, v):
        """Apply rotary position embedding"""
        phases = (phases % 1) * (2 * math.pi)
        cos = torch.cos(phases)
        sin = torch.sin(phases)
        v_rot = torch.stack((-v[..., 1::2], v[..., ::2]), dim=-1).view(*v.size())
        return (v * cos + v_rot * sin).to(v.dtype)

    def forward(self, Q, K, V):
        B, nh, T, N = Q.shape

        # Position phases for RoPE
        r_phases = torch.arange(T, device=Q.device, dtype=self.freqs.dtype).view(1, 1, -1, 1) * self.freqs

        # Apply RoPE
        QR = self.rope(r_phases, Q)
        KR = self.rope(r_phases, K)  # K = Q in BDH

        # Linear attention with STRICT causal mask (diagonal=-1, excludes self)
        scores = torch.matmul(QR, KR.transpose(-2, -1))  # (B, nh, T, T)
        scores = torch.tril(scores, diagonal=-1)  # Strictly lower triangular

        # No softmax - this is linear attention
        return torch.matmul(scores, V)


class BDHBlock(nn.Module):
    """
    BDH-style block (fixed to match original):
    1. Encode to high dimension
    2. ReLU for sparsity
    3. Linear attention
    4. Element-wise product (Hebbian)
    5. Decode back
    """
    def __init__(self, d_model, n_heads=4, expansion=32):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.N = d_model * expansion // n_heads  # High dimension per head

        # Encoder: project to high dimension (B, nh, T, D) -> (B, nh, T, N)
        self.encoder = nn.Parameter(torch.randn(n_heads, d_model, self.N) * 0.02)
        self.encoder_v = nn.Parameter(torch.randn(n_heads, d_model, self.N) * 0.02)

        # Decoder: project back to d_model
        self.decoder = nn.Parameter(torch.randn(n_heads * self.N, d_model) * 0.02)

        # Linear attention
        self.attn = LinearAttention(n_heads, self.N)

        # Layer norm
        self.ln = nn.LayerNorm(d_model, elementwise_affine=False)

    def forward(self, x):
        B, T, D = x.shape
        nh = self.n_heads
        N = self.N

        # x: (B, T, D) -> (B, 1, T, D) for broadcasting with heads
        x_in = x.unsqueeze(1)  # (B, 1, T, D)

        # Encode to high dimension: (B, nh, T, N)
        x_latent = torch.matmul(x_in, self.encoder)  # (B, nh, T, N)

        # ReLU for sparsity - key for biological plausibility
        x_sparse = F.relu(x_latent)  # (B, nh, T, N)

        # Linear attention - Q=K=sparse, V=original x (broadcast across heads)
        # V needs to be (B, nh, T, D) - broadcast x_in
        V = x_in.expand(B, nh, T, D)  # (B, nh, T, D)
        yKV = self.attn(Q=x_sparse, K=x_sparse, V=V)  # (B, nh, T, D)
        yKV = self.ln(yKV)

        # Encode attention output to high dimension
        y_latent = torch.matmul(yKV, self.encoder_v)  # (B, nh, T, N)
        y_sparse = F.relu(y_latent)

        # HEBBIAN: element-wise product
        # "neurons that fire together, wire together"
        xy_sparse = x_sparse * y_sparse  # (B, nh, T, N)

        # Decode back to d_model
        xy_flat = xy_sparse.transpose(1, 2).reshape(B, T, nh * N)  # (B, T, nh*N)
        y = torch.matmul(xy_flat, self.decoder)  # (B, T, D)
        y = self.ln(y)

        # Residual
        return self.ln(x + y)


class BDHModel(nn.Module):
    """
    BDH-style language model.
    """
    def __init__(self, vocab_size, d_model=64, n_layers=4, n_heads=4, expansion=32):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.ln_in = nn.LayerNorm(d_model, elementwise_affine=False)

        self.blocks = nn.ModuleList([
            BDHBlock(d_model, n_heads, expansion)
            for _ in range(n_layers)
        ])

        self.ln_out = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

        self.name = "BDH-Style"

    def forward(self, x):
        h = self.ln_in(self.embed(x))

        for block in self.blocks:
            h = block(h)

        return self.head(self.ln_out(h))


# =============================================================================
# STANDARD TRANSFORMER (for comparison)
# =============================================================================

class StandardTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=64, n_layers=4, n_heads=4):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 4, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

        self.name = "Standard Transformer"

    def forward(self, x):
        h = self.embed(x)

        # Causal mask
        T = x.size(1)
        mask = torch.triu(torch.ones(T, T, device=x.device) * float('-inf'), diagonal=1)

        h = self.transformer(h, mask=mask)
        return self.head(self.ln(h))


# =============================================================================
# OUR PREVIOUS CTM (tick-based)
# =============================================================================

class TickCTM(nn.Module):
    """Our previous approach: explicit ticks with accumulation"""
    def __init__(self, vocab_size, d_model=64, n_layers=2, n_heads=4, n_ticks=2):
        super().__init__()
        self.n_ticks = n_ticks

        self.embed = nn.Embedding(vocab_size, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 4, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

        self.name = "Tick-CTM (our previous)"

    def forward(self, x):
        h = self.embed(x)

        T = x.size(1)
        mask = torch.triu(torch.ones(T, T, device=x.device) * float('-inf'), diagonal=1)

        # Our approach: iterate with accumulation
        for tick in range(self.n_ticks):
            h_new = self.transformer(h, mask=mask)
            h = h + h_new  # Accumulation

        return self.head(self.ln(h))


# =============================================================================
# DATA: Sequence Reversal
# =============================================================================

def generate_reversal_data(n_samples, min_len=3, max_len=8):
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
# TRAINING
# =============================================================================

def train_model(model, train_data, n_epochs=100, lr=1e-3, device='cpu'):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

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

    with torch.no_grad():
        batch = pad_sequences(data)
        x = torch.tensor(batch, device=device)

        inputs = x[:, :-1]
        targets = x[:, 1:]

        logits = model(inputs)
        preds = logits.argmax(dim=-1)

        # Reversal-only accuracy
        reversal_correct = 0
        reversal_tokens = 0

        for i in range(len(data)):
            seq = data[i]
            sep_pos = seq.index(SEP)
            start = sep_pos
            end = len(seq) - 1
            if start < end:
                rev_preds = preds[i, start:end]
                rev_targets = targets[i, start:end]
                reversal_correct += (rev_preds == rev_targets).sum().item()
                reversal_tokens += (end - start)

        return reversal_correct / reversal_tokens if reversal_tokens > 0 else 0


def count_params(model):
    return sum(p.numel() for p in model.parameters())


# =============================================================================
# MAIN
# =============================================================================

print("\n1. Generating data...")
train_data = generate_reversal_data(500, min_len=3, max_len=6)
test_data = generate_reversal_data(100, min_len=3, max_len=6)
test_long = generate_reversal_data(50, min_len=7, max_len=10)  # OOD

print(f"   Train: {len(train_data)}, Test: {len(test_data)}, Test Long (OOD): {len(test_long)}")

# Device selection with MPS support
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')
print(f"   Device: {device}")

VOCAB_SIZE = 14
D_MODEL = 64

print("\n2. Creating models...")
models = {
    "Standard Transformer": StandardTransformer(VOCAB_SIZE, D_MODEL, n_layers=4),
    "Tick-CTM (1L x 8T)": TickCTM(VOCAB_SIZE, D_MODEL, n_layers=2, n_ticks=8),  # 2*8=16 passes
    "BDH-Style": BDHModel(VOCAB_SIZE, D_MODEL, n_layers=4, expansion=4),
}

for name, model in models.items():
    model = model.to(device)
    params = count_params(model)
    print(f"   {name}: {params:,} params")

print("\n3. Training...")
print("-" * 60)

results = {}
for name, model in models.items():
    print(f"\n   Training {name}...")
    try:
        model = train_model(model, train_data, n_epochs=150, device=device)

        test_acc = evaluate(model, test_data, device)
        long_acc = evaluate(model, test_long, device)

        results[name] = {
            'test': test_acc,
            'long': long_acc,
            'params': count_params(model)
        }
        print(f"   -> Test: {test_acc:.1%}, Long (OOD): {long_acc:.1%}")
    except Exception as e:
        print(f"   -> FAILED: {e}")
        results[name] = {'test': 0, 'long': 0, 'params': count_params(model)}

# =============================================================================
# RESULTS
# =============================================================================

print("\n" + "=" * 80)
print("RESULTS: Reversal Task")
print("=" * 80)

print(f"\n{'Model':<30} {'Test':>10} {'Long (OOD)':>12} {'Params':>12}")
print("-" * 65)

for name, r in results.items():
    print(f"{name:<30} {r['test']:>9.1%} {r['long']:>11.1%} {r['params']:>12,}")

print("\n" + "=" * 80)
print("ANALYSIS")
print("=" * 80)

if all(r['test'] > 0 for r in results.values()):
    bdh = results["BDH-Style"]['test']
    std = results["Standard Transformer"]['test']
    tick = results["Tick-CTM (1L x 8T)"]['test']

    bdh_ood = results["BDH-Style"]['long']
    std_ood = results["Standard Transformer"]['long']
    tick_ood = results["Tick-CTM (1L x 8T)"]['long']

    print(f"""
Configuration:
- Standard: 4 layers (4 passes)
- Tick-CTM: 1 layer * 8 ticks = 8 passes (SMALLEST model, MOST iterations)
- BDH: 4 layers

Test Accuracy:
- Standard:  {std:.1%}
- Tick-CTM:  {tick:.1%}
- BDH-Style: {bdh:.1%}

OOD Generalization (longer sequences):
- Standard:  {std_ood:.1%}
- Tick-CTM:  {tick_ood:.1%}
- BDH-Style: {bdh_ood:.1%}

BDH vs Standard: {bdh - std:+.1%} (test), {bdh_ood - std_ood:+.1%} (OOD)
Tick vs Standard: {tick - std:+.1%} (test), {tick_ood - std_ood:+.1%} (OOD)
""")

print("=" * 80)
