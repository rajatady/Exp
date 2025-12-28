"""
Real CTM Implementation - Based on the actual paper

What makes REAL CTM different from our "fake" CTM:

1. NEURON-LEVEL MODELS (NLMs):
   - Each neuron d has its OWN private MLP θ_d
   - Processes a history of M pre-activations for that neuron
   - z^d_{t+1} = g_θ_d(A^d_t) where A^d_t is M-dimensional time series

2. SYNCHRONIZATION AS REPRESENTATION:
   - Post-activation history: Z_t = [z^1, z^2, ..., z^t] ∈ R^{D×t}
   - Sync matrix: S_t = Z_t · Z_t^T ∈ R^{D×D}
   - This IS the representation, not an add-on

3. OUTPUT FROM SYNC:
   - y_t = W_out · subsample(S_t)
   - Queries: q_t = W_in · subsample(S_t)

4. SYNAPSE MODEL:
   - Recurrent MLP that produces pre-activations
   - a_t = f_θ_syn(concat(z_t, o_t))

Task: Sequence Reversal (same as test_reversal.py for comparison)
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
print("REAL CTM vs FAKE CTM vs Standard Transformer")
print("=" * 80)


# =============================================================================
# SHARED COMPONENTS
# =============================================================================

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, hidden_dim, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, hidden_dim)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, hidden_dim, 2).float() *
                            (-math.log(10000.0) / hidden_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


# =============================================================================
# REAL CTM - Implementing the actual architecture
# =============================================================================

class NeuronLevelModel(nn.Module):
    """
    Per-neuron private MLP.
    Each neuron processes its own M-dimensional history of pre-activations.
    """
    def __init__(self, history_len, hidden_dim=32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(history_len, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        # x: (batch, seq, history_len) for this neuron
        return self.mlp(x).squeeze(-1)  # (batch, seq)


class RealCTM(nn.Module):
    """
    Real CTM implementation following the paper:
    - Neuron-level models (NLMs) with private weights
    - Synchronization as representation (S = Z · Z^T)
    - Output from sync projection
    """
    def __init__(self, vocab_size, hidden_dim=32, n_ticks=4,
                 pre_history_len=4, n_sync_pairs=32):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_ticks = n_ticks
        self.pre_history_len = pre_history_len  # M in paper
        self.n_sync_pairs = n_sync_pairs  # How many (i,j) pairs to sample

        # Input embedding
        self.token_embed = nn.Embedding(vocab_size, hidden_dim)
        self.pos_encode = SinusoidalPositionalEncoding(hidden_dim)

        # Synapse model: produces pre-activations from (z_t, input)
        # In paper: a_t = f_θ_syn(concat(z_t, o_t))
        self.synapse = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )

        # Neuron-level models: one per neuron (dimension)
        # Each NLM processes M pre-activations and outputs 1 post-activation
        self.nlms = nn.ModuleList([
            NeuronLevelModel(pre_history_len, hidden_dim=16)
            for _ in range(hidden_dim)
        ])

        # Random sync pair indices (fixed at init for consistency)
        # Sample n_sync_pairs (i,j) pairs from DxD
        all_pairs = [(i, j) for i in range(hidden_dim) for j in range(hidden_dim)]
        self.sync_pairs_out = random.sample(all_pairs, min(n_sync_pairs, len(all_pairs)))
        self.sync_pairs_action = random.sample(all_pairs, min(n_sync_pairs, len(all_pairs)))

        # Output projection: from sync to logits
        # y_t = W_out · S_out
        self.output_proj = nn.Linear(n_sync_pairs, vocab_size)

        # Action projection: from sync to attention query
        # q_t = W_in · S_action
        self.action_proj = nn.Linear(n_sync_pairs, hidden_dim)

        # Simple attention for input modulation
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)

        self.name = "Real CTM"

    def compute_sync_features(self, Z, pairs):
        """
        Compute synchronization features by subsampling S = Z · Z^T

        Z: (batch, seq, n_ticks_so_far, hidden_dim)
        pairs: list of (i, j) pairs

        Returns: (batch, seq, n_pairs)
        """
        B, T, num_ticks, D = Z.shape

        # S = Z · Z^T would be (B, T, D, D)
        # But we only need specific (i,j) pairs
        sync_features = []
        for (i, j) in pairs:
            # Z[:, :, :, i] shape: (B, T, num_ticks)
            # Z[:, :, :, j] shape: (B, T, num_ticks)
            # Their dot product over time dimension
            zi = Z[:, :, :, i]  # (B, T, num_ticks)
            zj = Z[:, :, :, j]  # (B, T, num_ticks)
            sync_ij = (zi * zj).sum(dim=-1)  # (B, T)
            sync_features.append(sync_ij)

        return torch.stack(sync_features, dim=-1)  # (B, T, n_pairs)

    def forward(self, x):
        B, T = x.shape

        # Initial embedding
        h = self.pos_encode(self.token_embed(x))  # (B, T, D)

        # Initialize post-activation z_0 = h
        z = h.clone()

        # History of pre-activations A: (B, T, M, D) - most recent M
        pre_history = torch.zeros(B, T, self.pre_history_len, self.hidden_dim,
                                   device=x.device)

        # History of post-activations Z: will grow each tick
        post_history = [z.clone()]  # List of (B, T, D) tensors

        for tick in range(self.n_ticks):
            # 1. Synapse model: produce pre-activations
            # a_t = f_θ_syn(concat(z_t, o_t))
            # Here o_t is the input (we use h as standing input)
            synapse_input = torch.cat([z, h], dim=-1)  # (B, T, 2D)
            a = self.synapse(synapse_input)  # (B, T, D) pre-activations

            # 2. Update pre-activation history (FIFO)
            # Shift and add new
            pre_history = torch.cat([
                pre_history[:, :, 1:, :],
                a.unsqueeze(2)
            ], dim=2)  # (B, T, M, D)

            # 3. Neuron-level models: each neuron processes its history
            # z^d_{t+1} = g_θ_d(A^d_t)
            z_new_parts = []
            for d, nlm in enumerate(self.nlms):
                # History for neuron d: (B, T, M)
                neuron_history = pre_history[:, :, :, d]
                # NLM output: (B, T)
                z_d = nlm(neuron_history)
                z_new_parts.append(z_d)

            z_new = torch.stack(z_new_parts, dim=-1)  # (B, T, D)

            # 4. Store post-activation
            post_history.append(z_new)

            # 5. Compute synchronization from post-activation history
            # Z_t = [z^1, ..., z^t], S_t = Z_t · Z_t^T
            Z = torch.stack(post_history, dim=2)  # (B, T, t+1, D)

            # 6. Get sync features for output and action
            sync_out = self.compute_sync_features(Z, self.sync_pairs_out)  # (B, T, n_pairs)
            sync_action = self.compute_sync_features(Z, self.sync_pairs_action)  # (B, T, n_pairs)

            # 7. Action: use sync to modulate input via attention
            # q_t = W_in · S_action
            q = self.action_proj(sync_action)  # (B, T, D)
            k = self.k_proj(h)  # (B, T, D)
            v = self.v_proj(h)  # (B, T, D)

            # Causal attention
            attn_scores = torch.bmm(q, k.transpose(-2, -1)) / math.sqrt(self.hidden_dim)
            causal_mask = torch.tril(torch.ones(T, T, device=x.device)).bool()
            attn_scores = attn_scores.masked_fill(~causal_mask, float('-inf'))
            attn = F.softmax(attn_scores, dim=-1)
            attn_out = torch.bmm(attn, v)  # (B, T, D)

            # 8. Update z with attention output (as in paper)
            z = z_new + attn_out

        # Final output: y_t = W_out · S_out
        logits = self.output_proj(sync_out)  # (B, T, vocab_size)

        return logits


# =============================================================================
# FAKE CTM (what we had before)
# =============================================================================

class FakeCTM(nn.Module):
    """
    Our previous "CTM" - just looped transformer with history as add-on
    """
    def __init__(self, vocab_size, hidden_dim=32, n_layers=2, n_heads=4,
                 n_ticks=4, history_len=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_ticks = n_ticks
        self.history_len = history_len

        self.token_embed = nn.Embedding(vocab_size, hidden_dim)
        self.pos_encode = SinusoidalPositionalEncoding(hidden_dim)

        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim, nhead=n_heads,
                dim_feedforward=hidden_dim * 4, batch_first=True
            ) for _ in range(n_layers)
        ])

        # History as 0.1 add-on (NOT as representation)
        self.history_processor = nn.Sequential(
            nn.Linear(hidden_dim * history_len, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.ln_final = nn.LayerNorm(hidden_dim)
        self.output = nn.Linear(hidden_dim, vocab_size)
        self.name = "Fake CTM (our old impl)"

    def forward(self, x):
        B, T = x.shape
        h = self.pos_encode(self.token_embed(x))

        history = [h.clone()]

        for tick in range(self.n_ticks):
            # Standard transformer blocks
            for block in self.blocks:
                h = block(h)

            # History as add-on
            if len(history) >= self.history_len:
                recent = history[-self.history_len:]
            else:
                padding = [history[0]] * (self.history_len - len(history))
                recent = padding + history

            hist_concat = torch.cat(recent, dim=-1)
            hist_features = self.history_processor(hist_concat)
            h = h + 0.1 * hist_features  # THE PROBLEM: just 0.1 add-on

            history.append(h.clone())

        # Output from hidden state (NOT from sync)
        logits = self.output(self.ln_final(h))
        return logits


# =============================================================================
# STANDARD TRANSFORMER (baseline)
# =============================================================================

class StandardTransformer(nn.Module):
    def __init__(self, vocab_size, hidden_dim=32, n_layers=4, n_heads=4):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, hidden_dim)
        self.pos_encode = SinusoidalPositionalEncoding(hidden_dim)

        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim, nhead=n_heads,
                dim_feedforward=hidden_dim * 4, batch_first=True
            ) for _ in range(n_layers)
        ])

        self.ln_final = nn.LayerNorm(hidden_dim)
        self.output = nn.Linear(hidden_dim, vocab_size)
        self.name = "Standard Transformer"

    def forward(self, x):
        h = self.pos_encode(self.token_embed(x))
        for block in self.blocks:
            h = block(h)
        return self.output(self.ln_final(h))


# =============================================================================
# SIMPLE ACCUMULATOR (what actually worked)
# =============================================================================

class SimpleAccumulator(nn.Module):
    """What we found actually works: just accumulation h = h + h_new"""
    def __init__(self, vocab_size, hidden_dim=32, n_layers=2, n_heads=4, n_ticks=4):
        super().__init__()
        self.n_ticks = n_ticks

        self.token_embed = nn.Embedding(vocab_size, hidden_dim)
        self.pos_encode = SinusoidalPositionalEncoding(hidden_dim)

        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim, nhead=n_heads,
                dim_feedforward=hidden_dim * 4, batch_first=True
            ) for _ in range(n_layers)
        ])

        self.ln_final = nn.LayerNorm(hidden_dim)
        self.output = nn.Linear(hidden_dim, vocab_size)
        self.name = "Simple Accumulator"

    def forward(self, x):
        h = self.pos_encode(self.token_embed(x))

        for tick in range(self.n_ticks):
            h_new = h
            for block in self.blocks:
                h_new = block(h_new)
            h = h + h_new  # Just accumulation

        return self.output(self.ln_final(h))


# =============================================================================
# DATA GENERATION
# =============================================================================

def generate_reversal_data(n_samples, min_len=3, max_len=8):
    """Generate sequence reversal examples.
    Format: [BOS] a b c | c b a [EOS]
    Vocab: 0=PAD, 1=BOS, 2=EOS, 3=SEP(|), 4-13=digits 0-9
    """
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
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(n_epochs):
        model.train()
        random.shuffle(train_data)

        # Full batch for simplicity
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
# MAIN EXPERIMENT
# =============================================================================

print("\n1. Generating data...")
train_data = generate_reversal_data(500, min_len=3, max_len=6)
test_data = generate_reversal_data(100, min_len=3, max_len=6)
test_long = generate_reversal_data(50, min_len=7, max_len=10)

print(f"   Train: {len(train_data)}, Test: {len(test_data)}, Test Long: {len(test_long)}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"   Device: {device}")

VOCAB_SIZE = 14
HIDDEN_DIM = 64  # Increased to match original
N_TICKS = 4

print("\n2. Creating models...")
models = {
    "Standard Transformer": StandardTransformer(VOCAB_SIZE, HIDDEN_DIM, n_layers=4),
    "Simple Accumulator": SimpleAccumulator(VOCAB_SIZE, HIDDEN_DIM, n_ticks=N_TICKS),
    "Fake CTM": FakeCTM(VOCAB_SIZE, HIDDEN_DIM, n_ticks=N_TICKS),
    "Real CTM": RealCTM(VOCAB_SIZE, HIDDEN_DIM, n_ticks=N_TICKS, n_sync_pairs=128),
}

for name, model in models.items():
    model = model.to(device)
    params = count_params(model)
    print(f"   {name}: {params:,} params")

print("\n3. Training all models...")
print("-" * 60)

# More epochs for larger models
epochs_map = {
    "Standard Transformer": 200,
    "Simple Accumulator": 200,
    "Fake CTM": 200,
    "Real CTM": 500,  # More epochs for Real CTM
}

results = {}
for name, model in models.items():
    n_epochs = epochs_map[name]
    print(f"\n   Training {name} ({n_epochs} epochs)...")
    model = train_model(model, train_data, n_epochs=n_epochs, device=device)

    test_acc = evaluate(model, test_data, device)
    long_acc = evaluate(model, test_long, device)

    results[name] = {
        'test': test_acc,
        'long': long_acc,
        'params': count_params(model)
    }
    print(f"   → Test: {test_acc:.1%}, Long (OOD): {long_acc:.1%}")

# =============================================================================
# RESULTS
# =============================================================================

print("\n" + "=" * 80)
print("RESULTS: Reversal Task")
print("=" * 80)

print(f"\n{'Model':<25} {'Test Acc':>10} {'Long (OOD)':>12} {'Params':>10}")
print("-" * 60)

for name in ["Standard Transformer", "Simple Accumulator", "Fake CTM", "Real CTM"]:
    r = results[name]
    print(f"{name:<25} {r['test']:>9.1%} {r['long']:>11.1%} {r['params']:>10,}")

print("\n" + "=" * 80)
print("KEY COMPARISONS")
print("=" * 80)

fake = results["Fake CTM"]['test']
real = results["Real CTM"]['test']
simple = results["Simple Accumulator"]['test']
standard = results["Standard Transformer"]['test']

print(f"""
1. Does REAL CTM beat FAKE CTM?
   Real CTM:  {real:.1%}
   Fake CTM:  {fake:.1%}
   Gap: {real - fake:+.1%}

2. Does REAL CTM beat Simple Accumulator?
   Real CTM:          {real:.1%}
   Simple Accumulator: {simple:.1%}
   Gap: {real - simple:+.1%}

3. OOD Generalization (longer sequences):
   Real CTM:  {results["Real CTM"]['long']:.1%}
   Fake CTM:  {results["Fake CTM"]['long']:.1%}
   Standard:  {results["Standard Transformer"]['long']:.1%}
""")

if real > fake + 0.05:
    print("→ REAL CTM architecture provides SIGNIFICANT benefit over fake")
elif real > fake:
    print("→ Real CTM slightly better, but sync-as-representation helps")
else:
    print("→ Sync-as-representation doesn't help on this task")
    print("   Maybe reversal is too simple, or we need different task")

print("\n" + "=" * 80)
