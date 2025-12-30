"""
Proper CTM Components - Matching Real Implementation

Based on: /continuous-thought-machines/models/ctm.py and modules.py

Key differences from our previous attempts:
1. SuperLinear (NLM) uses einsum - NO PYTHON LOOPS
2. State trace is rolling buffer (B, T, D, M), not list
3. Sync uses exponential decay, not just dot product
4. All operations are fully vectorized

Adapted for transformer context:
- Real CTM: (B, D) neurons with M history
- Our CTM: (B, T, D) sequence positions, each with D neurons, M history per neuron
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import math
from dataclasses import dataclass
from typing import Tuple, Optional

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

print("=" * 80)
print("PROPER CTM IMPLEMENTATION")
print("=" * 80)


# =============================================================================
# PROPER CTM COMPONENTS
# =============================================================================

class SuperLinear(nn.Module):
    """
    Proper NLM implementation using einsum - NO PYTHON LOOPS.

    Applies N independent linear transformations to N neurons in parallel.

    Input: (B, T, D, M) where D=neurons, M=history length
    Output: (B, T, D) or (B, T, D, H) depending on out_dims

    From real CTM: torch.einsum('BDM,MHD->BDH', x, self.w1) + self.b1
    Adapted for sequence: torch.einsum('BTDM,MHD->BTDH', x, self.w1) + self.b1
    """
    def __init__(self, d_model: int, memory_length: int, hidden_dim: int = None,
                 out_dim: int = 1, dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.memory_length = memory_length
        self.out_dim = out_dim
        hidden_dim = hidden_dim or memory_length * 2

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Per-neuron weights: (M, H, D) - M=memory, H=hidden, D=neurons
        # Each neuron d has its own weight matrix w1[:, :, d]
        self.w1 = nn.Parameter(
            torch.empty(memory_length, hidden_dim, d_model).uniform_(
                -1/math.sqrt(memory_length + hidden_dim),
                1/math.sqrt(memory_length + hidden_dim)
            )
        )
        # Per-neuron bias: (1, 1, D, H)
        self.b1 = nn.Parameter(torch.zeros(1, 1, d_model, hidden_dim))

        # Second layer (if deep NLM)
        self.w2 = nn.Parameter(
            torch.empty(hidden_dim // 2, out_dim, d_model).uniform_(
                -1/math.sqrt(hidden_dim // 2 + out_dim),
                1/math.sqrt(hidden_dim // 2 + out_dim)
            )
        )
        self.b2 = nn.Parameter(torch.zeros(1, 1, d_model, out_dim))

        # Temperature scaling
        self.T = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, D, M) - batch, sequence, neurons, history
        returns: (B, T, D) - one output per neuron
        """
        x = self.dropout(x)

        # First layer: (B, T, D, M) @ (M, H, D) -> (B, T, D, H)
        # einsum applies different weights per neuron d
        h = torch.einsum('btdm,mhd->btdh', x, self.w1) + self.b1

        # GLU activation (halves hidden dim)
        h = F.glu(h, dim=-1)  # (B, T, D, H//2)

        # Second layer: (B, T, D, H//2) @ (H//2, O, D) -> (B, T, D, O)
        out = torch.einsum('btdh,hod->btdo', h, self.w2) + self.b2

        # GLU and squeeze
        out = F.glu(out, dim=-1)  # (B, T, D, O//2)
        out = out.squeeze(-1) / self.T  # (B, T, D)

        return out


class Synapses(nn.Module):
    """
    Synapse model - shares information across neurons.
    Takes concatenated [attention_out, activated_state] and produces pre-activations.

    Real CTM uses UNET for deep synapses, MLP for shallow.
    We use MLP for simplicity.
    """
    def __init__(self, d_input: int, d_model: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_input + d_model, d_model * 2),
            nn.GLU(),
            nn.LayerNorm(d_model)
        )

    def forward(self, attn_out: torch.Tensor, activated_state: torch.Tensor) -> torch.Tensor:
        """
        attn_out: (B, T, d_input)
        activated_state: (B, T, D)
        returns: (B, T, D) pre-activations
        """
        combined = torch.cat([attn_out, activated_state], dim=-1)
        return self.net(combined)


class SyncComputation(nn.Module):
    """
    Proper synchronisation computation with exponential decay.

    Sync between neuron i and j = correlation of their activations over time.
    Uses efficient linear recurrence (no need to store full history).

    decay_alpha = r * decay_alpha + pairwise_product
    decay_beta = r * decay_beta + 1
    sync = decay_alpha / sqrt(decay_beta)
    """
    def __init__(self, d_model: int, n_sync_pairs: int, neuron_select_type: str = 'random-pairing'):
        super().__init__()
        self.d_model = d_model
        self.n_sync_pairs = n_sync_pairs
        self.neuron_select_type = neuron_select_type

        # Learnable decay parameters (one per sync pair)
        self.decay_params = nn.Parameter(torch.zeros(n_sync_pairs))

        # Neuron indices for pairing
        if neuron_select_type == 'random-pairing':
            # Random pairs: n_sync_pairs pairs of neurons
            left_indices = torch.from_numpy(np.random.choice(d_model, size=n_sync_pairs))
            right_indices = torch.from_numpy(np.random.choice(d_model, size=n_sync_pairs))
        elif neuron_select_type == 'first-last':
            # First n neurons paired with themselves
            left_indices = torch.arange(n_sync_pairs)
            right_indices = torch.arange(n_sync_pairs)
        else:
            raise ValueError(f"Unknown neuron_select_type: {neuron_select_type}")

        self.register_buffer('left_indices', left_indices)
        self.register_buffer('right_indices', right_indices)

    def forward(self, activated_state: torch.Tensor,
                decay_alpha: Optional[torch.Tensor] = None,
                decay_beta: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        activated_state: (B, T, D) - current post-activations
        decay_alpha, decay_beta: (B, T, n_sync_pairs) - running accumulators

        returns: (sync, new_decay_alpha, new_decay_beta)
        """
        B, T, D = activated_state.shape

        # Clamp decay params to reasonable range
        decay_params_clamped = torch.clamp(self.decay_params, 0, 15)
        r = torch.exp(-decay_params_clamped)  # (n_sync_pairs,)
        r = r.view(1, 1, -1).expand(B, T, -1)  # (B, T, n_sync_pairs)

        # Get neuron activations for left and right indices
        left = activated_state[:, :, self.left_indices]   # (B, T, n_sync_pairs)
        right = activated_state[:, :, self.right_indices]  # (B, T, n_sync_pairs)

        # Pairwise product
        pairwise_product = left * right  # (B, T, n_sync_pairs)

        # Update decay accumulators
        if decay_alpha is None or decay_beta is None:
            decay_alpha = pairwise_product
            decay_beta = torch.ones_like(pairwise_product)
        else:
            decay_alpha = r * decay_alpha + pairwise_product
            decay_beta = r * decay_beta + 1

        # Compute sync
        sync = decay_alpha / torch.sqrt(decay_beta + 1e-8)

        return sync, decay_alpha, decay_beta


# =============================================================================
# PROPER CTM MODEL
# =============================================================================

@dataclass
class ProperCTMConfig:
    n_ticks: int = 8
    memory_length: int = 8  # How many ticks of history per neuron
    n_sync_pairs: int = 64
    nlm_hidden_dim: int = 32
    use_sync_output: bool = False  # Output from sync vs hidden
    use_sync_residual: bool = True  # Add sync as small residual
    sync_residual_scale: float = 0.1

    def name(self) -> str:
        parts = ["ProperCTM"]
        if self.use_sync_output:
            parts.append("+SyncOut")
        if self.use_sync_residual:
            parts.append("+SyncRes")
        return "".join(parts)


class ProperCTM(nn.Module):
    """
    Proper CTM implementation matching the real architecture.

    Adapted for transformer context:
    - Real CTM: processes single input, neurons have history over ticks
    - Our CTM: processes sequence, each position's neurons have history over ticks

    Key components:
    1. Base transformer for sequence processing
    2. NLM (SuperLinear) - per-neuron history processing
    3. State trace - rolling buffer of pre-activations
    4. Sync computation - correlation between neuron pairs with decay
    """
    def __init__(self, vocab_size: int, d_model: int = 64, n_layers: int = 2,
                 n_heads: int = 4, config: ProperCTMConfig = None):
        super().__init__()
        self.config = config or ProperCTMConfig()
        self.d_model = d_model

        # Embedding
        self.embed = nn.Embedding(vocab_size, d_model)

        # Base transformer (like synapses - shares info across neurons)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 4, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # NLM - processes each neuron's history
        self.nlm = SuperLinear(
            d_model=d_model,
            memory_length=self.config.memory_length,
            hidden_dim=self.config.nlm_hidden_dim,
            out_dim=2  # GLU will halve this to 1
        )

        # Sync computation
        self.sync = SyncComputation(
            d_model=d_model,
            n_sync_pairs=self.config.n_sync_pairs
        )

        # Sync to residual projection
        if self.config.use_sync_residual:
            self.sync_to_residual = nn.Sequential(
                nn.Linear(self.config.n_sync_pairs, d_model),
                nn.Tanh()
            )

        # Start states (learnable initial conditions)
        self.start_activated_state = nn.Parameter(
            torch.zeros(d_model).uniform_(-1/math.sqrt(d_model), 1/math.sqrt(d_model))
        )
        self.start_trace = nn.Parameter(
            torch.zeros(d_model, self.config.memory_length).uniform_(
                -1/math.sqrt(d_model + self.config.memory_length),
                1/math.sqrt(d_model + self.config.memory_length)
            )
        )

        # Output
        self.ln = nn.LayerNorm(d_model)
        if self.config.use_sync_output:
            self.head = nn.Linear(self.config.n_sync_pairs, vocab_size)
        else:
            self.head = nn.Linear(d_model, vocab_size)

        self.name = self.config.name()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        B, T = x.shape
        device = x.device

        # Causal mask for transformer
        mask = torch.triu(torch.ones(T, T, device=device) * float('-inf'), diagonal=1)

        # Initial embedding
        h = self.embed(x)  # (B, T, D)

        # Initialize state trace: (B, T, D, M)
        # Each position has D neurons, each with M history slots
        state_trace = self.start_trace.unsqueeze(0).unsqueeze(0).expand(B, T, -1, -1).clone()

        # Initialize activated state: (B, T, D)
        activated_state = self.start_activated_state.unsqueeze(0).unsqueeze(0).expand(B, T, -1).clone()

        # Initialize sync decay accumulators
        decay_alpha, decay_beta = None, None

        # Info for tracking
        info = {'ticks_used': self.config.n_ticks}

        # === MAIN TICK LOOP ===
        for tick in range(self.config.n_ticks):

            # 1. Transformer processes sequence (like synapses sharing info)
            # Combine embedding with current activated state
            transformer_input = h + activated_state
            pre_activations = self.transformer(transformer_input, mask=mask)  # (B, T, D)

            # 2. Update state trace (rolling buffer)
            # Drop oldest (index 0), append newest (pre_activations)
            state_trace = torch.cat([
                state_trace[:, :, :, 1:],  # Drop oldest
                pre_activations.unsqueeze(-1)  # Append newest
            ], dim=-1)  # (B, T, D, M)

            # 3. Apply NLM to get activated state
            # Each neuron processes its own history
            activated_state = self.nlm(state_trace)  # (B, T, D)

            # 4. Compute sync
            sync, decay_alpha, decay_beta = self.sync(
                activated_state, decay_alpha, decay_beta
            )  # sync: (B, T, n_sync_pairs)

            # 5. Optionally add sync as residual (gentle influence)
            if self.config.use_sync_residual:
                sync_residual = self.sync_to_residual(sync)  # (B, T, D)
                activated_state = activated_state + self.config.sync_residual_scale * sync_residual

        # === OUTPUT ===
        if self.config.use_sync_output:
            logits = self.head(sync)  # Output from sync
        else:
            logits = self.head(self.ln(activated_state))  # Output from hidden

        return logits, info


# =============================================================================
# BASELINE MODELS
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
        T = x.size(1)
        mask = torch.triu(torch.ones(T, T, device=x.device) * float('-inf'), diagonal=1)
        h = self.transformer(h, mask=mask)
        return self.head(self.ln(h)), {}


class TickCTM(nn.Module):
    """Our previous Tick-CTM (looped transformer with accumulation)"""
    def __init__(self, vocab_size, d_model=64, n_layers=2, n_heads=4, n_ticks=8):
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
        self.name = f"Tick-CTM ({n_layers}L x {n_ticks}T)"

    def forward(self, x):
        h = self.embed(x)
        T = x.size(1)
        mask = torch.triu(torch.ones(T, T, device=x.device) * float('-inf'), diagonal=1)
        for _ in range(self.n_ticks):
            h_new = self.transformer(h, mask=mask)
            h = h + h_new
        return self.head(self.ln(h)), {}


# =============================================================================
# DATA
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
# TRAINING & EVALUATION
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
        logits, _ = model(inputs)

        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            ignore_index=0
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if (epoch + 1) % 50 == 0:
            print(f"      Epoch {epoch+1}: loss={loss.item():.4f}")

    return model


def evaluate(model, data, device='cpu'):
    model.eval()
    SEP = 3

    with torch.no_grad():
        batch = pad_sequences(data)
        x = torch.tensor(batch, device=device)

        inputs = x[:, :-1]
        targets = x[:, 1:]

        logits, _ = model(inputs)
        preds = logits.argmax(dim=-1)

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

def main():
    print("\n1. Generating data...")
    train_data = generate_reversal_data(500, min_len=3, max_len=6)
    test_data = generate_reversal_data(100, min_len=3, max_len=6)
    test_long = generate_reversal_data(50, min_len=7, max_len=10)

    print(f"   Train: {len(train_data)}, Test: {len(test_data)}, Long (OOD): {len(test_long)}")

    # Device
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
        "Standard Transformer (4L)": StandardTransformer(VOCAB_SIZE, D_MODEL, n_layers=4),
        "Tick-CTM (2L x 8T)": TickCTM(VOCAB_SIZE, D_MODEL, n_layers=2, n_ticks=8),
        "ProperCTM": ProperCTM(VOCAB_SIZE, D_MODEL, n_layers=2,
                               config=ProperCTMConfig(n_ticks=8, use_sync_residual=False)),
        "ProperCTM+SyncRes": ProperCTM(VOCAB_SIZE, D_MODEL, n_layers=2,
                                        config=ProperCTMConfig(n_ticks=8, use_sync_residual=True)),
    }

    for name, model in models.items():
        model = model.to(device)
        params = count_params(model)
        print(f"   {name}: {params:,} params")

    print("\n3. Training...")
    print("-" * 70)

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
            import traceback
            traceback.print_exc()
            results[name] = {'test': 0, 'long': 0, 'params': count_params(model)}

    # =============================================================================
    # RESULTS
    # =============================================================================

    print("\n" + "=" * 70)
    print("RESULTS: Proper CTM vs Baselines")
    print("=" * 70)

    print(f"\n{'Model':<35} {'Test':>10} {'OOD':>10} {'Params':>12}")
    print("-" * 70)

    for name, r in sorted(results.items(), key=lambda x: x[1]['test'], reverse=True):
        print(f"{name:<35} {r['test']:>9.1%} {r['long']:>9.1%} {r['params']:>12,}")

    print("\n" + "=" * 70)
    print("KEY COMPARISON")
    print("=" * 70)

    if all(r['test'] > 0 for r in results.values()):
        std = results["Standard Transformer (4L)"]['test']
        tick = results["Tick-CTM (2L x 8T)"]['test']
        proper = results.get("ProperCTM", {}).get('test', 0)
        proper_sync = results.get("ProperCTM+SyncRes", {}).get('test', 0)

        print(f"""
Standard Transformer: {std:.1%}
Tick-CTM (our prev):  {tick:.1%} ({tick - std:+.1%})
ProperCTM:            {proper:.1%} ({proper - std:+.1%})
ProperCTM+SyncRes:    {proper_sync:.1%} ({proper_sync - std:+.1%})

Does proper CTM implementation help?
- vs Standard: {proper - std:+.1%}
- vs Tick-CTM: {proper - tick:+.1%}
""")


if __name__ == "__main__":
    main()
