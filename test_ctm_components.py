"""
Incremental CTM Component Analysis

Goal: Add real CTM components one-by-one to our Tick-CTM baseline.
See which components help, which break things, and why.

Components:
A. History Buffer - Store activations across ticks (Z tensor)
B. Sync Computation - S = Z·Z^T (correlation matrix)
C. Output from Sync - Predict from sync features, not hidden state
D. Per-neuron Processing - Each neuron processes its own history
E. Parallel Paths - Multiple refinement tracks (NEW IDEA)
F. Adaptive Halting - Stop when confident

The parallel paths idea:
- Instead of: tick1 → tick2 → tick3 (sequential)
- Try: [path1, path2, path3] all refine in parallel, then merge
- Like beam search for thinking, or mixture of experts for time
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from typing import Optional, List, Tuple
from dataclasses import dataclass

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

print("=" * 80)
print("INCREMENTAL CTM COMPONENT ANALYSIS")
print("=" * 80)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class CTMConfig:
    """Toggle components on/off"""
    use_history_buffer: bool = False      # A: Store Z across ticks
    use_sync_computation: bool = False    # B: Compute S = Z·Z^T (auxiliary)
    use_output_from_sync: bool = False    # C: Output from sync features
    use_sync_attention_bias: bool = False # C2: Use sync to bias attention (gentler)
    use_per_neuron_processing: bool = False  # D: Per-neuron history processing (OLD - last tick only)
    use_proper_nlm: bool = False          # D2: PROPER NLM - every tick, additive
    use_parallel_paths: bool = False      # E: Multiple refinement paths (independent)
    use_communicating_paths: bool = False # E2: Paths that share information
    use_adaptive_halting: bool = False    # F: Stop when confident
    use_sync_gating: bool = False         # G: Gate hidden state with sync
    use_sync_residual: bool = False       # H: Add sync as residual (gentle)

    n_ticks: int = 8
    n_parallel_paths: int = 4  # For component E/E2
    halt_threshold: float = 0.95  # For component F
    nlm_scale: float = 0.1  # Scale for NLM contribution

    def name(self) -> str:
        """Generate descriptive name"""
        parts = ["Tick-CTM"]
        if self.use_history_buffer:
            parts.append("+Hist")
        if self.use_sync_computation:
            parts.append("+Sync")
        if self.use_output_from_sync:
            parts.append("+SyncOut")
        if self.use_sync_attention_bias:
            parts.append("+SyncAttn")
        if self.use_sync_gating:
            parts.append("+SyncGate")
        if self.use_sync_residual:
            parts.append("+SyncRes")
        if self.use_per_neuron_processing:
            parts.append("+NLM(old)")
        if self.use_proper_nlm:
            parts.append("+NLM")
        if self.use_parallel_paths:
            parts.append(f"+Para({self.n_parallel_paths})")
        if self.use_communicating_paths:
            parts.append(f"+CommPara({self.n_parallel_paths})")
        if self.use_adaptive_halting:
            parts.append("+Halt")
        return "".join(parts) if len(parts) > 1 else "Tick-CTM (baseline)"


# =============================================================================
# MODULAR CTM MODEL
# =============================================================================

class PerNeuronMLP(nn.Module):
    """Component D: Per-neuron history processing (simplified NLM) - OLD VERSION"""
    def __init__(self, history_len: int, hidden_dim: int = 16):
        super().__init__()
        # Small MLP that processes one neuron's history
        self.mlp = nn.Sequential(
            nn.Linear(history_len, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        # history: (B, T, history_len) for one neuron
        return self.mlp(history).squeeze(-1)  # (B, T)


class ProperNLM(nn.Module):
    """
    Component D2: PROPER Neuron-Level Model (SuperLinear from real CTM)

    Based on real CTM: models/modules.py - SuperLinear class

    Key features:
    1. Uses EINSUM - NO PYTHON LOOPS over neurons
    2. Each neuron has UNIQUE weights (not shared)
    3. Processes ALL neurons in parallel efficiently
    4. Uses GLU activation like real CTM

    Real CTM formula:
        out = einsum('BDM,MHD->BDH', x, w1) + b1  # Per-neuron linear
        out = GLU(out)  # Gated activation

    Adapted for transformer (with sequence dim T):
        out = einsum('BTDM,MHD->BTDH', x, w1) + b1
    """
    def __init__(self, d_model: int, max_history: int = 8, hidden_dim: int = 32):
        super().__init__()
        self.d_model = d_model
        self.max_history = max_history
        import math

        # Per-neuron weights: (M, H, D) - unique weights for each of D neurons
        # M = memory/history length, H = hidden dim, D = num neurons
        self.w1 = nn.Parameter(
            torch.empty(max_history, hidden_dim * 2, d_model).uniform_(
                -1/math.sqrt(max_history + hidden_dim),
                1/math.sqrt(max_history + hidden_dim)
            )
        )
        # Per-neuron bias: (1, 1, D, H*2) for broadcasting
        self.b1 = nn.Parameter(torch.zeros(1, 1, d_model, hidden_dim * 2))

        # Second layer: (H, O, D) where O=1 for final output
        self.w2 = nn.Parameter(
            torch.empty(hidden_dim, 2, d_model).uniform_(
                -1/math.sqrt(hidden_dim + 1),
                1/math.sqrt(hidden_dim + 1)
            )
        )
        self.b2 = nn.Parameter(torch.zeros(1, 1, d_model, 2))

        # Temperature scaling (from real CTM)
        self.T = nn.Parameter(torch.ones(1))

    def forward(self, Z_list: list) -> torch.Tensor:
        """
        Z_list: list of tensors, each (B, T, D) - history so far
        Returns: (B, T, D) - contribution to add to hidden state
        """
        if len(Z_list) == 0:
            return None

        # Stack history: (B, T, n_ticks_so_far, D)
        Z = torch.stack(Z_list, dim=2)
        B, T, H, D = Z.shape

        # Pad or truncate to max_history
        if H < self.max_history:
            pad = torch.zeros(B, T, self.max_history - H, D, device=Z.device)
            Z = torch.cat([pad, Z], dim=2)
        elif H > self.max_history:
            Z = Z[:, :, -self.max_history:, :]

        # Z is now (B, T, M, D) where M=max_history
        # Transpose to (B, T, D, M) for einsum
        Z = Z.transpose(2, 3)  # (B, T, D, M)

        # First layer: einsum applies unique weights per neuron
        # (B, T, D, M) @ (M, H*2, D) -> (B, T, D, H*2)
        h = torch.einsum('btdm,mhd->btdh', Z, self.w1) + self.b1

        # GLU activation (halves hidden dim)
        h = F.glu(h, dim=-1)  # (B, T, D, H)

        # Second layer
        # (B, T, D, H) @ (H, 2, D) -> (B, T, D, 2)
        out = torch.einsum('btdh,hod->btdo', h, self.w2) + self.b2

        # GLU and squeeze
        out = F.glu(out, dim=-1)  # (B, T, D, 1)
        out = out.squeeze(-1) / self.T  # (B, T, D)

        return out


class ParallelPath(nn.Module):
    """Component E: One refinement path"""
    def __init__(self, d_model: int, n_heads: int = 4):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model)
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # Self-attention
        attn_out, _ = self.attn(x, x, x, attn_mask=mask, is_causal=False)
        x = self.ln1(x + attn_out)
        # FFN
        x = self.ln2(x + self.ffn(x))
        return x


class CommunicatingPaths(nn.Module):
    """
    Component E2: Parallel paths that communicate.

    Key insight: Independent parallel paths can't build on each other.
    Solution: After each "parallel tick", paths exchange information.

    Architecture:
    - Each path refines independently for one step
    - Then paths attend to each other (cross-path attention)
    - Repeat

    This is like having multiple "trains of thought" that periodically sync up.
    """
    def __init__(self, d_model: int, n_paths: int = 4, n_heads: int = 4):
        super().__init__()
        self.n_paths = n_paths
        self.d_model = d_model

        # Per-path refinement (independent)
        self.path_refine = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model * 2),
                nn.GELU(),
                nn.Linear(d_model * 2, d_model),
                nn.LayerNorm(d_model)
            ) for _ in range(n_paths)
        ])

        # Cross-path communication (paths attend to each other)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ln_cross = nn.LayerNorm(d_model)

        # Final merge
        self.merge = nn.Linear(d_model * n_paths, d_model)
        self.ln_out = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape

        # Initialize each path with the input
        paths = [x.clone() for _ in range(self.n_paths)]

        # Step 1: Independent refinement
        for i, refine in enumerate(self.path_refine):
            paths[i] = paths[i] + refine(paths[i])

        # Step 2: Cross-path communication
        # Stack paths: (B, T, n_paths, D) -> reshape for attention
        stacked = torch.stack(paths, dim=2)  # (B, T, n_paths, D)

        # Reshape: treat each position's paths as a sequence
        # (B * T, n_paths, D)
        cross_input = stacked.view(B * T, self.n_paths, D)

        # Cross attention (paths attend to each other at each position)
        cross_out, _ = self.cross_attn(cross_input, cross_input, cross_input)
        cross_out = self.ln_cross(cross_input + cross_out)

        # Reshape back: (B, T, n_paths, D)
        cross_out = cross_out.view(B, T, self.n_paths, D)

        # Update paths with cross-attention output
        for i in range(self.n_paths):
            paths[i] = cross_out[:, :, i, :]

        # Step 3: Merge paths
        concat = torch.cat(paths, dim=-1)  # (B, T, n_paths * D)
        merged = self.merge(concat)  # (B, T, D)

        return self.ln_out(merged)


class ModularCTM(nn.Module):
    """
    CTM with toggleable components.
    Start from Tick-CTM baseline, add components incrementally.
    """
    def __init__(self, vocab_size: int, d_model: int = 64, n_layers: int = 2,
                 n_heads: int = 4, config: CTMConfig = None):
        super().__init__()
        self.config = config or CTMConfig()
        self.d_model = d_model
        self.n_layers = n_layers

        # Embedding
        self.embed = nn.Embedding(vocab_size, d_model)

        # Base transformer layers (shared across ticks - weight tying)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 4, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Component D: Per-neuron processing (OLD - uses Python loops)
        if self.config.use_per_neuron_processing:
            self.nlms = nn.ModuleList([
                PerNeuronMLP(self.config.n_ticks, hidden_dim=16)
                for _ in range(d_model)
            ])

        # Component D2: Proper NLM (NEW - uses einsum, no loops)
        if self.config.use_proper_nlm:
            self.proper_nlm = ProperNLM(
                d_model=d_model,
                max_history=self.config.n_ticks,
                hidden_dim=32
            )

        # Component E: Parallel refinement paths (independent)
        if self.config.use_parallel_paths:
            self.parallel_paths = nn.ModuleList([
                ParallelPath(d_model, n_heads)
                for _ in range(self.config.n_parallel_paths)
            ])
            # Merge layer: combine parallel paths
            self.path_merger = nn.Linear(d_model * self.config.n_parallel_paths, d_model)

        # Component E2: Communicating parallel paths
        if self.config.use_communicating_paths:
            self.comm_paths = CommunicatingPaths(d_model, self.config.n_parallel_paths, n_heads)

        # Component C: Output from sync
        if self.config.use_output_from_sync:
            # Sync features dimension depends on how many pairs we sample
            self.n_sync_pairs = min(64, d_model * (d_model - 1) // 2)
            self.sync_pairs = self._generate_sync_pairs(d_model, self.n_sync_pairs)
            self.sync_head = nn.Linear(self.n_sync_pairs, vocab_size)

        # Component C2: Sync attention bias (gentler - use sync to bias attention)
        if self.config.use_sync_attention_bias:
            self.n_sync_pairs = min(64, d_model * (d_model - 1) // 2)
            self.sync_pairs = self._generate_sync_pairs(d_model, self.n_sync_pairs)
            # Project sync features to attention bias
            self.sync_to_attn = nn.Linear(self.n_sync_pairs, n_heads)

        # Component G: Sync gating (use sync to gate hidden state)
        if self.config.use_sync_gating:
            self.n_sync_pairs = min(64, d_model * (d_model - 1) // 2)
            self.sync_pairs = self._generate_sync_pairs(d_model, self.n_sync_pairs)
            # Project sync to gating values
            self.sync_to_gate = nn.Sequential(
                nn.Linear(self.n_sync_pairs, d_model),
                nn.Sigmoid()
            )

        # Component H: Sync residual (gentle - add sync as small residual)
        if self.config.use_sync_residual:
            self.n_sync_pairs = min(64, d_model * (d_model - 1) // 2)
            self.sync_pairs = self._generate_sync_pairs(d_model, self.n_sync_pairs)
            # Project sync to residual (small contribution)
            self.sync_to_residual = nn.Sequential(
                nn.Linear(self.n_sync_pairs, d_model),
                nn.Tanh()  # Bounded output
            )
            # Learnable scale (starts small)
            self.sync_scale = nn.Parameter(torch.tensor(0.1))

        # Component F: Adaptive halting
        if self.config.use_adaptive_halting:
            self.halt_predictor = nn.Linear(d_model, 1)

        # Output head
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

        self.name = self.config.name()

    def _generate_sync_pairs(self, d_model: int, n_pairs: int) -> List[Tuple[int, int]]:
        """Generate neuron pairs for sync computation"""
        all_pairs = [(i, j) for i in range(d_model) for j in range(i+1, d_model)]
        random.shuffle(all_pairs)
        return all_pairs[:n_pairs]

    def compute_sync_features(self, Z: torch.Tensor) -> torch.Tensor:
        """
        Component B: Compute sync features from history buffer.
        Z: (B, T, n_ticks, D) - activation history
        Returns: (B, T, n_sync_pairs)
        """
        B, T, n_ticks, D = Z.shape

        sync_features = []
        for (i, j) in self.sync_pairs:
            # Get histories for neurons i and j
            zi = Z[:, :, :, i]  # (B, T, n_ticks)
            zj = Z[:, :, :, j]  # (B, T, n_ticks)
            # Correlation across time (dot product of histories)
            sync_ij = (zi * zj).sum(dim=-1)  # (B, T)
            sync_features.append(sync_ij)

        return torch.stack(sync_features, dim=-1)  # (B, T, n_sync_pairs)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        B, T = x.shape

        # Causal mask
        mask = torch.triu(torch.ones(T, T, device=x.device) * float('-inf'), diagonal=1)

        # Initial embedding
        h = self.embed(x)

        # Component A: History buffer (use list to avoid inplace issues with gradients)
        if self.config.use_history_buffer:
            Z_list = []

        # Track halting info for component F
        halt_info = {'ticks_used': self.config.n_ticks, 'halted_early': False}

        # === MAIN TICK LOOP ===
        for tick in range(self.config.n_ticks):

            # Component E: Parallel paths (independent)
            if self.config.use_parallel_paths:
                path_outputs = []
                for path in self.parallel_paths:
                    path_out = path(h, mask)
                    path_outputs.append(path_out)
                # Concatenate and merge
                concat = torch.cat(path_outputs, dim=-1)  # (B, T, D * n_paths)
                h_new = self.path_merger(concat)  # (B, T, D)
            # Component E2: Communicating parallel paths
            elif self.config.use_communicating_paths:
                h_new = self.comm_paths(h)  # Paths refine and communicate
            else:
                # Standard transformer pass
                h_new = self.transformer(h, mask=mask)

            # Accumulation (key insight from our experiments)
            h = h + h_new

            # Component A: Store in history buffer
            if self.config.use_history_buffer:
                # Apply activation before storing (like CTM's post-activation)
                Z_list.append(torch.tanh(h))  # Append to list, no inplace op

            # Component G: Sync gating (gate hidden state with sync after tick 0)
            if self.config.use_sync_gating and self.config.use_history_buffer and tick > 0:
                # Stack history so far into tensor
                Z_partial = torch.stack(Z_list, dim=2)  # (B, T, tick+1, D)
                sync_features = self.compute_sync_features(Z_partial)  # (B, T, n_sync_pairs)
                gate = self.sync_to_gate(sync_features)  # (B, T, D) values in [0, 1]
                h = h * gate  # Gate the hidden state

            # Component H: Sync residual (gentle - add small sync contribution)
            if self.config.use_sync_residual and self.config.use_history_buffer and tick > 0:
                Z_partial = torch.stack(Z_list, dim=2)  # (B, T, tick+1, D)
                sync_features = self.compute_sync_features(Z_partial)  # (B, T, n_sync_pairs)
                sync_residual = self.sync_to_residual(sync_features)  # (B, T, D)
                h = h + self.sync_scale * sync_residual  # Small additive contribution

            # Component D2: Proper NLM (runs EVERY tick, uses einsum)
            if self.config.use_proper_nlm and self.config.use_history_buffer and tick > 0:
                nlm_contribution = self.proper_nlm(Z_list)  # (B, T, D)
                if nlm_contribution is not None:
                    h = h + self.config.nlm_scale * nlm_contribution  # Additive, scaled

            # Component D: Per-neuron processing (OLD - runs only on last tick)
            if self.config.use_per_neuron_processing and tick == self.config.n_ticks - 1:
                # On last tick, process each neuron's history
                Z = torch.stack(Z_list, dim=2)  # (B, T, n_ticks, D)
                nlm_outputs = []
                for neuron_idx, nlm in enumerate(self.nlms):
                    neuron_history = Z[:, :, :, neuron_idx]  # (B, T, n_ticks)
                    nlm_out = nlm(neuron_history)  # (B, T)
                    nlm_outputs.append(nlm_out)
                h = torch.stack(nlm_outputs, dim=-1)  # (B, T, D)

            # Component F: Adaptive halting
            if self.config.use_adaptive_halting and tick > 0:
                halt_prob = torch.sigmoid(self.halt_predictor(h)).mean()
                if halt_prob > self.config.halt_threshold:
                    halt_info['ticks_used'] = tick + 1
                    halt_info['halted_early'] = True
                    break

        # === OUTPUT ===

        # Component C: Output from sync
        if self.config.use_output_from_sync and self.config.use_history_buffer:
            Z = torch.stack(Z_list, dim=2)  # (B, T, n_ticks, D)
            sync_features = self.compute_sync_features(Z)  # (B, T, n_sync_pairs)
            logits = self.sync_head(sync_features)  # (B, T, vocab_size)
        else:
            # Standard output from hidden state
            logits = self.head(self.ln(h))

        return logits, halt_info


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
# DATA: Parity Task (count 1s modulo 2)
# =============================================================================

def generate_parity_data(n_samples, min_len=5, max_len=15):
    """
    Parity task: Given sequence of 0s and 1s, predict parity (count of 1s mod 2).
    This is a COUNTING task - transformers struggle, CTM should help.

    Format: [BOS, b1, b2, ..., bn, SEP, parity, EOS]
    """
    PAD, BOS, EOS, SEP = 0, 1, 2, 3
    ZERO, ONE = 4, 5  # Tokens for 0 and 1
    EVEN, ODD = 6, 7  # Tokens for parity result

    data = []
    for _ in range(n_samples):
        length = random.randint(min_len, max_len)
        bits = [random.randint(0, 1) for _ in range(length)]
        parity = sum(bits) % 2

        seq = [BOS]
        seq.extend([ZERO if b == 0 else ONE for b in bits])
        seq.append(SEP)
        seq.append(EVEN if parity == 0 else ODD)
        seq.append(EOS)

        data.append(seq)
    return data


def evaluate_parity(model, data, device='cpu'):
    """Evaluate on parity task - accuracy of predicting parity token."""
    model.eval()
    SEP = 3

    with torch.no_grad():
        batch = pad_sequences(data)
        x = torch.tensor(batch, device=device)

        inputs = x[:, :-1]
        targets = x[:, 1:]

        logits, _ = model(inputs)
        preds = logits.argmax(dim=-1)

        correct = 0
        for i in range(len(data)):
            seq = data[i]
            sep_pos = seq.index(SEP)
            # Parity prediction is at position sep_pos (predicting token after SEP)
            if preds[i, sep_pos] == targets[i, sep_pos]:
                correct += 1

        return correct / len(data)


# =============================================================================
# DATA: Addition Task (sum of digits)
# =============================================================================

def generate_addition_data(n_samples, min_len=2, max_len=5):
    """
    Addition task: Given sequence of single digits, predict sum.

    Format: [BOS, d1, d2, ..., dn, SEP, digit1, digit2, EOS]
    Sum is encoded as two digits (00-45 for max 5x9=45)
    """
    PAD, BOS, EOS, SEP = 0, 1, 2, 3
    DIGIT_OFFSET = 4  # Digits 0-9 map to tokens 4-13

    data = []
    for _ in range(n_samples):
        length = random.randint(min_len, max_len)
        digits = [random.randint(0, 9) for _ in range(length)]
        total = sum(digits)

        seq = [BOS]
        seq.extend([d + DIGIT_OFFSET for d in digits])
        seq.append(SEP)
        # Encode sum as two digits
        seq.append((total // 10) + DIGIT_OFFSET)
        seq.append((total % 10) + DIGIT_OFFSET)
        seq.append(EOS)

        data.append(seq)
    return data


def evaluate_addition(model, data, device='cpu'):
    """Evaluate on addition task - both sum digits must be correct."""
    model.eval()
    SEP = 3

    with torch.no_grad():
        batch = pad_sequences(data)
        x = torch.tensor(batch, device=device)

        inputs = x[:, :-1]
        targets = x[:, 1:]

        logits, _ = model(inputs)
        preds = logits.argmax(dim=-1)

        correct = 0
        for i in range(len(data)):
            seq = data[i]
            sep_pos = seq.index(SEP)
            # Sum is 2 tokens after SEP
            pred_d1 = preds[i, sep_pos]
            pred_d2 = preds[i, sep_pos + 1]
            target_d1 = targets[i, sep_pos]
            target_d2 = targets[i, sep_pos + 1]

            if pred_d1 == target_d1 and pred_d2 == target_d2:
                correct += 1

        return correct / len(data)


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

    return model


def evaluate(model, data, device='cpu'):
    model.eval()
    SEP = 3

    total_halt_ticks = 0
    n_samples = 0

    with torch.no_grad():
        batch = pad_sequences(data)
        x = torch.tensor(batch, device=device)

        inputs = x[:, :-1]
        targets = x[:, 1:]

        logits, halt_info = model(inputs)
        preds = logits.argmax(dim=-1)

        total_halt_ticks += halt_info['ticks_used']
        n_samples += 1

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

        acc = reversal_correct / reversal_tokens if reversal_tokens > 0 else 0
        avg_ticks = total_halt_ticks / n_samples

        return acc, avg_ticks


def count_params(model):
    return sum(p.numel() for p in model.parameters())


# =============================================================================
# EXPERIMENT CONFIGURATIONS
# =============================================================================

def get_experiment_configs():
    """Define all experimental configurations"""

    configs = [
        # ===================
        # BASELINE
        # ===================
        CTMConfig(n_ticks=8),

        # ===================
        # PROVEN WINNERS (from previous runs)
        # ===================

        # A: History buffer only
        CTMConfig(use_history_buffer=True, n_ticks=8),

        # A+B: History + Sync (PROVEN BEST: +2.6% test, +13% OOD)
        CTMConfig(use_history_buffer=True, use_sync_computation=True, n_ticks=8),

        # ===================
        # PROPER NLM (NEW - using einsum, no loops)
        # ===================

        # A+D2: History + Proper NLM
        CTMConfig(
            use_history_buffer=True,
            use_proper_nlm=True,
            n_ticks=8
        ),

        # A+B+D2: History + Sync + Proper NLM
        CTMConfig(
            use_history_buffer=True,
            use_sync_computation=True,
            use_proper_nlm=True,
            n_ticks=8
        ),

        # ===================
        # SYNC VARIATIONS
        # ===================

        # A+H: History + Sync Residual (gentle sync use)
        CTMConfig(
            use_history_buffer=True,
            use_sync_residual=True,
            n_ticks=8
        ),

        # A+B+H: History + Sync + SyncRes
        CTMConfig(
            use_history_buffer=True,
            use_sync_computation=True,
            use_sync_residual=True,
            n_ticks=8
        ),

        # ===================
        # PARALLEL VARIATIONS
        # ===================

        # E: Independent parallel paths
        CTMConfig(use_parallel_paths=True, n_parallel_paths=4, n_ticks=8),

        # E2: Communicating parallel paths
        CTMConfig(use_communicating_paths=True, n_parallel_paths=4, n_ticks=8),

        # ===================
        # ADAPTIVE HALTING
        # ===================

        # F: Halting only
        CTMConfig(use_adaptive_halting=True, halt_threshold=0.9, n_ticks=8),

        # A+B+F: Best combo + halting
        CTMConfig(
            use_history_buffer=True,
            use_sync_computation=True,
            use_adaptive_halting=True,
            halt_threshold=0.9,
            n_ticks=8
        ),
    ]

    return configs


# =============================================================================
# MAIN
# =============================================================================

def run_single_seed(seed, device):
    """Run all experiments with a single seed"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    train_data = generate_reversal_data(500, min_len=3, max_len=6)
    test_data = generate_reversal_data(100, min_len=3, max_len=6)
    test_long = generate_reversal_data(50, min_len=7, max_len=10)

    VOCAB_SIZE = 14
    D_MODEL = 64

    configs = get_experiment_configs()
    results = {}

    for config in configs:
        name = config.name()
        try:
            model = ModularCTM(VOCAB_SIZE, D_MODEL, n_layers=2, config=config)
            model = model.to(device)
            model = train_model(model, train_data, n_epochs=150, device=device)
            test_acc, _ = evaluate(model, test_data, device)
            long_acc, _ = evaluate(model, test_long, device)
            results[name] = {'test': test_acc, 'long': long_acc, 'params': count_params(model)}
        except Exception as e:
            print(f"   {name}: FAILED - {e}")
            results[name] = {'test': 0, 'long': 0, 'params': 0}

    return results


def run_multi_task_benchmark():
    """
    Test best configs across multiple tasks:
    1. Reversal (lookup)
    2. Parity (counting)
    3. Addition (arithmetic)
    """
    print("=" * 80)
    print("MULTI-TASK BENCHMARK")
    print("=" * 80)

    # Device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Device: {device}")

    # Best configs from reversal benchmark
    configs = {
        "Baseline": CTMConfig(n_ticks=8),
        "Hist+Sync": CTMConfig(use_history_buffer=True, use_sync_computation=True, n_ticks=8),
        "Hist+Sync+SyncRes": CTMConfig(use_history_buffer=True, use_sync_computation=True,
                                        use_sync_residual=True, n_ticks=8),
        "Hist+NLM": CTMConfig(use_history_buffer=True, use_proper_nlm=True, n_ticks=8),
    }

    # Task definitions
    tasks = {
        "Reversal": {
            "train_fn": lambda: generate_reversal_data(500, 3, 6),
            "test_fn": lambda: generate_reversal_data(100, 3, 6),
            "ood_fn": lambda: generate_reversal_data(50, 7, 10),
            "eval_fn": evaluate,
            "vocab_size": 14,
        },
        "Parity": {
            "train_fn": lambda: generate_parity_data(500, 5, 10),
            "test_fn": lambda: generate_parity_data(100, 5, 10),
            "ood_fn": lambda: generate_parity_data(50, 15, 25),
            "eval_fn": evaluate_parity,
            "vocab_size": 8,
        },
        "Addition": {
            "train_fn": lambda: generate_addition_data(500, 2, 4),
            "test_fn": lambda: generate_addition_data(100, 2, 4),
            "ood_fn": lambda: generate_addition_data(50, 5, 7),
            "eval_fn": evaluate_addition,
            "vocab_size": 14,
        },
    }

    # Results storage
    all_results = {task: {} for task in tasks}

    for task_name, task_info in tasks.items():
        print(f"\n{'='*60}")
        print(f"TASK: {task_name}")
        print(f"{'='*60}")

        for config_name, config in configs.items():
            print(f"\n  {config_name}...")

            # Generate data
            torch.manual_seed(42)
            np.random.seed(42)
            random.seed(42)

            train_data = task_info["train_fn"]()
            test_data = task_info["test_fn"]()
            ood_data = task_info["ood_fn"]()

            try:
                model = ModularCTM(task_info["vocab_size"], d_model=64, n_layers=2, config=config)
                model = model.to(device)
                model = train_model(model, train_data, n_epochs=150, device=device)

                test_acc = task_info["eval_fn"](model, test_data, device)
                ood_acc = task_info["eval_fn"](model, ood_data, device)

                if isinstance(test_acc, tuple):
                    test_acc = test_acc[0]
                if isinstance(ood_acc, tuple):
                    ood_acc = ood_acc[0]

                all_results[task_name][config_name] = {
                    "test": test_acc,
                    "ood": ood_acc
                }
                print(f"    Test: {test_acc:.1%}, OOD: {ood_acc:.1%}")

            except Exception as e:
                print(f"    FAILED: {e}")
                all_results[task_name][config_name] = {"test": 0, "ood": 0}

    # Summary
    print("\n" + "=" * 80)
    print("MULTI-TASK SUMMARY")
    print("=" * 80)

    # Header
    task_names = list(tasks.keys())
    header = f"{'Config':<25}"
    for task in task_names:
        header += f" | {task:>12} (T/O)"
    print(header)
    print("-" * len(header))

    # Rows
    for config_name in configs.keys():
        row = f"{config_name:<25}"
        for task in task_names:
            r = all_results[task].get(config_name, {"test": 0, "ood": 0})
            row += f" | {r['test']:>5.1%}/{r['ood']:<5.1%}"
        print(row)

    # Per-task best
    print("\n" + "-" * 60)
    print("BEST PER TASK:")
    for task in task_names:
        best_config = max(all_results[task].items(), key=lambda x: x[1]["test"])
        print(f"  {task}: {best_config[0]} ({best_config[1]['test']:.1%} test)")

    return all_results


def main():
    print("\n1. Setup...")

    # Device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"   Device: {device}")

    # Run multiple seeds
    SEEDS = [42, 123, 456]
    all_results = {}

    for seed in SEEDS:
        print(f"\n{'='*80}")
        print(f"SEED {seed}")
        print(f"{'='*80}")
        seed_results = run_single_seed(seed, device)

        for name, r in seed_results.items():
            if name not in all_results:
                all_results[name] = {'test': [], 'long': [], 'params': r['params']}
            all_results[name]['test'].append(r['test'])
            all_results[name]['long'].append(r['long'])
            print(f"   {name}: Test={r['test']:.1%}, OOD={r['long']:.1%}")

    # Aggregate results
    print("\n" + "=" * 80)
    print("AGGREGATED RESULTS (3 seeds)")
    print("=" * 80)
    print(f"\n{'Model':<40} {'Test Mean':>10} {'Test Std':>10} {'OOD Mean':>10} {'OOD Std':>10}")
    print("-" * 85)

    sorted_results = sorted(all_results.items(), key=lambda x: np.mean(x[1]['test']), reverse=True)
    for name, r in sorted_results:
        test_mean = np.mean(r['test'])
        test_std = np.std(r['test'])
        ood_mean = np.mean(r['long'])
        ood_std = np.std(r['long'])
        print(f"{name:<40} {test_mean:>9.1%} {test_std:>9.1%} {ood_mean:>9.1%} {ood_std:>9.1%}")

    # Print for research log
    print("\n" + "=" * 80)
    print("MARKDOWN TABLE FOR RESEARCH LOG")
    print("=" * 80)
    print("\n| Model | Test (mean±std) | OOD (mean±std) | Params |")
    print("|-------|-----------------|----------------|--------|")
    for name, r in sorted_results:
        test_mean = np.mean(r['test'])
        test_std = np.std(r['test'])
        ood_mean = np.mean(r['long'])
        ood_std = np.std(r['long'])
        print(f"| {name} | {test_mean:.1%}±{test_std:.1%} | {ood_mean:.1%}±{ood_std:.1%} | {r['params']:,} |")


def main_single():
    """Original single-seed main for quick testing"""
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

    print("\n2. Running experiments...")
    print("-" * 80)

    configs = get_experiment_configs()
    results = []

    for config in configs:
        name = config.name()
        print(f"\n   {name}")

        try:
            model = ModularCTM(VOCAB_SIZE, D_MODEL, n_layers=2, config=config)
            model = model.to(device)
            params = count_params(model)

            model = train_model(model, train_data, n_epochs=150, device=device)

            test_acc, test_ticks = evaluate(model, test_data, device)
            long_acc, long_ticks = evaluate(model, test_long, device)

            results.append({
                'name': name,
                'config': config,
                'test_acc': test_acc,
                'long_acc': long_acc,
                'params': params,
                'avg_ticks': test_ticks
            })

            print(f"      Test: {test_acc:.1%}, OOD: {long_acc:.1%}, Params: {params:,}, Ticks: {test_ticks:.1f}")

        except Exception as e:
            print(f"      FAILED: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                'name': name,
                'config': config,
                'test_acc': 0,
                'long_acc': 0,
                'params': 0,
                'avg_ticks': 0
            })

    # =============================================================================
    # RESULTS
    # =============================================================================

    print("\n" + "=" * 80)
    print("RESULTS: Component Analysis")
    print("=" * 80)

    # Sort by test accuracy
    results.sort(key=lambda x: x['test_acc'], reverse=True)

    print(f"\n{'Model':<45} {'Test':>8} {'OOD':>8} {'Params':>10} {'Ticks':>6}")
    print("-" * 80)

    baseline_acc = None
    for r in results:
        if 'baseline' in r['name'].lower():
            baseline_acc = r['test_acc']

        # Highlight if better than baseline
        marker = ""
        if baseline_acc and r['test_acc'] > baseline_acc:
            marker = " ✓"

        print(f"{r['name']:<45} {r['test_acc']:>7.1%} {r['long_acc']:>7.1%} {r['params']:>10,} {r['avg_ticks']:>6.1f}{marker}")

    # =============================================================================
    # ANALYSIS
    # =============================================================================

    print("\n" + "=" * 80)
    print("COMPONENT IMPACT ANALYSIS")
    print("=" * 80)

    baseline = next((r for r in results if 'baseline' in r['name'].lower()), results[-1])

    print(f"\nBaseline: {baseline['test_acc']:.1%} (test), {baseline['long_acc']:.1%} (OOD)")
    print("\nComponent Effects:")

    for r in results:
        if r['name'] != baseline['name']:
            test_delta = r['test_acc'] - baseline['test_acc']
            ood_delta = r['long_acc'] - baseline['long_acc']

            sign_test = "+" if test_delta >= 0 else ""
            sign_ood = "+" if ood_delta >= 0 else ""

            print(f"  {r['name']:<40} Test: {sign_test}{test_delta:.1%}, OOD: {sign_ood}{ood_delta:.1%}")

    print("\n" + "=" * 80)
    print("KEY INSIGHTS")
    print("=" * 80)

    # Find best overall
    best = max(results, key=lambda x: x['test_acc'] + x['long_acc'])

    print(f"""
Best Overall: {best['name']}
  - Test: {best['test_acc']:.1%}
  - OOD: {best['long_acc']:.1%}

Questions to answer:
1. Does history buffer help? (comparing baseline vs +History)
2. Does sync computation add value? (comparing +History vs +History+Sync)
3. Is output from sync better than output from hidden? (+History+Sync vs +History+Sync+SyncOut)
4. Do parallel paths beat sequential ticks? (baseline vs +Parallel)
5. Does adaptive halting save compute without hurting accuracy? (+Halt)
""")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "multi":
        run_multi_task_benchmark()
    else:
        main()
