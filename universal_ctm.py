"""
Universal CTM: Taking RealCTM's strengths, removing its weaknesses.

Key insights from RealCTM:
1. NLMs (per-neuron private weights) - enables temporal pattern recognition
2. History buffer - neurons "see" their past
3. Sync as representation - accumulation over time
4. Cross-attention each tick - iterative refinement
5. Phase transitions emerge naturally

RealCTM's weakness:
- Only works at training length (length=10 → 100%, others → 0%)
- Fixed sync dimensions tied to sequence structure

Goal: Architecture that beats transformers on BOTH:
- Counting tasks (parity) - requires temporal accumulation
- Memory tasks (reversal) - requires position mapping
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Tuple, Optional
import random

DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

# =============================================================================
# BUILDING BLOCKS
# =============================================================================

class NeuronLevelMLP(nn.Module):
    """
    Per-neuron private weights (like RealCTM's SuperLinear).

    Each of d_model neurons has its own MLP that processes its history.
    This is the key innovation - shared weights can't do counting.
    """
    def __init__(self, d_model: int, history_len: int, hidden_dim: int = 32):
        super().__init__()
        self.d_model = d_model
        self.history_len = history_len

        # Per-neuron weights: each neuron has its own linear transform
        # w1: (history_len, hidden_dim, d_model) - one per neuron
        # w2: (hidden_dim, 1, d_model) - one per neuron
        self.w1 = nn.Parameter(
            torch.empty(history_len, hidden_dim, d_model).uniform_(
                -1/math.sqrt(history_len), 1/math.sqrt(history_len)
            )
        )
        self.b1 = nn.Parameter(torch.zeros(1, d_model, hidden_dim))

        self.w2 = nn.Parameter(
            torch.empty(hidden_dim, 1, d_model).uniform_(
                -1/math.sqrt(hidden_dim), 1/math.sqrt(hidden_dim)
            )
        )
        self.b2 = nn.Parameter(torch.zeros(1, d_model, 1))

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        """
        history: (B, d_model, history_len) - each neuron's past states
        returns: (B, d_model) - activated state for each neuron
        """
        # First layer: (B, d_model, history_len) @ (history_len, hidden, d_model) -> (B, d_model, hidden)
        h = torch.einsum('bdh,hod->bdo', history, self.w1) + self.b1
        h = F.gelu(h)

        # Second layer: (B, d_model, hidden) @ (hidden, 1, d_model) -> (B, d_model, 1)
        out = torch.einsum('bdo,oid->bdi', h, self.w2) + self.b2
        return out.squeeze(-1)  # (B, d_model)


class TemporalAccumulator(nn.Module):
    """
    Accumulates information over iterations (like RealCTM's sync).

    Key insight: counting needs ACCUMULATION, not just residuals.

    Instead of fixed neuron pairs, we use a learnable accumulation:
    acc_t = decay * acc_{t-1} + f(state_t)
    """
    def __init__(self, d_model: int, acc_dim: int):
        super().__init__()
        self.d_model = d_model
        self.acc_dim = acc_dim

        # Project state to accumulator space
        self.state_to_acc = nn.Linear(d_model, acc_dim)

        # Learnable decay (like RealCTM's decay_params)
        self.decay_logit = nn.Parameter(torch.zeros(acc_dim))

    def forward(self, state: torch.Tensor, prev_acc: Optional[torch.Tensor],
                prev_count: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        state: (B, d_model)
        prev_acc: (B, acc_dim) or None
        prev_count: (B, acc_dim) or None

        Returns: (accumulator, new_acc, new_count)
        """
        B = state.shape[0]
        device = state.device

        # Project current state
        contribution = self.state_to_acc(state)  # (B, acc_dim)

        # Decay rate (0 to 1)
        decay = torch.sigmoid(self.decay_logit).unsqueeze(0)  # (1, acc_dim)

        if prev_acc is None:
            new_acc = contribution
            new_count = torch.ones(B, self.acc_dim, device=device)
        else:
            new_acc = decay * prev_acc + contribution
            new_count = decay * prev_count + 1

        # Normalized accumulator (like RealCTM's sync / sqrt(count))
        accumulator = new_acc / (torch.sqrt(new_count) + 1e-6)

        return accumulator, new_acc, new_count


class IterativeCrossAttention(nn.Module):
    """
    Re-read input at each iteration (like RealCTM).

    Key: Query comes from current state, allowing progressive refinement.
    """
    def __init__(self, d_model: int, n_heads: int = 4):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.q_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, state: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        """
        state: (B, d_model) - current state (used for query)
        kv: (B, T, d_model) - input embeddings (keys and values)

        Returns: (B, d_model) - attended information
        """
        q = self.q_proj(state).unsqueeze(1)  # (B, 1, d_model)
        attn_out, _ = self.attn(q, kv, kv)
        return self.norm(attn_out.squeeze(1))  # (B, d_model)


# =============================================================================
# UNIVERSAL CTM
# =============================================================================

class UniversalCTM(nn.Module):
    """
    Universal CTM: RealCTM's strengths + length-agnostic design.

    Architecture:
    1. Embed input tokens
    2. For each iteration:
       a. Cross-attend to input (query from accumulator)
       b. Update state through NLM (per-neuron processing of history)
       c. Accumulate over time
    3. Output from accumulator (not raw state)
    """

    def __init__(self, vocab_size: int, d_model: int = 128,
                 n_iterations: int = 30, history_len: int = 16,
                 acc_dim: int = 128, n_heads: int = 4):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_iterations = n_iterations
        self.history_len = history_len
        self.acc_dim = acc_dim

        # Input embedding
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(512, d_model)  # Learnable positional

        # Core components
        self.cross_attn = IterativeCrossAttention(d_model, n_heads)
        self.nlm = NeuronLevelMLP(d_model, history_len, hidden_dim=32)
        self.accumulator = TemporalAccumulator(d_model, acc_dim)

        # State update (like RealCTM's synapse, but simpler)
        self.synapse = nn.Sequential(
            nn.Linear(d_model * 2, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model)
        )

        # Output from accumulator
        self.output_proj = nn.Linear(acc_dim, vocab_size)

        # Initial states
        self.init_state = nn.Parameter(torch.randn(d_model) * 0.02)
        self.init_history = nn.Parameter(torch.randn(d_model, history_len) * 0.02)

        self.name = f"UniversalCTM(iter={n_iterations},d={d_model},hist={history_len})"

    def forward(self, x: torch.Tensor, track: bool = False) -> Tuple[torch.Tensor, dict]:
        """
        x: (B, T) token indices
        Returns: (B, T, vocab_size) logits
        """
        B, T = x.shape
        device = x.device

        # Embed input
        positions = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
        kv = self.embed(x) + self.pos_embed(positions)  # (B, T, d_model)

        # Initialize per-position states
        state = self.init_state.unsqueeze(0).unsqueeze(0).expand(B, T, -1).clone()  # (B, T, d_model)
        history = self.init_history.unsqueeze(0).unsqueeze(0).expand(B, T, -1, -1).clone()  # (B, T, d_model, history_len)

        # Accumulator state (per position)
        acc = None
        acc_count = None

        # Tracking
        tracking = {'states': [], 'accumulators': [], 'alpha': []} if track else None

        # Iterative refinement
        for iter_idx in range(self.n_iterations):
            # Process each position
            new_states = []
            new_histories = []

            for pos in range(T):
                pos_state = state[:, pos, :]  # (B, d_model)
                pos_history = history[:, pos, :, :]  # (B, d_model, history_len)

                # 1. Cross-attend to full input
                attn_out = self.cross_attn(pos_state, kv)  # (B, d_model)

                # 2. NLM processes history
                nlm_out = self.nlm(pos_history)  # (B, d_model)

                # 3. Synapse combines attention and NLM output
                combined = torch.cat([attn_out, nlm_out], dim=-1)
                delta = self.synapse(combined)  # (B, d_model)

                # 4. Update state
                new_state = pos_state + delta
                new_states.append(new_state)

                # 5. Update history (shift and append)
                new_hist = torch.cat([pos_history[:, :, 1:], new_state.unsqueeze(-1)], dim=-1)
                new_histories.append(new_hist)

                # Track alpha (for phase analysis)
                if track:
                    with torch.no_grad():
                        alpha = F.cosine_similarity(
                            pos_state.flatten(1), delta.flatten(1)
                        ).mean().item()
                        tracking['alpha'].append((iter_idx, pos, alpha))

            # Stack back
            state = torch.stack(new_states, dim=1)  # (B, T, d_model)
            history = torch.stack(new_histories, dim=1)  # (B, T, d_model, history_len)

            # 6. Accumulate (global over positions)
            state_mean = state.mean(dim=1)  # (B, d_model)
            accumulator, acc, acc_count = self.accumulator(state_mean, acc, acc_count)

            if track:
                tracking['states'].append(state.detach().cpu())
                tracking['accumulators'].append(accumulator.detach().cpu())

        # Output: use final accumulator to modulate per-position output
        # Expand accumulator to all positions
        acc_expanded = accumulator.unsqueeze(1).expand(-1, T, -1)  # (B, T, acc_dim)
        logits = self.output_proj(acc_expanded)  # (B, T, vocab_size)

        info = {'tracking': tracking} if track else {}
        return logits, info


class UniversalCTMv2(nn.Module):
    """
    Version 2: Simpler, faster, per-position accumulation.

    Changes from v1:
    - Per-position accumulator (not global mean)
    - Vectorized position processing
    - Output directly from accumulator per position
    """

    def __init__(self, vocab_size: int, d_model: int = 128,
                 n_iterations: int = 20, history_len: int = 8,
                 n_heads: int = 4):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_iterations = n_iterations
        self.history_len = history_len

        # Input
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(512, d_model)

        # Cross attention (re-read input each iteration)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.q_proj = nn.Linear(d_model, d_model)

        # NLM: per-neuron processing of history
        # Simpler version: 1D conv over history dimension
        self.nlm = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=history_len, groups=d_model),
            nn.GELU(),
        )

        # State update
        self.synapse = nn.Sequential(
            nn.Linear(d_model * 2, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model)
        )

        # Accumulator (per position)
        self.acc_proj = nn.Linear(d_model, d_model)
        self.decay_logit = nn.Parameter(torch.zeros(d_model))

        # Output
        self.output_proj = nn.Linear(d_model, vocab_size)

        # Initial states
        self.init_state = nn.Parameter(torch.randn(d_model) * 0.02)

        self.name = f"UniversalCTMv2(iter={n_iterations},d={d_model})"

    def forward(self, x: torch.Tensor, track: bool = False) -> Tuple[torch.Tensor, dict]:
        B, T = x.shape
        device = x.device

        # Embed
        positions = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
        kv = self.embed(x) + self.pos_embed(positions)

        # Initialize state per position
        state = self.init_state.view(1, 1, -1).expand(B, T, -1).clone()

        # History buffer: (B, T, d_model, history_len)
        history = torch.zeros(B, T, self.d_model, self.history_len, device=device)

        # Accumulator per position
        acc = torch.zeros(B, T, self.d_model, device=device)
        acc_count = torch.ones(B, T, self.d_model, device=device)
        decay = torch.sigmoid(self.decay_logit).view(1, 1, -1)

        tracking = {'alpha': []} if track else None

        for iter_idx in range(self.n_iterations):
            # 1. Cross-attention: state queries input
            q = self.q_proj(state)  # (B, T, d_model)
            attn_out, _ = self.cross_attn(q, kv, kv)  # (B, T, d_model)

            # 2. NLM: process history per position
            # Reshape for conv1d: (B*T, d_model, history_len)
            hist_flat = history.view(B * T, self.d_model, self.history_len)
            nlm_out = self.nlm(hist_flat).view(B, T, self.d_model)  # (B, T, d_model)

            # 3. Combine and update
            combined = torch.cat([attn_out, nlm_out], dim=-1)
            delta = self.synapse(combined)

            # Track alpha
            if track:
                with torch.no_grad():
                    alpha = F.cosine_similarity(
                        state.view(B, -1), delta.view(B, -1)
                    ).mean().item()
                    tracking['alpha'].append((iter_idx, alpha))

            # Update state
            new_state = state + delta

            # 4. Update history
            history = torch.cat([history[:, :, :, 1:], new_state.unsqueeze(-1)], dim=-1)

            # 5. Accumulate
            contribution = self.acc_proj(new_state)
            acc = decay * acc + contribution
            acc_count = decay * acc_count + 1

            state = new_state

        # Normalized accumulator
        accumulator = acc / (torch.sqrt(acc_count) + 1e-6)

        # Output
        logits = self.output_proj(accumulator)

        info = {'tracking': tracking} if track else {}
        return logits, info


class UniversalCTMv3(nn.Module):
    """
    Version 3: Closer to RealCTM's actual mechanics.

    Key changes:
    1. TRUE per-neuron weights (not grouped conv)
    2. Sync-like pairwise accumulation
    3. Output from sync representation
    4. Single global state (not per-position)
    """

    def __init__(self, vocab_size: int, d_model: int = 128,
                 n_iterations: int = 30, history_len: int = 16,
                 n_heads: int = 4, n_sync_pairs: int = 128):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_iterations = n_iterations
        self.history_len = history_len
        self.n_sync_pairs = n_sync_pairs

        # Input embedding
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(512, d_model)
        self.kv_proj = nn.Linear(d_model, d_model)

        # TRUE per-neuron weights (like RealCTM's SuperLinear)
        # Each neuron has its own MLP: history_len -> hidden -> 1
        nlm_hidden = 32
        self.nlm_w1 = nn.Parameter(torch.empty(history_len, nlm_hidden, d_model).uniform_(
            -1/math.sqrt(history_len), 1/math.sqrt(history_len)
        ))
        self.nlm_b1 = nn.Parameter(torch.zeros(1, d_model, nlm_hidden))
        self.nlm_w2 = nn.Parameter(torch.empty(nlm_hidden, 1, d_model).uniform_(
            -1/math.sqrt(nlm_hidden), 1/math.sqrt(nlm_hidden)
        ))
        self.nlm_b2 = nn.Parameter(torch.zeros(1, d_model))

        # Sync neuron pairs (random pairing like RealCTM)
        self.register_buffer('sync_left', torch.randint(0, d_model, (n_sync_pairs,)))
        self.register_buffer('sync_right', torch.randint(0, d_model, (n_sync_pairs,)))

        # Learnable decay for sync
        self.decay_logit = nn.Parameter(torch.zeros(n_sync_pairs))

        # Cross attention (query from sync)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.sync_to_query = nn.Linear(n_sync_pairs, d_model)

        # Synapse (state update)
        self.synapse = nn.Sequential(
            nn.Linear(d_model * 2, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model)
        )

        # Output from sync (like RealCTM)
        self.sync_to_output = nn.Linear(n_sync_pairs, vocab_size)

        # Initial state
        self.init_state = nn.Parameter(torch.randn(d_model) * 0.02)
        self.init_history = nn.Parameter(torch.randn(d_model, history_len) * 0.02)

        self.name = f"UniversalCTMv3(iter={n_iterations},d={d_model})"

    def compute_nlm(self, history: torch.Tensor) -> torch.Tensor:
        """
        Apply per-neuron MLP to history.
        history: (B, d_model, history_len)
        returns: (B, d_model)
        """
        # First layer: (B, d, h) @ (h, hidden, d) -> (B, d, hidden)
        h = torch.einsum('bdh,hod->bdo', history, self.nlm_w1) + self.nlm_b1
        h = F.gelu(h)
        # Second layer: (B, d, hidden) @ (hidden, 1, d) -> (B, d, 1)
        out = torch.einsum('bdo,oid->bdi', h, self.nlm_w2) + self.nlm_b2.unsqueeze(-1)
        return out.squeeze(-1)

    def compute_sync(self, state: torch.Tensor, prev_alpha: Optional[torch.Tensor],
                     prev_beta: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute sync representation (like RealCTM).
        state: (B, d_model)
        """
        B = state.shape[0]
        device = state.device

        # Get neuron values for sync pairs
        left = state[:, self.sync_left]   # (B, n_sync_pairs)
        right = state[:, self.sync_right]  # (B, n_sync_pairs)
        pairwise = left * right  # (B, n_sync_pairs)

        # Decay
        decay = torch.sigmoid(self.decay_logit).unsqueeze(0)  # (1, n_sync_pairs)

        if prev_alpha is None:
            alpha = pairwise
            beta = torch.ones(B, self.n_sync_pairs, device=device)
        else:
            alpha = decay * prev_alpha + pairwise
            beta = decay * prev_beta + 1

        sync = alpha / (torch.sqrt(beta) + 1e-6)
        return sync, alpha, beta

    def forward(self, x: torch.Tensor, track: bool = False) -> Tuple[torch.Tensor, dict]:
        B, T = x.shape
        device = x.device

        # Embed input
        positions = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
        embedded = self.embed(x) + self.pos_embed(positions)
        kv = self.kv_proj(embedded)  # (B, T, d_model)

        # Initialize global state (not per-position!)
        state = self.init_state.unsqueeze(0).expand(B, -1).clone()  # (B, d_model)
        history = self.init_history.unsqueeze(0).expand(B, -1, -1).clone()  # (B, d_model, history_len)

        # Sync accumulator
        sync_alpha = None
        sync_beta = None

        tracking = {'alpha': [], 'sync': []} if track else None

        for iter_idx in range(self.n_iterations):
            # 1. Compute sync from current state
            sync, sync_alpha, sync_beta = self.compute_sync(state, sync_alpha, sync_beta)

            # 2. Cross-attention: query from sync
            q = self.sync_to_query(sync).unsqueeze(1)  # (B, 1, d_model)
            attn_out, _ = self.cross_attn(q, kv, kv)  # (B, 1, d_model)
            attn_out = attn_out.squeeze(1)  # (B, d_model)

            # 3. NLM processes history
            nlm_out = self.compute_nlm(history)  # (B, d_model)

            # 4. Synapse combines
            combined = torch.cat([attn_out, nlm_out], dim=-1)
            delta = self.synapse(combined)

            # Track
            if track:
                with torch.no_grad():
                    alpha_val = F.cosine_similarity(state, delta).mean().item()
                    tracking['alpha'].append(alpha_val)
                    tracking['sync'].append(sync.mean().item())

            # 5. Update state
            state = state + delta

            # 6. Update history
            history = torch.cat([history[:, :, 1:], state.unsqueeze(-1)], dim=-1)

        # Final sync
        final_sync, _, _ = self.compute_sync(state, sync_alpha, sync_beta)

        # Output: expand sync to all positions (for seq2seq compatibility)
        # For parity, we only care about last position
        output = self.sync_to_output(final_sync)  # (B, vocab_size)

        # Expand to (B, T, vocab_size) - use contiguous expand
        logits = output.unsqueeze(1).expand(-1, T, -1).contiguous()

        info = {'tracking': tracking} if track else {}
        return logits, info


class UniversalCTMv5(nn.Module):
    """
    Version 5: Forced phase transitions.

    The key insight: v3/v4 stay stuck in AMPLIFY (α always positive).
    RealCTM naturally transitions because of the NLM + sync dynamics.

    Solution: Explicitly schedule phase transitions:
    - Iterations 0-9: AMPLIFY (positive gating)
    - Iterations 10-19: ROTATE (negative gating - subtract instead of add!)
    - Iterations 20-29: REFINE (small positive gating)
    """

    def __init__(self, vocab_size: int, d_model: int = 128,
                 n_iterations: int = 30, history_len: int = 16,
                 n_heads: int = 4):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_iterations = n_iterations
        self.history_len = history_len

        # Input
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(512, d_model)

        # Cross attention
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.q_proj = nn.Linear(d_model, d_model)

        # Per-neuron NLM
        nlm_hidden = 32
        self.nlm_w1 = nn.Parameter(torch.empty(history_len, nlm_hidden, d_model).uniform_(
            -1/math.sqrt(history_len), 1/math.sqrt(history_len)
        ))
        self.nlm_b1 = nn.Parameter(torch.zeros(1, d_model, nlm_hidden))
        self.nlm_w2 = nn.Parameter(torch.empty(nlm_hidden, 1, d_model).uniform_(
            -1/math.sqrt(nlm_hidden), 1/math.sqrt(nlm_hidden)
        ))
        self.nlm_b2 = nn.Parameter(torch.zeros(1, d_model))

        # Synapse
        self.synapse = nn.Sequential(
            nn.Linear(d_model * 2, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model)
        )

        # Phase gates: learned but initialized to encourage phase pattern
        # AMPLIFY -> ROTATE -> REFINE
        phase_init = (
            [1.0] * 10 +   # Amplify
            [-0.5] * 10 +  # Rotate (SUBTRACT!)
            [0.3] * 10     # Refine
        )
        self.phase_gates = nn.Parameter(torch.tensor(phase_init[:n_iterations]))

        # Accumulator
        self.acc_proj = nn.Linear(d_model, d_model)
        self.decay_logit = nn.Parameter(torch.zeros(d_model))

        # Output
        self.output_proj = nn.Linear(d_model, vocab_size)

        # Initial state
        self.init_state = nn.Parameter(torch.randn(d_model) * 0.02)
        self.init_history = nn.Parameter(torch.randn(d_model, history_len) * 0.02)

        self.name = f"UniversalCTMv5(iter={n_iterations},d={d_model})"

    def compute_nlm(self, history: torch.Tensor) -> torch.Tensor:
        h = torch.einsum('bdh,hod->bdo', history, self.nlm_w1) + self.nlm_b1
        h = F.gelu(h)
        out = torch.einsum('bdo,oid->bdi', h, self.nlm_w2) + self.nlm_b2.unsqueeze(-1)
        return out.squeeze(-1)

    def forward(self, x: torch.Tensor, track: bool = False) -> Tuple[torch.Tensor, dict]:
        B, T = x.shape
        device = x.device

        # Embed
        positions = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
        kv = self.embed(x) + self.pos_embed(positions)

        # Global state
        state = self.init_state.unsqueeze(0).expand(B, -1).clone()
        history = self.init_history.unsqueeze(0).expand(B, -1, -1).clone()

        # Accumulator
        acc = torch.zeros(B, self.d_model, device=device)
        acc_count = torch.ones(B, self.d_model, device=device)
        decay = torch.sigmoid(self.decay_logit).unsqueeze(0)

        tracking = {'alpha': [], 'gates': []} if track else None

        for iter_idx in range(self.n_iterations):
            # 1. Cross-attention
            q = self.q_proj(state).unsqueeze(1)
            attn_out, _ = self.cross_attn(q, kv, kv)
            attn_out = attn_out.squeeze(1)

            # 2. NLM
            nlm_out = self.compute_nlm(history)

            # 3. Synapse
            combined = torch.cat([attn_out, nlm_out], dim=-1)
            delta = self.synapse(combined)

            # 4. PHASE-GATED UPDATE (the key!)
            gate = self.phase_gates[iter_idx]
            state = state + gate * delta  # Gate can be negative!

            if track:
                with torch.no_grad():
                    # Compute effective alpha
                    effective_delta = gate * delta
                    alpha_val = F.cosine_similarity(
                        state - effective_delta, effective_delta
                    ).mean().item()
                    tracking['alpha'].append(alpha_val)
                    tracking['gates'].append(gate.item())

            # 5. Update history
            history = torch.cat([history[:, :, 1:], state.unsqueeze(-1)], dim=-1)

            # 6. Accumulate
            contrib = self.acc_proj(state)
            acc = decay * acc + contrib
            acc_count = decay * acc_count + 1

        # Normalized accumulator
        accumulator = acc / (torch.sqrt(acc_count) + 1e-6)

        # Output for all positions
        output = self.output_proj(accumulator)
        logits = output.unsqueeze(1).expand(-1, T, -1).contiguous()

        info = {'tracking': tracking} if track else {}
        return logits, info


class SimpleParity(nn.Module):
    """
    Simplified model for binary parity (0/1 input, 0/1 output).

    Inspired by RealCTM but stripped down:
    - Binary input (0/1)
    - Binary output (even/odd)
    - Global state with sync-like accumulation
    - No sequence overhead
    """

    def __init__(self, d_model: int = 64, n_iterations: int = 30,
                 history_len: int = 16, n_sync_pairs: int = 64, **kwargs):
        super().__init__()

        self.d_model = d_model
        self.n_iterations = n_iterations
        self.history_len = history_len
        self.n_sync_pairs = n_sync_pairs

        # Simple binary embedding (like RealCTM's parity backbone)
        self.embed = nn.Embedding(2, d_model)

        # Per-neuron NLM
        nlm_hidden = 16
        self.nlm_w1 = nn.Parameter(torch.empty(history_len, nlm_hidden, d_model).uniform_(
            -1/math.sqrt(history_len), 1/math.sqrt(history_len)
        ))
        self.nlm_b1 = nn.Parameter(torch.zeros(1, d_model, nlm_hidden))
        self.nlm_w2 = nn.Parameter(torch.empty(nlm_hidden, 1, d_model).uniform_(
            -1/math.sqrt(nlm_hidden), 1/math.sqrt(nlm_hidden)
        ))
        self.nlm_b2 = nn.Parameter(torch.zeros(1, d_model))

        # Sync pairs
        self.register_buffer('sync_left', torch.randint(0, d_model, (n_sync_pairs,)))
        self.register_buffer('sync_right', torch.randint(0, d_model, (n_sync_pairs,)))
        self.decay_logit = nn.Parameter(torch.zeros(n_sync_pairs))

        # Cross attention from sync
        self.sync_to_query = nn.Linear(n_sync_pairs, d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, 4, batch_first=True)

        # Synapse
        self.synapse = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model)
        )

        # Output from sync (binary classification)
        self.sync_to_output = nn.Linear(n_sync_pairs, 2)

        # Initial state
        self.init_state = nn.Parameter(torch.randn(d_model) * 0.02)
        self.init_history = nn.Parameter(torch.randn(d_model, history_len) * 0.02)

        self.name = f"SimpleParity(iter={n_iterations},d={d_model})"

    def compute_nlm(self, history):
        h = torch.einsum('bdh,hod->bdo', history, self.nlm_w1) + self.nlm_b1
        h = F.gelu(h)
        out = torch.einsum('bdo,oid->bdi', h, self.nlm_w2) + self.nlm_b2.unsqueeze(-1)
        return out.squeeze(-1)

    def compute_sync(self, state, prev_alpha, prev_beta):
        B = state.shape[0]
        device = state.device

        left = state[:, self.sync_left]
        right = state[:, self.sync_right]
        pairwise = left * right

        decay = torch.sigmoid(self.decay_logit).unsqueeze(0)

        if prev_alpha is None:
            alpha = pairwise
            beta = torch.ones(B, self.n_sync_pairs, device=device)
        else:
            alpha = decay * prev_alpha + pairwise
            beta = decay * prev_beta + 1

        sync = alpha / (torch.sqrt(beta) + 1e-6)
        return sync, alpha, beta

    def forward(self, x, track=False):
        """
        x: (B, L) binary tensor (0s and 1s)
        Returns: (B, 2) logits for even/odd
        """
        B, L = x.shape
        device = x.device

        # Embed input
        kv = self.embed(x)  # (B, L, d_model)

        # Initialize
        state = self.init_state.unsqueeze(0).expand(B, -1).clone()
        history = self.init_history.unsqueeze(0).expand(B, -1, -1).clone()

        sync_alpha = None
        sync_beta = None

        tracking = {'alpha': [], 'sync': []} if track else None

        for iter_idx in range(self.n_iterations):
            # 1. Compute sync
            sync, sync_alpha, sync_beta = self.compute_sync(state, sync_alpha, sync_beta)

            # 2. Cross-attention from sync
            q = self.sync_to_query(sync).unsqueeze(1)
            attn_out, _ = self.cross_attn(q, kv, kv)
            attn_out = attn_out.squeeze(1)

            # 3. NLM
            nlm_out = self.compute_nlm(history)

            # 4. Synapse
            combined = torch.cat([attn_out, nlm_out], dim=-1)
            delta = self.synapse(combined)

            if track:
                with torch.no_grad():
                    alpha_val = F.cosine_similarity(state, delta).mean().item()
                    tracking['alpha'].append(alpha_val)
                    tracking['sync'].append(sync.abs().mean().item())

            # 5. Update
            state = state + delta
            history = torch.cat([history[:, :, 1:], state.unsqueeze(-1)], dim=-1)

        # Final sync for output
        final_sync, _, _ = self.compute_sync(state, sync_alpha, sync_beta)
        logits = self.sync_to_output(final_sync)  # (B, 2)

        info = {'tracking': tracking} if track else {}
        return logits, info


class UniversalCTMv4(nn.Module):
    """
    Version 4: Hybrid - global accumulation + per-position output.

    Key insight: RealCTM works on parity because it has GLOBAL state.
    But for reversal, we need PER-POSITION output.

    Solution: Global accumulation for "thinking", then use it to
    modulate per-position attention for output.
    """

    def __init__(self, vocab_size: int, d_model: int = 128,
                 n_iterations: int = 30, history_len: int = 16,
                 n_heads: int = 4):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_iterations = n_iterations
        self.history_len = history_len

        # Input
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(512, d_model)
        self.kv_proj = nn.Linear(d_model, d_model)

        # Per-neuron weights for NLM
        nlm_hidden = 32
        self.nlm_w1 = nn.Parameter(torch.empty(history_len, nlm_hidden, d_model).uniform_(
            -1/math.sqrt(history_len), 1/math.sqrt(history_len)
        ))
        self.nlm_b1 = nn.Parameter(torch.zeros(1, d_model, nlm_hidden))
        self.nlm_w2 = nn.Parameter(torch.empty(nlm_hidden, 1, d_model).uniform_(
            -1/math.sqrt(nlm_hidden), 1/math.sqrt(nlm_hidden)
        ))
        self.nlm_b2 = nn.Parameter(torch.zeros(1, d_model))

        # Cross attention
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.q_proj = nn.Linear(d_model, d_model)

        # Synapse
        self.synapse = nn.Sequential(
            nn.Linear(d_model * 2, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model)
        )

        # Accumulator
        self.acc_proj = nn.Linear(d_model, d_model)
        self.decay_logit = nn.Parameter(torch.zeros(d_model))

        # Output attention: use accumulated state to query for per-position output
        self.output_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.output_q_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, vocab_size)

        # Initial state
        self.init_state = nn.Parameter(torch.randn(d_model) * 0.02)
        self.init_history = nn.Parameter(torch.randn(d_model, history_len) * 0.02)

        self.name = f"UniversalCTMv4(iter={n_iterations},d={d_model})"

    def compute_nlm(self, history: torch.Tensor) -> torch.Tensor:
        h = torch.einsum('bdh,hod->bdo', history, self.nlm_w1) + self.nlm_b1
        h = F.gelu(h)
        out = torch.einsum('bdo,oid->bdi', h, self.nlm_w2) + self.nlm_b2.unsqueeze(-1)
        return out.squeeze(-1)

    def forward(self, x: torch.Tensor, track: bool = False) -> Tuple[torch.Tensor, dict]:
        B, T = x.shape
        device = x.device

        # Embed
        positions = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
        embedded = self.embed(x) + self.pos_embed(positions)
        kv = self.kv_proj(embedded)

        # Global state
        state = self.init_state.unsqueeze(0).expand(B, -1).clone()
        history = self.init_history.unsqueeze(0).expand(B, -1, -1).clone()

        # Accumulator
        acc = torch.zeros(B, self.d_model, device=device)
        acc_count = torch.ones(B, self.d_model, device=device)
        decay = torch.sigmoid(self.decay_logit).unsqueeze(0)

        tracking = {'alpha': []} if track else None

        for iter_idx in range(self.n_iterations):
            # 1. Cross-attention
            q = self.q_proj(state).unsqueeze(1)
            attn_out, _ = self.cross_attn(q, kv, kv)
            attn_out = attn_out.squeeze(1)

            # 2. NLM
            nlm_out = self.compute_nlm(history)

            # 3. Synapse
            combined = torch.cat([attn_out, nlm_out], dim=-1)
            delta = self.synapse(combined)

            if track:
                with torch.no_grad():
                    alpha_val = F.cosine_similarity(state, delta).mean().item()
                    tracking['alpha'].append(alpha_val)

            # 4. Update
            state = state + delta
            history = torch.cat([history[:, :, 1:], state.unsqueeze(-1)], dim=-1)

            # 5. Accumulate
            contrib = self.acc_proj(state)
            acc = decay * acc + contrib
            acc_count = decay * acc_count + 1

        # Normalized accumulator
        accumulator = acc / (torch.sqrt(acc_count) + 1e-6)

        # Output: use accumulator to query input for per-position output
        out_q = self.output_q_proj(accumulator).unsqueeze(1).expand(-1, T, -1)
        out_attn, _ = self.output_attn(out_q, kv, kv)
        logits = self.output_proj(out_attn)

        info = {'tracking': tracking} if track else {}
        return logits, info


# =============================================================================
# BASELINE: Standard Transformer (for comparison)
# =============================================================================

class StandardTransformer(nn.Module):
    """Standard transformer for comparison."""

    def __init__(self, vocab_size: int, d_model: int = 128, n_layers: int = 4, n_heads: int = 4):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(512, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model*4,
            batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.output = nn.Linear(d_model, vocab_size)
        self.name = f"Transformer(L={n_layers},d={d_model})"

    def forward(self, x: torch.Tensor, track: bool = False) -> Tuple[torch.Tensor, dict]:
        B, T = x.shape
        positions = torch.arange(T, device=x.device).unsqueeze(0).expand(B, -1)
        h = self.embed(x) + self.pos_embed(positions)
        h = self.encoder(h)
        return self.output(h), {}


# =============================================================================
# DATA GENERATION
# =============================================================================

PAD, BOS, EOS, SEP = 0, 1, 2, 3
ZERO, ONE = 4, 5
EVEN, ODD = 6, 7


def make_parity_data(n_samples: int, length: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate parity task data."""
    data = []
    labels = []

    for _ in range(n_samples):
        bits = [random.choice([ZERO, ONE]) for _ in range(length)]
        count_ones = sum(1 for b in bits if b == ONE)
        parity = EVEN if count_ones % 2 == 0 else ODD

        seq = [BOS] + bits + [SEP]
        target = [PAD] * (length + 1) + [parity]

        data.append(seq)
        labels.append(target)

    return torch.tensor(data), torch.tensor(labels)


def make_simple_parity_data(n_samples: int, length: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Simple parity: just bits (0/1), output is 0 (even) or 1 (odd).
    This matches RealCTM's parity task more closely.
    """
    data = []
    labels = []

    for _ in range(n_samples):
        # Just 0s and 1s
        bits = [random.choice([0, 1]) for _ in range(length)]
        count_ones = sum(bits)
        parity = count_ones % 2  # 0 = even, 1 = odd

        data.append(bits)
        labels.append(parity)

    return torch.tensor(data), torch.tensor(labels)


def make_reversal_data(n_samples: int, length: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate reversal task data."""
    data = []
    labels = []

    for _ in range(n_samples):
        # Use tokens 4-13 for content
        content = [random.randint(4, 13) for _ in range(length)]
        reversed_content = content[::-1]

        seq = [BOS] + content + [SEP]
        target = [PAD] * (length + 1) + [SEP] + reversed_content + [EOS]

        # Pad to same length
        max_len = len(target)
        seq = seq + [PAD] * (max_len - len(seq))

        data.append(seq)
        labels.append(target)

    return torch.tensor(data), torch.tensor(labels)


# =============================================================================
# TRAINING & EVALUATION
# =============================================================================

def train_epoch(model, data, labels, optimizer, task='parity'):
    model.train()

    logits, _ = model(data)

    if task == 'parity':
        # Only care about the parity position
        parity_pos = -1
        loss = F.cross_entropy(logits[:, parity_pos, :], labels[:, parity_pos])
    else:
        # Full sequence loss for reversal
        mask = labels != PAD
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=PAD
        )

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    return loss.item()


def evaluate(model, data, labels, task='parity'):
    model.eval()
    with torch.no_grad():
        logits, info = model(data, track=True)

        if task == 'parity':
            preds = logits[:, -1, :].argmax(dim=-1)
            targets = labels[:, -1]
            acc = (preds == targets).float().mean().item()
        else:
            # For reversal, check after SEP
            sep_pos = (data == SEP).float().argmax(dim=1)
            correct = 0
            total = 0
            for i in range(len(data)):
                start = sep_pos[i].item() + 1
                pred = logits[i, start:, :].argmax(dim=-1)
                tgt = labels[i, start:]
                mask = tgt != PAD
                if mask.sum() > 0:
                    correct += (pred[mask] == tgt[mask]).sum().item()
                    total += mask.sum().item()
            acc = correct / total if total > 0 else 0

    return acc, info


def run_experiment(model_class, model_kwargs, task='parity', n_epochs=200,
                   train_length=10, test_lengths=[8, 10, 12]):
    """Run full training and evaluation."""

    print(f"\n{'='*60}")
    print(f"Training {model_class.__name__} on {task}")
    print(f"{'='*60}")

    # Create model
    model = model_class(**model_kwargs).to(DEVICE)
    print(f"Model: {model.name}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Generate data
    if task == 'parity':
        train_data, train_labels = make_parity_data(1000, train_length)
    else:
        train_data, train_labels = make_reversal_data(1000, train_length)

    train_data = train_data.to(DEVICE)
    train_labels = train_labels.to(DEVICE)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)

    # Training
    best_acc = 0
    for epoch in range(n_epochs):
        loss = train_epoch(model, train_data, train_labels, optimizer, task)
        scheduler.step()

        if (epoch + 1) % 20 == 0:
            acc, _ = evaluate(model, train_data[:200], train_labels[:200], task)
            print(f"Epoch {epoch+1:3d}: loss={loss:.4f}, acc={acc:.1%}")
            best_acc = max(best_acc, acc)

    # Test on different lengths
    print(f"\nLength generalization:")
    results = {}
    for length in test_lengths:
        if task == 'parity':
            test_data, test_labels = make_parity_data(500, length)
        else:
            test_data, test_labels = make_reversal_data(500, length)

        test_data = test_data.to(DEVICE)
        test_labels = test_labels.to(DEVICE)

        acc, info = evaluate(model, test_data, test_labels, task)
        results[length] = acc
        print(f"  Length {length}: {acc:.1%}")

        # Print phase info for parity
        if task == 'parity' and 'tracking' in info and info['tracking']:
            alphas = info['tracking'].get('alpha', [])
            if alphas:
                print(f"    Alpha trajectory: ", end='')
                step = max(1, len(alphas) // 5)
                for i in range(0, len(alphas), step):
                    if isinstance(alphas[i], tuple):
                        print(f"{alphas[i][1]:+.2f} ", end='')
                    else:
                        print(f"{alphas[i]:+.2f} ", end='')
                print()

    return model, results


# =============================================================================
# MAIN
# =============================================================================

def run_simple_parity_experiment():
    """Test SimpleParity model on raw binary parity task."""
    print("\n" + "="*60)
    print("SIMPLE PARITY EXPERIMENT")
    print("="*60)

    model = SimpleParity(d_model=64, n_iterations=30).to(DEVICE)
    print(f"Model: {model.name}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Generate simple parity data
    train_data, train_labels = make_simple_parity_data(2000, length=10)
    train_data = train_data.to(DEVICE)
    train_labels = train_labels.to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 300)

    for epoch in range(300):
        model.train()
        logits, _ = model(train_data)
        loss = F.cross_entropy(logits, train_labels)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if (epoch + 1) % 30 == 0:
            model.eval()
            with torch.no_grad():
                pred = logits.argmax(dim=-1)
                acc = (pred == train_labels).float().mean().item()
                _, info = model(train_data[:10], track=True)
                alphas = info['tracking']['alpha']
                print(f"Epoch {epoch+1:3d}: loss={loss:.4f}, acc={acc:.1%}")
                print(f"  Alpha: {alphas[0]:+.2f} → {alphas[9]:+.2f} → {alphas[19]:+.2f} → {alphas[-1]:+.2f}")

    # Test on different lengths
    print("\nLength generalization:")
    model.eval()
    for length in [8, 10, 12, 15, 20]:
        test_data, test_labels = make_simple_parity_data(500, length=length)
        test_data = test_data.to(DEVICE)
        test_labels = test_labels.to(DEVICE)

        with torch.no_grad():
            logits, _ = model(test_data)
            pred = logits.argmax(dim=-1)
            acc = (pred == test_labels).float().mean().item()
            print(f"  Length {length:2d}: {acc:.1%}")

    return model


if __name__ == "__main__":
    # First run simple parity experiment
    simple_model = run_simple_parity_experiment()

    print("\n" + "="*60)
    print("UNIVERSAL CTM EXPERIMENTS")
    print("="*60)

    vocab_size = 14  # PAD, BOS, EOS, SEP, ZERO, ONE, EVEN, ODD + extra

    # Models to test
    models_to_test = [
        (StandardTransformer, {'vocab_size': vocab_size, 'd_model': 128, 'n_layers': 4}),
        (UniversalCTMv5, {'vocab_size': vocab_size, 'd_model': 128, 'n_iterations': 30}),
    ]

    all_results = {}

    # Test on PARITY (counting task - transformers fail)
    print("\n" + "="*60)
    print("TASK 1: PARITY (counting - transformers should fail)")
    print("="*60)

    for model_class, kwargs in models_to_test:
        model, results = run_experiment(
            model_class, kwargs,
            task='parity',
            n_epochs=200,
            train_length=10,
            test_lengths=[8, 10, 12]
        )
        all_results[f"{model_class.__name__}_parity"] = results

    # Skip reversal for global-state models (they can't do per-position output)
    # Test on REVERSAL only for Transformer
    print("\n" + "="*60)
    print("TASK 2: REVERSAL (transformer only)")
    print("="*60)

    for model_class, kwargs in models_to_test:
        if 'Transformer' in model_class.__name__:
            model, results = run_experiment(
                model_class, kwargs,
                task='reversal',
                n_epochs=200,
                train_length=10,
                test_lengths=[8, 10, 12]
            )
            all_results[f"{model_class.__name__}_reversal"] = results

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"\n{'Model':<30} {'Parity@10':>12} {'Reversal@10':>12}")
    print("-"*56)

    for model_class, _ in models_to_test:
        name = model_class.__name__
        parity = all_results.get(f"{name}_parity", {}).get(10, 0)
        reversal = all_results.get(f"{name}_reversal", {}).get(10, 0)
        print(f"{name:<30} {parity:>11.1%} {reversal:>11.1%}")
