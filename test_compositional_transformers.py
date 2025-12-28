"""
Language Model Test for Compositional Hypothesis

Two models:
1. StandardTransformerLM - with residuals (baseline)
2. CompositionalTransformerLM - no residuals, early exit, confidence heads

Word-level tokenization on physics corpus.
Smaller hidden_dim to force generalization over memorization.

Measure:
- Perplexity (learning)
- Layer-wise perplexity (composition)
- Early exit distribution
- Confidence calibration
- Representation geometry
- Forgetting
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import os
import sys
from collections import defaultdict

# Load corpora from txt files
def load_corpus(path):
    with open(path, 'r') as f:
        return [line.strip() for line in f if line.strip()]

CORPUS_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_CORPUS = load_corpus(os.path.join(CORPUS_DIR, 'wikitext_train.txt'))
VAL_CORPUS = load_corpus(os.path.join(CORPUS_DIR, 'wikitext_val.txt'))


# ============================================================
# TOKENIZER
# ============================================================
class WordTokenizer:
    """Word-level tokenizer."""

    def __init__(self):
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.vocab_size = 0

        # Special tokens
        self.pad_token = '<PAD>'
        self.bos_token = '<BOS>'
        self.eos_token = '<EOS>'
        self.unk_token = '<UNK>'

    def fit(self, texts, max_vocab=3000):
        """Build vocabulary from texts, keeping only top max_vocab words."""
        import re
        from collections import Counter

        # Count word frequencies
        word_counts = Counter()
        for text in texts:
            tokens = re.findall(r'\w+|[^\w\s]', text.lower())
            word_counts.update(tokens)

        # Keep only top max_vocab words
        top_words = [word for word, count in word_counts.most_common(max_vocab)]

        # Add special tokens first
        self.word_to_idx[self.pad_token] = 0
        self.word_to_idx[self.bos_token] = 1
        self.word_to_idx[self.eos_token] = 2
        self.word_to_idx[self.unk_token] = 3

        # Add top words
        for i, word in enumerate(top_words):
            self.word_to_idx[word] = i + 4

        # Reverse mapping
        self.idx_to_word = {v: k for k, v in self.word_to_idx.items()}
        self.vocab_size = len(self.word_to_idx)

        # Calculate coverage
        total_tokens = sum(word_counts.values())
        covered_tokens = sum(word_counts[w] for w in top_words)
        coverage = covered_tokens / total_tokens

        print(f"Vocabulary size: {self.vocab_size}")
        print(f"Coverage: {coverage:.1%} of tokens")
        print(f"Sample words: {top_words[:10]}")

    def encode(self, text, add_special=True):
        """Encode text to token ids."""
        import re
        ids = []
        if add_special:
            ids.append(self.word_to_idx[self.bos_token])
        tokens = re.findall(r'\w+|[^\w\s]', text.lower())
        for token in tokens:
            if token in self.word_to_idx:
                ids.append(self.word_to_idx[token])
            else:
                ids.append(self.word_to_idx[self.unk_token])
        if add_special:
            ids.append(self.word_to_idx[self.eos_token])
        return ids

    def decode(self, ids):
        """Decode token ids to text."""
        words = []
        for idx in ids:
            if idx in self.idx_to_word:
                word = self.idx_to_word[idx]
                if word not in [self.pad_token, self.bos_token, self.eos_token]:
                    words.append(word)
        # Join with spaces, but handle punctuation
        result = []
        for i, word in enumerate(words):
            if word in '.,;:!?\'")-]':
                result.append(word)
            elif i > 0 and words[i-1] in '(\'"[-':
                result.append(word)
            else:
                if result:
                    result.append(' ')
                result.append(word)
        return ''.join(result)

    @property
    def pad_id(self):
        return self.word_to_idx[self.pad_token]

    @property
    def bos_id(self):
        return self.word_to_idx[self.bos_token]

    @property
    def eos_id(self):
        return self.word_to_idx[self.eos_token]

    # Aliases for compatibility
    @property
    def char_to_idx(self):
        return self.word_to_idx

    @property
    def idx_to_char(self):
        return self.idx_to_word


class CharTokenizer:
    """Character-level tokenizer."""

    def __init__(self):
        self.char_to_idx = {}
        self.idx_to_char = {}
        self.vocab_size = 0

        # Special tokens
        self.pad_token = '<PAD>'
        self.bos_token = '<BOS>'
        self.eos_token = '<EOS>'

    def fit(self, texts):
        """Build vocabulary from texts."""
        # Collect all unique characters
        chars = set()
        for text in texts:
            chars.update(text)

        # Sort for reproducibility
        chars = sorted(chars)

        # Add special tokens first
        self.char_to_idx[self.pad_token] = 0
        self.char_to_idx[self.bos_token] = 1
        self.char_to_idx[self.eos_token] = 2

        # Add characters
        for i, char in enumerate(chars):
            self.char_to_idx[char] = i + 3

        # Reverse mapping
        self.idx_to_char = {v: k for k, v in self.char_to_idx.items()}
        self.vocab_size = len(self.char_to_idx)

        print(f"Vocabulary size: {self.vocab_size}")
        print(f"Sample chars: {list(self.char_to_idx.keys())[3:13]}")

    def encode(self, text, add_special=True):
        """Encode text to token ids."""
        ids = []
        if add_special:
            ids.append(self.char_to_idx[self.bos_token])
        for char in text:
            if char in self.char_to_idx:
                ids.append(self.char_to_idx[char])
        if add_special:
            ids.append(self.char_to_idx[self.eos_token])
        return ids

    def decode(self, ids):
        """Decode token ids to text."""
        chars = []
        for idx in ids:
            if idx in self.idx_to_char:
                char = self.idx_to_char[idx]
                if char not in [self.pad_token, self.bos_token, self.eos_token]:
                    chars.append(char)
        return ''.join(chars)

    @property
    def pad_id(self):
        return self.char_to_idx[self.pad_token]

    @property
    def bos_id(self):
        return self.char_to_idx[self.bos_token]

    @property
    def eos_id(self):
        return self.char_to_idx[self.eos_token]


# ============================================================
# POSITIONAL ENCODING
# ============================================================
class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, hidden_dim, max_len=512):
        super().__init__()

        pe = torch.zeros(max_len, hidden_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_dim, 2).float() * (-math.log(10000.0) / hidden_dim))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))  # [1, max_len, hidden_dim]

    def forward(self, x):
        """Add positional encoding to input."""
        # x: [batch, seq_len, hidden_dim]
        return x + self.pe[:, :x.size(1), :]


# ============================================================
# CAUSAL ATTENTION
# ============================================================
class CausalSelfAttention(nn.Module):
    """Multi-head self-attention with causal mask."""

    def __init__(self, hidden_dim, n_heads, dropout=0.1):
        super().__init__()
        assert hidden_dim % n_heads == 0

        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads

        self.qkv = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        x: [batch, seq_len, hidden_dim]
        mask: [seq_len, seq_len] causal mask (True = attend, False = mask)
        """
        B, T, C = x.shape

        # Compute Q, K, V
        qkv = self.qkv(x)  # [B, T, 3*C]
        qkv = qkv.reshape(B, T, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, n_heads, T, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention scores
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # [B, n_heads, T, T]

        # Apply causal mask
        if mask is not None:
            scores = scores.masked_fill(~mask, float('-inf'))

        # Softmax and dropout
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        out = attn @ v  # [B, n_heads, T, head_dim]
        out = out.transpose(1, 2).reshape(B, T, C)  # [B, T, C]

        return self.proj(out)


# ============================================================
# TRANSFORMER BLOCKS
# ============================================================
class StandardTransformerBlock(nn.Module):
    """Transformer block WITH residual connections."""

    def __init__(self, hidden_dim, n_heads, dropout=0.1):
        super().__init__()
        self.attn = CausalSelfAttention(hidden_dim, n_heads, dropout)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.Dropout(dropout)
        )
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)

    def forward(self, x, mask=None):
        # Pre-norm with residual
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.mlp(self.ln2(x))
        return x


class CompositionalTransformerBlock(nn.Module):
    """Transformer block WITHOUT residual connections."""

    def __init__(self, hidden_dim, n_heads, dropout=0.1):
        super().__init__()
        self.attn = CausalSelfAttention(hidden_dim, n_heads, dropout)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.Dropout(dropout)
        )
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)

    def forward(self, x, mask=None):
        # NO RESIDUAL - strict composition
        x = self.attn(self.ln1(x), mask)
        x = self.mlp(self.ln2(x))
        return x


# ============================================================
# LANGUAGE MODELS
# ============================================================
class StandardTransformerLM(nn.Module):
    """Standard Transformer Language Model WITH residuals."""

    def __init__(self, vocab_size, hidden_dim, n_layers, n_heads, max_len=512, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        # Embeddings
        self.token_embed = nn.Embedding(vocab_size, hidden_dim)
        self.pos_encode = SinusoidalPositionalEncoding(hidden_dim, max_len)
        self.dropout = nn.Dropout(dropout)

        # Transformer blocks with residuals
        self.blocks = nn.ModuleList([
            StandardTransformerBlock(hidden_dim, n_heads, dropout)
            for _ in range(n_layers)
        ])

        self.ln_final = nn.LayerNorm(hidden_dim)

        # Output projection (tied with input embedding)
        self.output = nn.Linear(hidden_dim, vocab_size, bias=False)
        # Tie weights
        self.output.weight = self.token_embed.weight

    def get_causal_mask(self, seq_len, device):
        """Create causal mask: True where attention allowed."""
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool))
        return mask

    def forward(self, x, return_all=False):
        """
        x: [batch, seq_len] token ids
        return_all: if True, return outputs at each layer
        """
        B, T = x.shape

        # Embeddings
        h = self.token_embed(x)  # [B, T, hidden_dim]
        h = self.pos_encode(h)
        h = self.dropout(h)

        # Causal mask
        mask = self.get_causal_mask(T, x.device)

        all_logits = []
        all_hiddens = []

        # Transformer blocks
        for block in self.blocks:
            h = block(h, mask)
            all_hiddens.append(h.clone())

            if return_all:
                logits = self.output(self.ln_final(h))
                all_logits.append(logits)

        # Final output
        h = self.ln_final(h)
        logits = self.output(h)  # [B, T, vocab_size]

        if return_all:
            return logits, all_logits, all_hiddens
        return logits


class CompositionalTransformerLM(nn.Module):
    """Compositional Transformer Language Model WITHOUT residuals.

    Features:
    - No residual connections (strict composition)
    - Output head at each layer
    - Confidence head at each layer for early exit
    """

    def __init__(self, vocab_size, hidden_dim, n_layers, n_heads, max_len=512, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.vocab_size = vocab_size

        # Embeddings
        self.token_embed = nn.Embedding(vocab_size, hidden_dim)
        self.pos_encode = SinusoidalPositionalEncoding(hidden_dim, max_len)
        self.dropout = nn.Dropout(dropout)

        # Transformer blocks WITHOUT residuals
        self.blocks = nn.ModuleList([
            CompositionalTransformerBlock(hidden_dim, n_heads, dropout)
            for _ in range(n_layers)
        ])

        # Layer norm at each layer
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(n_layers)
        ])

        # Output head at each layer - INDEPENDENT (no weight tying)
        # Each layer learns its own output projection
        self.output_heads = nn.ModuleList([
            nn.Linear(hidden_dim, vocab_size)
            for _ in range(n_layers)
        ])

        # Confidence head at each layer (matches existing architecture)
        # Outputs probability that this layer's prediction is "good enough"
        self.confidence_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Sigmoid()
            )
            for _ in range(n_layers)
        ])

    def get_causal_mask(self, seq_len, device):
        """Create causal mask."""
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool))
        return mask

    def forward(self, x, return_all=False, exit_threshold=0.3):
        """
        x: [batch, seq_len] token ids
        return_all: if True, return outputs at each layer
        exit_threshold: confidence threshold for early exit (None = no early exit)
        """
        B, T = x.shape

        # Embeddings
        h = self.token_embed(x)  # [B, T, hidden_dim]
        h = self.pos_encode(h)
        h = self.dropout(h)

        # Causal mask
        mask = self.get_causal_mask(T, x.device)

        all_logits = []
        all_confidences = []
        all_hiddens = []
        exit_layers = torch.full((B,), self.n_layers - 1, dtype=torch.long, device=x.device)
        exited = torch.zeros(B, dtype=torch.bool, device=x.device)

        # Transformer blocks
        for i, (block, ln, head, conf_head) in enumerate(zip(
            self.blocks, self.layer_norms, self.output_heads, self.confidence_heads
        )):
            h = block(h, mask)  # NO RESIDUAL
            h_normed = ln(h)

            # Output at this layer
            logits = head(h_normed)  # [B, T, vocab_size]
            all_logits.append(logits)
            all_hiddens.append(h.clone())

            # Confidence at this layer (average over sequence)
            conf = conf_head(h_normed.mean(dim=1))  # [B, 1]
            all_confidences.append(conf.squeeze(-1))

            # Early exit logic (inference only)
            if not self.training and exit_threshold is not None:
                can_exit = (conf.squeeze(-1) > exit_threshold) & (~exited)
                exit_layers[can_exit] = i
                exited[can_exit] = True

                # If all samples have exited, stop computation
                if not return_all and exited.all():
                    break

        if return_all:
            return all_logits[-1], all_logits, all_confidences, all_hiddens, exit_layers

        # Return output at each sample's exit layer
        final_logits = torch.zeros(B, T, self.vocab_size, device=x.device)
        for i in range(self.n_layers):
            layer_mask = (exit_layers == i)
            if layer_mask.any() and i < len(all_logits):
                final_logits[layer_mask] = all_logits[i][layer_mask]

        return final_logits, exit_layers


# ============================================================
# CTM-INSPIRED TRANSFORMER
# ============================================================
class CTMInspiredTransformerLM(nn.Module):
    """CTM-Inspired Transformer LM.

    Key ideas from Continuous Thought Machine:
    1. Internal "thinking" ticks - iterate T times before outputting
    2. History of activations - neurons see their past states
    3. Synchronization as representation - how neurons fire together over time
    4. Certainty-based early exit

    We keep residuals for gradient stability.
    """

    def __init__(self, vocab_size, hidden_dim, n_layers, n_heads,
                 max_len=512, n_ticks=8, history_len=4, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.vocab_size = vocab_size
        self.n_ticks = n_ticks
        self.history_len = history_len

        # Embeddings
        self.token_embed = nn.Embedding(vocab_size, hidden_dim)
        self.pos_encode = SinusoidalPositionalEncoding(hidden_dim, max_len)
        self.dropout = nn.Dropout(dropout)

        # Transformer blocks WITH residuals (for stability)
        self.blocks = nn.ModuleList([
            StandardTransformerBlock(hidden_dim, n_heads, dropout)
            for _ in range(n_layers)
        ])

        # History processor (simplified neuron-level model)
        # Processes concatenated history to produce features
        self.history_processor = nn.Sequential(
            nn.Linear(hidden_dim * history_len, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Synchronization: we sample pairs of neurons and compute their correlation
        self.sync_dim = min(64, hidden_dim)  # number of sync pairs
        self.sync_proj = nn.Linear(self.sync_dim, hidden_dim)

        # Register random sync pairs (fixed after init)
        # Each pair (i, j) measures correlation between neurons i and j
        sync_pairs = torch.stack([
            torch.randint(0, hidden_dim, (self.sync_dim,)),
            torch.randint(0, hidden_dim, (self.sync_dim,))
        ], dim=1)
        self.register_buffer('sync_pairs', sync_pairs)

        self.ln_final = nn.LayerNorm(hidden_dim)

        # Output projection
        self.output = nn.Linear(hidden_dim, vocab_size)

        # Certainty head for early exit
        self.certainty_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def get_causal_mask(self, seq_len, device):
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool))
        return mask

    def compute_sync(self, history):
        """Compute synchronization from activation history.

        history: list of [B, T, D] tensors (past activations)
        Returns: [B, T, sync_dim] synchronization features
        """
        if len(history) < 2:
            B, T, D = history[0].shape
            return torch.zeros(B, T, self.sync_dim, device=history[0].device)

        # Stack history: [n_hist, B, T, D]
        H = torch.stack(history[-self.history_len:], dim=0)
        n_hist, B, T, D = H.shape

        # Compute sync for selected pairs
        sync_features = []
        for p in range(self.sync_dim):
            i, j = self.sync_pairs[p]
            # Get activations for neurons i and j over history
            h_i = H[:, :, :, i]  # [n_hist, B, T]
            h_j = H[:, :, :, j]  # [n_hist, B, T]
            # Correlation over history dimension (dot product normalized)
            sync = (h_i * h_j).mean(dim=0)  # [B, T]
            sync_features.append(sync)

        return torch.stack(sync_features, dim=-1)  # [B, T, sync_dim]

    def forward(self, x, return_all=False, certainty_threshold=0.8):
        """Forward pass with internal thinking ticks."""
        B, T = x.shape
        device = x.device

        # Initial embedding
        h = self.token_embed(x)
        h = self.pos_encode(h)
        h = self.dropout(h)

        mask = self.get_causal_mask(T, device)

        # History of activations
        history = [h.clone()]

        all_logits = []
        all_certainties = []
        all_hiddens = []
        exit_tick = self.n_ticks - 1

        # Thinking ticks
        for tick in range(self.n_ticks):
            # Process through transformer blocks (with residuals inside)
            for block in self.blocks:
                h = block(h, mask)

            # Build history input (pad if needed)
            if len(history) >= self.history_len:
                recent = history[-self.history_len:]
            else:
                padding = [history[0]] * (self.history_len - len(history))
                recent = padding + history

            # History features
            hist_concat = torch.cat(recent, dim=-1)  # [B, T, D*history_len]
            hist_features = self.history_processor(hist_concat)  # [B, T, D]

            # Synchronization features
            sync_features = self.compute_sync(history)  # [B, T, sync_dim]
            sync_embed = self.sync_proj(sync_features)  # [B, T, D]

            # Combine: current + history + sync (like a residual)
            h = h + 0.1 * hist_features + 0.1 * sync_embed

            # Update history
            history.append(h.clone())
            if len(history) > self.history_len + 2:
                history = history[-(self.history_len + 2):]  # Keep memory bounded

            # Compute output at this tick
            h_norm = self.ln_final(h)
            logits = self.output(h_norm)
            certainty = self.certainty_head(h_norm.mean(dim=1))  # [B, 1]

            all_logits.append(logits)
            all_certainties.append(certainty.squeeze(-1))
            all_hiddens.append(h.clone())

            # Early exit (inference only)
            if not self.training and certainty_threshold is not None:
                if certainty.mean() > certainty_threshold:
                    exit_tick = tick
                    break

        if return_all:
            return all_logits[-1], all_logits, all_certainties, all_hiddens, exit_tick

        return all_logits[-1], exit_tick


# ============================================================
# DATA PREPARATION
# ============================================================
def prepare_data(corpus, tokenizer, max_len=128):
    """Prepare training data from corpus."""
    all_sequences = []

    for text in corpus:
        ids = tokenizer.encode(text, add_special=True)

        # Truncate if too long
        if len(ids) > max_len:
            ids = ids[:max_len]

        all_sequences.append(ids)

    return all_sequences


def collate_batch(sequences, pad_id, max_len=None):
    """Collate sequences into padded batch."""
    if max_len is None:
        max_len = max(len(s) for s in sequences)

    batch = torch.full((len(sequences), max_len), pad_id, dtype=torch.long)

    for i, seq in enumerate(sequences):
        length = min(len(seq), max_len)
        batch[i, :length] = torch.tensor(seq[:length])

    return batch


# ============================================================
# TRAINING
# ============================================================
def compute_loss(logits, targets, pad_id):
    """Compute cross-entropy loss ignoring padding."""
    # logits: [B, T, vocab_size]
    # targets: [B, T]

    B, T, V = logits.shape

    # Flatten
    logits_flat = logits.reshape(-1, V)  # [B*T, V]
    targets_flat = targets.reshape(-1)  # [B*T]

    # Cross entropy (ignoring padding)
    loss = F.cross_entropy(logits_flat, targets_flat, ignore_index=pad_id, reduction='mean')

    return loss


def compute_perplexity(logits, targets, pad_id):
    """Compute perplexity."""
    loss = compute_loss(logits, targets, pad_id)
    return torch.exp(loss).item()


def train_epoch(model, sequences, tokenizer, optimizer, batch_size=32, max_len=128):
    """Train for one epoch."""
    model.train()
    device = next(model.parameters()).device

    # Shuffle sequences
    indices = torch.randperm(len(sequences)).tolist()

    total_loss = 0
    n_batches = 0

    for i in range(0, len(sequences), batch_size):
        batch_indices = indices[i:i+batch_size]
        batch_seqs = [sequences[idx] for idx in batch_indices]

        # Collate
        batch = collate_batch(batch_seqs, tokenizer.pad_id, max_len).to(device)

        # Input: all but last token, Target: all but first token
        x = batch[:, :-1]
        y = batch[:, 1:]

        optimizer.zero_grad()

        # Forward
        if isinstance(model, CTMInspiredTransformerLM):
            # CTM: get all tick outputs, train on all of them
            logits, all_logits, all_cert, all_hiddens, exit_tick = model(x, return_all=True)

            # Loss on all ticks (encourages each tick to be useful)
            loss = 0
            for tick_logits in all_logits:
                loss += compute_loss(tick_logits, y, tokenizer.pad_id)
            loss = loss / len(all_logits)

            # Certainty loss: certainty should predict if prediction is correct
            for tick_idx, cert in enumerate(all_cert):
                with torch.no_grad():
                    pred = all_logits[tick_idx].argmax(dim=-1)
                    correct = (pred == y).float()
                    mask = (y != tokenizer.pad_id).float()
                    acc_per_sample = (correct * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)

                cert_loss = F.binary_cross_entropy(cert, acc_per_sample)
                loss = loss + 0.1 * cert_loss

        elif isinstance(model, CompositionalTransformerLM):
            logits, all_logits, all_conf, _, _ = model(x, return_all=True)

            # Loss on all layers (for training confidence)
            loss = 0
            for layer_logits in all_logits:
                loss += compute_loss(layer_logits, y, tokenizer.pad_id)
            loss = loss / len(all_logits)

            # Confidence loss: confidence should predict if prediction is correct
            with torch.no_grad():
                for layer_idx, (layer_logits, conf) in enumerate(zip(all_logits, all_conf)):
                    pred = layer_logits.argmax(dim=-1)  # [B, T]
                    correct = (pred == y).float()  # [B, T]
                    # Mask padding
                    mask = (y != tokenizer.pad_id).float()
                    # Accuracy per sample
                    acc_per_sample = (correct * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)  # [B]

            # Add confidence loss
            for layer_idx, conf in enumerate(all_conf):
                with torch.no_grad():
                    pred = all_logits[layer_idx].argmax(dim=-1)
                    correct = (pred == y).float()
                    mask = (y != tokenizer.pad_id).float()
                    acc_per_sample = (correct * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)

                conf_loss = F.binary_cross_entropy(conf, acc_per_sample)
                loss = loss + 0.1 * conf_loss
        else:
            logits = model(x)
            loss = compute_loss(logits, y, tokenizer.pad_id)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


def evaluate(model, sequences, tokenizer, batch_size=32, max_len=128):
    """Evaluate model."""
    model.eval()
    device = next(model.parameters()).device

    all_perplexities = []
    layer_perplexities = defaultdict(list)  # Also used for tick perplexities in CTM
    all_exit_layers = []  # Also used for exit ticks in CTM
    all_confidences = defaultdict(list)  # Also used for certainties in CTM

    with torch.no_grad():
        for i in range(0, len(sequences), batch_size):
            batch_seqs = sequences[i:i+batch_size]
            batch = collate_batch(batch_seqs, tokenizer.pad_id, max_len).to(device)

            x = batch[:, :-1]
            y = batch[:, 1:]

            if isinstance(model, CTMInspiredTransformerLM):
                logits, all_logits, all_cert, all_hiddens, exit_tick = model(x, return_all=True, certainty_threshold=None)

                # Perplexity at each tick
                for tick_idx, tick_logits in enumerate(all_logits):
                    ppl = compute_perplexity(tick_logits, y, tokenizer.pad_id)
                    layer_perplexities[tick_idx].append(ppl)

                # Certainty at each tick
                for tick_idx, cert in enumerate(all_cert):
                    all_confidences[tick_idx].extend(cert.tolist())

                # Final perplexity
                ppl = compute_perplexity(logits, y, tokenizer.pad_id)
                all_perplexities.append(ppl)

                # Exit ticks (with threshold)
                _, exit_tick = model(x, return_all=False, certainty_threshold=0.3)
                all_exit_layers.append(exit_tick)  # Single value per batch

            elif isinstance(model, CompositionalTransformerLM):
                logits, all_logits, all_conf, all_hiddens, exit_layers = model(x, return_all=True, exit_threshold=None)

                # Perplexity at each layer
                for layer_idx, layer_logits in enumerate(all_logits):
                    ppl = compute_perplexity(layer_logits, y, tokenizer.pad_id)
                    layer_perplexities[layer_idx].append(ppl)

                # Confidence at each layer
                for layer_idx, conf in enumerate(all_conf):
                    all_confidences[layer_idx].extend(conf.tolist())

                # Final perplexity
                ppl = compute_perplexity(logits, y, tokenizer.pad_id)
                all_perplexities.append(ppl)

                # Exit layers (with threshold) - return_all=False returns (logits, exit_layers)
                _, exit_layers = model(x, return_all=False, exit_threshold=0.3)
                all_exit_layers.extend(exit_layers.tolist())
            else:
                logits, all_logits, all_hiddens = model(x, return_all=True)

                # Perplexity at each layer
                for layer_idx, layer_logits in enumerate(all_logits):
                    ppl = compute_perplexity(layer_logits, y, tokenizer.pad_id)
                    layer_perplexities[layer_idx].append(ppl)

                # Final perplexity
                ppl = compute_perplexity(logits, y, tokenizer.pad_id)
                all_perplexities.append(ppl)

    results = {
        'perplexity': np.mean(all_perplexities),
        'layer_perplexity': {k: np.mean(v) for k, v in layer_perplexities.items()},
        'exit_layers': all_exit_layers,
        'confidences': {k: np.mean(v) for k, v in all_confidences.items()},
    }

    return results


# ============================================================
# GENERATION
# ============================================================
def generate(model, tokenizer, prompt, max_new=50, temperature=0.8, exit_threshold=0.3):
    """Generate text from prompt."""
    model.eval()
    # Get device from model
    device = next(model.parameters()).device
    ids = tokenizer.encode(prompt, add_special=True)[:-1]  # Remove EOS

    for _ in range(max_new):
        x = torch.tensor([ids], device=device)

        if isinstance(model, CTMInspiredTransformerLM):
            logits, _ = model(x, certainty_threshold=exit_threshold)
        elif isinstance(model, CompositionalTransformerLM):
            logits, _ = model(x, exit_threshold=exit_threshold)
        else:
            logits = model(x)

        # Get last token logits
        next_logits = logits[0, -1, :] / temperature
        probs = F.softmax(next_logits, dim=-1)
        next_id = torch.multinomial(probs, 1).item()

        if next_id == tokenizer.eos_id:
            break

        ids.append(next_id)

    return tokenizer.decode(ids)


# ============================================================
# MAIN
# ============================================================
def main():
    print("="*80)
    print("LANGUAGE MODEL TEST - CTM-INSPIRED vs STANDARD")
    print("="*80)

    # Config - small model to force generalization
    # ~100K params for ~450K training tokens = 0.24 params/token
    HIDDEN_DIM = 16
    N_LAYERS = 4
    N_HEADS = 4
    MAX_LEN = 64  # Shorter sequences for efficiency
    BATCH_SIZE = 32
    N_EPOCHS = 50  # Fewer epochs needed with more data
    LR = 1e-3
    MAX_VOCAB = 3000
    # CTM-specific
    N_TICKS = 8  # Internal thinking iterations
    HISTORY_LEN = 4  # How many past states to remember

    # Prepare tokenizer (word-level, limited vocab)
    print("\n1. Preparing tokenizer...")
    tokenizer = WordTokenizer()
    tokenizer.fit(TRAIN_CORPUS + VAL_CORPUS, max_vocab=MAX_VOCAB)

    # Prepare data
    print("\n2. Preparing data...")
    train_seqs = prepare_data(TRAIN_CORPUS, tokenizer, MAX_LEN)
    val_seqs = prepare_data(VAL_CORPUS, tokenizer, MAX_LEN)

    train_tokens = sum(len(s) for s in train_seqs)
    val_tokens = sum(len(s) for s in val_seqs)
    print(f"   Train: {len(train_seqs)} sequences, {train_tokens:,} tokens")
    print(f"   Val: {len(val_seqs)} sequences, {val_tokens:,} tokens")

    # Device setup
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("\n   Using MPS (Apple Silicon GPU)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("\n   Using CUDA")
    else:
        device = torch.device("cpu")
        print("\n   Using CPU")

    # Create models
    print("\n3. Creating models...")

    std_model = StandardTransformerLM(
        vocab_size=tokenizer.vocab_size,
        hidden_dim=HIDDEN_DIM,
        n_layers=N_LAYERS,
        n_heads=N_HEADS,
        max_len=MAX_LEN
    ).to(device)

    ctm_model = CTMInspiredTransformerLM(
        vocab_size=tokenizer.vocab_size,
        hidden_dim=HIDDEN_DIM,
        n_layers=N_LAYERS,
        n_heads=N_HEADS,
        max_len=MAX_LEN,
        n_ticks=N_TICKS,
        history_len=HISTORY_LEN
    ).to(device)

    std_params = sum(p.numel() for p in std_model.parameters())
    ctm_params = sum(p.numel() for p in ctm_model.parameters())
    print(f"   StandardTransformerLM: {std_params:,} parameters")
    print(f"   CTMInspiredTransformerLM: {ctm_params:,} parameters (n_ticks={N_TICKS}, history_len={HISTORY_LEN})")

    # Optimizers
    std_optimizer = torch.optim.AdamW(std_model.parameters(), lr=LR)
    ctm_optimizer = torch.optim.AdamW(ctm_model.parameters(), lr=LR)

    # Training
    print("\n4. Training...")
    print("-"*60)

    import time

    for epoch in range(N_EPOCHS):
        epoch_start = time.time()

        # Train standard model
        std_loss = train_epoch(std_model, train_seqs, tokenizer, std_optimizer, BATCH_SIZE, MAX_LEN)

        # Train CTM model
        ctm_loss = train_epoch(ctm_model, train_seqs, tokenizer, ctm_optimizer, BATCH_SIZE, MAX_LEN)

        epoch_time = time.time() - epoch_start
        tokens_per_sec = (2 * train_tokens) / epoch_time  # 2x because both models

        # Show progress every epoch
        print(f"Epoch {epoch+1:3d}/{N_EPOCHS} | std_loss={std_loss:.3f} ctm_loss={ctm_loss:.3f} | {epoch_time:.1f}s | {tokens_per_sec/1000:.1f}K tok/s", end="")

        if (epoch + 1) % 10 == 0:
            # Evaluate
            std_results = evaluate(std_model, val_seqs, tokenizer, BATCH_SIZE, MAX_LEN)
            ctm_results = evaluate(ctm_model, val_seqs, tokenizer, BATCH_SIZE, MAX_LEN)

            print(f" | val_ppl: std={std_results['perplexity']:.1f} ctm={ctm_results['perplexity']:.1f}")

            # Layer-wise perplexity (Standard)
            print(f"  Standard layer PPL:     ", end="")
            for layer, ppl in std_results['layer_perplexity'].items():
                print(f"L{layer}={ppl:.1f} ", end="")
            print()

            # Tick-wise perplexity (CTM)
            print(f"  CTM tick PPL:           ", end="")
            for tick, ppl in ctm_results['layer_perplexity'].items():
                print(f"T{tick}={ppl:.1f} ", end="")
            print()

            # Certainty (CTM)
            if ctm_results['confidences']:
                print(f"  CTM certainty:          ", end="")
                for tick, cert in ctm_results['confidences'].items():
                    print(f"T{tick}={cert:.3f} ", end="")
                print()

            # Exit ticks (CTM)
            if ctm_results['exit_layers']:
                print(f"  CTM exit ticks: {ctm_results['exit_layers']}")
        else:
            print()  # Just newline for non-eval epochs

    # Final evaluation
    print("\n" + "="*80)
    print("FINAL EVALUATION")
    print("="*80)

    std_results = evaluate(std_model, val_seqs, tokenizer, BATCH_SIZE, MAX_LEN)
    ctm_results = evaluate(ctm_model, val_seqs, tokenizer, BATCH_SIZE, MAX_LEN)

    print(f"\nStandard Transformer:")
    print(f"  Perplexity: {std_results['perplexity']:.2f}")
    print(f"  Layer-wise PPL: {std_results['layer_perplexity']}")

    print(f"\nCTM-Inspired Transformer:")
    print(f"  Perplexity: {ctm_results['perplexity']:.2f}")
    print(f"  Tick-wise PPL: {ctm_results['layer_perplexity']}")
    print(f"  Certainty: {ctm_results['confidences']}")

    if ctm_results['exit_layers']:
        print(f"  Exit ticks: {ctm_results['exit_layers']}")

    # Sample generation
    print("\n" + "="*80)
    print("SAMPLE GENERATION")
    print("="*80)

    prompts = ["When you heat", "Sound travels", "Objects float"]

    for prompt in prompts:
        print(f"\nPrompt: '{prompt}'")
        print(f"Standard: {generate(std_model, tokenizer, prompt)}")
        print(f"CTM:      {generate(ctm_model, tokenizer, prompt)}")

    # ============================================================
    # SAVE MODELS
    # ============================================================
    print("\n" + "="*80)
    print("SAVING MODELS")
    print("="*80)

    save_dir = "./lm_checkpoints"
    os.makedirs(save_dir, exist_ok=True)

    # Save tokenizer
    import pickle
    tokenizer_path = f"{save_dir}/tokenizer.pkl"
    with open(tokenizer_path, 'wb') as f:
        pickle.dump(tokenizer, f)
    print(f"  Saved tokenizer to {tokenizer_path}")

    # Save standard model
    std_path = f"{save_dir}/standard_transformer_lm.pt"
    torch.save({
        'model_state': std_model.state_dict(),
        'config': {
            'vocab_size': tokenizer.vocab_size,
            'hidden_dim': HIDDEN_DIM,
            'n_layers': N_LAYERS,
            'n_heads': N_HEADS,
            'max_len': MAX_LEN,
        },
        'final_perplexity': std_results['perplexity'],
        'layer_perplexity': std_results['layer_perplexity'],
    }, std_path)
    print(f"  Saved StandardTransformerLM to {std_path}")

    # Save CTM model
    ctm_path = f"{save_dir}/ctm_transformer_lm.pt"
    torch.save({
        'model_state': ctm_model.state_dict(),
        'config': {
            'vocab_size': tokenizer.vocab_size,
            'hidden_dim': HIDDEN_DIM,
            'n_layers': N_LAYERS,
            'n_heads': N_HEADS,
            'max_len': MAX_LEN,
            'n_ticks': N_TICKS,
            'history_len': HISTORY_LEN,
        },
        'final_perplexity': ctm_results['perplexity'],
        'tick_perplexity': ctm_results['layer_perplexity'],
        'certainties': ctm_results['confidences'],
    }, ctm_path)
    print(f"  Saved CTMInspiredTransformerLM to {ctm_path}")

    # Save training sequences for reproducibility
    data_path = f"{save_dir}/data.pkl"
    with open(data_path, 'wb') as f:
        pickle.dump({
            'train_seqs': train_seqs,
            'val_seqs': val_seqs,
        }, f)
    print(f"  Saved data splits to {data_path}")

    print("\n" + "="*80)
    print("DONE")
    print("="*80)


def load_models(save_dir="./lm_checkpoints"):
    """Load saved models and tokenizer."""
    import pickle

    # Load tokenizer
    with open(f"{save_dir}/tokenizer.pkl", 'rb') as f:
        tokenizer = pickle.load(f)

    # Load standard model
    std_checkpoint = torch.load(f"{save_dir}/standard_transformer_lm.pt", map_location='cpu', weights_only=False)
    cfg = std_checkpoint['config']
    std_model = StandardTransformerLM(
        vocab_size=cfg['vocab_size'],
        hidden_dim=cfg['hidden_dim'],
        n_layers=cfg['n_layers'],
        n_heads=cfg['n_heads'],
        max_len=cfg['max_len']
    )
    std_model.load_state_dict(std_checkpoint['model_state'])

    # Load compositional model
    comp_checkpoint = torch.load(f"{save_dir}/compositional_transformer_lm.pt", map_location='cpu', weights_only=False)
    cfg = comp_checkpoint['config']
    comp_model = CompositionalTransformerLM(
        vocab_size=cfg['vocab_size'],
        hidden_dim=cfg['hidden_dim'],
        n_layers=cfg['n_layers'],
        n_heads=cfg['n_heads'],
        max_len=cfg['max_len']
    )
    comp_model.load_state_dict(comp_checkpoint['model_state'])

    # Load data
    with open(f"{save_dir}/data.pkl", 'rb') as f:
        data = pickle.load(f)

    return tokenizer, std_model, comp_model, data


if __name__ == "__main__":
    main()
