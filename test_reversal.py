"""
Sequence Reversal Task - A task transformers can solve with small data.

Task: Given "1 2 3 4 |" predict "4 3 2 1"

This tests:
- Attention mechanism (must look at position n-i for output i)
- Position encoding
- Whether CTM's thinking ticks help with algorithmic tasks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import math
from collections import defaultdict

# Set seeds
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# ============================================================
# REUSE MODEL COMPONENTS FROM ORIGINAL
# ============================================================

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


class CausalSelfAttention(nn.Module):
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
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(~mask, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = attn @ v
        out = out.transpose(1, 2).reshape(B, T, C)
        return self.proj(out)


class StandardTransformerBlock(nn.Module):
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
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.mlp(self.ln2(x))
        return x


class StandardTransformerLM(nn.Module):
    def __init__(self, vocab_size, hidden_dim, n_layers, n_heads, max_len=512, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.token_embed = nn.Embedding(vocab_size, hidden_dim)
        self.pos_encode = SinusoidalPositionalEncoding(hidden_dim, max_len)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            StandardTransformerBlock(hidden_dim, n_heads, dropout)
            for _ in range(n_layers)
        ])
        self.ln_final = nn.LayerNorm(hidden_dim)
        self.output = nn.Linear(hidden_dim, vocab_size)

    def get_causal_mask(self, seq_len, device):
        return torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool))

    def forward(self, x, return_all=False):
        B, T = x.shape
        h = self.dropout(self.pos_encode(self.token_embed(x)))
        mask = self.get_causal_mask(T, x.device)
        layer_hiddens = [h]
        for block in self.blocks:
            h = block(h, mask)
            layer_hiddens.append(h)
        h = self.ln_final(h)
        logits = self.output(h)
        if return_all:
            return logits, layer_hiddens
        return logits


class CTMTransformerLM(nn.Module):
    """Simplified CTM for reversal task."""
    def __init__(self, vocab_size, hidden_dim, n_layers, n_heads,
                 max_len=512, n_ticks=4, history_len=2, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_ticks = n_ticks
        self.history_len = history_len

        self.token_embed = nn.Embedding(vocab_size, hidden_dim)
        self.pos_encode = SinusoidalPositionalEncoding(hidden_dim, max_len)
        self.dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            StandardTransformerBlock(hidden_dim, n_heads, dropout)
            for _ in range(n_layers)
        ])

        # History processor
        self.history_processor = nn.Sequential(
            nn.Linear(hidden_dim * history_len, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.ln_final = nn.LayerNorm(hidden_dim)
        self.output = nn.Linear(hidden_dim, vocab_size)

    def get_causal_mask(self, seq_len, device):
        return torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool))

    def forward(self, x, return_all=False):
        B, T = x.shape
        h = self.dropout(self.pos_encode(self.token_embed(x)))
        mask = self.get_causal_mask(T, x.device)

        history = [h.clone()]
        all_logits = []

        for tick in range(self.n_ticks):
            for block in self.blocks:
                h = block(h, mask)

            # History features
            if len(history) >= self.history_len:
                recent = history[-self.history_len:]
            else:
                padding = [history[0]] * (self.history_len - len(history))
                recent = padding + history

            hist_concat = torch.cat(recent, dim=-1)
            hist_features = self.history_processor(hist_concat)
            h = h + 0.1 * hist_features

            history.append(h.clone())
            if len(history) > self.history_len + 1:
                history = history[-(self.history_len + 1):]

            h_norm = self.ln_final(h)
            logits = self.output(h_norm)
            all_logits.append(logits)

        if return_all:
            return logits, all_logits
        return logits


# ============================================================
# DATA GENERATION
# ============================================================

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
        # Generate random digits
        digits = [random.randint(0, 9) for _ in range(length)]

        # Build sequence: BOS + digits + SEP + reversed_digits + EOS
        seq = [BOS]
        seq.extend([d + DIGIT_OFFSET for d in digits])
        seq.append(SEP)
        seq.extend([d + DIGIT_OFFSET for d in reversed(digits)])
        seq.append(EOS)

        data.append(seq)

    return data


def pad_sequences(seqs, pad_id=0):
    """Pad sequences to same length."""
    max_len = max(len(s) for s in seqs)
    padded = []
    for s in seqs:
        padded.append(s + [pad_id] * (max_len - len(s)))
    return padded


# ============================================================
# TRAINING
# ============================================================

def train_epoch(model, data, optimizer, device):
    model.train()
    total_loss = 0
    total_correct = 0
    total_tokens = 0

    # Batch the data
    batch_size = 32
    random.shuffle(data)

    for i in range(0, len(data), batch_size):
        batch = data[i:i+batch_size]
        batch = pad_sequences(batch)
        x = torch.tensor(batch, device=device)

        # Input is all but last token, target is all but first
        inputs = x[:, :-1]
        targets = x[:, 1:]

        optimizer.zero_grad()
        logits = model(inputs)

        # Compute loss (ignore padding)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            ignore_index=0  # PAD
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * targets.numel()

        # Compute accuracy
        preds = logits.argmax(dim=-1)
        mask = targets != 0
        total_correct += ((preds == targets) & mask).sum().item()
        total_tokens += mask.sum().item()

    return total_loss / total_tokens, total_correct / total_tokens


def evaluate(model, data, device):
    model.eval()
    total_correct = 0
    total_tokens = 0
    seq_correct = 0

    # Also track accuracy on reversal part only (after SEP)
    reversal_correct = 0
    reversal_tokens = 0

    SEP = 3

    with torch.no_grad():
        batch = pad_sequences(data)
        x = torch.tensor(batch, device=device)

        inputs = x[:, :-1]
        targets = x[:, 1:]

        logits = model(inputs)
        preds = logits.argmax(dim=-1)

        # Overall accuracy
        mask = targets != 0
        total_correct = ((preds == targets) & mask).sum().item()
        total_tokens = mask.sum().item()

        # Per-sequence accuracy (full sequence correct)
        for i in range(len(data)):
            seq_len = len(data[i]) - 1
            if (preds[i, :seq_len] == targets[i, :seq_len]).all():
                seq_correct += 1

        # Reversal-only accuracy (tokens after SEP)
        for i in range(len(data)):
            seq = data[i]
            sep_pos = seq.index(SEP)
            # Targets after SEP position
            start = sep_pos  # In targets, this corresponds to the token after SEP
            end = len(seq) - 1  # Exclude EOS from counting
            if start < end:
                rev_preds = preds[i, start:end]
                rev_targets = targets[i, start:end]
                reversal_correct += (rev_preds == rev_targets).sum().item()
                reversal_tokens += (end - start)

    return {
        'token_acc': total_correct / total_tokens,
        'seq_acc': seq_correct / len(data),
        'reversal_acc': reversal_correct / reversal_tokens if reversal_tokens > 0 else 0
    }


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("SEQUENCE REVERSAL TASK - Standard vs CTM Transformer")
    print("=" * 70)

    # Hyperparameters
    VOCAB_SIZE = 14  # PAD, BOS, EOS, SEP, digits 0-9
    HIDDEN_DIM = 32
    N_LAYERS = 2
    N_HEADS = 4
    N_TICKS = 4
    HISTORY_LEN = 2
    N_EPOCHS = 100
    LR = 1e-3

    # Generate data
    print("\n1. Generating data...")
    train_data = generate_reversal_data(500, min_len=3, max_len=6)
    val_data = generate_reversal_data(100, min_len=3, max_len=6)
    test_data = generate_reversal_data(100, min_len=3, max_len=6)

    # Also test generalization to longer sequences
    test_long = generate_reversal_data(50, min_len=7, max_len=10)

    print(f"   Train: {len(train_data)} sequences")
    print(f"   Val: {len(val_data)} sequences")
    print(f"   Test: {len(test_data)} sequences")
    print(f"   Test (long): {len(test_long)} sequences (length 7-10)")

    # Show example
    example = train_data[0]
    digits = [str(t-4) if t >= 4 else ['P','B','E','|'][t] for t in example]
    print(f"   Example: {' '.join(digits)}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n   Using {device}")

    # Create models
    print("\n2. Creating models...")
    std_model = StandardTransformerLM(
        VOCAB_SIZE, HIDDEN_DIM, N_LAYERS, N_HEADS, dropout=0.1
    ).to(device)

    ctm_model = CTMTransformerLM(
        VOCAB_SIZE, HIDDEN_DIM, N_LAYERS, N_HEADS,
        n_ticks=N_TICKS, history_len=HISTORY_LEN, dropout=0.1
    ).to(device)

    std_params = sum(p.numel() for p in std_model.parameters())
    ctm_params = sum(p.numel() for p in ctm_model.parameters())
    print(f"   Standard: {std_params:,} params")
    print(f"   CTM: {ctm_params:,} params (n_ticks={N_TICKS})")

    # Optimizers
    std_opt = torch.optim.AdamW(std_model.parameters(), lr=LR)
    ctm_opt = torch.optim.AdamW(ctm_model.parameters(), lr=LR)

    # Training
    print("\n3. Training...")
    print("-" * 70)
    print(f"{'Epoch':>6} | {'Std Loss':>8} {'Std Acc':>8} | {'CTM Loss':>8} {'CTM Acc':>8} | {'Val Rev Acc':>12}")
    print("-" * 70)

    best_std_acc = 0
    best_ctm_acc = 0

    for epoch in range(1, N_EPOCHS + 1):
        std_loss, std_acc = train_epoch(std_model, train_data, std_opt, device)
        ctm_loss, ctm_acc = train_epoch(ctm_model, train_data, ctm_opt, device)

        if epoch % 10 == 0 or epoch == 1:
            std_val = evaluate(std_model, val_data, device)
            ctm_val = evaluate(ctm_model, val_data, device)

            best_std_acc = max(best_std_acc, std_val['reversal_acc'])
            best_ctm_acc = max(best_ctm_acc, ctm_val['reversal_acc'])

            print(f"{epoch:>6} | {std_loss:>8.4f} {std_acc:>7.1%} | "
                  f"{ctm_loss:>8.4f} {ctm_acc:>7.1%} | "
                  f"std={std_val['reversal_acc']:.1%} ctm={ctm_val['reversal_acc']:.1%}")

    # Final evaluation
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)

    std_test = evaluate(std_model, test_data, device)
    ctm_test = evaluate(ctm_model, test_data, device)
    std_long = evaluate(std_model, test_long, device)
    ctm_long = evaluate(ctm_model, test_long, device)

    print("\nTest Set (length 3-6, seen lengths):")
    print(f"  Standard: token_acc={std_test['token_acc']:.1%}, "
          f"seq_acc={std_test['seq_acc']:.1%}, reversal_acc={std_test['reversal_acc']:.1%}")
    print(f"  CTM:      token_acc={ctm_test['token_acc']:.1%}, "
          f"seq_acc={ctm_test['seq_acc']:.1%}, reversal_acc={ctm_test['reversal_acc']:.1%}")

    print("\nGeneralization Test (length 7-10, UNSEEN lengths):")
    print(f"  Standard: token_acc={std_long['token_acc']:.1%}, "
          f"seq_acc={std_long['seq_acc']:.1%}, reversal_acc={std_long['reversal_acc']:.1%}")
    print(f"  CTM:      token_acc={ctm_long['token_acc']:.1%}, "
          f"seq_acc={ctm_long['seq_acc']:.1%}, reversal_acc={ctm_long['reversal_acc']:.1%}")

    # Show some predictions
    print("\n" + "=" * 70)
    print("SAMPLE PREDICTIONS")
    print("=" * 70)

    def show_predictions(model, name, data, n=5):
        model.eval()
        print(f"\n{name}:")
        with torch.no_grad():
            for seq in data[:n]:
                x = torch.tensor([seq[:-1]], device=device)
                logits = model(x)
                preds = logits[0].argmax(dim=-1).tolist()

                # Find SEP position
                sep_pos = seq.index(3)

                # Format output
                input_digits = [str(t-4) if t >= 4 else '|' for t in seq[1:sep_pos+1]]
                target_digits = [str(t-4) for t in seq[sep_pos+1:-1]]
                pred_digits = [str(t-4) if t >= 4 else '?' for t in preds[sep_pos:sep_pos+len(target_digits)]]

                correct = '✓' if pred_digits == target_digits else '✗'
                print(f"  Input: {' '.join(input_digits)} → "
                      f"Target: {' '.join(target_digits)} | "
                      f"Pred: {' '.join(pred_digits)} {correct}")

    show_predictions(std_model, "Standard Transformer", test_data)
    show_predictions(ctm_model, "CTM Transformer", test_data)

    print("\nLong sequences (unseen lengths):")
    show_predictions(std_model, "Standard Transformer", test_long)
    show_predictions(ctm_model, "CTM Transformer", test_long)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"""
Task: Sequence Reversal (e.g., "1 2 3 |" → "3 2 1")
Data: 500 training sequences, lengths 3-6

Results:
  - Standard Transformer: {std_test['reversal_acc']:.1%} reversal accuracy
  - CTM Transformer:      {ctm_test['reversal_acc']:.1%} reversal accuracy

Generalization to longer sequences (7-10):
  - Standard: {std_long['reversal_acc']:.1%}
  - CTM:      {ctm_long['reversal_acc']:.1%}
    """)


if __name__ == "__main__":
    main()
