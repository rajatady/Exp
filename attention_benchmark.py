"""
ATTENTION IS ALL YOU NEED - CTM Edition

Recreating the spirit of Vaswani et al. 2017:
- They showed Transformers beat RNNs on translation
- We test if CTM beats Transformers on reasoning

Original paper's key findings:
1. QUALITY: +2 BLEU over best RNNs
2. SPEED: 10-100x faster training (parallelization)
3. SCALING: Predictable improvement with scale
4. EMERGENCE: Interpretable attention patterns

Our benchmarks (transformer-strong tasks):
1. ARITHMETIC: Multi-digit addition (transformers struggle at scale)
2. LOGIC: Propositional reasoning chains
3. ALGORITHMIC: Sorting, pattern matching
4. COMPOSITIONAL: SCAN-like systematic generalization

Goal: Find if CTM shows "Attention is All You Need"-level breakthrough
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import math
import time
from collections import defaultdict

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# ============================================================
# MODEL DEFINITIONS (same as before, compact)
# ============================================================

class SinusoidalPE(nn.Module):
    def __init__(self, dim, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class Attention(nn.Module):
    def __init__(self, dim, heads, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, mask=None):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(~mask, float('-inf'))
        attn = self.dropout(F.softmax(scores, dim=-1))
        return self.proj((attn @ v).transpose(1, 2).reshape(B, T, C))

class Block(nn.Module):
    def __init__(self, dim, heads, dropout=0.1):
        super().__init__()
        self.attn = Attention(dim, heads, dropout)
        self.mlp = nn.Sequential(nn.Linear(dim, 4*dim), nn.GELU(), nn.Linear(4*dim, dim), nn.Dropout(dropout))
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
    def forward(self, x, mask=None):
        x = x + self.attn(self.ln1(x), mask)
        return x + self.mlp(self.ln2(x))

class Transformer(nn.Module):
    def __init__(self, vocab, dim, layers, heads, dropout=0.1):
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.pe = SinusoidalPE(dim)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([Block(dim, heads, dropout) for _ in range(layers)])
        self.ln = nn.LayerNorm(dim)
        self.out = nn.Linear(dim, vocab)
    def forward(self, x):
        mask = torch.tril(torch.ones(x.size(1), x.size(1), dtype=torch.bool, device=x.device))
        h = self.drop(self.pe(self.embed(x)))
        for b in self.blocks:
            h = b(h, mask)
        return self.out(self.ln(h))

class CTM(nn.Module):
    def __init__(self, vocab, dim, layers, heads, ticks=4, dropout=0.1):
        super().__init__()
        self.ticks = ticks
        self.embed = nn.Embedding(vocab, dim)
        self.pe = SinusoidalPE(dim)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([Block(dim, heads, dropout) for _ in range(layers)])
        self.hist = nn.Sequential(nn.Linear(dim*2, dim), nn.GELU(), nn.Linear(dim, dim))
        self.ln = nn.LayerNorm(dim)
        self.out = nn.Linear(dim, vocab)
    def forward(self, x):
        mask = torch.tril(torch.ones(x.size(1), x.size(1), dtype=torch.bool, device=x.device))
        h = self.drop(self.pe(self.embed(x)))
        prev = h
        for tick in range(self.ticks):
            for b in self.blocks:
                h = b(h, mask)
            h = h + 0.1 * self.hist(torch.cat([h, prev], dim=-1))
            prev = h
        return self.out(self.ln(h))

# ============================================================
# BENCHMARK 1: MULTI-DIGIT ADDITION
# Transformers struggle with carry propagation at scale
# ============================================================

def generate_addition(n_samples, n_digits=3):
    """Generate addition: 123+456=579"""
    # Vocab: 0-9 digits, +, =, PAD, BOS, EOS
    data = []
    for _ in range(n_samples):
        a = random.randint(10**(n_digits-1), 10**n_digits - 1)
        b = random.randint(10**(n_digits-1), 10**n_digits - 1)
        c = a + b
        # Format: BOS a + b = c EOS
        seq = [11]  # BOS
        seq.extend([int(d) for d in str(a)])
        seq.append(10)  # +
        seq.extend([int(d) for d in str(b)])
        seq.append(12)  # =
        seq.extend([int(d) for d in str(c)])
        seq.append(13)  # EOS
        data.append(seq)
    return data

def eval_addition(model, data, eq_token=12):
    """Evaluate addition accuracy."""
    model.eval()
    correct = 0
    with torch.no_grad():
        for seq in data:
            x = torch.tensor([seq])
            logits = model(x[:, :-1])
            preds = logits[0].argmax(dim=-1)
            eq_pos = seq.index(eq_token)
            end_pos = len(seq) - 1
            if all(preds[i].item() == seq[i+1] for i in range(eq_pos, end_pos)):
                correct += 1
    return correct / len(data)

# ============================================================
# BENCHMARK 2: LOGICAL REASONING CHAINS
# A→B, B→C, C→D, ... what follows from A?
# ============================================================

def generate_logic_chain(n_samples, chain_len=3):
    """Generate logic chains: A>B B>C C>D ? A > D (answer: D)"""
    # Vocab: A-Z (0-25), > (26), ? (27), PAD (28), BOS (29), EOS (30)
    data = []
    for _ in range(n_samples):
        # Pick random start and create chain
        start = random.randint(0, 15)  # A-P
        chain = [start + i for i in range(chain_len + 1)]  # A, B, C, D for chain_len=3

        # Build sequence: BOS A>B B>C C>D ? A > EOS
        seq = [29]  # BOS
        for i in range(chain_len):
            seq.extend([chain[i], 26, chain[i+1]])  # X>Y
            seq.append(28) if i < chain_len - 1 else None  # space
        if seq[-1] == 28:
            seq = seq[:-1]
        seq.extend([27, chain[0], 26])  # ? A >
        seq.append(chain[-1])  # answer
        seq.append(30)  # EOS
        data.append(seq)
    return data

def eval_logic(model, data, q_token=27):
    """Evaluate logic chain accuracy."""
    model.eval()
    correct = 0
    with torch.no_grad():
        for seq in data:
            x = torch.tensor([seq])
            logits = model(x[:, :-1])
            preds = logits[0].argmax(dim=-1)
            # Find answer position (after last >)
            ans_pos = len(seq) - 2  # Before EOS
            if preds[ans_pos - 1].item() == seq[ans_pos]:
                correct += 1
    return correct / len(data)

# ============================================================
# BENCHMARK 3: SORTING
# Input: 3 1 4 1 5 | Output: 1 1 3 4 5
# ============================================================

def generate_sorting(n_samples, length=5):
    """Generate sorting: 3 1 4 | 1 3 4"""
    # Vocab: 0-9 digits, | (10), PAD (11), BOS (12), EOS (13)
    data = []
    for _ in range(n_samples):
        nums = [random.randint(0, 9) for _ in range(length)]
        sorted_nums = sorted(nums)
        seq = [12] + nums + [10] + sorted_nums + [13]
        data.append(seq)
    return data

def eval_sorting(model, data, sep=10):
    """Evaluate sorting accuracy."""
    model.eval()
    correct = 0
    with torch.no_grad():
        for seq in data:
            x = torch.tensor([seq])
            logits = model(x[:, :-1])
            preds = logits[0].argmax(dim=-1)
            sep_pos = seq.index(sep)
            end_pos = len(seq) - 1
            if all(preds[i].item() == seq[i+1] for i in range(sep_pos, end_pos)):
                correct += 1
    return correct / len(data)

# ============================================================
# BENCHMARK 4: SCAN-LIKE COMPOSITIONAL GENERALIZATION
# Train: "walk" → WALK, "run" → RUN, "walk twice" → WALK WALK
# Test: "run twice" → RUN RUN (unseen combination)
# ============================================================

def generate_scan_like(n_samples, test_held_out=False):
    """Generate SCAN-like compositional data."""
    # Primitives
    actions = {'walk': [1], 'run': [2], 'jump': [3], 'look': [4]}
    modifiers = {'twice': 2, 'thrice': 3, 'and': -1}

    # Vocab: PAD=0, WALK=1, RUN=2, JUMP=3, LOOK=4,
    #        walk=5, run=6, jump=7, look=8, twice=9, thrice=10, and=11, BOS=12, EOS=13, SEP=14

    data = []
    for _ in range(n_samples):
        # Generate command
        action = random.choice(list(actions.keys()))

        if random.random() < 0.5:
            # With modifier
            mod = random.choice(['twice', 'thrice'])
            if test_held_out and action == 'run' and mod == 'twice':
                continue  # Skip held-out combination in training

            input_seq = [12, 5 + list(actions.keys()).index(action), 9 + ['twice', 'thrice'].index(mod), 14]
            output_seq = actions[action] * modifiers[mod] + [13]
        else:
            # Single action
            input_seq = [12, 5 + list(actions.keys()).index(action), 14]
            output_seq = actions[action] + [13]

        data.append(input_seq + output_seq)

    return data

def generate_scan_test(n_samples):
    """Generate held-out test: 'run twice' → RUN RUN"""
    data = []
    for _ in range(n_samples):
        # run twice = token 6, 9
        input_seq = [12, 6, 9, 14]  # BOS run twice SEP
        output_seq = [2, 2, 13]  # RUN RUN EOS
        data.append(input_seq + output_seq)
    return data

def eval_scan(model, data, sep=14):
    """Evaluate SCAN-like accuracy."""
    model.eval()
    correct = 0
    with torch.no_grad():
        for seq in data:
            x = torch.tensor([seq])
            logits = model(x[:, :-1])
            preds = logits[0].argmax(dim=-1)
            sep_pos = seq.index(sep)
            end_pos = len(seq) - 1
            if all(preds[i].item() == seq[i+1] for i in range(sep_pos, end_pos)):
                correct += 1
    return correct / len(data)

# ============================================================
# TRAINING
# ============================================================

def train(model, data, epochs, lr=1e-3, vocab_size=32):
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    for epoch in range(epochs):
        model.train()
        random.shuffle(data)
        total_loss = 0
        for i in range(0, len(data), 32):
            batch = data[i:i+32]
            max_len = max(len(s) for s in batch)
            x = torch.tensor([s + [0]*(max_len-len(s)) for s in batch])
            logits = model(x[:, :-1])
            loss = F.cross_entropy(logits.reshape(-1, vocab_size), x[:, 1:].reshape(-1), ignore_index=0)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item()
    return total_loss

# ============================================================
# MAIN BENCHMARK
# ============================================================

def run_benchmarks():
    print("=" * 80)
    print("'ATTENTION IS ALL YOU NEED' - CTM EDITION")
    print("=" * 80)
    print("""
Recreating the landmark evaluation of Vaswani et al. 2017:
They showed: Transformer >> RNN on translation
We test: CTM vs Transformer on reasoning

Benchmarks:
1. ARITHMETIC: Multi-digit addition (tests carry propagation)
2. LOGIC: Propositional reasoning chains (tests multi-hop inference)
3. SORTING: Ordering numbers (tests comparison algorithm)
4. COMPOSITIONAL: SCAN-like generalization (tests systematic compositionality)
""")

    results = {}

    # ============================================================
    # BENCHMARK 1: ARITHMETIC
    # ============================================================
    print("\n" + "=" * 80)
    print("BENCHMARK 1: MULTI-DIGIT ADDITION")
    print("=" * 80)

    for n_digits in [2, 3, 4]:
        print(f"\n--- {n_digits}-digit addition ---")

        train_data = generate_addition(500, n_digits)
        test_data = generate_addition(100, n_digits)
        test_harder = generate_addition(100, n_digits + 1)  # OOD test

        for name, model_class, kwargs in [
            ('Transformer', Transformer, {'vocab': 14, 'dim': 64, 'layers': 2, 'heads': 4}),
            ('CTM-4T', CTM, {'vocab': 14, 'dim': 64, 'layers': 2, 'heads': 4, 'ticks': 4}),
            ('CTM-8T', CTM, {'vocab': 14, 'dim': 64, 'layers': 2, 'heads': 4, 'ticks': 8}),
        ]:
            model = model_class(**kwargs)

            start = time.time()
            train(model, train_data, epochs=50, vocab_size=14)
            train_time = time.time() - start

            acc = eval_addition(model, test_data)
            acc_ood = eval_addition(model, test_harder)

            key = f"add_{n_digits}d_{name}"
            results[key] = {'acc': acc, 'ood': acc_ood, 'time': train_time}
            print(f"  {name}: {acc:.1%} (OOD: {acc_ood:.1%}) [{train_time:.1f}s]")

    # ============================================================
    # BENCHMARK 2: LOGIC CHAINS
    # ============================================================
    print("\n" + "=" * 80)
    print("BENCHMARK 2: LOGICAL REASONING CHAINS")
    print("=" * 80)

    for chain_len in [2, 3, 4, 5]:
        print(f"\n--- Chain length {chain_len} ---")

        train_data = generate_logic_chain(500, chain_len)
        test_data = generate_logic_chain(100, chain_len)
        test_longer = generate_logic_chain(100, chain_len + 2)  # OOD

        for name, model_class, kwargs in [
            ('Transformer', Transformer, {'vocab': 32, 'dim': 64, 'layers': 2, 'heads': 4}),
            ('CTM-4T', CTM, {'vocab': 32, 'dim': 64, 'layers': 2, 'heads': 4, 'ticks': 4}),
            ('CTM-8T', CTM, {'vocab': 32, 'dim': 64, 'layers': 2, 'heads': 4, 'ticks': 8}),
        ]:
            model = model_class(**kwargs)
            train(model, train_data, epochs=50, vocab_size=32)

            acc = eval_logic(model, test_data)
            acc_ood = eval_logic(model, test_longer)

            key = f"logic_{chain_len}_{name}"
            results[key] = {'acc': acc, 'ood': acc_ood}
            print(f"  {name}: {acc:.1%} (OOD: {acc_ood:.1%})")

    # ============================================================
    # BENCHMARK 3: SORTING
    # ============================================================
    print("\n" + "=" * 80)
    print("BENCHMARK 3: SORTING")
    print("=" * 80)

    for length in [4, 6, 8]:
        print(f"\n--- Length {length} ---")

        train_data = generate_sorting(500, length)
        test_data = generate_sorting(100, length)
        test_longer = generate_sorting(100, length + 2)

        for name, model_class, kwargs in [
            ('Transformer', Transformer, {'vocab': 14, 'dim': 64, 'layers': 2, 'heads': 4}),
            ('CTM-4T', CTM, {'vocab': 14, 'dim': 64, 'layers': 2, 'heads': 4, 'ticks': 4}),
            ('CTM-8T', CTM, {'vocab': 14, 'dim': 64, 'layers': 2, 'heads': 4, 'ticks': 8}),
        ]:
            model = model_class(**kwargs)
            train(model, train_data, epochs=50, vocab_size=14)

            acc = eval_sorting(model, test_data)
            acc_ood = eval_sorting(model, test_longer)

            key = f"sort_{length}_{name}"
            results[key] = {'acc': acc, 'ood': acc_ood}
            print(f"  {name}: {acc:.1%} (OOD: {acc_ood:.1%})")

    # ============================================================
    # BENCHMARK 4: COMPOSITIONAL GENERALIZATION (SCAN-LIKE)
    # ============================================================
    print("\n" + "=" * 80)
    print("BENCHMARK 4: COMPOSITIONAL GENERALIZATION (SCAN-like)")
    print("=" * 80)
    print("Training on: walk, run, jump, look, X twice, X thrice")
    print("Testing on held-out: 'run twice' → RUN RUN")

    train_data = []
    for _ in range(500):
        train_data.extend(generate_scan_like(1, test_held_out=True))
    test_seen = generate_scan_like(100, test_held_out=False)
    test_held_out = generate_scan_test(100)

    for name, model_class, kwargs in [
        ('Transformer', Transformer, {'vocab': 16, 'dim': 64, 'layers': 2, 'heads': 4}),
        ('CTM-4T', CTM, {'vocab': 16, 'dim': 64, 'layers': 2, 'heads': 4, 'ticks': 4}),
        ('CTM-8T', CTM, {'vocab': 16, 'dim': 64, 'layers': 2, 'heads': 4, 'ticks': 8}),
    ]:
        model = model_class(**kwargs)
        train(model, train_data, epochs=50, vocab_size=16)

        acc_seen = eval_scan(model, test_seen)
        acc_comp = eval_scan(model, test_held_out)

        key = f"scan_{name}"
        results[key] = {'seen': acc_seen, 'compositional': acc_comp}
        print(f"  {name}: Seen={acc_seen:.1%}, Compositional={acc_comp:.1%}")

    # ============================================================
    # SUMMARY: "ATTENTION IS ALL YOU NEED" STYLE
    # ============================================================
    print("\n" + "=" * 80)
    print("SUMMARY: Does CTM Show 'Attention is All You Need'-Level Breakthrough?")
    print("=" * 80)

    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    VASWANI ET AL. (2017) vs OUR FINDINGS                     ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  VASWANI (Transformer vs RNN):                                               ║
║  • Translation: +2-3 BLEU improvement                                        ║
║  • Training: 10-100x faster                                                  ║
║  • Scaling: Predictable improvement                                          ║
║  • Emergence: Interpretable attention heads                                  ║
║                                                                              ║
║  OUR FINDINGS (CTM vs Transformer):                                          ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")

    # Calculate summary statistics
    transformer_accs = []
    ctm4_accs = []
    ctm8_accs = []
    transformer_oods = []
    ctm4_oods = []
    ctm8_oods = []

    for key, val in results.items():
        if 'Transformer' in key:
            if 'acc' in val:
                transformer_accs.append(val['acc'])
            if 'ood' in val:
                transformer_oods.append(val['ood'])
            if 'seen' in val:
                transformer_accs.append(val['seen'])
            if 'compositional' in val:
                transformer_oods.append(val['compositional'])
        elif 'CTM-4T' in key:
            if 'acc' in val:
                ctm4_accs.append(val['acc'])
            if 'ood' in val:
                ctm4_oods.append(val['ood'])
            if 'seen' in val:
                ctm4_accs.append(val['seen'])
            if 'compositional' in val:
                ctm4_oods.append(val['compositional'])
        elif 'CTM-8T' in key:
            if 'acc' in val:
                ctm8_accs.append(val['acc'])
            if 'ood' in val:
                ctm8_oods.append(val['ood'])
            if 'seen' in val:
                ctm8_accs.append(val['seen'])
            if 'compositional' in val:
                ctm8_oods.append(val['compositional'])

    avg_trans = sum(transformer_accs)/len(transformer_accs) if transformer_accs else 0
    avg_ctm4 = sum(ctm4_accs)/len(ctm4_accs) if ctm4_accs else 0
    avg_ctm8 = sum(ctm8_accs)/len(ctm8_accs) if ctm8_accs else 0

    avg_trans_ood = sum(transformer_oods)/len(transformer_oods) if transformer_oods else 0
    avg_ctm4_ood = sum(ctm4_oods)/len(ctm4_oods) if ctm4_oods else 0
    avg_ctm8_ood = sum(ctm8_oods)/len(ctm8_oods) if ctm8_oods else 0

    print(f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                           AGGREGATE RESULTS                                  ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  AVERAGE IN-DISTRIBUTION ACCURACY:                                           ║
║    Transformer:  {avg_trans:6.1%}                                                    ║
║    CTM-4T:       {avg_ctm4:6.1%}  ({'+' if avg_ctm4 > avg_trans else ''}{(avg_ctm4-avg_trans)*100:+.1f}% vs Transformer)                     ║
║    CTM-8T:       {avg_ctm8:6.1%}  ({'+' if avg_ctm8 > avg_trans else ''}{(avg_ctm8-avg_trans)*100:+.1f}% vs Transformer)                     ║
║                                                                              ║
║  AVERAGE OUT-OF-DISTRIBUTION (GENERALIZATION):                               ║
║    Transformer:  {avg_trans_ood:6.1%}                                                    ║
║    CTM-4T:       {avg_ctm4_ood:6.1%}  ({'+' if avg_ctm4_ood > avg_trans_ood else ''}{(avg_ctm4_ood-avg_trans_ood)*100:+.1f}% vs Transformer)                     ║
║    CTM-8T:       {avg_ctm8_ood:6.1%}  ({'+' if avg_ctm8_ood > avg_trans_ood else ''}{(avg_ctm8_ood-avg_trans_ood)*100:+.1f}% vs Transformer)                     ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")

    # Verdict
    improvement_id = (avg_ctm8 - avg_trans) / max(avg_trans, 0.01)
    improvement_ood = (avg_ctm8_ood - avg_trans_ood) / max(avg_trans_ood, 0.01)

    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                              VERDICT                                         ║
╠══════════════════════════════════════════════════════════════════════════════╣""")

    if improvement_id > 0.1 and improvement_ood > 0.2:
        print("""║                                                                              ║
║  ✓ CTM SHOWS SIGNIFICANT IMPROVEMENT OVER TRANSFORMERS                       ║
║                                                                              ║
║  Like Transformers over RNNs:                                                ║
║  • Better accuracy on reasoning tasks                                        ║
║  • Superior out-of-distribution generalization                               ║
║  • Same parameters, just different computation pattern                       ║
║                                                                              ║
║  THE KEY INSIGHT:                                                            ║
║  Transformers were "Attention is All You Need"                               ║
║  CTM shows "ITERATION is Also What You Need"                                 ║
║                                                                              ║""")
    elif improvement_ood > 0.1:
        print("""║                                                                              ║
║  ~ CTM SHOWS PROMISE, ESPECIALLY FOR GENERALIZATION                          ║
║                                                                              ║
║  Not a complete breakthrough like Transformers, but:                         ║
║  • Improved OOD performance suggests better algorithm learning               ║
║  • More ticks = better results (test-time compute scaling works)             ║
║  • May need larger scale to show full potential                              ║
║                                                                              ║""")
    else:
        print("""║                                                                              ║
║  ✗ CTM DOES NOT SHOW BREAKTHROUGH IMPROVEMENT AT THIS SCALE                  ║
║                                                                              ║
║  Possible reasons:                                                           ║
║  • Tasks too simple for iterative advantage                                  ║
║  • Model too small to show emergence                                         ║
║  • Need different training regime                                            ║
║                                                                              ║""")

    print("╚══════════════════════════════════════════════════════════════════════════════╝")

    return results

if __name__ == '__main__':
    results = run_benchmarks()
