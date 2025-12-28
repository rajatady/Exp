"""
DEEP OBSERVATION: Can compositional attention work without iteration?

The insight so far:
- CTM achieves multi-phase attention through iteration
- Phase 1: Attend to b,c (compute multiplication)
- Phase 2: Attend to a + result (compute addition)

The question: Can we get this composition WITHOUT iteration?

Approaches to test:
1. Hierarchical Multi-Head: Some heads attend locally, others globally
2. Two-stage attention in a single forward pass
3. Attention with learned grouping
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
print("COMPOSITIONAL ATTENTION: Can we get CTM benefits without iteration?")
print("=" * 80)

# =============================================================================
# DATA
# =============================================================================

def generate_expression_eval(n_samples, vocab_size=30):
    PLUS = vocab_size - 2
    TIMES = vocab_size - 3
    EQUALS = vocab_size - 1

    data, labels = [], []
    for _ in range(n_samples):
        a = random.randint(1, 5)
        b = random.randint(1, 5)
        c = random.randint(1, 5)
        result = a + b * c

        inp = [a, PLUS, b, TIMES, c, EQUALS, 0, 0]
        target = [0, 0, 0, 0, 0, 0,
                  result // 10 if result >= 10 else result,
                  result % 10 if result >= 10 else 0]

        data.append(inp)
        labels.append(target)

    return torch.tensor(data), torch.tensor(labels)

vocab_size = 30
train_data, train_labels = generate_expression_eval(1000, vocab_size)
test_data, test_labels = generate_expression_eval(200, vocab_size)

# =============================================================================
# MODEL 1: Standard Transformer (baseline)
# =============================================================================

class StandardTransformer(nn.Module):
    def __init__(self, vocab_size, hidden_dim=64, n_layers=4, n_heads=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embedding = nn.Embedding(256, hidden_dim)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim, nhead=n_heads,
                dim_feedforward=hidden_dim * 4, batch_first=True
            ) for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, vocab_size)
        self.name = "Standard Transformer"

    def forward(self, x):
        pos = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
        h = self.embedding(x) + self.pos_embedding(pos)
        for block in self.blocks:
            h = block(h)
        return self.head(self.ln_f(h))

# =============================================================================
# MODEL 2: CTM (what we're trying to match)
# =============================================================================

class CTM(nn.Module):
    def __init__(self, vocab_size, hidden_dim=64, n_layers=2, n_heads=4, n_ticks=4):
        super().__init__()
        self.n_ticks = n_ticks
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embedding = nn.Embedding(256, hidden_dim)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim, nhead=n_heads,
                dim_feedforward=hidden_dim * 4, batch_first=True
            ) for _ in range(n_layers)
        ])
        self.history_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.ln_f = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, vocab_size)
        self.name = "CTM (2 layers × 4 ticks)"

    def forward(self, x):
        pos = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
        h = self.embedding(x) + self.pos_embedding(pos)
        h_prev = torch.zeros_like(h)

        for tick in range(self.n_ticks):
            history = self.history_proj(torch.cat([h, h_prev], dim=-1))
            h_input = h + 0.1 * history
            h_new = h_input
            for block in self.blocks:
                h_new = block(h_new)
            h_prev = h.detach()
            h = h + h_new

        return self.head(self.ln_f(h))

# =============================================================================
# MODEL 3: Two-Stage Attention (compositional, no iteration)
# =============================================================================

class TwoStageAttention(nn.Module):
    """
    Idea: Do two attention passes in a single forward, with the second
    attending to the results of the first.

    Stage 1: Local attention (compute sub-expressions)
    Stage 2: Global attention (combine results)
    """
    def __init__(self, vocab_size, hidden_dim=64, n_heads=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads

        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embedding = nn.Embedding(256, hidden_dim)

        # Stage 1: First attention pass
        self.stage1_ln = nn.LayerNorm(hidden_dim)
        self.stage1_q = nn.Linear(hidden_dim, hidden_dim)
        self.stage1_k = nn.Linear(hidden_dim, hidden_dim)
        self.stage1_v = nn.Linear(hidden_dim, hidden_dim)
        self.stage1_o = nn.Linear(hidden_dim, hidden_dim)
        self.stage1_ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        self.stage1_ln2 = nn.LayerNorm(hidden_dim)

        # Stage 2: Second attention that sees stage 1 results
        self.stage2_ln = nn.LayerNorm(hidden_dim)
        self.stage2_q = nn.Linear(hidden_dim, hidden_dim)
        self.stage2_k = nn.Linear(hidden_dim * 2, hidden_dim)  # Sees original + stage1
        self.stage2_v = nn.Linear(hidden_dim * 2, hidden_dim)
        self.stage2_o = nn.Linear(hidden_dim, hidden_dim)
        self.stage2_ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        self.stage2_ln2 = nn.LayerNorm(hidden_dim)

        self.ln_f = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, vocab_size)
        self.name = "Two-Stage Attention"

    def forward(self, x):
        batch_size, seq_len = x.shape
        pos = torch.arange(seq_len, device=x.device).unsqueeze(0)
        h = self.embedding(x) + self.pos_embedding(pos)
        h_original = h.clone()

        # Stage 1: First attention pass
        h_normed = self.stage1_ln(h)
        q = self.stage1_q(h_normed)
        k = self.stage1_k(h_normed)
        v = self.stage1_v(h_normed)

        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        attn = F.softmax(torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim), dim=-1)
        attn_out = torch.matmul(attn, v)
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        h = h + self.stage1_o(attn_out)
        h = h + self.stage1_ffn(self.stage1_ln2(h))
        h_stage1 = h.clone()

        # Stage 2: Second attention with access to stage 1 results
        h_normed = self.stage2_ln(h)
        q = self.stage2_q(h_normed)

        # Key and value see both original and stage 1 results
        kv_input = torch.cat([h_original, h_stage1], dim=-1)  # [batch, seq, hidden*2]
        k = self.stage2_k(kv_input)
        v = self.stage2_v(kv_input)

        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        attn = F.softmax(torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim), dim=-1)
        attn_out = torch.matmul(attn, v)
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        h = h + self.stage2_o(attn_out)
        h = h + self.stage2_ffn(self.stage2_ln2(h))

        return self.head(self.ln_f(h))

# =============================================================================
# MODEL 4: Hierarchical Attention (inspired by syntax trees)
# =============================================================================

class HierarchicalAttention(nn.Module):
    """
    Idea: Explicitly process sub-expressions first, then combine.
    Uses soft grouping based on operators.
    """
    def __init__(self, vocab_size, hidden_dim=64, n_heads=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads

        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embedding = nn.Embedding(256, hidden_dim)

        # Operator-aware grouping
        self.group_scorer = nn.Linear(hidden_dim, 2)  # 2 priority levels

        # Low-priority attention (e.g., multiplication first)
        self.low_attn = nn.MultiheadAttention(hidden_dim, n_heads, batch_first=True)
        self.low_ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        self.low_ln1 = nn.LayerNorm(hidden_dim)
        self.low_ln2 = nn.LayerNorm(hidden_dim)

        # High-priority attention (e.g., addition after)
        self.high_attn = nn.MultiheadAttention(hidden_dim, n_heads, batch_first=True)
        self.high_ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        self.high_ln1 = nn.LayerNorm(hidden_dim)
        self.high_ln2 = nn.LayerNorm(hidden_dim)

        # Final processing
        self.final_attn = nn.MultiheadAttention(hidden_dim, n_heads, batch_first=True)
        self.final_ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        self.final_ln1 = nn.LayerNorm(hidden_dim)
        self.final_ln2 = nn.LayerNorm(hidden_dim)

        self.ln_f = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, vocab_size)
        self.name = "Hierarchical Attention"

    def forward(self, x):
        pos = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
        h = self.embedding(x) + self.pos_embedding(pos)

        # Compute soft grouping scores (which tokens should be processed first)
        group_scores = F.softmax(self.group_scorer(h), dim=-1)  # [batch, seq, 2]

        # Low-priority pass (weighted by group_scores[:,:,0])
        h_low = self.low_ln1(h)
        weight_low = group_scores[:, :, 0:1]  # [batch, seq, 1]
        h_low_weighted = h_low * weight_low
        attn_out, _ = self.low_attn(h_low_weighted, h_low_weighted, h_low_weighted)
        h = h + attn_out
        h = h + self.low_ffn(self.low_ln2(h))

        # High-priority pass (weighted by group_scores[:,:,1])
        h_high = self.high_ln1(h)
        weight_high = group_scores[:, :, 1:2]
        h_high_weighted = h_high * weight_high
        attn_out, _ = self.high_attn(h_high_weighted, h, h)  # Query from high, attend to all
        h = h + attn_out
        h = h + self.high_ffn(self.high_ln2(h))

        # Final pass
        h_final = self.final_ln1(h)
        attn_out, _ = self.final_attn(h_final, h_final, h_final)
        h = h + attn_out
        h = h + self.final_ffn(self.final_ln2(h))

        return self.head(self.ln_f(h))

# =============================================================================
# MODEL 5: Recurrence-Free CTM (can we get the benefit without iteration?)
# =============================================================================

class RecurrenceFreeMultiPass(nn.Module):
    """
    Idea: Instead of iterating, have multiple DIFFERENT layers that mimic
    what CTM learns to do across ticks.

    Layer 1-2: Process like tick 0 (focus on sub-expressions)
    Layer 3-4: Process like tick 1 (combine with previous results)
    """
    def __init__(self, vocab_size, hidden_dim=64, n_heads=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embedding = nn.Embedding(256, hidden_dim)

        # Tick 0 analog: 2 layers for initial processing
        self.tick0_blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim, nhead=n_heads,
                dim_feedforward=hidden_dim * 4, batch_first=True
            ) for _ in range(2)
        ])

        # Tick 1 analog: 2 layers that receive tick 0 output via cross-attention
        self.tick1_self_attn = nn.MultiheadAttention(hidden_dim, n_heads, batch_first=True)
        self.tick1_cross_attn = nn.MultiheadAttention(hidden_dim, n_heads, batch_first=True)
        self.tick1_ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        self.tick1_ln1 = nn.LayerNorm(hidden_dim)
        self.tick1_ln2 = nn.LayerNorm(hidden_dim)
        self.tick1_ln3 = nn.LayerNorm(hidden_dim)

        # Final block
        self.tick1_blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim, nhead=n_heads,
                dim_feedforward=hidden_dim * 4, batch_first=True
            ) for _ in range(1)
        ])

        self.ln_f = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, vocab_size)
        self.name = "Recurrence-Free Multi-Pass"

    def forward(self, x):
        pos = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
        h = self.embedding(x) + self.pos_embedding(pos)

        # Tick 0 analog
        h0 = h
        for block in self.tick0_blocks:
            h0 = block(h0)

        # Tick 1 analog: Self-attention + cross-attention to tick 0 results
        h1 = h0
        h1_norm = self.tick1_ln1(h1)
        attn_out, _ = self.tick1_self_attn(h1_norm, h1_norm, h1_norm)
        h1 = h1 + attn_out

        # Cross-attend to tick 0 results (this is like seeing h_prev)
        h1_norm = self.tick1_ln2(h1)
        attn_out, _ = self.tick1_cross_attn(h1_norm, h0, h0)
        h1 = h1 + attn_out

        h1 = h1 + self.tick1_ffn(self.tick1_ln3(h1))

        for block in self.tick1_blocks:
            h1 = block(h1)

        # Accumulate like CTM
        h = h + h0 + h1

        return self.head(self.ln_f(h))

# =============================================================================
# TRAINING
# =============================================================================

def train_and_eval(model, train_data, train_labels, test_data, test_labels, n_epochs=100):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    for epoch in range(n_epochs):
        model.train()
        logits = model(train_data)
        loss = criterion(logits.view(-1, logits.size(-1)), train_labels.view(-1))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    # Final evaluation
    model.eval()
    with torch.no_grad():
        logits = model(test_data)
        preds = logits.argmax(dim=-1)
        mask = test_labels != 0
        correct = (preds == test_labels) & mask
        return (correct.sum().float() / mask.sum().float()).item() * 100

def count_params(model):
    return sum(p.numel() for p in model.parameters())

# =============================================================================
# EXPERIMENT
# =============================================================================

print("\n" + "=" * 80)
print("EXPERIMENT: Can compositional attention match CTM?")
print("=" * 80)

models = [
    StandardTransformer(vocab_size),
    CTM(vocab_size),
    TwoStageAttention(vocab_size),
    HierarchicalAttention(vocab_size),
    RecurrenceFreeMultiPass(vocab_size),
]

results = {}
for model in models:
    print(f"\nTraining {model.name}...")
    print(f"  Parameters: {count_params(model):,}")
    acc = train_and_eval(model, train_data, train_labels, test_data, test_labels)
    results[model.name] = acc
    print(f"  Final accuracy: {acc:.1f}%")

# =============================================================================
# ANALYSIS
# =============================================================================

print("\n" + "=" * 80)
print("RESULTS SUMMARY")
print("=" * 80)

print(f"\n{'Model':<35} {'Params':>10} {'Accuracy':>10}")
print("-" * 57)
for model in models:
    name = model.name
    params = count_params(model)
    acc = results[name]
    print(f"{name:<35} {params:>10,} {acc:>9.1f}%")

# =============================================================================
# KEY INSIGHTS
# =============================================================================

print("\n" + "=" * 80)
print("KEY INSIGHTS")
print("=" * 80)

ctm_acc = results["CTM (2 layers × 4 ticks)"]
transformer_acc = results["Standard Transformer"]

print(f"""
CTM advantage over Transformer: {ctm_acc - transformer_acc:+.1f}%

Compositional alternatives:
""")

for name, acc in results.items():
    if name not in ["CTM (2 layers × 4 ticks)", "Standard Transformer"]:
        gap_to_ctm = acc - ctm_acc
        gap_to_trans = acc - transformer_acc
        print(f"  {name}: {acc:.1f}% (CTM gap: {gap_to_ctm:+.1f}%, Trans gap: {gap_to_trans:+.1f}%)")

print("""
The question: Can ANY non-iterative architecture match CTM?

If yes → The iteration isn't essential, just convenient
If no  → Iteration provides something fundamentally different

Key observation from attention analysis:
- CTM shifts attention from b,c to a across ticks
- This is MULTI-PHASE attention (compute b*c first, then add a)
- The phase 2 attention NEEDS phase 1 results to be computed first

This temporal dependency might be the reason iteration is necessary.
""")

print("=" * 80)
