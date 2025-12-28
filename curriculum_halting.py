"""
CURRICULUM-BASED ADAPTIVE HALTING

The hypothesis: Model doesn't learn to halt early because more ticks = better accuracy.
The fix: Train with 1 tick first, then ADD ticks when 1 isn't enough.

This forces the model to:
1. First learn to solve with minimal compute
2. Then learn WHEN to use more compute
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
print("CURRICULUM-BASED ADAPTIVE HALTING EXPERIMENT")
print("=" * 80)

# =============================================================================
# CTM WITH CURRICULUM HALTING
# =============================================================================

class CTMCurriculum(nn.Module):
    """
    CTM that learns to halt through curriculum:
    - Epoch 0-10: max 1 tick (MUST solve with 1)
    - Epoch 11-20: max 2 ticks (can use 2 if needed)
    - Epoch 21-30: max 3 ticks
    - Epoch 31+: max 4 ticks

    Key insight: The model learns to solve FIRST, then learns to think more.
    """
    def __init__(self, vocab_size, hidden_dim=64, n_layers=2, n_heads=4,
                 max_ticks=4, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_ticks = max_ticks
        self.n_layers = n_layers

        # Embeddings
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embedding = nn.Embedding(128, hidden_dim)

        # Transformer blocks (shared across ticks)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=n_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                batch_first=True
            ) for _ in range(n_layers)
        ])

        # History processor (key CTM component)
        self.history_processor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Per-position halting predictor
        self.halt_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

        # Output
        self.ln_f = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, vocab_size)

        # Track halting stats
        self.avg_ticks_used = 0

    def forward(self, x, current_max_ticks=None):
        """
        current_max_ticks: Curriculum-controlled maximum (changes during training)
        """
        if current_max_ticks is None:
            current_max_ticks = self.max_ticks

        batch_size, seq_len = x.shape

        # Initial embedding
        pos = torch.arange(seq_len, device=x.device).unsqueeze(0)
        h = self.embedding(x) + self.pos_embedding(pos)

        # History buffer
        history = [torch.zeros_like(h), torch.zeros_like(h)]

        # Halting state
        halted = torch.zeros(batch_size, seq_len, 1, device=x.device)
        accumulated_output = torch.zeros_like(h)
        remaining_prob = torch.ones(batch_size, seq_len, 1, device=x.device)

        ticks_used = torch.zeros(batch_size, seq_len, device=x.device)

        for tick in range(current_max_ticks):
            # Process history
            history_cat = torch.cat([history[0], history[1]], dim=-1)
            history_features = self.history_processor(history_cat)

            # Add history to hidden state
            h_with_history = h + 0.1 * history_features

            # Run through blocks
            h_new = h_with_history
            for block in self.blocks:
                h_new = block(h_new)

            # Residual connection
            h = h + h_new

            # Compute halt probability
            halt_prob = self.halt_predictor(h)  # [batch, seq, 1]

            # For positions not yet halted
            still_running = 1.0 - halted

            # How much probability to emit at this tick
            emit_prob = remaining_prob * halt_prob

            # Accumulate output weighted by emit probability
            accumulated_output = accumulated_output + emit_prob * h

            # Update remaining probability
            remaining_prob = remaining_prob * (1.0 - halt_prob)

            # Update halted status (halt if remaining prob < 0.01)
            newly_halted = (remaining_prob < 0.01).float()
            halted = torch.max(halted, newly_halted)

            # Track ticks used
            ticks_used = ticks_used + still_running.squeeze(-1)

            # Update history
            history[1] = history[0]
            history[0] = h.detach()

            # Early exit if all positions halted
            if halted.all():
                break

        # If still running after max ticks, emit remaining probability
        accumulated_output = accumulated_output + remaining_prob * h

        # Track average ticks
        self.avg_ticks_used = ticks_used.mean().item()

        # Output
        h_out = self.ln_f(accumulated_output)
        logits = self.head(h_out)

        # Ponder cost for regularization (encourage early halting)
        ponder_cost = ticks_used.mean()

        return logits, ponder_cost


class StandardTransformer(nn.Module):
    """Baseline for comparison"""
    def __init__(self, vocab_size, hidden_dim=64, n_layers=2, n_heads=4, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embedding = nn.Embedding(128, hidden_dim)

        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=n_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                batch_first=True
            ) for _ in range(n_layers)
        ])

        self.ln_f = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        pos = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
        h = self.embedding(x) + self.pos_embedding(pos)

        for block in self.blocks:
            h = block(h)

        h = self.ln_f(h)
        return self.head(h), 0


# =============================================================================
# MIXED DIFFICULTY DATASET
# =============================================================================

def generate_mixed_difficulty_data(n_samples, vocab_size=20):
    """
    Generate data with VARYING difficulty:
    - Easy: length 2-3 reversal (should need 1-2 ticks)
    - Medium: length 4-5 reversal (should need 2-3 ticks)
    - Hard: length 6-8 reversal (should need 3-4 ticks)

    This is crucial for curriculum learning - model must learn to
    allocate compute proportional to difficulty.
    """
    data = []
    labels = []
    difficulties = []

    for i in range(n_samples):
        # Vary difficulty
        difficulty = random.choice(['easy', 'medium', 'hard'])

        if difficulty == 'easy':
            length = random.randint(2, 3)
        elif difficulty == 'medium':
            length = random.randint(4, 5)
        else:
            length = random.randint(6, 8)

        # Generate sequence to reverse
        seq = [random.randint(1, vocab_size-2) for _ in range(length)]

        # Input: SEQ + SEP + padding
        # Output: REVERSED + padding
        sep_token = vocab_size - 1
        pad_token = 0

        max_len = 10
        inp = seq + [sep_token] + [pad_token] * (max_len - length - 1)
        target = seq[::-1] + [pad_token] * (max_len - length)

        data.append(inp)
        labels.append(target)
        difficulties.append(difficulty)

    return torch.tensor(data), torch.tensor(labels), difficulties


# =============================================================================
# CURRICULUM TRAINING
# =============================================================================

def get_curriculum_max_ticks(epoch, max_ticks=4):
    """
    Curriculum schedule:
    - Epochs 0-9: 1 tick (learn to solve minimally)
    - Epochs 10-19: 2 ticks (can use more if needed)
    - Epochs 20-29: 3 ticks
    - Epochs 30+: 4 ticks (full capacity)
    """
    if epoch < 10:
        return 1
    elif epoch < 20:
        return 2
    elif epoch < 30:
        return 3
    else:
        return max_ticks


def train_with_curriculum(model, train_data, train_labels, difficulties,
                          n_epochs=50, lr=0.001, ponder_weight=0.01, use_curriculum=True):
    """
    Train with curriculum-based tick allocation.

    Key: ponder_weight is LOWER than before (0.01 vs 0.1)
    Because curriculum already constrains compute - we don't need heavy regularization.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    history = {
        'epoch': [],
        'train_acc': [],
        'avg_ticks': [],
        'easy_acc': [],
        'medium_acc': [],
        'hard_acc': [],
        'max_ticks_allowed': []
    }

    for epoch in range(n_epochs):
        model.train()

        # Curriculum: control max ticks
        if use_curriculum:
            current_max_ticks = get_curriculum_max_ticks(epoch)
        else:
            current_max_ticks = getattr(model, 'max_ticks', 4)

        # Forward pass
        if hasattr(model, 'halt_predictor'):
            logits, ponder_cost = model(train_data, current_max_ticks)
        else:
            logits, ponder_cost = model(train_data)

        # Loss
        loss = criterion(logits.view(-1, logits.size(-1)), train_labels.view(-1))
        loss = loss + ponder_weight * ponder_cost

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Evaluate
        model.eval()
        with torch.no_grad():
            if hasattr(model, 'halt_predictor'):
                # At eval, use FULL ticks to see true capability
                logits_eval, _ = model(train_data, model.max_ticks)
            else:
                logits_eval, _ = model(train_data)

            preds = logits_eval.argmax(dim=-1)

            # Overall accuracy
            mask = train_labels != 0
            correct = (preds == train_labels) & mask
            acc = correct.sum().float() / mask.sum().float()

            # Per-difficulty accuracy
            easy_mask = [d == 'easy' for d in difficulties]
            medium_mask = [d == 'medium' for d in difficulties]
            hard_mask = [d == 'hard' for d in difficulties]

            def difficulty_acc(diff_mask):
                if sum(diff_mask) == 0:
                    return 0.0
                idx = torch.tensor([i for i, m in enumerate(diff_mask) if m])
                sub_preds = preds[idx]
                sub_labels = train_labels[idx]
                sub_mask = sub_labels != 0
                if sub_mask.sum() == 0:
                    return 0.0
                return ((sub_preds == sub_labels) & sub_mask).sum().float() / sub_mask.sum().float()

            easy_acc = difficulty_acc(easy_mask)
            medium_acc = difficulty_acc(medium_mask)
            hard_acc = difficulty_acc(hard_mask)

            avg_ticks = model.avg_ticks_used if hasattr(model, 'avg_ticks_used') else 0

        history['epoch'].append(epoch)
        history['train_acc'].append(acc.item() * 100)
        history['avg_ticks'].append(avg_ticks)
        history['easy_acc'].append(easy_acc.item() * 100 if isinstance(easy_acc, torch.Tensor) else easy_acc * 100)
        history['medium_acc'].append(medium_acc.item() * 100 if isinstance(medium_acc, torch.Tensor) else medium_acc * 100)
        history['hard_acc'].append(hard_acc.item() * 100 if isinstance(hard_acc, torch.Tensor) else hard_acc * 100)
        history['max_ticks_allowed'].append(current_max_ticks)

        if epoch % 10 == 0 or epoch == n_epochs - 1:
            print(f"Epoch {epoch:3d} | MaxTicks: {current_max_ticks} | "
                  f"Acc: {acc.item()*100:.1f}% | AvgTicks: {avg_ticks:.2f} | "
                  f"Easy: {history['easy_acc'][-1]:.1f}% | Hard: {history['hard_acc'][-1]:.1f}%")

    return history


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

print("\n" + "=" * 80)
print("EXPERIMENT 1: Curriculum vs No-Curriculum CTM")
print("=" * 80)

vocab_size = 20
n_train = 500

# Generate mixed difficulty data
train_data, train_labels, difficulties = generate_mixed_difficulty_data(n_train, vocab_size)

print(f"\nDataset: {n_train} samples")
print(f"  Easy (len 2-3): {sum(1 for d in difficulties if d == 'easy')}")
print(f"  Medium (len 4-5): {sum(1 for d in difficulties if d == 'medium')}")
print(f"  Hard (len 6-8): {sum(1 for d in difficulties if d == 'hard')}")

# Train CTM with curriculum
print("\n--- CTM with Curriculum ---")
ctm_curriculum = CTMCurriculum(vocab_size, hidden_dim=64, n_layers=2, n_heads=4, max_ticks=4)
history_curriculum = train_with_curriculum(
    ctm_curriculum, train_data, train_labels, difficulties,
    n_epochs=50, ponder_weight=0.01, use_curriculum=True
)

# Train CTM without curriculum (fixed max ticks)
print("\n--- CTM without Curriculum (Fixed 4 ticks) ---")
ctm_no_curriculum = CTMCurriculum(vocab_size, hidden_dim=64, n_layers=2, n_heads=4, max_ticks=4)
history_no_curriculum = train_with_curriculum(
    ctm_no_curriculum, train_data, train_labels, difficulties,
    n_epochs=50, ponder_weight=0.01, use_curriculum=False
)

# Train Standard Transformer
print("\n--- Standard Transformer (2 layers) ---")
transformer = StandardTransformer(vocab_size, hidden_dim=64, n_layers=2, n_heads=4)
history_transformer = train_with_curriculum(
    transformer, train_data, train_labels, difficulties,
    n_epochs=50, ponder_weight=0.0, use_curriculum=False
)

# =============================================================================
# ANALYSIS
# =============================================================================

print("\n" + "=" * 80)
print("RESULTS ANALYSIS")
print("=" * 80)

print("\n--- Final Performance ---")
print(f"                          Accuracy   Easy    Medium   Hard    AvgTicks")
print(f"CTM (Curriculum):         {history_curriculum['train_acc'][-1]:5.1f}%    "
      f"{history_curriculum['easy_acc'][-1]:5.1f}%  {history_curriculum['medium_acc'][-1]:5.1f}%  "
      f"{history_curriculum['hard_acc'][-1]:5.1f}%   {history_curriculum['avg_ticks'][-1]:.2f}")
print(f"CTM (No Curriculum):      {history_no_curriculum['train_acc'][-1]:5.1f}%    "
      f"{history_no_curriculum['easy_acc'][-1]:5.1f}%  {history_no_curriculum['medium_acc'][-1]:5.1f}%  "
      f"{history_no_curriculum['hard_acc'][-1]:5.1f}%   {history_no_curriculum['avg_ticks'][-1]:.2f}")
print(f"Standard Transformer:     {history_transformer['train_acc'][-1]:5.1f}%    "
      f"{history_transformer['easy_acc'][-1]:5.1f}%  {history_transformer['medium_acc'][-1]:5.1f}%  "
      f"{history_transformer['hard_acc'][-1]:5.1f}%   N/A")

# =============================================================================
# TEST: DOES CTM LEARN ADAPTIVE COMPUTE?
# =============================================================================

print("\n" + "=" * 80)
print("EXPERIMENT 2: Does CTM Learn Difficulty-Adaptive Halting?")
print("=" * 80)

# Generate fresh test data
test_data, test_labels, test_difficulties = generate_mixed_difficulty_data(300, vocab_size)

# Separate by difficulty
easy_idx = [i for i, d in enumerate(test_difficulties) if d == 'easy']
medium_idx = [i for i, d in enumerate(test_difficulties) if d == 'medium']
hard_idx = [i for i, d in enumerate(test_difficulties) if d == 'hard']

def analyze_ticks_by_difficulty(model, data, difficulties, name):
    """Analyze average ticks used per difficulty level."""
    model.eval()

    # Run with full ticks
    with torch.no_grad():
        logits, _ = model(data, model.max_ticks)

        # Access internal halting state - need to rerun and track
        # For now, use the avg_ticks_used from last forward pass

        print(f"\n{name}:")
        print(f"  Overall avg ticks: {model.avg_ticks_used:.2f}")

        # Run separately for each difficulty to get avg ticks
        for diff_name, idx_list in [('Easy', easy_idx), ('Medium', medium_idx), ('Hard', hard_idx)]:
            if len(idx_list) == 0:
                continue
            idx = torch.tensor(idx_list)
            sub_data = data[idx]

            _, _ = model(sub_data, model.max_ticks)
            avg = model.avg_ticks_used
            print(f"  {diff_name:8s} ({len(idx_list):3d} samples): {avg:.2f} avg ticks")

analyze_ticks_by_difficulty(ctm_curriculum, test_data, test_difficulties, "CTM (Curriculum)")
analyze_ticks_by_difficulty(ctm_no_curriculum, test_data, test_difficulties, "CTM (No Curriculum)")

# =============================================================================
# ECONOMIC ANALYSIS
# =============================================================================

print("\n" + "=" * 80)
print("ECONOMIC ANALYSIS: Is Curriculum CTM Cost-Effective?")
print("=" * 80)

print("""
The Key Question: Does curriculum training teach the model to allocate
compute proportionally to difficulty?

IDEAL BEHAVIOR:
- Easy tasks (length 2-3): ~1-2 ticks
- Medium tasks (length 4-5): ~2-3 ticks
- Hard tasks (length 6-8): ~3-4 ticks

If this works, CTM is economically BETTER than fixed-compute transformer:
- Same average FLOPs (weighted by task distribution)
- Higher peak capability on hard tasks
- Lower cost on easy tasks (energy savings)
""")

# Final verdict
curriculum_ticks = history_curriculum['avg_ticks'][-1]
no_curriculum_ticks = history_no_curriculum['avg_ticks'][-1]

print(f"\nFINAL VERDICT:")
print(f"  Curriculum CTM avg ticks: {curriculum_ticks:.2f}")
print(f"  No-Curriculum CTM avg ticks: {no_curriculum_ticks:.2f}")

if curriculum_ticks < no_curriculum_ticks * 0.9:
    print(f"  → Curriculum REDUCED compute by {(1 - curriculum_ticks/no_curriculum_ticks)*100:.1f}%")
    print(f"  → Curriculum learning WORKS for adaptive halting!")
else:
    print(f"  → Curriculum did NOT significantly reduce compute")
    print(f"  → Need different approach for adaptive halting")

# =============================================================================
# WHAT'S NEXT
# =============================================================================

print("\n" + "=" * 80)
print("WHAT'S NEXT")
print("=" * 80)

print("""
If curriculum works:
  → CTM has solved the "speed tax" problem
  → Same average cost as Transformer, higher peak capability
  → Path to Transformer obsolescence is clear

If curriculum doesn't work:
  → Need fundamentally different halting mechanism
  → Options:
    a) Explicit "done" token the model must output
    b) Confidence-based halting (output entropy threshold)
    c) Auxiliary halting reward (RL-style)
    d) Mixture of depths (different ticks for different positions)

Let's also check generalization...
""")

# Quick generalization test
print("\n--- Generalization Test (unseen lengths 9-10) ---")
def generate_ood_data(n_samples, vocab_size=20):
    data = []
    labels = []
    for _ in range(n_samples):
        length = random.randint(9, 10)  # Longer than training
        seq = [random.randint(1, vocab_size-2) for _ in range(length)]
        sep_token = vocab_size - 1
        pad_token = 0
        max_len = 12
        inp = seq + [sep_token] + [pad_token] * (max_len - length - 1)
        target = seq[::-1] + [pad_token] * (max_len - length)
        data.append(inp)
        labels.append(target)
    return torch.tensor(data), torch.tensor(labels)

ood_data, ood_labels = generate_ood_data(100, vocab_size)

for name, model in [("CTM Curriculum", ctm_curriculum),
                    ("CTM No-Curriculum", ctm_no_curriculum),
                    ("Transformer", transformer)]:
    model.eval()
    with torch.no_grad():
        if hasattr(model, 'halt_predictor'):
            logits, _ = model(ood_data, 4)  # Full ticks for OOD
        else:
            logits, _ = model(ood_data)
        preds = logits.argmax(dim=-1)
        mask = ood_labels != 0
        correct = (preds == ood_labels) & mask
        acc = correct.sum().float() / mask.sum().float()
        print(f"  {name:20s}: {acc.item()*100:.1f}%")

print("\n" + "=" * 80)
print("EXPERIMENT COMPLETE")
print("=" * 80)
