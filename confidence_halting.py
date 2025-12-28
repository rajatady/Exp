"""
CONFIDENCE-BASED ADAPTIVE HALTING

The problem: Learned halting doesn't work because there's no signal for when to halt.
The insight: The model ALREADY has a confidence signal - output entropy!

High entropy = uncertain = keep thinking
Low entropy = confident = can halt

This is emergent - no need to learn a separate halting function.
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
print("CONFIDENCE-BASED ADAPTIVE HALTING")
print("=" * 80)

# =============================================================================
# CTM WITH CONFIDENCE HALTING
# =============================================================================

class CTMConfidenceHalt(nn.Module):
    """
    CTM that halts based on output confidence (entropy).

    Key insight: Instead of learning WHEN to halt, we use the model's
    natural confidence as the halting signal.

    Low entropy at position = model is confident = don't need more thinking
    High entropy = uncertain = keep thinking
    """
    def __init__(self, vocab_size, hidden_dim=64, n_layers=2, n_heads=4,
                 max_ticks=8, entropy_threshold=0.5, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_ticks = max_ticks
        self.n_layers = n_layers
        self.vocab_size = vocab_size
        self.entropy_threshold = entropy_threshold  # Halt if entropy < this

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

        # History processor
        self.history_processor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Output
        self.ln_f = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, vocab_size)

        # Track stats
        self.avg_ticks_used = 0
        self.tick_history = []

    def compute_entropy(self, logits):
        """Compute per-position entropy of output distribution."""
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1)  # [batch, seq]
        # Normalize by max entropy (log vocab_size)
        max_entropy = math.log(self.vocab_size)
        return entropy / max_entropy  # 0 = certain, 1 = uniform

    def forward(self, x, use_adaptive=True):
        batch_size, seq_len = x.shape

        # Initial embedding
        pos = torch.arange(seq_len, device=x.device).unsqueeze(0)
        h = self.embedding(x) + self.pos_embedding(pos)

        # History buffer
        history = [torch.zeros_like(h), torch.zeros_like(h)]

        # Track when each position halted
        position_halted = torch.zeros(batch_size, seq_len, device=x.device)
        position_outputs = torch.zeros(batch_size, seq_len, self.vocab_size, device=x.device)
        ticks_per_position = torch.zeros(batch_size, seq_len, device=x.device)

        for tick in range(self.max_ticks):
            # Process history
            history_cat = torch.cat([history[0], history[1]], dim=-1)
            history_features = self.history_processor(history_cat)

            # Add history to hidden state
            h_with_history = h + 0.1 * history_features

            # Run through blocks
            h_new = h_with_history
            for block in self.blocks:
                h_new = block(h_new)

            # Residual
            h = h + h_new

            # Compute output and entropy
            h_out = self.ln_f(h)
            logits = self.head(h_out)  # [batch, seq, vocab]
            entropy = self.compute_entropy(logits)  # [batch, seq]

            if use_adaptive:
                # For positions not yet halted
                not_halted = (position_halted == 0).float()

                # Update ticks count for running positions
                ticks_per_position = ticks_per_position + not_halted

                # Halt positions where entropy is low
                should_halt = (entropy < self.entropy_threshold).float()
                newly_halted = should_halt * not_halted

                # Store output for newly halted positions
                for b in range(batch_size):
                    for s in range(seq_len):
                        if newly_halted[b, s] > 0:
                            position_outputs[b, s] = logits[b, s]

                # Update halted status
                position_halted = position_halted + newly_halted
                position_halted = torch.clamp(position_halted, 0, 1)

                # Early exit if all positions halted
                if position_halted.all():
                    break

            # Update history
            history[1] = history[0]
            history[0] = h.detach()

        # For positions that never halted, use final output
        if use_adaptive:
            not_halted_mask = (position_halted == 0)
            for b in range(batch_size):
                for s in range(seq_len):
                    if not_halted_mask[b, s]:
                        position_outputs[b, s] = logits[b, s]
                        ticks_per_position[b, s] = self.max_ticks

            self.avg_ticks_used = ticks_per_position.mean().item()
            self.tick_history.append(ticks_per_position.clone())
            return position_outputs, ticks_per_position.mean()
        else:
            self.avg_ticks_used = self.max_ticks
            return logits, 0


class StandardCTM(nn.Module):
    """Standard CTM with fixed ticks for comparison"""
    def __init__(self, vocab_size, hidden_dim=64, n_layers=2, n_heads=4,
                 max_ticks=8, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_ticks = max_ticks

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

        self.history_processor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.ln_f = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, vocab_size)

        self.avg_ticks_used = max_ticks

    def forward(self, x):
        pos = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
        h = self.embedding(x) + self.pos_embedding(pos)

        history = [torch.zeros_like(h), torch.zeros_like(h)]

        for tick in range(self.max_ticks):
            history_cat = torch.cat([history[0], history[1]], dim=-1)
            history_features = self.history_processor(history_cat)
            h_with_history = h + 0.1 * history_features
            h_new = h_with_history
            for block in self.blocks:
                h_new = block(h_new)
            h = h + h_new
            history[1] = history[0]
            history[0] = h.detach()

        h_out = self.ln_f(h)
        return self.head(h_out), 0


# =============================================================================
# DATA GENERATION
# =============================================================================

def generate_mixed_difficulty_data(n_samples, vocab_size=20):
    """Generate reversal data with varying difficulty."""
    data = []
    labels = []
    difficulties = []

    for _ in range(n_samples):
        difficulty = random.choice(['easy', 'medium', 'hard'])

        if difficulty == 'easy':
            length = random.randint(2, 3)
        elif difficulty == 'medium':
            length = random.randint(4, 5)
        else:
            length = random.randint(6, 8)

        seq = [random.randint(1, vocab_size-2) for _ in range(length)]
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
# TRAINING
# =============================================================================

def train_model(model, train_data, train_labels, n_epochs=100, lr=0.001,
                use_adaptive=False, adaptive_after=50):
    """
    Train with optional adaptive halting.

    Key: First train WITHOUT adaptive halting to learn the task.
    Then enable adaptive halting for efficiency.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    history = {'epoch': [], 'acc': [], 'ticks': []}

    for epoch in range(n_epochs):
        model.train()

        # Use adaptive halting only after model has learned the task
        current_adaptive = use_adaptive and (epoch >= adaptive_after)

        if hasattr(model, 'compute_entropy'):
            logits, ponder = model(train_data, use_adaptive=current_adaptive)
        else:
            logits, ponder = model(train_data)

        loss = criterion(logits.view(-1, logits.size(-1)), train_labels.view(-1))
        loss = loss + 0.001 * ponder

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Evaluate
        model.eval()
        with torch.no_grad():
            if hasattr(model, 'compute_entropy'):
                logits, _ = model(train_data, use_adaptive=current_adaptive)
            else:
                logits, _ = model(train_data)

            preds = logits.argmax(dim=-1)
            mask = train_labels != 0
            correct = (preds == train_labels) & mask
            acc = correct.sum().float() / mask.sum().float()

        history['epoch'].append(epoch)
        history['acc'].append(acc.item() * 100)
        history['ticks'].append(model.avg_ticks_used)

        if epoch % 20 == 0 or epoch == n_epochs - 1:
            adaptive_str = "ADAPTIVE" if current_adaptive else "FIXED"
            print(f"Epoch {epoch:3d} | {adaptive_str:8s} | "
                  f"Acc: {acc.item()*100:.1f}% | AvgTicks: {model.avg_ticks_used:.2f}")

    return history


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

print("\n" + "=" * 80)
print("EXPERIMENT: Confidence-Based Adaptive Halting")
print("=" * 80)

vocab_size = 20
n_train = 500
max_ticks = 8

train_data, train_labels, difficulties = generate_mixed_difficulty_data(n_train, vocab_size)

print(f"\nDataset: {n_train} samples")
print(f"  Easy (len 2-3): {sum(1 for d in difficulties if d == 'easy')}")
print(f"  Medium (len 4-5): {sum(1 for d in difficulties if d == 'medium')}")
print(f"  Hard (len 6-8): {sum(1 for d in difficulties if d == 'hard')}")

# Test different entropy thresholds
thresholds = [0.3, 0.5, 0.7]

print("\n--- Training with different entropy thresholds ---")

results = {}
for thresh in thresholds:
    print(f"\n=== Entropy threshold: {thresh} ===")
    model = CTMConfidenceHalt(vocab_size, hidden_dim=64, n_layers=2, n_heads=4,
                               max_ticks=max_ticks, entropy_threshold=thresh)
    history = train_model(model, train_data, train_labels, n_epochs=100,
                          use_adaptive=True, adaptive_after=50)
    results[thresh] = (model, history)

# Compare with fixed-tick CTM
print("\n=== Standard CTM (Fixed 8 ticks) ===")
fixed_model = StandardCTM(vocab_size, hidden_dim=64, n_layers=2, n_heads=4, max_ticks=8)
fixed_history = train_model(fixed_model, train_data, train_labels, n_epochs=100,
                            use_adaptive=False, adaptive_after=100)

# =============================================================================
# ANALYSIS
# =============================================================================

print("\n" + "=" * 80)
print("RESULTS ANALYSIS")
print("=" * 80)

print("\n--- Final Performance ---")
print(f"{'Model':<35} {'Accuracy':>10} {'Avg Ticks':>12}")
print("-" * 60)

for thresh in thresholds:
    model, hist = results[thresh]
    print(f"Confidence Halt (thresh={thresh}):     {hist['acc'][-1]:>8.1f}%  {hist['ticks'][-1]:>10.2f}")

print(f"Fixed 8 Ticks:                      {fixed_history['acc'][-1]:>8.1f}%  {8:>10}")

# Analyze ticks by difficulty for best adaptive model
print("\n" + "=" * 80)
print("TICKS BY DIFFICULTY (for threshold=0.5)")
print("=" * 80)

model = results[0.5][0]
model.eval()
model.tick_history = []

# Generate test data for each difficulty
for diff_name, length_range in [('Easy', (2, 3)), ('Medium', (4, 5)), ('Hard', (6, 8))]:
    test_data = []
    for _ in range(100):
        length = random.randint(*length_range)
        seq = [random.randint(1, vocab_size-2) for _ in range(length)]
        sep_token = vocab_size - 1
        pad_token = 0
        max_len = 10
        inp = seq + [sep_token] + [pad_token] * (max_len - length - 1)
        test_data.append(inp)
    test_data = torch.tensor(test_data)

    with torch.no_grad():
        _, _ = model(test_data, use_adaptive=True)
        avg_ticks = model.avg_ticks_used
        print(f"{diff_name:8s} (len {length_range}): {avg_ticks:.2f} avg ticks")

# =============================================================================
# SPEED-ACCURACY TRADEOFF
# =============================================================================

print("\n" + "=" * 80)
print("SPEED-ACCURACY TRADEOFF")
print("=" * 80)

print("""
The key question: Does confidence-based halting give BETTER speed-accuracy
tradeoff than fixed ticks?

Fixed 8 ticks: 8 passes through network, highest accuracy
Confidence halt: Variable passes, should be:
  - Fast on easy inputs (fewer passes)
  - Still good on hard inputs (more passes used automatically)
""")

# Test fresh data
test_data, test_labels, test_difficulties = generate_mixed_difficulty_data(300, vocab_size)

for thresh in thresholds:
    model = results[thresh][0]
    model.eval()

    with torch.no_grad():
        logits, _ = model(test_data, use_adaptive=True)
        preds = logits.argmax(dim=-1)
        mask = test_labels != 0
        correct = (preds == test_labels) & mask
        acc = correct.sum().float() / mask.sum().float()
        avg_ticks = model.avg_ticks_used

    # Compute efficiency: accuracy per tick
    efficiency = acc.item() * 100 / avg_ticks
    print(f"Threshold {thresh}: Acc={acc.item()*100:.1f}%, Ticks={avg_ticks:.2f}, "
          f"Efficiency={efficiency:.2f}%/tick")

fixed_model.eval()
with torch.no_grad():
    logits, _ = fixed_model(test_data)
    preds = logits.argmax(dim=-1)
    mask = test_labels != 0
    correct = (preds == test_labels) & mask
    acc = correct.sum().float() / mask.sum().float()
efficiency = acc.item() * 100 / 8
print(f"Fixed 8:     Acc={acc.item()*100:.1f}%, Ticks=8.00, "
      f"Efficiency={efficiency:.2f}%/tick")

# =============================================================================
# THE VERDICT
# =============================================================================

print("\n" + "=" * 80)
print("VERDICT: Does Confidence Halting Solve the Speed Tax?")
print("=" * 80)

best_thresh = min(thresholds, key=lambda t: results[t][1]['ticks'][-1])
best_model = results[best_thresh][0]
best_acc = results[best_thresh][1]['acc'][-1]
best_ticks = results[best_thresh][1]['ticks'][-1]

fixed_acc = fixed_history['acc'][-1]

speed_improvement = (8 - best_ticks) / 8 * 100
acc_loss = fixed_acc - best_acc

print(f"""
Best adaptive model (threshold={best_thresh}):
  Accuracy: {best_acc:.1f}%
  Avg ticks: {best_ticks:.2f}
  Speed improvement: {speed_improvement:.1f}% fewer ticks
  Accuracy loss: {acc_loss:.1f}%

Fixed 8-tick model:
  Accuracy: {fixed_acc:.1f}%
  Ticks: 8.00
""")

if speed_improvement > 30 and acc_loss < 5:
    print("✓ SUCCESS: Confidence halting provides significant speed improvement")
    print("  with minimal accuracy loss!")
elif speed_improvement > 10:
    print("△ PARTIAL SUCCESS: Some speed improvement, but room for optimization")
else:
    print("✗ FAILURE: Confidence halting doesn't reduce compute significantly")

print("\n" + "=" * 80)
print("KEY INSIGHT")
print("=" * 80)

print("""
The entropy-based halting is more principled than learned halting because:

1. No additional parameters to learn (just threshold tuning)
2. Natural emergence: confident = halt, uncertain = keep thinking
3. No conflict between task loss and halting loss
4. The model's own uncertainty guides compute allocation

However, there's a fundamental tension:
- Training with fixed ticks means model doesn't learn to be confident early
- Need to train WITH adaptive halting for model to learn to be confident faster

This is why curriculum might still help - train at lower ticks first to
teach the model to be confident with less thinking.
""")

print("=" * 80)
print("EXPERIMENT COMPLETE")
print("=" * 80)
