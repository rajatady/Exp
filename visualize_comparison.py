"""
Comparative Visualization: Standard Transformer vs Tick-CTM vs BDH-Style

Inspired by CTM's visualizations, we show:
1. How predictions evolve (over layers for Standard, over ticks for Tick-CTM)
2. Attention patterns
3. Hidden state dynamics
4. When/how the model "gets" the answer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import random
import math

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Device
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

print(f"Device: {device}")


# =============================================================================
# MODELS WITH TRACKING
# =============================================================================

class StandardTransformerWithTracking(nn.Module):
    """Standard Transformer that tracks intermediate states"""
    def __init__(self, vocab_size, d_model=64, n_layers=4, n_heads=4):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_enc = nn.Parameter(torch.randn(1, 512, d_model) * 0.02)

        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads,
                dim_feedforward=d_model * 4, batch_first=True
            ) for _ in range(n_layers)
        ])

        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
        self.name = "Standard Transformer"

    def forward(self, x, track=False):
        B, T = x.shape
        h = self.embed(x) + self.pos_enc[:, :T, :]

        # Causal mask
        mask = torch.triu(torch.ones(T, T, device=x.device) * float('-inf'), diagonal=1)

        # Track states and attention per layer
        states = [h.detach().cpu().numpy()]
        attentions = []
        predictions_per_layer = []

        for layer in self.layers:
            h = layer(h, src_mask=mask)
            if track:
                states.append(h.detach().cpu().numpy())
                # Get intermediate prediction
                pred = self.head(self.ln(h))
                predictions_per_layer.append(F.softmax(pred, dim=-1).detach().cpu().numpy())

        logits = self.head(self.ln(h))

        if track:
            return logits, {
                'states': np.array(states),  # (n_layers+1, B, T, D)
                'predictions': np.array(predictions_per_layer),  # (n_layers, B, T, V)
            }
        return logits


class TickCTMWithTracking(nn.Module):
    """Tick-based CTM that tracks intermediate states"""
    def __init__(self, vocab_size, d_model=64, n_layers=2, n_heads=4, n_ticks=8):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_ticks = n_ticks

        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_enc = nn.Parameter(torch.randn(1, 512, d_model) * 0.02)

        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads,
                dim_feedforward=d_model * 4, batch_first=True
            ) for _ in range(n_layers)
        ])

        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
        self.name = "Tick-CTM"

    def forward(self, x, track=False):
        B, T = x.shape
        h = self.embed(x) + self.pos_enc[:, :T, :]

        mask = torch.triu(torch.ones(T, T, device=x.device) * float('-inf'), diagonal=1)

        # Track states per tick
        states_per_tick = [h.detach().cpu().numpy()]
        predictions_per_tick = []
        certainties_per_tick = []

        for tick in range(self.n_ticks):
            h_new = h
            for layer in self.layers:
                h_new = layer(h_new, src_mask=mask)
            h = h + h_new  # Accumulation

            if track:
                states_per_tick.append(h.detach().cpu().numpy())
                pred = self.head(self.ln(h))
                probs = F.softmax(pred, dim=-1)
                predictions_per_tick.append(probs.detach().cpu().numpy())
                # Certainty = 1 - normalized entropy
                entropy = -(probs * torch.log(probs + 1e-10)).sum(-1)
                max_entropy = np.log(pred.size(-1))
                certainty = 1 - (entropy / max_entropy)
                certainties_per_tick.append(certainty.detach().cpu().numpy())

        logits = self.head(self.ln(h))

        if track:
            return logits, {
                'states': np.array(states_per_tick),  # (n_ticks+1, B, T, D)
                'predictions': np.array(predictions_per_tick),  # (n_ticks, B, T, V)
                'certainties': np.array(certainties_per_tick),  # (n_ticks, B, T)
            }
        return logits


# =============================================================================
# DATA
# =============================================================================

def generate_reversal_data(n_samples, min_len=3, max_len=6):
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

def train_model(model, train_data, n_epochs=150, lr=1e-3):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    for epoch in range(n_epochs):
        model.train()
        random.shuffle(train_data)
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


# =============================================================================
# VISUALIZATION
# =============================================================================

def visualize_standard_transformer(model, sample, ax_pred, ax_states, ax_title):
    """Visualize Standard Transformer: predictions per layer"""
    model.eval()
    with torch.no_grad():
        x = torch.tensor([sample[:-1]], device=device)
        target = sample[1:]
        logits, tracking = model(x, track=True)

        predictions = tracking['predictions']  # (n_layers, 1, T, V)
        states = tracking['states']  # (n_layers+1, 1, T, D)

        # Plot predictions over layers for reversal positions
        sep_pos = sample.index(3)  # SEP token
        reversal_start = sep_pos
        reversal_end = len(sample) - 2  # Before EOS

        n_layers = predictions.shape[0]
        correct_probs = []

        for layer_idx in range(n_layers):
            layer_correct = []
            for pos in range(reversal_start, reversal_end):
                target_token = target[pos]
                prob = predictions[layer_idx, 0, pos, target_token]
                layer_correct.append(prob)
            correct_probs.append(np.mean(layer_correct))

        ax_pred.plot(range(1, n_layers + 1), correct_probs, 'b-o', linewidth=2, markersize=8)
        ax_pred.set_xlabel('Layer')
        ax_pred.set_ylabel('Avg Prob of Correct Token')
        ax_pred.set_title('Prediction Confidence per Layer')
        ax_pred.set_ylim([0, 1])
        ax_pred.grid(True, alpha=0.3)

        # Plot state evolution (mean activation per layer)
        mean_activations = [np.mean(np.abs(states[i])) for i in range(states.shape[0])]
        ax_states.plot(range(len(mean_activations)), mean_activations, 'g-o', linewidth=2, markersize=8)
        ax_states.set_xlabel('Layer')
        ax_states.set_ylabel('Mean |Activation|')
        ax_states.set_title('Activation Magnitude per Layer')
        ax_states.grid(True, alpha=0.3)

        ax_title.text(0.5, 0.5, 'Standard Transformer\n(4 layers)',
                     ha='center', va='center', fontsize=14, fontweight='bold')
        ax_title.axis('off')


def visualize_tick_ctm(model, sample, ax_pred, ax_cert, ax_states, ax_title):
    """Visualize Tick-CTM: predictions and certainty per tick"""
    model.eval()
    with torch.no_grad():
        x = torch.tensor([sample[:-1]], device=device)
        target = sample[1:]
        logits, tracking = model(x, track=True)

        predictions = tracking['predictions']  # (n_ticks, 1, T, V)
        certainties = tracking['certainties']  # (n_ticks, 1, T)
        states = tracking['states']  # (n_ticks+1, 1, T, D)

        sep_pos = sample.index(3)
        reversal_start = sep_pos
        reversal_end = len(sample) - 2

        n_ticks = predictions.shape[0]
        correct_probs = []
        avg_certainties = []

        for tick_idx in range(n_ticks):
            tick_correct = []
            tick_cert = []
            for pos in range(reversal_start, reversal_end):
                target_token = target[pos]
                prob = predictions[tick_idx, 0, pos, target_token]
                tick_correct.append(prob)
                tick_cert.append(certainties[tick_idx, 0, pos])
            correct_probs.append(np.mean(tick_correct))
            avg_certainties.append(np.mean(tick_cert))

        # Plot prediction confidence
        ax_pred.plot(range(1, n_ticks + 1), correct_probs, 'b-o', linewidth=2, markersize=6)
        ax_pred.set_xlabel('Tick')
        ax_pred.set_ylabel('Avg Prob of Correct Token')
        ax_pred.set_title('Prediction Confidence per Tick')
        ax_pred.set_ylim([0, 1])
        ax_pred.grid(True, alpha=0.3)

        # Plot certainty
        ax_cert.plot(range(1, n_ticks + 1), avg_certainties, 'r-o', linewidth=2, markersize=6)
        ax_cert.set_xlabel('Tick')
        ax_cert.set_ylabel('Certainty (1 - Norm Entropy)')
        ax_cert.set_title('Model Certainty per Tick')
        ax_cert.set_ylim([0, 1])
        ax_cert.grid(True, alpha=0.3)

        # Plot state evolution
        mean_activations = [np.mean(np.abs(states[i])) for i in range(states.shape[0])]
        ax_states.plot(range(len(mean_activations)), mean_activations, 'g-o', linewidth=2, markersize=6)
        ax_states.set_xlabel('Tick')
        ax_states.set_ylabel('Mean |Activation|')
        ax_states.set_title('Activation Magnitude per Tick')
        ax_states.grid(True, alpha=0.3)

        ax_title.text(0.5, 0.5, 'Tick-CTM\n(2 layers x 8 ticks)',
                     ha='center', va='center', fontsize=14, fontweight='bold')
        ax_title.axis('off')


def visualize_comparison(std_model, tick_model, sample):
    """Create side-by-side comparison visualization"""
    fig = plt.figure(figsize=(16, 10))

    # Standard Transformer (left column)
    ax_std_title = fig.add_subplot(3, 2, 1)
    ax_std_pred = fig.add_subplot(3, 2, 3)
    ax_std_states = fig.add_subplot(3, 2, 5)

    # Tick-CTM (right column)
    ax_tick_title = fig.add_subplot(3, 2, 2)
    ax_tick_pred = fig.add_subplot(3, 2, 4)
    ax_tick_cert = fig.add_subplot(3, 2, 6)

    visualize_standard_transformer(std_model, sample, ax_std_pred, ax_std_states, ax_std_title)

    # For Tick-CTM, use cert plot instead of states
    tick_model.eval()
    with torch.no_grad():
        x = torch.tensor([sample[:-1]], device=device)
        target = sample[1:]
        logits, tracking = tick_model(x, track=True)

        predictions = tracking['predictions']
        certainties = tracking['certainties']

        sep_pos = sample.index(3)
        reversal_start = sep_pos
        reversal_end = len(sample) - 2

        n_ticks = predictions.shape[0]
        correct_probs = []
        avg_certainties = []

        for tick_idx in range(n_ticks):
            tick_correct = []
            tick_cert = []
            for pos in range(reversal_start, reversal_end):
                target_token = target[pos]
                prob = predictions[tick_idx, 0, pos, target_token]
                tick_correct.append(prob)
                tick_cert.append(certainties[tick_idx, 0, pos])
            correct_probs.append(np.mean(tick_correct))
            avg_certainties.append(np.mean(tick_cert))

        ax_tick_pred.plot(range(1, n_ticks + 1), correct_probs, 'b-o', linewidth=2, markersize=6)
        ax_tick_pred.set_xlabel('Tick')
        ax_tick_pred.set_ylabel('Avg Prob of Correct Token')
        ax_tick_pred.set_title('Prediction Confidence per Tick')
        ax_tick_pred.set_ylim([0, 1])
        ax_tick_pred.grid(True, alpha=0.3)

        ax_tick_cert.plot(range(1, n_ticks + 1), avg_certainties, 'r-o', linewidth=2, markersize=6)
        ax_tick_cert.set_xlabel('Tick')
        ax_tick_cert.set_ylabel('Certainty')
        ax_tick_cert.set_title('Model Certainty per Tick')
        ax_tick_cert.set_ylim([0, 1])
        ax_tick_cert.grid(True, alpha=0.3)

        ax_tick_title.text(0.5, 0.5, 'Tick-CTM\n(2 layers x 8 ticks)',
                          ha='center', va='center', fontsize=14, fontweight='bold')
        ax_tick_title.axis('off')

    # Add sample info
    sep_pos = sample.index(3)
    input_seq = sample[1:sep_pos]
    output_seq = sample[sep_pos+1:-1]
    token_map = {i+4: str(i) for i in range(10)}
    input_str = ''.join([token_map.get(t, '?') for t in input_seq])
    output_str = ''.join([token_map.get(t, '?') for t in output_seq])

    fig.suptitle(f'Reversal Task: "{input_str}" → "{output_str}"', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('comparison_visualization.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved to comparison_visualization.png")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Training models for visualization...")
    print("=" * 60)

    # Generate data
    train_data = generate_reversal_data(500, min_len=3, max_len=6)
    test_data = generate_reversal_data(20, min_len=4, max_len=5)

    VOCAB_SIZE = 14
    D_MODEL = 64

    # Create models
    std_model = StandardTransformerWithTracking(VOCAB_SIZE, D_MODEL, n_layers=4).to(device)
    tick_model = TickCTMWithTracking(VOCAB_SIZE, D_MODEL, n_layers=2, n_ticks=8).to(device)

    print(f"\nStandard Transformer: {sum(p.numel() for p in std_model.parameters()):,} params")
    print(f"Tick-CTM: {sum(p.numel() for p in tick_model.parameters()):,} params")

    # Train
    print("\nTraining Standard Transformer...")
    std_model = train_model(std_model, train_data, n_epochs=150)

    print("Training Tick-CTM...")
    tick_model = train_model(tick_model, train_data, n_epochs=150)

    # Visualize on a test sample
    print("\n" + "=" * 60)
    print("Creating visualization...")
    print("=" * 60)

    sample = test_data[0]
    visualize_comparison(std_model, tick_model, sample)

    print("\nDone!")
