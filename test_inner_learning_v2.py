"""
Experiment 1b: Inner Loop Learning - With Transformer Baseline

Fixed confounds from Experiment 1:
1. Using Transformer as baseline (not MLP)
2. Analyzing HOW each model solves the task
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

print("=" * 70)
print("INNER LOOP LEARNING v2: With Transformer Baseline")
print("=" * 70)


# =============================================================================
# TASK: Pattern Completion (same as before)
# =============================================================================

def generate_patterns(n_samples, pattern_len=8, mask_ratio=0.5):
    data = []
    for _ in range(n_samples):
        pattern = []
        val = random.randint(0, 1)
        for i in range(pattern_len):
            if random.random() < 0.7:
                val = 1 - val
            pattern.append(val)

        masked = pattern.copy()
        n_mask = int(pattern_len * mask_ratio)
        mask_positions = random.sample(range(pattern_len), n_mask)
        for pos in mask_positions:
            masked[pos] = -1

        data.append({
            'input': masked,
            'target': pattern,
            'mask_positions': mask_positions
        })
    return data


# =============================================================================
# MODEL 1: Standard Transformer
# =============================================================================

class TransformerBaseline(nn.Module):
    def __init__(self, pattern_len, hidden_dim=64, n_heads=4, n_layers=2):
        super().__init__()
        self.embed = nn.Embedding(3, hidden_dim)  # 0, 1, mask(-1->2)
        self.pos_embed = nn.Parameter(torch.randn(1, pattern_len, hidden_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads,
            dim_feedforward=hidden_dim*4, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.output = nn.Linear(hidden_dim, 2)
        self.pattern_len = pattern_len

        # For analysis: store attention weights
        self.last_attention = None

    def forward(self, x, return_attention=False):
        x_embed = x.clone()
        x_embed[x_embed == -1] = 2
        h = self.embed(x_embed) + self.pos_embed
        h = self.transformer(h)
        out = self.output(h)
        return out


# =============================================================================
# MODEL 2: Transformer + Dynamics (state evolves over ticks)
# =============================================================================

class TransformerDynamics(nn.Module):
    def __init__(self, pattern_len, hidden_dim=64, n_heads=4, n_layers=2, n_ticks=4):
        super().__init__()
        self.embed = nn.Embedding(3, hidden_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, pattern_len, hidden_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads,
            dim_feedforward=hidden_dim*4, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.output = nn.Linear(hidden_dim, 2)
        self.n_ticks = n_ticks

        # For analysis: store state evolution
        self.state_history = None

    def forward(self, x, return_states=False):
        x_embed = x.clone()
        x_embed[x_embed == -1] = 2
        h = self.embed(x_embed) + self.pos_embed

        states = [h.clone()]
        for tick in range(self.n_ticks):
            h_new = self.transformer(h)
            h = h + 0.5 * h_new  # Accumulation
            states.append(h.clone())

        self.state_history = states
        out = self.output(h)

        if return_states:
            return out, states
        return out


# =============================================================================
# MODEL 3: Transformer + Hebbian (weights change at inference)
# =============================================================================

class TransformerHebbian(nn.Module):
    def __init__(self, pattern_len, hidden_dim=64, n_heads=4, n_ticks=4, hebbian_lr=0.001):
        super().__init__()
        self.embed = nn.Embedding(3, hidden_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, pattern_len, hidden_dim) * 0.02)
        self.hidden_dim = hidden_dim
        self.n_ticks = n_ticks
        self.hebbian_lr = hebbian_lr

        # Simple attention (we'll modify Q,K weights with Hebbian)
        self.W_q = nn.Linear(hidden_dim, hidden_dim)
        self.W_k = nn.Linear(hidden_dim, hidden_dim)
        self.W_v = nn.Linear(hidden_dim, hidden_dim)
        self.W_o = nn.Linear(hidden_dim, hidden_dim)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )

        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.output = nn.Linear(hidden_dim, 2)

        # For analysis
        self.weight_changes = None

    def attention(self, h, W_q_weight, W_k_weight, W_v_weight):
        Q = F.linear(h, W_q_weight, self.W_q.bias)
        K = F.linear(h, W_k_weight, self.W_k.bias)
        V = F.linear(h, W_v_weight, self.W_v.bias)

        scores = torch.bmm(Q, K.transpose(-2, -1)) / math.sqrt(self.hidden_dim)
        attn = F.softmax(scores, dim=-1)
        out = torch.bmm(attn, V)
        return self.W_o(out), attn

    def forward(self, x, return_analysis=False):
        x_embed = x.clone()
        x_embed[x_embed == -1] = 2
        h = self.embed(x_embed) + self.pos_embed

        # Clone weights for this inference
        W_q = self.W_q.weight.clone()
        W_k = self.W_k.weight.clone()
        W_v = self.W_v.weight.clone()

        weight_deltas = []

        for tick in range(self.n_ticks):
            # Attention with current weights
            h_norm = self.ln1(h)
            attn_out, attn_weights = self.attention(h_norm, W_q, W_k, W_v)
            h = h + attn_out

            # FFN
            h = h + self.ffn(self.ln2(h))

            # Hebbian update on Q, K weights
            # Idea: strengthen connections that are used together
            with torch.no_grad():
                Q = F.linear(h_norm, W_q)
                K = F.linear(h_norm, W_k)

                # Hebbian: Δw = η * post ⊗ pre (averaged over batch and positions)
                # For Q weights: pre = h_norm, post = Q
                delta_q = torch.einsum('bpi,bpj->ij', Q, h_norm) / (h.size(0) * h.size(1))
                delta_k = torch.einsum('bpi,bpj->ij', K, h_norm) / (h.size(0) * h.size(1))

                # Normalize to prevent explosion
                delta_q = delta_q / (delta_q.norm() + 1e-8)
                delta_k = delta_k / (delta_k.norm() + 1e-8)

                W_q = W_q + self.hebbian_lr * delta_q
                W_k = W_k + self.hebbian_lr * delta_k

                weight_deltas.append({
                    'delta_q_norm': delta_q.norm().item(),
                    'delta_k_norm': delta_k.norm().item()
                })

        self.weight_changes = weight_deltas
        out = self.output(h)

        if return_analysis:
            return out, weight_deltas
        return out


# =============================================================================
# TRAINING & EVALUATION
# =============================================================================

def train_model(model, train_data, n_epochs=100, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(n_epochs):
        model.train()
        inputs = torch.tensor([d['input'] for d in train_data])
        targets = torch.tensor([d['target'] for d in train_data])

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = F.cross_entropy(outputs.view(-1, 2), targets.view(-1))
        loss.backward()
        optimizer.step()

    return model


def evaluate(model, data):
    model.eval()
    with torch.no_grad():
        inputs = torch.tensor([d['input'] for d in data])
        targets = torch.tensor([d['target'] for d in data])
        mask_positions = [d['mask_positions'] for d in data]

        outputs = model(inputs)
        preds = outputs.argmax(dim=-1)

        # Accuracy on masked positions only
        masked_correct = 0
        masked_total = 0
        for i, positions in enumerate(mask_positions):
            for pos in positions:
                if preds[i, pos] == targets[i, pos]:
                    masked_correct += 1
                masked_total += 1

        return masked_correct / masked_total if masked_total > 0 else 0


def analyze_model(model, data, name):
    """Analyze HOW the model solves the task"""
    model.eval()
    print(f"\n--- Analysis: {name} ---")

    inputs = torch.tensor([d['input'] for d in data[:5]])  # Just 5 examples

    with torch.no_grad():
        if isinstance(model, TransformerDynamics):
            outputs, states = model(inputs, return_states=True)

            # How much does state change per tick?
            print("State evolution (L2 distance between ticks):")
            for i in range(1, len(states)):
                delta = (states[i] - states[i-1]).norm(dim=-1).mean()
                print(f"  Tick {i}: Δ = {delta:.4f}")

        elif isinstance(model, TransformerHebbian):
            outputs, weight_changes = model(inputs, return_analysis=True)

            print("Weight changes per tick:")
            for i, wc in enumerate(weight_changes):
                print(f"  Tick {i+1}: ΔQ = {wc['delta_q_norm']:.4f}, ΔK = {wc['delta_k_norm']:.4f}")

        else:
            outputs = model(inputs)
            print("(Static model - no dynamics to analyze)")


# =============================================================================
# EXPERIMENT
# =============================================================================

PATTERN_LEN = 8
HIDDEN_DIM = 64

print("\n1. Generating data...")
train_data = generate_patterns(500, PATTERN_LEN, mask_ratio=0.5)
test_data = generate_patterns(100, PATTERN_LEN, mask_ratio=0.5)
test_hard = generate_patterns(100, PATTERN_LEN, mask_ratio=0.75)

print(f"   Train: {len(train_data)}, Test: {len(test_data)}, Hard: {len(test_hard)}")

print("\n2. Creating models...")
models = {
    "Transformer": TransformerBaseline(PATTERN_LEN, HIDDEN_DIM),
    "Transformer+Dynamics": TransformerDynamics(PATTERN_LEN, HIDDEN_DIM, n_ticks=4),
    "Transformer+Hebbian": TransformerHebbian(PATTERN_LEN, HIDDEN_DIM, n_ticks=4),
}

for name, model in models.items():
    n_params = sum(p.numel() for p in model.parameters())
    print(f"   {name}: {n_params:,} params")

print("\n3. Training...")
results = {}
for name, model in models.items():
    print(f"   Training {name}...")
    model = train_model(model, train_data, n_epochs=200)

    test_acc = evaluate(model, test_data)
    hard_acc = evaluate(model, test_hard)

    results[name] = {'test': test_acc, 'hard': hard_acc}
    print(f"   → Test: {test_acc:.1%}, Hard: {hard_acc:.1%}")

print("\n" + "=" * 70)
print("RESULTS")
print("=" * 70)
print(f"\n{'Model':<25} {'Test (50% masked)':<20} {'Hard (75% masked)':<20}")
print("-" * 65)
for name in models:
    r = results[name]
    print(f"{name:<25} {r['test']:<20.1%} {r['hard']:<20.1%}")

print("\n" + "=" * 70)
print("HOW EACH MODEL SOLVES THE TASK")
print("=" * 70)

for name, model in models.items():
    analyze_model(model, test_data, name)

print("\n" + "=" * 70)
print("INTERPRETATION")
print("=" * 70)

base = results["Transformer"]['hard']
dyn = results["Transformer+Dynamics"]['hard']
heb = results["Transformer+Hebbian"]['hard']

print(f"""
On hard test (75% masked):
- Transformer baseline: {base:.1%}
- + Dynamics:           {dyn:.1%} ({'+' if dyn > base else ''}{(dyn-base)*100:.1f}%)
- + Hebbian:            {heb:.1%} ({'+' if heb > base else ''}{(heb-base)*100:.1f}%)

Question: Does adding inner-loop learning help?
""")
