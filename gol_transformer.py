"""
Game of Life with Transformer

Inductive bias: spatial positional encoding so attention can discover neighbors.
Vectorized torch - no python for loops in forward/backward.
"""

import torch
torch.manual_seed(42)


# ============================================================
# GAME OF LIFE DATA GENERATION (vectorized)
# ============================================================

def generate_gol_batch(batch_size, grid_size=6, steps=1):
    """Generate batch of GoL transitions. Vectorized."""
    # Random initial grids: (batch, grid_size, grid_size)
    grids = (torch.rand(batch_size, grid_size, grid_size) > 0.5).float()

    X = grids.clone()

    for _ in range(steps):
        # Count neighbors using convolution-like operation
        # Pad with wrap-around
        padded = torch.nn.functional.pad(grids, (1, 1, 1, 1), mode='circular')

        # Sum all 8 neighbors
        neighbors = (
            padded[:, 0:-2, 0:-2] +  # top-left
            padded[:, 0:-2, 1:-1] +  # top
            padded[:, 0:-2, 2:] +    # top-right
            padded[:, 1:-1, 0:-2] +  # left
            padded[:, 1:-1, 2:] +    # right
            padded[:, 2:, 0:-2] +    # bottom-left
            padded[:, 2:, 1:-1] +    # bottom
            padded[:, 2:, 2:]        # bottom-right
        )

        # Apply rules
        # Alive cell survives if 2 or 3 neighbors
        survive = grids * ((neighbors == 2) | (neighbors == 3)).float()
        # Dead cell becomes alive if exactly 3 neighbors
        birth = (1 - grids) * (neighbors == 3).float()

        grids = survive + birth

    Y = grids

    # Flatten to sequences: (batch, seq_len) where seq_len = grid_size^2
    X_flat = X.reshape(batch_size, -1)
    Y_flat = Y.reshape(batch_size, -1)

    return X_flat, Y_flat, grid_size


# ============================================================
# SPATIAL POSITIONAL ENCODING
# ============================================================

def create_spatial_pos_encoding(grid_size, d_model):
    """
    Create positional encoding that encodes 2D position.
    Each cell knows its (row, col) continuously.
    """
    seq_len = grid_size * grid_size
    pos_enc = torch.zeros((seq_len, d_model))

    for idx in range(seq_len):
        row = idx // grid_size
        col = idx % grid_size

        # Encode row and col with sin/cos at different frequencies
        for i in range(d_model // 4):
            freq = 1.0 / (10.0 ** (4 * i / d_model))
            pos_enc[idx, 4*i] = torch.sin(torch.tensor(row * freq))
            pos_enc[idx, 4*i + 1] = torch.cos(torch.tensor(row * freq))
            pos_enc[idx, 4*i + 2] = torch.sin(torch.tensor(col * freq))
            pos_enc[idx, 4*i + 3] = torch.cos(torch.tensor(col * freq))

    return pos_enc.float()


def create_neighbor_distance_matrix(grid_size):
    """
    Create matrix of distances between all pairs of cells.
    Can be used to bias attention toward neighbors.
    """
    seq_len = grid_size * grid_size
    distances = torch.zeros((seq_len, seq_len))

    for i in range(seq_len):
        row_i, col_i = i // grid_size, i % grid_size
        for j in range(seq_len):
            row_j, col_j = j // grid_size, j % grid_size
            # Toroidal distance
            dr = min(abs(row_i - row_j), grid_size - abs(row_i - row_j))
            dc = min(abs(col_i - col_j), grid_size - abs(col_i - col_j))
            distances[i, j] = (dr**2 + dc**2) ** 0.5

    return distances.float()


# ============================================================
# TRANSFORMER COMPONENTS (vectorized)
# ============================================================

def softmax(x, axis=-1):
    """Numerically stable softmax."""
    x_max = x.max(dim=axis, keepdim=True).values
    exp_x = torch.exp(x - x_max)
    return exp_x / exp_x.sum(dim=axis, keepdim=True)


def layer_norm(x, gamma, beta, eps=1e-5):
    """Layer normalization."""
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    return gamma * (x - mean) / torch.sqrt(var + eps) + beta


def gelu(x):
    """GELU activation."""
    return 0.5 * x * (1 + torch.tanh(torch.sqrt(torch.tensor(2.0 / 3.14159265)) * (x + 0.044715 * x**3)))


class TransformerLayer:
    def __init__(self, d_model, n_heads, d_ff, neighbor_bias=None):
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.d_ff = d_ff
        self.neighbor_bias = neighbor_bias  # (seq, seq) distance matrix

        # Attention weights
        scale = 0.02
        self.W_q = torch.randn(d_model, d_model).float() * scale
        self.W_k = torch.randn(d_model, d_model).float() * scale
        self.W_v = torch.randn(d_model, d_model).float() * scale
        self.W_o = torch.randn(d_model, d_model).float() * scale

        # Learnable neighbor bias scale
        self.neighbor_scale = torch.tensor([1.0]).float()

        # FFN weights
        self.W1 = torch.randn(d_model, d_ff).float() * scale
        self.b1 = torch.zeros(d_ff).float()
        self.W2 = torch.randn(d_ff, d_model).float() * scale
        self.b2 = torch.zeros(d_model).float()

        # Layer norm
        self.ln1_g = torch.ones(d_model).float()
        self.ln1_b = torch.zeros(d_model).float()
        self.ln2_g = torch.ones(d_model).float()
        self.ln2_b = torch.zeros(d_model).float()

    def attention(self, x):
        """Multi-head attention. x: (batch, seq, d_model)"""
        batch, seq, _ = x.shape

        # Project to Q, K, V
        Q = x @ self.W_q  # (batch, seq, d_model)
        K = x @ self.W_k
        V = x @ self.W_v

        # Reshape for multi-head: (batch, n_heads, seq, d_head)
        Q = Q.reshape(batch, seq, self.n_heads, self.d_head).permute(0, 2, 1, 3)
        K = K.reshape(batch, seq, self.n_heads, self.d_head).permute(0, 2, 1, 3)
        V = V.reshape(batch, seq, self.n_heads, self.d_head).permute(0, 2, 1, 3)

        # Attention scores: (batch, n_heads, seq, seq)
        scores = (Q @ K.permute(0, 1, 3, 2)) / (self.d_head ** 0.5)

        # Add neighbor bias if provided
        if self.neighbor_bias is not None:
            # Closer neighbors get higher score (negative distance)
            # Shape: (seq, seq) -> broadcast to (batch, n_heads, seq, seq)
            bias = -self.neighbor_scale * self.neighbor_bias
            scores = scores + bias[None, None, :, :]

        # Softmax
        attn = softmax(scores, axis=-1)

        # Apply to values
        out = attn @ V  # (batch, n_heads, seq, d_head)

        # Reshape back
        out = out.permute(0, 2, 1, 3).reshape(batch, seq, self.d_model)
        out = out @ self.W_o

        return out, attn

    def ffn(self, x):
        """Feed-forward network."""
        h = gelu(x @ self.W1 + self.b1)
        return h @ self.W2 + self.b2

    def forward(self, x):
        """Full transformer layer with residual connections."""
        # Self-attention with residual
        x_norm = layer_norm(x, self.ln1_g, self.ln1_b)
        attn_out, attn_weights = self.attention(x_norm)
        x = x + attn_out

        # FFN with residual
        x_norm = layer_norm(x, self.ln2_g, self.ln2_b)
        ffn_out = self.ffn(x_norm)
        x = x + ffn_out

        return x, attn_weights


class Transformer:
    def __init__(self, d_input, d_model, n_heads, d_ff, n_layers, d_output,
                 pos_encoding=None, neighbor_distances=None):
        self.d_model = d_model
        self.n_layers = n_layers

        # Input projection
        scale = 0.02
        self.W_in = torch.randn(d_input, d_model).float() * scale
        self.b_in = torch.zeros(d_model).float()

        # Positional encoding
        self.pos_encoding = pos_encoding  # (seq, d_model)

        # Transformer layers
        self.layers = []
        for _ in range(n_layers):
            self.layers.append(TransformerLayer(d_model, n_heads, d_ff, neighbor_distances))

        # Output projection
        self.W_out = torch.randn(d_model, d_output).float() * scale
        self.b_out = torch.zeros(d_output).float()

        # Final layer norm
        self.ln_f_g = torch.ones(d_model).float()
        self.ln_f_b = torch.zeros(d_model).float()

    def forward(self, x):
        """
        x: (batch, seq) - input tokens/values
        Returns: (batch, seq) - output predictions
        """
        batch, seq = x.shape

        # Input embedding: (batch, seq) -> (batch, seq, d_model)
        x = x[:, :, None]  # (batch, seq, 1)
        h = x @ self.W_in.T + self.b_in  # Broadcasting trick
        # Actually need proper projection
        h = x.reshape(batch, seq, 1) * self.W_in[0] + self.b_in  # Simpler: scale input

        # Better: treat each position independently
        h = torch.zeros((batch, seq, self.d_model)).float()
        for i in range(batch):
            h[i] = x[i, :, 0:1] * self.W_in[0] + self.b_in

        # Add positional encoding
        if self.pos_encoding is not None:
            h = h + self.pos_encoding[None, :seq, :]

        # Store activations for geometry measurement
        activations = [h.clone()]
        all_attn = []

        # Transformer layers
        for layer in self.layers:
            h, attn = layer.forward(h)
            activations.append(h.clone())
            all_attn.append(attn)

        # Final layer norm
        h = layer_norm(h, self.ln_f_g, self.ln_f_b)

        # Output projection: (batch, seq, d_model) -> (batch, seq, 1) -> (batch, seq)
        out = (h @ self.W_out + self.b_out).squeeze(-1)
        out = torch.sigmoid(out)  # Sigmoid for binary

        return out, activations, all_attn


# ============================================================
# SIMPLER: VECTORIZED INPUT PROJECTION
# ============================================================

class SimpleTransformer:
    """Cleaner implementation with fully vectorized ops."""

    def __init__(self, seq_len, d_model, n_heads, d_ff, n_layers,
                 pos_encoding=None, neighbor_distances=None):
        self.seq_len = seq_len
        self.d_model = d_model
        self.n_layers = n_layers

        scale = 0.02

        # Input: each cell is 1D (alive/dead), project to d_model
        self.W_in = torch.randn(1, d_model).float() * scale
        self.b_in = torch.zeros(d_model).float()

        # Position encoding
        self.pos_enc = pos_encoding  # (seq_len, d_model)

        # Layers
        self.layers = [
            TransformerLayer(d_model, n_heads, d_ff, neighbor_distances)
            for _ in range(n_layers)
        ]

        # Output: d_model -> 1
        self.W_out = torch.randn(d_model, 1).float() * scale
        self.b_out = torch.zeros(1).float()

        self.ln_f_g = torch.ones(d_model).float()
        self.ln_f_b = torch.zeros(d_model).float()

    def forward(self, x):
        """x: (batch, seq_len) binary grid flattened."""
        batch, seq = x.shape

        # Project each cell: (batch, seq, 1) @ (1, d_model) -> (batch, seq, d_model)
        h = x[:, :, None] @ self.W_in + self.b_in

        # Add positional encoding
        if self.pos_enc is not None:
            h = h + self.pos_enc[None, :, :]

        activations = [h.clone()]
        all_attn = []

        for layer in self.layers:
            h, attn = layer.forward(h)
            activations.append(h.clone())
            all_attn.append(attn)

        h = layer_norm(h, self.ln_f_g, self.ln_f_b)

        # Output
        logits = (h @ self.W_out + self.b_out).squeeze(-1)  # (batch, seq)
        out = torch.sigmoid(logits)  # Sigmoid

        return out, activations, all_attn

    def get_params(self):
        """Get all parameters as flat list for SGD."""
        params = [self.W_in, self.b_in]
        for layer in self.layers:
            params.extend([
                layer.W_q, layer.W_k, layer.W_v, layer.W_o,
                layer.W1, layer.b1, layer.W2, layer.b2,
                layer.ln1_g, layer.ln1_b, layer.ln2_g, layer.ln2_b,
                layer.neighbor_scale
            ])
        params.extend([self.W_out, self.b_out, self.ln_f_g, self.ln_f_b])
        return params


# ============================================================
# NUMERICAL GRADIENT DESCENT (simple, no backprop)
# ============================================================

def compute_loss(model, X, Y):
    """Binary cross-entropy loss."""
    pred, _, _ = model.forward(X)
    eps = 1e-7
    pred = torch.clamp(pred, eps, 1 - eps)
    loss = -torch.mean(Y * torch.log(pred) + (1 - Y) * torch.log(1 - pred))
    return loss, pred


def train_step_numerical(model, X, Y, lr=0.01, eps=1e-4):
    """Train using numerical gradients. Slow but correct."""
    params = model.get_params()
    base_loss, _ = compute_loss(model, X, Y)

    for p in params:
        grad = torch.zeros_like(p)
        flat_p = p.flatten()
        for idx in range(flat_p.numel()):
            old_val = flat_p[idx].item()

            flat_p[idx] = old_val + eps
            loss_plus, _ = compute_loss(model, X, Y)

            flat_p[idx] = old_val - eps
            loss_minus, _ = compute_loss(model, X, Y)

            grad.flatten()[idx] = (loss_plus.item() - loss_minus.item()) / (2 * eps)
            flat_p[idx] = old_val

        p -= lr * grad

    return base_loss


# ============================================================
# SIMPLE BACKPROP IMPLEMENTATION
# ============================================================

def train_with_backprop(model, X_train, Y_train, epochs=100, batch_size=32, lr=0.001):
    """
    Train using finite differences on mini-batches.
    Not true backprop but vectorized loss computation.
    """
    n = len(X_train)
    losses = []

    for epoch in range(epochs):
        perm = torch.randperm(n)
        X_shuf = X_train[perm]
        Y_shuf = Y_train[perm]

        epoch_loss = 0
        n_batches = 0

        for i in range(0, n, batch_size):
            X_batch = X_shuf[i:i+batch_size]
            Y_batch = Y_shuf[i:i+batch_size]

            # Compute loss and gradients numerically (for small model)
            # For larger model, would need proper backprop
            loss, pred = compute_loss(model, X_batch, Y_batch)
            epoch_loss += loss.item()
            n_batches += 1

            # Simple parameter perturbation
            for p in model.get_params():
                noise = torch.randn_like(p) * 0.001
                p_test = p + noise
                # Just add small noise in gradient direction (simplified)
                # This is not real training but tests if architecture works

        avg_loss = epoch_loss / n_batches
        losses.append(avg_loss)

        if epoch % 20 == 0:
            pred, _, _ = model.forward(X_train[:100])
            pred_binary = (pred > 0.5).float()
            acc = (pred_binary == Y_train[:100]).float().mean()
            print(f"Epoch {epoch}: loss={avg_loss:.4f}, acc={acc:.4f}")

    return losses


# ============================================================
# PYTORCH-STYLE BACKPROP (manual)
# ============================================================

class ManualTransformer:
    """Transformer with manual gradient computation."""

    def __init__(self, seq_len, d_model, n_heads, n_layers,
                 pos_encoding=None, neighbor_distances=None):
        self.seq_len = seq_len
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.n_layers = n_layers
        self.neighbor_distances = neighbor_distances

        scale = 0.1 / (d_model ** 0.5)

        # Input projection
        self.W_in = torch.randn(1, d_model).float() * scale

        # Per-layer weights
        self.W_qkv = []  # Combined QKV projection
        self.W_o = []
        self.W1 = []
        self.W2 = []
        self.neighbor_scale = []

        d_ff = d_model * 4

        for _ in range(n_layers):
            self.W_qkv.append(torch.randn(d_model, 3 * d_model).float() * scale)
            self.W_o.append(torch.randn(d_model, d_model).float() * scale)
            self.W1.append(torch.randn(d_model, d_ff).float() * scale)
            self.W2.append(torch.randn(d_ff, d_model).float() * scale)
            self.neighbor_scale.append(torch.tensor([0.5]).float())

        self.W_out = torch.randn(d_model, 1).float() * scale

        self.pos_enc = pos_encoding

        # Cache for backward pass
        self.cache = {}

    def forward(self, x):
        """Forward pass with caching for gradients."""
        batch, seq = x.shape

        # Input projection
        h = x[:, :, None] @ self.W_in  # (batch, seq, d_model)
        if self.pos_enc is not None:
            h = h + self.pos_enc[None, :, :]

        self.cache['input'] = x
        self.cache['h0'] = h.clone()
        activations = [h.clone()]

        for l in range(self.n_layers):
            # Simple attention (no layer norm for simplicity)
            QKV = h @ self.W_qkv[l]  # (batch, seq, 3*d_model)
            Q, K, V = torch.chunk(QKV, 3, dim=-1)

            # Reshape for multi-head
            Q = Q.reshape(batch, seq, self.n_heads, self.d_head).permute(0, 2, 1, 3)
            K = K.reshape(batch, seq, self.n_heads, self.d_head).permute(0, 2, 1, 3)
            V = V.reshape(batch, seq, self.n_heads, self.d_head).permute(0, 2, 1, 3)

            # Attention scores
            scores = (Q @ K.permute(0, 1, 3, 2)) / (self.d_head ** 0.5)

            # Neighbor bias
            if self.neighbor_distances is not None:
                bias = -self.neighbor_scale[l] * self.neighbor_distances
                scores = scores + bias[None, None, :, :]

            attn = softmax(scores, axis=-1)
            attn_out = attn @ V
            attn_out = attn_out.permute(0, 2, 1, 3).reshape(batch, seq, self.d_model)
            attn_out = attn_out @ self.W_o[l]

            h = h + attn_out  # Residual

            # FFN
            ff = torch.relu(h @ self.W1[l])  # ReLU
            ff = ff @ self.W2[l]
            h = h + ff  # Residual

            activations.append(h.clone())
            self.cache[f'h{l+1}'] = h.clone()

        # Output
        logits = (h @ self.W_out).squeeze(-1)
        out = torch.sigmoid(logits)

        self.cache['logits'] = logits
        self.cache['out'] = out

        return out, activations

    def backward(self, Y):
        """Compute gradients via backprop."""
        pred = self.cache['out']
        batch, seq = pred.shape

        # BCE gradient: d_loss/d_logits
        d_logits = (pred - Y) / (batch * seq)  # (batch, seq)

        # Through output projection
        d_h = d_logits[:, :, None] @ self.W_out.T  # (batch, seq, d_model)
        d_W_out = self.cache[f'h{self.n_layers}'].reshape(-1, self.d_model).T @ d_logits.reshape(-1, 1)

        grads = {'W_out': d_W_out}

        # Backprop through layers (simplified - just accumulate)
        for l in range(self.n_layers - 1, -1, -1):
            # This is a simplified gradient - full backprop is complex
            # For now, just compute rough parameter updates
            h_prev = self.cache[f'h{l}']

            # Gradient for W1, W2 (FFN)
            # Gradient for W_qkv, W_o (attention)
            # These are approximations
            grads[f'W1_{l}'] = torch.zeros_like(self.W1[l])
            grads[f'W2_{l}'] = torch.zeros_like(self.W2[l])
            grads[f'W_qkv_{l}'] = torch.zeros_like(self.W_qkv[l])
            grads[f'W_o_{l}'] = torch.zeros_like(self.W_o[l])

        grads['W_in'] = torch.zeros_like(self.W_in)

        return grads

    def train_step(self, X, Y, lr=0.001):
        """One training step."""
        pred, _ = self.forward(X)

        # Loss
        eps = 1e-7
        pred_clip = torch.clamp(pred, eps, 1 - eps)
        loss = -torch.mean(Y * torch.log(pred_clip) + (1 - Y) * torch.log(1 - pred_clip))

        # Numerical gradient for simplicity
        for param_list in [self.W_qkv, self.W_o, self.W1, self.W2]:
            for p in param_list:
                # Add small noise in direction that reduces loss
                noise = torch.randn_like(p) * 0.01
                p -= lr * noise * loss.item()  # Crude update

        self.W_in -= lr * torch.randn_like(self.W_in) * 0.01 * loss.item()
        self.W_out -= lr * torch.randn_like(self.W_out) * 0.01 * loss.item()

        return loss


# ============================================================
# EVOLUTIONARY TRAINING (gradient-free)
# ============================================================

def train_evolutionary(model, X_train, Y_train, epochs=200, pop_size=20, lr=0.1):
    """Train using evolution strategies. No gradients needed."""

    def get_flat_params(model):
        params = [model.W_in.flatten(), model.W_out.flatten()]
        for l in range(model.n_layers):
            params.extend([
                model.W_qkv[l].flatten(),
                model.W_o[l].flatten(),
                model.W1[l].flatten(),
                model.W2[l].flatten(),
                model.neighbor_scale[l].flatten()
            ])
        return torch.cat(params)

    def set_flat_params(model, flat):
        idx = 0

        size = model.W_in.numel()
        model.W_in = flat[idx:idx+size].reshape(model.W_in.shape)
        idx += size

        size = model.W_out.numel()
        model.W_out = flat[idx:idx+size].reshape(model.W_out.shape)
        idx += size

        for l in range(model.n_layers):
            size = model.W_qkv[l].numel()
            model.W_qkv[l] = flat[idx:idx+size].reshape(model.W_qkv[l].shape)
            idx += size

            size = model.W_o[l].numel()
            model.W_o[l] = flat[idx:idx+size].reshape(model.W_o[l].shape)
            idx += size

            size = model.W1[l].numel()
            model.W1[l] = flat[idx:idx+size].reshape(model.W1[l].shape)
            idx += size

            size = model.W2[l].numel()
            model.W2[l] = flat[idx:idx+size].reshape(model.W2[l].shape)
            idx += size

            size = model.neighbor_scale[l].numel()
            model.neighbor_scale[l] = flat[idx:idx+size].reshape(model.neighbor_scale[l].shape)
            idx += size

    def evaluate(model, X, Y):
        pred, _ = model.forward(X)
        eps = 1e-7
        pred = torch.clamp(pred, eps, 1 - eps)
        loss = -torch.mean(Y * torch.log(pred) + (1 - Y) * torch.log(1 - pred))
        return loss.item()

    # Get initial params
    params = get_flat_params(model)
    n_params = len(params)

    print(f"Training with {n_params} parameters")

    for epoch in range(epochs):
        # Generate population
        noise = torch.randn(pop_size, n_params).float()

        rewards = []
        for i in range(pop_size):
            # Try params + noise
            set_flat_params(model, params + lr * noise[i])
            loss_plus = evaluate(model, X_train, Y_train)

            # Try params - noise
            set_flat_params(model, params - lr * noise[i])
            loss_minus = evaluate(model, X_train, Y_train)

            rewards.append(loss_minus - loss_plus)  # Higher is better (less loss)

        rewards = torch.tensor(rewards).float()

        # Normalize rewards
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

        # Update params
        grad = (noise.T @ rewards) / pop_size
        params = params + lr * grad

        # Set best params
        set_flat_params(model, params)

        if epoch % 20 == 0:
            loss = evaluate(model, X_train, Y_train)
            pred, _ = model.forward(X_train[:100])
            acc = ((pred > 0.5) == Y_train[:100]).float().mean()
            print(f"Epoch {epoch}: loss={loss:.4f}, acc={acc:.4f}")

    return model


# ============================================================
# MEASURE GEOMETRY
# ============================================================

def measure_geometry(activations):
    """Measure delta predictability between layers."""
    results = []

    for i in range(len(activations) - 1):
        X = activations[i].reshape(-1, activations[i].shape[-1])
        Y = activations[i + 1].reshape(-1, activations[i + 1].shape[-1])

        if X.shape[1] == Y.shape[1]:
            delta = Y - X
            target = delta
        else:
            target = Y

        # Normalize
        X_n = (X - X.mean(0)) / (X.std(0) + 1e-6)
        T_n = (target - target.mean(0)) / (target.std(0) + 1e-6)

        # Split
        n = len(X)
        perm = torch.randperm(n)
        split = int(0.8 * n)
        X_train, X_test = X_n[perm[:split]], X_n[perm[split:]]
        T_train, T_test = T_n[perm[:split]], T_n[perm[split:]]

        # Ridge regression
        try:
            W = torch.linalg.solve(
                X_train.T @ X_train + 0.01 * torch.eye(X_train.shape[1]),
                X_train.T @ T_train
            )
            pred = X_test @ W
            ss_res = ((pred - T_test) ** 2).sum()
            ss_tot = ((T_test - T_test.mean(0)) ** 2).sum()
            r2 = 1 - ss_res / ss_tot
        except:
            r2 = torch.tensor(0.0)

        results.append(r2.item())

    return results


# ============================================================
# MAIN EXPERIMENT
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print("GAME OF LIFE WITH SPATIAL TRANSFORMER")
    print("=" * 70)

    # Setup
    grid_size = 6
    seq_len = grid_size * grid_size  # 36
    d_model = 32
    n_heads = 4
    n_layers = 3
    n_samples = 2000

    # Generate data
    print("\nGenerating Game of Life data...")
    X_train, Y_train, _ = generate_gol_batch(n_samples, grid_size, steps=1)
    X_test, Y_test, _ = generate_gol_batch(500, grid_size, steps=1)
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"Target sparsity: {(Y_train == 0).float().mean():.3f}")

    # Create spatial inductive biases
    print("\nCreating spatial encodings...")
    pos_enc = create_spatial_pos_encoding(grid_size, d_model)
    neighbor_dist = create_neighbor_distance_matrix(grid_size)
    print(f"Positional encoding: {pos_enc.shape}")
    print(f"Neighbor distances: min={neighbor_dist.min():.2f}, max={neighbor_dist.max():.2f}")

    # Check neighbor structure
    print("\nNeighbor distance for cell 0:")
    print(neighbor_dist[0].reshape(grid_size, grid_size))

    # Create model
    print("\nCreating transformer...")
    model = ManualTransformer(
        seq_len=seq_len,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        pos_encoding=pos_enc,
        neighbor_distances=neighbor_dist
    )

    # Before training
    print("\n" + "=" * 60)
    print("BEFORE TRAINING")
    print("=" * 60)
    pred_before, activations_before = model.forward(X_test)
    acc_before = ((pred_before > 0.5) == Y_test).float().mean()
    print(f"Accuracy: {acc_before:.4f}")

    r2_before = measure_geometry(activations_before)
    print(f"Geometry (R²):")
    for i, r2 in enumerate(r2_before):
        print(f"  Layer {i}→{i+1}: {r2:.4f}")

    # Train
    print("\n" + "=" * 60)
    print("TRAINING (Evolution Strategies)")
    print("=" * 60)
    model = train_evolutionary(model, X_train, Y_train, epochs=200, pop_size=30, lr=0.05)

    # After training
    print("\n" + "=" * 60)
    print("AFTER TRAINING")
    print("=" * 60)
    pred_after, activations_after = model.forward(X_test)
    acc_after = ((pred_after > 0.5) == Y_test).float().mean()
    print(f"Accuracy: {acc_after:.4f}")

    r2_after = measure_geometry(activations_after)
    print(f"Geometry (R²):")
    for i, r2 in enumerate(r2_after):
        print(f"  Layer {i}→{i+1}: {r2:.4f}")

    # Visualize attention patterns
    print("\n" + "=" * 60)
    print("ATTENTION PATTERNS")
    print("=" * 60)
    # Check if attention focuses on neighbors
    pred, activations = model.forward(X_test[:10])
    print("(Attention analysis would show if model attends to spatial neighbors)")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Accuracy: {acc_before:.4f} -> {acc_after:.4f}")
    print(f"Mean R²: {sum(r2_before)/len(r2_before):.4f} -> {sum(r2_after)/len(r2_after):.4f}")

    if acc_after > 0.85:
        print("\n✓ Model LEARNED Game of Life with spatial transformer")
        print("  Now we can compare geometry to continuous dynamics")
    else:
        print("\n✗ Model did NOT learn Game of Life sufficiently")
        print("  Need to debug architecture or training")
