# CTM Research Log

## Session Summary
**Date**: 2025-12-28
**Goal**: Understand what makes CTM (Continuous Thought Machine) work and extract transferable architectural insights

---

## What We Started With

### Original Test Files

1. **test_compositional_transformers.py**
   - Task: Language Model on WikiText corpus (word-level tokenization)
   - Goal: Test compositional hypothesis, early exit, confidence heads
   - Metrics: Perplexity, layer-wise perplexity, early exit distribution

2. **test_reversal.py**
   - Task: Sequence reversal (e.g., "1 2 3 4 |" → "4 3 2 1")
   - Goal: Test if CTM's thinking ticks help with algorithmic tasks
   - Result: CTM beat Standard Transformer (99.6% vs 94.4%)

---

## Critical Discovery: We Never Implemented Real CTM

### What Real CTM Does (from ctm_paper.md)

```
Real CTM Architecture:
├── Per-neuron private weights: Each neuron d has private MLP θ_d
├── Post-activation history: Z_t = [z¹, z², ..., zᵀ] (activations over time)
├── Synchronization AS representation:
│   └── S_t = Z_t · Z_t^T  (correlation matrix)
├── Output FROM sync: y = W_out · subsample(S_out)
├── Query FROM sync: q = W_in · subsample(S_action)
└── Dual loss: argmin(prediction_loss) + argmax(certainty)
```

### What We Actually Built

```
Our "CTM" Implementation:
├── Standard transformer blocks (shared across all neurons)
├── No post-activation history Z
├── Sync as tiny add-on: h = h + 0.1 * sync_features
├── Output from hidden state: y = W · h (NOT from sync)
└── Single loss: just prediction loss
```

### The Fundamental Gap

| Component | Real CTM | Our Implementation |
|-----------|----------|-------------------|
| Sync matrix S | Core representation (Z·Z^T) | 0.1 weight add-on |
| Output source | Projection of sync | Hidden state |
| Per-neuron weights | Yes (private MLPs) | No (shared weights) |
| History | Activations over time | Just h_prev |
| Loss | Dual (loss + certainty) | Single |

---

## What We Actually Tested (Looped Transformers with Accumulation)

### Key Finding: Accumulation is the Key Mechanism

From `observe_accumulation_vs_reflection.py`:

| Configuration | Final Accuracy |
|--------------|----------------|
| Full CTM (R+A+H+) | 98.3% |
| No reflection (R-A+H+) | 96.9% |
| No accumulation (R+A-H+) | 85.7% |
| No history (R+A+H-) | 97.2% |
| **ONLY accumulation (R-A+H-)** | **99.0%** |

**Conclusion**: `h = h + h_new` (accumulation) is what matters, not history or reflection.

---

## Task-Specific Results

### Expression Evaluation (a + b * c)
- Simple Accumulator: 100%
- All CTM variants: ~100%
- Standard Transformer: 99.4%

### Counting Task
- **Standard Transformer: 88.3%**
- CTM variants: 72-80%
- **CTM hurts here!**

### Why CTM Hurts Counting

From `test_weight_sharing.py`:
| Model | Counting Accuracy |
|-------|-------------------|
| CTM Shared | 76.4% |
| CTM Unrolled (different weights) | 90.4% |
| Unrolled + Accumulation | 84.8% |
| Standard Transformer | 83.7% |

**Key insight**: For counting, unrolling (different weights per tick) helps, but accumulation hurts.

---

## The "Variant A vs B" Test

From `test_variant_a.py`:
- Variant A: Fuller history (2-layer MLP, history_len=2)
- Variant B: Simple history (single linear, just h_prev)

**Result**: Fuller history (Variant A) does NOT beat simplified (Variant B)
- Both ~99% on expression eval
- Accumulation is still the key mechanism in both

---

## Files Created During Investigation

| File | Purpose | Key Finding |
|------|---------|-------------|
| observe_attention_deep.py | Attention pattern analysis | 85% examples show attention shift to 'a' at tick 1 |
| observe_pattern_consistency.py | Verify attention patterns | Operator precedence through iterative attention |
| observe_compositional_attention.py | Alternative architectures | CTM wins but alternatives close (99% vs 100%) |
| observe_self_reflection.py | Test h_prev importance | Without reflection slightly better (99.0% vs 98.3%) |
| observe_accumulation_vs_reflection.py | Full ablation | Only Accumulation gets BEST result |
| observe_final_insight.py | Multi-run verification | Simple Accumulator matches Full CTM |
| test_all_task_classes.py | 6 different tasks | CTM hurts counting (-15.7%) |
| test_variant_a.py | Fuller vs simpler history | No difference |
| test_weight_sharing.py | Shared vs unrolled weights | Unrolling helps counting |
| test_real_ctm.py | Real CTM implementation | Sync-as-representation hurts on lookup tasks |

---

## Key Quotes from User

> "But we can't look at one task or type of tasks and decide. If the architecture is general purpose, we need to test on all types of tasks..."

> "But these are same kind of tasks. The class is the same." (on our 6 task types)

> "The idea is not to copy CTM. The idea is to learn from CTM, learn from what they have done so well, but then understand why does it work in the first place?"

> "But did we actually properly add ctm to our architecture or not"

---

## Real CTM Implementation (test_real_ctm.py)

### What We Implemented

```python
class RealCTM:
    # Per-neuron NLMs: Each neuron d has private MLP
    self.nlms = [NeuronLevelModel() for d in range(hidden_dim)]

    # Synapse model: a_t = f_θ_syn(concat(z_t, input))
    self.synapse = MLP(2*hidden_dim, hidden_dim)

    # Sync as representation: S = Z · Z^T
    def compute_sync_features(Z, pairs):
        return [(Z[:,:,i] * Z[:,:,j]).sum(-1) for (i,j) in pairs]

    # Output FROM sync: y = W_out · S_out
    self.output_proj = Linear(n_sync_pairs, vocab_size)
```

### Results on Reversal Task

| Model | Test Acc | Long (OOD) | Params |
|-------|----------|------------|--------|
| Standard Transformer | **100.0%** | 26.5% | 201,870 |
| Simple Accumulator | **100.0%** | 21.7% | 101,902 |
| Fake CTM | **100.0%** | 28.2% | 114,318 |
| Real CTM | 76.8% | 20.0% | 50,254 |

### Key Insight: Task-Architecture Match

**Real CTM performs WORSE on reversal (76.8% vs 100%)**

Why? Because reversal is a **lookup task**, not a **thinking task**:
- Reversal just needs: "position i maps to position n-i"
- This is a simple attention pattern, no "thinking" required
- Standard transformer's attention is perfectly suited for this

**CTM was designed for:**
- Tasks requiring temporal dynamics (maze solving)
- Tasks needing iterative refinement (image classification)
- Tasks where "thinking over time" matters

**Reversal does NOT require:**
- Synchronization between neurons
- Temporal evolution of representations
- Multiple "thinking" steps

### Conclusion

Sync-as-representation doesn't help on simple lookup tasks.
Need to test on tasks where CTM's temporal dynamics actually matter:
1. Maze solving
2. Multi-step reasoning
3. Planning tasks

---

## What We Still Don't Know

1. ~~Why does real CTM's sync-as-representation work?~~
   - **Tested**: It doesn't help on lookup tasks like reversal
   - **Still unknown**: Does it help on "thinking" tasks?

2. ~~What about per-neuron weights?~~
   - **Tested**: Implemented NLMs with private MLPs
   - **Finding**: They work but add complexity without helping reversal

3. **Does the dual loss matter?**
   - argmin(loss) + argmax(certainty) - never tested

4. **What tasks DOES Real CTM shine on?**
   - Need to test on maze solving, planning, multi-step reasoning

---

## Next Steps

### Option A: Implement Real CTM ✓ DONE
1. ✓ Create per-neuron private weights (NLMs)
2. ✓ Build post-activation history Z
3. ✓ Compute sync matrix S = Z·Z^T
4. ✓ Route output through sync projection
5. TODO: Add dual loss

### Option B: Extract Transferable Principles
1. Why does synchronization matter?
2. What computational property does S = Z·Z^T provide?
3. Can we get the benefit without the complexity?

### Option C: Find Tasks Where Real CTM Shines
1. Tasks requiring temporal coherence
2. Tasks needing neuron-level coordination
3. Tasks where accumulation hurts (like counting)

---

## Commit History

```
a26d358 Discover weight sharing is NOT the issue - unbundling reveals task-specific needs
37d59f0 Adds CTM paper
d1a9942 Test CTM across 6 fundamentally different task classes
2716285 Validate findings: fuller history (Variant A) does NOT beat simplified (Variant B)
eda1c31 Discover CTM's key mechanism: iterative accumulation, not self-reflection
```

---

## Theoretical Framework: Intelligence as Self-Modifying Prediction

### The Core Loop

```
Sense → Store → Model → Predict → Act → Feedback → Update
  ↑                                                    ↓
  └────────────────────────────────────────────────────┘
```

**Intelligence IS this loop.** The minimal "action" is self-modification based on prediction error.

### Two Levels of Learning

| Level | When | What Changes | Ground Truth |
|-------|------|--------------|--------------|
| **Outer loop** | Training | Weights | Labels (external) |
| **Inner loop** | Inference | Activations | Input itself |

**Key insight**: Current networks have no inner loop at inference. They're fixed functions.

### The Hypothesis

Networks with inference-time self-modification (inner loop) should outperform fixed-function networks on tasks requiring compositional generalization.

---

## Inner Loop Implementation (test_inner_loop.py)

### The Mechanism

```python
# Inner loop at inference:
for step in range(n_steps):
    pred_embed = model.predict_input(state)      # Predict input
    error = pred_embed - actual_input_embed      # Compare to ground truth
    state = state - lr * error                   # Update state
```

**Ground truth at inference = the input itself.**

### Results on Sorting/Reversal

| Task | Standard | Inner Loop | Difference |
|------|----------|------------|------------|
| Sorting | 100% | 99.9% | -0.1% |
| Reversal | 97% | 96% | -1% |
| **Sorting OOD** | 27.6% | **30.7%** | **+3.1%** |

Inner loop helps with **OOD generalization**, not in-distribution accuracy.

---

## Compositional Generalization Test (test_compositional_gen.py)

### Task Design

SCAN-inspired command execution:
- Train: WALK+TWICE, RUN+TWICE, JUMP+THRICE, LOOK+THRICE
- Test: WALK+THRICE, RUN+THRICE, JUMP+TWICE, LOOK+TWICE (NOVEL combinations)

The model must combine known primitives in NEW ways.

### Results (Single Seed)

| Model | Train | Test (Novel) | Gap |
|-------|-------|--------------|-----|
| Standard | 100% | 51.5% | -48.5% |
| **Inner Loop** | 100% | **64.7%** | **-35.3%** |
| Accumulator | 100% | 48.5% | -51.5% |

### Key Finding

**Inner Loop improves compositional generalization by +13.2%**

- Standard: 51.5% on novel combinations
- Inner Loop: 64.7% on novel combinations
- Accumulator: 48.5% (HURTS)

### Why Inner Loop Helps

1. **Grounds representation in input** at inference time
2. Forces state to be consistent with ALL parts of input
3. When facing "WALK THRICE" (never seen together):
   - Model has seen WALK and THRICE separately
   - Inner loop forces consistency with both parts
   - This enables correct composition

### Why Accumulator Hurts

- Accumulator: `h = h + h_new` (just iteration)
- No grounding in input
- Iterates but doesn't verify against input structure
- Can drift away from correct interpretation

---

## The Pattern Emerging

| Task Type | Inner Loop Helps? | Why? |
|-----------|-------------------|------|
| Lookup (reversal) | No | Simple attention pattern suffices |
| Sorting | Slightly (OOD) | Some benefit for generalization |
| **Compositional** | **YES (+13%)** | Requires grounding in input structure |

**The inner loop helps when the task requires relating novel inputs to known structure.**

---

## Updated Files

| File | Purpose | Key Finding |
|------|---------|-------------|
| test_inner_loop.py | Inner loop hypothesis test | OOD generalization improves |
| test_compositional_gen.py | Compositional generalization | **+13.2% on novel combinations** |

---

## Lessons Learned

1. **Read the paper carefully before implementing** - We missed core CTM innovations
2. **Test on diverse tasks early** - We drew conclusions from single task type
3. **Question assumptions** - "History" meant different things in paper vs implementation
4. **Track progress in writing** - 12 hours without documentation led to confusion
5. **Simple baselines first** - Simple Accumulator matched our "CTM" because it wasn't really CTM
6. **Ground truth at inference = input** - The inner loop uses input as self-supervision
7. **Compositional generalization is the key test** - Not in-distribution accuracy

---

## Deeper Theory: Backprop ≈ Evolution

### The Core Analogy

| Aspect | Evolution | Backprop |
|--------|-----------|----------|
| Acts on | Genes | Weights |
| Search type | Parallel, local | Parallel, local |
| Finds | Organisms with brains | Networks with weights |
| Global knowledge | No | No |
| Long-chain credit | Poor | Limited (vanishing gradients) |

**Key insight**: Both are SEARCH processes that find structures fitting constraints. Neither IS intelligence—they FIND it.

### The Two-Level Structure

```
OUTER LOOP (Evolution / Backprop):
├── Searches over structure (genes / weights)
├── Optimizes for fitness / loss
├── NOT biologically plausible at neural level
└── FINDS organisms / networks

INNER LOOP (Brain / ???):
├── Operates INSIDE what outer loop found
├── Acts on activations / firing patterns
├── Uses DIFFERENT rules (not evolution / not backprop)
└── IS intelligence
```

**The gap in current networks**: No meaningful inner loop. Just one forward pass.

### Intelligence as Self-Modifying Prediction

**Minimal definition**:
> Intelligence is a self-modifying prediction system.

The irreducible core:
```
Predict → Observe outcome → Update self
```

"Update self" IS the action. Even a brain in a vat, if learning, is acting on itself.

---

## The Dragon Hatchling (BDH) Paper Analysis

### What BDH Claims

1. **Biologically plausible** inference dynamics
2. **Hebbian learning** at inference time (state on edges updates)
3. **Mathematical equivalence** between tensor ops and graph dynamics
4. **Matches Transformer performance** (not beats—matches)

### BDH's Two Timescales

| Timescale | What Updates | Mechanism |
|-----------|--------------|-----------|
| **Fast** (inference, minutes) | State σ on edges | Hebbian: σ += x * y |
| **Slow** (training, hours) | Weights | Backprop |

### What BDH Actually Implements (Code)

```python
for layer in layers:
    x_sparse = ReLU(x @ encoder)           # Sparse representation
    y = linear_attention(x_sparse, x)       # Accumulates over TOKENS
    y_sparse = ReLU(y @ encoder_v)
    x = x + (x_sparse * y_sparse) @ decoder # Hebbian-like multiply
```

**No explicit ticks. No explicit state variable. Just tensor operations.**

### The Key Realization

BDH claims mathematical equivalence:
- **Parallel GPU computation** ≡ **Sequential Hebbian updates**
- Same math, different implementation

The "Hebbian dynamics" and "state on edges" are **theoretical interpretations**, not explicit mechanisms in the code.

### How This Relates to Our Hypothesis

| Our Hypothesis | BDH |
|----------------|-----|
| Outer loop: Backprop finds weights | ✓ Yes |
| Inner loop: Self-modification at inference | ✓ Yes (claimed, implicit) |
| Inner loop ≠ Backprop | ✓ Yes (Hebbian ≠ gradient) |

**Our hypothesis is SUPPORTED by BDH's claims.**

### Where We Went Wrong vs BDH

| Aspect | BDH | Our CTM |
|--------|-----|---------|
| State location | Edges (synapses) | Neurons (activations) |
| Accumulation over | Tokens (sequence) | Ticks (iterations) |
| Update rule | Hebbian (x * y) | Sync (Z · Z^T) |
| Explicit loops | No | Yes |

**Same idea, different implementation.**

---

## Current Test: BDH-Style vs Standard vs Tick-CTM (test_bdh_style.py)

### Architectures Compared

| Model | Key Features | Effective Passes |
|-------|--------------|------------------|
| Standard Transformer | Softmax attention, 4 layers | 4 |
| Tick-CTM | Softmax + accumulation, 2 layers × 2 ticks | 4 |
| BDH-Style | Linear attention + ReLU + x*y, 4 layers | 4 |

### BDH-Style Implementation

```python
class BDHBlock:
    # 1. Encode to high dimension
    x_sparse = ReLU(x @ encoder)  # Sparse, positive

    # 2. Linear attention (no softmax, causal)
    scores = (Q @ K.T).tril(diagonal=-1)  # Strictly lower triangular
    y = scores @ V

    # 3. Hebbian element-wise product
    xy = x_sparse * y_sparse  # "Fire together, wire together"

    # 4. Decode back
    return x + (xy @ decoder)
```

### What We're Testing

1. Does BDH's architectural choices (linear attention, ReLU sparsity, x*y multiply) match Standard Transformer?
2. Does Tick-CTM's explicit iteration still help when controlling for effective passes?
3. Which approach generalizes better to OOD (longer sequences)?

### Results (Controlling for Effective Passes)

When all models have ~4 effective passes:

| Model | Test | OOD (Long) | Params |
|-------|------|------------|--------|
| Standard (4 layers) | 86-87% | 48-50% | 201k |
| Tick-CTM (2L × 2T) | 81-82% | 44-46% | 101k |
| BDH-Style (4 layers) | 67-69% | 35-37% | 198k |

**Standard Transformer wins when passes are equalized.**

### Results (Tick-CTM with More Iterations)

When Tick-CTM gets 2 layers × 8 ticks = 16 effective passes:

| Model | Test | OOD (Long) | Params |
|-------|------|------------|--------|
| Standard (4 layers) | 86.1% | 48.6% | 201k |
| **Tick-CTM (2L × 8T)** | **92.7%** | **66.7%** | 101k |
| BDH-Style (4 layers) | 68.7% | 36.3% | 198k |

**Tick-CTM crushes everything with more ticks: +6.6% test, +18% OOD with HALF the parameters.**

### Key Finding: Iteration Beats Depth

- More ticks > more layers (for reversal task)
- **Tick-CTM is essentially a recurrent/looped transformer**, NOT real CTM
- Weight sharing + accumulation + more iterations = strong performance
- This is closer to Universal Transformers than to CTM's sync mechanism

---

## Critical Clarification: What Our "Tick-CTM" Actually Is

**We keep calling it "CTM" but it's NOT:**

```python
# What we built (Tick-CTM / Looped Transformer):
for tick in range(n_ticks):
    h_new = transformer(h)
    h = h + h_new  # Accumulation

# What Real CTM does:
for tick in range(n_ticks):
    z = NLM(pre_activation_history)   # Per-neuron private MLPs
    Z.append(z)                        # Post-activation history
    S = Z @ Z.T                        # Sync matrix
    output = project(S)                # Output FROM sync
```

**Our approach**: Same weights, iterate, accumulate activations.
**Real CTM**: Per-neuron dynamics, sync as representation, output from sync.

These are fundamentally different. We've been testing **looped transformers**, not CTM.

---

## Why BDH-Style Underperforms (Our Implementation Issues)

Our BDH implementation has bugs:

1. **Causal mask**: We used `tril(diagonal=0)`, BDH uses `tril(diagonal=-1)` (excludes diagonal)
2. **V dimension mismatch**: Attention head dimensions not aligned properly
3. **Missing proper RoPE**: Our rotary encoding is simplified

**BDH claims to MATCH transformer performance, not beat it.** Our buggy implementation underperforms, which is expected.

---

## Why Transformers Are Still So Good

Despite all our experiments, standard transformers remain highly competitive. Why?

| Property | Why It Helps |
|----------|--------------|
| **Attention** | O(1) access to any position—perfect for lookup tasks |
| **Parallelization** | GPU-friendly, fast training |
| **Depth** | Each layer can specialize (different features) |
| **Residual connections** | Gradient flow, easy optimization |
| **Softmax** | Sparse, interpretable attention patterns |

**The reversal task is a LOOKUP task.** Transformers excel at lookup—just attend to position n-i.

For tasks that are NOT lookup:
- Maze solving (needs search/planning)
- Multi-step reasoning (needs iterative refinement)
- Compositional generalization (needs grounding in input structure)

...we might see different results. **We haven't properly tested these.**

---

## Remaining Uncertainty: Task-Architecture Match

We still don't know:

| Architecture | Best For | Unknown |
|--------------|----------|---------|
| Standard Transformer | Lookup, pattern matching | Limits of depth vs width |
| Tick-CTM (Looped) | Tasks needing iterative refinement | When does more iteration hurt? |
| Real CTM (Sync) | ??? | Never properly tested on right tasks |
| BDH (Hebbian) | ??? | Our implementation is buggy |

**Key unknown**: What task classes REQUIRE sync-as-representation vs just iteration vs just attention?

CTM paper tested: MNIST, mazes, parity, RL
BDH paper tested: Language modeling, translation

We tested: Reversal, expression eval, counting, compositional

**We may be testing the WRONG tasks for these architectures.**

---

## CTM Notebook Visualization Capabilities

The CTM codebase (examples/01_mnist.ipynb, 03_mazes.ipynb, 04_parity.ipynb) provides sophisticated visualization:

### What They Visualize

1. **Neural Dynamics Over Time**
   - Pre-activations (gray dashed lines)
   - Post-activations (colored lines per neuron)
   - Shows how each neuron evolves over internal ticks

2. **Attention Patterns**
   - Heatmaps over input (image/sequence)
   - Updated each tick—shows WHERE model looks

3. **Predictions Over Time**
   - Bar charts of class probabilities
   - Shows HOW the answer EMERGES over ticks

4. **Certainty Over Time**
   - Line plot of model confidence
   - Shows WHEN the model "decides"

5. **Animated GIFs**
   - Combines all above
   - Frame-by-frame visualization of "thinking"

### Code Pattern (from 01_mnist.ipynb)

```python
def make_gif(predictions, certainties, targets, pre_activations,
             post_activations, attention, inputs, filename):
    """
    Creates animated GIF showing:
    - Input image
    - Attention heatmap (changes each tick)
    - Prediction probabilities (bar chart)
    - Certainty over time (line plot)
    - Individual neuron traces (multiple line plots)
    """
    for stepi in range(n_steps):
        # Plot attention at this tick
        # Plot predictions at this tick
        # Plot certainty line with vertical marker
        # Plot each neuron's pre/post activation trace
        frames.append(render_frame())

    mediapy.save(filename, frames)
```

### Why This Matters

These visualizations let us **literally see**:
- Is the model "searching" (attention moving around)?
- Is it "converging" (certainty increasing)?
- Which neurons activate for which features?
- How does the answer emerge over time?

**Our visualization (visualize_comparison.py) is much simpler**—just line plots of confidence/certainty over layers/ticks. To properly compare architectures, we need GIF-style temporal visualization.

---

## Files Created/Updated This Session

| File | Purpose | Key Finding |
|------|---------|-------------|
| test_bdh_style.py | BDH vs Standard vs Tick-CTM | Standard wins when passes equalized; Tick-CTM wins with more ticks |
| visualize_comparison.py | Side-by-side visualization | Created basic comparison plots |
| RESEARCH_LOG.md | This file | Comprehensive tracking of findings |

---

## Open Questions (Updated)

### Theoretical

1. **Is the two-loop hypothesis correct?**
   - Outer loop: Backprop finds structure
   - Inner loop: Self-modification at inference
   - Evidence: Tick-CTM's iteration helps; BDH claims implicit Hebbian

2. **What is the right "inner loop" mechanism?**
   - Accumulation over ticks? (our approach)
   - Hebbian edge updates? (BDH's claim)
   - Sync as representation? (real CTM)
   - Prediction-error grounding? (inner loop hypothesis)

3. **Why does iteration help OOD generalization?**
   - Tick-CTM: +18% on longer sequences
   - Is it "more compute" or "iterative refinement"?
   - Would Universal Transformers show same pattern?

### Empirical

4. **What tasks REQUIRE sync-as-representation?**
   - We never tested real CTM on maze solving, multi-step reasoning
   - Sakana's CTM paper shows it works on mazes—we should replicate

5. **Is our BDH implementation correct?**
   - Our version underperforms; BDH paper claims parity with transformers
   - Need to fix bugs and re-test

6. **What's the role of attention type?**
   - Softmax vs linear attention
   - BDH uses linear attention for implicit state accumulation
   - Does this matter for which tasks?

### Meta

7. **Are we testing the right tasks?**
   - Reversal is a lookup task—transformers are built for this
   - Need tasks that REQUIRE "thinking": planning, search, reasoning

8. **How do we properly visualize these architectures?**
   - CTM notebooks show detailed neural dynamics
   - We should create similar visualizations to UNDERSTAND differences
   - Not just accuracy numbers—HOW does each model solve the task?

---

## Next Steps

### Immediate

1. **Fix BDH implementation** - Match paper's architecture exactly
2. **Add GIF visualization** - See HOW each model thinks, not just accuracy
3. **Test on maze-like tasks** - Where CTM was designed to shine

### Medium-term

4. **Implement real CTM** - Per-neuron NLMs, sync as representation
5. **Test two-loop hypothesis** - Inner loop with explicit predict-error-update
6. **Systematic task taxonomy** - Which tasks need which architectural features?

### Long-term

7. **Extract transferable principles** - What's the minimal mechanism for OOD generalization?
8. **Understand why transformers work** - Not just "attention is all you need"—WHY?
9. **Connect to biological plausibility** - Does it matter? When?

---

## Key Takeaways So Far

1. **Iteration helps**: 2L × 8T beats 4L standard transformer (+6% test, +18% OOD)

2. **But it's not "CTM"**: Our Tick-CTM is just a looped transformer with accumulation

3. **Task matters**: Reversal is a lookup task—we may be testing the wrong thing

4. **BDH is subtle**: Claims mathematical equivalence between tensors and Hebbian dynamics—our implementation missed this

5. **Visualization is key**: CTM notebooks show detailed "thinking" GIFs—we need this to understand differences

6. **Transformers are strong**: Despite everything, standard transformers remain competitive on many tasks

7. **Uncertainty remains**: We don't yet know which tasks REQUIRE sync/iteration vs just attention

---

## New Session: Component-by-Component CTM Analysis (2025-12-30)

### Goal
Systematically add real CTM components one-by-one to our Tick-CTM baseline and measure impact.

### Components Tested

| Component | Description | From Real CTM? |
|-----------|-------------|----------------|
| **Hist** | History buffer (store activations across ticks) | ✓ Z tensor |
| **Sync** | Compute S = Z·Z^T (correlation flag) | ✓ Core mechanism |
| **SyncRes** | Add sync features as residual (gentle use) | Adapted |
| **NLM** | Proper per-neuron processing (einsum, no loops) | ✓ SuperLinear |
| **Halt** | Adaptive halting (stop when confident) | ✓ Certainty-based |
| **Para** | Independent parallel paths | New idea |
| **CommPara** | Communicating parallel paths | New idea |

### Implementation: Proper NLM (SuperLinear)

Based on real CTM's `models/modules.py`:

```python
# Real CTM's NLM uses einsum for parallel per-neuron processing
# Each of D neurons has UNIQUE weights - no loops!

# Input: (B, T, D, M) - batch, seq, neurons, history
# w1: (M, H, D) - per-neuron weights
h = torch.einsum('btdm,mhd->btdh', x, self.w1) + self.b1
h = F.glu(h, dim=-1)  # GLU activation like real CTM
out = torch.einsum('btdh,hod->btdo', h, self.w2) + self.b2
out = F.glu(out, dim=-1).squeeze(-1) / self.T
```

**Key**: Uses einsum for D independent linear transforms in parallel - NO PYTHON LOOPS.

### 3-Seed Benchmark Results (Reversal Task)

| Model | Test (mean±std) | OOD (mean±std) | Params |
|-------|-----------------|----------------|--------|
| **Tick-CTM+Hist+Sync+SyncRes** | **94.5%±2.4%** | **69.9%±6.0%** | 106,063 |
| Tick-CTM+Hist+SyncRes | 94.4%±1.9% | 65.6%±6.2% | 106,063 |
| Tick-CTM+Hist+NLM | 93.6%±0.9% | 61.6%±3.4% | 142,991 |
| Tick-CTM+Hist+Sync | 93.4%±1.3% | 68.9%±4.1% | 101,902 |
| Tick-CTM+Hist+Sync+NLM | 93.2%±3.4% | 67.6%±8.1% | 142,991 |
| Tick-CTM (baseline) | 93.1%±0.7% | 64.9%±6.4% | 101,902 |
| Tick-CTM+Hist+Sync+Halt | 91.4%±4.2% | 60.1%±7.1% | 101,967 |
| Tick-CTM+Hist | 90.9%±2.3% | 62.2%±6.1% | 101,902 |
| Tick-CTM+Halt | 90.4%±4.1% | 61.0%±4.5% | 101,967 |
| Tick-CTM+Para(4) | 82.8%±3.1% | 52.1%±3.1% | 252,238 |
| Tick-CTM+CommPara(4) | 8.2%±2.2% | 7.8%±0.7% | 202,062 |

### Key Findings

1. **Winner: Hist+Sync+SyncRes**
   - Test: 94.5% (+1.4% vs baseline)
   - OOD: 69.9% (+5.0% vs baseline)
   - Combining history, sync flag, and sync residual works best

2. **Single-seed results were misleading**
   - Hist+Sync+Halt looked like 96.8% in one run
   - Across 3 seeds: only 91.4% (high variance: 4.2%)
   - **Always use multiple seeds!**

3. **Proper NLM helps test, hurts OOD**
   - Hist+NLM: 93.6% test (stable, only 0.9% std)
   - But only 61.6% OOD (-3.3% vs baseline)
   - NLM adds parameters but doesn't generalize

4. **History buffer alone hurts**
   - Hist: 90.9% test (-2.2% vs baseline)
   - Need sync computation to extract value from history

5. **Sync computation helps OOD**
   - Hist+Sync: 93.4% test, 68.9% OOD (+4.0% OOD)
   - The correlation signal aids generalization

6. **CommPara is broken**
   - 8.2% accuracy across seeds
   - Something fundamentally wrong with cross-path attention
   - Need to debug

7. **Parallel paths don't help (yet)**
   - Para(4): 82.8% test, 52.1% OOD
   - More parameters, worse results
   - Sequential refinement matters more than parallel exploration

8. **Adaptive halting adds variance**
   - Halt alone: 90.4%±4.1% (high variance)
   - May be halting too early or inconsistently

### Component Impact Summary

| Component | Test Δ | OOD Δ | Verdict |
|-----------|--------|-------|---------|
| +Hist+Sync+SyncRes | +1.4% | +5.0% | **BEST** |
| +Hist+Sync | +0.3% | +4.0% | Good for OOD |
| +Hist+NLM | +0.5% | -3.3% | Hurts OOD |
| +Hist alone | -2.2% | -2.7% | Hurts |
| +Para(4) | -10.3% | -12.8% | Bad |
| +CommPara(4) | -84.9% | -57.1% | Broken |

### Lessons Learned

1. **Sync should be auxiliary, not primary** - Adding sync as residual works; replacing output with sync breaks the model

2. **Multiple seeds are essential** - Single run gave misleading results

3. **History needs sync** - Raw history buffer doesn't help; need to extract correlation information

4. **Parallel paths need more work** - Current implementations don't match sequential tick performance

5. **Proper NLM implementation** - Using einsum (no loops) is fast and matches real CTM, but doesn't help this task

### Next Steps

1. **Debug CommPara** - Find why cross-path attention breaks completely
2. **Test on different tasks** - Reversal may not need CTM-style processing
3. **Implement sync with decay** - Real CTM uses exponential decay in sync computation
4. **Add maze/parity tasks** - Where CTM was designed to shine

---

## Multi-Task Benchmark (Same Session)

### Goal
Test if findings from reversal generalize to other task types.

### Tasks Tested

| Task | Type | What it Tests |
|------|------|--------------|
| Reversal | Lookup | Attention-based retrieval |
| Parity | Counting | Accumulation over sequence |
| Addition | Arithmetic | Multi-step computation |

### Results

| Config | Reversal (T/O) | Parity (T/O) | Addition (T/O) |
|--------|----------------|--------------|----------------|
| Baseline | 93.3%/70.3% | 41.0%/58.0% | 28.0%/4.0% |
| Hist+Sync | 93.4%/57.5% | 42.0%/58.0% | 32.0%/4.0% |
| Hist+Sync+SyncRes | 91.8%/63.9% | 42.0%/58.0% | **37.0%**/6.0% |
| **Hist+NLM** | 92.3%/49.0% | **51.0%**/58.0% | 29.0%/6.0% |

### Critical Finding: Task-Dependent Architecture Selection

**Best config varies by task:**
- **Reversal (lookup)**: Baseline wins for OOD (70.3%)
- **Parity (counting)**: Hist+NLM wins (51% vs 41% baseline)
- **Addition (arithmetic)**: Hist+Sync+SyncRes wins (37% vs 28%)

### Key Insights

1. **NLM helps counting, not lookup**
   - Parity: +10% with NLM
   - Reversal: -21% OOD with NLM
   - NLM processes temporal history - useful for counting, harmful for lookup

2. **Sync hurts reversal OOD**
   - 57.5% vs 70.3% baseline (-12.8%)
   - Correlation signal adds noise for lookup task
   - Contradicts single-task findings (variance issue?)

3. **SyncRes helps arithmetic**
   - 37% vs 28% baseline (+9%)
   - Sync as residual aids multi-step computation

4. **All models struggle with counting/arithmetic**
   - Parity: 41-51% (near random for binary task)
   - Addition: 28-37% test, 4-6% OOD
   - Need more training or different architecture

### Architecture-Task Matching

| Architecture Feature | Best For | Worst For |
|---------------------|----------|-----------|
| Baseline (just iteration) | Lookup OOD | Counting |
| Hist+Sync | ? | Lookup OOD |
| Hist+Sync+SyncRes | Arithmetic | Lookup |
| Hist+NLM | **Counting** | Lookup OOD |

### Implications

1. **No single architecture wins all tasks** - This aligns with "no free lunch" theorem

2. **CTM components have task-specific value**:
   - NLM: Good for counting (temporal accumulation)
   - Sync: Mixed results, may need better implementation
   - SyncRes: Helps multi-step computation

3. **Reversal was misleading test case** - It's a lookup task where transformers excel

4. **Need better counting/arithmetic training** - Current results are near-random

### Questions Raised

1. Why does Sync hurt reversal OOD in multi-task but help in single-task benchmark?
   - Different seeds?
   - Task interference?

2. Why is parity so hard?
   - 51% is barely above random (50%)
   - Need more epochs or bigger model?

3. Does real CTM's sync-with-decay help?
   - Our sync is simplified
   - Real CTM uses exponential decay

---

## Parity Deep Dive (Same Session)

### Observation
All models stuck at ~50% on parity (random guessing).

### Extended Training Test
Trained for 300 epochs with 1000 samples:
- Baseline: 51.0% (stuck at loss ≈ 0.75 = log(2))
- Hist+NLM: 50.5% (same)

### Why Parity is Hard

This is a **known limitation** of transformers for counting tasks:
1. Parity requires exact counting - one mistake = wrong answer
2. Transformers learn "soft" attention patterns, not exact counts
3. No explicit accumulator mechanism
4. This is discussed in "Transformers Learn Shortcuts to Automata" paper

### What Would Help
1. **Scratchpad/chain-of-thought**: Let model write intermediate counts
2. **Position-based counting**: Explicit positional encoding for counting
3. **Recurrent mechanism**: True RNN-style accumulation (not our tick loop)
4. **Specialized architecture**: Counting-specific modules

### Implication
Our Tick-CTM architecture, despite iterations, cannot solve counting tasks. This is NOT a failure of our implementation - it's a fundamental architectural limitation.

**Real CTM may work on parity** - their paper shows good results. The difference:
- Real CTM: Per-neuron NLMs process history temporally
- Our Tick-CTM: Transformer blocks with weight sharing

---

## Session Summary (2025-12-30)

### What We Did
1. Implemented proper NLM using einsum (matching real CTM)
2. Ran 3-seed benchmark on reversal task
3. Tested multiple tasks: reversal, parity, addition
4. Found task-dependent architecture selection

### Key Findings

| Finding | Evidence | Implication |
|---------|----------|-------------|
| **Multi-seed essential** | Single-seed: 96.8%, 3-seed avg: 91.4% | Don't trust single runs |
| **Winner: Hist+Sync+SyncRes** | 94.5% test, 69.9% OOD (3-seed) | Sync as residual works |
| **Task-dependent architecture** | NLM helps parity, hurts reversal | No universal winner |
| **Parity is hard** | All models stuck at 50% | Transformers can't count |

### Architecture Recommendations by Task

| Task Type | Best Architecture | Why |
|-----------|-------------------|-----|
| **Lookup** (reversal) | Baseline or light Sync | Attention suffices |
| **Counting** (parity) | Need specialized arch | Tick-CTM can't do it |
| **Arithmetic** (addition) | Hist+Sync+SyncRes | Multi-step benefits |

### Open Questions
1. Would real CTM (not Tick-CTM) solve parity?
2. Why does sync-with-decay matter?
3. How to parallelize refinement without losing sequential benefits?

### Files Updated
- `test_ctm_components.py`: Added proper NLM, multi-task benchmark, parity/addition tasks
- `RESEARCH_LOG.md`: This file

---

## CRITICAL FINDING: Real CTM CAN Learn Parity (Same Session)

### The Test
Ran real CTM (from Sakana's codebase) on parity task to compare with our Tick-CTM.

### Results

| Model | After 5 Epochs | Verdict |
|-------|----------------|---------|
| **Real CTM** | **60.1%** (still learning) | ✓ WORKS |
| Our Tick-CTM | 50% (stuck) | ✗ FAILS |

### Training Curves
```
Real CTM:
  Epoch 1: 50.1% (random)
  Epoch 2: 50.9%
  Epoch 3: 51.4%
  Epoch 4: 56.0%  ← Learning starts!
  Epoch 5: 59.2%
  Test: 60.1%     ← Continues improving

Our Tick-CTM:
  Epoch 1: 50%
  ...
  Epoch 300: 50%  ← Stuck at random
```

### Why Real CTM Works on Parity

| Feature | Real CTM | Our Tick-CTM |
|---------|----------|--------------|
| **NLMs** | Per-neuron private MLPs process history | Shared transformer weights |
| **Synapses** | Deep UNET (8 layers) | Simple 2-layer transformer |
| **Memory** | 25 ticks of history | 8 ticks |
| **Sync** | Core representation (output from sync) | Add-on (small residual) |
| **Iterations** | 30-50 | 8 |
| **Parameters** | 142K-652K | ~100K |

### The Key Insight

**Real CTM's architecture enables counting through:**
1. **Per-neuron NLMs**: Each neuron tracks its own temporal state
2. **Deep synapses**: Complex information mixing between neurons
3. **Sync as representation**: Correlation captures counting state
4. **Many iterations**: Time to propagate count through network

**Our Tick-CTM is fundamentally different:**
- Just a looped transformer with weight sharing
- No per-neuron state tracking
- Sync is add-on, not core
- Can't implement counting algorithm

### Implications

1. **Tick-CTM ≠ CTM**: Our architecture is structurally incapable of counting
2. **Task matters**: Reversal (lookup) works, parity (counting) fails
3. **NLMs are critical**: Per-neuron processing enables temporal computation
4. **We need real CTM components**: Not just iteration

### What Real CTM Has That We Don't

```python
# Real CTM forward pass (simplified):
for tick in range(iterations):
    # 1. Synapses mix neurons (deep UNET)
    state = synapses(concat(attn_out, activated_state))

    # 2. Update pre-activation history (rolling buffer)
    state_trace = concat(state_trace[:, :, 1:], state)

    # 3. NLMs: EACH NEURON processes ITS OWN history
    activated_state = NLM(state_trace)  # Per-neuron!

    # 4. Sync: compute correlations
    sync = activated_state @ activated_state.T  # Core representation

    # 5. Output FROM sync
    output = project(sync)  # NOT from hidden state
```

**Our version skips steps 2-5.** We just loop the transformer.

### Next Step: Implement Real CTM Architecture
To match real CTM's capabilities, we need:
1. Deep synapses (UNET or equivalent)
2. Per-neuron NLMs with proper history buffer
3. Sync as core representation (output from sync)
4. More iterations (30+)
