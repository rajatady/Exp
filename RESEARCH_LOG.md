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

### Results (Multi-Seed: 5 seeds)

| Model | Test (mean ± std) |
|-------|-------------------|
| Standard | 52.6% ± 5.8% |
| **Inner Loop** | **56.5% ± 11.9%** |
| Accumulator | 51.5% ± 3.4% |

### Per-Seed Comparison (Inner Loop vs Standard)

| Seed | Difference |
|------|------------|
| 42 | +1.5% |
| 123 | +4.4% |
| 456 | +1.5% |
| 789 | -14.7% |
| 1024 | +26.5% |

**Mean improvement: +3.8% ± 13.2%**
**Wins: 4/5 seeds (80%)**

### Key Finding

**Inner Loop helps but effect is modest and variable**

- Inner Loop wins on 4/5 seeds (consistent direction)
- Average improvement: +3.8% (not the +13.2% from single seed)
- High variance: Some seeds benefit a lot (+26%), others don't (-15%)
- Accumulator consistently hurts (-1.2% average)

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

## Critical Discovery: Multi-Pass Wins (test_alternative_hypothesis.py)

### Hypothesis Testing

We tested what component actually provides the benefit:

| Model | Mean Accuracy | vs Standard |
|-------|---------------|-------------|
| Standard | 33.3% | baseline |
| Input Pred Loss | 36.3% | +2.9% |
| Inner Loop (Fixed) | 39.2% | +5.9% |
| Inner Loop (Train Only) | 51.0% | +17.6% |
| **Multi-Pass (4x)** | **60.3%** | **+27.0%** |

### The Real Mechanism

**Multi-Pass just reruns the encoder 4 times - no settling, no error, no input prediction:**
```python
for _ in range(4):
    h = encoder(h)  # Just rerun!
```

**The benefit is NOT from:**
- Input as ground truth ✗
- Self-modification ✗
- Settling dynamics ✗

**The benefit IS from:**
- Iterative refinement of representations ✓
- Multiple passes through attention ✓
- Shared weights reused across passes ✓

This is the **looped transformer** mechanism, not the "inner loop with feedback" hypothesis.

### Why Does This Help Compositional Generalization?

Multiple attention passes allow:
1. First pass: Recognize individual components (WALK, THRICE)
2. Subsequent passes: Integrate components, resolve bindings
3. Each pass refines the representation

Novel combinations benefit because the model can iteratively figure out how components relate, rather than pattern-matching in one shot.

### Comprehensive Analysis (comprehensive_multipass_analysis.py)

**Q1: Does multi-pass generalize across tasks?**
- Helps on compositional tasks with sufficient data (+17.6%)
- Neutral/hurts on simple tasks (copy, counting)
- Data size matters: 104 examples → helps; 8 examples → hurts

**Q2: What changes between passes?**
- Representation change: 5.7 → 1.9 (CONVERGING)
- Prediction accuracy: 7% → 86% (IMPROVING)
- This IS iterative refinement

**Q3: Connection to intelligence loop**
```
Intelligence Loop: Predict → Feedback → Update
Multi-Pass:        Predict → [Attention] → Refine
```

Multi-pass provides IMPLICIT feedback via attention:
- Each pass sees all updated representations
- If position A changes, position B notices on next pass
- No explicit error needed - attention IS the feedback mechanism

---

## Updated Files

| File | Purpose | Key Finding |
|------|---------|-------------|
| test_inner_loop.py | Inner loop hypothesis test | OOD generalization improves |
| test_compositional_gen.py | Compositional generalization | Inner Loop helps |
| test_compositional_gen_multi_seed.py | Multi-seed validation | +3.8% avg, 4/5 wins |
| analyze_seed_variance.py | Seed variance analysis | Error doesn't decrease in inner loop |
| diagnose_inner_loop.py | Inner loop diagnostics | Re-running encoder undoes updates |
| test_alternative_hypothesis.py | Alternative mechanisms | **Multi-Pass wins by +27%** |

---

## Lessons Learned

1. **Read the paper carefully before implementing** - We missed core CTM innovations
2. **Test on diverse tasks early** - We drew conclusions from single task type
3. **Question assumptions** - "History" meant different things in paper vs implementation
4. **Track progress in writing** - 12 hours without documentation led to confusion
5. **Simple baselines first** - Simple Accumulator matched our "CTM" because it wasn't really CTM
6. **Ground truth at inference = input** - The inner loop uses input as self-supervision
7. **Compositional generalization is the key test** - Not in-distribution accuracy
