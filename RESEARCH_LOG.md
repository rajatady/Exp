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

## Lessons Learned

1. **Read the paper carefully before implementing** - We missed core CTM innovations
2. **Test on diverse tasks early** - We drew conclusions from single task type
3. **Question assumptions** - "History" meant different things in paper vs implementation
4. **Track progress in writing** - 12 hours without documentation led to confusion
5. **Simple baselines first** - Simple Accumulator matched our "CTM" because it wasn't really CTM
