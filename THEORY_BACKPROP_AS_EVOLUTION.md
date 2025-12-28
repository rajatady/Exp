# Theory: Backprop as Evolution

## Meta
**Date**: 2025-12-28
**Status**: Raw exploration, not proven
**Goal**: Understand the principled "line" connecting all working architectures and predict what comes next

---

## The Core Hypothesis

### Raw Version (User's Words)
> "What if the thread connecting all the working architectures is backprop, and backprop itself is like evolution?"

> "Don't think of backprop as a learning rule for intelligence, think of it as a search rule for finding the structure that can navigate reality and survive, and intelligence just happens to be the best way to do it."

> "Think of backprop as doing evolution over multiple steps. Think of every single step of backprop as a new generation of the organism, but within that new step we have intelligence emerging."

> "What if we don't use backprop to train the inner loop?"

### Refined Version
```
OUTER LOOP (Search):
  - Evolution searches for organisms
  - Backprop searches for weight configurations
  - Both: parallel, local, pressure-driven

INNER LOOP (Intelligence):
  - Brains operate with their own dynamics (NOT evolution)
  - Networks should operate with their own dynamics (NOT backprop)
  - Both: the FOUND structure, not the SEARCH process
```

---

## The Analogy: Backprop ≈ Evolution

### Similarities (Refined)

| Property | Evolution | Backprop |
|----------|-----------|----------|
| Parallel | Millions of organisms simultaneously | Millions of parameters simultaneously |
| Local signal | "Did I survive?" (no long-chain credit) | Local gradient (vanishing over long chains) |
| No global knowledge | No organism knows the optimal form | No parameter knows the optimal config |
| Pressure-driven | Survival pressure shapes form | Loss pressure shapes weights |
| Finds local optima | Good enough to survive, not optimal | Good enough to minimize loss, not optimal |
| Continuous | Generations overlap, always ongoing | Steps blend, always refining |

### Initial Attacks (And Responses)

**Attack**: "Evolution is discrete (generations), backprop is continuous"
**Response**: Evolution is NOT discrete. Generations overlap. Mutations happen continuously. We just snapshot it.

**Attack**: "Backprop is deterministic, evolution is stochastic"
**Response**: Backprop is NOT deterministic. Random init, random batches, dropout, many paths through landscape.

**Attack**: "Evolution is open-ended, backprop is bounded by architecture"
**Response**: Evolution also has constraints (body plans rarely change after Cambrian). Both: fixed outer structure, fluid inner structure.

**Attack**: "Evolution is multi-objective, backprop is single loss"
**Response**: The thread is PRESSURE. Survival/reproduction/mate-selection are all pressures. Loss is pressure. Same principle, different form.

---

## The Two Levels

### Level 1: Search (Outer Loop)
- Evolution finds organisms
- Backprop finds weight configurations
- **Timescale**: Generations / Training iterations
- **Mechanism**: Parallel local optimization under pressure

### Level 2: Intelligence (Inner Loop)
- Organisms use brains to navigate reality
- Networks use forward pass to process inputs
- **Timescale**: Moments / Inference time
- **Mechanism**: ??? (This is the question)

### The Gap
- Evolution found brains with RICH internal dynamics (oscillations, sync, plasticity)
- Backprop finds networks with MINIMAL internal dynamics (just forward pass)
- **Maybe that's what's missing?**

---

## The Key Insight: Inner Loop Should NOT Use Backprop

### Raw Thought (User)
> "If backprop is equal to evolution, but evolution within evolution, the brain is also operating, but operating with a different principle, while both being qualitatively equally fast."

### What This Means
- Evolution (outer) found brains (inner)
- Brains do NOT run evolution internally
- Brains run something ELSE: neural dynamics, Hebbian learning, predictive coding, etc.

Similarly:
- Backprop (outer) finds networks (inner)
- Networks should NOT run backprop internally at inference
- Networks should run something ELSE: ???

### What Could "Something Else" Be?
1. **Temporal dynamics** (CTM's ticks)
2. **Energy minimization** (Hopfield networks, settling)
3. **Local learning rules** (Hebbian, STDP)
4. **Predictive coding** (minimize prediction error locally)
5. **Synchronization** (CTM's sync as representation)
6. **Search/exploration** (Monte Carlo, beam search internalized)

---

## The CTM Connection

### What CTM Does
- OUTER: Trained with backprop (standard)
- INNER: Has temporal dynamics (ticks, sync evolution)

### What This Suggests
CTM is an example of the theory: backprop finds a structure that has its own internal dynamics.

The dynamics (ticks, sync) are NOT backprop. They're something else.

### Open Question
Is CTM's specific mechanism (sync as representation) the RIGHT "something else"?
Or is it just ONE example of adding internal dynamics?

---

## Falsifiable Predictions

### Prediction 1: Dynamics Should Help on "Thinking" Tasks
- Tasks requiring search/exploration
- Tasks requiring iterative refinement
- Tasks with variable difficulty

**Test**: Compare feedforward vs dynamic on maze solving, planning, multi-step reasoning.
**Control**: Same total compute (layers × 1 = ticks × layers)

### Prediction 2: Inner Loop Learning Should Enable New Capabilities
If we add learning (not just dynamics) at inference time:
- Should adapt to new inputs without retraining
- Should handle distribution shift better
- Should show meta-learning properties

**Test**: Train on distribution A, test on distribution B, compare:
- Static network (forward pass only)
- Dynamic network (ticks, no inner learning)
- Adaptive network (inner learning rule at inference)

### Prediction 3: The Inner Loop Should NOT Be Backprop
If we run backprop at inference (like test-time training):
- Should be slower than a native inner mechanism
- Should be less biologically plausible
- Should work, but not be optimal

**Test**: Compare:
- Network with inference-time backprop (test-time training)
- Network with inference-time local rule (Hebbian, predictive coding)
- Measure: speed, accuracy, generalization

---

## Questions We're Asking

### Foundational
1. Is there a principled "line" connecting all working architectures?
2. Is backprop ≈ evolution a real pattern or curve fitting?
3. What would falsify this theory?

### Architectural
4. What should happen in the inner loop (inference time)?
5. Is CTM's temporal dynamics the right approach?
6. What's the minimal mechanism that adds "thinking"?

### Theoretical
7. Can we prove that some tasks REQUIRE internal dynamics?
8. What computational class do dynamic networks belong to?
9. Is there a task that feedforward provably can't solve but dynamic can?

### Biological
10. What rules does the brain use at inference time?
11. Why did evolution find dynamics instead of deeper feedforward?
12. Is biological plausibility a hint or a distraction?

---

## The Meta-Problem

### Pattern Matching Risk
> "You are a pattern matcher, so you will end up biasing towards the right or the wrong depending on what the user says"

This is true. Claude will pattern-match to user's framing.

### How Scientists Navigate This
1. **Predictions**: Theory must predict something specific and testable
2. **Surprise**: Good theories predict things you wouldn't expect otherwise
3. **Falsification**: Try to BREAK your theory, not prove it
4. **Consensus**: Many people trying to break it and failing

### Our Current Status
- We have an ANALOGY (backprop ≈ evolution)
- We have a HYPOTHESIS (inner loop needs different rules)
- We do NOT have a PREDICTION that would surprise us if true
- We do NOT have a clear FALSIFICATION condition

---

## Next Steps

### Immediate
1. Find/construct a task where we can PROVE feedforward can't solve it
2. Show that adding internal dynamics solves it
3. This would be evidence (not proof) for the theory

### Longer Term
1. Derive mathematically what "internal dynamics" provide
2. Find the MINIMAL mechanism (not full CTM complexity)
3. Test whether the mechanism generalizes across tasks

### The Big Question
> "What if we don't use backprop to train the inner loop?"

What WOULD we use? Options:
- Pre-defined dynamics (like CTM's ticks) - no learning, just evolution
- Local learning rules (Hebbian) - learning without backprop
- Energy minimization (Hopfield) - settling to attractors
- Predictive coding - hierarchical prediction error minimization

---

## Raw Thoughts Archive

### User's Raw Thoughts
1. "Backprop is like evolution" - parallel search for fit structures
2. "Intelligence emerges, it's not the goal" - side effect of survival/loss pressure
3. "Evolution without evolving the body" - backprop changes weights, not architecture
4. "Inner loop should be different from outer loop" - brain doesn't run evolution
5. "Both being qualitatively equally fast" - inner and outer operate at different timescales but both are efficient

### Implications Not Yet Explored
- If backprop = evolution, what is "mutation" in backprop? (Random init? Noise? Dropout?)
- If backprop = evolution, what is "reproduction" in backprop? (Spawning new models? Fine-tuning?)
- If backprop = evolution, what is "death" in backprop? (Pruning? Early stopping? Mode collapse?)
- What is the "environment" for backprop? (The dataset? The loss function? The task distribution?)

---

## Experiments

### Experiment 1: Inner Learning (test_inner_learning.py)

**Setup**:
- Task: Pattern completion (8 positions, 50-75% masked)
- Models:
  1. Standard: Forward pass only
  2. Dynamics: State evolves over 4 ticks (like CTM)
  3. Hebbian: Weights change at inference with `Δw = η * pre * post`

**Results (hard test, 75% masked)**:
| Model | Masked Accuracy |
|-------|-----------------|
| Standard | 61.7% |
| Dynamics | 60.2% (-1.5%) |
| Hebbian | 53.7% (-8.0%) |

**Observation**: Neither dynamics nor Hebbian helped. Both hurt.

**Known Confounds**:
1. Task too simple (can be memorized)
2. Hebbian rule too simplistic (no normalization, unstable)
3. Hebbian LR arbitrary (0.01, not tuned)
4. Not controlled for compute
5. More params in dynamics/Hebbian models

**Interpretation**:
This doesn't falsify the theory. It means:
- Either wrong task (pattern completion doesn't need "thinking")
- Or wrong implementation of inner learning
- Or the theory is wrong

**Next**: Try different task or different inner learning rule.

### Experiment 1b: Inner Learning with Transformer Baseline (test_inner_learning_v2.py)

**Fixed confounds**:
- Using Transformer as baseline (not MLP)
- Analyzing HOW each model solves the task

**Results (hard test, 75% masked)**:
| Model | Accuracy |
|-------|----------|
| Transformer | 54.0% |
| +Dynamics | 55.2% (+1.2%) |
| +Hebbian | 49.2% (-4.8%) |

**Analysis of internal mechanisms**:
- Dynamics: State changes by constant ~4.0 each tick (not settling/converging)
- Hebbian: Weight changes happening but hurting performance

**Interpretation**:
- Task may be wrong (pattern completion doesn't require iteration)
- Dynamics not settling = not doing useful refinement
- Hebbian interferes with what backprop learned

---

## Version History
- v1 (2025-12-28): Initial capture of theory and discussion
- v2 (2025-12-28): Added Experiment 1 (inner learning) - negative result
- v3 (2025-12-28): Added Experiment 1b with Transformer baseline - still negative
