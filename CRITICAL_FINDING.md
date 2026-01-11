# CRITICAL FINDING: RealCTM Never Learned to Count

## The Discovery

The reported 100% accuracy of RealCTM on parity was **an artifact of teacher forcing**, not actual counting ability.

## Evidence

### Training Setup (from behavioral_analysis.py)
```python
inputs = x[:, :-1]   # [BOS, bits, SEP, parity]
targets = x[:, 1:]   # [bits, SEP, parity, EOS]
```

For parity prediction at position `sep_pos`:
- Input at `sep_pos`: SEP (token 3)
- Input at `sep_pos + 1`: **parity token (6 or 7)**
- Target at `sep_pos`: parity token

The model sees the answer in its input!

### Evaluation (same flaw)
```python
inputs = x[:, :-1]  # Still includes parity!
preds[i, sep_pos] == targets[i, sep_pos]  # Checking parity prediction
```

### Fair Test Results

| Test Type | RealCTM Accuracy |
|-----------|------------------|
| Teacher forcing (parity in input) | 100% |
| **Fair test (no parity in input)** | **56%** |

56% is barely above random (50%).

## Implications

1. **RealCTM doesn't actually count** - it learned to copy
2. **Our 50% "failures" were correct** - that's the real baseline
3. **The research premise was flawed** - we were trying to replicate a non-existent capability

## What This Means for the Architecture

The 5 mechanisms we identified (NLM, sync, history, etc.) may still be valuable, but:
- They weren't proven to enable counting
- The claimed success was measurement error
- We need a different task/evaluation to validate these mechanisms

## Recommendations

1. **Fix the evaluation**: Remove parity from input when testing
2. **Test on truly sequential tasks**: Where the model can't peek at the answer
3. **Validate claims independently**: Don't trust reported accuracies without checking methodology

## Files Updated
- This finding invalidates results in:
  - `saved_models/behavioral_analysis_results.json`
  - `RESEARCH_LOG.md` references to "RealCTM 100% on parity"
  - `REALCTM_ANALYSIS.md` claims about counting ability
