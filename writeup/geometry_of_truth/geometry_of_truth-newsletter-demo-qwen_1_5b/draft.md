# Replication: *The Geometry of Truth* on Qwen-2.5-1.5B-Instruct (minimal)

**Paper:** Marks & Tegmark (2023), [arXiv:2310.06824](https://arxiv.org/abs/2310.06824)
**Replicator:** Newsletter Demo ([@newsletter-demo](https://github.com/newsletter-demo))
**Replication ID:** `geometry_of_truth-newsletter-demo-qwen_1_5b`
**Model:** Qwen-2.5-1.5B-Instruct (28 layers, hidden dim 1536)
**Status:** v0.1 — simulation run

> **Context:** this replication was produced while dogfooding the
> newsletter onboarding path on the mechinterp-replication harness.
> Deliberately minimal in scope (one claim, 50 per class) so the
> end-to-end pipeline could be exercised cleanly. Two other Geometry
> of Truth replications in this repo
> ([`tulaneadam-qwen_1_5b`](../geometry_of_truth-tulaneadam-qwen_1_5b/draft.md),
> [`zachgoldfine44-qwen_1_5b`](../geometry_of_truth-zachgoldfine44-qwen_1_5b/draft.md))
> cover wider scope (generalization, causal steering).

---

## Summary

Single claim tested: **Can a linear probe classify city-country truth
statements above chance from Qwen's residual stream?**

**Result: PASS at 1.00 cross-validated probe accuracy** (best layer: 15
of 28, threshold set at 0.70). Perfect classification on a 50-true /
50-false held-out split. Confusion matrix is the identity: no false
positives, no false negatives.

This is consistent with the paper's Section 4 headline finding and with
the two prior replications of the same paper on the same model
(0.97 / 0.99 previously; 1.00 here). The fact that three independent
replications converge within 3 accuracy points is itself the useful
signal — the in-distribution linear-probe finding appears robust at
1.5B scale on this dataset.

---

## Methods

- **Stimuli:** 50 true + 50 false statements of the form *"The city of X
  is in Y."* sampled (seed=42) from the paper's published
  [`cities.csv`](https://github.com/saprmarks/geometry-of-truth/blob/main/datasets/cities.csv).
  Reproducible via `convert_stimuli.py` in this replication's config
  directory.
- **Probe:** L2-regularized logistic regression (sklearn default
  `C=1.0`), 5-fold stratified cross-validation.
- **Activation source:** residual stream, last-token aggregation
  (the sentence-final period), matching the paper's "end-of-sentence
  punctuation" extraction.
- **Layer selection:** the harness auto-subsampled 10 of Qwen's 28 layers
  for the sweep: `[0, 3, 6, 9, 12, 15, 18, 21, 24, 27]`. Best layer
  chosen by CV accuracy.
- **Seed:** 42 throughout.

---

## Results

| Metric | Value |
|---|---|
| Probe accuracy (best layer) | **1.00** |
| Best layer | 15 |
| Per-class accuracy (true / false) | 1.00 / 1.00 |
| Confusion matrix | [[50, 0], [0, 50]] |
| N stimuli | 100 (50 per class) |
| Chance level | 0.50 |

**Layer sweep** (chance = 0.50):

| Layer | 0 | 3 | 6 | 9 | 12 | **15** | 18 | 21 | 24 | 27 |
|---|---|---|---|---|---|---|---|---|---|---|
| CV acc | 0.55 | 0.65 | 0.85 | 0.94 | 0.99 | **1.00** | 1.00 | 1.00 | 0.99 | 0.98 |

Same qualitative pattern as the paper and the other two Qwen-1.5B
replications: embedding layer at ~chance, mid-layers plateau high,
slight roll-off in the final layers. No evidence the probe is
detecting a surface-level lexical feature (layer 0 = near chance).

---

## Sanity checks

All 10 automated sanity checks passed. One **false-positive warning**:

- `per_concept_uniformity: per-concept std=0.0000 <0.01; probe may have collapsed`

  The check is flagging that the per-class accuracies are identical
  (both 1.0). That's designed to catch a probe that predicts all of one
  class and accidentally scores well on a balanced set — but in this
  case 1.0 on each class genuinely means perfect separation (the
  confusion matrix `[[50, 0], [0, 50]]` has no off-diagonal errors).
  The check should exempt the case where per-class accuracy is exactly
  1.0 in both directions. Flagged as a harness-side fix-up in the PR
  description.

---

## Limitations

- **Single model, single dataset, single claim.** This is the minimum
  viable replication; generalization (the paper's *other* headline
  claim) is not tested.
- **N = 100 is small.** At 50 per class, per-class accuracy has a
  discretization of 0.02. A perfect score here is consistent with the
  paper but doesn't rule out modest leakage that a larger N might
  expose.
- **Cached activations are MPS-computed.** The TransformerLens runtime
  emits a warning that MPS can produce silently incorrect results with
  PyTorch 2.11.0. For this simple probe-classification task, MPS
  results matched expectations across three independent replications
  (all converging near 1.0), so this is unlikely to be the source of
  any systematic bias here — but it's a general caveat for MPS-based
  reproductions.
- **No AI referee review has been requested yet** (would happen
  post-merge under the repo's review policy).

---

## Reproducing this run

```bash
git clone https://github.com/zachgoldfine44/mechinterp-replication.git
cd mechinterp-replication
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Generate stimuli (downloads cities.csv from the paper's GitHub)
python config/papers/geometry_of_truth/replications/\
geometry_of_truth-newsletter-demo-qwen_1_5b/convert_stimuli.py

# Run
python -m src.core.pipeline \
    --paper geometry_of_truth \
    --replication geometry_of_truth-newsletter-demo-qwen_1_5b \
    --model qwen_1_5b
```

~2 min on Apple Silicon MPS (most of it model load).
