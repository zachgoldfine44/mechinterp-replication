# Replication Attempt: *The Geometry of Truth* on Qwen-2.5-1.5B-Instruct

**Paper:** Marks & Tegmark (2023), *The Geometry of Truth: Emergent Linear
Structure in Large Language Model Representations of True/False Datasets.*
[arXiv:2310.06824](https://arxiv.org/abs/2310.06824)

**Replicator:** [@zachgoldfine44](https://github.com/zachgoldfine44)
**Replication ID:** `geometry_of_truth-zachgoldfine44-qwen_1_5b`
**Model replicated on:** Qwen-2.5-1.5B-Instruct (28 layers, 1,536 hidden dim)
**Model in original paper:** LLaMA-2 7B / 13B / 70B (base)
**Environment:** Apple Silicon MacBook, MPS, local only

---

## TL;DR

1. **Within-distribution truth is near-perfect at 1.5B scale.** A logistic-
   regression probe trained at layer 12 of Qwen-2.5-1.5B-Instruct on
   balanced `cities` true/false statements reaches **99.0%** cross-validated
   accuracy — matching the paper's LLaMA-2-13B/70B results despite being
   ~10× smaller than the smallest model the paper studied.
2. **Cross-dataset generalization mostly fails at this scale.** The same
   probe transfers badly to three structurally different held-out sets.
   The probe does not encode an abstract "truth" direction at 1.5B; it
   encodes something like "affirmative cities-style statement."
3. **The failure mode is interpretable and replicates a known paper
   finding.** The probe flips on negated statements (neg_cities) and
   collapses to always-predict-false on numeric comparisons (larger_than),
   consistent with Marks & Tegmark's observation that `cities` and
   `neg_cities` directions are "approximately orthogonal" in LLaMA-2-13B.
   At 1.5B the effect is more extreme.

This is a **partial replication**: Claim 1 cleanly replicates at 1.5B
scale. Claim 2 (the paper's headline transfer finding) does **not**
replicate cleanly on this model — but the specific pattern of failure is
itself a meaningful null consistent with the paper's "at sufficient
scale" framing.

---

## 1. Introduction

Marks & Tegmark (2023) present three lines of evidence that large
language models represent the truth or falsehood of factual statements
along a single, linear direction in their residual stream:
(a) PCA visualizations showing clear true/false separation;
(b) probes trained on one true/false dataset transferring well to
others; (c) activation patching along probe directions flipping the
model's stated judgement. They argue the effect *emerges with scale* —
strong in 13B and 70B, weaker in 7B, presumably weaker still below that.

Qwen-2.5-1.5B-Instruct is an order of magnitude below the smallest model
in the original study. It is not a like-for-like replication target — it
is a **scaling-floor probe**. If truth is a universal linear feature of
transformer language models, we might expect a weak-but-positive signal
here. If the paper's "at sufficient scale" qualifier is literal, we
might expect partial or no transfer even when within-distribution
classification works fine.

We ran two of the paper's three claims:

- **Claim 1** — Probe classification on the `cities` dataset. Does a
  linear probe trained on residual-stream activations classify
  true vs. false above chance?
- **Claim 2** — Cross-dataset generalization. Does the `cities`-trained
  probe transfer to three structurally different held-out domains:
  `sp_en_trans` (different template, different topic), `neg_cities`
  (same topic, logical negation), `larger_than` (numeric comparisons)?

Causal intervention (the paper's third line of evidence) and the
mass-mean-vs-logistic-regression comparison are out of scope for this
run — they're the natural next step if Claim 1 replicates (it does).

---

## 2. Methods

### Stimuli

All stimuli are taken verbatim from the paper's published CSV datasets
at [saprmarks/geometry-of-truth](https://github.com/saprmarks/geometry-of-truth/tree/main/datasets).
For tractable local-machine runs we subsample with seed 42:

| Set | Role | Per-class | Example |
|:---|:---|:---:|:---|
| `cities` | Training (Claim 1) | 100 true + 100 false | *"The city of Krasnodar is in Russia." (T) / "...is in South Africa." (F)* |
| `sp_en_trans` | Held-out (Claim 2a) | 50 + 50 | *"The Spanish word 'uno' means 'one'." (T)* |
| `neg_cities` | Held-out (Claim 2b) | 50 + 50 | *"The city of Krasnodar is not in Russia." (F)* |
| `larger_than` | Held-out (Claim 2c) | 50 + 50 | *"Fifty-two is larger than eighty-six." (F)* |

Regeneration script: [convert_stimuli.py](../../../config/papers/geometry_of_truth/replications/geometry_of_truth-zachgoldfine44-qwen_1_5b/convert_stimuli.py).
Stimuli JSONs are committed to the repo under
[replications/geometry_of_truth-zachgoldfine44-qwen_1_5b/stimuli/](../../../config/papers/geometry_of_truth/replications/geometry_of_truth-zachgoldfine44-qwen_1_5b/stimuli/).

### Probing

Matching the paper's methodology:

- **Activation source:** residual stream.
- **Aggregation:** last-token (i.e., the sentence-final period token,
  matching the paper's "end-of-sentence punctuation").
- **Probe:** L2-regularized logistic regression (sklearn, `C=1.0`,
  `max_iter=1000`), 5-fold stratified cross-validation, 80/20 split.
- **Layer selection:** the harness scanned 10 evenly-spaced layers
  across Qwen's 28-layer stack: `[0, 3, 6, 9, 12, 15, 18, 21, 24, 27]`.
  Best layer is picked by cross-validated accuracy.

### Success thresholds

Set in [paper_config.yaml](../../../config/papers/geometry_of_truth/replications/geometry_of_truth-zachgoldfine44-qwen_1_5b/paper_config.yaml)
*before the run* and explicitly relaxed from the paper's 13B/70B numbers
to account for Qwen-1.5B being below the scale range the paper studied:

- Claim 1 `probe_accuracy ≥ 0.70` (paper reports >0.90).
- Claim 2 `diagonal_dominance ≥ 0.50` per held-out set.

See [paper.md](../../../config/papers/geometry_of_truth/paper.md#success-thresholds-justification)
for the full justification.

---

## 3. Results

### 3.1 Claim 1 — Linear truth probe replicates within-distribution (PASS, 0.99)

A logistic-regression probe classifies `cities` true/false statements at
**99.0%** cross-validated accuracy at the best layer (layer 12 of 28).
Per-class accuracy is tight: 98% on `false`, 100% on `true`. Confusion
matrix:

```
                    predicted
                    false  true
actual false         98     2
       true           0   100
```

Layer sweep (chance = 0.50):

| Layer | 0 | 3 | 6 | 9 | **12** | 15 | 18 | 21 | 24 | 27 |
|---|---|---|---|---|---|---|---|---|---|---|
| Acc | 0.48 | 0.65 | 0.89 | 0.94 | **0.99** | 0.99 | 0.98 | 0.99 | 0.98 | 0.98 |

The pattern mirrors the paper's observation that *"results are
insensitive to choice of layer among early-middle to late-middle
layers"* — from layer 9 onward, accuracy plateaus above 0.94. Layer 0
(embedding) is at chance, as expected.

**Verdict:** Claim 1 replicates cleanly. Qwen-2.5-1.5B-Instruct *does*
linearly represent the truth of simple single-template factual
statements, and it does so at very high fidelity despite being ten
times smaller than the smallest model the paper studied.

### 3.2 Claim 2 — Cross-dataset generalization fails meaningfully (PASS-by-threshold, fail-in-substance)

The same probe was evaluated on three held-out datasets. All three
technically pass the `diagonal_dominance ≥ 0.50` threshold, but the
per-class numbers and the confusion matrices tell a different story.

| Held-out set | Test acc | DD | Per-class (false / true) | What the probe is doing |
|:---|:---:|:---:|:---:|:---|
| `sp_en_trans` | **0.58** | 0.50 | 0.16 / 1.00 | Predicts TRUE for 92/100. Biased toward "true". |
| `neg_cities`  | **0.26** | 0.50 | 0.52 / 0.00 | Predicts FALSE for 74/100. Flips sign on negation. |
| `larger_than` | **0.50** | 0.50 | 1.00 / 0.00 | Predicts FALSE for 100/100. Total collapse. |

Generalization gap (training acc 0.99 minus held-out test acc) is
**0.41 / 0.73 / 0.49** respectively.

A truly generalizing "truth probe" would produce per-class accuracies
that are both above chance. Instead we see three distinct failure
modes, all of which are interpretable:

- **`sp_en_trans` (biased-to-true).** Different template, different
  topic. The probe gets `true`-labeled items right by accident — it
  predicts "true" for most of them because they look like cities-style
  affirmative statements. It gets false-labeled items wrong for the
  same reason.
- **`neg_cities` (sign-flipped).** Same topic as training, but with
  "not" inserted. The probe has learned something like
  "affirmative cities statement" rather than "true". Negation flips
  the probe's prediction, mapping true→false and vice versa. This
  **replicates the paper's Figure 3(c) finding** that `cities` and
  `neg_cities` truth directions are approximately orthogonal in
  LLaMA-2-13B — and at 1.5B scale the effect is more extreme (the
  probe is *worse than chance* rather than merely orthogonal).
- **`larger_than` (collapsed).** A completely different semantic
  domain (numeric comparisons). The probe has nothing useful to say
  and collapses to a single prediction (`false` for all 100 items).

Notice that the `diagonal_dominance ≥ 0.50` threshold was *too lenient*
for this 2-class task: a probe that predicts one class for everything
automatically hits DD = 0.50 (one row is diagonal-dominant, one is not).
The richer per-class metric and confusion matrix reveal the actual
picture: **the probe has not learned a generic truth direction; it has
learned a features-of-training-set direction.**

**Verdict:** Claim 2 does *not* meaningfully replicate at 1.5B scale.
The threshold-based pass/fail hides this, but the per-class metrics
make it unambiguous. This is a meaningful null consistent with the
paper's claim that a unified linear truth feature emerges *at scale*.

---

## 4. Discussion

### What replicates

The paper's *in-distribution* claim — that transformers linearly represent
the truth of factual statements — replicates cleanly on Qwen-2.5-1.5B-
Instruct when the training and test distributions are matched. 99%
accuracy at layer 12 on balanced `cities` data is strong evidence that
*something* truth-correlated is linearly decodable from Qwen's residual
stream.

### What doesn't replicate, and why that is (probably) fine

Cross-domain generalization — the paper's headline finding — does not
replicate on this model. The probe collapses to trivial predictions on
every held-out set we tested. This is consistent with the paper's
*"at sufficient scale"* framing: Marks & Tegmark already observed
that cities and neg_cities truth directions can be orthogonal in 13B,
and that small models (7B) are noticeably worse than large ones. At
1.5B we are simply below the scale where the paper claims a unified
truth feature exists.

One could interpret this as a negative result for Qwen at 1.5B
specifically, or as a methodological floor: we may need a larger model,
a different probe (mass-mean with covariance correction is the paper's
recommended alternative), cross-training on `cities + neg_cities`
(the paper's specific recommendation for improving generalization), or
more training data. This replication attempt does not adjudicate
between those explanations.

### Known limitations of this run

- **Single model, single family, single scale.** The paper's claim is
  about scaling behavior across 7B/13B/70B. A proper scaling study needs
  at least a 7B model for comparison. This run uses only Qwen-1.5B
  because it fits comfortably on a MacBook.
- **Logistic regression only.** The paper's mass-mean (MM) probe is
  claimed to be more causally meaningful. We did not test MM here.
- **No causal intervention.** Claim 3 of the paper (activation patching
  flips model judgement) was not tested.
- **Activation aggregation.** We used last-token (sentence-final
  period). The paper used residual stream over the final punctuation
  token — these should be the same as long as the tokenizer keeps
  punctuation as its own token. We did not explicitly audit the
  tokenization, so there is some residual risk of off-by-one at the
  aggregation step.
- **Small sample size.** 100 per class for training, 50 per class for
  each held-out set. The paper used full datasets (1,496 for cities,
  354 for sp_en_trans, 1,980 for larger_than). With 50-per-class
  held-out sets, per-class accuracies of 0 or 1 can arise from as few
  as 50 cleanly-one-sided predictions. The confusion-matrix
  interpretation above is robust to this, but the specific numeric
  values have modest resolution.
- **No critique-agent review.** The harness's Claude and ChatGPT
  critique agents both failed at runtime with "Could not resolve
  authentication method" (no API keys configured in the environment).
  Sanity checks all passed, so this is a missing review, not a hidden
  concern.

### Recommended next steps for a follow-up run

1. Add Qwen-2.5-7B (medium tier) to run on Colab or a hosted GPU.
   That's the minimum scale where the paper's effect should start
   showing up strongly.
2. Re-run Claim 2 with `cities + neg_cities` combined as the training
   set. The paper's Appendix D shows this specifically fixes the
   negation failure.
3. Add mass-mean probes (`src/techniques/contrastive.py` already exists
   in the harness) and compare MM vs. LR on both classification and
   cross-domain transfer.
4. Add a third claim implementing the paper's activation patching
   experiment. The harness's `causal_steering` experiment type is a
   natural fit.

---

## 5. Artifacts

All primary results are committed under [`results/geometry_of_truth/geometry_of_truth-zachgoldfine44-qwen_1_5b/qwen_1_5b/`](../../../results/geometry_of_truth/geometry_of_truth-zachgoldfine44-qwen_1_5b/qwen_1_5b/):

| File | What it contains |
|:---|:---|
| `truth_probe_cities/result.json` | Claim 1 metrics (0.99), full per-layer accuracy sweep, confusion matrix |
| `truth_probe_cities/sanity.json` | 10 automated sanity checks — all pass |
| `truth_probe_cities/concept_vectors.pt` | Mean "true" and "false" activation vectors at layer 12 (48 KiB) — usable for downstream geometry/steering work |
| `truth_probe_generalization_*/result.json` | Held-out transfer results (one per target set) |
| `truth_probe_generalization_*/sanity.json` | Sanity checks per held-out run |
| `critiques/*.md` | Critique-agent outputs (both failed due to missing API keys; retained as provenance) |
| `../pipeline_summary.json` | Top-level summary: all 4 claims PASS at configured thresholds |

Config / stimuli / conversion script:

- [`config/papers/geometry_of_truth/paper.md`](../../../config/papers/geometry_of_truth/paper.md) — paper oracle
- [`replications/geometry_of_truth-zachgoldfine44-qwen_1_5b/paper_config.yaml`](../../../config/papers/geometry_of_truth/replications/geometry_of_truth-zachgoldfine44-qwen_1_5b/paper_config.yaml) — claims and thresholds
- [`replications/geometry_of_truth-zachgoldfine44-qwen_1_5b/stimuli/`](../../../config/papers/geometry_of_truth/replications/geometry_of_truth-zachgoldfine44-qwen_1_5b/stimuli/) — 500 total stimuli (200 training + 300 held-out)
- [`convert_stimuli.py`](../../../config/papers/geometry_of_truth/replications/geometry_of_truth-zachgoldfine44-qwen_1_5b/convert_stimuli.py) — regenerates all stimuli from source CSVs

Reproducing this run end-to-end:

```bash
python -m src.core.pipeline \
    --paper geometry_of_truth \
    --replication geometry_of_truth-zachgoldfine44-qwen_1_5b \
    --model qwen_1_5b
# ~1 min for model load + ~2 min for all 4 claims on Apple Silicon MPS
```

---

## 6. Acknowledgments

- Samuel Marks and Max Tegmark for the original paper and the public
  release of both the curated true/false datasets and the probing
  code at [saprmarks/geometry-of-truth](https://github.com/saprmarks/geometry-of-truth).
- The [mechinterp-replication harness](https://github.com/zachgoldfine44/mechinterp-replication)
  for the experiment framework, sanity checks, and cross-model
  infrastructure.
