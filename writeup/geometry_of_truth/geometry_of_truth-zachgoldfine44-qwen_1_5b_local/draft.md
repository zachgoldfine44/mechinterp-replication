# Replication Attempt: *The Geometry of Truth* on Qwen-2.5-1.5B-Instruct

**Paper:** Marks & Tegmark (2023), *The Geometry of Truth: Emergent Linear Structure in Large Language Model Representations of True/False Datasets.* [arXiv:2310.06824](https://arxiv.org/abs/2310.06824)

**Replicator:** [@zachgoldfine44](https://github.com/zachgoldfine44)
**Replication ID:** `geometry_of_truth-zachgoldfine44-qwen_1_5b_local`
**Model replicated on:** Qwen-2.5-1.5B-Instruct (28 layers, 1,536 hidden dim)
**Model in original paper:** LLaMA-2 7B / 13B / 70B (base)
**Environment:** Apple Silicon MacBook, MPS, local only (`python 3.9.6`, `torch 2.8.0`, `transformer-lens 2.18.0`)

> This is the third Geometry of Truth replication in the repo. It is also a
> fresh rerun of the earlier Zach Qwen-1.5B attempt in a clean
> per-replication namespace after fixing a dependency-path bug in the
> harness, so all downstream claims in this writeup are guaranteed to use the
> probes produced inside this namespace.

## TL;DR

1. A linear truth probe still replicates very strongly on `cities`: **98.5%**
   cross-validated accuracy, with a best layer at **18/28** and a broad
   high-accuracy plateau from layers 12 through 27.
2. Cross-dataset generalization is **partial**, not absent. Transfer to
   `sp_en_trans` is real (**78%** accuracy, **DD = 1.0**), `neg_cities`
   fails outright (**43%**, **DD = 0.0**), and `larger_than` only
   nominally passes the configured threshold (**53%**, **DD = 0.5**) while
   behaving like an almost-all-`false` classifier.
3. The cleanest interpretation is that Qwen-1.5B has some linearly decodable
   truth structure for affirmative factual statements, but not a robust,
   unified truth direction that survives negation or domain shift.

This is a **partial replication**. Claim 1 clearly replicates. Claim 2
partially replicates in substance: one held-out set transfers, one fails
cleanly, and one remains borderline enough that it should not be counted as
strong positive evidence.

## 1. Introduction

Marks & Tegmark (2023) argue that large language models represent the
truth or falsehood of factual statements along an approximately linear
direction in residual-stream space. Their evidence comes from:

1. high probe accuracy within single true/false datasets,
2. transfer of those probes to structurally different held-out datasets,
3. causal interventions that shift the model's stated judgement.

Qwen-2.5-1.5B-Instruct is far smaller than the models in the original paper,
so this is a scaling-floor replication rather than a like-for-like one. The
interesting question is not whether it exactly matches the paper's strongest
70B findings, but whether any of the same geometry is already visible at
1.5B scale.

This run tests two claims:

- **Claim 1:** a logistic-regression probe trained on `cities` residual-stream
  activations can classify true vs. false above chance.
- **Claim 2:** the `cities` probe generalizes to three held-out datasets:
  `sp_en_trans`, `neg_cities`, and `larger_than`.

I left causal intervention out of scope for this run. The harness has the
right primitives, but the result worth locking down first was whether the
fresh, namespaced rerun changed the representational story. It did.

## 2. Methods

### Stimuli

All stimuli come from the paper's published datasets at
[saprmarks/geometry-of-truth](https://github.com/saprmarks/geometry-of-truth/tree/main/datasets).
For a tractable local run, I used the same seed-42 subsampling scheme as the
earlier Zach attempt:

| Set | Role | Per class |
|:---|:---|:---:|
| `cities` | Probe training | 100 true + 100 false |
| `sp_en_trans` | Held-out transfer | 50 + 50 |
| `neg_cities` | Held-out transfer | 50 + 50 |
| `larger_than` | Held-out transfer | 50 + 50 |

Artifacts:

- [convert_stimuli.py](../../../config/papers/geometry_of_truth/replications/geometry_of_truth-zachgoldfine44-qwen_1_5b_local/convert_stimuli.py)
- [stimuli/](../../../config/papers/geometry_of_truth/replications/geometry_of_truth-zachgoldfine44-qwen_1_5b_local/stimuli/)

### Probing setup

- Activation source: residual stream
- Aggregation: last token / sentence-final punctuation
- Probe: sklearn logistic regression (`C=1.0`, `max_iter=1000`)
- Validation: 5-fold stratified CV, seed 42
- Layer sweep: 10 evenly spaced layers across the 28-layer stack

The configured thresholds were intentionally relaxed relative to the paper's
13B/70B numbers:

- Claim 1: `probe_accuracy >= 0.70`
- Claim 2: `diagonal_dominance >= 0.50`

Those thresholds are permissive enough for a 1.5B scaling-floor run, but
they are also loose enough that confusion matrices and sanity reports matter
more than the raw PASS/FAIL bit.

## 3. Results

### 3.1 Claim 1 — Probe classification on `cities` (PASS)

The `cities` probe again replicates strongly:

- Best cross-validated accuracy: **0.985**
- Best layer: **18**
- Per-class accuracy: **0.98** (`false`), **0.99** (`true`)

Confusion matrix:

```text
                    predicted
                    false  true
actual false         98     2
       true           1    99
```

Layer sweep:

| Layer | 0 | 3 | 6 | 9 | 12 | 15 | **18** | 21 | 24 | 27 |
|---|---|---|---|---|---|---|---|---|---|---|
| Acc | 0.50 | 0.68 | 0.885 | 0.94 | 0.985 | 0.98 | **0.985** | 0.985 | 0.985 | 0.985 |

This is the same qualitative picture as the older run: a near-chance
embedding layer, then rapid emergence of a highly linearly separable signal
by the middle of the network.

### 3.2 Claim 2 — Cross-dataset generalization (mixed)

The held-out transfer story changed materially in this clean rerun.

| Held-out set | Accuracy | DD | Per-class acc (false / true) | Verdict |
|:---|:---:|:---:|:---:|:---|
| `sp_en_trans` | **0.78** | **1.0** | 0.56 / 1.00 | Real positive transfer |
| `neg_cities` | **0.43** | **0.0** | 0.38 / 0.48 | Clear failure |
| `larger_than` | **0.53** | **0.5** | 1.00 / 0.06 | Threshold pass, substantive fail |

Confusion matrices:

`sp_en_trans`

```text
                    predicted
                    false  true
actual false         28    22
       true           0    50
```

`neg_cities`

```text
                    predicted
                    false  true
actual false         19    31
       true          26    24
```

`larger_than`

```text
                    predicted
                    false  true
actual false         50     0
       true          47     3
```

Interpretation:

- **`sp_en_trans`** is the main positive result. The probe is not just
  memorizing the exact `cities` template; it carries over to a different
  topic and sentence form. The false row is still weak, but both rows are
  diagonal-dominant and the overall accuracy is meaningfully above chance.
- **`neg_cities`** remains the sharpest failure. Adding negation is enough to
  break the direction completely. This lines up with the paper's broader
  point that `cities` and `neg_cities` do not share one clean truth axis at
  smaller scales.
- **`larger_than`** demonstrates why the configured DD threshold is too weak
  on a binary task. The harness marks this as a PASS, but the model is
  effectively predicting `false` for everything. The sanity check warning on
  near-chance test accuracy is correct.

## 4. Discussion

### What replicated

The central in-distribution claim replicated again, cleanly. Qwen-1.5B
contains a highly linearly decodable truth-correlated feature for simple
affirmative factual statements.

The more interesting update is that the transfer story is not uniformly
negative. In this clean rerun, `sp_en_trans` does generalize. That suggests
some part of the paper's headline claim is already visible below the 7B
scale floor of the original study.

### What did not replicate

The truth direction is still not robust enough to survive every structural
change:

- negation breaks it,
- arithmetic nearly collapses it,
- the held-out true/false balance is still handled asymmetrically.

So the best reading is **partial emergence** rather than a full small-model
replication of the paper's unified-truth-direction claim.

### Why this rerun is worth keeping separately

The earlier Zach row for this same model reported much weaker transfer on
`sp_en_trans`. This rerun was performed after fixing a harness bug where
dependent experiments could look outside the current replication namespace
for upstream probe results. That makes this run the cleaner measurement for
the exact same model-and-stimuli setup.

### Limitations

- Single model, single scale
- Logistic-regression probe only
- No mass-mean comparison
- No causal intervention
- Small held-out sets (50 per class)
- No external AI review artifacts were added because no API keys were
  configured in the environment

### Next steps

1. Run the same config on Qwen-2.5-7B to test whether the mixed transfer
   pattern sharpens into the paper's stronger large-model result.
2. Train on `cities + neg_cities`, which the paper itself recommends as a
   fix for the negation failure mode.
3. Add a causal intervention claim once the preferred steering metric is
   pinned down for binary truth judgements.

## 5. Artifacts

Committed result artifacts for this replication live under
[results/geometry_of_truth/geometry_of_truth-zachgoldfine44-qwen_1_5b_local/](../../../results/geometry_of_truth/geometry_of_truth-zachgoldfine44-qwen_1_5b_local/):

| File | Contents |
|:---|:---|
| `qwen_1_5b/truth_probe_cities/result.json` | Probe metrics, best layer, layer sweep, confusion matrix |
| `qwen_1_5b/truth_probe_cities/sanity.json` | Sanity report for Claim 1 |
| `qwen_1_5b/truth_probe_generalization_*/result.json` | Held-out transfer metrics |
| `qwen_1_5b/truth_probe_generalization_*/sanity.json` | Sanity reports for held-out runs |
| `pipeline_summary.json` | Top-level summary (3/4 configured claims pass) |

Heavy local artifacts such as probe fold files, concept vectors, and
per-stimulus activation caches remain under `local_data/` and are not part of
the PR.

Reproduce this run:

```bash
python -m src.core.pipeline \
  --paper geometry_of_truth \
  --replication geometry_of_truth-zachgoldfine44-qwen_1_5b_local \
  --model qwen_1_5b
```

## 6. Acknowledgments

- Samuel Marks and Max Tegmark for the original paper and the public
  dataset/code release
- The [mechinterp-replication harness](https://github.com/zachgoldfine44/mechinterp-replication)
  for the experiment framework and sanity checks
