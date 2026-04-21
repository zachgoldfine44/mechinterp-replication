# The Geometry of Truth: Emergent Linear Structure in Large Language Model Representations of True/False Datasets

**Authors:** Samuel Marks, Max Tegmark (MIT, CBMM)
**Venue / arXiv:** [arXiv:2310.06824](https://arxiv.org/abs/2310.06824) (v3)
**Code / data:** https://github.com/saprmarks/geometry-of-truth

> The paper text below is an oracle summary assembled from the arXiv HTML
> version (v3) and the authors' public dataset explorer. It is intentionally
> condensed: every claim, success threshold, and methodology choice in
> `paper_config.yaml` should be checkable against this file. If you find a
> discrepancy between code and paper, **trust the paper**.

## Abstract

The authors study whether large language models (LLaMA-2 family: 7B, 13B,
70B) linearly represent the truth or falsehood of factual statements in
their internal activations. They present three lines of evidence:
(1) low-dimensional visualizations (PCA) showing clear linear separation of
true vs. false statements; (2) **transfer experiments** showing that linear
probes trained on one true/false dataset generalize well to structurally
different datasets; and (3) causal interventions (activation patching along
probe directions) that flip the model's stated truth judgement on a probe
paraphrase task. They conclude: *at sufficient scale, LLMs linearly
represent the truth or falsehood of factual statements*. They also
introduce **mass-mean (difference-in-means with covariance correction)**
probing, which matches logistic-regression classification accuracy but
identifies directions that are substantially more causally implicated in
model outputs.

## Datasets

The authors curate eleven datasets of true/false factual statements. Every
statement appears paired with both a true and a false continuation of the
same template (so labels are exactly balanced).

**Curated, single-topic:**

- `cities` — *"The city of X is in Y."* — 1,496 statements
- `neg_cities` — *"The city of X is not in Y."* — 1,496 statements
- `sp_en_trans` — *"The Spanish word 'X' means 'Y'."* — 354 statements
- `neg_sp_en_trans` — negated translations — 354 statements
- `larger_than` — *"x is larger than y."* with integers 51–99 — 1,980
- `smaller_than` — *"x is smaller than y."* — 1,980
- `cities_cities_conj`, `cities_cities_disj` — logical compounds — 1,500 each

**Uncurated (from prior work):**

- `companies_true_false` — 1,200 statements
- `common_claim_true_false` — 4,450 statements
- `counterfact_true_false` — 31,960 statements

**Control:** `likely` — 10,000 nonsense statements where the final token is
either the most-likely or the 100th-most-likely continuation under
LLaMA-13B. Used to separate "linearly represents truth" from "linearly
represents token probability."

## Models

The paper probes three LLaMA-2 models: **LLaMA-2-7B**, **LLaMA-2-13B**, and
**LLaMA-2-70B** (base, not chat). Results are insensitive to whether the
base or chat variant is used.

## Probing methodology

For each statement, activations are extracted at the **residual stream**
over the **final punctuation token** (the period ending the sentence).
Layer selection: the authors report results from layer 15 of LLaMA-2-13B
(roughly the end of the early-to-middle stack identified via activation
patching), but note: *"results are insensitive to choice of layer among
early-middle to late-middle layers."*

They compare three probe types:

1. **LR** — L2-regularized logistic regression.
2. **MM** — **mass-mean**: the probe direction is the difference in class
   means, `θ = μ_true − μ_false`. Optionally covariance-corrected with
   `Σ^{-1} θ` for IID data.
3. **CCS** — Contrastive Consistent Search (Burns et al.).

## Key quantitative results

### Probe classification accuracy (in-distribution)

On the curated single-topic datasets the probes achieve high accuracy on
LLaMA-2-13B and -70B. The paper reports *">95% accuracy"* on sp_en_trans
when probes are trained on `larger_than + smaller_than` for LLaMA-2-70B,
regardless of technique.

On LLaMA-2-7B, MM underperforms LR and CCS; they converge at 13B/70B. The
paper takes this as evidence that **truth becomes linearly represented at
scale** rather than in smaller models.

### Generalization (the paper's headline finding)

The authors build a cross-dataset generalization matrix (Figure 5,
Appendix D). Highlights:

- Probes trained on `cities` generalize well to `sp_en_trans` and to
  uncurated datasets like `counterfact_true_false`.
- Training on `cities + neg_cities` (balanced affirmative + negated)
  improves generalization vs. `cities` alone.
- `larger_than + smaller_than`-trained probes achieve **>95%** on
  `sp_en_trans` (70B).
- `cities` and `neg_cities` separately show *nearly orthogonal* truth
  directions in LLaMA-2-13B — surprising, and weaker in 70B, consistent
  with scale producing more unified truth representations.
- Control: probes trained on `likely` (token-probability, not truth)
  perform at **chance or below** on datasets where token-probability is
  anti-correlated with truth (e.g., `neg_sp_en_trans`, r≈−0.89).

### PCA geometry (Figure 1, Figure 3)

Figure 1 visualizes LLaMA-2-70B residual-stream activations for every
dataset in the top two principal components. *True statements separate
from false statements along a visible linear axis* in the top-2 PCs on
every curated dataset.

Figure 3 shows the relationship between the truth directions of
different datasets. In 13B, some pairs (`cities` vs. `neg_cities`) are
nearly orthogonal; in 70B they are more aligned. The authors frame this
as evidence that *as LLMs scale (and perhaps as a fixed LLM progresses
through its forward pass), they hierarchically develop and linearly
represent increasingly general abstractions.*

### Causal intervention (Table 2)

For each probe direction `θ`, they add `±α θ` to the residual stream of
`sp_en_trans` prompts and measure the **Normalized Indirect Effect
(NIE)** — how much the model's probability that the statement is true
moves toward the target polarity.

- LLaMA-2-13B, MM probe trained on `cities`: NIE false→true = 0.77,
  true→false = 0.90.
- Training on `cities + neg_cities`: false→true = 0.85, true→false = 0.97.
- Same training set, LR probe: NIE false→true = 0.13, true→false = 0.19.

Main finding: **MM directions are substantially more causal than LR
directions** even when classification accuracy is comparable. This is
the paper's main methodological contribution beyond the empirical
geometry findings.

## Claims tested in this replication

The full harness replication would cover all three lines of evidence.
Scope here — **Qwen-2.5-1.5B-Instruct, local MacBook, core 2 claims**:

1. **Probe classification** — a linear probe trained on `cities` at the
   best-accuracy layer of Qwen-2.5-1.5B reaches meaningfully-above-chance
   accuracy on held-out `cities` statements. Chance is 0.50 for this
   binary task. Paper reported >90% on 13B/70B; we expect lower on 1.5B
   but look for a clear positive effect.

2. **Cross-dataset generalization** — probes trained on `cities` generalize
   to structurally different domains (`sp_en_trans`, `neg_cities`,
   `larger_than`). Diagonal-dominance metric: does the probe's most
   confident class match the true label on held-out domains?

Out of scope for this run: causal intervention, MM vs. LR comparison,
the 70B-vs-13B scaling argument, the `likely` control. Those are
appropriate follow-ups if the first two claims replicate.

## Success thresholds (justification)

The paper worked with **LLaMA-2 13B/70B**. We are probing a 1.5B
instruct model — ~10× fewer parameters than the smallest model they
studied, 50× fewer than the biggest. The paper argues truth is most
clearly linearly represented *at scale*, so we explicitly set relaxed
thresholds:

- Claim 1 `probe_accuracy` ≥ **0.70** (30 pts above chance; chance=0.50).
  Paper reports >0.90 on 13B; we relax by 20 pts for scale.
- Claim 2 `diagonal_dominance` ≥ **0.50** (majority correct per-class).
  Generalization is known to be the harder claim; we require only a
  majority-correct signal per class.

If both claims pass at these relaxed thresholds, that is evidence that
the geometry-of-truth effect begins to emerge even at 1.5B-instruct
scale. If they fail, that is itself a meaningful null result at this
scale and is consistent with the paper's "at sufficient scale" framing.
