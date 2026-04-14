# Claude Opus 4.6 (extended thinking) — Peer Review

**Reviewer:** Claude Opus 4.6 (extended thinking)
**Date:** April 2026
**Target:** [`writeup/emotions/draft.md`](../draft.md) (v3.6)
**Score:** **7 / 10**
**Recommendation:** Revise and Resubmit (Major Revisions)

---

## Prompts used

> Please read the following paper, and then the following replication of the paper, and rate the replication on a scale of 1-10: Paper: https://transformer-circuits.pub/2026/emotions/index.html / Replication: https://github.com/zachgoldfine44/mechinterp-replication/blob/main/writeup/emotions/draft.md

> Can you turn this into a harsher peer-review-style referee report with major concerns / minor concerns / accept-reject recommendation?

---

## Initial response (7/10)

**What's strong:** Testing across 3 families x 2 sizes is the right design. The numerical-magnitude contamination discovery (blueberries triggering the fear vector) is a genuinely clever contribution. The version history transparency is unusually honest. Full reproducibility exceeds most published papers.

**What limits the score:**

1. **The scale gap undermines the headline claim.** Models are ~10-50x smaller than Claude Sonnet 4.5. It's like testing whether a medication works by giving it to mice and reporting it doesn't work in mice.
2. **Not a "replication" in the strict sense.** Different models, different scale, different tools, different evaluation protocol. Better described as a "cross-model extension."
3. **15 of 171 emotions with 25 stimuli each** — easier 15-way task means probe accuracies aren't directly comparable.
4. **N=6 scaling analysis** — with 6 data points you can't distinguish scaling laws from noise.

The representational findings are solid; the behavioral null is uninformative rather than informative. For a solo independent replication, this is strong work. For a peer-reviewed publication making claims about emotion-behavior links, it needs larger models and nonzero baseline scenarios.

---

## Referee report

**Recommendation: Revise and Resubmit (Major Revisions)**

### Major Concerns

**1. The behavioral test is uninformative, not informative.** Fisher's exact comparing 0/N to 0/N yields p=1.0. This is a non-finding. The title ("without Emotion Function") and abstract treat it as affirmative. The competing hypotheses (protocol-limitation vs. representation-without-function) receive equal rhetorical weight when the data strongly favors protocol-limitation. Requested: Reframe explicitly as uninformative. Revise title.

**2. The scale mismatch invalidates the "replication" framing.** A replication applies the same method to the same system. This applies a different method to a different class of system. That is a transferability study. Requested: Reframe as cross-model generalization study.

**3. Stimulus confounds incompletely addressed.** Five positive-valence concepts in a 15-class task means aggregate accuracy conflates emotion discrimination with valence clustering. The 40% lexical baseline confirms substantial surface discriminability. Requested: Report confusion matrices. Test whether a 3-way valence classifier explains most performance.

**4. LLM-as-judge methodology insufficiently validated.** 24/24 agreement when base rate is 0% tells nothing about false-negative rate. Using the steered model as its own judge creates circularity. Requested: Use external judge or acknowledge circularity as limitation.

### Minor Concerns

5. Severity-pairs sign test p-values should be reported explicitly.
6. Version history reveals iterative analytical decisions — flag as exploratory.
7. Preference-steering null reported without details.
8. Spearman p-value on N=6 is misleading.
9. Mean-pooling interpretation overreaches from one delta on one model.
10. No related work section citing Turner et al. 2023, Rimsky et al. 2024, Zou et al. 2023.

### Summary

The representational findings are solid and useful. The severity-pairs methodology is a real contribution. The behavioral analysis needs reframing. With major revisions to framing and the additions noted above, this could be a solid contribution as a cross-model generalization study.
