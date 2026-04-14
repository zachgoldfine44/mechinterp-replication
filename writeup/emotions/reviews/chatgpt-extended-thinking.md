# ChatGPT (extended thinking) — Peer Review

**Reviewer:** ChatGPT (extended thinking)
**Date:** April 2026
**Target:** [`writeup/emotions/draft.md`](../draft.md) (v3.6)
**Score:** **7 / 10**
**Recommendation:** Reject

---

## Prompts used

> Please read the following paper, and then the following replication of the paper, and rate the replication on a scale of 1-10: Paper: https://transformer-circuits.pub/2026/emotions/index.html / Replication: https://github.com/zachgoldfine44/mechinterp-replication/blob/main/writeup/emotions/draft.md

> Can you turn this into a harsher peer-review-style referee report with major concerns / minor concerns / accept-reject recommendation?

---

## Initial response (7/10)

My score: **7/10**. This is a **good, useful partial replication / cross-model extension**, but not a strong enough methodological match to count as a clean replication of the full Anthropic result.

Why I'm at 7 instead of lower: the design has several things real replications often miss. It tests **multiple model families and scales**, uses **multiple converging checks**, includes **lexical baselines**, **seed robustness**, a **held-out generalization test**, and a genuinely valuable **negative control** that caught numerical-magnitude contamination in the severity analysis. The writeup is appropriately candid that the steering null is **uninformative rather than negative**, and adds a benign-behavior positive control showing the steering pipeline works.

Why I'm not higher: this is really a **cross-model extension with replication elements**, not a close replication. 15 emotions instead of 171, 25 stories instead of ~1,200, last-token activations, simpler vectors, and LLM-generated stimuli with circularity risk. The biggest weakness is the **behavioral test design** — 0% baseline unethical rate means no headroom. The sycophancy pushback result (Qwen-1.5B p=0.036) is **suggestive but not conclusive** with multiple models tested.

If graded strictly as a **replication**: ~6/10. If graded as an **independent exploratory cross-model extension with good controls**: ~8/10. Averaging: **7**.

---

## Referee report

**Recommendation: Reject** in current form.

### Major concerns

**1. Not a faithful replication.** The study departs from the original on 171→15 emotions, ~1,200→25 stories, token-averaged→last-token activations, orthogonalized→mean-difference vectors, multi-turn→single-turn evaluation, and SAE→linear probes. These are fundamental, not cosmetic. The paper should stop calling itself a replication and present as a partial conceptual reproduction under severe methodological simplification.

**2. The behavioral null is overstated.** A 0-vs-0 outcome is not evidence of failed replication; it is evidence that the test bed is uninformative. The framing invites readers to interpret the result as meaningful evidence against generalization, which is not justified.

**3. Headline claim is split between a modest representational result and an unsupported causal conclusion.** The representational results are solid. The causal claims are much weaker. The paper blurs this distinction.

**4. Stimulus circularity.** LLM-generated stimuli make it difficult to distinguish genuine internal representations from learned regularities about emotional narration. The lexical baseline at 40% confirms substantial surface-level discriminability.

**5. Scale gap.** 1B-9B vs. Claude Sonnet 4.5 (>100B) makes cross-model behavioral inference very weak.

**6. Exploratory fixes are commendable but not sufficient.** The negative control, severity-pairs fix, and sentiment positive control improve honesty but underscore how unstable the design still is.

### Minor concerns

1. "Replicates universally" language is not warranted from six instruction-tuned models in a narrow range.
2. Success thresholds appear permissive.
3. Preference-steering evidence is preliminary.
4. Difficult to separate planned vs. post-hoc analyses.
5. Too much value in discussion rather than decisive experiments.

### What would be needed

Reframe as partial cross-model reproduction. Add a behavioral assay with nonzero baseline variance. Strengthen evidence that probe results are not stimulus artifacts.
