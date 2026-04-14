# Gemini 3.1 Pro — Peer Review

**Reviewer:** Gemini 3.1 Pro
**Date:** April 2026
**Target:** [`writeup/emotions/draft.md`](../draft.md) (v3.6)
**Score:** **8.5 / 10**
**Recommendation:** Reject (Encourage Resubmission as Note/Short Paper)

---

## Prompts used

> Please read the following paper, and then the following replication of the paper, and rate the replication on a scale of 1-10: Paper: https://transformer-circuits.pub/2026/emotions/index.html / Replication: https://github.com/zachgoldfine44/mechinterp-replication/blob/main/writeup/emotions/draft.md

> Can you turn this into a harsher peer-review-style referee report with major concerns / minor concerns / accept-reject recommendation?

---

## Initial response (8.5/10)

**Rating: 8.5 / 10.** The replication offers a highly robust, analytical extension of the original Anthropic findings, successfully transitioning the study from a single proprietary model to a cross-model evaluation of six open-source transformers.

**Phase 1 (Representation):** Probes achieved 73.1-84.0% accuracy at roughly double the lexical ceiling.

**Phase 2 (Geometry):** PCA recovered a valence axis with r=0.67-0.83, three models within 0.02 of Claude's r=0.81.

**Phase 3 (Causal Steering):** Revealed a precise "gradient of influence" — sentiment shifts universally but ethical compliance is floor-gated.

**Phase 4 (Sycophancy v2):** Using GPT-5.4-mini to judge 2,520 responses eliminated self-judging circularity. Pushback-capitulation captured Qwen-1.5B (p=0.036) and borderline Gemma-9B (p=0.054).

**Critiques:** Stimulus circularity from LLM-generated stories. Severity-pairs underpowered (1/6 significant). 171→15 emotion compression limits full semantic mapping. The structured v3.4-v3.6 iterations elevate this from standard replication to robust scientific extension.

---

## Referee report

**Recommendation: Reject (Encourage Resubmission as Note/Short Paper)**

### Major Concerns

**1. Circularity in stimulus design (fatal flaw).** 25 LLM-generated stories per emotion likely capture superficial lexical correlates of how models write about emotions, not abstract functional representations. Without human-authored controls or rigorous cross-domain generalization, the probe accuracies are artifactual.

**2. Statistically irresponsible significance claims.** The sycophancy pushback result (Qwen-1.5B p=0.036) was tested across 6 models with no multiple-comparison correction. With Bonferroni, this is p=0.216 — non-significant. The "detectable on some complex behaviors" conclusion must be struck until replicated with N>1,000 and proper family-wise error rate corrections.

**3. Floor effects invalidate ethical steering.** You cannot measure a "gradient of causal influence" when the dependent variable has zero variance. To actually test if emotion vectors can override safety training, use models with higher baseline compliance failure rates or jailbreak-adjacent prompts.

**4. Oversimplification of original methods.** Abandoning neutral-transcript denoising and using last-token rather than position-averaged activations introduces massive noise into steering vectors. These are regressions in interpretability best-practices.

### Minor Concerns

- 5-fold CV on 25 samples = 5 test samples per fold. Highly susceptible to noise.
- Sentiment shifts (+0.01 to +0.53) reported without defined scale or standardized effect size.
- Severity-pairs test likely underpowered (Type II errors, not true negatives).
- No formal justification for which 15 emotions were selected.

### Conclusion

The representational findings are interesting preliminary data. The paper over-promises and under-delivers on causal claims. To achieve acceptance: scale dataset to N>500 per emotion, introduce human-authored controls, correct for multiple comparisons, and resubmit.
