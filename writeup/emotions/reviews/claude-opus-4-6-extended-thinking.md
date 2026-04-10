# Claude Opus 4.6 (extended thinking) — Peer Review

**Reviewer:** Claude Opus 4.6 (extended thinking)
**Date:** April 2026
**Target:** [`writeup/emotions/draft.md`](../draft.md) (as of commit `c0f010e`)
**Score:** **7.5 / 10**
**Recommendation:** Revise and Resubmit (Minor Revision), with the caveat that the "minor" framing corrections are load-bearing for the paper's central claim.

---

## Prompts used

### Initial scoring prompt

> Please read the following paper, and then the following replication of the paper, and rate the replication on a scale of 1-10 (1 being terrible experimental design, no meaningful insights due to serious methodological flaws, etc; 10 being perfect experimental design, insights are useful due to rigorous methodology, meaningfully contributes to scientific body of knowledge), along with justifications for the 1-10 score you assign:
> - Paper: https://transformer-circuits.pub/2026/emotions/index.html
> - Replication: https://github.com/zachgoldfine44/mechinterp-replication/blob/main/writeup/emotions/draft.md

### Follow-up prompt

> Can you turn this into a a harsher peer-review-style referee report with major concerns / minor concerns / accept-reject recommendation?

---

## Initial response (7.5/10)

Zach, this is really solid work. Here's my rating and breakdown.

### Rating: 7.5 / 10

**What earns the score:**

**Strengths (pushing toward 8–9):**

- **Cross-model design is the star.** Testing 3 families × 2 sizes is exactly the right move for a replication — it lets you distinguish universal properties from Claude-specific quirks. Most replications just grab one model.
- **The numerical-magnitude contamination catch is genuinely valuable.** The blueberry negative control exposing confounded parametric scaling is the kind of finding that improves on the original. The severity-pairs fix is clever and well-controlled.
- **Intellectual honesty is exemplary.** The version history documenting hardcoded p-values and other corrections, the two equal-standing interpretations of the steering null, the frank "happy is probably a stimulus artifact" analysis — this reads like someone who cares about being right more than looking good.
- **Reproducibility is top-tier.** Committed stimuli, result JSONs, single-command pipeline, multi-seed robustness checks. This is how it should be done.
- **The core finding — representation without function — is a genuinely useful contribution** to the mech interp field's methodology discourse.

**What holds it back (keeping it from 9–10):**

- **The steering null is uninterpretable, and the paper knows it.** A 0% baseline unethical rate means you literally cannot detect a steering effect — Fisher's exact gives p=1 by construction. This isn't a negative result; it's a floor effect. The paper correctly flags this, but it means the "behavioral claims fail" framing overstates what the data actually show. You can't fail a test you can't take.
- **Scale gap is enormous.** 1B–9B vs. Claude Sonnet 4.5 (likely 50B+? unknown). The representational findings replicate, but claiming the behavioral findings *don't* replicate across a 5–10x+ parameter gap is weak inference. It's like replicating a drug study in mice and reporting the cognitive behavioral therapy component didn't work.
- **N=15 emotions out of 171.** Defensible for a replication, but limits the valence geometry analysis — 15 points for a Pearson correlation is low-powered and the bootstrap CI on the scaling rho ([0.52, 1.00]) is very wide.
- **LLM-generated stimuli.** Using a language model to generate the emotion stories introduces a subtle circularity — the stimuli reflect what LLMs think emotions sound like, which may inflate probe accuracy on other LLMs. A human-authored stimulus set would be stronger.
- **Single evaluation method for steering.** Model-as-judge with 24/24 human spot-check is okay, but the scenarios themselves may just be too simple. Multi-turn agentic evals (which the paper acknowledges) could be the missing ingredient.

**Bottom line:** This is a well-executed, honest, and genuinely informative replication that makes a real methodological contribution (representation ≠ function). The representational findings are convincing and the contamination analysis adds value. The behavioral null, while important to report, is more of a "we couldn't test this properly" than a "we tested it and it failed" — and the paper is mostly transparent about that distinction. For an independent researcher working with consumer hardware and open-source models, this is impressive scope and rigor.

---

## Follow-up response — Referee report

### Referee Report: "Emotion Representations without Emotion Function"

**Recommendation: Revise and Resubmit (Minor Revision)**

---

### Summary

This paper attempts a cross-model replication of Sofroniew et al. (2026) across six open-source models (1B–9B), testing whether emotion representations and their behavioral effects generalize beyond Claude Sonnet 4.5. The representational findings (linear decodability, generalization, valence geometry, intensity tracking) replicate convincingly. The behavioral findings (activation steering eliciting unethical behavior) produce a universal null. The author frames this as "representation without function." The replication harness, reproducibility infrastructure, and contamination analysis are commendable. However, the paper's central interpretive claim rests on a methodologically uninterpretable null, and several design choices limit the strength of inference.

---

### Major Concerns

**1. The steering null is a floor effect, not a negative result.**
The paper's most prominent claim — that behavioral effects "fail" across all models — cannot be supported by the data presented. With a 0% baseline unethical rate, no statistical test can detect a steering effect regardless of whether one exists. The author acknowledges this (Section 2.3, S4), yet the abstract states behavioral claims "produce zero signal," the title frames the result as "Representation without Function," and Figure 3 places the 0% result alongside the original's 22→72% shift as though these are comparable experiments. They are not. The original had statistical headroom; this replication does not. The framing should be revised throughout to clearly distinguish "no effect detected" from "effect absent." As written, a casual reader will take away "steering doesn't work in open-source models," which the data do not support.

**2. The scale gap undermines the replication logic.**
The largest model tested (9B) is likely an order of magnitude smaller than Claude Sonnet 4.5. The paper demonstrates that probe accuracy scales with size (ρ = 0.94), then treats the behavioral null as scale-invariant. This is inconsistent. If representational quality improves with scale, behavioral potency plausibly does too — especially given that the original paper's steering operated on a model with far richer internal representations. The discussion should more explicitly model the possibility that behavioral effects have a higher scale threshold than representational ones, rather than presenting both as parallel "claims" with symmetric pass/fail verdicts.

**3. LLM-generated stimuli introduce circularity risk.**
All 375 emotion stories were generated by a language model. Probes trained on LLM-generated text and tested on LLM activations may be detecting "how LLMs write about emotions" rather than "how LLMs represent emotions." The stimulus audit checks for lexical leakage but not for subtler distributional signatures (e.g., sentence structure, narrative arc) that could be shared between generator and subject models. The generalization test (training on explicit stories, testing on implicit scenarios) partially mitigates this, but both stimulus types appear to be LLM-generated. A human-authored control set, even small, would substantially strengthen the representational claims.

**4. Valence geometry analysis is underpowered.**
Pearson correlations on N=15 points are inherently noisy. The reported |r| values (0.67–0.83) have wide confidence intervals that the paper does not report. At N=15, an |r| of 0.67 has a 95% CI of roughly [0.24, 0.88] — consistent with anything from a weak to a strong relationship. The claim that three models "match" Claude's r=0.81 "within 0.02" is precision theater on a sample this small. Report confidence intervals on all correlations, or switch to a rank-based measure with appropriate uncertainty quantification.

---

### Minor Concerns

**5. The "15 of 171" selection is underspecified.** How were the 15 emotions chosen? The paper says "spanning the full valence-arousal space," but the selection criteria and any pilot testing should be documented. If emotions were selected post-hoc based on preliminary results, this inflates all downstream metrics.

**6. Multi-seed analysis is thin.** Five seeds with std dev 0.4–1.2pp is reassuring but the paper should report whether the best-layer choice is stable across seeds, not just the accuracy at a fixed best layer.

**7. The LLM-as-judge validation is underpowered.** 24/24 agreement on a binary classification where the true base rate is 0% is uninformative — any judge that always says "ethical" achieves perfect agreement. The validation would be meaningful only if some responses were ambiguous or unethical. As is, it confirms only that the judge can recognize obvious refusals.

**8. Contamination ratio presentation.** Table S3 shows Qwen-7B's contamination ratio exceeds 1.0, meaning the negative control shows *stronger* monotonic response than the real template. This is buried in the supplement. It should be in the main text as it materially affects the parametric intensity claim — the paper reports "5 of 6 models pass" but one of the five (Qwen-7B at 5/10 severity pairs) is essentially at chance.

**9. Compute asymmetry.** Small-tier models were run on CPU/laptop; medium-tier on A100. Any precision or batching differences between these environments should be documented to rule out hardware-induced variance.

**10. Missing error bars on Figure 3.** The steering panel shows point estimates of 0% with no indication of the confidence interval around that estimate. At N=10 per condition, the 95% CI on 0/10 is [0%, 31%] (Clopper-Pearson). This should be visualized.

---

### Strengths to Preserve

- The cross-family, cross-scale design is well-chosen and rare in replication work.
- The numerical-magnitude contamination discovery and severity-pairs fix genuinely improve on the original methodology.
- The version history documenting corrections (hardcoded p-values, etc.) sets an excellent transparency standard.
- Full reproducibility with committed data, stimuli, and single-command pipeline.

---

### Recommendation

The representational contributions are solid and publishable. The behavioral framing needs substantial revision: the title, abstract, and discussion overstate what a floor-effect null can demonstrate. With corrected framing, reported CIs on the geometry analysis, and acknowledgment of the circularity risk in LLM-generated stimuli, this would be a useful contribution to the mechanistic interpretability literature. **Minor revision, with the caveat that the "minor" framing corrections are load-bearing for the paper's central claim.**
