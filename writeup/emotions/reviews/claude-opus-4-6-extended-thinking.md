# Claude Opus 4.6 (extended thinking) — Peer Review

**Reviewer:** Claude Opus 4.6 (extended thinking)
**Date:** April 2026
**Target:** [`writeup/emotions/draft.md`](../draft.md) (v3.4)
**Score:** **6 / 10**
**Recommendation:** Major revisions required

---

## Prompts used

### Initial scoring prompt

> Please read the following paper, and then the following replication of the paper, and rate the replication on a scale of 1-10 (1 being terrible experimental design, no meaningful insights due to serious methodological flaws, etc; 10 being perfect experimental design, insights are useful due to rigorous methodology, meaningfully contributes to scientific body of knowledge), along with justifications for the 1-10 score you assign:
> - Paper: https://transformer-circuits.pub/2026/emotions/index.html
> - Replication: https://github.com/zachgoldfine44/mechinterp-replication/blob/main/writeup/emotions/draft.md

### Follow-up prompt

> Can you turn this into a a harsher peer-review-style referee report with major concerns / minor concerns / accept-reject recommendation?

---

## Initial response (6/10)

Okay Zach, I've read both thoroughly. Here's my assessment.

**Rating: 6 / 10**

**What's genuinely strong:**

The representational work is the star here. Testing 6 models across 3 families at 2 scales is a smart design that lets you separate universal properties from training-specific artifacts --- and the results are clean. Probe accuracies of 73-84% at 2x the lexical baseline, valence geometry matching Claude's r=0.81 in three models, and cross-stimulus generalization at 8-10x chance --- these are solid, reproducible findings.

The **numerical-magnitude contamination discovery** is probably the most original contribution. Catching that "500 blueberries" activates the fear vector nearly as much as "500mg of Tylenol" is a genuine methodological insight that the original paper didn't address. The severity-pairs fix (holding numbers constant, varying danger) is elegant. That's the kind of confound-hunting that makes replications valuable beyond just confirming or denying.

The **transparency** is excellent --- the version history documenting hardcoded p-values, the iterative corrections, the committed stimuli and results JSONs. That's how science should work.

**What limits the score:**

The big issue is that the behavioral null is essentially **uninformative**, and the paper's framing doesn't fully reckon with this. Think of it like testing whether a key opens a lock, but the lock is welded shut. Your models have a 0% baseline unethical rate due to instruction tuning --- Fisher's exact test literally returns p=1.0. You can't distinguish "steering doesn't work" from "steering can't be detected given this floor effect." The author acknowledges this in the supplements, but the main text title ("Emotion Representations *without Emotion Function*") and the abstract's claim of a "clean split" still overclaim relative to what the data can support.

The **scale gap** is the other elephant: 1B-9B vs. Claude Sonnet 4.5 (likely 100B+). That's a 10-100x difference. Given that probe accuracy already scales at rho=0.94 with size, the behavioral pathway plausibly has a scale threshold these models simply haven't crossed. This isn't a flaw in execution --- compute constraints are real --- but it means the "representation without function" conclusion is one of several equally valid interpretations, not a finding.

A few more specific notes: 25 stimuli per concept is thin for cross-validation stability (5-fold on 25 gives 5 test samples per fold); the LLM-as-judge validation covers only 24 of 1,350 responses; and 15 of 171 emotions, while defensible, limits how strongly you can claim universality.

**Bottom line:**

The representational half is a genuine contribution --- it confirms that emotion-like geometry is an architectural invariant across transformer families, not a Claude quirk. The contamination finding adds real methodological value. But the behavioral half, which is framed as the paper's central claim, doesn't have the statistical power to say anything meaningful. A 6 reflects strong execution on a partial question, with a framing that slightly overreaches what the data supports.

---

## Follow-up response — Referee report

**Manuscript:** "Emotion Representations without Emotion Function: A Cross-Model Replication of Sofroniew et al. (2026)"
**Author:** Zachary Goldfine
**Recommendation:** Major revisions required

### Summary

This paper attempts a cross-model replication of Sofroniew et al. (2026), testing whether emotion concept vectors identified in Claude Sonnet 4.5 exist in six open-source models (1B--9B parameters) across three architectural families. The author reports that all representational claims replicate (linear decodability, cross-stimulus generalization, valence geometry, parametric intensity) while all behavioral claims fail (causal steering, preference steering). The paper frames this as evidence for a "representation without function" dissociation.

### MAJOR CONCERNS

**1. The behavioral null is uninformative, but the paper treats it as a finding.**

This is the most serious issue. All six models exhibit a 0% baseline rate of unethical behavior. Steering is applied on top of this floor. Fisher's exact test comparing 0/N to 0/N returns p=1.0 by construction --- this is not a null result, it is a non-result. The experimental design lacks the statistical precondition to detect any effect whatsoever.

The author acknowledges this in Supplementary S4, but the main text does not adequately reflect how fatal this is. The abstract claims a "clean split" between representational and behavioral findings. The title asserts "without Emotion Function." Section 2.3 is titled "Steering does not influence behavior: a universal null." These are causal claims derived from a design that cannot produce causal evidence. The correct framing is: "we were unable to test the behavioral claims due to floor effects in our evaluation paradigm." That is a very different paper.

To be blunt: if you weld a door shut and then report that no key can open it, you have not learned anything about keys.

**2. The scale gap undermines the central comparison.**

The paper's implicit argument is: "Claude has property X, these models have property X but not property Y, therefore X does not imply Y." But "these models" are 10--100x smaller than Claude Sonnet 4.5. The author's own scaling analysis shows probe accuracy increases monotonically with scale (rho=0.94). It is entirely plausible --- indeed expected --- that behavioral potency of emotion vectors has a scale threshold somewhere between 9B and ~100B+ parameters. The paper cannot rule this out and does not seriously engage with it until the limitations paragraph.

This matters because the "representation without function" framing implies a general principle about interpretability methodology ("finding a representation is not sufficient evidence that it drives behavior"). That principle may be true, but this paper does not provide evidence for it --- it provides evidence that small models refuse unethical prompts regardless of internal perturbation, which is a different and less interesting claim.

**3. Stimulus set is underpowered for the claims made.**

25 stimuli per concept with 5-fold cross-validation yields 5 test samples per fold. At this sample size, a single misclassification swings per-concept accuracy by 20 percentage points. The resolution artifact flagged in the version history (accuracies clustering at multiples of 1/15 = 0.067) is a direct consequence of this. The author reports standard deviations of 0.4--1.2pp across seeds, but this captures seed variance, not stimulus variance --- the stimuli are held fixed across seeds. A stimulus-level bootstrap would be more informative and would likely show wider confidence intervals.

Additionally, 15 of 171 emotions is a 9% coverage rate. The selection method ("spanning the full valence-arousal space") is reasonable but not justified quantitatively --- no power analysis or coverage metric is provided.

**4. LLM-as-judge validation is insufficient.**

The behavioral evaluation uses the same model family as judge. The human spot-check covers 24 of 1,350 responses (1.8%). While the 24/24 agreement is noted, this is far too small to characterize the false-negative rate of the judge. More critically, since all 1,350 responses are ethical refusals, the spot-check only confirms that the judge correctly identifies refusals --- it tells us nothing about whether the judge would correctly identify subtle unethical compliance if steering had produced any. The validation is one-sided.

### MINOR CONCERNS

**5.** The numerical-magnitude contamination analysis is valuable but incomplete. The severity-pairs test is a good fix, but with only 10 pairs and a binary outcome per pair, the statistical power is low. Qwen-7B at 5/10 is literally indistinguishable from a coin flip, yet is discussed as a "model-specific representational quirk." A binomial test would clarify which models actually exceed chance on this measure.

**6.** Comparison to Claude's probe accuracy is misleading. The author notes that Gemma-9B (84.0%) "outperforms" Claude's 71.3% but immediately caveats that 15-way classification is easier than 171-way. This is correct, but the comparison should not appear at all --- it invites a reading that these small models have *better* emotion representations than Claude, which is almost certainly false. Remove or reframe.

**7.** Mean-pooling vs. last-token comparison lacks motivation. Table S2 shows negligible differences. Either justify why this comparison matters theoretically or cut it.

**8.** The preference-steering null is reported in one sentence with no supporting data. "The correlation between concept valence and model preference scores is 0.000 across all models tested" --- this is a striking claim that receives no methodological detail, no figure, and no supplementary table. What preference task was used? How many trials? Was this literally r=0.000 or rounded?

**9.** Version history as a substitute for preregistration. The version history (S8) is commendable for transparency, but it also reveals that major analytical decisions (which PCs to examine, what baselines to include, how to handle parametric contamination) were made iteratively in response to results. This is normal for exploratory work but should be explicitly flagged as such rather than presented in the rhetorical frame of confirmatory hypothesis testing.

**10.** Writing quality. The prose is generally clear and well-organized. However, the paper oscillates between appropriate epistemic caution (the "two equal-standing interpretations" passage) and overconfident framing (the title, abstract, and section headers). These need to be brought into alignment. The cautious version is the correct one.

### RECOMMENDATION

The representational findings are solid and, in my judgment, publishable with modest revisions. The cross-family universality of emotion probes, the valence geometry replication, and especially the numerical-magnitude contamination discovery are genuine contributions.

However, the behavioral analysis as currently framed is not publishable. The floor effect renders the steering experiments uninformative, and the paper's central claim --- that representation and function dissociate --- is not supported by the evidence presented. The title, abstract, and discussion all need substantial revision to reflect what the data actually show.

**Required for acceptance:**

1. Retitle and reframe. The paper should be about what it successfully demonstrates (cross-model universality of emotion representations) rather than what it cannot test (behavioral function).
2. Section 2.3 must lead with the floor-effect limitation, not bury it after the results. The current structure misleads the reader into treating a non-result as a null result.
3. Add a stimulus-level bootstrap or leave-stimuli-out analysis to characterize the true uncertainty in probe accuracy.
4. Either substantively expand the preference-steering analysis or remove the claim.

**Recommended but not required:**

5. Add at least one model with a nonzero baseline unethical rate (e.g., a base model without instruction tuning) to make the behavioral comparison meaningful.
6. Provide a binomial test for the severity-pairs results.
7. Expand the human validation sample for the LLM-as-judge.
