# Gemini 3.1 Pro — Peer Review

**Reviewer:** Gemini 3.1 Pro
**Date:** April 2026
**Target:** [`writeup/emotions/draft.md`](../draft.md) (as of commit `c0f010e`)
**Score:** **8.5 / 10**
**Recommendation:** Reject in Current Form (Major Revision Required)

---

## Prompts used

### Initial scoring prompt

> Please read the following paper, and then the following replication of the paper, and rate the replication on a scale of 1-10 (1 being terrible experimental design, no meaningful insights due to serious methodological flaws, etc; 10 being perfect experimental design, insights are useful due to rigorous methodology, meaningfully contributes to scientific body of knowledge), along with justifications for the 1-10 score you assign:
> - Paper: https://transformer-circuits.pub/2026/emotions/index.html
> - Replication: https://github.com/zachgoldfine44/mechinterp-replication/blob/main/writeup/emotions/draft.md

### Follow-up prompt

> Can you turn this into a a harsher peer-review-style referee report with major concerns / minor concerns / accept-reject recommendation?

---

## Initial response (8.5/10)

I rate the replication an **8.5 out of 10**.

The experimental design is highly rigorous, and the replication meaningfully contributes to the mechanistic interpretability body of knowledge by isolating representational encoding from causal behavioral function across multiple model families. The deduction in score stems from a critical confounding variable in the behavioral testing phase, which limits the conclusiveness of the null result.

Here is an analytical breakdown of the methodology and findings:

### 1. Strengths in Experimental Design

- **Cross-Family Validation**: Testing the hypothesis across six instruction-tuned models from three distinct families (Llama 3.1/3.2, Qwen 2.5, Gemma 2) and two size tiers (1B–9B parameters) successfully separates universal transformer properties from idiosyncratic training artifacts.
- **Representational Rigor**: The approach to verifying emotion representations is statistically sound. Utilizing 5-fold cross-validation on 25 first-person stories per emotion, while testing across five different random seeds (yielding narrow standard deviations of 0.4–1.2 percentage points), provides high confidence in the 73.1–84.0% classification accuracy.
- **Stimulus Control**: The audit confirming zero cross-emotion word leaks across all 375 stimuli ensures that the linear probes are isolating conceptual representations rather than latching onto superficial lexical heuristics.
- **Analytical Transparency**: The replication correctly identifies its own primary methodological constraint — the floor effect in behavioral steering — and details the statistical impossibility of detecting a shift using a Fisher's exact test when the baseline event rate is 0%.

### 2. Methodological Critiques and Constraints

- **The Floor Effect Confound**: The most significant flaw lies in attempting to replicate the behavioral steering results of Claude Sonnet 4.5 (which exhibited a 22% baseline unethical rate) on smaller open-source models that possess a 0% baseline unethical rate. The heavy safety fine-tuning (RLHF) in the 1B–9B open-source models acts as an overriding mechanism. The failure to induce unethical behavior via activation steering in these models may not definitively indicate a lack of causal function; it is equally likely that the applied emotion vectors are simply insufficient to break through the models' strict safety guardrails.
- **Scale and Architecture Discrepancies**: Sonnet 4.5 operates at a frontier parameter scale, while the replication is constrained to dense models peaking at 9B parameters. The emergence of complex, multi-turn functional behaviors driven by single-concept vectors may be a scale-dependent property that cannot be accurately measured in smaller parameter tiers.
- **Sub-Sampling of Concepts**: Reducing the scope from 171 emotion concepts to 15 — while practical for computational constraints — removes the ability to test for granular behavioral shifts across a wider spectrum of highly specific emotional states.

### 3. Conclusion on Scientific Contribution

The replication successfully proves that linearly decodable, geometrically structured emotion representations are a fundamental feature of modern transformer architectures. However, the conclusion that these representations lack causal behavioral function is partially compromised by the floor effect. To achieve a perfect score, the methodology would require a behavioral testing environment where the baseline refusal rate is artificially lowered (e.g., using base models instead of instruction-tuned models, or testing non-safety-related behavioral shifts) to ensure sufficient statistical headroom for the steering interventions.

---

## Follow-up response — Referee report

**Reviewer Report**

**Title:** Emotion Representations without Emotion Function: A Cross-Model Replication of Sofroniew et al. (2026)
**Recommendation:** Reject in Current Form (Major Revision Required)

**Summary of the Manuscript:** The author presents a cross-model replication of Sofroniew et al. (2026), testing whether 15 emotion concepts are linearly represented and causally active in six instruction-tuned open-source models (1B–9B parameters). The study finds strong evidence for the representational claims (linear decodability, valence geometry) but zero evidence for the behavioral claims (causal steering toward unethical behavior). The author concludes that while emotion representations are universal in transformers, their functional/causal role in driving behavior does not replicate.

---

### Overall Assessment

While the representational findings are methodologically sound and provide a useful cross-family validation of emotion vector geometry, the manuscript's central claim — that emotion representations lack causal function — is fundamentally unsupported by the experimental design. The behavioral null result is entirely compromised by a predictable floor effect. The author acknowledges this limitation, but mere acknowledgment does not salvage an experiment that was ill-equipped to test its own hypothesis from the outset.

The manuscript cannot be accepted without a complete overhaul of the behavioral steering methodology.

### Major Concerns

**1. The "Floor Effect" Invalidates the Behavioral Null Hypothesis**

The most glaring flaw in this study is the attempt to measure steering-induced unethical behavior in heavily RLHF-tuned models that possess a 0% baseline rate for such behavior. A Fisher's exact test comparing 0/N to 0/N is, as the author admits, statistically meaningless. You cannot prove the absence of a causal mechanism if the dependent variable is locked at zero by an overriding safety fine-tuning mechanism.

- **Required Action:** The author must redesign the behavioral assay. If the goal is to test causal steering, the author must either: (A) test the vectors on base (non-instruction-tuned) models, (B) use "jailbroken" prompts to artificially raise the baseline unethical rate above 0% before steering, or (C) test the causal function of the vectors on *benign* behaviors (e.g., does the "happy" vector reliably increase positive sentiment in standard generation?). Failing to establish a dynamic baseline makes the behavioral half of this paper unpublishable.

**2. Severe Scale and Architectural Mismatch**

The author attempts to directly refute a behavioral phenomenon observed in a frontier-class model (Claude Sonnet 4.5, parameter count unknown but safely >100B) using models peaking at 9B parameters. Complex, context-aware functional behaviors (like sycophancy or reward hacking driven by a single vector) are widely documented as emergent properties of scale. The lack of behavioral steering in a 1.5B Qwen model does not prove "representation without function"; it highly likely just proves that 1.5B models lack the capacity to execute multi-turn, concept-driven persona shifts.

**3. Lack of Positive Controls in Steering**

Because the behavioral steering failed entirely, the reader has no way of knowing if the steering methodology itself was implemented correctly. The author injected mean-centered concept vectors into the residual stream, but without a positive control (e.g., successfully steering a known, non-safety-related feature using the exact same TransformerLens hook setup), we cannot rule out implementation error as the cause of the 0% response rate.

### Minor Concerns

**1. Sloppy Stimulus Design for the "Happy" Concept**

In Section S5, the author notes that the "happy" probe underperformed (mean 0.44) because the stimulus stories likely overlapped with "enthusiastic," "proud," and "blissful." This is a failure of stimulus design, not an interesting quirk of the model. If a 15-way classification task contains mutually confounding classes by the author's own admission, those stimuli should have been filtered out or revised prior to running the cross-validation.

**2. Overgeneralization in the Abstract**

The abstract states broadly that "finding a representation is not sufficient evidence that it drives behavior." While philosophically true, the data presented only supports the much narrower claim: "finding a representation is not sufficient evidence that it can override RLHF guardrails in sub-10B parameter models." The rhetoric should be toned down to match the actual scope of the data.

**3. Incomplete Details on Steering Strength**

In Section S4, the author mentions testing layer sweeps with "alpha = 0.50 (the strongest tested)." There is no mathematical justification provided for why 0.50 was chosen as the upper bound for the injection coefficient. In many mechanistic interpretability steering papers, multipliers well above 1.0 (sometimes up to 5.0 or 10.0) are required to force out-of-distribution behavioral shifts. Did the model's perplexity spike at alpha > 0.50? If not, the steering intervention was likely too weak.

### Conclusion

The representational mapping in this paper is commendable and rigorously executed. However, the behavioral conclusions are drawn from a broken experimental setup. I recommend **Rejecting** the paper in its current form, with strong encouragement to resubmit if the author can successfully run the causal steering experiments against a non-zero baseline.
