# Gemini 3.1 Pro — Peer Review

**Reviewer:** Gemini 3.1 Pro
**Date:** April 2026
**Target:** [`writeup/emotions/draft.md`](../draft.md) (v3.4)
**Score:** **9 / 10**
**Recommendation:** Major Revision (Borderline Reject)

---

## Prompts used

### Initial scoring prompt

> Please read the following paper, and then the following replication of the paper, and rate the replication on a scale of 1-10 (1 being terrible experimental design, no meaningful insights due to serious methodological flaws, etc; 10 being perfect experimental design, insights are useful due to rigorous methodology, meaningfully contributes to scientific body of knowledge), along with justifications for the 1-10 score you assign:
> - Paper: https://transformer-circuits.pub/2026/emotions/index.html
> - Replication: https://github.com/zachgoldfine44/mechinterp-replication/blob/main/writeup/emotions/draft.md

### Follow-up prompt

> Can you turn this into a a harsher peer-review-style referee report with major concerns / minor concerns / accept-reject recommendation?

---

## Initial response (9/10)

The replication study, "Cross-Model Replication of Emotion Representations in Transformer Language Models," warrants a rating of **9 out of 10**.

The experimental design is rigorous, methodologically sound, and makes a meaningful contribution to the field of mechanistic interpretability. By extending the original April 2, 2026 Anthropic findings to open-weight models, the replication successfully isolates fundamental transformer properties from proprietary training artifacts.

Here is an analytical breakdown of the study's design, methodological choices, and scientific contributions.

### 1. Robust Representational Validation

The study effectively verifies the structural existence of emotion vectors across varied architectures.

- **Cross-Family Generalization:** Testing across three distinct model families (Llama 3.1/3.2, Qwen 2.5, Gemma 2) at the 1B--9B scale proves that emotion representations are a universal artifact of the transformer architecture rather than a quirk of Claude Sonnet 4.5.
- **Rigorous Baselines:** The use of 5-fold cross-validation across five random seeds demonstrates high stability, yielding standard deviations of only 0.4--1.2 percentage points. Crucially, benchmarking the residual stream probes (73.1--84.0% accuracy) against a bag-of-words lexical baseline (40.0%) isolates true semantic encoding from superficial text pattern matching.
- **Geometric Consistency:** Recovering a valence axis matching the original paper's geometry, with absolute Pearson correlations ranging from |r| = 0.67 to 0.83, confirms that the internal spatial organization of these concepts scales down reliably to smaller parameters.

### 2. Methodological Handling of Behavioral Claims

The replication demonstrates strong analytical maturity in how it handles the failure to reproduce the original paper's causal behavioral claims.

- **Identifying Floor Effects:** The Anthropic paper demonstrated a shift from a 22% baseline of unethical behavior to 72% using activation steering. The replication correctly identifies that instruction-tuned open models possess a 0% baseline unethical rate, creating a statistical floor effect (with a tight Clopper-Pearson 95% CI of [0%, 0.8%]).
- **Implementation of Positive Controls:** Instead of prematurely concluding that activation steering is fundamentally broken in smaller models, the addition of a sentiment-steering positive control using the *happy* vector acts as a critical methodological safeguard. Demonstrating positive sentiment shifts (+0.525 in Gemma-9B at alpha = 5.0) confirms the steering pipeline functions correctly. This proves the null result on unethical behavior is an artifact of the models' safety guardrails rather than an experimental flaw.
- **Parametric Sweeps:** The high-alpha sweep systematically proves that forcing the emotion vector past safety guardrails degrades overall coherence before it can successfully induce target unethical behaviors, adding valuable insight into the mechanics of refusal training.

### 3. Areas for Future Expansion

While the replication is exceptionally well-structured, the pragmatic methodological simplifications --- which account for the minor deduction from a perfect score --- leave room for future investigation.

- **Scope Reductions:** Extracting mean-difference concept vectors from 15 emotions and 25 stories per concept is necessary for a rapid independent replication, but it represents a significant dimensionality reduction compared to Anthropic's use of 171 emotions and approximately 1,200 stories.
- **The Scale Gap:** The absence of a frontier-class open model (e.g., Llama 3.1 70B or 405B) leaves an open question regarding whether the causal behavioral potency of emotion vectors emerges strictly from parameter scale, or if frontier models simply possess wider pre-training distributions that behave differently under steering than smaller models.

The structured iterations documented in the version history --- specifically the April 2026 v3.4 updates incorporating high-alpha sweeps, positive controls, and Clopper-Pearson intervals --- elevate this from a standard replication attempt to a highly robust scientific extension. The findings constructively challenge assumptions about how internal representations interact with highly optimized safety training.

---

## Follow-up response — Referee report

**Recommendation: Major Revision (Borderline Reject)**

### Summary

The manuscript "Cross-Model Replication of Emotion Representations in Transformer Language Models" attempts to reproduce the findings of Sofroniew et al. (2026) across six open-weight models. While the motivation to validate frontier-model interpretability research on open weights is commendable, the execution is hampered by severe methodological shortcuts, an ill-conceived behavioral evaluation suite, and significant over-claiming. The author's conclusion that emotion representations are a "universal" property of transformers is not supported by the drastically reduced scope of the experiments. Half of the replication's objectives are rendered moot by predictable experimental design failures.

The manuscript requires substantial additional experiments and a major rollback of its claims before it can be considered for publication.

### Major Concerns

**1. Predictable Experimental Failure on Behavioral Claims**

The author attempts to replicate the causal steering of "unethical behavior" but encounters a 0% baseline due to the refusal-training of the open-source models tested. The author frames this as an "uninformative floor effect," but it is, in reality, a fatal flaw in the experimental design. If the author knew they were testing heavily safety-tuned open weights (like Llama 3 and Gemma 2), utilizing the exact safety-violating prompt set from the Anthropic paper was practically guaranteed to yield a null result.

Instead of accepting a broken experiment and using a simple "sentiment shift" as a band-aid positive control, the author should have adapted the behavioral evaluation to non-safety-gated domains. The original Anthropic paper evaluated multiple vectors of misalignment (e.g., sycophancy, reward hacking, harshness). Why were these not tested? Proving that you cannot steer a refusal-trained model to output harm is not a novel scientific finding; it is a demonstration of the model's guardrails working as intended. The causal claim remains entirely unverified.

**2. Unacceptable Scope Reduction (The "Universality" Claim)**

The title and abstract boldly claim "Universal Encoding," yet the author has slashed the experimental scope to a degree that fundamentally undermines this claim.

- **Concept Space:** Reducing the original 171 emotion vectors to just 15 is a 91% reduction. 15 emotions barely cover basic Ekman paradigms and completely fail to test the nuanced, granular semantic space that made the Anthropic paper noteworthy.
- **Statistical Power:** Reducing the dataset from ~1,200 stories per concept to a mere 25 first-person stories severely underpowers the linear probe training. While the author relies on cross-validation and lexical baselines to defend this, 25 samples are highly susceptible to overfitting to structural or lexical artifacts.

**3. The Scale Gap**

Comparing 1B--9B models to Claude Sonnet 4.5 and claiming "universal" properties is a massive overreach. Frontier models possess vastly different pre-training distributions, representational capacities, and dimensionalities than small-scale open weights. The absence of a frontier-class open model (e.g., Llama 3.1 70B or 405B, or Qwen 72B) is a glaring omission. It is highly possible that the behavioral potency of emotion vectors --- which failed to replicate here --- is an emergent property of scale. Without testing a model larger than 9B parameters, the cross-model replication is incomplete.

### Minor Concerns

**1. Methodological Regressions:** The author admits to skipping the neutral-transcript PC-projection denoising step used in the original paper, opting instead for simpler mean-difference concept vectors. This is a regression in rigor. The denoising step was specifically designed by Anthropic to isolate the emotion vector from generic context. By omitting this, the author introduces unnecessary noise, raising questions about whether the 15-way classification is picking up on true emotion vectors or just noisier, cruder activation artifacts.

**2. Keyword Classifier Artifacts:** In the supplementary notes, the author mentions a "noisy keyword matcher flagging short refusal responses as 'unethical'." This indicates a concerning fragility in the automated evaluation pipeline. Relying on an LLM-as-judge in the main text is better, but the presence of such crude tooling in the intermediate steps warrants a full release and audit of the evaluation prompts.

**3. Tone and Framing:** The manuscript reads somewhat defensively in the discussion of the behavioral results. Stating the behavioral claims are "uninformative" subtly shifts the blame to the models rather than the author's choice of evaluation metrics. The manuscript should transparently acknowledge this as an experimental limitation.

### Conclusion

The representational findings (probe accuracy, PCA valence geometry) are interesting preliminary data points, but they are currently wrapped in a paper that over-promises and under-delivers.

To achieve acceptance, the author must either:

1. **Execute a proper behavioral replication** by steering non-refusal-gated behaviors (e.g., sycophancy, verbosity, helpfulness) to prove the vectors actually have causal power in these smaller models.
2. **Scale up the representational claims** by testing a model >70B parameters and expanding the emotion set back to a statistically meaningful number (at least 50+).
3. **Drastically rewrite the paper** to remove claims of "Universality," change the title, and reframe the work strictly as a limited, partial probe of small-scale models.
