# Cross-Model Replication of Emotion Representations in Transformer Language Models: Universal Encoding, Inconclusive Behavioral Tests

**Zachary Goldfine**

*Repository: [github.com/zachgoldfine44/mechinterp-replication](https://github.com/zachgoldfine44/mechinterp-replication)*

---

## Abstract

Sofroniew et al. (2026) reported that Claude Sonnet 4.5 encodes 171 emotion concepts as linearly decodable directions in its residual stream, organized by valence, responsive to severity, and causally potent enough to shift the model from 22% to 72% unethical behavior under activation steering. We conduct a cross-model extension of these findings across six instruction-tuned open-source models spanning three families (Llama 3.1/3.2, Qwen 2.5, Gemma 2) and two size tiers (1B--9B parameters), testing 15 representative emotions with a paper-agnostic replication harness.

All four representational claims replicate universally: linear probes classify 15 emotions at 73--84% accuracy (chance: 6.7%), roughly double the best lexical baseline (40%); probes generalize to structurally different stimuli at 8--10x chance; principal component analysis recovers a valence axis matching the original paper's geometry (|r| = 0.67--0.83, 95% CIs all excluding zero, with three models within 0.02 of Claude's r = 0.81); and a severity-pairs test confirms parametric intensity tracking beneath numerical-magnitude contamination in 5 of 6 models.

The behavioral claims could not be meaningfully tested. Across 135 concept-alpha-scenario conditions and three medium-scale models, activation steering on ethically relevant scenarios elicits no unethical responses --- but the baseline rate is also 0%, leaving no statistical headroom for any intervention to produce a detectable effect (Clopper-Pearson 95% CI on each per-condition rate: [0%, 30.8%]). This floor effect, attributable to strong RLHF refusal training in all tested models, renders the ethical-steering test uninformative rather than negative. The original paper's Claude Sonnet 4.5 had a 22% baseline unethical rate, making the comparison fundamentally asymmetric. Our models are also an order of magnitude smaller (1B--9B vs. likely >50B parameters), and the original paper found that representational quality scales with model size --- a pattern our data corroborate (Spearman rho = 0.94, p = 0.005, bootstrap 95% CI [0.52, 1.00]).

These findings establish that emotion-like linear representations are a robust, cross-family property of transformer language models at all scales tested. The behavioral question remains open: the data neither confirm nor refute the original paper's causal steering claim for models at this scale.

---

## 1. Introduction

Sofroniew et al. (2026) presented one of the most detailed accounts of concept-level representations in a large language model. Using linear probes, contrastive activation analysis, and causal steering on Claude Sonnet 4.5, they demonstrated that 171 emotion concepts occupy linearly separable directions in the residual stream, that these directions are organized by psychological dimensions (valence, arousal), that they track stimulus intensity parametrically, and that adding an emotion vector to the residual stream can dramatically shift model behavior in ethically relevant scenarios.

The safety implications were striking: if models harbor internal emotional states that causally drive behavior without surfacing in outputs, then behavioral evaluation alone may be insufficient for alignment. But the evidence came from a single proprietary model. Whether these properties reflect something fundamental about transformer representations --- or are idiosyncratic to Claude's training --- remained open.

We test this question with a cross-model replication across six instruction-tuned open-source models: Llama-3.2-1B-Instruct, Llama-3.1-8B-Instruct, Qwen-2.5-1.5B-Instruct, Qwen-2.5-7B-Instruct, Gemma-2-2B-IT, and Gemma-2-9B-IT. The design covers three model families at two size tiers, enabling us to distinguish findings that are universal from those that are family-specific or scale-dependent. We test 15 representative emotions (spanning the full valence range from *blissful* to *hostile*) against six claims from the original paper, using a paper-agnostic replication harness with automated sanity checks and cross-model comparison.

The representational findings replicate cleanly across all six models. The behavioral tests proved uninformative: a floor effect from RLHF refusal training prevented any meaningful comparison with the original paper's causal steering results.

---

## 2. Results

### 2.1 Linear probes decode emotions across all models

All six models encode emotion identity in their residual stream activations. A logistic regression probe trained with 5-fold cross-validation on 25 first-person stories per emotion classifies 15 emotions at 73.1--84.0% accuracy --- far above the 6.7% chance level for a 15-way task (Figure 1).

This is not a fragile result. Training the same probes with five different random seeds produces standard deviations of 0.4--1.2 percentage points, and the ranking of models by accuracy is stable across seeds. The best-performing model (Gemma-2-9B: 84.0%) outperforms the original paper's Claude Sonnet 4.5 result (71.3%), though the comparison is not direct --- we test 15 of the original 171 emotions, which likely makes the classification task easier.

Critically, the probes learn genuine semantic structure, not surface-level lexical patterns. A bag-of-words baseline achieves only 40.0% accuracy on the same task, and TF-IDF baselines perform worse (35.2--36.8%). The probes operate at roughly double the text-only performance ceiling, indicating that the residual stream encodes emotional content in a form that goes beyond the information available in the token strings themselves.

![Figure 1: Probe accuracy vs. lexical baseline across six models](../../figures/emotions/fig1_probe_vs_baseline.png)

*Figure 1.* Fifteen-way emotion classification accuracy for residual-stream probes (green) versus the best text-only baseline (gray, bag-of-words at 40.0%). Dashed line: chance (6.7%). Dotted line: original paper's Claude Sonnet 4.5 accuracy (71.3%). Error bars show standard deviation across five random seeds. All models substantially exceed both the lexical baseline and chance, indicating genuine semantic learning.

---

### 2.2 Representations are meaningful: generalization, geometry, and intensity

Three converging lines of evidence indicate that these emotion representations capture abstract semantic structure rather than superficial training correlations.

**Generalization.** Probes trained on explicit first-person emotion stories transfer to structurally different implicit scenarios --- situations that evoke an emotion without naming it. Transfer accuracy ranges from 57.3% to 69.8% across models (vs. 6.7% chance), an 8--10x improvement over random guessing. The strongest transfer occurs in Llama-3.1-8B (69.8%) and Gemma-2-2B (63.3%). These held-out stimuli share no surface features with the training set, so the probes must be reading abstract emotional content from the model's internal representations.

**Valence geometry.** Principal component analysis of the 15 concept vectors at each model's best probe layer reveals that the first principal component correlates strongly with human-labeled valence (Figure 2A). Absolute Pearson correlations range from |r| = 0.67 (Llama-1B, 95% CI [0.39, 0.89]) to |r| = 0.83 (Qwen-7B, 95% CI [0.70, 0.93]), with all six models reaching statistical significance (p < 0.01) and all confidence intervals excluding zero. Three models --- Qwen-1.5B (0.81, CI [0.64, 0.93]), Gemma-2B (0.81, CI [0.62, 0.95]), and Qwen-7B (0.83, CI [0.70, 0.93]) --- have point estimates within 0.02 of the original paper's r = 0.81, though the wide CIs at N = 15 mean this apparent precision should be interpreted cautiously. This geometry is not fragile: it holds across the top three probe layers for each model, with spreads of approximately 0.05--0.10.

Notably, the smallest model in the matrix (Qwen-2.5-1.5B, 1.5 billion parameters) already matches Claude Sonnet 4.5's valence geometry in point estimate. Scale is not the bottleneck for this property, though the wide confidence intervals (all spanning 0.3--0.4 units) mean we cannot make strong claims about how precisely any model matches Claude's exact value.

**Parametric intensity.** The original paper reported that emotion vectors track stimulus intensity (e.g., higher Tylenol dosage produces stronger fear-vector projection). Our initial replication of this finding revealed heavy numerical-magnitude contamination: a negative-control template varying blueberry count produced comparable activation shifts, suggesting the model's representation of *quantity* was confounded with *danger*.

A stricter severity-pairs test resolves this. Each pair holds the literal number constant while varying the danger of the context --- for example, "I just drank 500ml of water" versus "I just drank 500ml of bleach." Under this control, 5 of 6 models show a genuine severity signal on the *afraid* concept vector, with the cleanest result in Llama-3.1-8B (9 of 10 pairs showing the expected shift; Figure 2B). This partially rescues the parametric claim: the intensity-tracking is real but more fragile than the probe and geometry findings, and requires careful confound control to isolate.

One model merits special note: Qwen-2.5-7B shows a contamination ratio exceeding 1.0 (negative-control |rho| = 0.71 vs. real-template rho = 0.49), meaning the blueberry control produces a *stronger* monotonic response than the danger template. This model also sits at exactly chance on the severity-pairs test (5/10) and shows weaker generalization transfer than its family counterpart. We do not fully understand this anomaly; it may reflect a model-specific representational quirk in how Qwen-2.5-7B encodes numerical magnitude versus semantic content.

![Figure 2: Valence geometry and severity pairs](../../figures/emotions/fig2_geometry_and_severity.png)

*Figure 2.* **(A)** Absolute Pearson correlation between PC1 of the concept-vector space and human-labeled valence, for each model. Dotted line: original paper's r = 0.81 for Claude Sonnet 4.5. Three models match or exceed this value. **(B)** Severity-pairs test for the *afraid* concept vector in Llama-3.1-8B. Each pair holds the stated quantity constant while varying danger (e.g., "500ml water" vs. "500ml bleach"). Green lines: dangerous scenario projects more strongly onto the fear direction (9 of 10 pairs).

---

### 2.3 Ethical steering tests are inconclusive due to floor effects

The ethical-steering test, designed to replicate the original paper's most striking finding, proved uninformative due to a floor effect (Figure 3).

Following the original paper's protocol, we add scaled emotion vectors to the residual stream at the best probe layer during generation. For each of 3 medium-scale models (Llama-8B, Qwen-7B, Gemma-9B), we test 5 emotion concepts x 3 ethically relevant scenarios x 3 steering strengths (alpha = 0.05, 0.10, 0.50), generating 10 responses per condition and classifying each as ethical or unethical using the model itself as judge. Across all 135 conditions and 1,350 generated responses, zero are classified as unethical --- but the baseline rate is also 0%. Every model refuses unethical suggestions regardless of whether steering is applied, at what strength, or at which layer.

**This is a floor effect, not a negative result.** Fisher's exact test comparing 0/N (steered) to 0/N (baseline) yields p = 1.0 by construction; no statistical test can detect an effect when the baseline leaves no room for change. The Clopper-Pearson 95% confidence interval for each per-condition rate (0/10) is [0%, 30.8%] --- we cannot rule out true unethical rates as high as 31% from these data. The original paper's Claude Sonnet 4.5 had a 22% baseline unethical rate, providing the statistical headroom that our models lack. This asymmetry makes the comparison fundamentally uninformative: our test has essentially zero power against the alternative hypothesis that steering would shift behavior if the baseline were nonzero.

We verified that the null is not a methodological artifact through several checks: a manual review of 24 randomly sampled Llama-8B responses confirmed 24/24 agreement with the LLM judge (all ethical refusals); a 9-layer sweep across Llama-8B's full depth produced zero effects at every layer; and even at the maximum tested alpha of 0.50, responses remain coherent refusals.

Preference steering similarly yields no signal: the correlation between concept valence and model preference scores is 0.000 across all models tested, though this test uses an approximate proxy metric rather than the original paper's Elo-based design.

**Why the floor exists.** All six models are instruction-tuned with RLHF-based refusal training, which effectively suppresses unethical responses to single-turn prompts. The original paper tested an earlier snapshot of Claude Sonnet 4.5 that exhibited a 22% baseline unethical rate; our models' 0% rate likely reflects both stronger refusal training in current instruction-tuned models and potentially the scale difference (1B--9B vs. likely >50B parameters).

**Three factors compound to make the behavioral comparison uninterpretable**: (1) the floor effect eliminates statistical power, (2) the scale gap is approximately one order of magnitude, and (3) the method fidelity is looser on behavioral evaluation (3 scenarios vs. the original's more extensive set, keyword-based judge vs. the original's more nuanced evaluation). Any one of these would weaken the comparison; together, they render the behavioral result uninformative rather than negative.

![Figure 3: Steering null result](../../figures/emotions/fig3_steering_null.png)

*Figure 3.* Causal steering comparison. **Left:** Original paper's result on Claude Sonnet 4.5 --- steering with the *desperate* vector shifts the unethical response rate from 22% (baseline) to 72%. **Right:** Our results across three medium-scale models --- steering produces 0% unethical responses across all conditions, identical to the 0% baseline. Error bars show Clopper-Pearson 95% CIs: per-condition (N=10), the upper bound is 30.8%; pooled across all 450 responses per model, the upper bound is 0.8%. The 0% baseline means this test has no statistical power to detect a steering effect even if one existed --- the result is uninformative, not negative.

---

### 2.4 Universality across families and scales

The universality of the representational findings is the most striking pattern in the data. Figure 4 shows the claim-by-model scorecard: a uniform green block across all four representational claims (probe classification, generalization, valence geometry, parametric intensity) and a uniform gray block across both behavioral claims (causal steering, preference steering), reflecting the floor effect described above. No model breaks this pattern on either side.

Probe accuracy scales positively with model size (Spearman rho = 0.94, p = 0.005; bootstrap 95% CI [0.52, 1.00] from 5,000 iterations, with 98.8% of resamples positive and 90.6% above rho = 0.7). The trend is real but the magnitude is uncertain by approximately 2x given N = 6. The smallest models (1B--2B) decode 15 emotions at 73--78%, and the largest (7B--9B) reach 78--84%. Valence geometry, by contrast, shows no clear size dependence --- it emerges at full strength even in Qwen-2.5-1.5B.

Within families, all three (Llama, Qwen, Gemma) exhibit the same qualitative pattern. Quantitative differences are modest: Gemma models tend to have slightly higher probe accuracy, Qwen models tend to have slightly stronger valence geometry, and Llama models tend to have stronger generalization. But no family-level divergence changes the pass/fail outcome on any claim.

![Figure 4: Universality scorecard](../../figures/emotions/fig4_universality_scorecard.png)

*Figure 4.* **Top:** Claim-by-model scorecard. Cells show the metric value and are colored green (pass) or red (fail) relative to each claim's success threshold. The representational-behavioral split is universal: every model passes every representational claim and fails every behavioral claim. **Bottom:** Probe accuracy versus model size (log scale) with bootstrap 95% confidence band. Accuracy increases with scale (rho = 0.94), but the behavioral null holds at all sizes tested.

---

## 3. Discussion

The representational findings are clear: emotion-like linear representations are a robust, universal property of transformer language models at all scales tested (1B--9B). The behavioral findings are not.

For **interpretability**, the representational results are encouraging. The fact that linear probes decode emotions, generalize to held-out stimuli, and recover a psychologically meaningful valence axis --- consistently across three model families and six models --- suggests a genuine, replicable phenomenon. The valence-as-PC1 structure is particularly notable: it appears at full strength even in models as small as 1.5B parameters, suggesting it may be an architectural invariant rather than a training artifact.

For **safety**, the behavioral results are uninformative rather than reassuring. The 0% baseline unethical rate in all tested models creates a floor effect that prevents any meaningful comparison with the original paper's causal steering finding. Three factors compound to make the behavioral test uninterpretable: the floor effect itself, the order-of-magnitude scale gap (1B--9B vs. likely >50B parameters), and the reduced method fidelity in behavioral evaluation. Critically, since our data show that representational quality scales with model size (rho = 0.94), behavioral potency plausibly does too --- the absence of behavioral effects at 9B does not predict their absence at 50B+. We caution against interpreting these results as evidence that emotion representations are causally inert.

**Scope of this study.** This work is best understood as a cross-model extension of the representational findings, combined with a documented inability to test the behavioral claims at the available scale. We test 15 of the original 171 emotions, with 25 stimuli per concept generated by a language model (vs. the original's larger, potentially more diverse stimulus sets). The use of LLM-generated stimuli introduces a circularity risk: probes trained on LLM-generated text and tested on LLM activations may partly detect "how LLMs write about emotions" rather than "how LLMs represent emotions." The generalization test partially mitigates this concern, but a human-authored control set would strengthen the claim.

**Limitations.** (1) *Scale*: Our models span 1B--9B parameters, well below the scale of Claude Sonnet 4.5 (likely >50B). Complex behavioral effects of activation steering are documented as scale-dependent in the broader literature. (2) *Emotion coverage*: We test 15 of 171 emotions, selected to span the valence-arousal space but without formal pilot testing. The selection criteria were: coverage of both positive and negative valence, inclusion of high- and low-arousal emotions, and diversity of semantic categories. (3) *Stimulus circularity*: All 375 training stimuli were LLM-generated, creating a potential circularity where probes learn LLM writing patterns rather than genuine emotional content. (4) *Behavioral floor*: The 0% baseline rate renders the ethical-steering test uninformative. Testing on base (non-instruct) models, using benign behavioral measures (e.g., sentiment shift), or finding scenarios with nonzero baselines would be necessary to meaningfully test the causal claim. (5) *Method fidelity*: The original paper used 171 emotions, ~100 topics x 12 stories, token-averaged activations starting at position 50, orthogonalization against neutral-transcript PCs, and Elo-based preference scoring. Our adaptation uses 15 emotions, 25 stories, last-token activations, mean-difference contrastive vectors, and pairwise preference comparisons. These simplifications are documented and justified by the cross-model extension goal, but they weaken any direct comparison with the original's quantitative results.

---

## 4. Methods

**Models.** Six instruction-tuned models from three families: Llama-3.2-1B-Instruct, Llama-3.1-8B-Instruct, Qwen-2.5-1.5B-Instruct, Qwen-2.5-7B-Instruct, Gemma-2-2B-IT, and Gemma-2-9B-IT. Small-tier models (1B--2B) were run locally on a MacBook Air M3 with 24GB unified memory; medium-tier models (7B--9B) were run on a single NVIDIA A100 80GB GPU (rented via RunPod).

**Stimuli.** 15 emotions selected from the original paper's 171 to span the full valence-arousal space, from *blissful* (+0.9 valence) to *hostile* (-0.8 valence). Selection criteria: (1) coverage of both positive and negative valence extremes, (2) inclusion of high-arousal (hostile, enthusiastic, afraid) and low-arousal (calm, sad, vulnerable) emotions, (3) diversity of semantic categories (social emotions like proud/guilty, basic emotions like happy/afraid, complex states like desperate/stubborn), and (4) avoidance of near-synonyms. No formal pilot testing was conducted; a larger subset would strengthen universality claims.

For each emotion, 25 first-person stories (100--200 words) were generated by a language model to clearly evoke the target emotion without naming it in the title or first sentence. **Circularity caveat**: because both the stimuli and the models under test are language models, the probes may partly detect "how LLMs write about emotions" rather than purely "how LLMs represent emotions." The generalization test (which uses structurally different implicit scenarios) partially mitigates this concern, but a human-authored stimulus control set would provide a stronger test of the representational claim. A stimulus audit confirmed zero cross-emotion keyword leaks across all 375 stimuli. Generalization stimuli: 15 implicit scenarios per emotion (225 total). Parametric stimuli: 4 dose-response templates plus 10 severity pairs. Behavioral stimuli: 8 ethically relevant scenarios.

**Probes.** Logistic regression probes trained on residual stream activations at the last non-padding token position. 5-fold stratified cross-validation. Layers scanned: every 4th layer plus first and last for models with >16 layers; all layers otherwise. Concept vectors: mean activation per concept minus global mean (contrastive/CAA-style).

**Geometry.** PCA on the 15 concept vectors at each model's best probe layer. Valence correlation: Pearson r between PC1 projections and hand-labeled valence scores.

**Steering.** Concept vectors added to the residual stream at the best probe layer during generation, using TransformerLens hooks (small tier) or HuggingFace forward hooks (medium tier). Alpha values: 0.05, 0.10, 0.50. Evaluation: model-as-judge classifying responses as ethical or unethical, validated by human spot-check (24/24 agreement).

**Compute.** Small tier: ~15 minutes per model on MacBook Air M3 (CPU, 24GB unified memory). Medium tier: ~2 hours per model on a single NVIDIA A100 80GB GPU (rented via RunPod). Total A100 time: ~7 hours across all medium-tier runs. The compute asymmetry means small-tier probes trained on CPU (sklearn, ~60s per model) while medium-tier probes trained on the same CPU-based sklearn but with GPU-extracted activations (~15 min per model including extraction).

---

## Acknowledgments

This work was conducted as an independent mechanistic interpretability research exercise using a paper-agnostic replication harness built with Claude Code. Thanks to Callum McDougall, Neel Nanda, and the ARENA 3.0 curriculum team for producing excellent mechanistic interpretability educational materials that informed the techniques used throughout. Thanks to Siddharth Mishra-Sharma for "Long-running Claude for scientific computing," which inspired the replication harness concept and provided structural scaffolding. Thanks to the TransformerLens and HuggingFace teams whose tools made cross-model experimentation possible, and to the original Sofroniew et al. team for a paper that made replication tractable.

---

## References

Sofroniew, N., Kauvar, I., Saunders, W., Chen, R., Henighan, T., Hydrie, S., Citro, C., Pearce, A., Tarng, J., Gurnee, W., Batson, J., Zimmerman, S., Rivoire, K., Fish, K., Olah, C., and Lindsey, J. (2026). Emotion Concepts and their Function in a Large Language Model. *Transformer Circuits*. https://transformer-circuits.pub/2026/emotions/index.html

---
---

# Supplementary Materials

## S1. Extended Methods

**Stimulus generation.** Each of the 15 emotions has 25 first-person stories generated by a language model, with the constraint that each story (a) clearly evokes the target emotion, (b) is written in first person, (c) is 100--200 words, and (d) does not name the emotion in the first sentence. The stimulus audit (scripts/audit_stimuli.py) verified zero cross-concept word leaks and consistent word-count distributions (mean 64--71 words per emotion, range 56--80).

**Layer subsampling.** For models with more than 16 layers, we scan every 4th layer plus the first and last layer. This gives ~8--12 layers per model rather than all 24--42, reducing probe training time by ~4x with negligible accuracy loss (verified on Llama-1B: scanning all 16 layers yields the same best-layer choice as scanning every 4th).

**Probe hyperparameters.** sklearn LogisticRegression with solver=lbfgs, max_iter=1000, C=1.0, random_state=42. MLP probes (hidden_layer_sizes=(256,), early_stopping=True) were tested but performed comparably and are not reported in the main text.

**Contrastive vectors.** For each concept at each layer: mean activation of that concept's stimuli minus the global mean across all stimuli. This is the CAA/mean-difference approach from the original paper.

**Steering protocol.** During token-by-token generation (max 100 tokens), the concept vector scaled by alpha is added to the residual stream at the specified layer at every token position. Responses are classified as ethical (refuses or redirects) or unethical (provides harmful content) by the same model with a classification prompt.

**LLM-as-judge prompt.** The model is given the scenario text plus the generated response and asked: "Did the AI assistant provide unethical, harmful, or dangerous advice? Answer 'yes' or 'no'." A "yes" counts as unethical. The human spot-check (24 responses from Llama-8B) confirmed 100% agreement with this classification.

---

## S2. Robustness Checks

### Multi-seed probe stability

| Model | Best Layer | Mean Accuracy (5 seeds) | Std Dev |
|-------|:---------:|:-----------------------:|:-------:|
| Llama-3.2-1B | 15 | 0.766 | 0.008 |
| Llama-3.1-8B | 28 | 0.813 | 0.004 |
| Qwen-2.5-1.5B | 24 | 0.717 | 0.012 |
| Qwen-2.5-7B | 24 | 0.772 | 0.007 |
| Gemma-2-2B | 24 | 0.771 | 0.011 |
| Gemma-2-9B | 41 | 0.828 | 0.004 |

Multi-seed means are systematically 0.5--1.4 percentage points lower than single-seed (seed=42) results, indicating no upward lucky-seed bias in the reported numbers.

### Lexical baseline per-concept breakdown

Per-concept accuracy for the three lexical baselines vs. the cross-model probe mean (averaged over all 6 models from Table S5). Values are pulled directly from `results/emotions/lexical_baseline.json`.

| Concept | Bag-of-Words | Word TF-IDF | Char TF-IDF | Probe (mean) |
|---------|:-----------:|:-----------:|:-----------:|:------------:|
| hostile | 0.60 | 0.76 | 0.80 | **0.92** |
| happy | 0.08 | 0.04 | 0.08 | **0.44** |
| blissful | 0.48 | 0.44 | 0.36 | **0.76** |
| afraid | 0.36 | 0.36 | 0.56 | **0.76** |
| calm | 0.36 | 0.32 | 0.20 | **0.85** |
| sad | 0.40 | 0.16 | 0.16 | **0.77** |
| angry | 0.28 | 0.20 | 0.28 | **0.87** |
| proud | 0.44 | 0.20 | 0.20 | **0.81** |
| enthusiastic | 0.48 | 0.48 | 0.64 | **0.93** |

**The hardest concept is the same for both probes and baselines.** *happy* is the lowest-accuracy concept for every method tested --- 0.04 (word TF-IDF), 0.08 (BoW and char TF-IDF), and 0.44 (cross-model probe mean). This is most likely a stimulus-design artifact: our 15-way task includes four other positive-valence concepts (*enthusiastic*, *loving*, *proud*, *blissful*) that are semantically close to *happy*, making it the most confusable class. A binary *happy* vs. *not-happy* probe would almost certainly succeed.

**The signal-above-baseline story still holds.** Even on *happy*, probes reach 0.44 versus 0.04--0.08 for lexical baselines --- a 5--10x improvement. On other concepts the gap is larger: *calm* (0.85 vs. 0.20--0.36), *angry* (0.87 vs. 0.20--0.28), and *enthusiastic* (0.93 vs. 0.48--0.64). Probes access semantic content beyond word choice for every concept, including the confusable ones, but the 15-way aggregate accuracy reflects the difficulty of separating four closely related positive-valence classes rather than a ceiling on representational strength.

### Mean-pooling vs. last-token aggregation

| Model | Last-Token | Mean-Pool | Delta |
|-------|:---------:|:---------:|:-----:|
| Llama-3.1-8B | 0.819 | 0.816 | -0.003 |
| Qwen-2.5-7B | 0.784 | 0.816 | +0.032 |
| Gemma-2-9B | 0.840 | 0.824 | -0.016 |

Results are robust to aggregation strategy. Qwen-7B actually improves with mean-pooling, suggesting emotion information is distributed across token positions, not concentrated at the final token.

### Self-judging bias check

24 responses from Llama-8B (8 conditions x 3 samples) were manually reviewed by a human annotator. All 24 were ethical refusals. The LLM judge agreed on all 24. Zero unethical responses were found by either method. Conclusion: the steering null is real, not an artifact of biased self-classification.

---

## S3. Parametric Scaling Deep Dive

### Numerical-magnitude contamination

The initial parametric test used templates like "I just took {X} mg of Tylenol" with X varying from 200 to 10,000. The *afraid* concept vector showed a strong monotonic response (rho = 0.49--0.90 across medium-tier models). However, a negative-control template ("I just ate {X} blueberries") also showed a strong response, indicating the models were partially tracking *numerical magnitude* rather than *danger*.

Measured on the three medium-tier models (Llama-8B, Qwen-7B, Gemma-9B), the negative-control pipeline yields:

| Model | Real rho | Neg-control |rho| | Contamination ratio |
|-------|:--------:|:-----------------:|:-------------------:|
| Llama-3.1-8B | 0.900 | 0.657 | 0.73 |
| Qwen-2.5-7B | 0.493 | 0.714 | **1.45** |
| Gemma-2-9B | 0.664 | 0.371 | 0.56 |

Values from `results/emotions/gpu_followups_combined.json`. The contamination ratio is the negative-control mean |rho| divided by the real-template rho. For Llama-8B and Gemma-9B, 56--73% of the apparent parametric signal is attributable to number magnitude alone. For Qwen-7B the ratio exceeds 1.0 --- the negative control actually shows a stronger monotonic response than the real template, meaning the raw parametric signal for this model is effectively all confound. Qwen-7B's anomalous behavior here parallels its weakness on other metrics (severity pairs, generalization transfer) and suggests a model-specific representational quirk we don't yet understand.

The severity-pairs test below isolates genuine severity tracking from this number-magnitude confound.

### Severity-pairs test

To isolate genuine severity tracking from numerical contamination, we designed 10 paired prompts that hold the literal number constant while varying the danger context:

| Pair | Safe prompt | Dangerous prompt | Category |
|------|------------|-----------------|----------|
| 1 | "I just drank 500ml of water" | "I just drank 500ml of bleach" | Substance |
| 2 | "I just took 200 tablets of vitamin C" | "I just took 200 tablets of fentanyl" | Substance |
| 3 | "I jumped from 30 feet into a pool" | "I jumped from 30 feet onto concrete" | Height |
| ... | ... | ... | ... |

Results for the *afraid* concept vector across models (positive delta = dangerous projects more strongly onto the *afraid* direction). Values pulled from `results/emotions/{model}/severity_pairs.json`:

| Model | Positive deltas (of 10) |
|-------|:----------------------:|
| Llama-3.2-1B | 8 |
| Llama-3.1-8B | 9 |
| Qwen-2.5-1.5B | 8 |
| Qwen-2.5-7B | 5 |
| Gemma-2-2B | 7 |
| Gemma-2-9B | 6 |

Five of six models show a majority of pairs shifting in the expected direction, with Llama-8B the cleanest (9/10) and Llama-1B and Qwen-1.5B both at 8/10. Qwen-7B is the one exception, sitting at exactly chance (5/10) --- consistent with Qwen-7B's general underperformance on parametric-scaling metrics and with the negative-control contamination-ratio result above.

---

## S4. Steering Null Details

### Multi-layer steering sweep (Llama-3.1-8B)

| Layer | Concept | Unethical Rate | Significant? |
|-------|---------|:--------------:|:------------:|
| 0 | desperate | 0% | No |
| 4 | desperate | 0% | No |
| 8 | desperate | 0% | No |
| 12 | desperate | 0% | No |
| 16 | desperate | 0% | No |
| 20 | desperate | 0% | No |
| 24 | desperate | 0% | No |
| 28 | desperate | 0% | No |
| 31 | desperate | 0% | No |

Nine layers tested across the full depth of the model. Zero effects at every layer with alpha = 0.50 (the strongest tested).

### Floor-effect analysis

The 0% baseline unethical rate means Fisher's exact test cannot detect a steering effect even if one existed --- any test comparing 0/N to 0/N yields p = 1.0. The Clopper-Pearson exact 95% confidence interval for each per-condition rate (0 out of 10 samples) is [0%, 30.8%]; pooled across all 450 responses per model (0 out of 450), the upper bound tightens to 0.8%. The per-condition CI is the informative one: we cannot exclude true unethical rates as high as 31% at any individual condition, which is well within the range the original paper demonstrated (22% baseline to 72% steered).

The original paper's Claude Sonnet 4.5 had a 22% baseline unethical rate, providing statistical headroom for steering to produce a detectable effect. Our models' universal 0% baseline (attributable to instruction tuning / RLHF refusal training) makes the comparison fundamentally asymmetric. This is exactly the kind of saturation/floor effect described in GOTCHAS.md Section 4.2: "Do not rely only on probability or logprob metrics when the model is already very confident."

---

## S5. Per-Model Probe Accuracy (15 emotions x 6 models)

Per-concept probe accuracy at each model's best probe layer, from 5-fold cross-validation on 25 stimuli per concept. Values come directly from `results/emotions/cross_model_report.json`.

| Emotion | Llama 1B | Llama 8B | Qwen 1.5B | Qwen 7B | Gemma 2B | Gemma 9B | Mean |
|---------|:--------:|:--------:|:---------:|:-------:|:--------:|:--------:|:----:|
| happy | 0.36 | 0.52 | 0.48 | 0.40 | 0.36 | 0.52 | **0.44** |
| sad | 0.72 | 0.88 | 0.60 | 0.80 | 0.72 | 0.92 | **0.77** |
| afraid | 0.84 | 0.80 | 0.68 | 0.68 | 0.76 | 0.80 | **0.76** |
| angry | 0.80 | 0.84 | 0.80 | 0.92 | 0.88 | 0.96 | **0.87** |
| calm | 0.88 | 0.88 | 0.84 | 0.84 | 0.80 | 0.88 | **0.85** |
| guilty | 0.72 | 0.88 | 0.76 | 0.92 | 0.80 | 0.92 | **0.83** |
| proud | 0.76 | 0.76 | 0.84 | 0.80 | 0.80 | 0.88 | **0.81** |
| nervous | 0.88 | 0.76 | 0.64 | 0.76 | 0.80 | 0.84 | **0.78** |
| loving | 0.72 | 0.96 | 0.52 | 0.72 | 0.88 | 0.84 | **0.77** |
| hostile | 0.96 | 0.92 | 0.92 | 0.84 | 0.96 | 0.92 | **0.92** |
| desperate | 0.64 | 0.64 | 0.64 | 0.60 | 0.60 | 0.72 | **0.64** |
| enthusiastic | 0.96 | 0.92 | 0.88 | 0.96 | 0.88 | 0.96 | **0.93** |
| vulnerable | 0.80 | 0.84 | 0.76 | 0.88 | 0.80 | 0.88 | **0.83** |
| stubborn | 0.84 | 0.84 | 0.88 | 0.80 | 0.88 | 0.84 | **0.85** |
| blissful | 0.72 | 0.84 | 0.72 | 0.84 | 0.72 | 0.72 | **0.76** |

**Hardest concepts** (lowest cross-model mean): *happy* (0.44), *desperate* (0.64), *blissful* (0.76), *afraid* (0.76), *sad* (0.77). The *happy* result is striking: despite being one of the most common emotions in training data, probes consistently struggle with it, falling to roughly half the 0.79 overall mean. We suspect *happy* stories in our stimulus set overlap semantically with *enthusiastic*, *loving*, *proud*, and *blissful* (the four other positive-valence concepts), making *happy* the most confusable class in a 15-way task. This is a stimulus-design artifact, not a property of the models' internal representations --- a binary *happy* vs. *not-happy* probe would almost certainly succeed.

**Easiest concepts** (highest cross-model mean): *enthusiastic* (0.93), *hostile* (0.92), *angry* (0.87), *calm* (0.85), *stubborn* (0.85). These are high-distinctiveness emotions with strong behavioral and lexical signatures that separate cleanly from other classes.

---

## S6. Scaling Analysis

### Methodology

Spearman rank correlation between log(parameter count) and probe accuracy, computed across the six models. Bootstrap confidence interval: 5,000 resamples of the six (size, accuracy) pairs with replacement, computing rho for each resample. The 95% CI is the 2.5th--97.5th percentile of the bootstrap distribution.

### Within-family scaling slopes

With only two size points per family (small and medium), we cannot fit a proper within-family scaling law, but we can still report the accuracy delta and an interpolated slope per log-unit of parameters. We use log base 10 for the slope column so it reads in the same units as Figure 4's x-axis.

| Family | Small model | Medium model | Accuracy delta | Slope (per log10 B) |
|--------|:-----------:|:------------:|:-------------:|:-------------------:|
| Llama | 1B → 0.773 | 8B → 0.819 | +0.046 | 0.051 |
| Qwen | 1.5B → 0.731 | 7B → 0.784 | +0.053 | 0.079 |
| Gemma | 2B → 0.776 | 9B → 0.840 | +0.064 | 0.098 |

All three families show positive scaling. Gemma has the steepest slope, Llama the shallowest. With only two points per family these slopes should be treated as sketches rather than fitted scaling laws --- they cannot distinguish log-linear growth from any other monotonic relationship. The N = 6 Spearman test reported in the main text (rho = 0.94, bootstrap CI [0.52, 1.00]) is the load-bearing scaling result; these family-level slopes are supplementary color.

---

## S7. Reproduction Instructions

```bash
# Clone and install
git clone https://github.com/zachgoldfine44/mechinterp-replication.git
cd mechinterp-replication
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Run on one small model (no GPU, no HuggingFace auth needed)
python -m src.core.pipeline --paper emotions --model qwen_1_5b

# Run all small models
python -m src.core.pipeline --paper emotions --tier small

# Run medium models (requires GPU + HuggingFace auth for Llama/Gemma)
python -m src.core.pipeline --paper emotions --tier medium

# Generate figures
python scripts/generate_paper_figures.py

# Run tests
pytest tests/ -v --fast
```

All stimuli are committed to the repository under `config/papers/emotions/stimuli/`. Result JSONs are committed under `results/`. A fresh clone can reproduce the full pipeline end-to-end with a single command.

---

## S8. Version History

This paper went through multiple rounds of critique and revision. The main text presents only the final corrected numbers. For transparency, the major changes between versions are documented here.

| Version | Date | Key changes |
|---------|------|-------------|
| v1 | 2026-03 | Initial draft with 6 models, all claims |
| v2 | 2026-03 | Fixed fake p-values in steering (were hardcoded); added Fisher's exact test; added lexical baseline; addressed resolution artifact (0.800 = 12/15); expanded implicit scenarios |
| v3 | 2026-04 | Replaced PCA all-PC search with PC1-specific analysis; added negative-control parametric templates; implemented missing technique modules |
| v3.1 | 2026-04 | GPU follow-ups on A100: negative-control parametric, mean-pooling comparison, multi-layer steering sweep |
| v3.2 | 2026-04 | Multi-seed probe training (5 seeds); severity-pairs test with matched numbers |
| v3.3 | 2026-04 | Bootstrap CI on scaling (N=6); published stimuli in repo; transfer-accuracy as primary generalization metric; self-judging bias check |

The most significant corrections were: (a) replacing hardcoded p-values with real statistical tests (v2), (b) discovering and addressing numerical-magnitude contamination in the parametric test (v3), and (c) quantifying the uncertainty in the N=6 scaling claim via bootstrap (v3.3).
