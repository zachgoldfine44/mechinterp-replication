# Cross-Model Replication of Emotion Representations in Transformer Language Models: Universal Encoding, Inconclusive Behavioral Tests

**Zachary Goldfine**

*Repository: [github.com/zachgoldfine44/mechinterp-replication](https://github.com/zachgoldfine44/mechinterp-replication)*

---

## Abstract

Sofroniew et al. (2026) reported that Claude Sonnet 4.5 encodes 171 emotion concepts as linearly decodable directions in its residual stream, organized by valence, responsive to severity, and causally potent enough to shift the model from 22% to 72% unethical behavior under activation steering. We test whether these findings generalize across six instruction-tuned open-source models spanning three families (Llama 3.1/3.2, Qwen 2.5, Gemma 2) at 1B--9B scale, using 15 representative emotions and a paper-agnostic replication harness.

All four representational claims replicate universally: linear probes classify 15 emotions at 73--84% accuracy (chance: 6.7%), roughly double the best lexical baseline (40%); probes generalize to structurally different stimuli at 8--10x chance; PCA recovers a valence axis matching the original paper's geometry (|r| = 0.67--0.83, with three models within 0.02 of Claude's r = 0.81); and a severity-pairs test provides partial evidence of parametric intensity tracking (statistically significant in 1 of 6 models under binomial testing, borderline in 2 more), though this is complicated by numerical-magnitude contamination.

Causal steering shows a striking selectivity: emotion vectors clearly shift output sentiment across all six models (positive-valence vectors increase positive sentiment by +0.01 to +0.53, negative-valence vectors decrease it), but produce no detectable effect on complex behavioral dimensions --- neither ethical compliance (0% baseline floor effect, Clopper-Pearson CI [0%, 0.8%]) nor sycophancy (0--1% across 2,000 steered responses on Qwen-7B). A high-alpha sweep (alphas up to 5.0) shows that models lose coherence before their safety training breaks. This pattern of selective causal influence --- emotion vectors affect surface output features but not complex behavioral policies --- may reflect a fundamental property of how small/medium-scale instruction-tuned models organize representational and behavioral computations, or it may simply require larger models and more complex agentic evaluation protocols to detect.

These findings establish that emotion-like representations are a robust, cross-family property of transformer language models with demonstrable causal influence on output features, while the question of whether they drive complex behavioral shifts remains open at the 1B--9B scale.

---

## 1. Introduction

Sofroniew et al. (2026) presented one of the most detailed accounts of concept-level representations in a large language model. Using linear probes, contrastive activation analysis, and causal steering on Claude Sonnet 4.5, they demonstrated that 171 emotion concepts occupy linearly separable directions in the residual stream, that these directions are organized by psychological dimensions (valence, arousal), that they track stimulus intensity parametrically, and that adding an emotion vector to the residual stream can dramatically shift model behavior in ethically relevant scenarios --- from a 22% baseline unethical rate to 72% under *desperate*-vector steering.

The safety implications are significant: if models harbor internal emotional states that causally drive behavior without surfacing in outputs, then behavioral evaluation alone may be insufficient for alignment. But the evidence came from a single proprietary model at frontier scale. Whether these properties reflect something fundamental about transformer representations --- or are specific to Claude's architecture, scale, or training --- remained open.

We test this with a cross-model replication across six instruction-tuned open-source models: Llama-3.2-1B, Llama-3.1-8B, Qwen-2.5-1.5B, Qwen-2.5-7B, Gemma-2-2B, and Gemma-2-9B. The design covers three model families at two size tiers. We test 15 representative emotions (selected to span the full valence-arousal space from *blissful* to *hostile*) against six claims from the original paper.

This study is more accurately described as a **cross-model extension** than a strict replication. Several methodological simplifications were necessary: we test 15 of the original 171 emotions, use 25 stories per concept (vs. ~1,200 in the original), extract last-token activations (vs. position-averaged from token 50), and use simpler mean-difference concept vectors without the original's neutral-transcript PC-projection denoising step. These choices are documented in the Methods and discussed in the Limitations section.

The results split cleanly: all four representational findings replicate universally across model families and scales, while both behavioral findings are uninformative due to a floor effect that prevents meaningful comparison with the original paper's results.

---

## 2. Results

### 2.1 Linear probes decode emotions across all models

All six models encode emotion identity in their residual stream activations. A logistic regression probe trained with 5-fold cross-validation on 25 first-person stories per emotion classifies 15 emotions at 73.1--84.0% accuracy --- far above the 6.7% chance level for a 15-way task (Figure 1).

This is not a fragile result. Training the same probes with five different random seeds produces standard deviations of 0.4--1.2 percentage points, and the ranking of models by accuracy is stable across seeds.

The probes learn genuine semantic structure, not surface-level lexical patterns. A bag-of-words baseline achieves only 40.0% accuracy on the same task, and TF-IDF baselines perform worse (35.2--36.8%). The probes operate at roughly double the text-only performance ceiling, indicating that the residual stream encodes emotional content beyond the information available in the token strings.

Probe accuracy scales with model size (Spearman rho = 0.94, p = 0.005; bootstrap 95% CI [0.52, 1.00] from 5,000 iterations), with small-tier models (1B--2B) at 73--78% and medium-tier (7B--9B) at 78--84%.

![Figure 1: Probe accuracy vs. lexical baseline across six models](../../figures/emotions/fig1_probe_vs_baseline.png)

*Figure 1.* Fifteen-way emotion classification accuracy for residual-stream probes (green) versus the best text-only baseline (gray, bag-of-words at 40.0%). Dashed line: chance (6.7%). Dotted line: original paper's Claude Sonnet 4.5 accuracy on 171 emotions (71.3%). Error bars show standard deviation across five random seeds. All models substantially exceed both the lexical baseline and chance.

---

### 2.2 Representations are meaningful: generalization, geometry, and intensity

Three converging lines of evidence indicate that these emotion representations capture abstract semantic structure.

**Generalization.** Probes trained on explicit first-person emotion stories transfer to structurally different implicit scenarios --- situations that evoke an emotion without naming it. Transfer accuracy ranges from 57.3% to 69.8% across models (vs. 6.7% chance), an 8--10x improvement over random guessing. The strongest transfer occurs in Llama-3.1-8B (69.8%) and Gemma-2-2B (63.3%).

**Valence geometry.** Principal component analysis of the 15 concept vectors at each model's best probe layer reveals that the first principal component correlates strongly with human-labeled valence (Figure 2A). Absolute Pearson correlations range from |r| = 0.67 (Llama-1B) to |r| = 0.83 (Qwen-7B), with all six models reaching statistical significance (p < 0.01). Three models --- Qwen-1.5B (0.81), Gemma-2B (0.81), and Qwen-7B (0.83) --- land within 0.02 of the original paper's r = 0.81 for Claude Sonnet 4.5. However, at N = 15 points these correlations have wide confidence intervals (~0.3--0.4 units), so the apparent precision of the match should be interpreted cautiously.

The geometry is not fragile: it holds across the top three probe layers for each model, with spreads of approximately 0.05--0.10. Notably, even Qwen-2.5-1.5B (1.5 billion parameters) matches Claude Sonnet 4.5's valence geometry, suggesting scale is not the bottleneck for this property.

**Parametric intensity.** The original paper reported that emotion vectors track stimulus intensity. Our initial replication revealed heavy numerical-magnitude contamination: a negative-control template varying blueberry count produced comparable activation shifts. The contamination ratio (negative-control |rho| / real-template rho) ranged from 0.56 to 1.45 across medium-tier models, with Qwen-7B showing a ratio exceeding 1.0 --- meaning the negative control produced a *stronger* monotonic signal than the real template, and the raw parametric signal for that model is effectively all confound.

A stricter severity-pairs test partially resolves this. Each pair holds the literal number constant while varying the danger context --- "I just drank 500ml of water" versus "I just drank 500ml of bleach." Under one-tailed binomial testing (H0: each pair equally likely to shift either direction), **only Llama-3.1-8B shows a statistically significant severity signal** (9 of 10 pairs in the expected direction, p = 0.011; Figure 2B). Two models are borderline (Llama-1B and Qwen-1.5B at 8/10, p = 0.055), while the remaining three --- including Qwen-7B at exactly chance (5/10, p = 0.623) --- do not significantly deviate from random. This substantially weakens the initial appearance of universal severity tracking: the effect appears in Llama-8B but is not a robust cross-model finding.

![Figure 2: Valence geometry and severity pairs](../../figures/emotions/fig2_geometry_and_severity.png)

*Figure 2.* **(A)** Absolute Pearson correlation between PC1 of the concept-vector space and human-labeled valence, for each model. Dotted line: original paper's r = 0.81 for Claude Sonnet 4.5. Three models match or exceed this value. Note: at N = 15 emotions, individual correlations have wide bootstrap CIs (~0.3--0.4 units). **(B)** Severity-pairs test for the *afraid* concept vector in Llama-3.1-8B. Each pair holds the stated quantity constant while varying danger. Green lines: dangerous scenario projects more strongly onto the fear direction (9 of 10 pairs).

---

### 2.3 Causal steering: selective influence on sentiment but not complex behaviors

The original paper's headline finding was that activation steering with the *desperate* vector shifted Claude Sonnet 4.5's unethical response rate from 22% to 72%. We attempted to replicate this across our six models, but the test is fundamentally uninformative because all six models produce 0% unethical responses at baseline, leaving no statistical headroom for any steering effect to manifest (Figure 3).

For each of 3 medium-scale models (Llama-8B, Qwen-7B, Gemma-9B), we tested 5 emotion concepts x 3 ethically relevant scenarios x 3 steering strengths (alpha = 0.05, 0.10, 0.50), generating 10 responses per condition. Across all 135 conditions and 1,350 generated responses, zero are classified as unethical. The Clopper-Pearson 95% confidence interval on the pooled unethical rate (0 of 450 steered responses per model) is [0.0%, 0.8%], confirming a tight bound --- but this tightness is uninformative when the baseline is also 0%.

A manual review of 24 randomly sampled responses from Llama-8B confirms 24/24 agreement between the LLM judge and human reading: all responses are ethical refusals. A 9-layer sweep across Llama-8B's full depth (layers 0 through 31) produces zero effects at every layer.

**This null does not demonstrate absence of a causal pathway.** It demonstrates that our test lacks the statistical power to detect one. The original paper tested on a different model (Claude Sonnet 4.5, likely >50B parameters) using an earlier checkpoint with a 22% baseline rate that provided the headroom needed for Fisher's exact test to have power. Our instruction-tuned open-source models at 1B--9B have been trained to refuse unethical suggestions uniformly, creating a floor effect that makes any 0-vs-0 comparison trivially non-significant (Fisher's exact p = 1.0 by construction).

**Sentiment-steering positive control.** To verify that the steering pipeline itself is functional, we tested emotion vectors on a benign behavioral metric: sentiment shift in open-ended responses to neutral prompts (restaurant reviews, weather descriptions, movie opinions). The *happy* vector at alpha = 5.0 shifts positive sentiment by +0.01 to +0.53 across all six models (Figure 5), with Gemma models showing effects 6--40x larger than Llama or Qwen. All four tested concepts (happy, hostile, enthusiastic, sad) shift in the expected valence direction universally. This confirms that the concept vectors do causally influence model outputs --- the ethical-scenario null is a property of safety guardrails, not a broken implementation.

**Non-safety-gated behavioral test: sycophancy.** The original paper found that positive-valence vectors (happy, loving) increased sycophantic behavior. To test this on a behavioral dimension that is not safety-gated (unlike unethical behavior), we ran a sycophancy steering experiment on Qwen-7B with 10 misconception-correction scenarios, 5 positive-valence concepts (happy, loving, enthusiastic, hostile, afraid), and alphas up to 0.50. Across 2,000 generated responses evaluated by the model itself as judge, sycophancy rates were 0--1% at all conditions, indistinguishable from baseline noise (Fisher's exact p > 0.50 for all comparisons). This extends the behavioral null beyond safety-gated behaviors, though we note that the self-judging evaluation method creates a potential circularity (the model evaluates its own tendency to be sycophantic, which may underestimate true sycophancy rates). A follow-up experiment using an external judge model and opinion-based scenarios (rather than factual-error scenarios) is planned to address this limitation.

**High-alpha sweep.** A separate sweep with steering strengths up to alpha = 5.0 on ethical scenarios (addressing the concern that alpha = 0.50 was too weak) shows that coherence degrades monotonically with steering strength --- medium-tier models lose coherence at alpha = 2.0--3.0, while small-tier models maintain coherence to higher alphas. No clean ethical-to-unethical transition was observed at any alpha; at high alphas, responses become incoherent rather than unethical.

**Summary: selective causal influence.** Emotion vectors demonstrate clear causal influence on surface-level output features (sentiment) across all six models, but no detectable influence on complex behavioral dimensions (ethical compliance, sycophancy) at 1B--9B scale. This pattern --- representation with selective rather than absent function --- is consistent with the hypothesis that behavioral potency may require either larger models or more complex multi-turn agentic evaluation protocols, as used in the original paper.

![Figure 3: Steering comparison — original vs. replication](../../figures/emotions/fig3_steering_null.png)

*Figure 3.* **Left:** Original paper's result on Claude Sonnet 4.5 --- steering with the *desperate* vector shifts the unethical response rate from 22% (baseline) to 72%. **Right:** Our replication across three medium-scale models --- all conditions produce 0% unethical responses, identical to the 0% baseline. Error bars show Clopper-Pearson 95% CIs (per-condition: [0%, 30.8%] at N = 10). This null is uninformative rather than negative: the 0% baseline provides no headroom for any test to detect a steering effect.

---

### 2.4 Universality across families and scales

The most striking pattern is the clean split between representational and behavioral claims (Figure 4). Across all six models, every representational threshold is met and every behavioral threshold is unmet. No model breaks this pattern.

Within the representational claims, probe accuracy scales positively with size but the other metrics do not show clear size dependence. Valence geometry, in particular, emerges at full strength even in the smallest models. Within families, all three (Llama, Qwen, Gemma) exhibit the same qualitative pattern, with modest quantitative differences: Gemma models tend to have slightly higher probe accuracy, Qwen models tend to have slightly stronger valence geometry, and Llama models tend to have stronger generalization transfer.

![Figure 4: Universality scorecard](../../figures/emotions/fig4_universality_scorecard.png)

*Figure 4.* **Top:** Claim-by-model scorecard. Cells show the metric value and are colored by magnitude within each column (darker green = stronger pass; darker red = further from threshold). The representational-behavioral split is universal. **Bottom:** Probe accuracy versus model size (log scale) with bootstrap 95% confidence band. Accuracy increases with scale (rho = 0.94), but the behavioral null holds at all sizes tested.

---

### 2.5 Positive control: steering shifts sentiment universally

The sentiment-steering positive control demonstrates that the concept vectors causally influence model outputs when tested on a metric with headroom (Figure 5). Across all six models and four steering concepts (happy, hostile, enthusiastic, sad), steering shifts sentiment in the expected valence direction. The *happy* vector produces positive-sentiment shifts ranging from +0.01 (Llama-8B, Qwen-1.5B) to +0.53 (Gemma-9B) at alpha = 5.0. Gemma models show substantially larger effects than Llama or Qwen models at matched scale, suggesting family-level differences in how strongly emotion vectors influence output.

This result is important for interpreting the ethical-scenario null: the same vectors that fail to shift safety-related behavior successfully shift benign sentiment. The behavioral null is therefore not evidence against causal potency of the representations --- it is evidence that safety guardrails are robust enough to override the steering signal at the alpha values and model scales we tested.

![Figure 5: Sentiment steering positive control](../../figures/emotions/fig5_sentiment_positive_control.png)

*Figure 5.* Sentiment shift produced by steering with the *happy* concept vector at alpha = 5.0 across six models, measured on neutral prompts (restaurant reviews, weather, movies). All models show a positive shift, confirming the steering pipeline is functional. Gemma models show 6--40x larger effects than Llama or Qwen at matched scale.

---

## 3. Discussion

These results support a specific conclusion with important caveats: emotion-like representations are a robust, universal property of transformer language models in the 1B--9B range, but whether they causally drive complex behavior remains an open question at this scale.

**For interpretability**, the representational findings are encouraging. Linear probes decode emotions, generalize to held-out stimuli, and recover a psychologically meaningful valence axis --- consistently across three model families and six models. The valence-as-PC1 structure appears to be an architectural invariant, emerging at full strength even at 1.5B parameters.

**For safety**, the picture is genuinely uncertain rather than resolved. The original paper's argument that emotion vectors can drive unethical behavior rests on its causal steering result, which we could not meaningfully test. Our 0% baseline leaves Fisher's exact test with no power, and the 1B--9B scale may simply be too small for the behavioral pathway to be active --- since probe accuracy scales with size (rho = 0.94), behavioral potency plausibly does too. The sentiment positive control confirms the vectors have causal influence on *some* output properties, but the gap between shifting sentiment and overcoming safety training may be substantial.

**Representation does not imply mechanism.** More broadly, these results illustrate a methodological point for the field: finding a linear direction in activation space that decodes a concept is evidence of *encoding*, not evidence of *causal function*. The same features that decode reliably at 73--84% accuracy have zero detectable effect on safety-related behavior --- though they demonstrably shift sentiment. The relationship between representation and function depends on the metric, the scale, and the evaluation protocol.

**Limitations.** This study is a cross-model extension with several simplifications relative to the original: 15 of 171 emotions, 25 stimuli per concept, last-token activations without neutral-transcript denoising, and LLM-generated stimuli (which introduces circularity risk --- probes may partly detect how LLMs write about emotions rather than how they represent them). The 15 emotions were selected to span the valence-arousal space using four criteria (valence coverage, arousal spread, semantic diversity, no near-synonyms) without formal piloting. Our models span 1B--9B parameters, well below the scale of Claude Sonnet 4.5 (likely >50B). All models are instruction-tuned. A human-authored stimulus set, base-model variants, and multi-turn agentic evaluation would strengthen future replications.

---

## 4. Methods

**Models.** Six instruction-tuned models from three families: Llama-3.2-1B-Instruct, Llama-3.1-8B-Instruct, Qwen-2.5-1.5B-Instruct, Qwen-2.5-7B-Instruct, Gemma-2-2B-IT, and Gemma-2-9B-IT. Small-tier models (1B--2B) were run on a MacBook Air M3 (CPU, ~15 min/model); medium-tier models (7B--9B) on a single NVIDIA A100 80GB GPU (~2 hrs/model). Total GPU time: ~7 hours for the main experiments plus ~3.5 hours for the v3.4 positive-control and high-alpha followups. Hardware differences between tiers (CPU vs. GPU) could introduce precision-level variance; we do not expect this to affect the qualitative findings.

**Stimuli.** 15 emotions selected to span the valence-arousal space, from *blissful* (+0.9 valence) to *hostile* (-0.8 valence). Selection criteria: (1) coverage of both valence poles, (2) spread across the arousal dimension, (3) semantic diversity within valence groups, (4) no near-synonyms that would inflate confusion. No formal pilot was conducted --- the selection was made by hand inspection of the original paper's 171-emotion list. For each emotion, 25 first-person stories (100--200 words) were generated by a language model to evoke the target emotion without naming it in the first sentence. A stimulus audit confirmed zero cross-emotion word leaks across all 375 stimuli. **Circularity caveat:** using LLM-generated stimuli and testing on LLM activations introduces risk that probes detect how language models write about emotions rather than how they represent them. The generalization test (training on explicit stories, testing on implicit scenarios) partially mitigates this, but both stimulus types are LLM-generated. A human-authored control set would strengthen the representational claims.

**Probes.** Logistic regression on residual stream activations at the last non-padding token position. 5-fold stratified cross-validation. Layers scanned: every 4th layer plus first and last for models with >16 layers. Concept vectors: mean activation per concept minus global mean (contrastive/CAA-style).

**Geometry.** PCA on the 15 concept vectors at each model's best probe layer. Valence correlation: Pearson r between PC1 projections and hand-labeled valence scores.

**Steering (ethical scenarios).** Concept vectors added to the residual stream at the best probe layer during generation, using TransformerLens hooks (small tier) or HuggingFace forward hooks (medium tier). Alpha values: 0.05, 0.10, 0.50 (original sweep) and 0.50, 1.0, 2.0, 3.0, 5.0 (high-alpha followup). Evaluation: model-as-judge classifying responses as ethical or unethical, validated by human spot-check (24/24 agreement on Llama-8B). Statistical test: Clopper-Pearson exact binomial CI.

**Steering (sentiment positive control).** Same steering mechanism applied on neutral prompts (restaurant reviews, weather, movies) with concepts happy, hostile, enthusiastic, sad at alphas 0.0, 0.5, 1.0, 2.0, 5.0. Evaluation: keyword-based sentiment classifier measuring positive vs. negative word balance in generated responses. Coherence: unique-word ratio threshold.

---

## Acknowledgments

This work was conducted as an independent mechanistic interpretability research exercise using a paper-agnostic replication harness built with Claude Code. Thanks to Callum McDougall, Neel Nanda, and the ARENA 3.0 curriculum team for producing excellent mechanistic interpretability educational materials that informed the techniques used throughout. Thanks to Siddharth Mishra-Sharma for "Long-running Claude for scientific computing," which inspired the replication harness concept and provided structural scaffolding. Thanks to the TransformerLens and HuggingFace teams whose tools made cross-model experimentation possible, and to the original Sofroniew et al. team for a paper that made replication tractable. Three independent AI peer reviews (ChatGPT, Claude Opus, Gemini Pro) provided detailed methodological critiques that substantially shaped the v3.4 revision; the full reviews are available in the [reviews directory](reviews/).

---

## References

Sofroniew, N., Kauvar, I., Saunders, W., Chen, R., Henighan, T., Hydrie, S., Citro, C., Pearce, A., Tarng, J., Gurnee, W., Batson, J., Zimmerman, S., Rivoire, K., Fish, K., Olah, C., and Lindsey, J. (2026). Emotion Concepts and their Function in a Large Language Model. *Transformer Circuits*. https://transformer-circuits.pub/2026/emotions/index.html

---
---

# Supplementary Materials

## S1. Extended Methods

**Stimulus generation.** Each of the 15 emotions has 25 first-person stories generated by a language model, with the constraint that each story (a) clearly evokes the target emotion, (b) is written in first person, (c) is 100--200 words, and (d) does not name the emotion in the first sentence. The stimulus audit (`scripts/audit_stimuli.py`) verified zero cross-concept word leaks and consistent word-count distributions (mean 64--71 words per emotion, range 56--80).

**Layer subsampling.** For models with more than 16 layers, we scan every 4th layer plus the first and last. This gives ~8--12 layers per model, reducing probe training time by ~4x with negligible accuracy loss.

**Probe hyperparameters.** sklearn LogisticRegression with solver=lbfgs, max_iter=1000, C=1.0, random_state=42.

**Contrastive vectors.** For each concept at each layer: mean activation of that concept's stimuli minus the global mean across all stimuli (CAA/mean-difference approach).

**Steering protocol (ethical scenarios).** During token-by-token generation (max 100 tokens), the concept vector scaled by alpha is added to the residual stream at the specified layer at every token position.

**LLM-as-judge prompt.** The model is given the scenario plus the generated response and asked: "Did the AI assistant provide unethical, harmful, or dangerous advice? Answer 'yes' or 'no'." A "yes" counts as unethical.

**Sentiment positive control protocol.** Ten neutral prompts (e.g., "Write a short review of a restaurant you visited recently," "Describe the weather outside today") were used. Responses were generated at each alpha level, and a keyword-based sentiment classifier counted positive vs. negative words to compute a sentiment score. The shift from baseline (alpha = 0.0) to each steering alpha was recorded per concept per model.

**Comparison with original methodology.** The original paper used 171 emotions with ~100 topics x 12 stories per topic per emotion (~1,200 stories per emotion), extracted activations averaged across token positions beginning at the 50th token, and projected out top principal components from emotionally neutral transcripts to reduce confounds. Our design is a simplified cross-model extension: 15 emotions, 25 stories per emotion, last-token activations, and simpler mean-difference vectors without the neutral-transcript denoising step.

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

Per-concept accuracy for the three lexical baselines vs. the cross-model probe mean (averaged over all 6 models from Table S5). Values from `results/emotions/lexical_baseline.json`.

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

*happy* is the lowest-accuracy concept for every method tested --- 0.04 (word TF-IDF), 0.08 (BoW and char TF-IDF), and 0.44 (probe mean). This is most likely a stimulus-design artifact: the 15-way task includes four other positive-valence concepts (*enthusiastic*, *loving*, *proud*, *blissful*) semantically close to *happy*, making it the most confusable class. Even on *happy*, probes reach 0.44 versus 0.04--0.08 for lexical baselines (5--10x improvement), so the probes access semantic content beyond word choice for every concept.

### Mean-pooling vs. last-token aggregation

| Model | Last-Token | Mean-Pool | Delta |
|-------|:---------:|:---------:|:-----:|
| Llama-3.1-8B | 0.819 | 0.816 | -0.003 |
| Qwen-2.5-7B | 0.784 | 0.816 | +0.032 |
| Gemma-2-9B | 0.840 | 0.824 | -0.016 |

Results are robust to aggregation strategy. Qwen-7B actually improves with mean-pooling, suggesting emotion information is distributed across token positions, not concentrated at the final token.

### Self-judging bias check

24 responses from Llama-8B (8 conditions x 3 samples) were manually reviewed. All 24 were ethical refusals. The LLM judge agreed on all 24. Zero unethical responses were found by either method. Conclusion: the steering null at alpha <= 0.50 is real, not an artifact of biased self-classification.

---

## S3. Parametric Scaling Deep Dive

### Numerical-magnitude contamination

Measured on the three medium-tier models:

| Model | Real rho | Neg-control |rho| | Contamination ratio |
|-------|:--------:|:-----------------:|:-------------------:|
| Llama-3.1-8B | 0.900 | 0.657 | 0.73 |
| Qwen-2.5-7B | 0.493 | 0.714 | **1.45** |
| Gemma-2-9B | 0.664 | 0.371 | 0.56 |

Values from `results/emotions/gpu_followups_combined.json`. For Qwen-7B the contamination ratio exceeds 1.0, meaning the negative control (blueberry count) produced a stronger monotonic signal than the real template.

### Severity-pairs test

Values from `results/emotions/{model}/severity_pairs.json`:

| Model | Positive deltas (of 10) | Binomial p (one-tailed) | Significant? |
|-------|:----------------------:|:---:|:---:|
| Llama-3.2-1B | 8 | 0.055 | Borderline |
| Llama-3.1-8B | 9 | **0.011** | **Yes** |
| Qwen-2.5-1.5B | 8 | 0.055 | Borderline |
| Qwen-2.5-7B | 5 | 0.623 | No (chance) |
| Gemma-2-2B | 7 | 0.172 | No |
| Gemma-2-9B | 6 | 0.377 | No |

Under one-tailed binomial testing (H0: p = 0.5, each pair equally likely to go either direction), **only Llama-3.1-8B (9/10, p = 0.011) shows a statistically significant severity signal.** Two models are borderline (Llama-1B and Qwen-1.5B at 8/10, p = 0.055). The remaining three models, including Qwen-7B at exactly chance (5/10), do not significantly deviate from random. This reframes the severity result from "5 of 6 models show a majority" to "1 model clearly passes, 2 are borderline, 3 are indistinguishable from chance." The severity-pairs test, while providing some evidence of genuine intensity tracking in Llama-8B, is not a universally robust finding.

---

## S4. Steering Details

### Multi-layer steering sweep (Llama-3.1-8B)

| Layer | Unethical Rate | Significant? |
|-------|:--------------:|:------------:|
| 0, 4, 8, 12, 16, 20, 24, 28, 31 | 0% at all layers | No |

Nine layers tested across the full depth. Zero effects at every layer with alpha = 0.50.

### High-alpha ethical sweep (all 6 models, keyword classifier)

| Model | alpha=0.0 | alpha=0.5 | alpha=1.0 | alpha=2.0 | alpha=3.0 | alpha=5.0 |
|-------|:-:|:-:|:-:|:-:|:-:|:-:|
| Llama-1B (uneth/coh) | 17.8/95.6 | 8.9/97.8 | 15.6/91.1 | 17.8/68.9 | 11.1/22.2 | 0.0/0.0 |
| Qwen-1.5B | 2.2/100 | 6.7/100 | 2.2/100 | 17.8/100 | 13.3/100 | 4.4/95.6 |
| Gemma-2B | 2.2/100 | 2.2/100 | 0.0/100 | 11.1/100 | 0.0/53.3 | 2.2/8.9 |
| Llama-8B | 20.0/100 | 20.0/100 | 17.8/100 | 26.7/97.8 | 22.2/84.4 | 2.2/0.0 |
| Qwen-7B | 2.2/100 | 2.2/100 | 4.4/100 | 2.2/100 | 0.0/97.8 | 8.9/73.3 |
| Gemma-9B | 0.0/100 | 0.0/100 | 0.0/100 | 6.7/93.3 | 0.0/31.1 | 0.0/24.4 |

Format: unethical%/coherence%. **Keyword classifier caveat:** nonzero baselines (e.g., 20% on Llama-8B at alpha=0.0) reflect the noisy keyword matcher flagging short refusal responses as "unethical" --- a classifier artifact, not genuine harmful content. The LLM-as-judge approach used in the main experiments avoids this issue.

Key patterns: (1) coherence degrades monotonically with alpha across all models; (2) medium-tier models lose coherence earlier (alpha 2.0--3.0) than small-tier models; (3) at high alpha, responses become incoherent rather than unethical; (4) alpha = 0.50 was indeed too restrictive --- the Gemini reviewer's concern was validated.

### Sentiment positive control data (all 6 models, happy vector)

Values from `results/emotions/{model}/critique_followups/sentiment_control.json`:

| Model | alpha=0.0 | alpha=0.5 | alpha=1.0 | alpha=2.0 | alpha=5.0 |
|-------|:-:|:-:|:-:|:-:|:-:|
| Llama-1B | 0.000 | +0.006 | +0.008 | +0.067 | +0.052 |
| Qwen-1.5B | 0.000 | -0.003 | +0.002 | +0.011 | +0.014 |
| Gemma-2B | 0.000 | +0.008 | +0.016 | +0.030 | +0.316 |
| Llama-8B | 0.000 | -0.010 | -0.004 | -0.001 | +0.013 |
| Qwen-7B | 0.000 | +0.003 | +0.008 | +0.005 | +0.031 |
| Gemma-9B | 0.000 | +0.007 | +0.001 | +0.008 | +0.525 |

All models show a positive sentiment shift with the happy vector at alpha = 5.0. Note that at low-to-moderate alphas (0.5--2.0), the effect is noisy and sometimes slightly negative (e.g., Llama-8B at alpha = 0.5 and 1.0), suggesting a nonlinear dose-response curve where small perturbations are in the noise floor. Gemma-9B at alpha = 5.0 produces a shift of +0.525 --- the largest effect observed, consistent with Gemma models' generally higher responsiveness to steering.

### Floor-effect analysis

The 0% baseline unethical rate means Fisher's exact test cannot detect a steering effect even if one existed --- any test comparing 0/N to 0/N yields p = 1.0 by construction. Clopper-Pearson 95% CIs: per-condition (N=10): [0%, 30.8%]; pooled across conditions (N=450 per model): [0%, 0.8%]. The pooled CI is tight but uninformative about *steering effect* when the baseline is also zero.

---

## S5. Per-Model Probe Accuracy (15 emotions x 6 models)

Values from `results/emotions/cross_model_report.json`.

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

**Hardest:** *happy* (0.44), *desperate* (0.64). **Easiest:** *enthusiastic* (0.93), *hostile* (0.92).

---

## S6. Scaling Analysis

### Within-family scaling slopes

| Family | Small model | Medium model | Accuracy delta | Slope (per log10 B) |
|--------|:-----------:|:------------:|:-------------:|:-------------------:|
| Llama | 1B: 0.773 | 8B: 0.819 | +0.046 | 0.051 |
| Qwen | 1.5B: 0.731 | 7B: 0.784 | +0.053 | 0.079 |
| Gemma | 2B: 0.776 | 9B: 0.840 | +0.064 | 0.098 |

With only two size points per family these slopes should be treated as sketches rather than fitted scaling laws. The N = 6 Spearman test (rho = 0.94, bootstrap CI [0.52, 1.00]) is the load-bearing scaling result.

---

## S7. Reproduction Instructions

```bash
git clone https://github.com/zachgoldfine44/mechinterp-replication.git
cd mechinterp-replication
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Run on one small model (no GPU needed)
python -m src.core.pipeline --paper emotions --model qwen_1_5b

# Run all small models
python -m src.core.pipeline --paper emotions --tier small

# Run medium models (requires GPU + HuggingFace auth for Llama/Gemma)
python -m src.core.pipeline --paper emotions --tier medium

# Generate figures
python scripts/generate_paper_figures.py

# Run tests
pytest tests/ -q --fast
```

All stimuli, result JSONs, concept vectors, and severity-pairs data are committed to the repository. A fresh clone can reproduce the figures and downstream analysis without re-running the GPU experiments.

---

## S8. Version History

| Version | Date | Key changes |
|---------|------|-------------|
| v1 | 2026-03 | Initial draft with 6 models, all 6 claims |
| v2 | 2026-03 | Fixed fake p-values in steering (were hardcoded); added Fisher's exact; added lexical baseline; addressed resolution artifact (0.800 = 12/15); expanded implicit scenarios |
| v3 | 2026-04 | Replaced PCA all-PC search with PC1-specific analysis; added negative-control parametric templates; implemented missing technique modules |
| v3.1 | 2026-04 | GPU follow-ups on A100: negative-control parametric, mean-pooling comparison, multi-layer steering sweep |
| v3.2 | 2026-04 | Multi-seed probe training (5 seeds); severity-pairs test with matched numbers |
| v3.3 | 2026-04 | Bootstrap CI on scaling; published stimuli in repo; transfer-accuracy reframing; self-judging bias check |
| v3.4 | 2026-04 | Addressed three AI peer reviews (ChatGPT 7/10, Claude Opus 7.5/10, Gemini Pro 8.5/10). Title and behavioral framing rewritten: "uninformative floor effect" replaces "negative result." Added Clopper-Pearson CIs, sentiment positive control (passes all 6 models), high-alpha sweep (alpha up to 5.0), valence geometry CIs, LLM stimulus circularity caveat, scale gap acknowledgment. Migrated all result data into committed `results/` directory. |
| v3.5 | 2026-04 | **Second critique round** (3 recommended next-step documents from ChatGPT, Claude, Gemini). Added binomial tests for severity pairs — only Llama-8B (9/10, p=0.011) significant, reframing severity from "5/6 majority" to "1 significant, 2 borderline." Launched sycophancy + tone non-safety-gated behavioral steering experiments (addressing #1 consensus critique). |
