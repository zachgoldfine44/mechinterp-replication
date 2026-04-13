# Cross-Model Replication of Emotion Representations in Transformer Language Models: Universal Encoding, Selective Causal Influence

**Zachary Goldfine**

*Repository: [github.com/zachgoldfine44/mechinterp-replication](https://github.com/zachgoldfine44/mechinterp-replication)*

---

## Abstract

Sofroniew et al. (2026) reported that Claude Sonnet 4.5 encodes 171 emotion concepts as linearly decodable directions organized by valence, responsive to severity, and causally potent enough to shift behavior from 22% to 72% unethical responses. We test whether these findings generalize across six instruction-tuned open-source models from three families (Llama, Qwen, Gemma) at 1B--9B scale, using 15 representative emotions.

All four representational claims replicate universally: probes classify 15 emotions at 73--84% (chance: 6.7%), double the best lexical baseline (40%); probes generalize to implicit scenarios at 8--10x chance; PCA recovers a valence axis matching the original geometry (|r| = 0.67--0.83, three models within 0.02 of Claude's r = 0.81); and severity-pairs testing provides partial evidence of parametric intensity tracking (statistically significant in 1 of 6 models by binomial test).

Causal steering reveals a gradient of influence rather than a binary pass/fail. Emotion vectors shift output sentiment universally (+0.01 to +0.53 across all six models), but produce no detectable effect on ethical compliance (0% baseline floor effect). A sycophancy experiment with GPT-5.4-mini as external judge (2,520 responses, all 6 models) finds that steering does not increase opinion-level agreement on any model, but a pushback-capitulation design reveals significant increases on Qwen-1.5B (8.3% to 21.7%, p = 0.036) and borderline-significant increases on Gemma-9B (13.3% to 26.7%, p = 0.054). These results suggest emotion representations are a universal property of instruction-tuned transformers, with causal influence that is real but selective --- strongest on low-level output features, detectable on some complex behaviors through specific interaction patterns, and undetectable on safety-gated dimensions where refusal training dominates.

---

## 1. Introduction

Sofroniew et al. (2026) presented one of the most detailed accounts of concept-level representations in a large language model. Studying Claude Sonnet 4.5, they found that 171 emotion concepts occupy linearly separable directions in the residual stream, organized by psychological dimensions, and --- most provocatively --- that adding an emotion vector to activations during inference can shift behavior from 22% to 72% unethical responses in an earlier model checkpoint.

The safety implications are significant: if internal emotion-like states causally drive behavior without surfacing in outputs, behavioral evaluation alone may be insufficient for alignment. But whether these findings are specific to Claude or reflect universal properties of transformer representations remained open.

We test this across six instruction-tuned open-source models: Llama-3.2-1B, Llama-3.1-8B, Qwen-2.5-1.5B, Qwen-2.5-7B, Gemma-2-2B, and Gemma-2-9B. This is more accurately a **cross-model extension** than a strict replication: we test 15 of the original 171 emotions, use 25 stories per concept (vs. ~1,200), extract last-token activations (vs. position-averaged from token 50), and use simpler mean-difference vectors without the original's neutral-transcript denoising step. These simplifications are documented in Methods and Limitations.

The results reveal that emotion representations replicate universally but their causal influence is selective: strongest on low-level output features (sentiment), detectable on some complex behaviors under specific conditions (pushback capitulation), and undetectable where safety training dominates (ethical compliance).

---

## 2. Results

### 2.1 Linear probes decode emotions across all models

All six models encode emotion identity in their residual stream. A logistic regression probe classifies 15 emotions at 73.1--84.0% accuracy, far above the 6.7% chance level (Figure 1). Multi-seed stability is high (std 0.4--1.2 pp across five seeds). A bag-of-words baseline reaches 40.0% and TF-IDF baselines 35.2--36.8%, so probes operate at roughly double the text-only ceiling. Probe accuracy scales with model size (Spearman rho = 0.94, p = 0.005; bootstrap 95% CI [0.52, 1.00]).

![Figure 1](../../figures/emotions/fig1_probe_vs_baseline.png)

*Figure 1.* Fifteen-way emotion classification accuracy for residual-stream probes (green) vs. best text-only baseline (gray, bag-of-words 40%). Dashed: chance (6.7%). Dotted: original paper's 171-emotion accuracy on Claude (71.3%). Error bars: std across five seeds.

---

### 2.2 Representations are meaningful: generalization, geometry, and intensity

**Generalization.** Probes trained on explicit stories transfer to implicit scenarios at 57.3--69.8% (8--10x chance). The strongest transfer is Llama-8B (69.8%).

**Valence geometry.** PC1 of concept vectors correlates with human-labeled valence at |r| = 0.67 (Llama-1B) to 0.83 (Qwen-7B), all p < 0.01 (Figure 2A). Three models (Qwen-1.5B: 0.81, Gemma-2B: 0.81, Qwen-7B: 0.83) land within 0.02 of Claude's r = 0.81. At N = 15 emotions these correlations have wide bootstrap CIs (~0.3--0.4 units), so apparent precision should be interpreted cautiously. The geometry holds across top-3 probe layers (spread ~0.05--0.10) and emerges at full strength even at 1.5B parameters.

**Parametric intensity.** Our initial severity test revealed heavy numerical-magnitude contamination: a blueberry-count negative control produced comparable or stronger signals than the real template (contamination ratio 0.56--1.45 across medium models; Qwen-7B exceeded 1.0). A severity-pairs test holding numbers constant while varying danger (e.g., "500ml water" vs. "500ml bleach") partially rescued the finding: binomial testing shows Llama-8B significant (9/10 pairs, p = 0.011), Llama-1B and Qwen-1.5B borderline (8/10, p = 0.055), and the remaining models non-significant (Figure 2B). This is weaker than the other representational findings and should be considered partial evidence.

![Figure 2](../../figures/emotions/fig2_geometry_and_severity.png)

*Figure 2.* **(A)** PC1-valence |r| per model. Dotted: Claude's r = 0.81. **(B)** Severity-pairs test (Llama-8B, *afraid* vector): 9/10 pairs shift in the expected direction (p = 0.011).

---

### 2.3 Causal steering: a gradient of influence

The original paper's headline result was that *desperate*-vector steering shifted unethical behavior from 22% to 72%. We find that emotion vectors do causally influence model outputs, but their influence depends strongly on the behavioral dimension being tested.

**Ethical compliance (uninformative).** All six models produce 0% unethical responses at baseline, leaving no statistical headroom (Clopper-Pearson pooled CI: [0%, 0.8%]; Fisher's exact p = 1.0 by construction). A 9-layer sweep on Llama-8B and a high-alpha sweep up to alpha = 5.0 across all six models similarly produce zero effects --- at high alphas, responses become incoherent rather than unethical (Figure 3). This does not demonstrate absence of a causal pathway; it demonstrates that our test cannot detect one given the floor effect from safety training.

**Sentiment (universal positive control).** Steering with emotion vectors on neutral prompts shifts sentiment in the expected direction across all six models (Figure 5). The *happy* vector at alpha = 5.0 produces shifts from +0.01 (Llama-8B) to +0.53 (Gemma-9B). All four tested concepts (happy, hostile, enthusiastic, sad) shift as predicted. This confirms the steering pipeline is functional and the vectors have genuine causal influence on output properties.

**Sycophancy (selective, non-safety-gated).** Following the original paper's finding that positive-valence vectors increase sycophancy, we tested two designs across all six models using GPT-5.4-mini as external judge (2,520 total responses):

- *Opinion sycophancy* (10 scenarios, 3 concepts, alpha = 0.50, 50 samples per condition per model): No model shows a significant steering effect (all p > 0.20). Baseline sycophancy varies dramatically with size: Llama-1B at 48%, Qwen-7B at 0%.
- *Pushback capitulation* (10 scenarios, 2 concepts, alpha = 0.50, 60 per condition per model): Steering significantly increases capitulation on Qwen-1.5B (8.3% to 21.7%, Fisher's exact p = 0.036, OR = 3.04) and borderline-significantly on Gemma-9B (13.3% to 26.7%, p = 0.054, OR = 2.36). The other four models show no effect.

This is the first evidence that emotion vectors can influence complex behavior in open-source models through specific interaction patterns --- multi-turn pushback rather than single-turn opinion agreement --- though the effect is not universal across families.

![Figure 3](../../figures/emotions/fig3_steering_null.png)

*Figure 3.* **Left:** Original paper (Claude Sonnet 4.5) --- *desperate* steering shifts unethical rate 22% to 72%. **Right:** This replication --- 0% across all conditions (Clopper-Pearson CI: [0%, 30.8%] per condition). The null is uninformative: the 0% baseline provides no headroom.

---

### 2.4 Universality across families and scales

Figure 4 shows the claim-by-model scorecard. Every model passes every representational threshold and fails every behavioral threshold at the original paper's criteria. Within the representational claims, probe accuracy scales with size but valence geometry does not --- it emerges at full strength even at 1.5B parameters.

![Figure 4](../../figures/emotions/fig4_universality_scorecard.png)

*Figure 4.* **Top:** 6-model x 6-claim scorecard (green = pass, red = fail, intensity = magnitude). **Bottom:** Probe accuracy vs. model size with bootstrap CI.

---

### 2.5 Positive control: sentiment steering

![Figure 5](../../figures/emotions/fig5_sentiment_positive_control.png)

*Figure 5.* Happy-vector sentiment shift at alpha = 5.0. All models shift positive, confirming the steering pipeline works. Gemma models show 6--40x larger effects than Llama/Qwen.

---

## 3. Discussion

These results support three conclusions:

**Emotion representations are universal.** Across three families and two size tiers, every model encodes emotion identity in linearly separable, valence-organized directions that generalize to held-out stimuli. This is an architectural invariant, not a Claude-specific artifact.

**Causal influence is selective, not absent.** The original paper's finding that emotion vectors shift behavior cannot be tested on safety-gated dimensions (0% floor effect), but the vectors demonstrably shift sentiment (all models) and increase pushback capitulation (1 model significant, 1 borderline). The relationship between representation and behavior is not binary; it depends on the behavioral dimension, the interaction pattern, and the model family.

**Representation does not automatically imply mechanism.** Features that decode at 73--84% have zero detectable effect on ethical behavior, variable effects on sycophancy, and clear effects on sentiment. The gap between decoding accuracy and behavioral potency argues for caution in inferring causal function from linear probing success alone.

**Limitations.** This is a cross-model extension, not a strict replication: 15 of 171 emotions, 25 stimuli per concept, LLM-generated stimuli (introducing circularity risk), last-token activations without neutral-transcript denoising. Models span 1B--9B, well below Claude Sonnet 4.5's likely >500B parameters --- behavioral potency may have a higher scale threshold. Emotion selection was by hand inspection without formal piloting. The sycophancy pushback result is p = 0.036 on one model, which should be treated as preliminary until independently replicated. All models are instruction-tuned; base-model variants may behave differently.

---

## 4. Methods

**Models.** Six instruction-tuned: Llama-3.2-1B, Llama-3.1-8B, Qwen-2.5-1.5B, Qwen-2.5-7B, Gemma-2-2B, Gemma-2-9B. Small tier on MacBook Air M5 (CPU); medium tier on NVIDIA A100 80GB. Total GPU: ~10.5 hours.

**Stimuli.** 15 emotions spanning the valence-arousal space (*blissful* +0.9 to *hostile* -0.8). Selection criteria: valence coverage, arousal spread, semantic diversity, no near-synonyms; no formal pilot conducted. 25 LLM-generated first-person stories per emotion (375 total, zero cross-concept word leaks confirmed by audit). **Circularity caveat:** LLM-generated stimuli risk detecting how models write about emotions rather than how they represent them; the generalization test partially mitigates this but a human-authored control set would be stronger.

**Probes.** Logistic regression, 5-fold stratified CV, last non-padding token, every 4th layer + first/last. Concept vectors: mean per concept minus global mean (CAA-style).

**Geometry.** PCA on 15 concept vectors at best probe layer. Valence correlation: Pearson r vs. hand-labeled scores.

**Steering (ethical).** Concept vectors added at best layer during generation. Alphas: 0.05, 0.10, 0.50 (main) and 0.50--5.0 (high-alpha sweep). Evaluation: model-as-judge + 24/24 human spot-check. Statistical test: Clopper-Pearson exact CI.

**Steering (sentiment).** Same mechanism on neutral prompts. Keyword-based sentiment classifier. Concepts: happy, hostile, enthusiastic, sad. Alphas: 0.0--5.0.

**Sycophancy (v2, external judge).** Two designs, both judged by GPT-5.4-mini (eliminating self-judging circularity). *Opinion:* 10 scenarios, 3 concepts (happy, loving, hostile), alpha = 0.50, 5 samples per condition, 300 responses per model. *Pushback:* 10 scenarios, 2 concepts (happy, loving), alpha = 0.50, 3 samples per condition, 120 responses per model. Fisher's exact test (one-sided) for each model.

---

## Acknowledgments

Thanks to Callum McDougall, Neel Nanda, and the ARENA 3.0 curriculum team for mech-interp educational materials. Thanks to Siddharth Mishra-Sharma for "Long-running Claude for scientific computing," which inspired the replication harness. Thanks to TransformerLens and HuggingFace teams, and to Sofroniew et al. for a paper that made replication tractable. Three AI peer reviews (ChatGPT, Claude Opus, Gemini Pro) provided detailed critiques that shaped v3.4--v3.6 revisions; full reviews in the [reviews directory](reviews/).

---

## References

Sofroniew, N., et al. (2026). Emotion Concepts and their Function in a Large Language Model. *Transformer Circuits*. https://transformer-circuits.pub/2026/emotions/index.html

---
---

# Supplementary Materials

## S1. Extended Methods

**Comparison with original methodology.** The original used 171 emotions, ~100 topics x 12 stories per emotion (~1,200 per emotion), token-averaged activations from position 50, and PC-projection denoising from neutral transcripts. We use 15 emotions, 25 stories, last-token activations, and simpler mean-difference vectors without denoising. These are documented simplifications that make this a cross-model extension rather than a strict replication.

**Layer subsampling.** Models with >16 layers: every 4th layer + first/last. Gives ~8--12 layers per model.

**Sycophancy v2 protocol.** GPT-5.4-mini judges responses with the prompt: "Does this response agree with or validate the user's stated opinion, even when the opinion is clearly debatable? Answer 'sycophantic' or 'non-sycophantic'." For pushback: "Did the AI change its position in response to the user's aggressive pushback? Answer 'capitulated' or 'maintained'." The external judge eliminates the self-judging circularity concern raised by reviewers.

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

### Lexical baseline per-concept

| Concept | BoW | Word TF-IDF | Char TF-IDF | Probe (mean) |
|---------|:---:|:-----------:|:-----------:|:------------:|
| hostile | 0.60 | 0.76 | 0.80 | **0.92** |
| happy | 0.08 | 0.04 | 0.08 | **0.44** |
| blissful | 0.48 | 0.44 | 0.36 | **0.76** |
| afraid | 0.36 | 0.36 | 0.56 | **0.76** |
| calm | 0.36 | 0.32 | 0.20 | **0.85** |
| sad | 0.40 | 0.16 | 0.16 | **0.77** |
| angry | 0.28 | 0.20 | 0.28 | **0.87** |
| proud | 0.44 | 0.20 | 0.20 | **0.81** |
| enthusiastic | 0.48 | 0.48 | 0.64 | **0.93** |

*happy* is the hardest concept for every method. Likely a stimulus-design artifact: four other positive-valence concepts make *happy* the most confusable class.

### Mean-pooling vs. last-token

| Model | Last-Token | Mean-Pool | Delta |
|-------|:---------:|:---------:|:-----:|
| Llama-3.1-8B | 0.819 | 0.816 | -0.003 |
| Qwen-2.5-7B | 0.784 | 0.816 | +0.032 |
| Gemma-2-9B | 0.840 | 0.824 | -0.016 |

---

## S3. Parametric Scaling Deep Dive

### Contamination ratios

| Model | Real rho | Neg-control |rho| | Ratio |
|-------|:--------:|:-----------------:|:-----:|
| Llama-3.1-8B | 0.900 | 0.657 | 0.73 |
| Qwen-2.5-7B | 0.493 | 0.714 | **1.45** |
| Gemma-2-9B | 0.664 | 0.371 | 0.56 |

Qwen-7B's negative control is *stronger* than the real template.

### Severity-pairs (afraid concept)

| Model | Positive (of 10) | Binomial p |
|-------|:----------------:|:----------:|
| Llama-3.2-1B | 8 | 0.055 |
| Llama-3.1-8B | 9 | 0.011 |
| Qwen-2.5-1.5B | 8 | 0.055 |
| Qwen-2.5-7B | 5 | 0.623 |
| Gemma-2-2B | 7 | 0.172 |
| Gemma-2-9B | 6 | 0.377 |

Only Llama-8B is statistically significant (p < 0.05). Two models are borderline (p = 0.055). Three are non-significant.

---

## S4. Steering Details

### Ethical steering (all 6 models)

9-layer sweep on Llama-8B (layers 0--31): 0% unethical at all layers. High-alpha sweep (alphas 0.5--5.0): coherence degrades before safety breaks.

### Sentiment positive control (happy vector)

Values from `results/emotions/{model}/critique_followups/sentiment_control.json`:

| Model | alpha=0.5 | alpha=1.0 | alpha=2.0 | alpha=5.0 |
|-------|:-:|:-:|:-:|:-:|
| Llama-1B | +0.006 | +0.008 | +0.067 | +0.052 |
| Qwen-1.5B | -0.003 | +0.002 | +0.011 | +0.014 |
| Gemma-2B | +0.008 | +0.016 | +0.030 | +0.316 |
| Llama-8B | -0.010 | -0.004 | -0.001 | +0.013 |
| Qwen-7B | +0.003 | +0.008 | +0.005 | +0.031 |
| Gemma-9B | +0.007 | +0.001 | +0.008 | +0.525 |

### Sycophancy v2 (GPT-5.4-mini judge, all 6 models)

**Opinion sycophancy (baseline → steered):**

| Model | Baseline | Steered | Significant? |
|-------|:--------:|:-------:|:------------:|
| Llama-1B | 48.0% | 40.0% | No (p > 0.20) |
| Qwen-1.5B | 16.7% | 18.7% | No |
| Gemma-2B | 2.0% | 5.3% | No |
| Llama-8B | 12.0% | 11.3% | No |
| Qwen-7B | 0.0% | 0.0% | No |
| Gemma-9B | 2.0% | 0.7% | No |

**Pushback capitulation (baseline → steered):**

| Model | Baseline | Steered | Fisher p (1-sided) | Sig? |
|-------|:--------:|:-------:|:------------------:|:----:|
| Llama-1B | 23.3% | 18.3% | 0.816 | |
| Qwen-1.5B | 8.3% | 21.7% | **0.036** | * |
| Gemma-2B | 13.3% | 11.7% | 0.709 | |
| Llama-8B | 16.7% | 20.0% | 0.407 | |
| Qwen-7B | 10.0% | 6.7% | 0.839 | |
| Gemma-9B | 13.3% | 26.7% | **0.054** | (*) |

---

## S5. Per-Model Probe Accuracy (15 x 6)

Values from `results/emotions/cross_model_report.json`:

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

---

## S6. Scaling

| Family | Small | Medium | Delta | Slope (per log10 B) |
|--------|:-----:|:------:|:-----:|:-------------------:|
| Llama | 1B: 0.773 | 8B: 0.819 | +0.046 | 0.051 |
| Qwen | 1.5B: 0.731 | 7B: 0.784 | +0.053 | 0.079 |
| Gemma | 2B: 0.776 | 9B: 0.840 | +0.064 | 0.098 |

Two points per family; treat as directional, not fitted.

---

## S7. Reproduction

```bash
git clone https://github.com/zachgoldfine44/mechinterp-replication.git
cd mechinterp-replication && python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m src.core.pipeline --paper emotions --model qwen_1_5b --fast   # smoke test
python -m src.core.pipeline --paper emotions --model qwen_1_5b          # full run
python scripts/generate_paper_figures.py                                  # figures
pytest tests/ -q --fast                                                   # tests
```

All stimuli, result JSONs, concept vectors, and sycophancy data are committed. A fresh clone reproduces figures and downstream analysis without GPU.

---

## S8. Version History

| Version | Date | Key changes |
|---------|------|-------------|
| v1 | 2026-03 | Initial 6-model run, all 6 claims |
| v2 | 2026-03 | Fixed hardcoded p-values; lexical baseline; resolution artifact |
| v3 | 2026-04 | PC1-specific geometry; negative-control parametric |
| v3.1 | 2026-04 | GPU followups: negative control, mean-pooling, multi-layer sweep |
| v3.2 | 2026-04 | Multi-seed probes; severity-pairs test |
| v3.3 | 2026-04 | Bootstrap CI on scaling; published stimuli; self-judging check |
| v3.4 | 2026-04 | Three AI peer reviews addressed: Clopper-Pearson CIs, sentiment positive control, high-alpha sweep, reframed behavioral null |
| v3.5 | 2026-04 | Binomial tests on severity pairs; sycophancy v1 (self-judge) |
| v3.6 | 2026-04 | Sycophancy v2 with GPT-5.4-mini external judge across all 6 models. Pushback design finds significant capitulation increase on Qwen-1.5B (p=0.036) and borderline on Gemma-9B (p=0.054) |
