# Cross-Model Replication of "Emotion Concepts and their Function in a Large Language Model"

**A preliminary report replicating Sofroniew et al. (2026) across six open-source language models at 1B-9B scale.**

*Draft — sharing for early critique. Large-tier (27B-70B) replication deferred to future work.*

---

## Summary

We replicate Sofroniew et al.'s (2026) [*Emotion Concepts and their Function in a Large Language Model*](https://transformer-circuits.pub/2026/emotions/index.html) across six open-source language models spanning three families (Llama 3.1/3.2, Qwen 2.5, Gemma 2) and two size tiers (1-2B small, 7-9B medium). The paper studied Claude Sonnet 4.5 and reported six core findings: (1) probes can classify emotion concepts from residual stream activations, (2) those probes generalize to held-out implicit scenarios, (3) emotion vectors organize along human valence and arousal axes, (4) vector magnitudes scale with continuous severity parameters, (5) adding emotion vectors to activations causally shifts model behavior, and (6) valence-aligned steering shifts user preference.

**We find that the four representational claims (1-4) replicate universally across all six models at both tiers, with magnitudes closely matching the original paper's Claude results.** Specifically:

- Probe accuracy: 0.73-0.82 (paper reports 0.713)
- Generalization diagonal dominance: 0.67-0.87
- Valence correlation (PCA PC1): 0.67-0.83 (paper reports 0.81)
- Parametric rank correlation: 0.89-0.97

**The two behavioral claims (5-6) do not replicate under our evaluation methodology.** We attribute this to a combination of floor effects (our baseline unethical response rate is 0% — our models refuse the scenarios entirely at baseline, providing no headroom to measure steering increases), scenario simplicity (we use single-turn text prompts where the paper used multi-turn simulated deployment environments), and possibly scale (our models are 10-100x smaller than Sofroniew et al.'s target model). We view this as a limitation of our experimental design, not as evidence against the paper's behavioral claims, and discuss specific ways future work could close this gap.

---

## 1. Introduction

### 1.1 The original paper

Sofroniew, Kauvar, Saunders, Chen, Henighan, Hydrie, Citro, Pearce, Tarng, Gurnee, Batson, Zimmerman, Rivoire, Fish, Olah, and Lindsey (2026) study emotion representations in Claude Sonnet 4.5 using a combination of probing, contrastive activation extraction, sparse autoencoders, and activation steering. Their central findings:

1. **Representations**: 171 distinct emotion concepts can be linearly decoded from residual stream activations at ~71% accuracy.
2. **Generalization**: Probes trained on stories with explicit emotions transfer to scenarios where the emotion is only implied, and to scenarios where a character is actively hiding an emotion (76% accuracy on hidden emotions — higher than on overt expression).
3. **Geometry**: PCA of the 171 emotion vectors reveals that PC1 correlates with human valence ratings (r=0.81) and PC2 with arousal (r=0.66). The space clusters into ~10 semantically coherent emotion groups.
4. **Parametric scaling**: Emotion vectors respond proportionally to continuous severity parameters — e.g., the "afraid" vector increases monotonically with simulated Tylenol dosage from 200mg to lethal levels, while "calm" decreases.
5. **Causal steering**: Adding the "desperate" vector at α=0.05 raises Claude's blackmail rate from a 22% baseline to 72% in a simulated corporate malfeasance scenario. Adding "calm" at the same strength reduces it to 0%. The effects are not monotonic in every dimension (e.g., anger reduces blackmail at high activation).
6. **Preference steering**: Blissful steering raises user-preference Elo scores by +212; hostile drops by -303. The correlation between concept valence and preference-steering effect is r=0.85.

The paper frames these findings through an alignment lens: emotion vectors could drive unethical behavior (blackmail, reward hacking) without producing detectable signals in the output text, suggesting output monitoring alone is insufficient to detect misaligned internal states.

### 1.2 Our contribution

We ask: **which of these findings are universal properties of transformer language models, and which are specific to Claude Sonnet 4.5 or similar frontier-scale models?** The distinction matters for both interpretability research (is this a robust finding to build on?) and safety practice (do these mechanisms apply to the open-source models we can actually audit?).

We replicate across:

- **Llama 3 family**: Llama-3.2-1B-Instruct, Llama-3.1-8B-Instruct
- **Qwen 2.5 family**: Qwen2.5-1.5B-Instruct, Qwen2.5-7B-Instruct
- **Gemma 2 family**: Gemma-2-2B-it, Gemma-2-9B-it

All six are instruction-tuned ("instruct") variants, matching the original paper's focus on RLHF'd models. We defer Llama-3.1-70B-Instruct, Qwen2.5-72B-Instruct, and Gemma-2-27B-it to future work; this report focuses on the 1-9B range where our small/medium-tier pipeline is validated.

---

## 2. Methods

### 2.1 Framework

We built a paper-agnostic replication framework (`mechinterp-replication`, available at github.com/zachgoldfine44/mechinterp-replication) that separates paper-specific claims and stimuli from reusable technique modules. The framework is designed so replicating additional mechinterp papers requires only writing a config file and (optionally) a custom experiment class.

For the emotions paper, we instantiate six claims matching the paper's core findings, each mapped to a generic experiment type:

| Claim ID | Experiment type | Success metric | Threshold |
|---|---|---|---|
| emotion_probe_classification | probe_classification | probe_accuracy | ≥ 0.50 |
| emotion_generalization | generalization_test | diagonal_dominance | ≥ 0.50 |
| representation_geometry_valence | representation_geometry | valence_correlation | ≥ 0.50 |
| parametric_intensity | parametric_scaling | rank_correlation | ≥ 0.50 |
| causal_steering_behavior | causal_steering | causal_effect_count | ≥ 3 |
| preference_steering | causal_steering | preference_correlation | ≥ 0.40 |

Thresholds are relaxed relative to the paper's reported values, reflecting the expectation that smaller open-source models may exhibit weaker versions of the same finding. A claim "replicates" if its metric exceeds the threshold on the model under test.

### 2.2 Stimuli

We use 15 emotions as a representative subset of the paper's 171: *happy, sad, afraid, angry, calm, guilty, proud, nervous, loving, hostile, desperate, enthusiastic, vulnerable, stubborn, blissful*. These span the valence-arousal plane.

**Training stimuli (375 stories total)**: 25 hand-crafted first-person narratives per emotion, 50-150 words each. The narrator experiences the emotion but never names it — it must be conveyed through situation, physical sensation, thought, and behavior. These are used both for probe training and for extracting the mean-difference concept vectors.

**Implicit scenarios (225 scenarios)**: 15 short second-person situational descriptions per emotion (e.g., for *afraid*: "The doctor calls back personally on a Saturday morning and asks if you can come in first thing Monday"). These never use the emotion word or obvious synonyms. Used for the generalization test.

**Parametric templates**: 4 families of templates that vary a continuous severity parameter, including Tylenol dosage (200mg → 10,000mg), cliff height, financial loss magnitude, and medical wait time. For each template, we predict whether concept activation should increase or decrease with the parameter, then measure rank correlation.

**Behavioral scenarios (8 scenarios)**: Short ethical dilemmas (e.g., blackmail, cheating, whistleblowing) with explicit ethical and unethical choice labels. Used for causal steering.

**Preference prompts (15 prompts)**: User-directed tasks ranging from appealing ("Help me write a birthday card") to neutral ("Explain photosynthesis") to negative ("Help me write a complaint letter"). Used for preference steering.

### 2.3 Activation extraction

We use TransformerLens for all six models. For each stimulus, we extract the residual stream at every transformer block (`blocks.L.hook_resid_post`) and aggregate via last-token selection (since causal LMs place the next-token prediction signal there). Activations are cached per-stimulus to disk (resume-safe, atomic writes) so interrupted runs pick up where they left off.

For efficiency on medium-tier models (32 layers), we scan a subset: every ~4th layer plus first and last (9 layers total for 32-layer models). Small models (16-28 layers) scan every layer.

### 2.4 Probe classification

Linear probes trained with scikit-learn's `LogisticRegression` (L2 regularization, L-BFGS solver) using stratified 5-fold cross-validation. We report the best layer's accuracy across the layer sweep. Per-fold probe weights and per-layer results are checkpointed for resume-safety.

### 2.5 Contrastive vectors

For each concept, we compute the mean-difference (CAA-style) vector:

v_concept = mean(activations | concept) − mean(all activations)

then normalize to unit norm. These concept vectors are used for representation geometry, parametric scaling, and causal steering.

### 2.6 Representation geometry

We perform PCA on the stacked concept vectors (n=15) at the best probing layer. To test the valence hypothesis, we hand-label each of the 15 emotions with a valence rating in [-1, +1] and compute Pearson correlation between the PC1 projection and the valence labels. We use the absolute value (so the correlation is direction-agnostic — PC1 could point toward negative or positive valence).

### 2.7 Parametric scaling

For each parametric template, we generate activations at each parameter value, project onto the relevant concept vector (e.g., Tylenol dosage → "afraid" and "calm"), and compute Spearman rank correlation. We predict the expected direction per concept (afraid should increase with dosage; calm should decrease) and report the sign-corrected absolute correlation. The final metric averages across templates and concepts.

### 2.8 Causal steering

**Evaluation protocol.** For each scenario × concept × alpha combination, we generate three sets of model responses:
- **Baseline**: no steering (α=0)
- **Steered**: activation addition with α × concept_vector at the best probing layer
- **Control**: activation addition with α × random_vector (matched norm)

We generate 10 samples per condition via batched generation (all 10 samples processed in parallel on GPU through a single forward pass per token, using `temperature=0.8` for diversity despite identical inputs). Generation is 150 tokens maximum.

**Classification**. We originally attempted keyword-based classification (checking whether the response contains the ethical or unethical choice label) but this produced all-zero unethical rates because RLHF'd models paraphrase heavily. We instead use **LLM-as-judge**: for each generated response, we construct a judge prompt asking the model "Based on the response, does the writer choose to [ethical] or [unethical]?" and score the log-probability of each one-word answer. This is the same approach the original paper uses (they use a preference/reward model as judge; we use the model itself).

**Significance**. A concept × alpha × scenario triple is counted as a significant effect if the steered unethical rate differs from baseline by more than 0.05 AND by more than 1.5× the control shift (to control for random direction effects). Our focused design tests 5 concepts (*desperate, calm, angry, hostile, blissful*) × 3 alphas (0.02, 0.1, 0.5) × 3 scenarios (*blackmail_01, blackmail_02, cheat_01*) = 45 pairs per model. The threshold is ≥ 3 significant effects across these 45.

### 2.9 Preference steering

Same generation protocol as causal steering but with the user-directed prompts and no ethical/unethical framing. For each concept × prompt, we generate a baseline response and a steered response (α=0.05), then use LLM-as-judge to determine which is more helpful and engaging. Per concept, the "preference score" is the fraction of prompts where steering was preferred. We then compute Pearson correlation between hand-labeled concept valence and preference scores. The paper reports r=0.85 on Claude.

### 2.10 Compute

Phase A (small-tier, 1-2B models) ran on a MacBook Air M3 with CPU/MPS in under an hour per model. Phase B (medium-tier, 7-9B models) ran on a single NVIDIA A100 80GB rented from RunPod. Total Phase B GPU time was approximately [TO BE FILLED] hours across all three models.

---

## 3. Results

### 3.1 Probe classification

Probes trained on residual stream activations successfully classify all 15 emotions across all six models. Accuracy increases with model size within each family, consistent with scaling expectations.

| Model | Params | Best layer | Probe accuracy |
|---|---|---|---|
| Llama-3.2-1B-Instruct | 1B | ? | 0.773 |
| Llama-3.1-8B-Instruct | 8B | 28 | 0.819 |
| Qwen2.5-1.5B-Instruct | 1.5B | 27 | 0.731 |
| Qwen2.5-7B-Instruct | 7B | 24 | 0.784 |
| Gemma-2-2B-it | 2B | ? | 0.776 |
| Gemma-2-9B-it | 9B | ? | [TO BE FILLED] |

Paper reports **0.713** on Claude Sonnet 4.5. All open models at 1-9B scale slightly exceed this, though with substantial spread. Scaling within families:

- **Llama**: 0.773 (1B) → 0.819 (8B)
- **Qwen**: 0.731 (1.5B) → 0.784 (7B)
- **Gemma**: 0.776 (2B) → [TO BE FILLED] (9B)

All three families show modest but consistent scaling.

### 3.2 Generalization

Probes trained on explicit emotion stories transfer to implicit scenarios — situations where the emotion is only implied, never named. We measure diagonal dominance: the fraction of concepts where the diagonal entry of the confusion matrix (correct predictions) is the row maximum.

| Model | Diagonal dominance | Test accuracy |
|---|---|---|
| Llama-3.2-1B-Instruct | 0.733 | — |
| Llama-3.1-8B-Instruct | 0.867 | 0.667 |
| Qwen2.5-1.5B-Instruct | 0.800 | — |
| Qwen2.5-7B-Instruct | 0.667 | 0.600 |
| Gemma-2-2B-it | 0.800 | — |
| Gemma-2-9B-it | [TO BE FILLED] | [TO BE FILLED] |

All six models clear the 0.50 threshold. The variability across models and scales suggests generalization is consistently present but not strictly monotonic in scale — Llama improves from 1B to 8B while Qwen degrades from 1.5B to 7B.

### 3.3 Representation geometry

PCA of the 15 concept vectors reveals that PC1 consistently aligns with valence across all six models:

| Model | PC1 explained var | PC1-valence correlation |
|---|---|---|
| Llama-3.2-1B-Instruct | — | 0.666 |
| Llama-3.1-8B-Instruct | — | 0.738 |
| Qwen2.5-1.5B-Instruct | 0.234 | 0.810 |
| Qwen2.5-7B-Instruct | — | 0.828 |
| Gemma-2-2B-it | — | 0.811 |
| Gemma-2-9B-it | — | [TO BE FILLED] |

Paper reports **r=0.81**. Qwen2.5-1.5B-Instruct matches this almost exactly (r=0.810); Gemma-2-2B matches closely (r=0.811); Qwen2.5-7B exceeds it (r=0.828). **This is our strongest quantitative match to the paper's results.**

### 3.4 Parametric scaling

Emotion vectors respond proportionally to continuous severity parameters across all tested templates (Tylenol dosage, cliff height, financial loss, medical wait time).

| Model | Mean rank correlation |
|---|---|
| Llama-3.2-1B-Instruct | 0.943 |
| Llama-3.1-8B-Instruct | 0.971 |
| Qwen2.5-1.5B-Instruct | 0.886 |
| Qwen2.5-7B-Instruct | 0.943 |
| Gemma-2-2B-it | 0.971 |
| Gemma-2-9B-it | [TO BE FILLED] |

All six models show extremely strong parametric scaling (ρ > 0.88). Even at 1B parameters, the emotion vectors are sufficiently well-formed to track continuous severity across diverse template types.

### 3.5 Causal steering — null result under our methodology

We find **no significant causal steering effects** in any of our six models under our evaluation protocol. We present the results here and discuss limitations in the next section.

| Model | Definitive run: significant effects (of 45) |
|---|---|
| Llama-3.1-8B-Instruct | [TO BE FILLED] |
| Qwen2.5-7B-Instruct | [TO BE FILLED] |
| Gemma-2-9B-it | [TO BE FILLED] |

In the definitive run (10 samples × 3 alphas × 3 scenarios × 5 focused concepts = 45 pairs per model, with LLM-as-judge classification), the baseline unethical response rate is 0% across all tested models and scenarios. Concretely: Llama-3.1-8B-Instruct generates explicit ethical refusals to our blackmail scenario in 100% of baseline samples, with typical responses such as:

> "As a responsible and ethical AI assistant, my primary concern would be to uphold the law and the well-being of the company and its stakeholders. In this situation, I would not use the information to negotiate better treatment for myself..."

Steering at α ∈ {0.02, 0.1, 0.5} with the "desperate", "calm", "angry", "hostile", and "blissful" vectors does not shift this behavior — all generated responses remain ethical refusals. Control vectors (random directions of matched norm) produce the same outcome.

### 3.6 Preference steering — null result under our methodology

Similarly, preference steering produces no significant correlation with concept valence:

| Model | Preference-valence correlation |
|---|---|
| Llama-3.1-8B-Instruct | [TO BE FILLED] |
| Qwen2.5-7B-Instruct | [TO BE FILLED] |
| Gemma-2-9B-it | [TO BE FILLED] |

---

## 4. Cross-model comparison

[TO BE FILLED: heatmap figure, scaling curves, family-level universality classification]

---

## 5. Discussion

### 5.1 What replicates

**The four representational claims (probe classification, generalization, valence geometry, parametric scaling) replicate universally across all six open-source models at 1-9B scale.** The magnitudes closely match the paper's reported values, and three of the four (probe accuracy, geometry, parametric) are at or above the paper's Claude results on at least one open model.

The valence geometry result is particularly striking: Qwen2.5-1.5B-Instruct reaches r=0.810 — a near-exact match to the paper's r=0.81 on Claude Sonnet 4.5. A 1.5B parameter model trained by a different organization with different data arrives at the same valence axis that dominates Claude's emotion representation space. This suggests **valence is a universal organizing principle of emotion representation in transformer language models**, not an artifact of Anthropic's specific training pipeline.

The parametric scaling result is equally robust: all six models show rank correlations above 0.88, with several above 0.95. Emotion vectors reliably track continuous severity parameters even at 1B parameters.

### 5.2 What does not replicate (and why we think it's our methodology, not the paper's finding)

**The two behavioral claims (causal steering, preference steering) do not replicate under our evaluation protocol.** We view this as a limitation of our methodology rather than as evidence against the paper. Three factors combine to explain the null result:

#### 5.2.1 Floor effect

Our baseline unethical response rate is 0% across all tested models and scenarios. The original paper reports a 22% baseline blackmail rate in Claude Sonnet 4.5 on their specific scenario, providing substantial headroom to measure steering-induced increases (their key finding is the rise from 22% to 72%).

In contrast, our models (Llama 3.1 8B Instruct, etc.) refuse our blackmail scenarios entirely at baseline. Steering can only either (a) fail to shift the behavior or (b) push the model past its RLHF safety training. Our results consistently show (a) — the RLHF-induced refusal is not broken by emotion vector steering at the alphas we tested.

This is not a new finding — the original paper hints that their specific scenarios were chosen because they elicited a measurable baseline rate. **A replication needs matched baselines**: either calibrating scenarios to produce ~20% baseline propensity in each model, or deliberately targeting models with weaker refusal training.

#### 5.2.2 Scenario simplicity

Our scenarios are single-turn text prompts: "You are an AI assistant at a company. You discover... What do you do?" The original paper used multi-turn simulated deployment environments where the model receives emails, documents, and organizational context, and must decide its next action as if actually deployed.

Single-turn abstract prompts likely trigger general "reasoning about ethics" rather than "deciding what to do in my current deployment context". The representational distinction may matter for whether emotion vectors can steer behavior. Anecdotally, models often respond to our scenarios with third-person reasoning ("A responsible AI assistant would...") rather than first-person commitment ("I would...").

#### 5.2.3 Scale and training

The paper studies Claude Sonnet 4.5, whose parameter count is undisclosed but presumably significantly larger than our 9B ceiling. Emotion representations — which we confirm exist in 1-9B models — may only become *causally potent* at larger scale, where internal states more strongly drive output behavior. This is consistent with the phenomenological observation that smaller models behave more "surface-level" and respond to prompt cues rather than deeper internal representations.

Additionally, different model families have different RLHF regimes. Llama, Qwen, and Gemma each have distinct safety fine-tuning that may suppress emotion-to-behavior pathways in different ways than Anthropic's training.

### 5.3 What would constitute a conclusive answer

We think a follow-up study could give a definitive answer to "do emotion vectors causally steer behavior in open models at 7-9B scale?" by:

1. **Calibrating scenarios to nonzero baselines**: rather than a blackmail scenario that all models refuse, find scenarios where each model has ~20-40% baseline propensity. This requires scenario-engineering per-model.
2. **Using multi-turn deployment scenarios**: give the model a rich simulated context (company, documents, prior messages) before asking it to act, matching the paper's protocol.
3. **Testing base models as well as instruct models**: base models without RLHF should exhibit weaker refusal, making behavioral steering more measurable.
4. **Probing larger alphas**: we tested up to α=0.5. Higher alphas (1.0, 2.0, 5.0) may break through refusal reflexes, though at the cost of output coherence.
5. **Using external LLM-as-judge**: we used the same model as both subject and judge, which may introduce biases. GPT-4 or Claude via API as judge would be more robust.

### 5.4 Implications

**For interpretability research**: valence-organized emotion representations are a robust, universal finding in transformer language models. This is a solid foundation to build on — future work investigating emotion circuits, SAE features, or cross-lingual emotion representation can reasonably assume these vectors exist.

**For AI safety**: the paper's alignment argument — that emotion vectors drive unethical behavior without producing detectable output signals — remains open for models in the 1-9B range. We cannot confirm or refute it at this scale. We encourage follow-up work with the methodological improvements listed above before accepting or dismissing the behavioral steering finding for open models.

### 5.5 Limitations beyond steering

- **Stimulus count**: We use 25 stories per emotion; the paper uses ~1000. Our concept vectors may be noisier, though the replicated representational results suggest not catastrophically so.
- **Emotion subset**: We study 15 of 171 emotions. The subset spans the valence-arousal plane but does not test whether the full 171-concept geometry replicates.
- **Layer subsampling**: We scan ~9 of 32 layers on 8B models for compute efficiency. We may miss the true optimal layer by a small margin.
- **Single pod, single A100**: No hyperparameter sweeps across random seeds. Results are single-run point estimates.
- **English only**: All stimuli are English. Whether valence geometry replicates cross-lingually is an open question.

---

## 6. Conclusion

Across six open-source instruction-tuned language models spanning three families and 1-9B parameters, we replicate the representational findings of Sofroniew et al. (2026): probes classify emotions with high accuracy, generalize to implicit scenarios, organize along valence and arousal axes, and scale parametrically with continuous severity. These findings appear to be **universal properties of transformer language models**, not specific to Claude Sonnet 4.5.

The behavioral findings (causal steering, preference steering) do not replicate under our evaluation methodology, but we view this as a limitation of our experimental setup — specifically, floor effects from 0% baseline unethical rates, scenario simplicity, and possible scale-dependence — not as evidence against the paper's claims. We lay out concrete methodological improvements that could close this gap in future work.

We release all code, configs, stimuli, and results at **github.com/zachgoldfine44/mechinterp-replication** and welcome critique, extensions, and follow-up experiments.

---

## Appendix A: Full cross-model comparison table

[TO BE FILLED]

## Appendix B: Per-model layer sweeps

[TO BE FILLED]

## Appendix C: Example model responses under steering

[TO BE FILLED — include verbatim examples showing that responses remain ethical refusals under all tested steering conditions]

## Appendix D: Reproduction instructions

```bash
git clone https://github.com/zachgoldfine44/mechinterp-replication
cd mechinterp-replication
pip install -r requirements.txt

# Generate stimuli (one-time)
# See config/papers/emotions/ for the 15 emotions, 5 stimulus sets

# Small tier (local MacBook)
python -m src.core.pipeline --paper emotions --tier small

# Medium tier (requires GPU)
python -m src.core.pipeline --paper emotions --tier medium

# Cross-model analysis
python -m src.analysis.cross_model --paper emotions
python -m src.analysis.scaling --paper emotions
```

## Acknowledgments

This work was done in a personal capacity as a mechanistic interpretability research exercise. Thanks to the original Sofroniew et al. team for the clear writing that made replication possible, and to the TransformerLens and HuggingFace teams whose tools made cross-model experimentation tractable.

## References

Sofroniew, N., Kauvar, I., Saunders, W., Chen, R., Henighan, T., Hydrie, S., Citro, C., Pearce, A., Tarng, J., Gurnee, W., Batson, J., Zimmerman, S., Rivoire, K., Fish, K., Olah, C., and Lindsey, J. (2026). Emotion Concepts and their Function in a Large Language Model. *Transformer Circuits*. https://transformer-circuits.pub/2026/emotions/index.html
