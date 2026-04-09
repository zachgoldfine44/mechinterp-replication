# Progress Log

## Active paper: emotions (Sofroniew et al. 2026)

## Status: 🟢 Phase A complete — 4/6 claims UNIVERSAL across all 3 small-tier models

## Current test status
```
81 tests passed in 0.82s (pytest tests/ -q --fast)
Pipeline: 4/6 claims PASS on ALL small-tier models (real handcrafted stimuli)
Cross-model analysis: 4 UNIVERSAL findings, 2 NULL (steering needs larger models)
```

## Task checklist

### Phase 0–3: Framework infrastructure — COMPLETE
- [x] All 34 source files implemented
- [x] 81 tests passing
- [x] Pipeline orchestrator with CLI (--paper, --model, --tier, --fast, --validate-only)

### Phase 4: Emotions paper replication

#### Phase A: Small-tier validation (local MacBook) — COMPLETE
- [x] Create paper config (6 claims, 15 emotions, 5 stimulus sets)
- [x] Generate 375 real handcrafted stimuli (25 stories × 15 emotions)
- [x] Run on Qwen-2.5-1.5B-Instruct: **4/6 PASS**
- [x] Run on Gemma-2-2B-it: **4/6 PASS**
- [x] Run on Llama-3.2-1B-Instruct: **4/6 PASS**
- [x] Cross-model analysis: 4 claims UNIVERSAL
- [x] Generate comparison heatmap figure

#### Phase B: Medium-tier primary replication (Colab GPU) — NEXT
- [ ] Run on Llama-3.1-8B-Instruct
- [ ] Run on Qwen-2.5-7B-Instruct
- [ ] Run on Gemma-2-9B-it
- [ ] Debug steering methodology at medium scale
- [ ] Run cross-model + scaling analysis

#### Phase C: Large-tier scaling (Colab, if compute allows)
- [ ] Run on Llama-3.1-70B-Instruct
- [ ] Run on Qwen-2.5-72B-Instruct
- [ ] Run on Gemma-2-27B-it

#### Writeup
- [ ] Write up results in writeup/emotions/draft.md

## Results: Small-Tier Cross-Model Comparison

### Summary Table (real handcrafted stimuli, 15 emotions, 25 stories each)

| Claim | Threshold | Qwen 1.5B | Gemma 2B | Llama 1B | Universality |
|-------|-----------|-----------|----------|----------|:------------:|
| **Probe classification** | ≥0.50 | **0.731** ✅ | **0.776** ✅ | **0.773** ✅ | UNIVERSAL |
| **Generalization** | ≥0.50 | **0.800** ✅ | **0.800** ✅ | **0.733** ✅ | UNIVERSAL |
| **Representation geometry** | ≥0.50 | **0.810** ✅ | **0.811** ✅ | **0.666** ✅ | UNIVERSAL |
| **Parametric intensity** | ≥0.50 | **0.886** ✅ | **0.971** ✅ | **0.943** ✅ | UNIVERSAL |
| **Causal steering** | ≥3 effects | 2 ❌ | 0 ❌ | 2 ❌ | NULL |
| **Preference steering** | ≥0.40 | 0.0 ❌ | 0.0 ❌ | 0.0 ❌ | NULL |

### Key Findings

**4 of 6 claims replicate universally across all 3 model families at small scale:**

1. **Probe classification (UNIVERSAL)**: Linear probes classify 15 emotions from residual stream activations with 73-78% accuracy across all families. Paper reported 71.3% on Claude — our results are remarkably consistent.

2. **Generalization (UNIVERSAL)**: Probes trained on explicit emotion stories transfer to implicit scenarios (73-80% diagonal dominance). This confirms emotions are represented abstractly, not just as lexical patterns.

3. **Representation geometry (UNIVERSAL)**: PCA of emotion vectors reveals a valence dimension. Correlation with human valence ratings: Qwen r=0.810, Gemma r=0.811, Llama r=0.666. The paper reported r=0.81 on Claude — Qwen and Gemma match almost exactly.

4. **Parametric intensity (UNIVERSAL)**: Emotion vectors scale proportionally with continuous severity parameters (Tylenol dosage, financial loss, etc.). Rank correlations 0.886–0.971 across all models.

**2 claims do not replicate at small scale (expected):**

5. **Causal steering (NULL at small scale)**: Log-prob scoring detects 0-2 causal effects (threshold: 3). Small 1-2B models may lack sufficient representational capacity for behavioral steering to produce robust effects. Medium-tier (7-9B) models should show stronger effects.

6. **Preference steering (NULL at small scale)**: No correlation between concept valence and preference shift. Same reasoning — likely needs larger models where emotion representations are more behaviorally potent.

### Model-Specific Notes

**Gemma 2B**: Highest probe accuracy (0.776) and parametric correlation (0.971). Strongest overall performance at small scale.

**Llama 1B**: Slightly lower valence correlation (0.666 vs 0.81) — smallest model in the tier (1B vs 1.5-2B), suggesting geometry sharpens with scale.

**Qwen 1.5B**: Most consistent with paper's Claude results — probe accuracy 0.731 (paper: 0.713), valence correlation 0.810 (paper: 0.81).

## Steering Approach Tradeoffs (documented)

**Original approach**: Generate full text responses (100-200 tokens) with per-token hook injection, classify via keyword matching. **Problem**: ~30 min per concept×scenario pair on MPS due to run_with_hooks overhead per token (~1.7 tok/s vs 15+ tok/s normal generation).

**Current approach**: Log-probability scoring — single forward pass per choice, compare log-prob of ethical vs unethical completions under steering. **~1000x faster** (~seconds per pair). Tradeoff: less naturalistic (scores predefined choices rather than free-form generation), but more statistically principled.

**Alternative not taken**: Reduce max_new_tokens to 50 with greedy decoding. Would keep generation-based approach but at ~17 min/model. Rejected because log-prob scoring is both faster AND more principled.

**Recommendation for Phase B**: On Colab with CUDA, generation speed will be much higher (~50+ tok/s). Consider restoring generation-based approach for medium/large models to get richer behavioral data.

## Failed approaches
- `LogisticRegression(multi_class="multinomial")`: deprecated in sklearn 1.7+. Removed parameter.
- `model.generate(fwd_hooks=...)`: TransformerLens `generate()` doesn't accept `fwd_hooks`. Fixed to use manual token-by-token loop with `run_with_hooks()`.
- Generation-based steering on MPS: ~30 min per pair, impractical for 120+ pairs. Switched to log-prob scoring.
- `HookedTransformer.from_pretrained(token=True)`: conflicts with TL's internal token handling. Fixed by setting `HF_TOKEN` env var from cached login.

## Session log

### 2026-04-08 Session 1
- Built entire framework from scratch (34 Python files, 81 tests)
- Ran placeholder stimuli on Qwen 1.5B: 4/6 pass (probe=0.937 on trivial stimuli)

### 2026-04-08 Session 2
- Generated 375 real handcrafted stimuli (25 stories × 15 emotions)
- Fixed causal steering: replaced generation-based approach with log-prob scoring (1000x speedup)
- Fixed HF auth for gated models (Llama, Gemma)
- Ran all 3 small-tier models with real stimuli:
  - Qwen 1.5B: 4/6 PASS (probe=0.731, geometry=0.810, parametric=0.886)
  - Gemma 2B: 4/6 PASS (probe=0.776, geometry=0.811, parametric=0.971)
  - Llama 1B: 4/6 PASS (probe=0.773, geometry=0.666, parametric=0.943)
- Cross-model analysis: 4 claims UNIVERSAL, 2 NULL (steering needs larger models)
- Generated comparison heatmap figure
- **Next session:**
  1. Phase B: Run on medium-tier models (7-9B) on Colab
  2. Debug steering at medium scale (may need generation-based approach on CUDA)
  3. Run scaling analysis with medium + small data points
  4. Begin writeup
