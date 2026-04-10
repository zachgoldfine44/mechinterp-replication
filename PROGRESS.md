# Progress Log

## Active paper: emotions (Sofroniew et al. 2026)

## Status: 🟢 Phase A + Phase B COMPLETE; writeup v3.2 + harness v2 incorporating four rounds of critique fixes

**Writeup v3.2 (multi-seed + severity pairs, ~7 min new compute):**
- Multi-seed probe runs on cached activations: 5 seeds × top-3 layers × 6 models in 94 seconds (CPU only). Std 0.004-0.012; multi-seed means systematically 0.005-0.014 lower than v1 single-seed (random_state=42 was a slight favorable bias). Qualitative result unchanged.
- Severity-pair test: 10 hand-designed paired prompts with shared numbers but different danger contexts. **5/6 models show real severity signal** when number is held constant. Llama-3.1-8B is cleanest with 4/6 concepts at p<0.01 (`desperate` larger in 10/10 pairs, `happy` smaller in 0/10 pairs). Qwen-2.5-7B is the only failure — converging with v3.1's parametric break.
- Claim 4 reframed: "5/6 pass real-only metric, all 6 contaminated, but 5/6 also pass stricter severity-pairs validity check (real signal exists underneath)".
- New scripts: `scripts/multi_seed_probes.py`, `scripts/severity_pairs_test.py`.
- New data: `drive_data/data/emotions/severity_pairs.json` (10 pairs).

**Writeup v3.1 (GPU follow-ups, 21 min A100 time):**
- Negative control parametric on medium tier:
  - Llama 8B: contamination 0.730 (was 1.000 small)
  - Qwen 7B: real-only rank correlation 0.493 — **FAILS** the 0.50 threshold; contamination 1.449 (negative control STRONGER than real)
  - Gemma 9B: contamination 0.559 (was 1.000 small)
  - Headline parametric verdict downgraded: "5/6 pass real-only metric, 1 fails outright, all 6 contaminated"
- Mean-pooling vs last-token comparison: cross-model deltas tiny (-0.016 to +0.032). Qwen 7B benefits from mean-pooling.
- Multi-layer steering on Llama 8B (9 layers × 1 concept × 1 alpha × 1 scenario × 5 samples): only layer 12 has 1/5 sample flipped, well within noise. Does NOT rescue the steering null.
- New `scripts/gpu_followups.py` (582 LOC, runnable on any pod with HF token).


**Writeup v3 (low-hanging-fruit pass on the v2 critiques):**
- Parametric scaling: added negative-control template ("blueberries"). Contamination ratio = 1.000 across all 3 small models — fear vector activates monotonically with blueberry count just as strongly as with Tylenol dosage. Claim 4 reframed as contaminated, not "passes universally".
- Geometry: layer-stability check across top-3 probe layers; spread is small (~0.1) so the result is NOT a brittle single-layer artifact. Saved to `geometry_layer_stability.json`.
- Stimulus audit: new `scripts/audit_stimuli.py` runs against actual stimuli; **0 leaks across 375 stimuli**. Runtime `validate_stimuli_for_leaks` added to dataset loader.
- Run manifest: git SHA, lib versions, torch device/dtype now attached to every fresh ExperimentResult.metadata['run_manifest'] via the base Experiment class.
- Cascading dependency failure: pipeline now skips dependent claims cleanly when their dependency failed (was producing cascading FileNotFoundErrors).
- Topological sort: deque.popleft (O(1)).
- `--fast` mode: loud multi-line warning that results are not comparable to full mode.
- Behavioral scenarios schema: verified correct (no bug).
- Stimulus count discrepancy fixed (config aligned to actual 25/concept).


**Harness v2 (technique modules + tests, in response to "implement the missing files" critique):**
- 5 previously-vapor technique modules now exist with full implementations and unit tests:
  - `src/techniques/patching.py` (13 tests) — `causal_trace`, `patch_residual_stream`, denoising/noising sweeps
  - `src/techniques/attention.py` (14 tests) — pattern extraction, head attribution, induction-head detection, entropy
  - `src/techniques/sae.py` (17 tests) — `SimpleSAE` train-from-scratch + optional SAELens loader, top-features-per-concept
  - `src/techniques/logit_lens.py` (9 tests) — projections, per-layer top tokens, target-token trajectory
  - `src/techniques/circuit_discovery.py` (16 tests) — edge attribution patching, single-edge path patching, ACDC stub
- New `circuit_identification` experiment type (12 tests) wires patching/attention/circuit_discovery into a paper-agnostic experiment for IOI/induction-head/causal-trace style replications
- Consolidated `_get_layer_modules` duplication: now `src/utils/activations.get_hf_layer_modules` is the single canonical implementation (was duplicated across 4 files)
- New integration tests (10) load a real pythia-14m via TransformerLens and exercise the full harness end-to-end (extract → probe → contrastive → steer → patch → attention → logit lens → EAP)
- New regression tests (10) pin structural properties of activation extraction (norms, monotonicity with depth, dtype preservation, determinism, etc.) to catch silent regressions
- Updated DESIGN.md: status column on the techniques table; all 8 modules now ✅ except ACDC which is 🟡

**Test counts after harness v2:** 162 unit tests pass in 1.16s; 20 integration tests pass against real pythia-14m in 2.8s when run with `--integration` flag.

**v2 critique-response fixes from previous round (analysis-only, no GPU rerun needed):**
- Real Fisher's exact test on CUDA steering path; placeholder p-values removed
- PCA geometry now scores PC1 specifically (was max-over-PCs); + bootstrap 95% CIs
- Lexical baseline added (text-only TF-IDF/BoW): 0.35-0.40 vs activation 0.73-0.84
- Per-concept probe accuracy reported (`happy` is the worst class at 0.36-0.52)
- Stimulus count discrepancy fixed (config aligned to actual 25/concept)
- Dead `_compare_responses` word-count preference heuristic removed
- Writeup language downgraded from "universally replicates" to "passes our thresholds"
- Steering null reframed as two equal-standing hypotheses (not just "our methodology")
- Errata + change log added as §0 / Appendix D of writeup

## Current test status
```
81 tests passed (pytest tests/ -q --fast)
6/6 models completed (3 small tier + 3 medium tier)
4/6 claims REPLICATE universally across all families, both tiers
2/6 claims NULL (causal + preference steering) — framed as methodology limitation
Writeup draft complete at writeup/emotions/draft.md, ready for external critique
```

## Final Cross-Model Results

| Model | Probe | Generalize | Geometry | Parametric | Steer | Preference |
|---|:-:|:-:|:-:|:-:|:-:|:-:|
| Llama-3.2-1B | 0.773 ✅ | 0.733 ✅ | 0.666 ✅ | 0.943 ✅ | 2* ❌ | 0.0 ❌ |
| Qwen2.5-1.5B | 0.731 ✅ | 0.800 ✅ | 0.810 ✅ | 0.886 ✅ | 2* ❌ | 0.0 ❌ |
| Gemma-2-2B | 0.776 ✅ | 0.800 ✅ | 0.811 ✅ | 0.971 ✅ | 0* ❌ | 0.0 ❌ |
| Llama-3.1-8B | 0.819 ✅ | 0.867 ✅ | 0.738 ✅ | 0.971 ✅ | 0/45 ❌ | 0.0 ❌ |
| Qwen2.5-7B | 0.784 ✅ | 0.667 ✅ | 0.828 ✅ | 0.943 ✅ | 0/45 ❌ | 0.0 ❌ |
| Gemma-2-9B | 0.840 ✅ | 0.800 ✅ | 0.790 ✅ | 0.571 ✅ | 0/45 ❌ | 0.0 ❌ |
| **Universality** | UNIVERSAL | UNIVERSAL | UNIVERSAL | UNIVERSAL | NULL | NULL |

*Small-tier steering used noisier protocol; medium tier used definitive 10-sample batched generation with LLM-as-judge.

Key findings:
- Probe accuracy scales cleanly with parameters (Spearman r=0.943, p=0.005)
- Qwen2.5-7B matches paper's valence geometry exactly (0.828 vs paper's 0.81)
- Behavioral null driven by 0% baseline unethical rate (RLHF floor effect) — all 3 medium models refuse scenarios at all alphas up to 0.5

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

#### Phase C: Large-tier scaling — **DEFERRED**
*Decision 2026-04-09: Stop after medium-tier (7-9B) runs. Write up small+medium
results and share for external critique first. Revisit large-tier (27B-70B) in
a future session if the writeup feedback suggests it would add value.*
- [ ] ~~Run on Llama-3.1-70B-Instruct~~ (deferred)
- [ ] ~~Run on Qwen-2.5-72B-Instruct~~ (deferred)
- [ ] ~~Run on Gemma-2-27B-it~~ (deferred)

#### Writeup (next up)
- [x] Draft `writeup/emotions/draft.md` with Phase A results + placeholders for Phase B
- [ ] Fill Phase B numbers once definitive run completes
- [ ] Generate publication-ready figures (heatmap, scaling curves, PCA scatter)
- [ ] Share draft for external critique

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

## Phase B: Medium-tier replication (RunPod A100, in progress)

### Infrastructure setup
- **First pod** (213.173.102.5): A100 80GB with only 20GB root disk + MFS workspace with ~20GB quota.
  Encountered multiple disk quota issues; Llama 8B and Qwen 7B completed with log-prob scoring
  (4/6 PASS each) but workspace corrupted file writes blocked further progress.
- **Second pod** (154.54.102.48): A100 80GB with **100GB root disk** — fresh install, full project
  sync, reliable reads/writes. Current active pod.

### Code optimizations (Phase B)

1. **Probe training: scan ~9 layers instead of 32** (4x speedup)
   - sklearn `LogisticRegression` on 4096-dim activations for 32 layers × 5 folds = 160 fits × ~5s each = ~60 min per model on CPU.
   - Fix: auto-select every 4th layer + first + last when `n_layers > 16`.
   - Result: ~10-15 min per model for probe classification on A100.

2. **Generation-based steering restored on CUDA + log-prob fallback on CPU/MPS**
   - Phase A used log-prob scoring (fast but insensitive) because generation-based was impractical on MacBook (~30 min/pair).
   - On A100, generation is much faster (~50+ tok/s normal, ~5-10 tok/s with per-token hooks).
   - Fix: `_is_cuda(model)` check routes to generation-based `_run_choice_shift` on CUDA, falls back to log-prob scoring on CPU/MPS.
   - Preference steering: LLM-as-judge implemented on CUDA (matches paper's preference-model approach; paper used a preference model, not human raters).

3. **Steering runtime tuning (2026-04-09)**
   - First medium run with default params took 18 min/pair on Llama 8B (360 pairs = ~108 hrs).
   - Root cause: `run_with_hooks()` per-token is slow even on A100; TransformerLens doesn't batch.
   - Fix (applied during Phase B): reduced `n_samples_per_condition` from 5 to 1, `steering_alpha` from [0.01, 0.05, 0.1] to [0.05], and `max_new_tokens` from 100 to 30.
   - New estimate: ~120 pairs × ~10s = ~20 min per model = ~1-2 hrs total for all 3 medium models.
   - Tradeoff: noisier individual measurements (single sample per condition) but still detects behavioral shifts.

4. **Implicit scenarios expanded from 30 to 225** (2026-04-09)
   - Noticed multiple models scoring exactly 0.800 on generalization — a resolution artifact: with only 2 test samples per concept, diagonal dominance was quantized to multiples of 1/15.
   - Fix: generated 15 situational scenarios per emotion (225 total) for meaningful granularity.
   - Will re-run generalization claim for all models with expanded scenarios.

5. **Infrastructure workarounds**
   - `TORCHDYNAMO_DISABLE=1`: prevents torch.compile from hanging during model load.
   - `HF_HOME=/workspace/hf_cache`: routes model weights to high-capacity filesystem.
   - `HF_HUB_OFFLINE=1`: when cache is valid, avoids redundant download attempts.
   - `_ensure_hf_token_env()`: sets `HF_TOKEN` from cached login so TransformerLens can access gated repos.
   - `nohup python3 run_all.py`: server-side sequential runner so SSH disconnects don't interrupt the pipeline.

## Failed approaches
- `LogisticRegression(multi_class="multinomial")`: deprecated in sklearn 1.7+. Removed parameter.
- `model.generate(fwd_hooks=...)`: TransformerLens `generate()` doesn't accept `fwd_hooks`. Fixed to use manual token-by-token loop with `run_with_hooks()`.
- Generation-based steering on MPS: ~30 min per pair, impractical for 120+ pairs on MacBook. Log-prob scoring there.
- `HookedTransformer.from_pretrained(token=True)`: conflicts with TL's internal token handling. Fixed by setting `HF_TOKEN` env var.
- **RunPod pod #1 (20GB root + 20GB workspace quota)**: workspace filesystem corrupted `torch.save` writes, blocked code sync. Abandoned in favor of 100GB pod.
- **PyTorch 2.11.0+cu130 preinstalled on RunPod**: incompatible with CUDA driver 12.4. Fixed by force-reinstalling `torch==2.4.1+cu124`.
- **Default steering params (5 samples × 3 alphas × 100 tokens)**: would take ~5 days per model on A100 due to `run_with_hooks` overhead. Reduced to 1 sample × 1 alpha × 30 tokens.
- **Default 2 implicit scenarios per concept**: produced 0.800 resolution artifact on diagonal dominance. Expanded to 15 per concept.

## Session log

### 2026-04-08 Session 1
- Built entire framework from scratch (34 Python files, 81 tests)
- Ran placeholder stimuli on Qwen 1.5B: 4/6 pass

### 2026-04-08 Session 2
- Generated 375 real handcrafted stimuli (25 stories × 15 emotions)
- Fixed causal steering: log-prob scoring on CPU/MPS
- Fixed HF auth for gated models
- Ran all 3 small-tier models (Phase A): 4/6 PASS UNIVERSAL
- Generated comparison heatmap figure

### 2026-04-09 Session 3 (Phase B)
- Set up RunPod A100 pod #1 (213.173.102.5) — hit workspace quota issues
- Ran Llama 8B + Qwen 7B on pod #1: 4/6 PASS each with log-prob steering (NULL)
- Could not complete Gemma 9B due to disk quota corruption
- Switched to RunPod A100 pod #2 (154.54.102.48) with 100GB disk
- Implemented generation-based steering on CUDA + LLM-as-judge for preference
- Expanded implicit scenarios from 30 to 225 (15/emotion)
- Optimized probe training: scan ~9 layers instead of 32
- Reduced steering scope: n_samples=1, alpha=[0.05], max_tokens=30
- Launched server-side `nohup` run of all 3 medium models (PID 2222 on pod #2)
- **Next:** When run completes, sync results back, run cross-model + scaling analysis, update this file with Phase B results

## Phase B: Partial results (from RunPod pod #1, log-prob steering)

| Model | Probe | Generalize | Geometry | Parametric | Steer | Preference |
|-------|:-:|:-:|:-:|:-:|:-:|:-:|
| llama_8b | **0.819** ✅ | 0.800* ✅ | **0.738** ✅ | **0.971** ✅ | 2 ❌ | 0.0 ❌ |
| qwen_7b | **0.784** ✅ | 0.800* ✅ | **0.828** ✅ | **0.943** ✅ | 0 ❌ | 0.0 ❌ |
| gemma_9b | (pending) | | | | | |

*Generalization was measured against only 30 implicit scenarios → resolution artifact at 0.800.
Final Phase B values will use 225 expanded scenarios.

**Scaling confirmed**: Probe accuracy increases with model size.
- Llama: 0.773 (1B) → 0.819 (8B) ✅
- Qwen: 0.731 (1.5B) → 0.784 (7B) ✅
