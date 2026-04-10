# Design Document: Cross-Model Mechinterp Replication Framework

## Overview

This document describes the paper-agnostic framework for replicating mechanistic
interpretability papers across open-source models. The key abstraction: separate
the *what* (paper-specific claims, stimuli, concepts) from the *how* (techniques,
cross-model execution, analysis).

Paper-specific details live entirely in `config/papers/{paper_id}/`. The
framework code in `src/` should never reference a specific paper.

---

## 1. Architecture

### The Experiment base class

All experiments inherit from `src/core/experiment.py`:

```python
class Experiment(ABC):
    """Base class for all replication experiments."""

    def __init__(self, config: ClaimConfig, model_key: str, paths: ProjectPaths):
        self.config = config        # from paper_config.yaml
        self.model_key = model_key
        self.paths = paths
        self.results_dir = paths.results / config.paper_id / model_key / config.claim_id

    @abstractmethod
    def run(self, model, activations_cache) -> ExperimentResult:
        """Execute the experiment. Returns structured results."""
        ...

    @abstractmethod
    def evaluate(self, result: ExperimentResult) -> bool:
        """Check if result meets the success criterion."""
        ...

    def load_or_run(self, model, activations_cache) -> ExperimentResult:
        """Resume-aware: load cached result if exists, else run."""
        result_path = self.results_dir / "result.json"
        if result_path.exists():
            return ExperimentResult.load(result_path)
        result = self.run(model, activations_cache)
        result.save(result_path)
        return result
```

### The Claim dataclass

```python
@dataclass(frozen=True)
class ClaimConfig:
    paper_id: str
    claim_id: str
    description: str
    experiment_type: str       # maps to an Experiment subclass
    params: dict               # passed to the experiment
    success_metric: str
    success_threshold: float
    depends_on: str | None     # claim_id this depends on, or None
```

### The Pipeline orchestrator

`src/core/pipeline.py` is the main entry point:

```
1. Load paper config → list of ClaimConfigs
2. Load model config → list of models to run
3. For each model:
   a. Load model (or reuse if already loaded)
   b. For each claim (in dependency order):
      i.  Resolve experiment_type → Experiment subclass
      ii. Call experiment.load_or_run(model, cache)
      iii. Call experiment.evaluate(result)
      iv. Log result to manifest
   c. Unload model
4. Run cross-model analysis
5. Generate figures
```

### The Technique modules

`src/techniques/` contains reusable building blocks. These are NOT experiments
— they're tools that experiments compose:

**Status legend:** ✅ implemented · 🟡 stub / placeholder · ❌ planned, not yet written

| Status | Module | What it provides | Used by which experiments |
|:--:|--------|-----------------|--------------------------|
| ✅ | `probes.py` | Train/eval linear probes, logistic regression, MLP probes | probe_classification, generalization_test |
| ✅ | `contrastive.py` | Mean-difference vectors, CAA extraction, PCA, similarity matrices | probe_classification, causal_steering, representation_geometry |
| ✅ | `steering.py` | Activation addition at inference time, control vectors | causal_steering |
| ✅ | `patching.py` | Activation patching, causal tracing, denoising/noising sweeps | circuit_identification |
| ✅ | `attention.py` | Attention pattern extraction, head attribution, induction-head detection, attention entropy | circuit_identification |
| ✅ | `sae.py` | `SimpleSAE` train-from-scratch + optional `sae_lens` pretrained loader, feature extraction, top-features-per-concept | representation_geometry, circuit_identification |
| ✅ | `logit_lens.py` | Logit lens projections + per-layer top tokens + target-token trajectory; tuned-lens hook | representation_geometry, circuit_identification |
| ✅ | `circuit_discovery.py` | Edge attribution patching (gradient-based), path patching, subgraph extraction | circuit_identification |
| 🟡 | `circuit_discovery.py::acdc` | Full ACDC algorithm (Conmy et al.) | (raises NotImplementedError; use EAP + path_patch instead) |

All eight technique modules now exist with working implementations and unit tests
(150+ unit tests across the harness, plus 20 integration tests against a real
pythia-14m model). Coverage of mechinterp methods supported by the harness:
**probing, contrastive activation steering, activation patching, attention head
analysis, SAE features, logit lens, edge attribution patching, single-edge path
patching.** This is enough surface area to attempt replications of papers like
IOI (Wang et al.), induction heads (Olsson et al.), and ROME-style causal traces
(Meng et al.) — see the `circuit_identification` experiment type for the glue.

Notable gaps: ACDC is stubbed (use EAP as a fast approximation), and tuned lens
requires bringing your own checkpoint (no automatic training pipeline).

---

## 2. Model matrix

### Configuration (`config/models.yaml`)

The model matrix is paper-agnostic. Each paper config specifies
`model_variant: "base"` or `"instruct"`, and the loader picks the right one.

**Instruct variants** (for papers studying RLHF'd models):

| Family | Small | Medium | Large |
|--------|-------|--------|-------|
| Llama 3.1/3.2 | 1B-Instruct | 8B-Instruct | 70B-Instruct |
| Qwen 2.5 | 1.5B-Instruct | 7B-Instruct | 72B-Instruct |
| Gemma 2 | 2B-it | 9B-it | 27B-it |

**Base variants** (for papers studying pretrained models):

| Family | Small | Medium | Large |
|--------|-------|--------|-------|
| Llama 3.1/3.2 | 1B | 8B | 70B |
| Qwen 2.5 | 1.5B | 7B | 72B |
| Gemma 2 | 2B | 9B | 27B |

**Special models** (for papers studying specific architectures):

Some papers study a specific model (e.g., IOI studies GPT-2 Small). The
paper config can specify `additional_models` to include these:

```yaml
additional_models:
  - hf_id: "openai-community/gpt2"
    key: "gpt2_small"
    reason: "Original model from the paper"
```

### Model loading strategy

```
Priority order:
1. TransformerLens (if supported) — cleanest hook API
2. nnsight (good for models TL doesn't support) — proxy-based hooks
3. Raw HuggingFace + register_forward_hook — fallback
```

### Execution order: Small → Medium → Large

1. **Small tier first (locally on MacBook Air M3)**. Validates that the full
   pipeline runs — model loading, activation extraction, caching/resume,
   probe training, result saving. Catches infrastructure bugs for free
   before spending Colab GPU-hours. Null results on 1-2B models are expected;
   this tier establishes the scaling floor.
2. **Medium tier second (on Colab)**. Primary replication target. Debug
   methodology here until results are interpretable.
3. **Large tier last (on Colab, if compute allows)**. Scaling ceiling.

Within each tier: start with the smallest model (faster loading, faster debugging).

---

## 3. Stimulus design (paper-specific)

Stimuli are defined entirely in `config/papers/{paper_id}/stimuli_config.yaml`.
The framework supports four stimulus types:

### `generated` — LLM-generated stimuli
Use when the paper needs many varied examples (e.g., 50 stories per emotion).
```yaml
training_stimuli:
  type: "generated"
  generation_prompt: |
    Write a short first-person story (100-200 words) where the narrator
    experiences {concept}. ...
  per_concept: 50
  output_dir: "data/{paper_id}/training/"
```

Generation is resume-aware: each stimulus is saved individually, so if
generation is interrupted at item 37/50, it resumes at 38.

### `hardcoded` — Manually curated stimuli
Use for small, carefully designed test sets (e.g., 12 implicit scenarios).
```yaml
implicit_scenarios:
  type: "hardcoded"
  file: "data/{paper_id}/implicit_scenarios.json"
```

### `dataset` — External dataset
Use when the paper uses a known dataset (e.g., true/false claims dataset).
```yaml
factual_claims:
  type: "dataset"
  source: "huggingface"
  dataset_id: "notrichman/true_false_dataset"
  split: "train"
  concept_column: "label"
  text_column: "statement"
```

### `programmatic` — Parameterized templates
Use when stimuli are generated from templates with variable substitution.
```yaml
parameterized:
  type: "programmatic"
  templates:
    - template: "The {subject} is {adjective}."
      variables:
        subject: ["cat", "dog", "bird"]
        adjective: ["happy", "sad"]
```

---

## 4. Activation extraction

### What to extract

The extraction strategy depends on the experiment type:

| Experiment type | What to extract | Aggregation |
|----------------|-----------------|-------------|
| probe_classification | Residual stream at target layer(s) | Last token, or mean over tokens |
| generalization_test | Same as probe_classification | Same |
| parametric_scaling | Same as probe_classification | Same |
| causal_steering | Full forward pass with hooks (no pre-extraction) | N/A — modifies activations in-place |
| circuit_identification | All components: residual, attention patterns, MLP outputs | Per-token, per-component |
| representation_geometry | Residual stream at all layers | Last token or specific positions |

### Caching strategy

All activations cached per-item to Google Drive (via `get_data_root()`):
```
{DATA_ROOT}/activations/{paper_id}/{model_key}/{stimulus_set}_{stimulus_id}.pt
```

Uses the skip-if-exists + atomic write pattern (see CLAUDE.md checkpointing
section). Code lives in the GitHub repo; data files live on Google Drive.
See CLAUDE.md "Git, GitHub, and multi-environment workflow" for the full setup.

### Memory management

| Environment | Models | Batch size | Precision |
|------------|--------|------------|-----------|
| MacBook Air M3 | 1-2B only | 2-4 | float16 on MPS (CPU fallback) |
| Colab A100/H100 | 7-9B | 4-8 | float16 |
| Colab A100/H100 | 70B+ | 1 | 4-bit (bitsandbytes) |

---

## 5. Technique details

### Linear probes (`src/techniques/probes.py`)

Start simple. Linear probes (logistic regression) are the default. The
linearity of a representation is itself a finding. Only escalate to MLP
probes if linear probes fail across all models.

**Training protocol** (configurable per paper):
- k-fold CV (default k=5), stratified by concept
- L2 regularization (default λ=1.0, tunable)
- Each fold saved separately for resume
- Reports: mean ± std accuracy, per-concept breakdown

### Contrastive vectors (`src/techniques/contrastive.py`)

Mean-difference approach (CAA-style):
```
v_concept = mean(activations | concept) - mean(all activations)
v_concept = v_concept / ||v_concept||
```

Also supports paired contrastive extraction:
```
v = mean(activations | condition_A) - mean(activations | condition_B)
```

### Steering (`src/techniques/steering.py`)

Activation addition: `h'_l = h_l + α * v`

**Configurable parameters**:
- α: steering strength (default 0.5, sweep configurable)
- Layers: which layers to steer at (default: middle third)
- Controls: random vector (null), negated vector (reversal)

### Activation patching (`src/techniques/patching.py`)

Supports:
- **Activation patching**: Replace activations from clean → corrupted run
- **Path patching**: Trace specific computational paths
- **Causal tracing**: Restore activations one-at-a-time to find critical components

### Circuit discovery (`src/techniques/circuit_discovery.py`)

Supports:
- **ACDC**: Automatic circuit discovery via ablation
- **Edge attribution**: Score edges in the computation graph
- **Subgraph extraction**: Extract minimal faithful subgraph

---

## 6. Cross-model analysis

This is paper-agnostic. For any set of experiment results:

### Per-claim comparison table
Rows = models (sorted by family, then size). Columns = key metrics.
Color-coded by success criterion (green = passes, red = fails).

### Scaling plot
X = model params (log scale), Y = key metric. One line per family.
Question: does the finding get stronger with scale?

### Family comparison
Bar chart at matched size (~7-9B). Question: does the finding depend on
which model family, or is it universal?

### Statistical tests
- Probe accuracies: paired t-test across concepts
- Effect sizes: bootstrap confidence intervals
- Scaling trends: Spearman correlation with model size

### What to look for
- **Universal findings**: Replicate on all models → strongest evidence.
- **Family-specific**: Replicate on Llama but not Qwen → training dependence.
- **Scale-dependent**: Only replicate above a threshold → emergent capability.
- **Null results**: Don't replicate anywhere → original-model-specific.

---

## 7. Known risks and mitigations

| Risk | Mitigation |
|------|-----------|
| Paper's methodology is too model-specific to generalize | Start with the most standard techniques; note deviations |
| Models can't process the paper's stimuli | Verify basic comprehension before running experiments |
| Activation extraction differs across loading frameworks | Unit test that TL/nnsight/HF extractions agree |
| Large models don't fit in memory | 4-bit quantization; note as limitation |
| Too many statistical comparisons → false positives | Bonferroni correction; report all results |
| Paper uses a technique not in our toolkit | Add to `src/techniques/` or `src/experiments/paper_specific/` |
| Base vs. instruct model confound | Paper config specifies variant; flag as limitation if mixed |

---

## 8. Dependencies on ARENA 3.0 methodology

This project draws on the ARENA curriculum's techniques:

| ARENA chapter | Technique | Our module |
|--------------|-----------|------------|
| Ch1 §1-2: Transformers + TransformerLens | Hook-based activation extraction | `src/utils/activations.py` |
| Ch1 §3: IOI / circuits | Activation patching, circuit analysis | `src/techniques/patching.py`, `circuit_discovery.py` |
| Ch1 §4: Superposition | SAE-based feature extraction | `src/techniques/sae.py` |
| Ch1 §5: Probing | Linear probes on activations | `src/techniques/probes.py` |
| Ch1 §6: Function vectors | Steering vectors / activation addition | `src/techniques/steering.py` |

Code should use ARENA-style patterns (einops, TransformerLens hooks, jaxtyping)
for accessibility to the ARENA/MATS community.

---

## 9. Adding a new technique

When a paper requires a technique not yet in `src/techniques/`:

1. Create `src/techniques/{technique_name}.py`
2. Follow the existing module pattern: pure functions, type hints, docstrings
3. Add a corresponding test in `tests/test_techniques.py`
4. If the technique requires a new experiment type, add it to `src/experiments/`
5. Document in this file (Section 5)

The technique module should be paper-agnostic — it operates on tensors and
configs, not on specific concepts or stimuli.
