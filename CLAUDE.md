# Mechinterp Paper Replication Engine

## What is this?

A general-purpose framework for replicating mechanistic interpretability papers
across multiple open-source model families (Llama, Qwen, Gemma) at various
scales, then writing up the results.

Think of it like a replication study in psychology: same experimental protocol,
different populations (models), measuring what transfers and what doesn't.
But instead of one paper hardwired in, this is a replication *engine* — you
feed it a paper config, and it designs experiments, runs them, and writes up
whether the findings generalize.

## Quick reference

* **Design document**: `DESIGN.md` (framework architecture, replication protocol)
* **Progress log**: `PROGRESS.md` (read this first every session)
* **Active paper config**: `config/papers/{paper_id}/paper_config.yaml`
* **Model matrix**: `config/models.yaml`

## Setup

```bash
# Create environment
python -m venv .venv && source .venv/bin/activate

# Install core dependencies
pip install torch transformers transformer-lens sae-lens nnsight
pip install einops jaxtyping beartype fancy_einsum
pip install pandas numpy scipy scikit-learn matplotlib seaborn plotly
pip install pytest pyyaml tqdm wandb huggingface_hub

# Verify GPU access
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"

# Run smoke test
pytest tests/test_smoke.py -v

# Run full test suite (fast mode)
pytest tests/ -v --fast
```

---

## Orientation (read this first every session)

1. `git pull` — get latest code.
2. Read `PROGRESS.md` — what's done, what's next, what failed.
3. Check which paper is active: `cat config/active_paper.txt`
4. Read the active paper config: `cat config/papers/{paper_id}/paper_config.yaml`
5. Run `pytest tests/ -v --fast 2>&1 | tail -30` — current test status.
6. Pick the next unchecked item from PROGRESS.md.
7. When you finish a unit of work, update PROGRESS.md, commit, and push.

---

## Project structure

```
.
├── CLAUDE.md                       # You are here
├── DESIGN.md                       # Framework architecture & replication protocol
├── PROGRESS.md                     # Session-to-session memory
├── config/
│   ├── active_paper.txt            # Single line: which paper_id is active
│   ├── models.yaml                 # Model matrix (families × sizes)
│   └── papers/                     # One folder per paper to replicate
│       ├── emotions/               # Example: Anthropic emotions paper
│       │   ├── paper_config.yaml   # Claims, success criteria, experiment types
│       │   └── stimuli_config.yaml # Paper-specific stimulus definitions
│       ├── geometry_of_truth/      # Example: Geometry of Truth
│       │   ├── paper_config.yaml
│       │   └── stimuli_config.yaml
│       └── ioi/                    # Example: IOI circuit in GPT-2 Small
│           ├── paper_config.yaml
│           └── stimuli_config.yaml
├── src/
│   ├── core/                       # Paper-agnostic framework
│   │   ├── experiment.py           # Abstract Experiment base class
│   │   ├── claim.py                # Claim dataclass (hypothesis + success criterion)
│   │   ├── pipeline.py             # Orchestrator: load config → run experiments → analyze
│   │   └── config_loader.py        # Load & validate paper configs
│   ├── models/
│   │   ├── loader.py               # Unified model loading (HF + TransformerLens + nnsight)
│   │   └── registry.py             # Model metadata (layer count, hidden dim, etc.)
│   ├── techniques/                 # Reusable mechinterp building blocks
│   │   ├── probes.py               # Linear probes, logistic regression, MLP probes
│   │   ├── contrastive.py          # Contrastive activation extraction (CAA / mean-diff)
│   │   ├── steering.py             # Activation addition / steering vectors
│   │   ├── patching.py             # Activation patching (causal tracing)
│   │   ├── attention.py            # Attention pattern analysis, head attribution
│   │   ├── sae.py                  # SAE-based feature extraction
│   │   ├── logit_lens.py           # Logit lens / tuned lens
│   │   └── circuit_discovery.py    # Path patching, ACDC, edge attribution
│   ├── experiments/                # Paper-specific experiment implementations
│   │   ├── __init__.py             # Auto-discovers experiment classes
│   │   ├── probe_classification.py # Generic: train probes to classify concepts
│   │   ├── generalization_test.py  # Generic: test if probes generalize to held-out
│   │   ├── parametric_scaling.py   # Generic: test if probe tracks continuous variable
│   │   ├── causal_steering.py      # Generic: steer with vectors, measure behavioral shift
│   │   ├── circuit_identification.py   # Generic: find circuits for a behavior
│   │   ├── representation_geometry.py  # Generic: analyze geometry of representation space
│   │   └── paper_specific/         # Custom experiments that don't fit generic patterns
│   │       └── README.md           # When to add a custom experiment
│   ├── analysis/
│   │   ├── cross_model.py          # Compare results across model matrix
│   │   ├── scaling.py              # How do results change with model size?
│   │   └── visualize.py            # Plots, heatmaps, similarity matrices
│   └── utils/
│       ├── activations.py          # Extract, cache, manage activations
│       ├── datasets.py             # Stimulus/dataset generation & loading
│       ├── env.py                  # Environment detection (MacBook vs Colab), paths
│       └── metrics.py              # Cosine sim, accuracy, effect sizes, CIs
├── tests/
│   ├── test_smoke.py               # Quick sanity (loads 1 model, runs 1 technique)
│   ├── test_techniques.py          # Each technique module works correctly
│   ├── test_experiments.py         # Generic experiment classes run end-to-end
│   ├── test_config.py              # Paper configs parse and validate
│   └── test_cross_model.py         # Cross-model comparison pipeline
├── data/                           # Generated/loaded stimuli per paper
│   └── {paper_id}/
├── results/                        # Raw outputs per paper × model × experiment
│   └── {paper_id}/{model_key}/
├── figures/                        # Publication-ready plots per paper
│   └── {paper_id}/
└── writeup/
    └── {paper_id}/
        ├── draft.md
        └── appendix.md
```

---

## How to replicate a new paper

This is the core workflow. It generalizes to any mechinterp paper.

### Step 1: Read the paper and extract claims

Read the paper carefully. Identify its 3-7 key claims — the findings that,
if they replicate, would constitute a successful replication. Each claim must
be falsifiable and have a quantitative success criterion.

### Step 2: Create a paper config

Create `config/papers/{paper_id}/paper_config.yaml`. This file defines:

```yaml
paper:
  id: "my_paper"
  title: "Paper Title"
  authors: "Author et al."
  url: "https://..."
  original_model: "What model the paper studied"
  model_variant: "instruct"  # or "base" — determines which models we use

# What mechinterp techniques does this paper use?
# Determines which modules from src/techniques/ are needed.
techniques_required:
  - probes           # linear probes for classification
  - contrastive      # mean-difference / CAA vectors
  - steering         # activation addition
  # Other options: patching, attention, sae, logit_lens, circuit_discovery

# The paper's key claims, each mapped to a generic experiment type
claims:
  - id: "claim_1"
    description: "Human-readable description of the claim"
    experiment_type: "probe_classification"  # which generic experiment
    params:                                   # experiment-specific parameters
      concept_set: ["concept_a", "concept_b", "concept_c"]
      n_stimuli_per_concept: 50
      probe_type: "logistic_regression"
    success_metric: "probe_accuracy"
    success_threshold: 0.70
    notes: "Additional context for interpreting results"

  - id: "claim_2"
    experiment_type: "generalization_test"
    depends_on: "claim_1"                    # must run claim_1 first
    params:
      test_set: "implicit"
    success_metric: "diagonal_dominance"
    success_threshold: 0.67

  - id: "claim_3"
    experiment_type: "causal_steering"
    depends_on: "claim_1"
    params:
      steering_alpha: 0.5
      behavior_metric: "preference_shift"
    success_metric: "causal_effect_count"
    success_threshold: 3
```

### Step 3: Create a stimuli config

Create `config/papers/{paper_id}/stimuli_config.yaml`. This defines what
data each experiment needs:

```yaml
stimulus_sets:
  training_stimuli:
    type: "generated"        # Use LLM to create; options: generated, hardcoded, dataset, programmatic
    generation_prompt: |
      Write a short first-person story (100-200 words) where the narrator
      experiences {concept}. The story should clearly evoke {concept}.
    per_concept: 50
    output_dir: "data/{paper_id}/training/"

  implicit_scenarios:
    type: "hardcoded"
    file: "data/{paper_id}/implicit_scenarios.json"

  parameterized_templates:
    type: "programmatic"
    templates:
      - template: "I just took {X} mg of tylenol for my back pain."
        variable: "X"
        values: [200, 500, 1000, 2000, 5000, 10000]
        expected_response: {afraid: "increasing", calm: "decreasing"}

  factual_claims:
    type: "dataset"
    source: "huggingface"
    dataset_id: "notrichman/true_false_dataset"
    split: "train"
```

### Step 4: Set the active paper and run

```bash
echo "my_paper" > config/active_paper.txt
python -m src.core.pipeline --paper my_paper --validate-only  # dry run
python -m src.core.pipeline --paper my_paper --model llama_1b  # single small model (local)
python -m src.core.pipeline --paper my_paper --tier small      # all small models (local)
python -m src.core.pipeline --paper my_paper --tier medium     # all medium models (Colab)
python -m src.core.pipeline --paper my_paper --all             # everything
```

### Step 5: Analyze and write up

```bash
python -m src.analysis.cross_model --paper my_paper
python -m src.analysis.scaling --paper my_paper
# Then edit writeup/{paper_id}/draft.md
```

---

## Generic experiment types

These are the reusable experiment patterns that cover most mechinterp papers.
Each maps to a class in `src/experiments/`.

| Experiment type | What it tests | Example papers |
|----------------|---------------|----------------|
| `probe_classification` | Can a linear probe classify concepts from activations? | Emotions, Geometry of Truth, representation probing |
| `generalization_test` | Do probes generalize to held-out / structurally different stimuli? | Any paper claiming abstract representations |
| `parametric_scaling` | Does probe activation scale with a continuous parameter? | Emotions (intensity), dose-response studies |
| `causal_steering` | Does adding a vector to activations change model behavior? | Emotions, steering vectors, CAA, function vectors |
| `circuit_identification` | Can we find a circuit responsible for a behavior? | IOI, induction heads, greater-than circuit |
| `representation_geometry` | What's the geometric structure of a representation space? | Geometry of Truth, linear representation hypothesis |

When a paper's methodology doesn't fit any generic type, write a custom
experiment in `src/experiments/paper_specific/`. It must still inherit from
the `Experiment` base class.

---

## Principles for autonomous development

Adapted from the clax/C-compiler pattern.

### 1. The original paper is the oracle

The paper's reported results on its original model are our ground truth.
We can't always reproduce them exactly (different model, maybe no internals
access), but we CAN verify our methodology produces analogous patterns.

**Rules:**
- First, validate the pipeline runs end-to-end on a small model locally.
  This catches infrastructure bugs for free before using Colab.
- Then, validate the methodology produces interpretable results on a medium
  model. If it doesn't work on any 7-9B model, the pipeline has a bug —
  don't scale garbage.
- Write the test for each experiment BEFORE implementing it.
- When results look wrong, check upstream: is the activation extraction
  correct? Are the probes/circuits correct? Is the stimulus set valid?

### 2. Concise output (context window hygiene)

**Rules:**
- Tests print at most 5-10 lines on success, ~20 lines on failure.
- Use `pytest -q --fast` by default.
- Log verbose diagnostics to `results/{paper_id}/logs/`, not stdout.
- Error messages: `ERROR: {paper} {model} {experiment} — {reason}` on one line.

### 3. Fast tests to avoid time blindness

**Rules:**
- `--fast` flag runs on 2 concepts (not all), 1 model (not all),
  10 stimuli (not full set).
- Default dev cycle: `--fast` after every change, full suite before commit.
- Model loading is the bottleneck — cache aggressively.
- Always develop and test against small models locally first. Only move to
  Colab once the pipeline passes all local tests.

### 4. Keep PROGRESS.md current

**Rules:**
- Update after every meaningful unit of work.
- Record: what worked, what didn't, accuracy numbers, failed approaches.
- **Record failed approaches** so they aren't re-attempted.
- Track per-model results in a table.

### 5. Prevent regressions

- Run `pytest tests/ -q --fast` before every commit.
- Never commit code that breaks existing passing tests.

### 6. Small, testable commits

Each commit: one experiment on one model, or one infrastructure piece.

### 7. Document for the next session

Every module docstring explains what it does, what it expects, what it outputs,
and known model-specific limitations.

---

## The model matrix

See `config/models.yaml` for the full matrix:

| Family | Small | Medium | Large |
|--------|-------|--------|-------|
| Llama 3.1/3.2 | 1B | 8B | 70B |
| Qwen 2.5 | 1.5B | 7B | 72B |
| Gemma 2 | 2B | 9B | 27B |

### Execution order: Small → Medium → Large

**Small tier first (locally on MacBook Air M3).** This catches all
infrastructure bugs — loading, extraction, caching, resume, path resolution —
for free before burning Colab GPU-hours. If probes are broken or stimuli are
malformed, you find out in 2 minutes not 45. Null results on 1B models are
expected and fine; treat small-tier as "pipeline validation + scaling floor."

**Medium tier second (on Colab).** This is the primary replication target.
Debug methodology here until results are interpretable. If a finding doesn't
show up on a 7-9B model, that's a meaningful null result worth reporting.

**Large tier last (on Colab, if compute allows).** Scaling ceiling. Only run
after medium tier is fully analyzed.

**Note**: Some papers study base models (e.g., IOI studied GPT-2 Small).
The `paper_config.yaml` specifies `model_variant: "base"` or `"instruct"`
and the loader picks the right variant.

---

## The replication protocol

For each paper claim, the workflow has three phases:

### Phase A: Pipeline validation (Small tier, local MacBook)
1. **Operationalize**: State claim as testable hypothesis with success criterion.
2. **Implement**: Build or configure the experiment.
3. **Run on one small model locally** (e.g., Llama-3.2-1B): confirm the pipeline
   runs end-to-end — loading, extraction, caching, probing, saving results.
   Null results are expected and OK at this scale. You're testing plumbing, not
   the finding.
4. Run on remaining small-tier models to verify cross-model infrastructure.

### Phase B: Primary replication (Medium tier, Colab)
5. **Run on one medium model** (e.g., Llama-3.1-8B): this is where you debug
   methodology. If results aren't interpretable here, the approach needs work —
   don't scale further until they are.
6. Scale to all medium-tier models. Cache everything.

### Phase C: Scaling analysis (Large tier, Colab, if compute allows)
7. Run on large-tier models for scaling ceiling.
8. **Analyze**: Cross-model comparison. Effect sizes and CIs.
9. **Write up**: Add to `writeup/{paper_id}/draft.md`.

---

## Coding conventions

- **Python 3.10+**. Type hints everywhere.
- **torch** for all tensor ops. No mixing frameworks.
- **TransformerLens** for hook-based activation extraction where possible.
  Fall back to `nnsight` or raw HuggingFace hooks for unsupported models.
- **einops** for tensor reshaping.
- **Results are always saved** to `results/`. Use `torch.save()` for tensors, JSON for metrics.
- **Reproducibility**: Set seeds everywhere. Log all hyperparameters.
- **Memory**: `torch.no_grad()` and `model.eval()` for all inference.
  For 70B+, use 4-bit quantization via bitsandbytes.
- **Naming**: `{paper_id}/{model_key}/{experiment_id}_{timestamp}.pt`.

---

## Checkpoint everything — assume runs WILL be interrupted

Long runs get killed constantly — Colab timeouts, laptop sleeps, OOM crashes.
Every pipeline must be designed so that interruption loses at most one item.

**Rules:**

- **Per-item saves**: Save after EACH item, not at the end:
  ```python
  for i, item in enumerate(items):
      out_path = results_dir / f"item_{i:04d}.pt"
      if out_path.exists():
          continue  # already done — skip
      result = expensive_compute(item)
      torch.save(result, out_path.with_suffix(".tmp"))
      out_path.with_suffix(".tmp").rename(out_path)  # atomic
  ```
- **Resume-aware loops**: Every long-running function checks what's done on
  disk before starting.
- **Atomic writes**: Write `.tmp`, then `os.rename()` to final path.
- **Progress manifests**: JSON file listing completed items per experiment.
- **Probe checkpoints**: Save each CV fold separately.
- **Activation extraction**: Save per-stimulus, not as one giant tensor.

---

## Git, GitHub, and multi-environment workflow

### The split: code in GitHub, data in Google Drive

Two things need to stay in sync across MacBook and Colab, and they have
different characteristics:

| What | Where | Why |
|------|-------|-----|
| **Code** (src/, tests/, config/, CLAUDE.md, etc.) | **GitHub repo** | Small files, needs version history, needs to be identical across environments |
| **Data** (activations, stimuli, probe weights, results) | **Google Drive** | Large files (GBs of tensors), no version history needed, needs to persist across Colab sessions |

These two storage layers should NEVER be mixed:
- Never commit activations or large tensors to GitHub.
- Never put source code only in Google Drive.

### GitHub repo setup (one-time)

```bash
# On MacBook — create repo and push
cd ~/projects/mechinterp-replication
git init
git remote add origin git@github.com:zachgoldfine44/mechinterp-replication.git

# .gitignore — keep data out of git
cat > .gitignore << 'EOF'
# Data and results (live on Google Drive, not in git)
data/
results/
figures/
activations/
checkpoints/
drive_results/

# Environment
.venv/
__pycache__/
*.pyc
.ipynb_checkpoints/

# Large files
*.pt
*.safetensors
*.bin
*.npy
*.npz

# OS
.DS_Store
EOF

git add .
git commit -m "Initial project structure"
git push -u origin main
```

### Google Drive data folder setup (one-time)

Create this folder structure in Google Drive (manually or via script):
```
Google Drive/
└── mechinterp-replication/
    ├── data/                  # Generated stimuli per paper
    ├── results/               # All experiment outputs
    ├── checkpoints/           # Probe weights, intermediate state
    └── activations/           # Cached activations (largest files)
```

### MacBook local setup

```bash
# Clone repo
cd ~/projects
git clone git@github.com:zachgoldfine44/mechinterp-replication.git
cd mechinterp-replication

# Create venv
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Symlink Google Drive data folder into the repo directory
ln -s ~/Library/CloudStorage/GoogleDrive-*/My\ Drive/mechinterp-replication ./drive_data

# Set environment variable (add to ~/.zshrc)
export MECHINTERP_DATA_ROOT="./drive_data"
```

### Colab notebook setup (top of every notebook)

```python
# ── 1. Mount Google Drive (data persistence) ──
from google.colab import drive
drive.mount('/content/drive')
DATA_ROOT = "/content/drive/MyDrive/mechinterp-replication"

# ── 2. Clone GitHub repo (latest code) ──
import os
REPO_DIR = "/content/mechinterp-replication"
if not os.path.exists(REPO_DIR):
    !git clone https://github.com/zachgoldfine44/mechinterp-replication.git {REPO_DIR}
else:
    !cd {REPO_DIR} && git pull

# ── 3. Install dependencies ──
!cd {REPO_DIR} && pip install -q -r requirements.txt

# ── 4. Set data root ──
os.environ["MECHINTERP_DATA_ROOT"] = DATA_ROOT

# ── 5. Run ──
%cd {REPO_DIR}
```

To pull code updates mid-session (e.g., after pushing a fix from MacBook):
```python
!cd {REPO_DIR} && git pull
```

### Environment detection (`src/utils/env.py`)

```python
import platform, os
from pathlib import Path

def detect_environment() -> str:
    """Returns 'colab', 'macbook', or 'other'."""
    try:
        get_ipython()
        if "google.colab" in str(type(get_ipython())):
            return "colab"
    except NameError:
        pass
    if platform.system() == "Darwin" and platform.processor() == "arm":
        return "macbook"
    return "other"

def get_data_root() -> Path:
    """Returns the root for all data/results (Google Drive path)."""
    env_override = os.environ.get("MECHINTERP_DATA_ROOT")
    if env_override:
        return Path(env_override)

    env = detect_environment()
    if env == "colab":
        return Path("/content/drive/MyDrive/mechinterp-replication")
    elif env == "macbook":
        return Path("./drive_data")
    else:
        return Path("./local_data")

def get_device() -> str:
    """Returns best available torch device for current environment."""
    import torch
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
```

### Git workflow rules

- **Commit after every meaningful unit of work.** Each commit should pass
  `pytest tests/ -q --fast`.
- **Never commit data files.** The `.gitignore` blocks them, but be vigilant
  about `git add .` — always review `git status` first.
- **Commit messages should be greppable**: `"Add probe_classification experiment"`
  not `"updates"`.
- **Push before switching environments.** If you're done working on MacBook and
  moving to Colab, push first. If you made code changes in a Colab notebook
  (rare — prefer making code changes locally), commit and push from Colab:
  ```python
  !cd {REPO_DIR} && git add -A && git commit -m "Fix extraction batch size" && git push
  ```
- **Pull at the start of every session.** Both locally and on Colab.
- **Don't edit code in Colab notebooks** unless it's a quick fix. The pattern
  is: develop code locally on MacBook → push → pull on Colab → run experiments.
  Colab is for execution, MacBook is for development.

### What lives where — summary

| Item | GitHub repo | Google Drive | .gitignore'd |
|------|:-----------:|:------------:|:------------:|
| `src/`, `tests/`, `config/` | ✓ | | |
| `CLAUDE.md`, `DESIGN.md`, `PROGRESS.md` | ✓ | | |
| `requirements.txt`, `.gitignore` | ✓ | | |
| `data/` (generated stimuli) | | ✓ | ✓ |
| `results/` (experiment outputs) | | ✓ | ✓ |
| `activations/` (cached tensors) | | ✓ | ✓ |
| `checkpoints/` (probe weights) | | ✓ | ✓ |
| `figures/` (generated plots) | | ✓ | ✓ |
| `writeup/` (draft markdown) | ✓ | | |

### Rules for multi-environment work

- **Never hardcode paths.** Always use `get_data_root()` from `src/utils/env.py`.
- **Save data to Google Drive, not local /tmp.** Colab wipes /tmp on disconnect.
- **Small models locally, big models on Colab.**
- **Generate stimuli locally, extract activations on Colab.** Stimulus generation
  doesn't need a GPU; activation extraction does.
- **Resume logic is especially critical on Colab** — per-item saves to Drive
  mean a killed session just picks up where it left off.
- **MPS compatibility**: Some TransformerLens operations don't work on MPS.
  Fall back to CPU for probe training locally — it's fast enough for small
  models.

---

## Critical rules to prevent mechinterp bugs

### Never cherry-pick models or concepts
Report ALL results, including nulls. Null results are findings.

### Validate probes before trusting them
Test on stimuli that don't name the concept, and on confound stimuli.

### Control for confounds
Verify models comprehend stimuli before probing internals.

### Report effect sizes, not just p-values
For every experiment: effect size, 95% CI, sample size, null baseline.

### Layer selection matters
Scan all layers; report both best layer and paper's recommended depth.

### Respect the paper's methodology before deviating
First replicate the exact method. If it fails, THEN try variations.
Record which method you used.

---

## Writeup structure

Final deliverable is `writeup/{paper_id}/draft.md`:

1. **Introduction**: Paper being replicated, why cross-model replication matters.
2. **Methods**: Model matrix, stimuli, technique config. Reproducible detail.
3. **Results by claim**: Table/figure per claim across all models.
4. **Cross-model analysis**: What generalizes vs. what's family/scale-specific.
5. **Discussion**: Universality implications. Limitations. Future work.
6. **Appendix**: Per-model details, hyperparameters, all figures.

---

## Example paper configs (for reference)

These show how different papers map onto the framework:

| Paper | Techniques | Experiment types | Concepts | Model variant |
|-------|-----------|-----------------|----------|---------------|
| Emotions (Sofroniew+ 2026) | probes, contrastive, steering | probe_classification → generalization → parametric_scaling → causal_steering | 12 emotions | instruct |
| Geometry of Truth (Marks+ 2023) | probes, representation_geometry | probe_classification → representation_geometry | true/false across domains | base |
| IOI (Wang+ 2022) | patching, attention, circuit_discovery | circuit_identification | indirect object identification | base |
| Steering Vectors (Turner+ 2023) | contrastive, steering | probe_classification → causal_steering | behavioral traits (sycophancy, etc.) | instruct |
| Toy Models of Superposition (Elhage+ 2022) | sae, representation_geometry | representation_geometry | synthetic features | train_from_scratch |

See `config/papers/` for full config files for each.
