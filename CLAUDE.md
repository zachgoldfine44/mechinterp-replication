# Mechinterp Replication Engine — Operational Guide

> **For humans**: If you're looking for an overview of what this project is
> and how to use it, see [README.md](README.md). If you want to contribute
> a replication or improve the harness, see [CONTRIBUTING.md](CONTRIBUTING.md).
>
> **This file** is the operational guide for Claude Code agents (and
> power-users who want to understand the internal conventions). It documents
> session workflow, commit cadence, storage layout, and coding standards
> that the autonomous development pipeline relies on.

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
* **Mech-interp guardrails**: `GOTCHAS.md` (preflight + post-result checklists — read before designing experiments and after every positive result)
* **Milestone log**: `PROGRESS.md` (read this first every session — big-picture state)
* **Running changelog**: `CHANGELOG.md` (append-only entry per work unit — your incremental memory)
* **Active paper**: `config/papers/{paper_id}/` — config + paper text + stimuli
* **Active paper config**: `config/papers/{paper_id}/paper_config.yaml`
* **Paper as ground truth**: `config/papers/{paper_id}/paper.md` (full paper text — see "Paper as oracle" below)
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
2. Read `PROGRESS.md` — milestone state.
3. Read the last ~30 entries of `CHANGELOG.md` — fine-grained recent history (what failed yesterday, what I tried last hour).
4. Check which paper is active: `cat config/active_paper.txt`
5. Read the active paper config: `cat config/papers/{paper_id}/paper_config.yaml`
6. **Read the paper itself**: `cat config/papers/{paper_id}/paper.md` (or skim if it's long). This is the oracle. Every claim, every success threshold, every methodology choice should be checkable against the paper.
7. Skim `GOTCHAS.md` if you're about to design an experiment (preflight checklist) or interpret a result (post-result checklist).
8. Run `pytest tests/ -v --fast 2>&1 | tail -30` — current test status.
9. **Make an initial commit at the start of a session** — even if it's just appending to CHANGELOG.md saying "session started, plan: X". This proves the commit pipe works and gives you a stable base to revert to.
10. Pick the next unchecked item from PROGRESS.md.
11. When you finish a unit of work: append to CHANGELOG.md, commit, push. **Commits should flow throughout the session, not pile up at the end.**

## Commit cadence rules (important)

The user explicitly wants to see work progressing in git history, not a single end-of-session dump. Specifically:

- **Initial commit** at the start of every session, even if it's just a CHANGELOG entry.
- **Commit after every meaningful unit of work** — a passing experiment, a finished refactor, a stimulus generation pass, a fixed test. If you find yourself with more than ~30 minutes of uncommitted changes during active development, that's too long.
- **Commit before moving to a new file** — don't pile multi-file changes into one commit unless they're intrinsically coupled.
- **Push after each commit** unless the user has said otherwise. Local-only commits don't show up in the user's web view of GitHub, so they think nothing is happening.
- **Each commit must pass `pytest tests/ -q --fast`** — no exceptions, no broken-tests-on-main.
- **Never `git add .` blindly.** Always `git status` first, then `git add` specific files. This is especially important now that small result files ARE meant to be committed (see "Storage layout" below) — be careful not to accidentally commit large `.pt` cache files.

## CHANGELOG.md vs PROGRESS.md

These are different documents serving different purposes:

| | PROGRESS.md | CHANGELOG.md |
|---|---|---|
| Cadence | Milestones (claim done, paper done, version bump) | Every commit / work unit / failed attempt |
| Style | Curated, big-picture, replaceable sections | Append-only running log |
| Editing | OK to rewrite a section when status changes | Never rewrite past entries; always append |
| Scope | "Where does this paper replication stand?" | "What did I try at 3pm and did it work?" |
| Length cap | Keep tight (~200 lines) | Grows forever; that's fine |

When in doubt, append to CHANGELOG.md. PROGRESS.md should not be touched on every commit — only when a milestone changes.

The user explicitly called out that PROGRESS.md was getting overwritten in a previous session. **Do not overwrite past PROGRESS.md content.** Treat its sections as updates-in-place for current state, never as a place to delete history. Move retired information to CHANGELOG.md or the writeup before deleting from PROGRESS.md.

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

**You MUST save the paper text into the repo at the start.** Without this,
future sessions and sub-agents will drift from the paper's methodology and
silently invent things. Specifically:

1. Get the paper as text (HTML → markdown, PDF → markdown via `pdftotext`
   or `pandoc`, or copy from Anthropic / arXiv directly).
2. Save it as `config/papers/{paper_id}/paper.md`. This is the **oracle**
   for everything that comes after.
3. Reference it in `paper_config.yaml`:
   ```yaml
   paper:
     id: "my_paper"
     title: "..."
     paper_text_path: "paper.md"   # relative to config/papers/{paper_id}/
   ```
4. **Every claim in `paper_config.yaml` should have a `paper_section` field
   pointing at the section/page of `paper.md` it comes from.** Future
   sessions need to be able to verify "did we actually do what the paper
   said?" without re-downloading the paper.
5. **Every methodology choice that copies the paper** (steering layer,
   alpha, sample count, evaluation prompt, judge model) **should have a
   citation comment** linking to where in `paper.md` it came from.

The pipeline loads `paper.md` once and exposes it via
`PaperConfig.paper_text` so experiments and critique agents can reference
it. The critique agents are explicitly instructed to compare results
against the paper as a sanity check.

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
- The paper text MUST live at `config/papers/{paper_id}/paper.md`. If it
  doesn't, that's the first thing to fix in any new session — without it,
  every methodology decision is a guess.
- Open `paper.md` regularly. When the user asks "what about preference
  steering?" or "how did they evaluate?", the answer is in the paper, not
  in your memory of the paper.
- Every claim row in `paper_config.yaml` should cite a paper section. When
  you implement a claim, re-read that section of the paper before writing
  code.
- First, validate the pipeline runs end-to-end on a small model locally.
  This catches infrastructure bugs for free before using Colab.
- Then, validate the methodology produces interpretable results on a medium
  model. If it doesn't work on any 7-9B model, the pipeline has a bug —
  don't scale garbage.
- Write the test for each experiment BEFORE implementing it.
- When results look wrong, check upstream: is the activation extraction
  correct? Are the probes/circuits correct? Is the stimulus set valid?
- When in doubt, **diff your methodology against the paper.** Most
  silent-divergence bugs come from "I forgot the paper used pairwise
  Elo, not absolute scoring" — exactly the kind of thing that has bitten
  us in past sessions.

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

### 4. Keep CHANGELOG.md and PROGRESS.md current

**Rules:**
- Append to CHANGELOG.md after every commit. One bullet, dated, including
  what worked, what didn't, and any accuracy numbers worth remembering.
- Update PROGRESS.md only on milestones (claim done, paper done, version
  bump). Treat its sections as live status; never delete history without
  first moving it to CHANGELOG or the writeup.
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

## Storage layout (single source of truth: the repo)

**Past sessions used a Google Drive symlink (`./drive_data`) and Colab
mount magic. That layer caused two problems**: results were invisible to
the user (they didn't know where to look), and nothing flowed into git
during a run. We've removed it. Everything now lives inside the repo,
with a clear distinction between what is committed vs. what is local-only.

### What lives where — current model

| Item | Path | Committed to git? | Notes |
|------|------|:-:|------|
| Source code | `src/`, `tests/`, `scripts/` | ✓ | |
| Process docs | `CLAUDE.md`, `DESIGN.md`, `PROGRESS.md`, `CHANGELOG.md`, `GOTCHAS.md` | ✓ | |
| Paper configs | `config/papers/{paper_id}/paper_config.yaml` | ✓ | |
| **Paper text (oracle)** | `config/papers/{paper_id}/paper.md` | ✓ | New — see "Paper as oracle" |
| Stimuli (small JSONs) | `config/papers/{paper_id}/stimuli/` | ✓ | Tracked so a fresh clone can run end-to-end |
| `result.json` per claim | `results/{paper_id}/{model_key}/{claim_id}/result.json` | ✓ | Whitelisted in `.gitignore` |
| Sanity-check + critique reports | `results/{paper_id}/{model_key}/{claim_id}/critiques/*.json` | ✓ | Whitelisted |
| Figures | `figures/{paper_id}/*.png` | ✓ | Whitelisted |
| Writeups | `writeup/{paper_id}/draft.md` | ✓ | |
| **Activation caches** | `local_data/cache/activations/{model_key}/*.npy` | ✗ | Local-only, regenerated on demand |
| **Probe weights** | `local_data/results/{paper_id}/{model_key}/{claim_id}/probes/*.pt` | ✗ | Heavy, regenerated on demand |
| **Per-stimulus activation files** | `local_data/results/{paper_id}/{model_key}/{claim_id}/activations/stimulus_*.pt` | ✗ | |

The principle: **anything that documents what happened gets committed;
anything that's a re-derivable cache stays local.** A fresh clone of the
repo plus a model download is enough to re-run everything end-to-end.

### Paths in code

`src/utils/env.py` exposes two functions:

- `get_data_root()` — root for **local-only** caches and heavy artifacts.
  Defaults to `<repo>/local_data/`. Can be overridden via
  `MECHINTERP_DATA_ROOT` for an external mount (e.g., a Colab disk you
  want to keep across sessions).
- `get_committed_artifacts_root()` — root for **git-tracked** result
  files. Always `<repo>/results/`. Never overridden.

Experiments that produce small artifacts (`result.json`, figures,
critique reports) write under `get_committed_artifacts_root()`.
Experiments that produce heavy artifacts (cached activations, probe
weights) write under `get_data_root()`.

### Colab usage

Colab is still supported for medium/large-tier runs, but the pattern is
much simpler now:

```python
import os, subprocess
REPO_DIR = "/content/mechinterp-replication"
if not os.path.exists(REPO_DIR):
    subprocess.run(["git", "clone",
                    "https://github.com/zachgoldfine44/mechinterp-replication.git",
                    REPO_DIR], check=True)
%cd {REPO_DIR}
!git pull
!pip install -q -r requirements.txt
# That's it. local_data/ lives in /content/mechinterp-replication/local_data/.
# It will not survive Colab disconnect — that's fine, the pipeline is
# resume-aware and the *important* artifacts get committed to git as we go.
```

If you want a Colab session's caches to survive disconnects, set
`MECHINTERP_DATA_ROOT=/content/drive/MyDrive/...` before importing
anything from `src.`. This is opt-in, not the default.

### Git workflow rules

- **Commit after every meaningful unit of work.** Each commit should pass
  `pytest tests/ -q --fast`.
- **`git add` specific files, never `git add .`** — even though large
  files are gitignored, an oversight is one stray commit away. Be
  particularly careful about `*.pt` files, `local_data/`, and the
  symlinked HF cache.
- **Commit messages should be greppable**: `"Add probe_classification
  experiment"` not `"updates"`. Use the `topic: short summary` format.
- **Push after every commit** unless the user says otherwise. The user
  cannot see local-only commits and will assume nothing is happening.
- **Pull at the start of every session.**
- **Don't edit code in Colab notebooks.** Pattern: develop locally → push
  → pull on Colab → run. Colab is for execution.
- **MPS compatibility**: Some TransformerLens operations don't work on
  MPS. Fall back to CPU for probe training locally — it's fast enough
  for small models.

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

### Run the live sanity checks; do not wait until the end to look at results
The pipeline runs `src.core.sanity_checks` after every experiment finishes
and writes the report to `results/{paper_id}/{model_key}/{claim_id}/sanity.json`.
It catches things like:
- confusion-matrix resolution artifacts ("0.800 = 12/15" trap that bit a
  past session — see CHANGELOG entry for that incident)
- chance-level accuracy reported as a positive result
- exactly-identical metrics across models (impossible coincidence)
- suspiciously round numbers (0.5, 0.8, 1.0)
- per-concept accuracy that's all the same (probe collapsed)
- success_metric that doesn't match what `paper_config.yaml` declared

If a sanity check fires, you must address it before moving on.
The pipeline also surfaces these warnings in stdout — read them, don't
let them scroll past.

### Run the critique agents at the end of every model
After each model finishes its full claim sweep, the pipeline calls
`src.core.critique` which spawns:
- A Claude-Sonnet critique agent reviewing the model's results against
  the paper text and GOTCHAS.md
- An optional ChatGPT critique agent doing the same (requires
  `OPENAI_API_KEY`; silently skipped if absent)
- An evaluator agent that reads both critiques and produces a prioritized
  list of next steps

Critique outputs land in `results/{paper_id}/{model_key}/critiques/` and
are committed to git so they show up in the user's view of the repo.

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
