# TL;DR

To quick-start a replication of a mechinterp paper, copy and paste the following into Claude Code or Codex:

>Please replicate the Geometry of Truth paper using the zachgoldfine44/mechinterp-replication harness on the Qwen-2.5-1.5B-Instruct model locally, and then open a pull request to that repo with the results as a replication attempt when done.

It will likely take ~30-60 minutes to complete.

You can substitue the "Geometry of Truth" paper with any mechinterp paper you'd like to replicate. 

From there, you can evaluate the results, ask Claude Code or Codex to make revisions, ask Claude Code or Codex to extend the replication to more models or different configurations, or submit a pull request to contribute your replication attempt. 

**When the replication finishes, the pipeline will print a review prompt and write an artifact bundle to `local_data/reviews/{paper}/{replication}/self_review_bundle.md`. Paste the prompt and upload the bundle into an AI *other than* the one that just helped you run the replication** — if you used Claude Code, try ChatGPT or Gemini; if you used Codex, try Claude or Gemini. This gives you a neutral second-opinion review on strengths and weaknesses before you open a PR, catching blind spots that the AI you co-wrote with is more likely to miss.

We're excited to see what papers and experiments you replicate!

# Mechinterp Replication Harness

[![Tests](https://img.shields.io/badge/tests-212%20passing-brightgreen)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)]()

**An open-source harness for replicating mechanistic interpretability papers across open-source model families.**

Built so independent researchers can validate published findings, contribute replications, and help the community build a more complete picture of how AI systems work internally — making it more likely that these systems are well-understood and emerge safely.

---

## Why replication matters

Most mechanistic interpretability papers study a single model (often a proprietary one). The findings are fascinating — emotion concepts, truth directions, induction circuits — but we rarely know whether they generalize. Does the same circuit exist in Llama? In Gemma? At 1B scale vs. 70B?

**Independent replication is how science self-corrects**, but it's rare in mechinterp because:

- Setting up the tooling from scratch for each paper is expensive
- Cross-model comparison requires handling multiple architectures
- There's no standard framework for "run the same experiment on six models and compare"

This harness exists to make that tractable. You feed it a link to a mechinterp paper, and it determines what claims to test, what stimuli to use, and what success looks like. And then it runs the experiments across a matrix of open-source models, catches common pitfalls, and produces a writeup.

The goal: **lower the barrier so that more papers get replicated more often, by more people, across more models.** The field benefits when findings are tested independently, null results are reported, and the community accumulates evidence about what's robust and what isn't.

---

## What's included

### Framework

- **6 generic experiment types** that cover many mechinterp papers: probe classification, generalization testing, parametric scaling, causal steering, circuit identification, representation geometry
- **8 reusable technique modules**: linear probes, contrastive activation addition (CAA), activation steering, activation patching, attention analysis, SAE feature extraction, logit lens, and circuit discovery
- **3 model families x 3 scales**: Llama 3.1/3.2 (1B, 8B, 70B), Qwen 2.5 (1.5B, 7B, 72B), Gemma 2 (2B, 9B, 27B) — instruct and base variants
- **Automated sanity checks and guardrails** that catch common issues (resolution traps, chance-level probes, confounded metrics) after every experiment
- **Critique agents** (Claude + ChatGPT (optional)) that review results against the original paper after each experiment and provides a critique

### Completed replications

Each row below is **one replication attempt** — a specific (paper, replicator, model-set) triple. Multiple replications of the same paper coexist under `config/papers/{paper}/replications/{replication_id}/` and each owns its own config, stimuli, results, and writeup. See [CONTRIBUTING.md](CONTRIBUTING.md) for how to submit your own.

The table below is regenerated from each replication's `metadata.yaml` by [`scripts/generate_replications_table.py`](scripts/generate_replications_table.py) — don't hand-edit it, edit the metadata.

<!-- replications-table:start -->
| Paper | Replicator | Replication ID | Models | Status | Key findings | Writeup | Config | AI peer reviews |
|-------|:---:|---|:---:|:---:|---|:---:|:---:|:---|
| [Emotion Concepts and their Function in a Large Language Model](https://transformer-circuits.pub/2026/emotions/index.html) | [@zachgoldfine44](https://github.com/zachgoldfine44) | `emotions-zachgoldfine44-6models` | 7 | v3.6 | All 4 representational claims replicate universally (probes 73-84%, valence geometry matching original). Causal influence is selective: sentiment shifts universally, pushback capitulation increases on 1-2 models (p=0.036), ethical compliance uninformative (0% floor). | [draft.md](writeup/emotions/emotions-zachgoldfine44-6models/draft.md) | [config/](config/papers/emotions/replications/emotions-zachgoldfine44-6models/) | [ChatGPT 7/10](writeup/emotions/emotions-zachgoldfine44-6models/reviews/chatgpt-extended-thinking.md) · [Claude 7/10](writeup/emotions/emotions-zachgoldfine44-6models/reviews/claude-opus-4-6-extended-thinking.md) · [Gemini 8.5/10](writeup/emotions/emotions-zachgoldfine44-6models/reviews/gemini-3-1-pro.md) |
| [The Geometry of Truth: Emergent Linear Structure in Large Language Model Representations of True/False Datasets](https://arxiv.org/abs/2310.06824) | [@tulaneadam](https://github.com/tulaneadam) | `geometry_of_truth-tulaneadam-qwen_1_5b` | 1 | v1 | Basic truth classification replicates cleanly on Qwen-2.5-1.5B-Instruct (probe accuracy 0.97 vs. 0.70 threshold). Cross-dataset generalization fails at the 0.70 threshold (DD 0.50). Causal steering measured via preference-correlation metric. | [draft.md](writeup/geometry_of_truth/geometry_of_truth-tulaneadam-qwen_1_5b/draft.md) | [config/](config/papers/geometry_of_truth/replications/geometry_of_truth-tulaneadam-qwen_1_5b/) |  |
| *Your replication here* | *(you)* | `{paper}-{handle}-{model_scope}` | | | | | | |
<!-- replications-table:end -->

We want this table to grow, with multiple independent replications per paper. If you replicate a paper using this harness, please open a PR to add your replication (see [Contributing](#contributing)).

Every replication submitted to this repo is reviewed by three frontier LLM referees (currently ChatGPT with extended thinking, Claude Opus with extended thinking, and Gemini Pro) using a standardized prompt. The reviews are stored alongside the writeup and linked from this table. The reviews are not a substitute for human peer review — they're a low-cost, high-signal way to stress-test claims, catch methodological issues, and give authors immediate feedback. See [CONTRIBUTING.md](CONTRIBUTING.md#ai-review-policy) for the protocol.

---

## Quick start (for humans)

### 1. Clone and install

```bash
git clone https://github.com/zachgoldfine44/mechinterp-replication.git
cd mechinterp-replication

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run the smoke test

```bash
pytest tests/test_smoke.py -v
```

### 3. Run the emotions replication on a small model

Start with a fast smoke test (~2 min on Apple Silicon / ~5 min on CPU):

```bash
# Qwen 2.5 1.5B doesn't require HuggingFace authentication.
# --fast mode uses 2 concepts and 10 stimuli per concept — good for
# verifying the pipeline runs end-to-end, not for reporting numbers.
python -m src.core.pipeline --paper emotions --model qwen_1_5b --fast
```

For a real replication run with all 15 concepts and 25 stimuli per concept (~30--60 min on Apple Silicon / longer on pure CPU), drop the `--fast` flag:

```bash
python -m src.core.pipeline --paper emotions --model qwen_1_5b
```

Either command will:
- Load the paper config from `config/papers/emotions/paper_config.yaml`
- Download Qwen-2.5-1.5B-Instruct from HuggingFace (~3 GB, cached after first run)
- Run all six claims: probe classification, generalization, valence geometry, parametric intensity, causal steering, preference steering
- Write `result.json` + `sanity.json` for each claim under `local_data/results/emotions/qwen_1_5b/`
- Optionally run critique agents at the end (Claude requires `ANTHROPIC_API_KEY`; ChatGPT requires `OPENAI_API_KEY`; both are skipped gracefully if missing)

### 4. Run the full test suite

```bash
pytest tests/ -q --fast   # ~211 tests pass, ~20 skipped (integration), <5s
```

---

## How to replicate a new paper (for agents)

The full workflow is documented in [CLAUDE.md](CLAUDE.md) (the operational guide) and [DESIGN.md](DESIGN.md) (the architecture doc). Here's the summary:

### Step 1: Read the paper and extract claims

Identify 3-7 key falsifiable claims with quantitative success criteria. Save the paper text as `config/papers/{paper_id}/paper.md` — this is the ground truth everything checks against.

### Step 2: Create a paper config

Create `config/papers/{paper_id}/paper_config.yaml` mapping each claim to a generic experiment type:

```yaml
paper:
  id: "my_paper"
  title: "Paper Title"
  authors: "Author et al."
  url: "https://..."
  original_model: "The model the paper studied"
  model_variant: "instruct"  # or "base"

claims:
  - id: "claim_1"
    description: "Linear probes classify concepts above chance"
    experiment_type: "probe_classification"
    paper_section: "Section 3.1"  # cite the paper
    params:
      concept_set: ["concept_a", "concept_b"]
      n_stimuli_per_concept: 50
    success_metric: "probe_accuracy"
    success_threshold: 0.70
```

### Step 3: Create stimuli

Define stimuli in `config/papers/{paper_id}/stimuli_config.yaml` or place JSON files directly in `config/papers/{paper_id}/stimuli/`.

### Step 4: Run

```bash
python -m src.core.pipeline --paper my_paper --model llama_1b    # one small model
python -m src.core.pipeline --paper my_paper --tier small         # all small models
python -m src.core.pipeline --paper my_paper --tier medium        # all medium models
```

### Step 5: Write up

Results land in `results/{paper_id}/{model_key}/`. The framework generates cross-model comparisons. Write your findings in `writeup/{paper_id}/draft.md`.

---

## Repo structure

```
.
├── README.md                       # You are here
├── CLAUDE.md                       # Operational guide (for Claude Code agents)
├── DESIGN.md                       # Framework architecture
├── GOTCHAS.md                      # Mechinterp guardrails & checklists
├── CONTRIBUTING.md                 # How to contribute replications or code
├── config/
│   ├── models.yaml                 # Model matrix (families x sizes)
│   └── papers/                     # One folder per paper
│       └── emotions/               # Completed: Anthropic emotions paper
│           ├── paper_config.yaml   # Claims, experiments, success criteria
│           ├── paper.md            # Paper text (ground truth)
│           ├── stimuli_config.yaml # Stimulus definitions
│           └── stimuli/            # Generated/curated stimulus JSONs
├── src/
│   ├── core/                       # Framework engine
│   │   ├── pipeline.py             # Orchestrator: config → run → analyze
│   │   ├── experiment.py           # Abstract experiment base class
│   │   ├── claim.py                # Claim + ExperimentResult dataclasses
│   │   ├── sanity_checks.py        # Post-experiment artifact detection
│   │   └── critique.py             # Claude + ChatGPT critique agents
│   ├── experiments/                # 6 generic experiment types
│   │   ├── probe_classification.py
│   │   ├── generalization_test.py
│   │   ├── parametric_scaling.py
│   │   ├── causal_steering.py
│   │   ├── circuit_identification.py
│   │   └── representation_geometry.py
│   ├── techniques/                 # 8 reusable mechinterp building blocks
│   │   ├── probes.py
│   │   ├── contrastive.py
│   │   ├── steering.py
│   │   ├── patching.py
│   │   ├── attention.py
│   │   ├── sae.py
│   │   ├── logit_lens.py
│   │   └── circuit_discovery.py
│   ├── models/                     # Model loading (HF, TransformerLens, nnsight)
│   ├── analysis/                   # Cross-model comparison & visualization
│   └── utils/                      # Activations, datasets, caching, metrics
├── tests/                          # ~211 unit tests + integration suite
├── writeup/                        # Replication writeups (one per paper)
│   └── emotions/draft.md
├── results/                        # Per-paper, per-model result artifacts
└── figures/                        # Publication-ready plots
```

---

## The model matrix

The harness tests findings across three model families at three scales:

| Family | Small (local CPU/MPS) | Medium (single GPU) | Large (multi-GPU) |
|--------|:---:|:---:|:---:|
| **Llama 3.1/3.2** | 1B | 8B | 70B |
| **Qwen 2.5** | 1.5B | 7B | 72B |
| **Gemma 2** | 2B | 9B | 27B |

Small-tier models run locally on a laptop for pipeline validation. Medium-tier is the primary replication target. Large-tier is for scaling analysis if compute allows.

---

## Built-in safety nets

### Sanity checks (after every experiment)

The harness runs [10 automated checks](src/core/sanity_checks.py) after each experiment and writes a `sanity.json` report. Examples:

- **Resolution artifact detection**: catches when a metric like `diagonal_dominance = 0.800` is actually `12/15` because there are only 2 test samples per concept (this happened during the emotions replication and took a while to debug manually, but is now caught instantly)
- **Chance-level probe warning**: flags when probe accuracy is within 0.05 of random guessing
- **Negative-control contamination**: errors if a control condition shows an effect as large as the real condition
- **Stimulus count mismatch**: e.g., warns when the config says 50 stimuli per concept but only 25 loaded

### Critique agents (after each model)

After each model finishes its claim sweep, [critique agents](src/core/critique.py) are spawned:

1. **Claude critic** reviews results against the paper text and GOTCHAS.md
2. **ChatGPT critic** (optional, requires `OPENAI_API_KEY`) does the same independently
3. **Evaluator** reads both critiques and produces a prioritized list of concerns

This automates the "paste results into ChatGPT and ask what's wrong" loop that researchers do manually.

### Mechinterp guardrails

[GOTCHAS.md](GOTCHAS.md) is a comprehensive checklist of ways to fool yourself in mechinterp research: tokenization traps, metric saturation, cherry-picking, probe confounds, steering illusions, and more. The harness references it during experiment design and the critique agents use it to ground their reviews.

---

## Contributing

We want this to become a hub for independent mechinterp replications. See [CONTRIBUTING.md](CONTRIBUTING.md) for details. The two main ways to contribute:

1. **Replicate a paper** — point an LLM at this repo and include a paper you want to replicate, then open a PR when it's done
2. **Improve the harness** — please open a PR or create an issue to create or request new experiment types, new model families, better sanity checks, improved documentation, new tests, etc.

Every replication that lands here — whether it confirms, partially confirms, or refutes the original findings — adds to the community's understanding of what's robust in mechanistic interpretability.

---

## Acknowledgments

- **Callum McDougall, Neel Nanda, and the [ARENA 3.0](https://www.arena.education/) curriculum team** for producing excellent mechanistic interpretability educational materials that informed the techniques and best practices used throughout this harness
- **Siddharth Mishra-Sharma** for ["Long-running Claude for scientific computing"](https://www.anthropic.com/research/long-running-Claude), which inspired the idea of building a science replication harness and provided much of the structural scaffolding (incremental checkpointing, changelog discipline, paper-as-oracle pattern)
- **TransformerLens** and **HuggingFace** teams whose tools made cross-model experimentation possible
- The original authors of papers replicated using this harness, starting with **Sofroniew et al.** for the [Emotion Concepts](https://transformer-circuits.pub/2026/emotions/index.html) paper

---

## License

[MIT](LICENSE) — use it, fork it, build on it.
