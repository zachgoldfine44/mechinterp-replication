# TL;DR

To quick-start a replication of a mechinterp paper, copy and paste the following into Claude Code or Codex:

>Please replicate the Geometry of Truth paper using the zachgoldfine44/mechinterp-replication harness on the Qwen-2.5-1.5B-Instruct model locally, and then open a pull request to that repo with the results as a replication attempt when done.

It will likely take ~30-60 minutes to complete.

_Note: You can substitue the "Geometry of Truth" paper with any mechinterp paper you'd like to replicate._

Then, evaluate the results. Which parts of the paper were replicated successfully? Which weren't? What parts of the results are useful? What changes might make the replication attempt stronger? 

The next steps are up to you: make revisions, extend the replication to more models or different configurations, or submit a pull request to contribute your replication attempt as-is. 

We're excited to see what you do!

# Mechinterp Replication Harness

[![Tests](https://img.shields.io/badge/tests-211%20passing-brightgreen)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)]()

**An open-source harness for replicating mechanistic interpretability papers across open-source model families.**

Built so independent researchers can validate published findings, contribute replications, and help the community build a more complete picture of how AI systems work internally — making it more likely that these systems are well-understood and emerge safely.

---

## Why replication matters

Most mechanistic interpretability papers study a single model (often a proprietary one). The findings are fascinating — emotion concepts, truth directions, induction circuits — but we rarely know whether they generalize. Does the same circuit exist in Llama? In Gemma? At 1B scale vs. 70B?

**Independent replication is an important component of scientific progress**, but it's rare because:

- Setting up the tooling from scratch for each paper is expensive
- Cross-model comparison requires handling multiple architectures
- There's no standardized framework for running experiments and comparing results 

This harness exists to make these problems tractable. You feed it a link to a mechinterp paper, and it determines what claims to test, what stimuli to use, and what success looks like. And then it can run the experiments across a matrix of open-source models, catch common pitfalls, and produce a writeup.

The goal: **lower the barrier so that more papers get replicated more often, by more people, across more models.** The field benefits when findings are tested independently, null results are reported, and the community accumulates evidence about what's robust and what isn't.

---

## What's included

### Standardized Framework

- **6 generic experiment types** that cover many mechinterp papers: probe classification, generalization testing, parametric scaling, causal steering, circuit identification, representation geometry
- **8 reusable technique modules**: linear probes, contrastive activation addition (CAA), activation steering, activation patching, attention analysis, SAE feature extraction, logit lens, and circuit discovery
- **Configs for 3 model families x 3 scales**: Llama 3.1/3.2 (1B, 8B, 70B), Qwen 2.5 (1.5B, 7B, 72B), Gemma 2 (2B, 9B, 27B) — instruct and base variants
- **Automated sanity checks and guardrails** that can catch common issues (e.g., resolution traps, chance-level probes, confounded metrics) after every experiment
- **Critique agents** (Claude + ChatGPT) that review results against the original paper after each experiment and provides a critique. _Note: this requires adding API keys for these models.
_
### Completed Replications

Each row below is **one replication attempt** — a specific (paper, replication author, model-set) triple. Multiple replications of the same paper coexist under `config/papers/{paper}/replications/{replication_id}/` and each owns its own config, stimuli, results, and writeup. See [CONTRIBUTING.md](CONTRIBUTING.md) for how to submit your own.

<!-- replications-table:start -->
| Paper | Replicator | Replication ID | Models | Status | Key findings | Writeup | Config | AI peer reviews |
|-------|:---:|---|:---:|:---:|---|:---:|:---:|:---|
| [Emotion Concepts and their Function in a Large Language Model](https://transformer-circuits.pub/2026/emotions/index.html) | [@zachgoldfine44](https://github.com/zachgoldfine44) | `emotions-zachgoldfine44-6models` | 7 | v3.6 | All 4 representational claims replicate universally (probes 73-84%, valence geometry matching original). Causal influence is selective: sentiment shifts universally, pushback capitulation increases on 1-2 models (p=0.036), ethical compliance uninformative (0% floor). | [draft.md](writeup/emotions/emotions-zachgoldfine44-6models/draft.md) | [config/](config/papers/emotions/replications/emotions-zachgoldfine44-6models/) | [ChatGPT 7/10](writeup/emotions/emotions-zachgoldfine44-6models/reviews/chatgpt-extended-thinking.md) · [Claude 7/10](writeup/emotions/emotions-zachgoldfine44-6models/reviews/claude-opus-4-6-extended-thinking.md) · [Gemini 8.5/10](writeup/emotions/emotions-zachgoldfine44-6models/reviews/gemini-3-1-pro.md) |
| [The Geometry of Truth: Emergent Linear Structure in LLM Representations of True/False Datasets](https://arxiv.org/abs/2310.06824) | [@zachgoldfine44](https://github.com/zachgoldfine44) | `geometry_of_truth-zachgoldfine44-qwen_1_5b` | 1 | v0.1 partial | Within-distribution probe replicates cleanly (0.99 on cities, layer 12 — matches paper 13B/70B despite being 10x smaller). Cross-dataset generalization does NOT replicate at 1.5B: probe collapses to trivial predictions on sp_en_trans (92/100 "true"), flips sign on neg_cities (replicates paper's Fig 3c orthogonal-directions finding), collapses to all-false on larger_than. Consistent with paper's "at sufficient scale" framing. | [draft.md](writeup/geometry_of_truth/geometry_of_truth-zachgoldfine44-qwen_1_5b/draft.md) | [config/](config/papers/geometry_of_truth/replications/geometry_of_truth-zachgoldfine44-qwen_1_5b/) |  |
| *Your replication here* | *(you)* | `{paper}-{handle}-{model_scope}` | | | | | | |
<!-- replications-table:end -->

We want this table to grow, with multiple independent replications per paper. If you replicate a paper using this harness, please open a PR to add your replication (see [Contributing](#contributing)).

_Note: The table above is regenerated from each replication's `metadata.yaml` by [`scripts/generate_replications_table.py`](scripts/generate_replications_table.py). Please edit the metadata instead of hand-editing the table_.

Every replication submitted to this repo is reviewed by three frontier LLM referees (currently ChatGPT with extended thinking, Claude Opus with extended thinking, and Gemini Pro) using a standardized prompt. The reviews are stored alongside the writeup and linked from this table. The reviews are not a substitute for human peer review — they're a low-cost, high-signal way to stress-test claims, catch methodological issues, and give authors immediate feedback. See [CONTRIBUTING.md](CONTRIBUTING.md#ai-review-policy) for the protocol.


---

## Contributing

We want this to become a hub for independent mechinterp replications. See [CONTRIBUTING.md](CONTRIBUTING.md) for details. The two main ways to contribute:

1. **Replicate a paper**: point an LLM at this repo and include a paper you want to replicate, then open a PR when it's done
2. **Improve the harness**: please open a PR or create an issue to create or request new experiment types, new model families, better sanity checks, improved documentation, new tests, etc.

Every replication that lands here — whether it confirms, partially confirms, extends, or refutes the original findings — adds to the community's understanding of what's robust or not in mechanistic interpretability.

---

## Acknowledgments

- **Callum McDougall, Neel Nanda, and the [ARENA 3.0](https://www.arena.education/) curriculum team** for producing excellent mechanistic interpretability educational materials that informed the techniques and best practices used throughout this harness
- **Siddharth Mishra-Sharma** for ["Long-running Claude for scientific computing"](https://www.anthropic.com/research/long-running-Claude), which inspired the idea of building a science replication harness and provided much of the structural scaffolding (incremental checkpointing, changelog discipline, paper-as-oracle pattern)
- **TransformerLens** and **HuggingFace** teams whose tools made cross-model experimentation easier
- **Johnny Lin and the Neuronpedia.org team** for encouraging and supporting further development of this harness, and for creating and hosting useful tools for doing open-source mechinterp research
- The original authors of papers replicated using this harness, starting with **Sofroniew et al.** for the [Emotion Concepts](https://transformer-circuits.pub/2026/emotions/index.html) paper

---

## License

[MIT](LICENSE) — use it, fork it, build on it.



---
---
---
---

## How to replicate a new paper (for agents)

The full workflow is documented in [CLAUDE.md](CLAUDE.md) (the operational guide) and [DESIGN.md](DESIGN.md) (the architecture doc). Here's the summary:

### Step 1: Read the paper and extract claims

Identify 3-7 key falsifiable claims with quantitative success criteria. Save the paper text as `config/papers/{paper_id}/paper.md` — this is the ground truth everything checks against.

### Step 2: Pick a replication ID and create its config

Pick an ID of the form `{paper_id}-{github_handle}-{model_scope}` (e.g. `my_paper-alice-qwen_1_5b`; full convention in [CONTRIBUTING.md](CONTRIBUTING.md)). Then create two small files under `config/papers/{paper_id}/replications/{replication_id}/`:

- `paper_config.yaml` — the claims this attempt tests, mapped to generic experiment types and success thresholds
- `metadata.yaml` — row data for the README table (regenerated by [`scripts/generate_replications_table.py`](scripts/generate_replications_table.py); don't hand-edit the table)

Here's the `paper_config.yaml` shape:

```yaml
paper:
  id: "my_paper"
  title: "Paper Title"
  authors: "Author et al."
  url: "https://..."
  original_model: "The model the paper studied"
  model_variant: "instruct"  # or "base"

replication:
  id: "my_paper-alice-qwen_1_5b"
  replicator: "Alice"
  github_handle: "alice"

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

Define stimuli in `config/papers/{paper_id}/replications/{replication_id}/stimuli_config.yaml`, or place JSON files directly under `config/papers/{paper_id}/stimuli/` when the data is shared across all replications of the paper.

### Step 4: Run

```bash
python -m src.core.pipeline --paper my_paper --replication my_paper-alice-qwen_1_5b --model llama_1b    # one small model
python -m src.core.pipeline --paper my_paper --replication my_paper-alice-qwen_1_5b --tier small        # all small models
python -m src.core.pipeline --paper my_paper --replication my_paper-alice-qwen_1_5b --tier medium       # all medium models
```

If `--replication` is omitted, the pipeline reads the ID from the yaml's `replication.id` field — so you can leave it off after setting it there once.

### Step 5: Write up

Results land in `results/{paper_id}/{replication_id}/{model_key}/`. The framework generates cross-model comparisons. Write your findings in `writeup/{paper_id}/{replication_id}/draft.md`, and track milestones / work units in the sibling `PROGRESS.md` + `CHANGELOG.md`.


---

## Repo structure

Everything paper-specific or replication-specific is namespaced under
`{paper_id}/` and `{paper_id}/{replication_id}/`, so multiple
independent replications of the same paper can coexist without
colliding on configs, stimuli, results, writeups, or logs.

```
.
├── README.md                       # You are here
├── CLAUDE.md                       # Operational guide (for Claude Code agents)
├── CONTRIBUTING.md                 # How to contribute replications or code
├── DESIGN.md                       # Framework architecture
├── GOTCHAS.md                      # Mechinterp guardrails & checklists
├── CHANGELOG.md                    # Harness changelog (framework-level changes only)
├── LICENSE                         # MIT
├── requirements.txt                # Python dependencies
├── config/
│   ├── active_paper.txt            # Default paper_id for the pipeline
│   ├── models.yaml                 # Model matrix (families × sizes)
│   └── papers/                     # One folder per paper studied
│       └── emotions/
│           ├── paper.md            # Paper text — the ground-truth oracle
│           ├── paper_config.yaml   # (optional) paper-level legacy config
│           ├── stimuli_config.yaml # (optional) paper-level legacy stimuli
│           ├── stimuli/            # Generated/curated stimulus JSONs (shared)
│           └── replications/       # One subfolder per replication attempt
│               └── emotions-zachgoldfine44-6models/
│                   ├── paper_config.yaml   # This attempt's claims + thresholds
│                   ├── stimuli_config.yaml # This attempt's stimuli config
│                   └── metadata.yaml       # Row data for the README table
├── src/
│   ├── core/                       # Framework engine
│   │   ├── pipeline.py             # Orchestrator: config → run → analyze
│   │   ├── config_loader.py        # Loads paper + replication configs
│   │   ├── experiment.py           # Abstract experiment base class
│   │   ├── claim.py                # Claim + ExperimentResult dataclasses
│   │   ├── sanity_checks.py        # Post-experiment artifact detection
│   │   ├── critique.py             # Claude + ChatGPT critique agents
│   │   └── review.py               # Post-run AI peer-review prompt builder
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
├── scripts/                        # CLI helpers (review prompts, README table regen, …)
├── tests/                          # ~211 unit tests + integration suite
├── writeup/                        # Per-replication writeups
│   └── emotions/
│       └── emotions-zachgoldfine44-6models/
│           ├── draft.md            # The writeup
│           ├── PROGRESS.md         # Milestone state for this replication
│           ├── CHANGELOG.md        # Per-work-unit log for this replication
│           └── reviews/            # AI peer-review responses
├── results/                        # Per-replication, per-model result artifacts
│   └── emotions/
│       └── emotions-zachgoldfine44-6models/
│           └── {model_key}/        # result.json, sanity.json, critiques/
├── figures/                        # Per-replication publication-ready plots
│   └── emotions/
│       └── emotions-zachgoldfine44-6models/
└── local_data/                     # Gitignored: activation caches, probe weights
```

Two different scopes of log coexist: the **root `CHANGELOG.md`** is the
harness changelog — framework changes only (`src/`, `tests/`,
`scripts/`, repo-level docs). Scientific progress on a specific
replication lives in that replication's own
`writeup/{paper}/{id}/PROGRESS.md` + `CHANGELOG.md`.


---

## The model matrix

The initial harness is set up to test findings across three model families at three scales:

| Family | Small (local CPU/MPS) | Medium (single GPU) | Large (multi-GPU) |
|--------|:---:|:---:|:---:|
| **Llama 3.1/3.2** | 1B | 8B | 70B |
| **Qwen 2.5** | 1.5B | 7B | 72B |
| **Gemma 2** | 2B | 9B | 27B |

Small-tier models run locally on a laptop for pipeline validation. Medium-tier is the primary replication target. Large-tier is for scaling analysis if compute allows. We hope to see support for many more models added to this harness.

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
