# Contributing to the Mechinterp Replication Harness

Thank you for your interest in contributing. The goal of this project is to
make it easy for independent researchers to replicate mechanistic
interpretability papers across open-source models, and to build a shared
repository of replications the community can learn from.

There are two main ways to contribute:

---

## 1. Submit a paper replication

This is the primary contribution we're hoping for. You've replicated (or
attempted to replicate) a mechinterp paper using this harness, and you want
to add your replication to the repo so others can see the results and build
on your work.

### What to include in your PR

```
config/papers/{paper_id}/
    paper_config.yaml       # Claims, experiments, success criteria
    paper.md                # Full paper text (markdown) for oracle reference
    stimuli_config.yaml     # Stimulus definitions
    stimuli/                # JSON stimulus files (keep small; generated
                            #   data can be regenerated from the config)

writeup/{paper_id}/
    draft.md                # Your writeup (methods, results, discussion)

results/{paper_id}/         # Only the small documenting artifacts:
    {model_key}/
        {claim_id}/
            result.json     # Metric outputs
            sanity.json     # Sanity-check report
        critiques/
            evaluator.json  # Critique agent outputs (if you ran them)
```

**Do not include** large files (activation caches, probe weights, `.pt`
files). The `.gitignore` blocks them, but double-check `git status` before
committing.

### Step-by-step

1. **Fork** the repo and create a branch: `git checkout -b replicate/{paper_id}`
2. **Read the paper** and save it as `config/papers/{paper_id}/paper.md`
3. **Extract 3-7 claims** and create `paper_config.yaml` (see the
   [emotions config](config/papers/emotions/paper_config.yaml) as a template)
4. **Create stimuli** — either hand-written JSONs or a `stimuli_config.yaml`
   that generates them
5. **Run the pipeline** on at least one small model locally:
   ```bash
   python -m src.core.pipeline --paper {paper_id} --model qwen_1_5b
   ```
6. **Run on medium models** if you have GPU access:
   ```bash
   python -m src.core.pipeline --paper {paper_id} --tier medium
   ```
7. **Write up your findings** in `writeup/{paper_id}/draft.md`. Include:
   - Which claims replicated and which didn't
   - Cross-model comparison (what generalizes, what's model-specific)
   - Limitations of your replication
   - Anything surprising
8. **Open a PR** with a clear title like: `Replication: {Paper Title} ({Author} {Year})`

### What makes a good replication

- Tests at least 2 models (ideally 3+, across families)
- Reports null results honestly — a claim that doesn't replicate is still
  a finding
- Includes sanity-check reports showing no artifact warnings
- Discusses limitations and alternative explanations
- Cites specific paper sections for each claim (using the `paper_section`
  field in the config)

Partial replications are welcome. If you only tested claims 1-3 out of 6,
that's still valuable — note what's left undone and someone else may pick
it up.

---

## 2. Improve the harness

If you want to contribute to the framework itself rather than (or in
addition to) a replication, here's what we'd find most useful:

### High-value contributions

- **New experiment types** — if a paper's methodology doesn't fit the
  existing 6 generic types, add a new one in `src/experiments/`. It must
  inherit from the `Experiment` base class and include tests.
- **New technique modules** — reusable building blocks in
  `src/techniques/` (e.g., representation engineering, probing with
  concept bottleneck models, improved circuit discovery methods).
- **New model families** — add entries to `config/models.yaml` and verify
  that `src/models/loader.py` and `src/utils/activations.py` handle the
  architecture correctly.
- **Better sanity checks** — if you've been bitten by an artifact the
  existing checks don't catch, add it to `src/core/sanity_checks.py`.
- **Documentation and tutorials** — especially "How to replicate paper X"
  walkthroughs that help onboard new contributors.

### Code standards

- **Python 3.10+** with type hints on all public functions
- **Tests required** for new features. Must pass `pytest tests/ -q --fast`
- **Follow existing patterns** — look at how similar features are
  implemented before starting from scratch
- **One logical change per commit**, with a descriptive message
- **No large binary files** in git (`.pt`, `.npy`, `.safetensors`, etc.)
- **Review [GOTCHAS.md](GOTCHAS.md)** before implementing new
  mechinterp experiments

### Running tests

```bash
# Fast mode (default for development — uses minimal data)
pytest tests/ -q --fast

# Full suite including integration tests (requires model download)
pytest tests/ -v --integration
```

---

## Code of conduct

Be kind and constructive. Null results are welcome. Critiques of methods
should be specific and actionable. The goal is to collectively improve our
understanding of how models work, not to score points.

---

## Questions?

Open an issue on GitHub. If you're unsure whether a paper is a good
candidate for replication, or how to handle a tricky methodology, ask —
we'd rather help you succeed than have you struggle in silence.
