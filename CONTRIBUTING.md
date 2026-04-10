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

### AI review policy

After your PR is merged, the repo maintainer runs **three AI referees** on
your replication:

- **ChatGPT** with extended thinking
- **Claude Opus** with extended thinking
- **Gemini Pro** with extended thinking

Each referee gets the exact same two-prompt protocol — first a 1-to-10
scoring request, then a follow-up asking for a harsher peer-review-style
referee report with major/minor concerns and an accept/reject
recommendation. The prompts and the raw responses are committed to
`writeup/{paper_id}/reviews/` alongside the draft, and the scores are
surfaced in the [Completed replications](README.md#completed-replications)
table in the README.

**Why the maintainer runs the reviews, not the submitter:**

1. **Comparability.** If the score is to mean anything when comparing
   replications, the same three reviewers with the same prompts have to
   run on every submission. Otherwise a "7/10" from one submitter isn't
   commensurable with an "8/10" from another.
2. **Anti-cherry-picking.** A submitter with access to their own reviews
   has an obvious incentive to only submit runs that gave favorable
   scores. Maintainer-run reviews eliminate this.
3. **Lower contribution barrier.** Not every researcher has API access to
   all three frontier providers.
4. **Archival integrity.** Reviews are pinned to a specific commit hash
   of the writeup, so a later edit doesn't invalidate them — the full
   history stays in git.

**What the reviews are not:**

- They are **not** a substitute for human peer review at a real venue.
- They are **not** a gatekeeper. The PR merge decision is based on whether
  the replication follows the methodology laid out in `CLAUDE.md` and
  `GOTCHAS.md`, not on the AI review scores. A harshly reviewed paper can
  still be a valuable contribution if the methodology is sound.
- They are **not** authoritative. Different reviewers will disagree; look
  at the full reports, not just the scores.

**What the reviews are:**

- A low-cost, high-signal way to stress-test claims against three
  independent frontier models.
- Useful feedback for authors to improve future drafts.
- A transparent record that each replication has been externally
  scrutinized in a reproducible way.

The exact prompt text and reviewer model identifiers live in the
[emotions reviews directory](writeup/emotions/reviews/README.md) as a
concrete example. The prompt protocol may evolve over time; when it does,
older reviews stay unchanged and new ones use the updated protocol (the
protocol version is recorded in each review file's header).

If you disagree with a review and want to respond, feel free to open a
follow-up PR adding an `author-response.md` to the `reviews/` directory.
Dialog is welcome.

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
