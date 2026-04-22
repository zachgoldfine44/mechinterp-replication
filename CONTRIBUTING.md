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

### Replication IDs and the per-replication layout

The repo is organized so that **multiple independent replications of the
same paper coexist** — so ten people can each submit their own attempt at
Geometry of Truth without colliding on configs, stimuli, results, or the
README row. Every replicator picks a **replication ID** and everything
they own lives under that namespace.

**Replication ID convention:** `{paper_id}-{github_handle}-{model_scope}`

Examples:
- `emotions-zachgoldfine44-6models` —  6 models across the small, medium, and large tiers
- `geometry_of_truth-zachgoldfine44-qwen_1_5b` — one model
- `geometry_of_truth-alice-llama-70b` — hypothetical second replication
  by a different person on a larger model

`{model_scope}` can be a specific model key (`qwen_1_5b`) or a compact
label when you run several (`6models`, `small-tier`, `llama-family`).
Including the paper_id in the ID keeps them globally unique so you can
replicate many papers on the same model without ambiguity.

### What to include in your PR

```
config/papers/{paper_id}/
    paper.md                # Full paper text — shared across all replications
                            #   of this paper (the "oracle"). Already exists
                            #   if someone's replicated this paper before;
                            #   don't overwrite it if so.
    replications/{replication_id}/
        paper_config.yaml   # YOUR claims, thresholds, experiment params
        metadata.yaml       # YOUR replicator info (used by the README generator)
        stimuli_config.yaml # YOUR stimulus definitions (OK to share the
                            #   canonical dataset with other replicators;
                            #   but your file lives under your namespace)
        stimuli/            # Small JSON stimulus files (generated data
                            #   can be regenerated from the config)

writeup/{paper_id}/{replication_id}/
    draft.md                # Your writeup (methods, results, discussion)

results/{paper_id}/{replication_id}/
    {model_key}/
        {claim_id}/
            result.json     # Metric outputs
            sanity.json     # Sanity-check report
        critiques/          # Critique agent outputs (if you ran them)

figures/{paper_id}/{replication_id}/
    *.png, *.pdf            # Your figures
```

**Do not include** large files (activation caches, probe weights, `.pt`
files). The `.gitignore` blocks them, but double-check `git status`
before committing.

**Do not modify** anything at the paper level (`paper.md`, or files
under another replicator's `replications/{their_id}/`). Those belong to
other people's replications or the shared paper oracle.

### Step-by-step

1. **Fork** the repo and create a branch named after your replication:
   ```bash
   git checkout -b replication/{replication_id}
   ```
2. **Pick your replication ID** (see "Replication IDs" above).
3. **Read the paper.** If the paper has never been replicated before,
   save the full text as `config/papers/{paper_id}/paper.md`. If it has
   been, leave that file alone — your replication shares it as an oracle.
4. **Scaffold your replication directory** by copying the template:
   ```bash
   cp -r config/replication_template \
     config/papers/{paper_id}/replications/{replication_id}
   ```
   Then edit `paper_config.yaml`, `stimuli_config.yaml`, and
   `metadata.yaml` with your claims, thresholds, and replicator info.
5. **Create stimuli** — either hand-written JSONs in
   `config/papers/{paper_id}/replications/{replication_id}/stimuli/` or
   a `stimuli_config.yaml` that generates them.
6. **Run the pipeline** on at least one small model locally:
   ```bash
   python -m src.core.pipeline \
     --paper {paper_id} --replication {replication_id} --model qwen_1_5b
   ```
7. **Run on medium models** if you have GPU access:
   ```bash
   python -m src.core.pipeline \
     --paper {paper_id} --replication {replication_id} --tier medium
   ```
8. **Write up your findings** in `writeup/{paper_id}/{replication_id}/draft.md`.
   Include:
   - Which claims replicated and which didn't
   - Cross-model comparison (what generalizes, what's model-specific)
   - Limitations of your replication
   - Anything surprising
9. **Get a pre-PR self-review from a different AI.** The pipeline
   prints a "NEXT STEP" block at the end of every non-fast run with a
   paste-ready prompt and a bundle file at
   `local_data/reviews/{paper}/{your_id}/self_review_bundle.md`.
   Upload that bundle into a web AI *other than* the one that helped
   you run the replication — the goal is independent feedback from a
   model that isn't already anchored on your choices. You can also
   invoke it standalone:
   ```bash
   python scripts/review_prompt.py \
       --paper {paper} --replication {your_id} \
       --self-review --used-ai Claude   # or ChatGPT / Gemini / Codex
   ```
   Save any responses you decide are worth keeping under
   `writeup/{paper}/{your_id}/reviews/{reviewer}.md` before committing.
   (This is separate from the post-merge referee reviews the
   maintainer runs — see [AI review policy](#ai-review-policy) below.)
10. **Regenerate the README table** (don't hand-edit it):
    ```bash
    python scripts/generate_replications_table.py
    ```
    This scans every `metadata.yaml` and rewrites the table between its
    sentinels. It adds your row automatically from the metadata you
    wrote in step 4.
11. **Open a PR** with a clear title like:
    `Replication: {replication_id}`
    (e.g., `Replication: geometry_of_truth-alice-llama-70b`). The PR
    shouldn't touch paths under anyone else's `replications/{their_id}/`.

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

### What a "fresh and clean" replication looks like

The point of the per-replication layout is that each submission stands
on its own. When in doubt:

- **Don't copy another replicator's results**, probe weights, or
  concept vectors. Rerun the pipeline on your own machine / hosted GPU,
  from your own stimuli.
- **It's fine to share the paper's original datasets** (e.g., the
  Marks–Tegmark truth CSVs). The novelty of your replication is in
  *where you run them, with what model, with what choices*.
- **It's fine to read existing replications for inspiration.** The
  point of the per-replication namespace isn't to prevent you from
  learning what thresholds others picked — it's to prevent your results
  from silently overwriting theirs.
- **If you deviate from another replicator's methodology deliberately**
  (e.g., you pick different thresholds or a different probe type),
  explain why in your writeup. Deliberate methodological choices are
  exactly what independent replications are for.

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
`writeup/{paper_id}/{replication_id}/reviews/` alongside the draft, and
the scores are surfaced in the [Completed
replications](README.md#completed-replications) table in the README.

To prepare a replication for review, use the helper:

```bash
# One-shot "give me everything I need to run a review right now":
# writes a self-contained bundle under
# local_data/reviews/{paper}/{id}/reviewer_bundle.md and prints
# step-by-step upload-and-paste instructions plus the two prompts.
# Works around URL-fetch restrictions in claude.ai / gemini /
# chatgpt — the reviewer uploads the bundle, no fetching needed.
# This is the default you want for ad-hoc reviews.
python scripts/review_prompt.py --paper {paper} --replication {id} \
    --reviewer-ready

# Prints a ready-to-paste prompt with the standardized protocol + GitHub
# URLs to the writeup/config/results/figures. Requires the reviewer's
# AI to fetch URLs, which some web chat UIs now block as a
# provenance check. Prefer --reviewer-ready if you hit that.
python scripts/review_prompt.py --paper {paper} --replication {id} \
    --pr {pr_number}

# Concatenates every text artifact (paper.md, writeup, configs, per-claim
# result.json + sanity.json, any harness critiques) into a single
# markdown file. Good for ad-hoc offline use when you don't need the
# prompt scaffolding --reviewer-ready gives you.
python scripts/review_prompt.py --paper {paper} --replication {id} \
    --bundle -o /tmp/review_bundle.md
```

`--replication` is never auto-selected: if you omit it, the script
lists the available IDs and exits.

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
