# Harness CHANGELOG

Append-only log of framework/harness changes — i.e. anything touching
`src/`, `tests/`, `scripts/`, `config/models.yaml`, or repo-level docs
(`CLAUDE.md`, `CONTRIBUTING.md`, `DESIGN.md`, `GOTCHAS.md`,
`README.md`). Scientific progress on a specific replication belongs in
that replication's own log at
`writeup/{paper_id}/{replication_id}/CHANGELOG.md`.

Newest first. One entry per meaningful unit of work. Include the
commit SHA so entries are greppable against `git log`.

---

## 2026-04-21 — `review_prompt.py`: add `--reviewer-ready` mode

- **Commit:** `f3d3bb0`
- **Files:** `src/core/review.py`, `scripts/review_prompt.py`,
  `CLAUDE.md`, `CONTRIBUTING.md`
- **Why:** the PR-review mode produces a prompt with GitHub URLs, and
  claude.ai / gemini / chatgpt are rejecting those fetches on
  provenance grounds (markdown link formatting, machine-pasted URLs,
  GitHub hosts all seem to trip it). The self-review mode already
  sidesteps this by bundling artifacts, but its framing is narrowly
  contributor-focused.
- **What:** new fourth mode, `--reviewer-ready`, with the same bundle
  mechanics as `--self-review` but neutral framing (no "different AI"
  nudge). Emits three-step upload-paste-follow-up instructions. CLAUDE.md
  now tells future agents to use this mode when the user asks
  "generate a review prompt for {id}".

---

## 2026-04-21 — withdraw `geometry_of_truth-tulaneadam-qwen_1_5b`

- **Commit:** `d65ffdf`
- **Files:** deletions under `config/papers/geometry_of_truth/replications/`,
  `writeup/geometry_of_truth/`, `results/geometry_of_truth/`; edits to
  `README.md` (regenerated), `CONTRIBUTING.md`, `src/core/pipeline.py`,
  `src/core/config_loader.py`, and the surviving
  `geometry_of_truth-zachgoldfine44-qwen_1_5b` writeup + paper_config.
- **Why:** The attempt used a weaker AI assistant and the maintainer did not consider
  it review-worthy. It was primarily a test run for the harness.
- **What:** hard delete of the three namespaces, plus pruning of the
  two real cross-references in `zachgoldfine44`'s artifacts, plus
  swapping the now-dead handle for `zachgoldfine44` in docstring/CLI
  examples so the docs still point at a real replication.

---

## 2026-04-09 — harness improvements pass

User gave a substantial set of improvements after the emotions
replication run. This entry covers framework-level work that followed;
emotions-specific followups live in the emotions replication's
CHANGELOG.

### Done

- `edbaa39` Added `GOTCHAS.md` (mech-interp guardrails document) to
  repo root and wired it into `CLAUDE.md` as the canonical preflight +
  post-result checklist for new paper implementations.
- `2e18e00` Major `CLAUDE.md` rewrite covering:
  - **Commit cadence**: explicit "initial commit at session start, commit
    after every work unit, push every commit, pile-ups not allowed."
  - **CHANGELOG.md vs PROGRESS.md** distinction: changelog is the running
    log, progress is curated milestone state.
  - **Paper as oracle**: every new paper now requires
    `config/papers/{paper_id}/paper.md` saved at session start, with
    `paper_section` citations on every claim and methodology comment.
  - **Storage layout**: dropped Google Drive symlink magic in favor of
    everything in the repo. Small artifacts (`result.json`, figures,
    critique reports) are committed to git; heavy artifacts (activation
    caches, probe weights) stay under `local_data/` (gitignored).
  - References to the new sanity-check + critique systems (built in
    follow-on commits).
- `a8e3a48` **Paper as oracle** loader. `PaperConfig.paper_text` reads
  `config/papers/{paper_id}/paper.md` automatically; `ClaimConfig.paper_section`
  added so each claim points at the section it came from. Placeholder
  `paper.md` written for the emotions paper. 4 new tests.
- `958783d` **Sanity-check system** (`src/core/sanity_checks.py`, ~420 LOC).
  Ten checks fire after every experiment, write `sanity.json` next to
  `result.json`, and surface warnings/errors in stdout. Specifically
  catches the `0.800 = 12/15` resolution-artifact case that took hours
  to debug in a previous session. 11 new tests.
- `621b0ee` **Critique agent system** (`src/core/critique.py`, ~500 LOC).
  At end of each model: Claude critic + optional ChatGPT critic +
  evaluator. Outputs four files to
  `results/{paper}/{model}/critiques/` which are git-tracked and flow
  into commit history. 9 new tests, all mocked. anthropic + openai
  added to requirements.txt.
- Pipeline cleanup: use `paper_config.paper_text` directly instead of
  re-reading from disk; build the critique payload from the loaded
  `PaperConfig` including `paper_section` pointers.

### Where things stand after this session

- 212 unit tests passing (was 180 at start of session; +32 across
  paper-loading, sanity, critique, and the prior Tier-2 work).
- Six failure modes from the user's experience are now structurally
  prevented or surfaced automatically. See list below.
- Three Tier-2 items (#3 multi-layer × multi-alpha steering sweep, #4
  external LLM-as-judge, #5 base-model variants, #6 multi-turn
  scenarios) remain and need GPU or API budget.

### Bug classes structurally addressed

| Original observation | Mechanism that now prevents it |
|---|---|
| Nothing committed during run | `CLAUDE.md` commit-cadence rules + small artifacts whitelisted in `.gitignore` |
| Nothing visible in Google Drive | Drive layer dropped; `local_data/` for caches, `results/` for committed artifacts |
| `PROGRESS.md` overwritten | `CLAUDE.md` "never delete history" rule + new `CHANGELOG.md` for incremental log |
| Paper not loaded as oracle | `PaperConfig.paper_text` field + mandatory `paper.md` per paper |
| Steering forgot / mis-implemented | `paper_section` pointers + critique agents that compare against paper text |
| Suspicious results caught only by user | `src/core/sanity_checks.py` runs after every experiment |

### Bug-class context for future Claude sessions

The user observed several failure modes during the emotions replication
that drove the changes above. They are listed here so future sessions
recognize them when they recur:

1. **Nothing committed during the run** until the user manually prompted
   it. Fix: explicit commit cadence rules in `CLAUDE.md`.
2. **Nothing visible in Google Drive** (or unclear where to look). Fix:
   storage simplification, results committed to git.
3. **`PROGRESS.md` overwritten** by past sessions instead of appended.
   Fix: explicit "never delete history without first moving it to
   CHANGELOG.md or the writeup" rule.
4. **Paper not loaded anywhere in the repo**, leading to drift from the
   paper's methodology. Fix: `paper.md` is now mandatory per paper, and
   experiments + critique agents can read it.
5. **Steering experiment forgot in small models, then implemented wrong
   way in medium models** ("Elo from humans? No, the paper used
   pairwise LLM judging"). Same root cause as #4.
6. **Suspicious results caught only by the user** (the
   `diagonal_dominance = 0.800 = 12/15` resolution artifact). Fix: the
   new `src/core/sanity_checks.py` module runs after every experiment
   and flags this kind of thing in stdout.

### Past entries (pre-changelog)

Previous harness work was captured ad-hoc in PROGRESS.md (now split
per-replication) and in commit messages. Highlights:

- Built the paper-agnostic harness with 6 experiment types.
- Tier-2 refactors: aggregation as a first-class concept, ActivationCache
  wired through the pipeline, the `__new__` shim removed from the four
  experiment classes (commits `547c454`, `4087d90`).
- 188 unit tests passing at end of that era (20 skipped — integration
  tests requiring a real model).
