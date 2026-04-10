# CHANGELOG

Append-only running log of work in this repo. One entry per commit (or per
meaningful work unit, whichever is larger). For milestone-level state see
`PROGRESS.md`. **Never rewrite past entries here.**

Format: each entry starts with an ISO date, the commit short-hash if known,
and a one-line summary, followed by optional bullets.

---

## 2026-04-09 — harness improvements pass

The user gave a substantial set of improvements after replicating the
Anthropic emotions paper. This entry covers the work that follows.

### Done in this session

- `edbaa39` Added `GOTCHAS.md` (Sofroniew/anonymous mech-interp guardrails
  document) to repo root and wired it into `CLAUDE.md` as the canonical
  preflight + post-result checklist for new paper implementations.
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
- *(this commit, hash TBD)* Pipeline cleanup: use
  `paper_config.paper_text` directly instead of re-reading from disk;
  build the critique payload from the loaded `PaperConfig` including
  `paper_section` pointers. CHANGELOG rolled forward.

### Where things stand after this session

- 212 unit tests passing (was 180 at start of session; +32 across
  paper-loading, sanity, critique, and the prior Tier-2 work).
- Six failure modes from the user's experience are now structurally
  prevented or surfaced automatically. See list below.
- Three Tier-2 items (#3 multi-layer × multi-alpha steering sweep, #4
  external LLM-as-judge, #5 base-model variants, #6 multi-turn
  scenarios) remain and need GPU or API budget.

### Bugs structurally addressed by this session

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

Previous work spanned multiple sessions and is captured in `PROGRESS.md`
and the writeup at `writeup/emotions/draft.md`. Highlights:

- Built the paper-agnostic harness with 6 experiment types.
- Replicated the Anthropic emotions paper across 6 models (Llama 1B/8B,
  Qwen 1.5B/7B, Gemma 2B/9B).
- Multiple critique-response passes producing draft v3.3.
- Tier-2 refactors: aggregation as a first-class concept, ActivationCache
  wired through the pipeline, the `__new__` shim removed from the four
  experiment classes (commits `547c454`, `4087d90`).
- Currently 188 unit tests passing (20 skipped — integration tests
  requiring a real model).
