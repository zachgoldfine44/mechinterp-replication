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
- *(this commit, hash TBD)* Major `CLAUDE.md` rewrite covering:
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
