# CHANGELOG

Append-only running log of work in this repo. One entry per commit (or per
meaningful work unit, whichever is larger). For milestone-level state see
`PROGRESS.md`. **Never rewrite past entries here.**

Format: each entry starts with an ISO date, the commit short-hash if known,
and a one-line summary, followed by optional bullets.

---

## 2026-04-13 — v3.5: Second round of critique responses (3 recommended next-step docs)

Three new recommended-next-step documents (from ChatGPT, Claude, and Gemini)
converged on these priorities:
1. Test causal behavioral influence on non-safety-gated dimensions (sycophancy, tone)
2. Add binomial tests for severity pairs
3. Stimulus-level bootstrap CIs
4. Human-authored stimulus control

### Completed so far

- **Binomial tests for severity pairs**: Only Llama-8B (9/10, p=0.011)
  passes. Two borderline (Llama-1B, Qwen-1.5B at 8/10, p=0.055). Three
  models indistinguishable from chance. Saved to
  `results/emotions/severity_pairs_binomial.json`. Writeup severity section
  reframed from "5/6 show majority" to "1 model significant, 2 borderline."
- **Sycophancy + tone steering experiments**: Script
  `scripts/behavioral_steering.py` created with sycophancy (10 scenarios ×
  LLM-as-judge) and tone (10 neutral questions × 1-5 scale judge).
  Running on Qwen-7B on A100. Partial results for happy concept:
  - Baseline: 0% sycophancy (0/100)
  - α=0.05: 0% (0/100)
  - α=0.10: 0% (0/100)
  - α=0.50: 1% (1/100) — first non-zero but not significant
  - Remaining concepts (loving, enthusiastic, hostile, afraid) + tone
    experiment still running on A100 (PID running autonomously).

---

## 2026-04-12 — v3.4: Address three external peer reviews

Three independent peer reviews (ChatGPT extended thinking 7/10, Claude Opus 4.6
extended thinking 7.5/10, Gemini 3.1 Pro 8.5/10) identified consensus issues.
This session addresses the highest-priority items.

### Writeup changes (addressing all three reviewers)

- **Reframed behavioral null as floor effect, not negative result.** Title changed
  from "Emotion Representations without Emotion Function" to "Cross-Model
  Replication of Emotion Representations...: Universal Encoding, Inconclusive
  Behavioral Tests." Abstract, Section 2.3, Discussion, and Figure 3 caption
  all rewritten to clearly distinguish "uninformative due to floor effect" from
  "tested and failed." This was the consensus #1 issue across all three reviews.
- **Added Clopper-Pearson 95% CIs to Figure 3 steering results**: per-condition
  (N=10) CI is [0%, 30.8%], pooled (N=450) CI is [0%, 0.8%]. Saved to
  `results/emotions/figure3_clopper_pearson.json`.
- **Geometry CIs already computed and now cited in main text**: all six models
  have bootstrap 95% CIs on PC1-valence |r|, and the text now notes the wide
  CIs (~0.3-0.4 units at N=15) to temper precision claims.
- **Scale gap explicitly acknowledged**: Discussion now notes that behavioral
  potency plausibly scales with model size, and that absence at 9B doesn't
  predict absence at 50B+.
- **Emotion selection criteria documented**: Methods section now lists the four
  selection principles (valence coverage, arousal spread, semantic diversity,
  no near-synonyms) and notes no formal pilot was conducted.
- **LLM stimulus circularity risk acknowledged**: Methods section now includes a
  circularity caveat about LLM-generated stimuli and notes that a human-authored
  control set would strengthen claims.
- **Method fidelity framing softened**: Paper now explicitly describes itself as
  a "cross-model extension" rather than a strict replication, with a detailed
  comparison of original vs. our methodology in the Limitations paragraph.
- **Qwen-7B contamination ratio >1.0 moved from supplement to main text** in
  Section 2.2.
- **M3/M5 editorial inconsistency fixed**: all references now say MacBook Air M3.
- **Compute asymmetry documented**: Methods and Compute sections describe the
  CPU vs GPU difference explicitly.

### New experiments (addressing Gemini's positive control concern)

- **Sentiment steering positive control**: Script `scripts/critique_followups_hf.py`
  steers with happy/hostile/enthusiastic/sad vectors on neutral sentiment prompts
  (restaurant reviews, weather, movies, etc.) and measures keyword-based sentiment
  shift. Tests alphas [0.0, 0.5, 1.0, 2.0, 5.0]. If steering shifts sentiment on
  benign prompts, it demonstrates the pipeline works and the ethical-scenario null
  is a floor effect rather than a broken pipeline.
- **High-alpha ethical steering sweep**: Same script sweeps alphas [0.5, 1.0, 2.0,
  3.0, 5.0] on ethical scenarios with coherence monitoring (unique-word ratio).
  Addresses Gemini's concern that alpha=0.50 upper bound was too weak.
- Ran on all 6 models via A100 (~3.5 hrs total). Key findings:
  - **Sentiment positive control PASSES on all 6 models**: happy vector shifts
    sentiment +0.013 to +0.525 at alpha=5.0. Gemma models show 10-50x larger
    effects than Llama/Qwen. All 4 steering concepts (happy, hostile,
    enthusiastic, sad) shift in expected directions universally.
  - **High-alpha ethical sweep on all 6 models**: coherence degrades with model
    size (Gemma-9B loses coherence at α=3.0, Qwen-1.5B stays coherent to α=5.0).
    Small models more susceptible to keyword-based "unethical" detection but
    don't degrade. Alpha=0.50 was too low — effects appear at α=1.0-3.0.
  - **Keyword classifier caveat** documented: the noisy keyword matcher produces
    nonzero baselines on some models (e.g., 20% on Llama-8B), which the LLM-as-judge
    approach avoids.
  - Data saved to `results/emotions/{model_key}/critique_followups/` for all 6 models.

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
