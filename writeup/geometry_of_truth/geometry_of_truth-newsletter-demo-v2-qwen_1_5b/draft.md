# Replication (v2): *The Geometry of Truth* on Qwen-2.5-1.5B-Instruct (post-#8 verification)

**Paper:** Marks & Tegmark (2023), [arXiv:2310.06824](https://arxiv.org/abs/2310.06824)
**Replicator:** Newsletter Demo v2 ([@newsletter-demo-v2](https://github.com/newsletter-demo-v2))
**Replication ID:** `geometry_of_truth-newsletter-demo-v2-qwen_1_5b`
**Model:** Qwen-2.5-1.5B-Instruct (28 layers, hidden dim 1536)
**Status:** v0.2 — second simulation

> **Context:** this is the second dogfood of the newsletter onboarding
> flow, run against main after PR #8 landed the three must-fix
> friction points surfaced by the first simulation (PR #7). The
> science is a carbon copy of v1 — the value of this PR is
> documenting that the UX path now works without manual workarounds.

---

## Summary

Same minimal single-claim replication as v1. **Result: PASS at 1.00
cross-validated probe accuracy** (best layer: 15, confusion matrix
`[[50, 0], [0, 50]]`). Deterministic match with the v1 run — same
stimuli (seed=42), same extraction, same probe.

The interesting finding is **procedural, not scientific**: every step
of the newsletter onboarding path now works without the workarounds
the first simulation needed.

---

## Did the fixes take?

Tested against main at commit `da96547` (tip of PR #8).

### Fix #1 (Python 3.9 blocker) — ✅ FIXED

Followed the README's new step 1 verbatim:

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Clean install in ~30s. No `sae-lens` wheel incompatibility. The
explicit `3.11` pin + the callout about `brew install python@3.11`
catches the most common macOS default-Python trap.

### Fix #2 (results all routed to local_data/) — ✅ FIXED

This was the biggest win. After `python -m src.core.pipeline … --model qwen_1_5b`:

```
results/geometry_of_truth/geometry_of_truth-newsletter-demo-v2-qwen_1_5b/
  pipeline_summary.json            # committed, written directly
  qwen_1_5b/
    truth_probe_cities/
      result.json                  # committed
      sanity.json                  # committed
      concept_vectors.pt           # committed (gitignore whitelist)
      probes/layer_*_result.json   # committed
    critiques/*.{json,md}          # committed

local_data/results/geometry_of_truth/geometry_of_truth-newsletter-demo-v2-qwen_1_5b/
  qwen_1_5b/truth_probe_cities/
    activations/stimulus_*.pt  (×100)  # cache only, gitignored
```

**A `git add -A` after the run stages 25 files** (config + stimuli +
results + writeup) with zero manual copies. In v1 I had to hand-move
seven files out of `local_data/` into `results/` before the PR would
look right.

### Fix #3 (template `paper_text_path`) — ✅ FIXED

New template default is `paper.md`. Pipeline log shows:

```
INFO | src.core.config_loader | Loaded paper text from .../paper.md (77194 chars)
```

No "No paper text found" warning. Paper oracle loads correctly on
every run.

---

## Results (same as v1, deterministic)

| Metric | Value |
|---|---|
| Probe accuracy (best layer) | **1.00** |
| Best layer | 15 |
| Per-class accuracy (true / false) | 1.00 / 1.00 |
| Confusion matrix | [[50, 0], [0, 50]] |
| Runtime | ~24s claim time + ~90s model load on Apple Silicon MPS |

Layer sweep (chance = 0.50):

| Layer | 0 | 3 | 6 | 9 | 12 | **15** | 18 | 21 | 24 | 27 |
|---|---|---|---|---|---|---|---|---|---|---|
| CV acc | 0.55 | 0.65 | 0.85 | 0.94 | 0.99 | **1.00** | 1.00 | 1.00 | 0.99 | 0.98 |

Four Qwen-1.5B replications of this paper are now in-repo; probe
accuracies (0.97 / 0.99 / 1.00 / 1.00) are tight-clustered — evidence
that the in-distribution probe finding is robust at this scale.

---

## Remaining friction (untouched by #8, deferred)

Same "should-fix" + "nice-to-have" items that the v1 simulation
surfaced, still present:

- `cp -r config/replication_template` drags in the template's own
  `README.md` + `stimuli/.gitkeep`. I had to delete them before
  `git add -A` — otherwise they'd commit as noise in the replication
  directory.
- Sanity check `per_concept_uniformity` still fires a false-positive
  WARN on perfect classification. Non-blocking. Same ask to exempt
  `(1.0, 1.0)` as legitimate rather than a collapse signal.
- Three Geometry-of-Truth replications were already in the README
  table before this run; the README still doesn't nudge on why a
  fourth coexistence is valuable. One sentence could close the gap.

None of these stop the PR from opening cleanly now that #8 is in.

---

## Reproducing

```bash
git clone https://github.com/zachgoldfine44/mechinterp-replication.git
cd mechinterp-replication
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Scaffold + fill in your replication_id (this part is still manual)
cp -r config/replication_template \
  config/papers/geometry_of_truth/replications/geometry_of_truth-newsletter-demo-v2-qwen_1_5b
# then edit paper_config.yaml, stimuli_config.yaml, metadata.yaml,
# and rm the template README.md + stimuli/.gitkeep

# Regenerate the cities stimuli (50 per class, seed=42)
python config/papers/geometry_of_truth/replications/\
geometry_of_truth-newsletter-demo-v2-qwen_1_5b/convert_stimuli.py

# Run
python -m src.core.pipeline \
    --paper geometry_of_truth \
    --replication geometry_of_truth-newsletter-demo-v2-qwen_1_5b \
    --model qwen_1_5b
```

~2 min on Apple Silicon MPS.
