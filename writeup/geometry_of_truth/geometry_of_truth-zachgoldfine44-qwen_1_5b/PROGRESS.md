# PROGRESS — `geometry_of_truth-zachgoldfine44-qwen_1_5b`

Milestone log for this replication. Update in-place when the status
changes; move retired content to `CHANGELOG.md` before deleting from
here. Keep this file tight (~200 lines).

**Paper:** The Geometry of Truth: Emergent Linear Structure in Large
Language Model Representations of True/False Datasets (Marks & Tegmark
2023) — https://arxiv.org/abs/2310.06824
**Replicator:** @zachgoldfine44
**Models tested:** Qwen-2.5-1.5B-Instruct (original paper: LLaMA-2 7B/13B/70B)
**Current version:** v0.1 partial

## Current status

See `draft.md` for the full writeup. Per the README table:
within-distribution probe replicates cleanly (0.99 on `cities`, layer 12
— matches paper 13B/70B despite being 10× smaller). Cross-dataset
generalization does NOT replicate at 1.5B: probe collapses to trivial
predictions on `sp_en_trans` (92/100 "true"), flips sign on
`neg_cities` (replicates paper's Fig 3c orthogonal-directions finding),
collapses to all-false on `larger_than`. Consistent with the paper's
"at sufficient scale" framing.

## Open questions / next steps

_(empty — fill in as work continues)_

## Pointers

- Paper oracle: `config/papers/geometry_of_truth/paper.md`
- Config: `config/papers/geometry_of_truth/replications/geometry_of_truth-zachgoldfine44-qwen_1_5b/`
- Results: `results/geometry_of_truth/geometry_of_truth-zachgoldfine44-qwen_1_5b/`
- Reviews: `writeup/geometry_of_truth/geometry_of_truth-zachgoldfine44-qwen_1_5b/reviews/`
  (empty at time of writing)
- Commit history for this replication:
  `git log -- writeup/geometry_of_truth/geometry_of_truth-zachgoldfine44-qwen_1_5b/
  config/papers/geometry_of_truth/replications/geometry_of_truth-zachgoldfine44-qwen_1_5b/
  results/geometry_of_truth/geometry_of_truth-zachgoldfine44-qwen_1_5b/`
