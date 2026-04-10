# Peer Reviews: Emotions Concept Replication

This directory contains AI referee reviews of [`writeup/emotions/draft.md`](../draft.md). Three frontier language models were given the same standardized prompts and asked to (a) rate the replication on a 1–10 scale with justification and (b) produce a peer-review-style referee report with major concerns, minor concerns, and an accept/reject recommendation.

## Summary of scores

| Reviewer | Score | Recommendation | Full review |
|---|:---:|:---:|---|
| ChatGPT (extended thinking) | **7 / 10** | Weak reject / major revision | [chatgpt-extended-thinking.md](chatgpt-extended-thinking.md) |
| Claude Opus 4.6 (extended thinking) | **7.5 / 10** | Revise and resubmit (minor revision) | [claude-opus-4-6-extended-thinking.md](claude-opus-4-6-extended-thinking.md) |
| Gemini 3.1 Pro | **8.5 / 10** | Reject in current form (major revision) | [gemini-3-1-pro.md](gemini-3-1-pro.md) |
| **Mean** | **7.67** | | |

## What the reviewers agreed on

All three reviewers independently identified the same core issues:

1. **The steering null is a floor effect, not a negative result.** With a 0% baseline unethical rate across all six open-source models, Fisher's exact test has no statistical headroom to detect a steering effect even if one existed. The title and abstract's "representation without function" framing overstates what a 0-vs-0 comparison can demonstrate.
2. **The scale gap is material.** Tested models (1B–9B parameters) are an order of magnitude smaller than Claude Sonnet 4.5. Claiming the causal behavioral findings "don't replicate" across this gap is weak inference, especially since the paper shows probe accuracy scales with size (ρ = 0.94).
3. **Method fidelity is looser than a strict replication.** The original used 171 emotions with richer extraction machinery; this replication uses 15 emotions with simpler mean-difference vectors and a narrower generalization test (2 implicit scenarios per emotion).

## What they disagreed on

- **Severity of the behavioral framing issue**: ChatGPT (harshest) and Gemini (harshest) both recommended rejection in current form; Claude called it "minor revision" but noted the minor revision is load-bearing for the central claim.
- **The contamination discovery**: Claude and Gemini rated this as a meaningful methodological contribution that partially offsets the behavioral null weakness. ChatGPT gave it less weight.
- **Overall scientific contribution**: Gemini was most generous (8.5), viewing the representational findings as a solid standalone contribution. ChatGPT was harshest (7) on the behavioral framing.

## How these reviews were produced

Each reviewer was prompted with:

> Please read the following paper, and then the following replication of the paper, and rate the replication on a scale of 1-10 (1 being terrible experimental design, no meaningful insights due to serious methodological flaws, etc; 10 being perfect experimental design, insights are useful due to rigorous methodology, meaningfully contributes to scientific body of knowledge), along with justifications for the 1-10 score you assign:
> - Paper: https://transformer-circuits.pub/2026/emotions/index.html
> - Replication: https://github.com/zachgoldfine44/mechinterp-replication/blob/main/writeup/emotions/draft.md

Followed by:

> Can you turn this into a harsher peer-review-style referee report with major concerns / minor concerns / accept-reject recommendation?

The exact same prompts were given to all three reviewers to ensure comparability. This is the review protocol documented in [CONTRIBUTING.md](../../../CONTRIBUTING.md#ai-review-policy).

Reviews were conducted in April 2026 against the draft as of commit [`c0f010e`](https://github.com/zachgoldfine44/mechinterp-replication/commit/c0f010e).
