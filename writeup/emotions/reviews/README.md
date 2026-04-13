# Peer Reviews: Emotions Concept Replication

This directory contains AI referee reviews of [`writeup/emotions/draft.md`](../draft.md). Three frontier language models were given the same standardized prompts and asked to (a) rate the replication on a 1-10 scale with justification and (b) produce a peer-review-style referee report with major concerns, minor concerns, and an accept/reject recommendation.

## Summary of scores

| Reviewer | Score | Recommendation | Full review |
|---|:---:|:---:|---|
| ChatGPT (extended thinking) | **7 / 10** | Weak reject | [chatgpt-extended-thinking.md](chatgpt-extended-thinking.md) |
| Claude Opus 4.6 (extended thinking) | **6 / 10** | Major revisions required | [claude-opus-4-6-extended-thinking.md](claude-opus-4-6-extended-thinking.md) |
| Gemini 3.1 Pro | **9 / 10** (score) · **Borderline reject** (referee) | Major revision (borderline reject) | [gemini-3-1-pro.md](gemini-3-1-pro.md) |
| **Mean** | **7.33** | | |

Note: Gemini gave the highest initial score (9/10) but the harshest referee recommendation (borderline reject). This divergence reflects different evaluation stances: the 9/10 credits the experimental design and methodological controls, while the borderline-reject focuses on the gap between the paper's claims and the evidence available to support them.

## What the reviewers agreed on

All three reviewers independently identified the same core issues:

1. **The steering null is a floor effect, not a negative result.** With a 0% baseline unethical rate across all six open-source models, Fisher's exact test has no statistical headroom to detect a steering effect even if one existed. The title and abstract's framing overstates what a 0-vs-0 comparison can demonstrate.
2. **The scale gap is material.** Tested models (1B-9B parameters) are an order of magnitude smaller than Claude Sonnet 4.5. Claiming the causal behavioral findings "don't replicate" across this gap is weak inference, especially since probe accuracy scales with size (rho = 0.94).
3. **Method fidelity is looser than a strict replication.** The original used 171 emotions with richer extraction machinery; this replication uses 15 emotions with simpler mean-difference vectors and a narrower generalization test.
4. **Universality claims are overstated.** Six models across three families in the 1B-9B range, all instruction-tuned, does not justify claims about transformers in general.

## What they disagreed on

- **Severity of the issues**: Claude (6/10) was the harshest overall scorer, viewing the behavioral framing as a fundamental flaw. ChatGPT (7/10) was moderate. Gemini gave 9/10 on the initial scoring pass but borderline-reject on the referee report.
- **The sentiment positive control**: Gemini praised it strongly ("critical methodological safeguard"). Claude acknowledged it but called it "a much less demanding test than altering safety-relevant decision-making." ChatGPT called the pipeline validation "helpful" but not sufficient.
- **The contamination discovery**: Claude and ChatGPT rated this as a meaningful original contribution. Gemini mentioned it but focused less on it.
- **Whether the paper should test non-safety behaviors**: Gemini and ChatGPT specifically suggested testing sycophancy, verbosity, or helpfulness instead of ethical refusal. Claude focused more on the framing issue.

## How these reviews were produced

Each reviewer was prompted with:

> Please read the following paper, and then the following replication of the paper, and rate the replication on a scale of 1-10 (1 being terrible experimental design, no meaningful insights due to serious methodological flaws, etc; 10 being perfect experimental design, insights are useful due to rigorous methodology, meaningfully contributes to scientific body of knowledge), along with justifications for the 1-10 score you assign:
> - Paper: https://transformer-circuits.pub/2026/emotions/index.html
> - Replication: https://github.com/zachgoldfine44/mechinterp-replication/blob/main/writeup/emotions/draft.md

Followed by:

> Can you turn this into a harsher peer-review-style referee report with major concerns / minor concerns / accept-reject recommendation?

The exact same prompts were given to all three reviewers to ensure comparability. This is the review protocol documented in [CONTRIBUTING.md](../../../CONTRIBUTING.md#ai-review-policy).

Reviews were conducted in April 2026 against the v3.4 draft.
