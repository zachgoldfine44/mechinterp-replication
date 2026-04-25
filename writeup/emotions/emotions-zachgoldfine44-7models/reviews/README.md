# Peer Reviews: Emotions Concept Replication

This directory contains AI referee reviews of [`writeup/emotions/draft.md`](../draft.md) (v3.6). Three frontier language models were given the same standardized prompts and asked to (a) rate the replication on a 1-10 scale with justification and (b) produce a peer-review-style referee report with major concerns, minor concerns, and an accept/reject recommendation.

## Summary of scores

| Reviewer | Score | Recommendation | Full review |
|---|:---:|:---:|---|
| ChatGPT (extended thinking) | **7 / 10** | Reject | [chatgpt-extended-thinking.md](chatgpt-extended-thinking.md) |
| Claude Opus 4.6 (extended thinking) | **7 / 10** | Revise and resubmit (major revisions) | [claude-opus-4-6-extended-thinking.md](claude-opus-4-6-extended-thinking.md) |
| Gemini 3.1 Pro | **8.5 / 10** (score) · **Reject** (referee) | Reject (encourage resubmission as note/short paper) | [gemini-3-1-pro.md](gemini-3-1-pro.md) |
| **Mean** | **7.50** | | |

## Score evolution across draft versions

| Reviewer | v3.3 | v3.4 | v3.6 | Trend |
|---|:---:|:---:|:---:|---|
| ChatGPT | 7 | 7 | 7 | Stable |
| Claude | 7.5 | 6 | 7 | V-shaped (dropped on framing, recovered with sycophancy data) |
| Gemini | 8.5 | 9 | 8.5 | Peaked at v3.4 (positive control valued), back to 8.5 with harsher referee |
| Mean | 7.67 | 7.33 | 7.50 | Stable in the 7-7.5 range |

## What the reviewers agreed on

All three independently identified the same core issues:

1. **Not a strict replication.** All three say this is better framed as a "cross-model extension" or "generalization study." The methodological simplifications (15/171 emotions, 25 vs. ~1,200 stories, simpler vectors, no denoising) are too substantial for strict replication claims.
2. **The ethical-steering null is a floor effect, not a finding.** The 0% baseline makes the test uninformative rather than negative. The title and abstract may still overstate what can be concluded.
3. **The scale gap matters.** 1B-9B vs. Claude Sonnet 4.5 (likely >100B) is a fundamentally different regime. Behavioral potency may have a scale threshold.
4. **LLM-generated stimuli introduce circularity risk.** Probes may detect how models write about emotions rather than how they represent them.
5. **The sycophancy pushback result is suggestive but preliminary.** p=0.036 on one model without multiple-comparison correction is not conclusive.

## What they disagreed on

- **Overall severity**: Claude and ChatGPT both give 7/10 with fairly aligned critiques. Gemini gives 8.5/10 on the initial score but the harshest referee report (Reject, calling the work an "exploratory weekend project").
- **Whether to publish**: ChatGPT says Reject. Claude says Revise and Resubmit. Gemini says Reject but encourages resubmission as a note/short paper. No reviewer recommends acceptance in current form.
- **The positive controls**: Claude credits the severity-pairs confound discovery as the "most original contribution." ChatGPT credits the sentiment positive control and sycophancy design. Gemini credits the overall experimental design maturity.
- **Specific statistical concerns**: Gemini specifically calls out the lack of multiple-comparison correction on the sycophancy tests. Claude focuses on the LLM-as-judge circularity. ChatGPT focuses on the framing gap between "replication" and "extension."

## How these reviews were produced

Each reviewer was prompted with:

> Please read the following paper, and then the following replication of the paper, and rate the replication on a scale of 1-10 (1 being terrible experimental design, no meaningful insights due to serious methodological flaws, etc; 10 being perfect experimental design, insights are useful due to rigorous methodology, meaningfully contributes to scientific body of knowledge), along with justifications for the 1-10 score you assign:
> - Paper: https://transformer-circuits.pub/2026/emotions/index.html
> - Replication: https://github.com/zachgoldfine44/mechinterp-replication/blob/main/writeup/emotions/draft.md

Followed by:

> Can you turn this into a harsher peer-review-style referee report with major concerns / minor concerns / accept-reject recommendation?

Reviews conducted in April 2026 against the v3.6 draft. This is the review protocol documented in [CONTRIBUTING.md](../../../../CONTRIBUTING.md#ai-review-policy).
