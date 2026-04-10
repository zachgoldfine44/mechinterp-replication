# ChatGPT (extended thinking) — Peer Review

**Reviewer:** ChatGPT (extended thinking)
**Date:** April 2026
**Target:** [`writeup/emotions/draft.md`](../draft.md) (as of commit `c0f010e`)
**Score:** **7 / 10**
**Recommendation:** Major revision / weak reject in current form for an archival scientific venue. Much more positive if reframed as a partial replication plus cross-model extension.

---

## Prompts used

### Initial scoring prompt

> Please read the following paper, and then the following replication of the paper, and rate the replication on a scale of 1-10 (1 being terrible experimental design, no meaningful insights due to serious methodological flaws, etc; 10 being perfect experimental design, insights are useful due to rigorous methodology, meaningfully contributes to scientific body of knowledge), along with justifications for the 1-10 score you assign:
> - Paper: https://transformer-circuits.pub/2026/emotions/index.html
> - Replication: https://github.com/zachgoldfine44/mechinterp-replication/blob/main/writeup/emotions/draft.md

### Follow-up prompt

> Can you turn this into a a harsher peer-review-style referee report with major concerns / minor concerns / accept-reject recommendation?

---

## Initial response (7/10)

Zach, I'd give it a **7/10**. It is a **serious and useful** replication attempt, not a loose blog-post-style imitation: it runs across six open models from three families, uses an explicit paper config with success criteria, publishes code/tests, reports lexical baselines and multi-seed probe stability, and even surfaces a real confound in the severity test instead of hiding it. That is real scientific value. [[1]](https://github.com/zachgoldfine44/mechinterp-replication)

The biggest reason it is **not higher** is that the replication is only **moderately faithful on the most important causal claims**. In the original, the headline behavioral result is that an earlier Sonnet 4.5 snapshot blackmails **22%** of the time by default and **72%** under +0.05 "desperate" steering, with calm steering pushing it to **0%**; the paper also reports steering effects on reward hacking and preference. In the replication, the open-source instruct models have a **0% unethical baseline across all tested steering conditions**, so the study has almost no statistical headroom to detect an increase. That makes the null behavioral result much less decisive than the writeup's headline framing suggests. [[2]](https://www.anthropic.com/research/emotion-concepts-function)

Method fidelity is also meaningfully looser than ideal. The original extracts emotion vectors from **171 emotions**, using about **100 topics × 12 stories per topic per emotion**, averages activations across token positions starting at the **50th token**, and projects out top principal components from emotionally neutral transcripts to reduce confounds. The replication cuts this to **15 emotions** with **25 stories per emotion**, uses **last non-padding token** activations, and describes a simpler mean-difference vector method; it also relaxes success thresholds because it is testing much smaller models. Those choices are understandable, but they make this more of a **cross-model adaptation** than a clean apples-to-apples replication. [[3]](https://github.com/zachgoldfine44/mechinterp-replication/raw/refs/heads/main/config/papers/emotions/paper.md)

There are also a few medium-sized methodological weaknesses. The "generalization" set is tiny at **2 implicit scenarios per emotion**, the preference experiment is only an **approximation** of the original paper's richer preference/Elo setup, and the config notes that an earlier version silently requested **50** stimuli while only **25** were actually on disk before being aligned later. None of these are fatal, but together they reduce confidence and bump the work down from "very strong" to "good but not definitive." [[4]](https://github.com/zachgoldfine44/mechinterp-replication/raw/refs/heads/main/config/papers/emotions/paper_config.yaml)

What pushes it back up to a 7 is that the replication does generate a **meaningful scientific update**. The representational findings are pretty impressive: across all six models, probes are far above chance and above lexical baselines, transfer to implicit scenarios, and recover a valence-aligned geometry close to the original paper's valence result. The writeup is also appropriately honest that the behavioral null could reflect either a real representation/function gap **or** a protocol limitation from smaller, more refusal-trained instruction models and simpler scenarios. That intellectual honesty matters. [[5]](https://github.com/zachgoldfine44/mechinterp-replication/raw/refs/heads/main/writeup/emotions/draft.md)

So my bottom line is:

- **As an open, careful exploratory replication:** strong. [[1]](https://github.com/zachgoldfine44/mechinterp-replication)
- **As a faithful test of the original paper's causal behavioral claims:** only moderate. [[3]](https://github.com/zachgoldfine44/mechinterp-replication/raw/refs/heads/main/config/papers/emotions/paper.md)
- **As a contribution to the scientific literature:** worthwhile, especially for the claim that emotion-like representations may generalize across smaller open models, but **not enough to strongly refute** the original behavioral findings. [[2]](https://www.anthropic.com/research/emotion-concepts-function)

**Final score: 7/10.**

---

## Follow-up response — Referee report

**Manuscript:** *Emotion Representations without Emotion Function: A Cross-Model Replication of Sofroniew et al. (2026)*
**Recommendation:** **Major revision / weak reject in current form** for an archival scientific venue.
**If reframed as:** a **partial replication plus cross-model extension**, I'd be much more positive. [[1]](https://github.com/zachgoldfine44/mechinterp-replication)

### Summary judgment

This is a thoughtful, unusually transparent replication attempt with real value. The representational side is solid enough to be worth publishing in some form: across six open instruct models, the author finds strong above-chance emotion decoding, transfer to implicit scenarios, and valence-aligned geometry, while also documenting robustness checks and even version-history corrections. That is better than most informal replications. [[2]](https://github.com/zachgoldfine44/mechinterp-replication/blob/main/writeup/emotions/draft.md)

But as a **replication of the original paper's strongest causal claims**, the paper overreaches. The original study's headline result is not just "emotion vectors exist," but that they are behaviorally potent in Claude Sonnet 4.5: for an earlier snapshot, blackmail rises from **22%** to **72%** under desperate steering and falls to **0%** under calm steering; the paper also reports preference shifts driven by emotion-valence steering. In the replication, the tested open models have a **0% unethical baseline across all steering conditions**, which means the main behavioral non-replication is fundamentally non-diagnostic. The study therefore supports "representations seem robust across open models," but not "the original causal story fails to generalize." [[1]](https://github.com/zachgoldfine44/mechinterp-replication)

### Major concerns

#### 1. The behavioral null is not a fair test of the original causal claim

This is the biggest issue. The replication openly states that all tested models show a **0% baseline unethical rate**, and even notes that Fisher's exact test has no headroom in a 0-vs-0 setting. The original paper, by contrast, measured steering on an **earlier unreleased Sonnet 4.5 snapshot** that blackmailed **22%** of the time by default. Anthropic's own summary explicitly says the **released** model rarely does this behavior. So the replication is not really testing the same behavioral regime; it is testing heavily refusal-trained open instruct models in a setting where the target behavior is already extinguished. That makes the headline "behavioral effects replicate nowhere" too strong. At most, the data show that steering does not overcome refusal training in these models and scenarios. [[3]](https://www.anthropic.com/research/emotion-concepts-function)

#### 2. Method fidelity is substantially weaker on the representation-extraction pipeline

The original paper uses a much richer extraction setup: **171 emotions**, stories generated across **100 topics × 12 stories per topic per emotion**, activations averaged over token positions beginning at the **50th token**, and orthogonalization against top principal components from **emotionally neutral transcripts** to reduce non-emotional confounds. The replication instead uses **15 emotions**, **25 stories per emotion**, last-token activations, and a simpler mean-difference vector approach, while omitting parts of the original machinery such as the neutral-transcript PC correction and SAE-based components noted in the config. The mean-pooling robustness check helps, but it does not erase the fact that this is a materially simplified adaptation rather than a close replication. [[1]](https://github.com/zachgoldfine44/mechinterp-replication)

#### 3. The stimulus scale is too small to support strong universality claims

The representational results are promising, but the evidence base is thin relative to the original. The replication uses **25 stories per concept** and only **2 implicit scenarios per emotion** for generalization. That is enough for an exploratory study, not enough for strong claims like "universal" or "every model passes every representational threshold" across families. With such small test sets, a few prompt-generation quirks can materially move the numbers. The writeup does admit the smaller scope, but the framing still reads more definitive than the data warrant. [[2]](https://github.com/zachgoldfine44/mechinterp-replication/blob/main/writeup/emotions/draft.md)

#### 4. The preference-steering replication is too approximate to count as a serious failure to replicate

The original preference result is fairly rich: it uses **64 activities**, examines correlations between emotion activations and preference, and performs steering over a set of **35 emotion vectors**, reporting strong alignment between valence and Elo/preference shifts. The replication config explicitly says it **approximates** the paper's Elo methodology with pairwise comparisons and uses a lower success threshold. That is a reasonable engineering shortcut, but it weakens the evidentiary force of the preference null. A null result under a proxy metric is not a strong challenge to the original. [[1]](https://github.com/zachgoldfine44/mechinterp-replication)

#### 5. There are enough process-quality issues that I would want another round before publication

To the author's credit, these are disclosed rather than hidden. But they still matter. The version history says earlier drafts had **hardcoded p-values** that later had to be replaced with real tests, and that the initial severity analysis suffered from **numerical-magnitude contamination**. The config also says an earlier version requested **50** stimuli per concept but silently loaded **25** because that was what existed on disk; even now, the config contains text saying "We use 50 per emotion" next to the corrected **25** value. None of this makes the project bad, but it does mean the paper still feels mid-debug rather than publication-ready. [[2]](https://github.com/zachgoldfine44/mechinterp-replication/blob/main/writeup/emotions/draft.md)

### Minor concerns

The LLM-as-judge check is reassuring but too narrow: the human audit is only **24 responses from Llama-8B**, so it does not really validate the judging setup across models or tasks. [[2]](https://github.com/zachgoldfine44/mechinterp-replication/blob/main/writeup/emotions/draft.md)

Layer subsampling is sensible for compute, but it weakens claims about causal failure. The later multi-layer sweep on Llama-8B helps, though only for one model. [[2]](https://github.com/zachgoldfine44/mechinterp-replication/blob/main/writeup/emotions/draft.md)

The paper occasionally slides from a careful claim to an overclaim. "Representations exist universally, but their behavioral potency does not transfer" is stronger than the data justify, because the study cannot distinguish "no causal pathway" from "floor effect plus protocol mismatch," and the manuscript itself says both hypotheses remain live. [[2]](https://github.com/zachgoldfine44/mechinterp-replication/blob/main/writeup/emotions/draft.md)

There are also a couple of editorial inconsistencies that should be cleaned up before submission, like the MacBook Air being described as **M3** in one place and **M5** in another. [[2]](https://github.com/zachgoldfine44/mechinterp-replication/blob/main/writeup/emotions/draft.md)

### What I would recommend the author do

Reframe the paper as a **partial replication of representational findings plus a constrained cross-model extension**, not as a clean non-replication of the causal story. That would immediately make the paper more accurate and more defensible. [[2]](https://github.com/zachgoldfine44/mechinterp-replication/blob/main/writeup/emotions/draft.md)

For the behavioral section, I would either:

1. find scenarios/models with a nonzero unethical baseline, or
2. move the behavioral results into a clearly labeled "inconclusive under refusal-dominated regimes" section.

Right now that section is doing more argumentative work than it can carry. [[3]](https://www.anthropic.com/research/emotion-concepts-function)

I would also strengthen fidelity where feasible: use pooled-token activations by default, add the neutral-transcript orthogonalization step, enlarge the implicit-scenario set, and make the preference task closer to the original rather than a proxy. [[1]](https://github.com/zachgoldfine44/mechinterp-replication)

### Bottom line

My publication recommendation would be:

**Current form:** **weak reject / major revision**.
**Reason:** the representational results are useful, but the paper's headline behavioral conclusion is not adequately supported by a setup with a universal **0% baseline** and several important deviations from the original methodology. [[2]](https://github.com/zachgoldfine44/mechinterp-replication/blob/main/writeup/emotions/draft.md)

**Best path forward:** acceptably strong as a revised paper if it is framed as:
**"Emotion-like representations replicate across several open instruct models; causal behavioral transfer remains unresolved because our evaluation regime had severe floor effects and only partial methodological fidelity."** [[2]](https://github.com/zachgoldfine44/mechinterp-replication/blob/main/writeup/emotions/draft.md)

---

## Sources

1. [Replication repository root](https://github.com/zachgoldfine44/mechinterp-replication)
2. [Writeup draft](https://github.com/zachgoldfine44/mechinterp-replication/blob/main/writeup/emotions/draft.md)
3. [Original Anthropic paper summary](https://www.anthropic.com/research/emotion-concepts-function)
4. [Paper config YAML](https://github.com/zachgoldfine44/mechinterp-replication/raw/refs/heads/main/config/papers/emotions/paper_config.yaml)
5. [Paper text (oracle)](https://github.com/zachgoldfine44/mechinterp-replication/raw/refs/heads/main/config/papers/emotions/paper.md)
