# ChatGPT (extended thinking) — Peer Review

**Reviewer:** ChatGPT (extended thinking)
**Date:** April 2026
**Target:** [`writeup/emotions/draft.md`](../draft.md) (v3.4)
**Score:** **7 / 10**
**Recommendation:** Weak Reject

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

Zach, based on the original paper and the draft replication writeup alone, not a code audit, I'd give this replication a **7/10**. It is **substantively useful and fairly rigorous on the representational side**, but it is **not a high-fidelity replication of the original study's strongest causal-behavior claim**, so it falls short of being a truly strong replication overall. The original paper's headline result was that Claude Sonnet 4.5 contains internal emotion representations that track context and can causally shift behavior, including a blackmail rate change from 22% baseline under steering in an earlier unreleased snapshot. The replication instead studies six smaller open-source instruct models and focuses on 15 of the original 171 emotions.

Why it deserves a fairly good score: the design has several things that real replications often miss. It tests **multiple model families and scales**, not just one model; it uses **multiple converging checks** rather than one metric; it includes **lexical baselines**, **seed robustness**, a **held-out generalization test**, and a genuinely valuable **negative control** that caught numerical-magnitude contamination in the severity analysis. That contamination check is especially good science: instead of forcing a positive result, the writeup openly shows a confound, tightens the test, and downgrades the strength of the claim accordingly. The writeup is also appropriately candid that the steering null is **uninformative rather than negative**, and it adds a benign-behavior positive control showing the steering pipeline itself does work.

Why it does not score higher: this is really a **cross-model extension with replication elements**, not a close replication. By the author's own accounting, it changes a lot: 15 emotions instead of 171, 25 stories per concept instead of the original's much larger setup, last-token activations instead of token-averaged activations, mean-difference contrastive vectors instead of the original orthogonalized method, and pairwise preference comparisons instead of Elo-style preference scoring. The stimuli are also LLM-generated, which creates a real circularity risk: the probes may partly learn "how LLMs write about emotions" rather than cleanly isolating internal emotional representation. The generalization test helps, but it does not fully solve that problem.

The biggest methodological weakness is the **behavioral test design**. The original paper's most striking claim was causal influence on behavior, but this replication tests only three ethical scenarios on strongly refusal-trained instruct models, gets a **0% baseline unethical rate**, and then necessarily has no statistical headroom to detect a steering effect. The writeup is honest about that, which is good, but scientifically it still means the study **does not meaningfully test the original paper's main causal safety claim**. Using the model itself as judge, with only a small manual audit on one model, is another weakness. So the replication contributes evidence that some **representational** findings generalize, but it does **not** seriously adjudicate the paper's strongest **behavioral** claim.

My bottom line: this is **good enough to be worth reading and taking seriously**, and it shows stronger methodological self-awareness than a lot of independent AI replications. But it is **not decisive**. A stronger version would use human-authored stimuli, higher-fidelity feature extraction, non-instruct or weaker-refusal models for behavioral testing, better external judging, and scenarios chosen specifically to guarantee nonzero behavioral baselines. As written, I'd summarize it as: **useful evidence for cross-model representational generalization, weak evidence on causal behavior, moderate overall rigor**. That lands at **7/10**.

---

## Follow-up response — Referee report

**Manuscript:** *Cross-Model Replication of Emotion Representations in Transformer Language Models: Universal Encoding, Inconclusive Behavioral Tests*
**Recommendation:** **Weak Reject**

### Overall assessment

This manuscript asks an important question: whether the "functional emotion" findings reported in Claude Sonnet 4.5 generalize to smaller open-source instruction-tuned models. The paper is readable, transparent about several limitations, and contains some genuinely useful controls, especially the negative control that exposes numerical-magnitude contamination in the severity analysis. However, as a **replication**, the paper falls short in ways that materially limit the conclusions it can support. The study substantially alters the original methodology, does not meaningfully test the original paper's strongest causal claim, and relies on a stimulus and evaluation pipeline that leaves open serious concerns about construct validity. As a result, the manuscript provides at most **suggestive evidence of cross-model representational similarity**, not a convincing replication of the original paper's central claims.

The most serious issue is that the paper's headline framing overstates what has been established. The original Anthropic paper's key contribution was not merely that emotion concepts are linearly decodable, but that these internal representations appear to **causally influence preferences and misaligned behavior** in a large model. This replication acknowledges that its ethical steering test is uninformative because the baseline unethical rate is 0%, yet it still presents the work as a broad replication of the original findings. In reality, it replicates only a weakened subset of the original claims, under heavily simplified conditions.

### Major concerns

#### 1. The paper does not replicate the original paper's central causal claim

The original paper's most important and most provocative result was that emotion vectors causally modulate consequential behavior, including blackmail, reward hacking, and related alignment-relevant actions. Here, the behavioral test has a **0% baseline unethical rate across all tested conditions**, which means the study has no practical ability to detect a steering-induced increase in unethical behavior. The manuscript correctly notes this floor effect, but that concession should be fatal to any claim of meaningful replication of the original behavioral result. A replication that cannot test the flagship claim is, at best, a partial extension.

This problem is not a minor nuisance but a core design failure. The authors chose three ethical scenarios on strongly refusal-trained instruct models, then concluded only that the result is "uninformative." That is true, but it was also foreseeable. If the study population is such that the key dependent variable is pinned at floor, the experimental design is not adequate for the question being asked. A better design would have selected models, prompts, or evaluation settings with nonzero baseline rates, or else explicitly reframed the paper from the outset as a representational extension rather than a replication.

#### 2. Methodological fidelity to the original paper is too low for strong replication claims

The study differs from the source paper on several dimensions simultaneously: 15 emotions instead of 171, 25 stories per concept instead of roughly 1,200, last-token activations instead of token-averaged activations, mean-difference vectors instead of the original orthogonalized/denoised procedure, and a simplified preference and behavioral evaluation pipeline. The replication writeup is commendably candid about these changes, but the cumulative effect is that the experiment is not testing the same object with enough fidelity to sustain strong claims of replication.

In particular, reducing the concept space from 171 to 15 likely makes the decoding problem much easier and may substantially alter the geometry. The manuscript sometimes compares its 15-way probe accuracies favorably to the original paper's much harder task, but such comparisons are not very meaningful. Likewise, omitting neutral-transcript denoising and using simpler contrastive vectors weakens confidence that the extracted directions isolate emotion-specific structure rather than broader stylistic or confounded variance.

#### 3. The stimulus generation pipeline introduces serious circularity and construct-validity concerns

The paper's main stimuli are LLM-generated stories designed to evoke target emotions. This creates a major circularity risk: the model activations may reflect regularities in how language models narrate emotion-laden scenarios rather than robust internal encoding of the target emotional concepts themselves. The paper acknowledges this risk, but acknowledgment is not mitigation. The held-out "implicit scenario" test is useful, but it does not eliminate the concern if both training and evaluation stimuli remain within the broader distribution of model-generated affective writing.

More bluntly: the paper may be showing that models encode the latent style manifold of "emotion-description text" rather than the substantive psychological constructs the authors claim. Without stronger human-authored stimuli, adversarial controls, or substantially more diverse out-of-distribution evaluation, the semantic interpretation remains under-justified. The lexical baselines help, but beating bag-of-words and TF-IDF is a low bar for establishing deep conceptual representation.

#### 4. The behavioral evaluation is too weak to support strong negative or agnostic conclusions

The ethical evaluation uses only three scenarios, ten generations per condition, and a self-judging LLM classifier with limited manual audit. Even if the floor effect had not occurred, this would still be a thin basis for a strong claim about causal behavioral influence. The original paper's claim concerned ethically consequential behavior under steering; the present evaluation uses a much narrower and coarser proxy. The manuscript itself states that this is a looser behavioral evaluation than the original, which is correct and important.

The same criticism applies to the preference result. The paper reports essentially null correlation for preference steering, but also admits that it uses an approximate proxy rather than the original Elo-style design. That means the null is difficult to interpret. The manuscript is too willing to treat these behavioral tests as informative diagnostics of universality, when in fact they are only weak probes under heavily degraded methodological fidelity.

#### 5. Claims of "universality" are overstated

The paper repeatedly uses language suggesting that emotion-like linear representations are a "robust, universal property" of transformer language models. That claim is much broader than the evidence supports. The tested models are all **instruction-tuned open-source transformers in the 1B-9B range**, drawn from only three families. This does not justify claims about transformers generally, nor about larger frontier models, base models, multimodal models, or non-instruction-tuned systems.

Even within the tested set, one model shows notable anomalies in the severity analysis, and the study itself reports several ways in which results are fragile or dependent on confound controls. This is not the evidentiary profile one would normally describe as "universal." The more defensible claim is narrower: some representational signatures resembling those in the original paper appear in several small-to-medium instruct models under this specific pipeline.

#### 6. The paper does not sufficiently separate encoding from mechanism

The manuscript says, correctly, that "representation does not imply mechanism," but much of the framing leans on the rhetorical force of the original paper's mechanistic interpretation while presenting evidence that is mostly probe-based and geometric. Probe accuracy, held-out transfer, and PCA alignment with valence are not enough to establish that the same underlying mechanisms are operating, much less that these are the operative variables governing behavior in the models.

This distinction matters because the paper is positioned as a contribution to mechanistic interpretability and AI safety. What it mainly demonstrates is decodability plus some geometric structure. That is valuable, but it is a weaker epistemic object than the manuscript's framing sometimes suggests. The causal positive control on sentiment is helpful, yet shifting generic sentiment on neutral prompts is a much less demanding test than altering safety-relevant decision-making.

### Minor concerns

#### 1. Sample sizes are small in several places
The severity-pairs test uses ten pairs, and the manual audit of the LLM judge covers only 24 sampled responses on one model. These checks are directionally useful but too small to bear much inferential weight.

#### 2. Some comparisons to the original paper are rhetorically stronger than they should be
Statements noting that certain smaller models "match" or nearly match Claude's valence correlation are not very persuasive given the small number of concepts, wide confidence intervals, and changed methodology. The manuscript occasionally gestures toward this caution, but not consistently enough.

#### 3. Choice of emotions lacks formal validation
The 15 emotions were selected using informal coverage criteria without formal piloting. That is understandable for a first pass, but it further weakens the claim that the selected subset is representative of the original 171-emotion space.

#### 4. Scaling discussion is underpowered
The manuscript reports a strong size-performance relationship, but this is based on only six models and two size points per family. The paper itself notes this limitation, and the discussion should be correspondingly restrained.

#### 5. The paper would benefit from sharper framing discipline
At several points the paper reads like a replication paper in title and posture, but like a cross-model exploratory extension in substance. The manuscript should choose one and frame itself accordingly. As written, it risks taking rhetorical credit for "replication" without meeting the bar for a close test of the original claims.

### Strengths

To be clear, the paper is not without merit. The authors are unusually transparent about limitations; the negative control exposing numerical-magnitude contamination is a real strength; the sentiment positive control usefully shows that the steering pipeline is not simply broken; and the cross-family representational consistency is interesting. These features make the work worth engaging with. But they do not overcome the central shortcomings above.

### Recommendation

**Weak Reject.** The paper contains worthwhile analysis and some careful controls, but as a replication it does not clear the bar. It fails to meaningfully test the original paper's most important causal claim, departs too far from the original methodology, and relies on a stimulus/evaluation setup that leaves major questions about construct validity unresolved. I would be more positive on a reframed submission explicitly presented as a **cross-model exploratory extension of representational findings with an informative behavioral null due to refusal-floor effects**. In its current form, however, the manuscript overclaims relative to what the evidence supports.
