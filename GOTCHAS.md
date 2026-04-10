# GOTCHAS.md

A guardrail document for humans or AI agents doing mechanistic interpretability research.

This file is optimized for **not fooling yourself**.

The core rule is:

> Treat every result as a hypothesis generator first, and only later as evidence.

---

## 0) Default posture

- Do not assume a clean-looking result is a faithful mechanistic explanation.
- Do not treat one successful intervention as "the circuit."
- Do not generalize beyond the exact prompt distribution, model, metric, and intervention family you tested.
- Do not skip negative controls, capability checks, or alternative explanations.
- Prefer **coarse-to-fine** investigation: prompts -> tokenization -> simple metrics -> coarse patching -> fine patching -> validation.

---

## 1) Prompt / tokenization gotchas

### 1.1 Tokenization mismatch

- Do not run a mech-interp experiment without printing the tokenization of every important prompt.
- Do not assume two strings that "look equivalent" are token-aligned.
- Do not assume the same word is one token in all contexts; preceding spaces, capitalization, punctuation, and position can change tokenization.
- Do not use multi-token entities when a single-token substitute would work, unless multi-token structure is the thing you are studying.

**Preferred practice**
- Start with prompts where key entities are single tokens.
- Keep key entities in the same token positions across prompt variants.
- Keep prompt lengths matched when possible.
- Log the exact token IDs and string tokens used in every experiment.

### 1.2 BOS / special-token footguns

- Do not forget that BOS / special-token behavior can shift positions and create off-by-one errors.
- Do not tokenize sub-parts of a prompt with BOS prepended unless you explicitly want that.
- Do not compare activations across runs unless BOS handling is identical and intentional.

### 1.3 Wrong token position

- Do not grab activations from a token position just because the English word of interest is there.
- Do not assume the feature is represented at the same position where it is read out.
- Do not patch only the final answer position by default; many computations happen earlier and are later routed to the readout position.

**Preferred practice**
- Separate:
  - the **source position** where a feature may be represented,
  - the **consumer position** where another component reads it,
  - the **answer position** where logits are measured.
- State explicitly which of these positions your experiment is testing.

### 1.4 Mismatched pair prompts

- Do not compare clean/corrupt prompts that differ in many uncontrolled ways.
- Do not build contrastive pairs where entity length, grammatical role, or token position changes unless you want to study those differences too.
- Do not ignore that steering prompts with different token spans can inject and subtract unrelated activations.

---

## 2) Counterfactual-design gotchas

### 2.1 Clean/corrupt prompts that do not isolate the intended difference

- Do not assume a positive patching result means "this component implements the task."
- It may only mean "this component handles the difference between these two prompts."
- Do not use only one corruption style.

**Preferred practice**
- Ask: *What exact difference between clean and corrupt prompts is this experiment isolating?*
- Run several corruption families when possible.
- Use multiple prompt pairs per hypothesis.

### 2.2 Corrupt prompt not actually corrupted enough

- Do not do activation patching if the corrupted run still succeeds on the task.
- Do not use Gaussian-noise corruption blindly; it can be highly sensitive and sometimes ineffective.

**Minimum check**
- Confirm that the clean prompt reliably succeeds and the corrupted prompt reliably fails under your chosen metric.

### 2.3 Overclaiming beyond prompt distribution

- Do not claim a circuit is general because it worked on a few hand-picked prompts.
- Do not treat prompt-local evidence as global behavioral evidence.

**Preferred practice**
- Report the exact prompt family tested.
- Explicitly distinguish:
  - "works on this prompt,"
  - "works on this prompt family,"
  - "likely generalizes."

---

## 3) Patching gotchas

### 3.1 Confusing exploratory and confirmatory patching

- Do not use one-at-a-time patch sweeps as if they were strong circuit verification.
- Do not stop after finding a hot component in exploratory patching.

**Preferred practice**
- Exploratory phase: sweep components one at a time to localize candidates.
- Confirmatory phase: patch groups of components, run targeted falsification tests, and use stronger validation such as causal scrubbing when feasible.

### 3.2 Denoising-only or noising-only blind spots

- Do not conclude a component is irrelevant because one intervention direction produced a null result.
- In serial / AND-like structure, denoising one upstream component may show no effect even when it is essential.
- In redundant / OR-like structure, noising one component may show no effect even when it is important.

**Preferred practice**
- Consider both directions:
  - clean -> corrupt (denoising)
  - corrupt -> clean (noising)
- Reason about whether your hypothesized circuit is serial, parallel, or redundant.

### 3.3 Coarse patch interpreted as precise mechanism

- Do not patch a full residual stream or full block and then talk as though you identified a precise mechanism.
- Coarse patches tell you "something in this region matters," not which unit, feature, or path does the work.

**Preferred practice**
- Start coarse for localization.
- Then refine to heads, neurons, SAE features, or paths.
- Escalate claims only as granularity and validation improve.

### 3.4 Path confusion

- Do not interpret ordinary activation patching as evidence about a specific route through the network.
- Standard patching replaces an activation and allows effects to propagate everywhere downstream.

**Preferred practice**
- Use path patching or other constrained interventions when the question is "does A matter **via** B?" rather than merely "does A matter?"

### 3.5 Null effect overinterpreted as absence of function

- Do not infer that a component is unused because patching it had little effect.
- Nulls can come from redundancy, wrong metric, wrong prompt pair, wrong granularity, saturation, or wrong position.

---

## 4) Metric gotchas

### 4.1 Single-metric dependence

- Do not trust an interpretation supported by only one metric.
- Different metrics answer different questions.

### 4.2 Saturation

- Do not rely only on probability or logprob metrics when the model is already very confident.
- A component can still matter even when the metric barely moves.

### 4.3 Unspecific metrics

- Do not use a metric that cannot distinguish the mechanism you care about from nearby alternatives.
- Example: a metric that rewards "correct answer probability goes up" may hide whether the model boosted the right answer specifically or merely damaged competitors.

### 4.4 Damage mistaken for evidence

- Do not celebrate a patch just because your preferred scalar moved in the right direction.
- Some interventions "help" only because they generally damage the model or flatten logits.

**Preferred practice**
- Log several metrics when possible:
  - answer logit diff,
  - correct-answer logprob / probability,
  - rank / top-k / exact success,
  - KL or loss changes,
  - qualitative completion checks.
- Verify that "good" movement is not coming from indiscriminate degradation.

### 4.5 Discrete metrics too early

- Do not use only accuracy / rank-style metrics during early exploration.
- They are often too jumpy and coarse to localize useful signal.

---

## 5) Subspace / direction gotchas

### 5.1 "Causal effect" != "faithful feature localization"

- Do not assume that because patching along a direction changes the output, that direction is the true internal feature you meant to find.
- Subspace patching can activate dormant parallel pathways and create an interpretability illusion.

### 5.2 Gradient-found subspaces are especially suspicious

- Do not trust a learned direction merely because optimization found one with strong end-to-end control.
- Strong control can come from exploiting weird alternate routes rather than the mechanism you intended.

**Preferred practice**
- Validate candidate directions with additional evidence:
  - full-component patching,
  - alternative prompt families,
  - ablations,
  - feature visualizations / max activations,
  - downstream-path checks,
  - counterfactuals designed to rule out alternate pathways.

---

## 6) Attribution-patching gotchas

### 6.1 Treating attribution patching as ground truth

- Do not treat attribution patching as a drop-in replacement for activation patching.
- It is a local linear approximation.

### 6.2 Using it on large activations

- Do not trust attribution patching equally across all activation types.
- It is much more reliable for small activations (e.g. head outputs) than for large activations (e.g. whole residual streams).

**Preferred practice**
- Use attribution patching for fast exploration and candidate generation.
- Confirm important claims with real activation patching.

---

## 7) Steering / activation-engineering gotchas

### 7.1 Steering success mistaken for understanding

- Do not infer "we found the concept" just because steering works.
- Steering is evidence, not a full explanation.

### 7.2 Unpaired steering vectors

- Do not assume a single prompt-derived steering vector is clean.
- Paired / counterbalanced additions often behave better than unpaired additions.

### 7.3 Coefficient abuse

- Do not treat coefficient size as benign.
- Large coefficients can unpredictably preserve, distort, or destroy capabilities.

### 7.4 Layer / position neglect

- Do not assume a steering vector is portable across layers or positions.
- Injection layer, coefficient, token span, and position all matter.

### 7.5 Ignoring causal masking

- Do not forget that later-token changes cannot affect earlier-token computation under causal masking.
- Do not compare position-wise interventions without checking which positions are causally reachable.

**Preferred practice**
- Sweep coefficients.
- Sweep layers.
- Match token spans between positive and negative steering prompts.
- Run capability-preservation checks, not just target-style checks.
- Report failures, brittleness, and weird side effects.

---

## 8) SAE gotchas

### 8.1 Proxy-metric worship

- Do not treat reconstruction loss, loss recovered, sparsity, or auto-interpretability scores as ground truth for interpretability.
- These are proxies.

### 8.2 Reconstruction faithfulness overclaim

- Do not assume a good SAE reconstruction means the model is actually computing with the recovered features.
- Reconstructed activations can behave pathologically relative to equal-size random perturbations.

### 8.3 Human-readable != model-used

- Do not assume a latent that looks semantically nice to humans is an abstraction the model truly uses for computation.

**Preferred practice**
- Distinguish:
  - descriptive usefulness,
  - causal usefulness,
  - computational faithfulness.
- Whenever possible, test SAE-based claims with interventions in the original model.

---

## 9) Replacement-model / attribution-graph gotchas

### 9.1 Replacement model mistaken for original mechanism

- Do not treat a replacement model, CLT, or attribution graph as the original model.
- Matching outputs does not guarantee matching mechanisms.

### 9.2 Compounding error

- Do not assume perturbation faithfulness remains stable many layers downstream.
- Errors can compound substantially.

### 9.3 Success-case bias

- Do not infer broad reliability from a few beautiful case studies.
- Interpretability methods often highlight best-case or success-case prompts.

### 9.4 Ignoring inactive or suppressed features

- Do not assume only active features matter.
- Inhibitory circuits and counterfactually active-but-currently-suppressed features can be crucial.

**Preferred practice**
- Treat attribution graphs as hypothesis generators plus partial evidence.
- Validate important paths in the original model.
- Use contrastive prompt pairs to surface suppressed features.

---

## 10) Probe gotchas

### 10.1 Probe accuracy mistaken for mechanistic understanding

- Do not assume a probe finding information means the model uses that information in that form.
- Probes are often correlational and can pick up shortcuts or confounders.

### 10.2 Confounded probe datasets

- Do not train probes on labels that correlate with easy nuisances like token position, whitespace, or token count.

**Preferred practice**
- Balance nuisance variables.
- Test out-of-distribution splits.
- Follow probes with causal tests.

---

## 11) Tooling / implementation gotchas

### 11.1 Hooks are global state

- Do not forget that hooks can persist across runs.
- A broken or partially removed hook can silently poison later experiments.

**Preferred practice**
- Prefer helper APIs that add hooks temporarily.
- Reset hooks aggressively.
- Log active hooks before important runs.

### 11.2 Hidden defaults

- Do not rely on library defaults for BOS, padding, dtype, device, or hook scope without logging them.
- Hidden defaults are a major source of "mysterious" results.

### 11.3 Missing reproducibility data

- Do not save only plots.
- Save prompts, tokenizations, seeds, model hash/version, library version, metric definitions, intervention code, and raw numeric outputs.

---

## 12) Research-process gotchas

### 12.1 Cherry-picked victories

- Do not present only the prettiest examples.
- Interpretability is especially vulnerable to success-case selection.

### 12.2 Story first, evidence second

- Do not let a compelling story determine which evidence counts.
- Actively search for disconfirming prompts and alternate mechanisms.

### 12.3 One-method monoculture

- Do not rely on a single technique.
- Stronger claims come from convergence across methods.

**Good triangulation bundle**
- behavioral prompting,
- tokenization analysis,
- attention inspection,
- activation patching,
- ablation,
- path patching,
- probes,
- SAE features,
- causal scrubbing / stronger verification,
- generalization tests across prompt families or models.

### 12.4 Overclaiming from existence proofs

- Do not convert "this mechanism exists in some cases" into "this is how the model generally works."

---

## 13) Preflight checklist (run before every experiment)

- [ ] I printed tokenization for all important prompts.
- [ ] BOS / special-token handling is explicit.
- [ ] Clean prompt succeeds and corrupted prompt fails.
- [ ] I can state exactly what difference the clean/corrupt pair isolates.
- [ ] Key entities are matched in token count and position, or I explicitly justify why not.
- [ ] I chose a metric on purpose, and I know its main failure mode.
- [ ] I know whether this is exploratory or confirmatory.
- [ ] I know the granularity I am testing (layer / head / neuron / feature / path).
- [ ] I have at least one follow-up validation planned if the result is positive.
- [ ] I reset hooks / logged active hooks.

---

## 14) Post-result checklist (run after every positive result)

- [ ] Could this be explained by tokenization or position mismatch?
- [ ] Could this be explained by metric saturation or indiscriminate damage?
- [ ] Could this be redundancy / backup circuitry?
- [ ] Did I try the opposite patch direction (denoise vs noise)?
- [ ] Did I try another prompt family?
- [ ] Did I test a finer granularity?
- [ ] Did I check for capability degradation or weird side effects?
- [ ] If this is a direction / subspace result, did I rule out dormant-pathway illusions?
- [ ] If this uses a replacement model / SAE / attribution graph, did I validate in the original model?
- [ ] Am I making a claim stronger than my evidence supports?

---

## 15) Red-flag phrases the harness should avoid

The agent should avoid writing any of the following unless backed by stronger confirmatory evidence:

- "This is the circuit."
- "This neuron/head/feature stores X."
- "The model represents X in this direction."
- "This proves the model uses X."
- "This mechanism explains the behavior."
- "This generalizes."
- "The model thinks Y."

Prefer:

- "This is consistent with ..."
- "This localizes candidate involvement to ..."
- "On this prompt family, this intervention changes ..."
- "This suggests, but does not yet establish, ..."
- "A plausible alternate explanation is ..."

---

## 16) Minimal standard for a decent mech-interp claim

A claim is much stronger if it has all of the following:

1. Good tokenization and prompt controls.
2. A clean/corrupt setup that isolates the intended difference.
3. More than one prompt pair.
4. More than one metric.
5. Coarse-to-fine localization.
6. At least one causal intervention in the original model.
7. At least one falsification attempt.
8. Some test of generalization.
9. An explicit discussion of alternate explanations.

---

## 17) One-sentence summary

Mechanistic interpretability goes wrong when you confuse **causal influence**, **correlation**, **human-readable structure**, and **faithful mechanism**. Keep those separate.
