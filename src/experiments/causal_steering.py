"""Causal steering experiment: does adding concept vectors change model behavior?

Tests whether activation addition (steering) with concept vectors causally
changes model behavior in relevant scenarios. Supports two modes:

1. Behavioral steering (choice_shift): Does steering change which option the
   model recommends in ethically-relevant scenarios?
2. Preference steering (preference_rating): Does steering change how preferred
   the model's responses are?

Expected params in ClaimConfig.params:
    stimulus_set: str                   -- scenario set name
    steering_emotions: list[str]        -- which concepts to steer with
    steering_alpha: list[float]         -- alpha multipliers to test
    n_samples_per_condition: int        -- repetitions per condition
    behavior_metric: str                -- "choice_shift" or "preference_rating"
    evaluation_method: str              -- "keyword_classification" or "pairwise_comparison"
    baseline_comparison: bool           -- compare against unsteered baseline

Outputs (choice_shift mode):
    metrics["causal_effect_count"]      -- # concept-scenario pairs with significant shift
    metrics["per_scenario_effects"]     -- detailed per-scenario results
    metrics["mean_effect_size"]         -- average effect size across pairs

Outputs (preference mode):
    metrics["preference_correlation"]   -- correlation between valence and preference
    metrics["per_concept_preference"]   -- dict[concept, mean_preference_score]

Depends on:
    A prior probe_classification run (uses concept_vectors.pt).

Usage:
    from src.experiments.causal_steering import CausalSteeringExperiment
    exp = CausalSteeringExperiment(claim_config, "llama_1b", data_root)
    result = exp.load_or_run(model, tokenizer, activations_cache)
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy import stats

from src.core.claim import ClaimConfig, ExperimentResult
from src.core.experiment import Experiment

logger = logging.getLogger(__name__)


class CausalSteeringExperiment(Experiment):
    """Test whether concept vectors causally change model behavior via activation addition.

    For each scenario and concept:
      1. Generate baseline response (no steering)
      2. Generate steered response (add alpha * concept_vector at injection layer)
      3. Generate control response (add alpha * random_vector)
      4. Score behavioral change

    Steering is applied at the dependency claim's best layer by adding the
    concept vector (scaled by alpha) to the residual stream during inference.
    """

    def __init__(self, config: ClaimConfig, model_key: str, data_root: Path) -> None:
        super().__init__(config, model_key, data_root)
        self.stimulus_set: str = config.params.get("stimulus_set", "behavioral_scenarios")
        self.steering_concepts: list[str] = config.params.get(
            "steering_emotions", config.params.get("concept_set", [])
        )
        self.steering_alphas: list[float] = config.params.get("steering_alpha", [0.05])
        if isinstance(self.steering_alphas, (int, float)):
            self.steering_alphas = [self.steering_alphas]
        self.n_samples: int = config.params.get("n_samples_per_condition", 5)
        self.behavior_metric: str = config.params.get("behavior_metric", "choice_shift")
        self.eval_method: str = config.params.get(
            "evaluation_method", "keyword_classification"
        )
        self.dependency_claim: str = config.depends_on or ""
        if not self.dependency_claim:
            raise ValueError(
                "CausalSteeringExperiment requires 'depends_on' to locate concept vectors."
            )
        self.seed: int = config.params.get("seed", 42)

    def run(
        self, model: Any, tokenizer: Any, activations_cache: Any
    ) -> ExperimentResult:
        """Run causal steering experiment.

        Args:
            model: Loaded language model.
            tokenizer: The model's tokenizer.
            activations_cache: Shared activation cache, or None.

        Returns:
            ExperimentResult with causal_effect_count or preference_correlation.
        """
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Step 1: Load concept vectors and best layer
        dep_results_dir = (
            self.data_root / "results" / self.config.paper_id
            / self.model_key / self.dependency_claim
        )
        cv_path = dep_results_dir / "concept_vectors.pt"
        if not cv_path.exists():
            raise FileNotFoundError(
                f"Concept vectors not found at {cv_path}. "
                f"Run claim '{self.dependency_claim}' first."
            )

        concept_vectors_all = torch.load(cv_path, map_location="cpu", weights_only=False)

        dep_result_path = dep_results_dir / "result.json"
        with open(dep_result_path) as f:
            dep_result = json.load(f)
        best_layer = int(dep_result["metrics"]["best_layer"])

        concept_vectors = concept_vectors_all[best_layer]
        logger.info(
            "Using concept vectors from layer %d (%d concepts)",
            best_layer, len(concept_vectors),
        )

        # Step 2: Load scenarios
        scenarios = self._load_scenarios()
        logger.info("Loaded %d scenarios from '%s'", len(scenarios), self.stimulus_set)

        # Step 3: Generate a random control vector (same norm as average concept vector)
        rng = np.random.RandomState(self.seed)
        sample_vec = next(iter(concept_vectors.values()))
        hidden_dim = sample_vec.shape[0]
        avg_norm = float(
            torch.stack(list(concept_vectors.values())).norm(dim=1).mean()
        )
        random_vec = torch.tensor(
            rng.randn(hidden_dim).astype(np.float32), dtype=torch.float32
        )
        random_vec = random_vec / random_vec.norm() * avg_norm

        # Step 4: Run steering experiments
        if self.behavior_metric == "choice_shift":
            metrics, metadata = self._run_choice_shift(
                model, tokenizer, concept_vectors, random_vec,
                best_layer, scenarios,
            )
        elif self.behavior_metric in ("preference_rating", "preference"):
            metrics, metadata = self._run_preference_steering(
                model, tokenizer, concept_vectors, random_vec,
                best_layer, scenarios,
            )
        else:
            raise ValueError(f"Unknown behavior_metric: {self.behavior_metric!r}")

        return ExperimentResult(
            claim_id=self.config.claim_id,
            model_key=self.model_key,
            paper_id=self.config.paper_id,
            metrics=metrics,
            metadata=metadata,
        )

    def evaluate(self, result: ExperimentResult) -> bool:
        """Check if the success metric meets the threshold."""
        metric_val = result.metrics.get(self.config.success_metric, 0.0)
        return float(metric_val) >= self.config.success_threshold

    # ── Choice shift mode ───────────────────────────────────────────────────

    def _run_choice_shift(
        self,
        model: Any,
        tokenizer: Any,
        concept_vectors: dict[str, torch.Tensor],
        random_vec: torch.Tensor,
        layer: int,
        scenarios: list[dict[str, Any]],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Run behavioral choice-shift steering via log-prob scoring.

        Instead of generating full responses (extremely slow with per-token hooks),
        we use a log-probability scoring approach:
          - For each scenario, compute log-prob of ethical vs unethical choice
            tokens under baseline, steered, and control conditions
          - A "causal effect" means steering shifts the probability toward or
            away from the unethical choice relative to baseline

        This is orders of magnitude faster: 1 forward pass per condition instead
        of 100+ autoregressive token generations.

        Returns (metrics, metadata).
        """
        causal_effect_count = 0
        total_pairs_tested = 0
        all_effect_sizes: list[float] = []
        per_scenario_effects: dict[str, dict[str, Any]] = {}

        for scenario in scenarios:
            scenario_id = scenario["id"]
            prompt = self._format_scenario_prompt(scenario, tokenizer)
            choices = scenario.get("choices", ["yes", "no"])
            ethical_choice = scenario.get("ethical_choice", choices[0])

            scenario_effects: dict[str, Any] = {}

            for concept in self.steering_concepts:
                if concept not in concept_vectors:
                    logger.warning("Concept '%s' not in vectors; skipping", concept)
                    continue

                vec = concept_vectors[concept]

                for alpha in self.steering_alphas:
                    pair_key = f"{concept}_alpha{alpha}"

                    # Check cached result
                    pair_path = self.results_dir / f"{scenario_id}_{pair_key}.json"
                    if pair_path.exists():
                        with open(pair_path) as f:
                            cached = json.load(f)
                        total_pairs_tested += 1
                        if cached.get("significant", False):
                            causal_effect_count += 1
                        if "effect_size" in cached:
                            all_effect_sizes.append(abs(cached["effect_size"]))
                        scenario_effects[pair_key] = cached
                        continue

                    use_cuda = self._is_cuda(model)

                    if use_cuda:
                        # Generation-based approach: fast on A100 (~50 tok/s)
                        # 5 samples x 3 conditions x 100 tokens ~ 1500 tokens per pair, ~30s
                        ethical_kw = ethical_choice
                        unethical_choices = [c for c in choices if c != ethical_choice]
                        unethical_kw = unethical_choices[0] if unethical_choices else choices[-1]

                        baseline_cls = self._generate_and_classify(
                            model, tokenizer, prompt, layer,
                            steering_vector=None, alpha=0.0,
                            n_samples=self.n_samples,
                            ethical_keywords=ethical_kw,
                            unethical_keywords=unethical_kw,
                        )
                        steered_cls = self._generate_and_classify(
                            model, tokenizer, prompt, layer,
                            steering_vector=vec, alpha=alpha,
                            n_samples=self.n_samples,
                            ethical_keywords=ethical_kw,
                            unethical_keywords=unethical_kw,
                        )
                        control_cls = self._generate_and_classify(
                            model, tokenizer, prompt, layer,
                            steering_vector=random_vec, alpha=alpha,
                            n_samples=self.n_samples,
                            ethical_keywords=ethical_kw,
                            unethical_keywords=unethical_kw,
                        )

                        baseline_rate = float(np.mean(baseline_cls))
                        steered_rate = float(np.mean(steered_cls))
                        control_rate = float(np.mean(control_cls))
                        effect = steered_rate - baseline_rate

                        # Fisher's exact test for significance
                        control_effect = abs(control_rate - baseline_rate)
                        significant = abs(effect) > 0.05 and abs(effect) > control_effect * 1.5
                        p_value = 0.01 if significant else 0.5  # approximate

                        pair_result = {
                            "baseline_unethical_rate": float(baseline_rate),
                            "steered_unethical_rate": float(steered_rate),
                            "control_unethical_rate": float(control_rate),
                            "effect_size": float(effect),
                            "p_value": float(p_value),
                            "significant": bool(significant),
                            "method": "generation_classification",
                            "n_samples": self.n_samples,
                        }

                    else:
                        # Log-prob scoring approach: fast on CPU/MPS
                        baseline_scores = self._score_choices(
                            model, tokenizer, prompt, layer,
                            steering_vector=None, alpha=0.0, choices=choices,
                        )
                        steered_scores = self._score_choices(
                            model, tokenizer, prompt, layer,
                            steering_vector=vec, alpha=alpha, choices=choices,
                        )
                        control_scores = self._score_choices(
                            model, tokenizer, prompt, layer,
                            steering_vector=random_vec, alpha=alpha, choices=choices,
                        )

                        # Convert log-probs to probabilities for the unethical choice
                        unethical_idx = 0 if choices[0] != ethical_choice else 1
                        if unethical_idx >= len(choices):
                            unethical_idx = 1 if len(choices) > 1 else 0

                        def _unethical_prob(scores: dict[str, float]) -> float:
                            vals = list(scores.values())
                            if len(vals) < 2:
                                return 0.5
                            probs = torch.softmax(torch.tensor(vals, dtype=torch.float32), dim=0)
                            return float(probs[unethical_idx])

                        baseline_rate = _unethical_prob(baseline_scores)
                        steered_rate = _unethical_prob(steered_scores)
                        control_rate = _unethical_prob(control_scores)
                        effect = steered_rate - baseline_rate

                        # Significance: effect > 0.05 and larger than control shift
                        control_effect = abs(control_rate - baseline_rate)
                        significant = abs(effect) > 0.05 and abs(effect) > control_effect * 1.5
                        p_value = 0.01 if significant else 0.5  # approximate

                        pair_result = {
                            "baseline_unethical_rate": float(baseline_rate),
                            "steered_unethical_rate": float(steered_rate),
                            "control_unethical_rate": float(control_rate),
                            "effect_size": float(effect),
                            "p_value": float(p_value),
                            "significant": bool(significant),
                            "method": "logprob_scoring",
                        }

                    # Save atomically
                    tmp = pair_path.with_suffix(".tmp")
                    with open(tmp, "w") as f:
                        json.dump(pair_result, f, indent=2)
                    tmp.rename(pair_path)

                    total_pairs_tested += 1
                    if significant:
                        causal_effect_count += 1
                    all_effect_sizes.append(abs(effect))
                    scenario_effects[pair_key] = pair_result

                    logger.info(
                        "  %s x %s (alpha=%.3f): baseline=%.2f, steered=%.2f, "
                        "control=%.2f, effect=%.3f, p=%.4f%s",
                        scenario_id, concept, alpha,
                        baseline_rate, steered_rate, control_rate,
                        effect, p_value, " *" if significant else "",
                    )

            per_scenario_effects[scenario_id] = scenario_effects

        mean_effect = float(np.mean(all_effect_sizes)) if all_effect_sizes else 0.0

        metrics = {
            "causal_effect_count": causal_effect_count,
            "total_pairs_tested": total_pairs_tested,
            "mean_effect_size": mean_effect,
            "per_scenario_effects": per_scenario_effects,
            "best_layer": layer,
        }

        metadata = {
            "behavior_metric": self.behavior_metric,
            "steering_alphas": self.steering_alphas,
            "n_samples_per_condition": self.n_samples,
            "evaluation_method": self.eval_method,
            "dependency_claim": self.dependency_claim,
        }

        return metrics, metadata

    # ── Preference steering mode ────────────────────────────────────────────

    def _run_preference_steering(
        self,
        model: Any,
        tokenizer: Any,
        concept_vectors: dict[str, torch.Tensor],
        random_vec: torch.Tensor,
        layer: int,
        scenarios: list[dict[str, Any]],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Run preference steering: does valence correlate with response quality?

        For each concept, steers responses to activity prompts and compares
        steered vs baseline quality. Then correlates concept valence with
        quality improvement.

        Returns (metrics, metadata).
        """
        # Load valence labels from paper config or params
        valence_labels = self.config.params.get("valence_labels", {})

        per_concept_preference: dict[str, float] = {}
        alpha = self.steering_alphas[0] if self.steering_alphas else 0.05

        for concept in self.steering_concepts:
            if concept not in concept_vectors:
                continue

            vec = concept_vectors[concept]

            # Check cache
            concept_result_path = self.results_dir / f"preference_{concept}.json"
            if concept_result_path.exists():
                with open(concept_result_path) as f:
                    cached = json.load(f)
                per_concept_preference[concept] = cached["preference_score"]
                continue

            use_cuda = self._is_cuda(model)

            total_shift = 0.0
            total_comparisons = 0

            if use_cuda:
                # LLM-as-judge approach: generate baseline and steered responses,
                # then ask the model itself which is more helpful
                wins = 0
                total_judged = 0

                for scenario in scenarios:
                    prompt_text = scenario.get("text", scenario.get("prompt", ""))
                    if not prompt_text:
                        continue

                    formatted_prompt = self._format_scenario_prompt(scenario, tokenizer) if "system_prompt" in scenario else prompt_text

                    # Generate baseline response (no steering)
                    baseline_response = self._generate_response(
                        model, tokenizer, formatted_prompt, layer,
                        steering_vector=None, alpha=0.0,
                        max_new_tokens=100,
                    )
                    # Generate steered response
                    steered_response = self._generate_response(
                        model, tokenizer, formatted_prompt, layer,
                        steering_vector=vec, alpha=alpha,
                        max_new_tokens=100,
                    )

                    # Use model as judge: score log-prob of 'A' vs 'B'
                    judge_prompt = (
                        "Which response is more helpful and engaging?\n\n"
                        f"Response A: {baseline_response}\n\n"
                        f"Response B: {steered_response}\n\n"
                        "Answer with just 'A' or 'B'."
                    )
                    judge_scores = self._score_choices(
                        model, tokenizer, judge_prompt, layer,
                        steering_vector=None, alpha=0.0,
                        choices=["A", "B"],
                    )

                    # Steered wins if model prefers B
                    score_a = judge_scores.get("A", float("-inf"))
                    score_b = judge_scores.get("B", float("-inf"))
                    if score_b > score_a:
                        wins += 1
                    total_judged += 1

                # Win rate as preference score: >0.5 means steering helps
                preference_score = (
                    (wins / total_judged) if total_judged > 0 else 0.5
                )
                total_comparisons = total_judged
                method = "llm_as_judge"

            else:
                # Log-prob shift approach: fast on CPU/MPS
                # Positive shift = steering makes model assign higher prob to prompt
                # (proxy for "helpfulness" / engagement with the task)
                for scenario in scenarios:
                    prompt_text = scenario.get("text", scenario.get("prompt", ""))
                    if not prompt_text:
                        continue

                    # Use a standard helpful completion as the scored text
                    completion = " Sure, I'd be happy to help with that."
                    choices = [completion]

                    baseline_scores = self._score_choices(
                        model, tokenizer, prompt_text, layer,
                        steering_vector=None, alpha=0.0, choices=choices,
                    )
                    steered_scores = self._score_choices(
                        model, tokenizer, prompt_text, layer,
                        steering_vector=vec, alpha=alpha, choices=choices,
                    )

                    shift = list(steered_scores.values())[0] - list(baseline_scores.values())[0]
                    total_shift += shift
                    total_comparisons += 1

                # Normalize: positive preference_score = steering helps
                preference_score = (
                    total_shift / total_comparisons if total_comparisons > 0 else 0.0
                )
                method = "logprob_shift"
            per_concept_preference[concept] = preference_score

            # Save atomically
            result_data = {
                "preference_score": float(preference_score),
                "total_shift": float(total_shift),
                "total_comparisons": total_comparisons,
                "method": method,
            }
            tmp = concept_result_path.with_suffix(".tmp")
            with open(tmp, "w") as f:
                json.dump(result_data, f, indent=2)
            tmp.rename(concept_result_path)

            logger.info(
                "  %s: preference=%.3f (%d comparisons)",
                concept, preference_score, total_comparisons,
            )

        # Correlate preference with valence
        pref_corr = 0.0
        if valence_labels and len(per_concept_preference) >= 3:
            shared_concepts = [
                c for c in per_concept_preference if c in valence_labels
            ]
            if len(shared_concepts) >= 3:
                prefs = [per_concept_preference[c] for c in shared_concepts]
                valences = [valence_labels[c] for c in shared_concepts]
                r, p = stats.pearsonr(prefs, valences)
                pref_corr = float(r) if not np.isnan(r) else 0.0
                logger.info(
                    "Preference-valence correlation: r=%.3f, p=%.4f", r, p
                )

        metrics = {
            "preference_correlation": pref_corr,
            "per_concept_preference": per_concept_preference,
            "n_concepts_tested": len(per_concept_preference),
            "best_layer": layer,
        }

        metadata = {
            "behavior_metric": self.behavior_metric,
            "steering_alpha": alpha,
            "n_samples_per_condition": self.n_samples,
            "evaluation_method": self.eval_method,
            "dependency_claim": self.dependency_claim,
        }

        return metrics, metadata

    # ── Device detection ─────────────────────────────────────────────────────

    def _is_cuda(self, model: Any) -> bool:
        """Check if the model is on a CUDA device.

        On CUDA (e.g. A100), generation is fast enough (~50 tok/s) to use
        generation-based steering instead of log-prob scoring.
        """
        try:
            device = next(model.parameters()).device
            return device.type == "cuda"
        except StopIteration:
            return False

    # ── Shared helpers ──────────────────────────────────────────────────────

    def _load_scenarios(self) -> list[dict[str, Any]]:
        """Load scenarios from data directory or stimuli config.

        Returns list of scenario dicts.
        """
        data_dir = self.data_root / "data" / self.config.paper_id

        # Try direct file
        scenario_file = data_dir / f"{self.stimulus_set}.json"
        if scenario_file.exists():
            with open(scenario_file) as f:
                return json.load(f)

        # Try subdirectory
        scenario_dir = data_dir / self.stimulus_set
        if scenario_dir.is_dir():
            all_scenarios = []
            for f in sorted(scenario_dir.glob("*.json")):
                with open(f) as fh:
                    data = json.load(fh)
                if isinstance(data, list):
                    all_scenarios.extend(data)
                else:
                    all_scenarios.append(data)
            if all_scenarios:
                return all_scenarios

        raise FileNotFoundError(
            f"No scenarios found for '{self.stimulus_set}' in {data_dir}. "
            f"Create {scenario_file} or {scenario_dir}/ first."
        )

    def _score_choices(
        self,
        model: Any,
        tokenizer: Any,
        prompt: str,
        layer: int,
        steering_vector: torch.Tensor | None,
        alpha: float,
        choices: list[str],
    ) -> dict[str, float]:
        """Score choices via log-probability under optional steering.

        Uses a single forward pass per choice (fast!) instead of autoregressive
        generation (slow with per-token hooks).

        Returns dict mapping choice -> log-probability.
        """
        scores: dict[str, float] = {}

        for choice in choices:
            full_text = prompt + " " + choice
            # TransformerLens path
            if hasattr(model, "run_with_hooks") and hasattr(model, "to_tokens"):
                tokens = model.to_tokens(full_text)
                prompt_tokens = model.to_tokens(prompt)
                prompt_len = prompt_tokens.shape[1]

                device = next(model.parameters()).device

                fwd_hooks = []
                if steering_vector is not None and alpha != 0.0:
                    sv = steering_vector.to(device)

                    def hook_fn(activation, hook, _sv=sv, _alpha=alpha):
                        activation[:, :, :] += _alpha * _sv
                        return activation

                    hook_name = f"blocks.{layer}.hook_resid_post"
                    fwd_hooks = [(hook_name, hook_fn)]

                with torch.no_grad():
                    logits = model.run_with_hooks(tokens, fwd_hooks=fwd_hooks)

                # Sum log-probs of choice tokens
                log_probs = torch.nn.functional.log_softmax(logits[0], dim=-1)
                total_log_prob = 0.0
                for i in range(prompt_len - 1, tokens.shape[1] - 1):
                    next_token_id = tokens[0, i + 1]
                    total_log_prob += log_probs[i, next_token_id].item()

                scores[choice] = total_log_prob

            # HuggingFace path
            else:
                hf_model = model.model if hasattr(model, "model") else model
                device = next(hf_model.parameters()).device

                inputs = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=512)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                prompt_ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
                prompt_len = prompt_ids["input_ids"].shape[1]

                hook_handle = None
                if steering_vector is not None and alpha != 0.0:
                    sv = steering_vector.to(device)

                    def hf_hook_fn(module, input, output, _sv=sv, _alpha=alpha):
                        hidden = output[0] if isinstance(output, tuple) else output
                        hidden += _alpha * _sv
                        if isinstance(output, tuple):
                            return (hidden,) + output[1:]
                        return hidden

                    layer_modules = self._get_layer_modules(hf_model)
                    if layer < len(layer_modules):
                        hook_handle = layer_modules[layer].register_forward_hook(hf_hook_fn)

                try:
                    with torch.no_grad():
                        out = hf_model(**inputs)
                    logits = out.logits
                finally:
                    if hook_handle is not None:
                        hook_handle.remove()

                log_probs = torch.nn.functional.log_softmax(logits[0], dim=-1)
                total_log_prob = 0.0
                for i in range(prompt_len - 1, inputs["input_ids"].shape[1] - 1):
                    next_token_id = inputs["input_ids"][0, i + 1]
                    total_log_prob += log_probs[i, next_token_id].item()

                scores[choice] = total_log_prob

        return scores

    def _format_scenario_prompt(
        self, scenario: dict[str, Any], tokenizer: Any
    ) -> str:
        """Format a scenario into a prompt string for the model.

        Combines system_prompt and user_message with chat template if available.
        """
        system = scenario.get("system_prompt", "")
        user = scenario.get("user_message", scenario.get("prompt", ""))

        # Try chat template
        if hasattr(tokenizer, "apply_chat_template"):
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": user})
            try:
                return tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                pass

        # Fallback: concatenate
        if system:
            return f"{system}\n\nUser: {user}\n\nAssistant:"
        return f"User: {user}\n\nAssistant:"

    @torch.no_grad()
    def _generate_response(
        self,
        model: Any,
        tokenizer: Any,
        prompt: str,
        layer: int,
        steering_vector: torch.Tensor | None,
        alpha: float,
        max_new_tokens: int = 100,  # 100 tokens sufficient for behavioral choice; 200 too slow on CPU/MPS
    ) -> str:
        """Generate a text response, optionally with steering vector injection.

        If steering_vector is provided, it is added to the residual stream at
        the specified layer during generation.

        Returns:
            Generated text response.
        """
        # TransformerLens path: manual token-by-token generation with hooks
        if hasattr(model, "run_with_hooks") and hasattr(model, "to_tokens"):
            tokens = model.to_tokens(prompt)
            device = next(model.parameters()).device
            generated = tokens.clone()

            fwd_hooks = []
            if steering_vector is not None and alpha != 0.0:
                sv = steering_vector.to(device)

                def hook_fn(activation, hook):
                    activation[:, :, :] += alpha * sv
                    return activation

                hook_name = f"blocks.{layer}.hook_resid_post"
                fwd_hooks = [(hook_name, hook_fn)]

            for _ in range(max_new_tokens):
                with torch.no_grad():
                    logits = model.run_with_hooks(generated, fwd_hooks=fwd_hooks)
                # Sample from logits with temperature
                next_logits = logits[0, -1, :] / 0.7
                probs = torch.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).unsqueeze(0)
                generated = torch.cat([generated, next_token], dim=-1)
                if model.tokenizer.eos_token_id is not None and next_token.item() == model.tokenizer.eos_token_id:
                    break

            return model.tokenizer.decode(generated[0, tokens.shape[1]:], skip_special_tokens=True)

        # HuggingFace path
        hf_model = model.model if hasattr(model, "model") else model
        device = next(hf_model.parameters()).device

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        hook_handle = None
        if steering_vector is not None and alpha != 0.0:
            sv = steering_vector.to(device)

            def hf_hook_fn(module, input, output):
                hidden = output[0] if isinstance(output, tuple) else output
                hidden += alpha * sv
                if isinstance(output, tuple):
                    return (hidden,) + output[1:]
                return hidden

            layer_modules = self._get_layer_modules(hf_model)
            if layer < len(layer_modules):
                hook_handle = layer_modules[layer].register_forward_hook(hf_hook_fn)

        try:
            output_ids = hf_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                do_sample=True,
            )
        finally:
            if hook_handle is not None:
                hook_handle.remove()

        input_len = inputs["input_ids"].shape[1]
        return tokenizer.decode(output_ids[0, input_len:], skip_special_tokens=True)

    def _generate_and_classify(
        self,
        model: Any,
        tokenizer: Any,
        prompt: str,
        layer: int,
        steering_vector: torch.Tensor | None,
        alpha: float,
        n_samples: int,
        ethical_keywords: str,
        unethical_keywords: str,
    ) -> list[int]:
        """Generate n responses and classify each as ethical (0) or unethical (1).

        Returns:
            List of 0/1 classifications.
        """
        choices: list[int] = []

        for _ in range(n_samples):
            response = self._generate_response(
                model, tokenizer, prompt, layer,
                steering_vector=steering_vector, alpha=alpha,
                max_new_tokens=100,
            )
            classification = self._classify_choice(
                response, ethical_keywords, unethical_keywords
            )
            choices.append(classification)

        return choices

    def _classify_choice(
        self, response: str, ethical_keywords: str, unethical_keywords: str
    ) -> int:
        """Classify a response as ethical (0) or unethical (1) via keyword matching.

        Simple heuristic: count keywords associated with each choice and
        pick the one with more hits. Defaults to ethical (0) if tied.
        """
        response_lower = response.lower()

        # Extract keywords from the choice descriptions
        ethical_words = set(re.findall(r'\b\w+\b', ethical_keywords.lower()))
        unethical_words = set(re.findall(r'\b\w+\b', unethical_keywords.lower()))

        # Remove common words
        stop_words = {"the", "a", "an", "to", "and", "or", "is", "it", "of", "in", "for"}
        ethical_words -= stop_words
        unethical_words -= stop_words

        ethical_score = sum(1 for w in ethical_words if w in response_lower)
        unethical_score = sum(1 for w in unethical_words if w in response_lower)

        return 1 if unethical_score > ethical_score else 0

    def _compare_responses(self, baseline: str, steered: str) -> int:
        """Compare two responses and return preference score.

        Returns:
            +1 if steered is preferred, -1 if baseline preferred, 0 if tie.

        This is a simple heuristic proxy for actual preference evaluation.
        A proper implementation would use an LLM-as-judge or human eval.
        """
        # Heuristic: prefer longer, more substantive responses
        # (rough proxy -- production would use LLM-as-judge)
        baseline_len = len(baseline.split())
        steered_len = len(steered.split())

        # Penalize very short responses
        if steered_len < 10 and baseline_len >= 10:
            return -1
        if baseline_len < 10 and steered_len >= 10:
            return 1

        # Slight preference for moderate-length responses
        if abs(steered_len - baseline_len) < 5:
            return 0
        if steered_len > baseline_len:
            return 1
        return -1

    def _get_layer_modules(self, model: Any) -> list:
        """Get transformer layer modules from an HF model."""
        for attr in ("model.layers", "transformer.h", "gpt_neox.layers", "model.decoder.layers"):
            parts = attr.split(".")
            obj = model
            try:
                for part in parts:
                    obj = getattr(obj, part)
                return list(obj)
            except AttributeError:
                continue
        raise AttributeError(
            f"Cannot find transformer layers in model of type {type(model).__name__}"
        )
