"""Parametric scaling experiment: do concept vectors respond to continuous parameters?

Tests whether concept probe activations scale monotonically with a continuous
real-world parameter (e.g., medication dosage, height, financial loss). This
validates that representations track meaningful gradations, not just categories.

Expected params in ClaimConfig.params:
    stimulus_set: str               -- name of parametric stimulus set
    target_emotions: list[str]      -- which concepts to test
    scaling_variable: str           -- name of the continuous variable
    expected_directions: dict       -- {concept: "increasing"|"decreasing"|"stable_high"}
    correlation_type: str           -- "spearman" (default) or "pearson"

Outputs:
    metrics["rank_correlation"]         -- mean |rho| across significant pairs
    metrics["per_template_correlations"]-- dict[template_id, dict[concept, rho]]
    metrics["significant_pairs"]        -- count of (template, concept) pairs with |rho|>threshold
    metrics["total_pairs_tested"]       -- total pairs tested

Depends on:
    A prior probe_classification run (uses concept_vectors.pt for projections).

Usage:
    from src.experiments.parametric_scaling import ParametricScalingExperiment
    exp = ParametricScalingExperiment(claim_config, "llama_1b", data_root)
    result = exp.load_or_run(model, tokenizer, activations_cache)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy import stats

from src.core.claim import ClaimConfig, ExperimentResult
from src.core.experiment import Experiment

logger = logging.getLogger(__name__)


class ParametricScalingExperiment(Experiment):
    """Test whether concept vectors respond proportionally to continuous parameters.

    Loads concept vectors from a prior experiment, generates parametric stimuli
    (e.g., "I took {X} mg of tylenol"), extracts activations for each parameter
    value, projects onto concept vectors, and computes rank correlations between
    parameter values and projections.
    """

    def __init__(self, config: ClaimConfig, model_key: str, data_root: Path) -> None:
        super().__init__(config, model_key, data_root)
        self.stimulus_set: str = config.params.get("stimulus_set", "parametric_dosage")
        self.target_concepts: list[str] = config.params.get(
            "target_emotions", config.params.get("concept_set", [])
        )
        self.scaling_variable: str = config.params.get("scaling_variable", "")
        self.expected_directions: dict[str, str] = config.params.get(
            "expected_directions", {}
        )
        self.correlation_type: str = config.params.get("correlation_type", "spearman")
        self.dependency_claim: str = config.depends_on or ""
        if not self.dependency_claim:
            raise ValueError(
                "ParametricScalingExperiment requires 'depends_on' to locate concept vectors."
            )

    def run(
        self, model: Any, tokenizer: Any, activations_cache: Any
    ) -> ExperimentResult:
        """Run parametric scaling analysis.

        Args:
            model: Loaded language model.
            tokenizer: The model's tokenizer.
            activations_cache: Shared activation cache, or None.

        Returns:
            ExperimentResult with rank_correlation and per-template metrics.
        """
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Step 1: Load concept vectors and best layer from dependency
        from src.core.experiment import _results_dir_for
        dep_results_dir = _results_dir_for(
            data_root=self.data_root,
            paper_id=self.config.paper_id,
            replication_id=self.config.replication_id,
            model_key=self.model_key,
            claim_id=self.dependency_claim,
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
        aggregation = dep_result.get("metadata", {}).get("aggregation", "last_token")

        concept_vectors = concept_vectors_all[best_layer]
        logger.info("Using concept vectors from layer %d", best_layer)

        # Step 2: Load parametric templates
        templates = self._load_templates()
        logger.info("Loaded %d parametric templates", len(templates))

        # Step 3: For each template, generate stimuli and extract activations.
        # Templates can be one of two types:
        #   - "real": expected to produce a positive scaling effect
        #     (e.g., tylenol_dosage, height_ledge)
        #   - "negative_control": expected to produce a NEAR-ZERO effect
        #     (e.g., blueberries_control). If a negative control gives a
        #     large effect, the parametric finding is contaminated by
        #     numerical-magnitude artifacts.
        # We aggregate them separately so the headline metric isn't inflated
        # by negative-control hits, and we report a contamination signal.
        per_template_correlations: dict[str, dict[str, float]] = {}
        per_template_is_neg_control: dict[str, bool] = {}
        real_correlations: list[float] = []     # |rho| for real templates only
        neg_control_correlations: list[float] = []  # |rho| for negative controls
        significant_pairs = 0  # real templates only
        total_pairs = 0        # real templates only

        for template_info in templates:
            template_id = template_info["id"]
            template_str = template_info["template"]
            variable = template_info["variable"]
            values = template_info["values"]
            is_neg_control = bool(template_info.get("is_negative_control", False))
            per_template_is_neg_control[template_id] = is_neg_control
            # Per-template expected_response from the stimuli config (if present)
            # overrides the global expected_directions on a per-concept basis.
            template_expected = template_info.get("expected_response", {}) or {}

            # Check for cached results
            template_result_path = self.results_dir / f"template_{template_id}.json"
            if template_result_path.exists():
                with open(template_result_path) as f:
                    cached = json.load(f)
                per_template_correlations[template_id] = cached["correlations"]
                for concept, rho in cached["correlations"].items():
                    abs_rho = abs(rho)
                    if is_neg_control:
                        neg_control_correlations.append(abs_rho)
                    else:
                        real_correlations.append(abs_rho)
                        total_pairs += 1
                        if abs_rho >= 0.5:
                            significant_pairs += 1
                continue

            # Generate stimuli from template
            texts = [template_str.replace(f"{{{variable}}}", str(v)) for v in values]

            # Extract activations (thread the shared in-memory cache through
            # so the same parametric prompt is never re-extracted)
            activations = self._extract_activations_single_layer(
                model, tokenizer, texts, best_layer, aggregation,
                activations_cache=activations_cache,
            )  # (n_values, hidden_dim)

            # Project onto concept vectors and compute correlations
            template_corrs: dict[str, float] = {}
            for concept in self.target_concepts:
                if concept not in concept_vectors:
                    logger.warning("Concept '%s' not in concept vectors; skipping", concept)
                    continue

                vec = concept_vectors[concept].numpy()
                vec_norm = vec / (np.linalg.norm(vec) + 1e-8)

                # Project each stimulus activation onto concept vector
                projections = activations @ vec_norm  # (n_values,)

                # Compute correlation
                param_values = np.array(values, dtype=float)
                if self.correlation_type == "spearman":
                    rho, p_val = stats.spearmanr(param_values, projections)
                else:
                    rho, p_val = stats.pearsonr(param_values, projections)

                # Resolve expected direction. Per-template config wins over
                # the global. "stable" / "stable_high" → expectation of
                # near-zero correlation; we record the raw rho but do NOT
                # sign-adjust (sign-adjusting a stable target would just
                # measure the same magnitude either way).
                expected = template_expected.get(
                    concept, self.expected_directions.get(concept, "increasing")
                )
                if expected == "decreasing":
                    rho_adjusted = -float(rho) if not np.isnan(rho) else 0.0
                elif expected in ("stable", "stable_high"):
                    rho_adjusted = float(rho) if not np.isnan(rho) else 0.0
                else:  # "increasing" or anything else
                    rho_adjusted = float(rho) if not np.isnan(rho) else 0.0

                template_corrs[concept] = rho_adjusted
                abs_rho = abs(float(rho)) if not np.isnan(rho) else 0.0
                if is_neg_control:
                    neg_control_correlations.append(abs_rho)
                else:
                    real_correlations.append(abs_rho)
                    total_pairs += 1
                    if abs_rho >= 0.5:
                        significant_pairs += 1

                logger.info(
                    "  %s x %s: rho=%.3f (expected=%s, adjusted=%.3f)%s",
                    template_id, concept, float(rho), expected, rho_adjusted,
                    " [NEG_CONTROL]" if is_neg_control else "",
                )

            per_template_correlations[template_id] = template_corrs

            # Save per-template result (resume-safe)
            template_result = {
                "correlations": template_corrs,
                "values": values,
                "is_negative_control": is_neg_control,
            }
            tmp = template_result_path.with_suffix(".tmp")
            with open(tmp, "w") as f:
                json.dump(template_result, f, indent=2)
            tmp.rename(template_result_path)

        # Step 4: Aggregate. Headline metric uses REAL templates only;
        # negative-control templates are aggregated separately so they don't
        # inflate the headline. The negative_control_mean is reported as a
        # contamination signal: if it is comparable to rank_correlation, the
        # parametric finding is partly an artifact of numerical magnitude in
        # the prompt rather than a genuine emotion-from-severity effect
        # (critique #2-4).
        rank_correlation = float(np.mean(real_correlations)) if real_correlations else 0.0
        neg_control_mean = (
            float(np.mean(neg_control_correlations)) if neg_control_correlations else None
        )
        # Contamination ratio: how much of the headline effect could be
        # explained by numerical magnitude alone. 1.0 means the negative
        # control is just as strong as the real templates (full contamination);
        # 0.0 means the negative control is silent.
        contamination_ratio = (
            (neg_control_mean / rank_correlation)
            if (neg_control_mean is not None and rank_correlation > 0)
            else None
        )

        metrics = {
            "rank_correlation": rank_correlation,
            "per_template_correlations": per_template_correlations,
            "per_template_is_negative_control": per_template_is_neg_control,
            "significant_pairs": significant_pairs,
            "total_pairs_tested": total_pairs,
            "negative_control_mean_abs_rho": neg_control_mean,
            "negative_control_contamination_ratio": contamination_ratio,
            "n_negative_control_pairs": len(neg_control_correlations),
            "best_layer": best_layer,
        }

        metadata = {
            "stimulus_set": self.stimulus_set,
            "target_concepts": self.target_concepts,
            "correlation_type": self.correlation_type,
            "expected_directions": self.expected_directions,
            "dependency_claim": self.dependency_claim,
        }

        return ExperimentResult(
            claim_id=self.config.claim_id,
            model_key=self.model_key,
            paper_id=self.config.paper_id,
            metrics=metrics,
            metadata=metadata,
        )

    def evaluate(self, result: ExperimentResult) -> bool:
        """Check if rank correlation meets the success threshold."""
        metric_val = result.metrics.get(self.config.success_metric, 0.0)
        return float(metric_val) >= self.config.success_threshold

    # ── Private helpers ─────────────────────────────────────────────────────

    def _load_templates(self) -> list[dict[str, Any]]:
        """Load parametric templates from stimuli config or data directory.

        Looks for:
          1. data/{paper_id}/parametric/{stimulus_set}.json
          2. Falls back to stimuli_config.yaml definitions

        Returns:
            List of template dicts with keys: id, template, variable, values.
        """
        from src.utils.datasets import resolve_stimulus_dir
        data_dir = resolve_stimulus_dir(
            self.config.paper_id, self.data_root,
            replication_id=self.config.replication_id,
        )

        # Try JSON file in data directory
        templates_file = data_dir / "parametric" / f"{self.stimulus_set}.json"
        if templates_file.exists():
            with open(templates_file) as f:
                return json.load(f)

        # Try a combined parametric file
        combined_file = data_dir / f"{self.stimulus_set}.json"
        if combined_file.exists():
            with open(combined_file) as f:
                data = json.load(f)
            if isinstance(data, list):
                return data
            if "templates" in data:
                return data["templates"]

        # Fall back to stimuli_config.yaml
        from src.core.config_loader import load_stimuli_config
        try:
            stim_config = load_stimuli_config(
                self.config.paper_id,
                replication_id=self.config.replication_id,
            )
            stim_sets = stim_config.get("stimulus_sets", {})
            if self.stimulus_set in stim_sets:
                return stim_sets[self.stimulus_set].get("templates", [])
        except FileNotFoundError:
            pass

        raise FileNotFoundError(
            f"No parametric templates found for stimulus set '{self.stimulus_set}'. "
            f"Expected at {templates_file} or in stimuli_config.yaml."
        )

    def _extract_activations_single_layer(
        self,
        model: Any,
        tokenizer: Any,
        texts: list[str],
        layer: int,
        aggregation: str = "last_token",
        activations_cache: Any | None = None,
    ) -> np.ndarray:
        """Extract activations for a list of texts at a single layer.

        Thin wrapper around ``extract_for_experiment`` for one layer. The
        per-template results are cached at the experiment level (one file
        per parametric template) so disk caching is unnecessary here, but
        the in-memory ``activations_cache`` is still threaded through so a
        prompt that has already been extracted by another experiment on the
        same model is reused.
        """
        from src.utils.extraction import extract_for_experiment

        acts_by_layer = extract_for_experiment(
            model=model,
            tokenizer=tokenizer,
            texts=texts,
            layers=[layer],
            aggregation=aggregation,
            cache_dir=None,
            activations_cache=activations_cache,
        )
        return acts_by_layer[layer]
