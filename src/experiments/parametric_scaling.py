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
        aggregation = dep_result.get("metadata", {}).get("aggregation", "last_token")

        concept_vectors = concept_vectors_all[best_layer]
        logger.info("Using concept vectors from layer %d", best_layer)

        # Step 2: Load parametric templates
        templates = self._load_templates()
        logger.info("Loaded %d parametric templates", len(templates))

        # Step 3: For each template, generate stimuli and extract activations
        per_template_correlations: dict[str, dict[str, float]] = {}
        all_correlations: list[float] = []
        significant_pairs = 0
        total_pairs = 0

        for template_info in templates:
            template_id = template_info["id"]
            template_str = template_info["template"]
            variable = template_info["variable"]
            values = template_info["values"]

            # Check for cached results
            template_result_path = self.results_dir / f"template_{template_id}.json"
            if template_result_path.exists():
                with open(template_result_path) as f:
                    cached = json.load(f)
                per_template_correlations[template_id] = cached["correlations"]
                for concept, rho in cached["correlations"].items():
                    all_correlations.append(abs(rho))
                    total_pairs += 1
                    if abs(rho) >= 0.5:
                        significant_pairs += 1
                continue

            # Generate stimuli from template
            texts = [template_str.replace(f"{{{variable}}}", str(v)) for v in values]

            # Extract activations
            activations = self._extract_activations_single_layer(
                model, tokenizer, texts, best_layer, aggregation
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

                # Adjust sign based on expected direction
                expected = self.expected_directions.get(concept, "increasing")
                if expected == "decreasing":
                    # For "decreasing", a negative correlation is correct
                    rho_adjusted = -float(rho) if not np.isnan(rho) else 0.0
                else:
                    rho_adjusted = float(rho) if not np.isnan(rho) else 0.0

                template_corrs[concept] = rho_adjusted
                all_correlations.append(abs(float(rho)) if not np.isnan(rho) else 0.0)
                total_pairs += 1
                if abs(float(rho)) >= 0.5 and not np.isnan(rho):
                    significant_pairs += 1

                logger.info(
                    "  %s x %s: rho=%.3f (expected=%s, adjusted=%.3f)",
                    template_id, concept, float(rho), expected, rho_adjusted,
                )

            per_template_correlations[template_id] = template_corrs

            # Save per-template result (resume-safe)
            template_result = {"correlations": template_corrs, "values": values}
            tmp = template_result_path.with_suffix(".tmp")
            with open(tmp, "w") as f:
                json.dump(template_result, f, indent=2)
            tmp.rename(template_result_path)

        # Step 4: Aggregate
        mean_abs_corr = float(np.mean(all_correlations)) if all_correlations else 0.0

        metrics = {
            "rank_correlation": mean_abs_corr,
            "per_template_correlations": per_template_correlations,
            "significant_pairs": significant_pairs,
            "total_pairs_tested": total_pairs,
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
        data_dir = self.data_root / "data" / self.config.paper_id

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
            stim_config = load_stimuli_config(self.config.paper_id)
            stim_sets = stim_config.get("stimulus_sets", {})
            if self.stimulus_set in stim_sets:
                return stim_sets[self.stimulus_set].get("templates", [])
        except FileNotFoundError:
            pass

        raise FileNotFoundError(
            f"No parametric templates found for stimulus set '{self.stimulus_set}'. "
            f"Expected at {templates_file} or in stimuli_config.yaml."
        )

    @torch.no_grad()
    def _extract_activations_single_layer(
        self,
        model: Any,
        tokenizer: Any,
        texts: list[str],
        layer: int,
        aggregation: str = "last_token",
    ) -> np.ndarray:
        """Extract activations for a list of texts at a single layer.

        Returns:
            Array of shape (n_texts, hidden_dim).
        """
        from src.experiments.probe_classification import ProbeClassificationExperiment

        # Create a minimal instance for extraction
        temp = ProbeClassificationExperiment.__new__(ProbeClassificationExperiment)
        temp.config = self.config
        temp.model_key = self.model_key
        temp.data_root = self.data_root
        temp.results_dir = self.results_dir
        temp.aggregation = aggregation
        temp.concept_set = []
        temp.n_stimuli = 0
        temp.probe_type = "logistic_regression"
        temp.layers = [layer]
        temp.n_folds = 1
        temp.train_split = 0.8
        temp.seed = 42

        acts_by_layer = temp._extract_activations(
            model, tokenizer, texts, [layer], None
        )
        return acts_by_layer[layer]
