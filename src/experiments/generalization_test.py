"""Generalization test experiment: do probes transfer to held-out stimuli?

Tests whether probes trained on one stimulus distribution (e.g., explicit
emotion stories) generalize to a structurally different distribution (e.g.,
implicit scenarios where the concept is never named).

Expected params in ClaimConfig.params:
    training_stimulus_set: str   -- name of training set (for reference)
    test_stimulus_set: str       -- name of held-out set (e.g., "implicit_scenarios")
    training_claim_id: str       -- claim_id that produced the trained probes
                                    (inferred from depends_on if not set)
    evaluation_method: str       -- "confusion_matrix" (default)

Outputs:
    metrics["diagonal_dominance"]     -- fraction of concepts where diagonal
                                         entry is row-maximum in confusion matrix
    metrics["test_accuracy"]          -- overall accuracy on held-out set
    metrics["per_concept_accuracy"]   -- dict[str, float] on test set
    metrics["confusion_matrix"]       -- (n_concepts, n_concepts) as nested list
    metrics["generalization_gap"]     -- training accuracy minus test accuracy

Depends on:
    A prior probe_classification run (uses its trained probes and best layer).

Usage:
    from src.experiments.generalization_test import GeneralizationTestExperiment
    exp = GeneralizationTestExperiment(claim_config, "llama_1b", data_root)
    result = exp.load_or_run(model, tokenizer, activations_cache)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix as sk_confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder

from src.core.claim import ClaimConfig, ExperimentResult
from src.core.experiment import Experiment
from src.utils.metrics import diagonal_dominance

logger = logging.getLogger(__name__)


class GeneralizationTestExperiment(Experiment):
    """Test whether probes from a prior experiment generalize to held-out stimuli.

    This experiment loads trained probes from a dependency (typically
    probe_classification), extracts activations for a new stimulus set, and
    evaluates the probes on it. The key metric is diagonal dominance in the
    confusion matrix: for each concept, is the probe's highest-confidence
    prediction the correct concept?
    """

    def __init__(self, config: ClaimConfig, model_key: str, data_root: Path) -> None:
        super().__init__(config, model_key, data_root)
        self.test_stimulus_set: str = config.params.get(
            "test_stimulus_set", "implicit_scenarios"
        )
        self.training_claim_id: str = config.params.get(
            "training_claim_id", config.depends_on or ""
        )
        if not self.training_claim_id:
            raise ValueError(
                "GeneralizationTestExperiment requires 'training_claim_id' in params "
                "or a 'depends_on' claim_id to locate trained probes."
            )

    def run(
        self, model: Any, tokenizer: Any, activations_cache: Any
    ) -> ExperimentResult:
        """Run generalization test on held-out stimuli.

        Args:
            model: Loaded language model.
            tokenizer: The model's tokenizer.
            activations_cache: Shared activation cache, or None.

        Returns:
            ExperimentResult with diagonal_dominance and generalization metrics.
        """
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Step 1: Load training experiment results to find best layer and probes
        training_results_dir = (
            self.data_root
            / "results"
            / self.config.paper_id
            / self.model_key
            / self.training_claim_id
        )
        training_result_path = training_results_dir / "result.json"

        if not training_result_path.exists():
            raise FileNotFoundError(
                f"Training experiment results not found at {training_result_path}. "
                f"Run claim '{self.training_claim_id}' first."
            )

        with open(training_result_path) as f:
            training_result = json.load(f)

        best_layer = int(training_result["metrics"]["best_layer"])
        concept_set = list(training_result["metrics"]["per_concept_accuracy"].keys())
        training_accuracy = float(training_result["metrics"]["probe_accuracy"])
        n_folds = int(training_result["metadata"].get("n_folds", 5))

        logger.info(
            "Using probes from '%s', best layer %d (train acc %.4f)",
            self.training_claim_id, best_layer, training_accuracy,
        )

        # Step 2: Load trained probes for best layer
        probes, label_encoder = self._load_probes(
            training_results_dir / "probes", best_layer, n_folds
        )
        logger.info("Loaded %d probe folds for layer %d", len(probes), best_layer)

        # Step 3: Load held-out test stimuli
        test_texts, test_labels = self._load_test_stimuli(concept_set)
        logger.info(
            "Loaded %d test stimuli across %d concepts",
            len(test_texts), len(set(test_labels)),
        )

        # Step 4: Extract activations for test stimuli at best layer
        test_activations = self._extract_test_activations(
            model, tokenizer, test_texts, best_layer, activations_cache
        )

        # Step 5: Evaluate probes on test set
        y_true = label_encoder.transform(test_labels)
        test_acc, per_concept, cm, dd = self._evaluate_on_test(
            probes, test_activations, y_true, label_encoder
        )

        logger.info(
            "Test accuracy: %.4f, Diagonal dominance: %.4f, Gap: %.4f",
            test_acc, dd, training_accuracy - test_acc,
        )

        metrics = {
            "diagonal_dominance": dd,
            "test_accuracy": test_acc,
            "per_concept_accuracy": per_concept,
            "confusion_matrix": cm.tolist(),
            "generalization_gap": training_accuracy - test_acc,
            "training_accuracy": training_accuracy,
            "best_layer": best_layer,
            "n_test_stimuli": len(test_texts),
        }

        metadata = {
            "training_claim_id": self.training_claim_id,
            "test_stimulus_set": self.test_stimulus_set,
            "n_probe_folds": len(probes),
        }

        return ExperimentResult(
            claim_id=self.config.claim_id,
            model_key=self.model_key,
            paper_id=self.config.paper_id,
            metrics=metrics,
            metadata=metadata,
        )

    def evaluate(self, result: ExperimentResult) -> bool:
        """Check if diagonal dominance meets the success threshold."""
        metric_val = result.metrics.get(self.config.success_metric, 0.0)
        return float(metric_val) >= self.config.success_threshold

    # ── Private helpers ─────────────────────────────────────────────────────

    def _load_probes(
        self, probes_dir: Path, layer: int, n_folds: int
    ) -> tuple[list[Any], LabelEncoder]:
        """Load trained probes for a specific layer across all CV folds.

        Returns:
            (list_of_probes, label_encoder)
        """
        probes: list[Any] = []
        label_encoder = LabelEncoder()

        for fold_idx in range(n_folds):
            probe_path = probes_dir / f"layer_{layer}_fold_{fold_idx}.pt"
            if not probe_path.exists():
                logger.warning("Missing probe: %s", probe_path)
                continue

            probe_data = torch.load(probe_path, map_location="cpu", weights_only=False)
            classes = probe_data["classes"]
            label_encoder.classes_ = np.array(classes)

            if probe_data.get("probe_type", "logistic_regression") == "logistic_regression":
                probe = LogisticRegression(max_iter=1)  # placeholder
                probe.classes_ = np.arange(len(classes))
                probe.coef_ = probe_data["coef"].numpy()
                probe.intercept_ = probe_data["intercept"].numpy()
                probes.append(probe)
            else:
                # MLP: the full sklearn model was saved
                probes.append(probe_data["model_state"])

        if not probes:
            raise FileNotFoundError(
                f"No probes found in {probes_dir} for layer {layer}"
            )

        return probes, label_encoder

    def _load_test_stimuli(
        self, concept_set: list[str]
    ) -> tuple[list[str], list[str]]:
        """Load held-out test stimuli from the test stimulus set.

        Looks for:
          1. data/{paper_id}/{test_stimulus_set}.json (combined file)
          2. data/{paper_id}/{test_stimulus_set}/{concept}.json (per-concept)

        Returns:
            (texts, labels)
        """
        from src.utils.datasets import resolve_stimulus_dir
        data_dir = resolve_stimulus_dir(self.config.paper_id, self.data_root)
        texts: list[str] = []
        labels: list[str] = []

        # Try combined file first
        combined_path = data_dir / f"{self.test_stimulus_set}.json"
        if combined_path.exists():
            with open(combined_path) as f:
                items = json.load(f)
            for item in items:
                concept = item.get("concept", item.get("label", ""))
                if concept in concept_set:
                    texts.append(item["text"])
                    labels.append(concept)
            return texts, labels

        # Try per-concept files in a subdirectory
        set_dir = data_dir / self.test_stimulus_set
        if set_dir.is_dir():
            for concept in concept_set:
                concept_file = set_dir / f"{concept}.json"
                if concept_file.exists():
                    with open(concept_file) as f:
                        items = json.load(f)
                    for item in items:
                        texts.append(item["text"])
                        labels.append(concept)

        if not texts:
            raise FileNotFoundError(
                f"No test stimuli found for set '{self.test_stimulus_set}' "
                f"in {data_dir}. Create {combined_path} or {set_dir}/ first."
            )

        return texts, labels

    def _extract_test_activations(
        self,
        model: Any,
        tokenizer: Any,
        texts: list[str],
        layer: int,
        activations_cache: Any,
    ) -> np.ndarray:
        """Extract activations for held-out test stimuli at a single layer.

        Aggregation is read from the training experiment's saved metadata so
        the test extraction matches what the probes were trained on.
        """
        from src.utils.extraction import extract_for_experiment

        training_results_dir = (
            self.data_root / "results" / self.config.paper_id
            / self.model_key / self.training_claim_id
        )
        training_result_path = training_results_dir / "result.json"
        aggregation = "last_token"
        if training_result_path.exists():
            with open(training_result_path) as f:
                tr = json.load(f)
            aggregation = tr.get("metadata", {}).get("aggregation", "last_token")

        acts_by_layer = extract_for_experiment(
            model=model,
            tokenizer=tokenizer,
            texts=texts,
            layers=[layer],
            aggregation=aggregation,
            cache_dir=self.results_dir / "activations",
            activations_cache=activations_cache,
        )
        return acts_by_layer[layer]

    def _evaluate_on_test(
        self,
        probes: list[Any],
        X_test: np.ndarray,
        y_true: np.ndarray,
        label_encoder: LabelEncoder,
    ) -> tuple[float, dict[str, float], np.ndarray, float]:
        """Evaluate ensemble of probes on test data.

        Uses majority vote across CV folds for predictions.

        Returns:
            (accuracy, per_concept_accuracy, confusion_matrix, diagonal_dominance)
        """
        # Collect predictions from each fold
        all_fold_preds = []
        for probe in probes:
            preds = probe.predict(X_test)
            all_fold_preds.append(preds)

        # Majority vote across folds
        fold_preds_arr = np.stack(all_fold_preds)  # (n_folds, n_test)
        y_pred = np.zeros(len(y_true), dtype=int)
        for i in range(len(y_true)):
            votes = fold_preds_arr[:, i]
            y_pred[i] = int(np.bincount(votes.astype(int)).argmax())

        # Overall accuracy
        test_acc = float(accuracy_score(y_true, y_pred))

        # Per-concept accuracy
        per_concept: dict[str, float] = {}
        for class_idx, concept in enumerate(label_encoder.classes_):
            mask = y_true == class_idx
            if mask.sum() > 0:
                per_concept[concept] = float(accuracy_score(y_true[mask], y_pred[mask]))

        # Confusion matrix
        cm = sk_confusion_matrix(y_true, y_pred)

        # Diagonal dominance
        dd = diagonal_dominance(cm.astype(float))

        return test_acc, per_concept, cm, dd
