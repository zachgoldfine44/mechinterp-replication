"""Probe classification experiment: train linear probes on residual stream activations.

Tests whether a linear probe can classify which concept (e.g., emotion) is
expressed in a stimulus, using residual stream activations at one or more layers.

Expected params in ClaimConfig.params:
    concept_set: list[str]          -- concepts to classify
    n_stimuli_per_concept: int      -- stimuli per concept
    probe_type: str                 -- "logistic_regression" or "mlp"
    layers: list[int] | None        -- specific layers, or None to scan all
    aggregation: str                -- "last_token", "first_token", "mean",
                                        "max", or "last_k:N" (see
                                        src.utils.aggregation)
    cross_validation_folds: int     -- CV folds (default 5)
    train_test_split: float         -- train fraction (default 0.8)

Outputs:
    metrics["probe_accuracy"]       -- best layer's cross-validated accuracy
    metrics["best_layer"]           -- which layer achieved best accuracy
    metrics["per_layer_accuracy"]   -- dict[int, float] for every scanned layer
    metrics["per_concept_accuracy"] -- dict[str, float] at best layer
    metrics["confusion_matrix"]     -- (n_concepts, n_concepts) as nested list

Saves:
    results_dir/probes/layer_{L}_fold_{F}.pt  -- trained probe weights
    results_dir/concept_vectors.pt            -- mean activation per concept (contrastive)
    results_dir/result.json                   -- final metrics

Usage:
    from src.experiments.probe_classification import ProbeClassificationExperiment
    exp = ProbeClassificationExperiment(claim_config, "llama_1b", data_root)
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
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder

from src.core.claim import ClaimConfig, ExperimentResult
from src.core.experiment import Experiment
from src.utils.extraction import extract_for_experiment

logger = logging.getLogger(__name__)


class ProbeClassificationExperiment(Experiment):
    """Train linear probes to classify concepts from residual stream activations.

    This is the most common first experiment in a mechinterp replication:
    can we decode what concept the model is representing?

    The experiment:
      1. Loads stimuli for each concept from data/{paper_id}/training/
      2. Extracts residual stream activations (from cache or compute)
      3. Trains probes at each specified layer (or scans all layers)
      4. Reports best layer's accuracy and per-concept breakdown
      5. Saves concept vectors (mean activation per concept) for downstream use
    """

    def __init__(self, config: ClaimConfig, model_key: str, data_root: Path) -> None:
        super().__init__(config, model_key, data_root)
        self.concept_set: list[str] = config.params["concept_set"]
        self.n_stimuli: int = config.params.get("n_stimuli_per_concept", 50)
        self.probe_type: str = config.params.get("probe_type", "logistic_regression")
        self.layers: list[int] | None = config.params.get("layers", None)
        self.aggregation: str = config.params.get("aggregation", "last_token")
        self.n_folds: int = config.params.get("cross_validation_folds", 5)
        self.train_split: float = config.params.get("train_test_split", 0.8)
        self.seed: int = config.params.get("seed", 42)

    def run(
        self, model: Any, tokenizer: Any, activations_cache: Any
    ) -> ExperimentResult:
        """Execute probe classification experiment.

        Args:
            model: Loaded language model (TransformerLens HookedTransformer,
                nnsight LanguageModel, or HuggingFace model).
            tokenizer: The model's tokenizer.
            activations_cache: Shared ActivationsCache instance, or None.

        Returns:
            ExperimentResult with probe_accuracy and per-concept metrics.
        """
        self.results_dir.mkdir(parents=True, exist_ok=True)
        probes_dir = self.results_dir / "probes"
        probes_dir.mkdir(exist_ok=True)

        # Step 1: Load stimuli
        stimuli, labels = self._load_stimuli()
        logger.info(
            "Loaded %d stimuli across %d concepts", len(stimuli), len(self.concept_set)
        )

        # Step 2: Determine layers to scan
        # For models with many layers, scanning every layer is CPU-expensive
        # (sklearn probe training is O(n_layers * n_folds * n_features^2)).
        # Default: scan every 4th layer + first + last for models with >16 layers.
        # This gives good coverage of the layer sweep while being ~4x faster.
        # Set layers explicitly in config to override.
        n_layers = self._get_n_layers(model)
        if self.layers is not None:
            layers_to_scan = self.layers
        elif n_layers > 16:
            layers_to_scan = sorted(set(
                [0] + list(range(0, n_layers, max(1, n_layers // 8))) + [n_layers - 1]
            ))
            logger.info(
                "Auto-selecting %d/%d layers for probe sweep (every ~%d layers)",
                len(layers_to_scan), n_layers, max(1, n_layers // 8),
            )
        else:
            layers_to_scan = list(range(n_layers))
        logger.info("Scanning %d layers: %s", len(layers_to_scan), layers_to_scan)

        # Step 3: Extract activations for all stimuli at all layers
        activations_by_layer = self._extract_activations(
            model, tokenizer, stimuli, layers_to_scan, activations_cache
        )

        # Step 4: Compute concept vectors (mean activation per concept per layer)
        concept_vectors = self._compute_concept_vectors(activations_by_layer, labels, layers_to_scan)
        cv_path = self.results_dir / "concept_vectors.pt"
        torch.save(concept_vectors, cv_path.with_suffix(".tmp"))
        cv_path.with_suffix(".tmp").rename(cv_path)
        logger.info("Saved concept vectors to %s", cv_path)

        # Step 5: Train and evaluate probes at each layer
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(labels)

        per_layer_accuracy: dict[int, float] = {}
        best_layer = -1
        best_accuracy = -1.0
        best_per_concept: dict[str, float] = {}
        best_cm: np.ndarray | None = None

        for layer in layers_to_scan:
            X = activations_by_layer[layer]  # (n_stimuli, hidden_dim)

            # Check for cached probe results at this layer
            layer_result_path = probes_dir / f"layer_{layer}_result.json"
            if layer_result_path.exists():
                with open(layer_result_path) as f:
                    cached = json.load(f)
                layer_acc = cached["accuracy"]
                logger.info("Layer %d: cached accuracy = %.4f", layer, layer_acc)
                per_layer_accuracy[layer] = layer_acc
                if layer_acc > best_accuracy:
                    best_accuracy = layer_acc
                    best_layer = layer
                    best_per_concept = cached.get("per_concept", {})
                    best_cm = np.array(cached.get("confusion_matrix", []))
                continue

            # Cross-validated probe training
            layer_acc, per_concept, cm = self._train_and_eval_probe(
                X, y_encoded, label_encoder, layer, probes_dir
            )
            per_layer_accuracy[layer] = layer_acc

            # Save per-layer result for resume
            layer_result = {
                "accuracy": layer_acc,
                "per_concept": per_concept,
                "confusion_matrix": cm.tolist(),
            }
            tmp = layer_result_path.with_suffix(".tmp")
            with open(tmp, "w") as f:
                json.dump(layer_result, f)
            tmp.rename(layer_result_path)

            logger.info("Layer %d: accuracy = %.4f", layer, layer_acc)

            if layer_acc > best_accuracy:
                best_accuracy = layer_acc
                best_layer = layer
                best_per_concept = per_concept
                best_cm = cm

        # Step 6: Build result
        metrics = {
            "probe_accuracy": best_accuracy,
            "best_layer": best_layer,
            "per_layer_accuracy": {str(k): v for k, v in per_layer_accuracy.items()},
            "per_concept_accuracy": best_per_concept,
            "confusion_matrix": best_cm.tolist() if best_cm is not None else [],
            "n_concepts": len(self.concept_set),
            "n_stimuli_total": len(stimuli),
            "chance_level": 1.0 / len(self.concept_set),
        }

        metadata = {
            "probe_type": self.probe_type,
            "aggregation": self.aggregation,
            "n_folds": self.n_folds,
            "layers_scanned": layers_to_scan,
            "seed": self.seed,
        }

        return ExperimentResult(
            claim_id=self.config.claim_id,
            model_key=self.model_key,
            paper_id=self.config.paper_id,
            metrics=metrics,
            metadata=metadata,
        )

    def evaluate(self, result: ExperimentResult) -> bool:
        """Check if probe accuracy meets the success threshold."""
        metric_val = result.metrics.get(self.config.success_metric, 0.0)
        return float(metric_val) >= self.config.success_threshold

    # ── Private helpers ─────────────────────────────────────────────────────

    def _load_stimuli(self) -> tuple[list[str], list[str]]:
        """Load stimuli texts and their concept labels from data directory.

        Looks for JSON files in data/{paper_id}/training/ with format:
        [{"concept": str, "text": str, ...}, ...]

        Returns:
            (texts, labels) where texts[i] has concept labels[i].
        """
        from src.utils.datasets import resolve_stimulus_dir
        stimulus_dir = resolve_stimulus_dir(self.config.paper_id, self.data_root)
        training_dir = stimulus_dir / "training"

        texts: list[str] = []
        labels: list[str] = []

        per_concept_loaded: dict[str, int] = {}
        for concept in self.concept_set:
            concept_file = training_dir / f"{concept}.json"
            if concept_file.exists():
                with open(concept_file) as f:
                    items = json.load(f)
                taken = items[: self.n_stimuli]
                per_concept_loaded[concept] = len(taken)
                if len(items) < self.n_stimuli:
                    logger.warning(
                        "Concept '%s': requested %d stimuli, only %d available on disk. "
                        "Using all %d. Update n_stimuli_per_concept in paper_config.yaml "
                        "to reflect what is actually loaded so reported counts match.",
                        concept, self.n_stimuli, len(items), len(items),
                    )
                for item in taken:
                    texts.append(item["text"])
                    labels.append(concept)
                continue

            # Try a single combined file
            combined_file = training_dir / "all_stimuli.json"
            if combined_file.exists():
                with open(combined_file) as f:
                    all_items = json.load(f)
                concept_items = [
                    it for it in all_items if it.get("concept") == concept
                ]
                taken = concept_items[: self.n_stimuli]
                per_concept_loaded[concept] = len(taken)
                if len(concept_items) < self.n_stimuli:
                    logger.warning(
                        "Concept '%s': requested %d stimuli, only %d available in "
                        "combined file. Using all %d.",
                        concept, self.n_stimuli, len(concept_items), len(concept_items),
                    )
                for item in taken:
                    texts.append(item["text"])
                    labels.append(concept)
                continue

            logger.warning(
                "No stimuli found for concept '%s' in %s", concept, training_dir
            )

        # Sanity check: warn if per-concept counts are unbalanced (probe will
        # be biased toward over-represented concepts otherwise).
        if per_concept_loaded:
            counts = list(per_concept_loaded.values())
            if max(counts) - min(counts) > 1:
                logger.warning(
                    "Per-concept stimulus counts are unbalanced: min=%d max=%d. "
                    "This can bias probe accuracy and confusion matrices.",
                    min(counts), max(counts),
                )

        if len(texts) == 0:
            raise FileNotFoundError(
                f"No stimuli found in {training_dir}. "
                f"Generate them first (see stimuli_config.yaml)."
            )

        return texts, labels

    def _get_n_layers(self, model: Any) -> int:
        """Determine number of layers from the model object."""
        # TransformerLens
        if hasattr(model, "cfg") and hasattr(model.cfg, "n_layers"):
            return model.cfg.n_layers
        # HuggingFace
        if hasattr(model, "config"):
            for attr in ("num_hidden_layers", "n_layer", "n_layers"):
                if hasattr(model.config, attr):
                    return getattr(model.config, attr)
        # nnsight
        if hasattr(model, "model") and hasattr(model.model, "config"):
            cfg = model.model.config
            for attr in ("num_hidden_layers", "n_layer", "n_layers"):
                if hasattr(cfg, attr):
                    return getattr(cfg, attr)
        logger.warning("Could not determine layer count; defaulting to 32")
        return 32

    def _extract_activations(
        self,
        model: Any,
        tokenizer: Any,
        stimuli: list[str],
        layers: list[int],
        activations_cache: Any,
    ) -> dict[int, np.ndarray]:
        """Extract residual stream activations for all stimuli at specified layers.

        Thin wrapper around ``src.utils.extraction.extract_for_experiment``.
        Kept as a method for backward compatibility with any caller that
        accesses the experiment's instance state; new callers should import
        ``extract_for_experiment`` directly.
        """
        return extract_for_experiment(
            model=model,
            tokenizer=tokenizer,
            texts=stimuli,
            layers=layers,
            aggregation=self.aggregation,
            cache_dir=self.results_dir / "activations",
            activations_cache=activations_cache,
        )

    def _train_and_eval_probe(
        self,
        X: np.ndarray,
        y: np.ndarray,
        label_encoder: LabelEncoder,
        layer: int,
        probes_dir: Path,
    ) -> tuple[float, dict[str, float], np.ndarray]:
        """Train a probe with cross-validation and return accuracy metrics.

        Returns:
            (accuracy, per_concept_accuracy, confusion_matrix)
        """
        np.random.seed(self.seed)
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.seed)

        fold_accuracies: list[float] = []
        all_preds: list[np.ndarray] = []
        all_true: list[np.ndarray] = []

        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            probe_path = probes_dir / f"layer_{layer}_fold_{fold_idx}.pt"

            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Train probe
            probe = self._make_probe()
            probe.fit(X_train, y_train)

            # Evaluate
            y_pred = probe.predict(X_test)
            fold_acc = accuracy_score(y_test, y_pred)
            fold_accuracies.append(fold_acc)

            all_preds.append(y_pred)
            all_true.append(y_test)

            # Save probe weights (for downstream experiments)
            probe_data = {
                "probe_type": self.probe_type,
                "layer": layer,
                "fold": fold_idx,
                "classes": label_encoder.classes_.tolist(),
            }
            if self.probe_type == "logistic_regression":
                probe_data["coef"] = torch.tensor(probe.coef_, dtype=torch.float32)
                probe_data["intercept"] = torch.tensor(probe.intercept_, dtype=torch.float32)
            else:
                # For MLP, save the full sklearn object state
                probe_data["model_state"] = probe

            tmp = probe_path.with_suffix(".tmp")
            torch.save(probe_data, tmp)
            tmp.rename(probe_path)

        mean_acc = float(np.mean(fold_accuracies))

        # Aggregate predictions for confusion matrix
        all_preds_arr = np.concatenate(all_preds)
        all_true_arr = np.concatenate(all_true)

        cm = sk_confusion_matrix(all_true_arr, all_preds_arr)

        # Per-concept accuracy
        per_concept: dict[str, float] = {}
        for class_idx, concept in enumerate(label_encoder.classes_):
            mask = all_true_arr == class_idx
            if mask.sum() > 0:
                per_concept[concept] = float(accuracy_score(all_true_arr[mask], all_preds_arr[mask]))

        return mean_acc, per_concept, cm

    def _make_probe(self) -> LogisticRegression | MLPClassifier:
        """Create a fresh probe instance based on config."""
        if self.probe_type == "logistic_regression":
            return LogisticRegression(
                max_iter=1000,
                solver="lbfgs",
                random_state=self.seed,
                C=1.0,
            )
        elif self.probe_type == "mlp":
            return MLPClassifier(
                hidden_layer_sizes=(256,),
                max_iter=500,
                random_state=self.seed,
                early_stopping=True,
                validation_fraction=0.1,
            )
        else:
            raise ValueError(f"Unknown probe type: {self.probe_type!r}")

    def _compute_concept_vectors(
        self,
        activations_by_layer: dict[int, np.ndarray],
        labels: list[str],
        layers: list[int],
    ) -> dict[int, dict[str, torch.Tensor]]:
        """Compute mean activation vector per concept per layer.

        These are the contrastive / CAA-style concept vectors used by
        downstream experiments (steering, geometry, parametric scaling).

        Returns:
            Dict[layer, Dict[concept, Tensor of shape (hidden_dim,)]]
        """
        labels_arr = np.array(labels)
        concept_vectors: dict[int, dict[str, torch.Tensor]] = {}

        # Compute global mean across all stimuli (for mean-subtraction)
        for layer in layers:
            X = activations_by_layer[layer]
            global_mean = X.mean(axis=0)
            concept_vectors[layer] = {}

            for concept in self.concept_set:
                mask = labels_arr == concept
                if mask.sum() > 0:
                    concept_mean = X[mask].mean(axis=0)
                    # Contrastive vector: concept mean minus global mean
                    vec = torch.tensor(concept_mean - global_mean, dtype=torch.float32)
                    concept_vectors[layer][concept] = vec

        return concept_vectors
