"""Tests for experiment classes with mocked models (no real model downloads).

Tests the experiment registry, evaluate() methods with known metrics,
and ExperimentResult save/load round-tripping.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.core.claim import ClaimConfig, ExperimentResult


# ---------------------------------------------------------------------------
# Experiment registry
# ---------------------------------------------------------------------------

class TestExperimentRegistry:
    """Verify the experiment registry maps all expected types."""

    def test_experiment_registry_complete(self) -> None:
        """All 5 generic experiment types should be registered."""
        from src.experiments import EXPERIMENT_REGISTRY

        expected_types = {
            "probe_classification",
            "generalization_test",
            "representation_geometry",
            "parametric_scaling",
            "causal_steering",
        }
        assert set(EXPERIMENT_REGISTRY.keys()) == expected_types

    def test_get_experiment_class_valid(self) -> None:
        """get_experiment_class should return a class for each known type."""
        from src.experiments import get_experiment_class

        for exp_type in [
            "probe_classification",
            "generalization_test",
            "causal_steering",
        ]:
            cls = get_experiment_class(exp_type)
            assert callable(cls)

    def test_get_experiment_class_invalid(self) -> None:
        """Unknown experiment_type should raise KeyError."""
        from src.experiments import get_experiment_class

        with pytest.raises(KeyError, match="Unknown experiment type"):
            get_experiment_class("nonexistent_experiment")

    def test_registry_classes_inherit_from_experiment(self) -> None:
        """Every registered class should be a subclass of Experiment."""
        from src.core.experiment import Experiment
        from src.experiments import EXPERIMENT_REGISTRY

        for name, cls in EXPERIMENT_REGISTRY.items():
            assert issubclass(cls, Experiment), (
                f"{name!r} -> {cls.__name__} does not inherit from Experiment"
            )


# ---------------------------------------------------------------------------
# Helper: make a ClaimConfig for a given experiment type
# ---------------------------------------------------------------------------

def _make_claim(
    experiment_type: str,
    success_metric: str,
    threshold: float,
    depends_on: str | None = None,
    extra_params: dict | None = None,
) -> ClaimConfig:
    params: dict = extra_params or {}
    if experiment_type == "probe_classification":
        params.setdefault("concept_set", ["happy", "sad"])
    return ClaimConfig(
        paper_id="emotions",
        claim_id=f"test_{experiment_type}",
        description=f"Test claim for {experiment_type}",
        experiment_type=experiment_type,
        params=params,
        success_metric=success_metric,
        success_threshold=threshold,
        depends_on=depends_on,
    )


# ---------------------------------------------------------------------------
# ProbeClassificationExperiment.evaluate()
# ---------------------------------------------------------------------------

class TestProbeClassificationEvaluate:
    """Test evaluate() on ProbeClassificationExperiment."""

    def test_evaluate_passes(self, data_root: Path) -> None:
        """Result with accuracy above threshold should evaluate True."""
        from src.experiments.probe_classification import ProbeClassificationExperiment

        claim = _make_claim("probe_classification", "probe_accuracy", 0.50)
        exp = ProbeClassificationExperiment(claim, "llama_1b", data_root)

        result = ExperimentResult(
            claim_id=claim.claim_id,
            model_key="llama_1b",
            paper_id="emotions",
            metrics={"probe_accuracy": 0.73},
        )
        assert exp.evaluate(result) is True

    def test_evaluate_fails(self, data_root: Path) -> None:
        """Result with accuracy below threshold should evaluate False."""
        from src.experiments.probe_classification import ProbeClassificationExperiment

        claim = _make_claim("probe_classification", "probe_accuracy", 0.50)
        exp = ProbeClassificationExperiment(claim, "llama_1b", data_root)

        result = ExperimentResult(
            claim_id=claim.claim_id,
            model_key="llama_1b",
            paper_id="emotions",
            metrics={"probe_accuracy": 0.30},
        )
        assert exp.evaluate(result) is False


# ---------------------------------------------------------------------------
# GeneralizationTestExperiment.evaluate()
# ---------------------------------------------------------------------------

class TestGeneralizationEvaluate:
    """Test evaluate() on GeneralizationTestExperiment."""

    def test_evaluate_passes(self, data_root: Path) -> None:
        from src.experiments.generalization_test import GeneralizationTestExperiment

        claim = _make_claim(
            "generalization_test",
            "diagonal_dominance",
            0.50,
            depends_on="emotion_probe_classification",
        )
        exp = GeneralizationTestExperiment(claim, "llama_1b", data_root)

        result = ExperimentResult(
            claim_id=claim.claim_id,
            model_key="llama_1b",
            paper_id="emotions",
            metrics={"diagonal_dominance": 0.67},
        )
        assert exp.evaluate(result) is True

    def test_evaluate_fails(self, data_root: Path) -> None:
        from src.experiments.generalization_test import GeneralizationTestExperiment

        claim = _make_claim(
            "generalization_test",
            "diagonal_dominance",
            0.50,
            depends_on="emotion_probe_classification",
        )
        exp = GeneralizationTestExperiment(claim, "llama_1b", data_root)

        result = ExperimentResult(
            claim_id=claim.claim_id,
            model_key="llama_1b",
            paper_id="emotions",
            metrics={"diagonal_dominance": 0.33},
        )
        assert exp.evaluate(result) is False


# ---------------------------------------------------------------------------
# RepresentationGeometryExperiment.evaluate()
# ---------------------------------------------------------------------------

class TestRepresentationGeometryEvaluate:
    """Test evaluate() on RepresentationGeometryExperiment."""

    def test_evaluate_passes(self, data_root: Path) -> None:
        from src.experiments.representation_geometry import RepresentationGeometryExperiment

        claim = _make_claim(
            "representation_geometry",
            "valence_correlation",
            0.50,
            depends_on="emotion_probe_classification",
            extra_params={"valence_labels": {"happy": 0.8, "sad": -0.7, "afraid": -0.6}},
        )
        exp = RepresentationGeometryExperiment(claim, "llama_1b", data_root)

        result = ExperimentResult(
            claim_id=claim.claim_id,
            model_key="llama_1b",
            paper_id="emotions",
            metrics={"valence_correlation": 0.65},
        )
        assert exp.evaluate(result) is True

    def test_evaluate_fails(self, data_root: Path) -> None:
        from src.experiments.representation_geometry import RepresentationGeometryExperiment

        claim = _make_claim(
            "representation_geometry",
            "valence_correlation",
            0.50,
            depends_on="emotion_probe_classification",
            extra_params={"valence_labels": {"happy": 0.8, "sad": -0.7, "afraid": -0.6}},
        )
        exp = RepresentationGeometryExperiment(claim, "llama_1b", data_root)

        result = ExperimentResult(
            claim_id=claim.claim_id,
            model_key="llama_1b",
            paper_id="emotions",
            metrics={"valence_correlation": 0.30},
        )
        assert exp.evaluate(result) is False


# ---------------------------------------------------------------------------
# ParametricScalingExperiment.evaluate()
# ---------------------------------------------------------------------------

class TestParametricScalingEvaluate:
    """Test evaluate() on ParametricScalingExperiment."""

    def test_evaluate_passes(self, data_root: Path) -> None:
        from src.experiments.parametric_scaling import ParametricScalingExperiment

        claim = _make_claim(
            "parametric_scaling",
            "rank_correlation",
            0.50,
            depends_on="emotion_probe_classification",
            extra_params={"target_emotions": ["afraid", "calm"]},
        )
        exp = ParametricScalingExperiment(claim, "llama_1b", data_root)

        result = ExperimentResult(
            claim_id=claim.claim_id,
            model_key="llama_1b",
            paper_id="emotions",
            metrics={"rank_correlation": 0.72},
        )
        assert exp.evaluate(result) is True

    def test_evaluate_fails(self, data_root: Path) -> None:
        from src.experiments.parametric_scaling import ParametricScalingExperiment

        claim = _make_claim(
            "parametric_scaling",
            "rank_correlation",
            0.50,
            depends_on="emotion_probe_classification",
            extra_params={"target_emotions": ["afraid", "calm"]},
        )
        exp = ParametricScalingExperiment(claim, "llama_1b", data_root)

        result = ExperimentResult(
            claim_id=claim.claim_id,
            model_key="llama_1b",
            paper_id="emotions",
            metrics={"rank_correlation": 0.25},
        )
        assert exp.evaluate(result) is False


# ---------------------------------------------------------------------------
# CausalSteeringExperiment.evaluate()
# ---------------------------------------------------------------------------

class TestCausalSteeringEvaluate:
    """Test evaluate() on CausalSteeringExperiment."""

    def test_evaluate_passes(self, data_root: Path) -> None:
        from src.experiments.causal_steering import CausalSteeringExperiment

        claim = _make_claim(
            "causal_steering",
            "causal_effect_count",
            3.0,
            depends_on="emotion_probe_classification",
            extra_params={"steering_emotions": ["desperate", "calm"]},
        )
        exp = CausalSteeringExperiment(claim, "llama_1b", data_root)

        result = ExperimentResult(
            claim_id=claim.claim_id,
            model_key="llama_1b",
            paper_id="emotions",
            metrics={"causal_effect_count": 5},
        )
        assert exp.evaluate(result) is True

    def test_evaluate_fails(self, data_root: Path) -> None:
        from src.experiments.causal_steering import CausalSteeringExperiment

        claim = _make_claim(
            "causal_steering",
            "causal_effect_count",
            3.0,
            depends_on="emotion_probe_classification",
            extra_params={"steering_emotions": ["desperate", "calm"]},
        )
        exp = CausalSteeringExperiment(claim, "llama_1b", data_root)

        result = ExperimentResult(
            claim_id=claim.claim_id,
            model_key="llama_1b",
            paper_id="emotions",
            metrics={"causal_effect_count": 1},
        )
        assert exp.evaluate(result) is False


# ---------------------------------------------------------------------------
# ExperimentResult save/load round-trip
# ---------------------------------------------------------------------------

class TestResultSaveLoad:
    """Test ExperimentResult serialization to JSON and back."""

    def test_result_save_load(self, tmp_path: Path) -> None:
        """Save and load should produce an identical ExperimentResult."""
        original = ExperimentResult(
            claim_id="test_claim",
            model_key="llama_1b",
            paper_id="emotions",
            metrics={
                "probe_accuracy": 0.73,
                "best_layer": 10,
                "per_concept": {"happy": 0.81, "sad": 0.65},
            },
            success=True,
            metadata={"seed": 42, "n_folds": 5},
        )

        path = tmp_path / "result.json"
        original.save(path)
        assert path.exists()

        loaded = ExperimentResult.load(path)

        assert loaded.claim_id == original.claim_id
        assert loaded.model_key == original.model_key
        assert loaded.paper_id == original.paper_id
        assert loaded.metrics == original.metrics
        assert loaded.success == original.success
        assert loaded.metadata == original.metadata

    def test_result_save_atomic(self, tmp_path: Path) -> None:
        """After save, no .tmp file should remain (atomic rename)."""
        result = ExperimentResult(
            claim_id="test",
            model_key="llama_1b",
            paper_id="emotions",
            metrics={"accuracy": 0.5},
        )
        path = tmp_path / "result.json"
        result.save(path)

        assert path.exists()
        assert not path.with_suffix(".tmp").exists()

    def test_result_save_creates_parents(self, tmp_path: Path) -> None:
        """save() should create parent directories if they don't exist."""
        result = ExperimentResult(
            claim_id="test",
            model_key="llama_1b",
            paper_id="emotions",
            metrics={"accuracy": 0.5},
        )
        path = tmp_path / "deep" / "nested" / "dir" / "result.json"
        result.save(path)
        assert path.exists()
