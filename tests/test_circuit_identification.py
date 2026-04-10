"""Unit tests for CircuitIdentificationExperiment.

These tests use mocked configs and synthetic payloads -- no real model is
loaded. They verify:

- evaluate() correctly compares metric_recovery against the threshold
- __init__ rejects malformed params (missing prompts, unknown method)
- The experiment is registered in EXPERIMENT_REGISTRY
- _build_result wraps a payload into a sane ExperimentResult
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.core.claim import ClaimConfig, ExperimentResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_circuit_claim(
    method: str = "causal_trace",
    threshold: float = 0.5,
    *,
    success_metric: str = "metric_recovery",
    extra: dict | None = None,
) -> ClaimConfig:
    """Build a minimal valid ClaimConfig for circuit_identification."""
    params: dict = {
        "clean_prompt": "When Mary and John went to the store, John gave a drink to",
        "corrupted_prompt": "When Alice and John went to the store, John gave a drink to",
        "answer_token": "Mary",
        "distractor_token": "John",
        "method": method,
        "top_k_components": 5,
        "metric": "logit_diff",
    }
    if extra:
        params.update(extra)

    return ClaimConfig(
        paper_id="ioi",
        claim_id="test_circuit",
        description="Test claim for circuit identification",
        experiment_type="circuit_identification",
        params=params,
        success_metric=success_metric,
        success_threshold=threshold,
    )


# ---------------------------------------------------------------------------
# evaluate()
# ---------------------------------------------------------------------------

class TestEvaluate:
    """Verify evaluate() compares metric_recovery against the threshold."""

    def test_evaluate_passes_above_threshold(self, data_root: Path) -> None:
        """metric_recovery >= threshold should evaluate True."""
        from src.experiments.circuit_identification import CircuitIdentificationExperiment

        claim = _make_circuit_claim(threshold=0.5)
        exp = CircuitIdentificationExperiment(claim, "gpt2_small", data_root)

        result = ExperimentResult(
            claim_id=claim.claim_id,
            model_key="gpt2_small",
            paper_id="ioi",
            metrics={"metric_recovery": 0.72},
        )
        assert exp.evaluate(result) is True

    def test_evaluate_fails_below_threshold(self, data_root: Path) -> None:
        """metric_recovery < threshold should evaluate False."""
        from src.experiments.circuit_identification import CircuitIdentificationExperiment

        claim = _make_circuit_claim(threshold=0.5)
        exp = CircuitIdentificationExperiment(claim, "gpt2_small", data_root)

        result = ExperimentResult(
            claim_id=claim.claim_id,
            model_key="gpt2_small",
            paper_id="ioi",
            metrics={"metric_recovery": 0.31},
        )
        assert exp.evaluate(result) is False

    def test_evaluate_at_exact_threshold(self, data_root: Path) -> None:
        """metric_recovery == threshold should evaluate True (>=, not >)."""
        from src.experiments.circuit_identification import CircuitIdentificationExperiment

        claim = _make_circuit_claim(threshold=0.5)
        exp = CircuitIdentificationExperiment(claim, "gpt2_small", data_root)

        result = ExperimentResult(
            claim_id=claim.claim_id,
            model_key="gpt2_small",
            paper_id="ioi",
            metrics={"metric_recovery": 0.5},
        )
        assert exp.evaluate(result) is True

    def test_evaluate_missing_metric_returns_false(self, data_root: Path) -> None:
        """Result lacking metric_recovery key should evaluate False (default 0.0)."""
        from src.experiments.circuit_identification import CircuitIdentificationExperiment

        claim = _make_circuit_claim(threshold=0.5)
        exp = CircuitIdentificationExperiment(claim, "gpt2_small", data_root)

        result = ExperimentResult(
            claim_id=claim.claim_id,
            model_key="gpt2_small",
            paper_id="ioi",
            metrics={},
        )
        assert exp.evaluate(result) is False


# ---------------------------------------------------------------------------
# __init__ validation
# ---------------------------------------------------------------------------

class TestInitValidation:
    """Verify __init__ rejects bad params with clear errors."""

    def test_init_requires_clean_and_corrupted_prompts(self, data_root: Path) -> None:
        """Missing clean_prompt or corrupted_prompt should raise ValueError."""
        from src.experiments.circuit_identification import CircuitIdentificationExperiment

        # Missing clean_prompt
        claim = ClaimConfig(
            paper_id="ioi",
            claim_id="bad_claim",
            description="Bad claim - no clean_prompt",
            experiment_type="circuit_identification",
            params={
                "corrupted_prompt": "something",
                "answer_token": "X",
                "method": "causal_trace",
            },
            success_metric="metric_recovery",
            success_threshold=0.5,
        )
        with pytest.raises(ValueError, match="clean_prompt"):
            CircuitIdentificationExperiment(claim, "gpt2_small", data_root)

        # Missing corrupted_prompt
        claim2 = ClaimConfig(
            paper_id="ioi",
            claim_id="bad_claim2",
            description="Bad claim - no corrupted_prompt",
            experiment_type="circuit_identification",
            params={
                "clean_prompt": "something",
                "answer_token": "X",
                "method": "causal_trace",
            },
            success_metric="metric_recovery",
            success_threshold=0.5,
        )
        with pytest.raises(ValueError, match="corrupted_prompt"):
            CircuitIdentificationExperiment(claim2, "gpt2_small", data_root)

    def test_init_requires_answer(self, data_root: Path) -> None:
        """Missing answer_token AND answer_token_id should raise ValueError."""
        from src.experiments.circuit_identification import CircuitIdentificationExperiment

        claim = ClaimConfig(
            paper_id="ioi",
            claim_id="bad_claim",
            description="Bad claim - no answer",
            experiment_type="circuit_identification",
            params={
                "clean_prompt": "p1",
                "corrupted_prompt": "p2",
                "method": "causal_trace",
            },
            success_metric="metric_recovery",
            success_threshold=0.5,
        )
        with pytest.raises(ValueError, match="answer_token"):
            CircuitIdentificationExperiment(claim, "gpt2_small", data_root)

    def test_init_unknown_method_raises(self, data_root: Path) -> None:
        """Unknown method string should raise ValueError listing valid options."""
        from src.experiments.circuit_identification import CircuitIdentificationExperiment

        claim = _make_circuit_claim(method="not_a_real_method")
        with pytest.raises(ValueError, match="Unknown method"):
            CircuitIdentificationExperiment(claim, "gpt2_small", data_root)

    def test_init_unknown_metric_raises(self, data_root: Path) -> None:
        """Unknown metric string should raise ValueError listing valid options."""
        from src.experiments.circuit_identification import CircuitIdentificationExperiment

        claim = _make_circuit_claim(extra={"metric": "not_a_metric"})
        with pytest.raises(ValueError, match="Unknown metric"):
            CircuitIdentificationExperiment(claim, "gpt2_small", data_root)

    def test_init_accepts_token_ids_directly(self, data_root: Path) -> None:
        """Passing answer_token_id (int) instead of answer_token should work."""
        from src.experiments.circuit_identification import CircuitIdentificationExperiment

        claim = ClaimConfig(
            paper_id="ioi",
            claim_id="ok_claim",
            description="OK claim with explicit ids",
            experiment_type="circuit_identification",
            params={
                "clean_prompt": "p1",
                "corrupted_prompt": "p2",
                "answer_token_id": 12345,
                "distractor_token_id": 67890,
                "method": "causal_trace",
            },
            success_metric="metric_recovery",
            success_threshold=0.5,
        )
        exp = CircuitIdentificationExperiment(claim, "gpt2_small", data_root)
        assert exp.answer_token_id == 12345
        assert exp.distractor_token_id == 67890


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class TestRegistry:
    """Verify circuit_identification is registered."""

    def test_registry_includes_circuit_identification(self) -> None:
        """The experiment must be findable by name in EXPERIMENT_REGISTRY."""
        from src.experiments import EXPERIMENT_REGISTRY, get_experiment_class
        from src.experiments.circuit_identification import CircuitIdentificationExperiment

        assert "circuit_identification" in EXPERIMENT_REGISTRY
        assert EXPERIMENT_REGISTRY["circuit_identification"] is CircuitIdentificationExperiment

        cls = get_experiment_class("circuit_identification")
        assert cls is CircuitIdentificationExperiment

    def test_inherits_from_experiment(self) -> None:
        """CircuitIdentificationExperiment must subclass Experiment."""
        from src.core.experiment import Experiment
        from src.experiments.circuit_identification import CircuitIdentificationExperiment

        assert issubclass(CircuitIdentificationExperiment, Experiment)


# ---------------------------------------------------------------------------
# _build_result wrapping
# ---------------------------------------------------------------------------

class TestBuildResult:
    """Verify _build_result correctly wraps a payload into an ExperimentResult."""

    def test_build_result_round_trip(self, data_root: Path) -> None:
        """A synthetic payload should populate metrics and metadata correctly."""
        from src.experiments.circuit_identification import CircuitIdentificationExperiment

        claim = _make_circuit_claim()
        exp = CircuitIdentificationExperiment(claim, "gpt2_small", data_root)

        payload = {
            "method": "causal_trace",
            "metric_name": "logit_diff",
            "top_k_components": [("L5P3", 0.82), ("L7P4", 0.41)],
            "all_ranked_components": [("L5P3", 0.82), ("L7P4", 0.41)],
            "clean_metric_value": 3.2,
            "corrupted_metric_value": -1.1,
            "metric_recovery": 0.82,
            "n_components_above_threshold": 2,
        }

        result = exp._build_result(payload)
        assert result.claim_id == "test_circuit"
        assert result.paper_id == "ioi"
        assert result.model_key == "gpt2_small"

        assert result.metrics["metric_recovery"] == 0.82
        assert result.metrics["clean_metric_value"] == 3.2
        assert result.metrics["corrupted_metric_value"] == -1.1
        assert result.metrics["n_components_above_threshold"] == 2
        assert result.metrics["method"] == "causal_trace"
        assert result.metrics["top_k_components"] == [("L5P3", 0.82), ("L7P4", 0.41)]

        # Metadata should preserve the original prompts and config knobs.
        assert result.metadata["method"] == "causal_trace"
        assert result.metadata["metric"] == "logit_diff"
        assert result.metadata["top_k"] == 5
        assert "clean_prompt" in result.metadata
        assert "corrupted_prompt" in result.metadata
