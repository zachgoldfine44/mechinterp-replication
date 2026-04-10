"""Tests for src/core/sanity_checks.py."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from src.core.claim import ClaimConfig, ExperimentResult
from src.core.sanity_checks import (
    check_chance_level_accuracy,
    check_confusion_matrix_resolution_artifact,
    check_finite_metrics,
    check_negative_control_contamination,
    run_sanity_checks,
)


def _make_claim(
    success_metric: str = "probe_accuracy",
    threshold: float = 0.7,
    concept_set: list[str] | None = None,
    n_stimuli_per_concept: int | None = None,
    experiment_type: str = "probe_classification",
) -> ClaimConfig:
    params: dict = {}
    if concept_set is not None:
        params["concept_set"] = concept_set
    if n_stimuli_per_concept is not None:
        params["n_stimuli_per_concept"] = n_stimuli_per_concept
    return ClaimConfig(
        paper_id="test",
        claim_id="claim_x",
        description="test claim",
        experiment_type=experiment_type,
        params=params,
        success_metric=success_metric,
        success_threshold=threshold,
    )


def _zero_cm(n: int) -> list[list[int]]:
    return [[0] * n for _ in range(n)]


# ---------------------------------------------------------------------------
# #1: 0.800 = 12/15 resolution artifact
# ---------------------------------------------------------------------------

def test_confusion_matrix_resolution_artifact_12_of_15():
    n = 15
    # 30 total samples (2 per row), 12 correct on diagonal -> 0.8 exactly
    cm = _zero_cm(n)
    # First 12 rows: both samples correct (diag += 2) -> 24 diag contributions
    # But we only want 12 diag, so: put 1 on diag + 1 off-diag per row for first 12 rows,
    # and 0 diag + 2 off-diag for last 3 rows. That yields diag=12, total=30.
    for i in range(12):
        cm[i][i] = 1
        cm[i][(i + 1) % n] = 1
    for i in range(12, 15):
        cm[i][(i + 1) % n] = 2
    total = sum(sum(row) for row in cm)
    assert total == 30
    diag = sum(cm[i][i] for i in range(n))
    assert diag == 12  # 0.8 exactly
    result = ExperimentResult(
        claim_id="claim_x",
        model_key="llama_1b",
        paper_id="test",
        metrics={
            "diagonal_dominance": 0.8,
            "confusion_matrix": cm,
        },
    )
    claim = _make_claim(success_metric="diagonal_dominance", threshold=0.67)
    cr = check_confusion_matrix_resolution_artifact(result, claim)
    assert cr.severity in ("warn", "error")
    assert not cr.passed
    assert cr.details["total_samples"] == 30
    assert cr.details["n_concepts"] == 15
    assert cr.details.get("matched_fraction") == "12/15"


def test_confusion_matrix_resolution_artifact_per_class_under_3():
    n = 15
    cm = _zero_cm(n)
    for i in range(n):
        cm[i][i] = 1  # per-class = 1
    result = ExperimentResult(
        claim_id="claim_x",
        model_key="m",
        paper_id="test",
        metrics={"diagonal_dominance": 1.0, "confusion_matrix": cm},
    )
    claim = _make_claim(success_metric="diagonal_dominance")
    cr = check_confusion_matrix_resolution_artifact(result, claim)
    assert cr.severity == "error"


# ---------------------------------------------------------------------------
# #2: chance-level probe
# ---------------------------------------------------------------------------

def test_chance_level_accuracy_fires():
    result = ExperimentResult(
        claim_id="claim_x",
        model_key="m",
        paper_id="test",
        metrics={"probe_accuracy": 0.07},
    )
    claim = _make_claim(
        success_metric="probe_accuracy",
        concept_set=[f"c{i}" for i in range(15)],
    )
    cr = check_chance_level_accuracy(result, claim)
    assert cr.severity == "warn"
    assert not cr.passed
    assert "chance" in cr.message.lower()


def test_chance_level_accuracy_clean():
    result = ExperimentResult(
        claim_id="claim_x",
        model_key="m",
        paper_id="test",
        metrics={"probe_accuracy": 0.85},
    )
    claim = _make_claim(
        success_metric="probe_accuracy",
        concept_set=[f"c{i}" for i in range(15)],
    )
    cr = check_chance_level_accuracy(result, claim)
    assert cr.severity == "info"
    assert cr.passed


# ---------------------------------------------------------------------------
# #9: NaN metric
# ---------------------------------------------------------------------------

def test_finite_metrics_nan_fires():
    result = ExperimentResult(
        claim_id="claim_x",
        model_key="m",
        paper_id="test",
        metrics={"probe_accuracy": float("nan")},
    )
    claim = _make_claim(success_metric="probe_accuracy")
    cr = check_finite_metrics(result, claim)
    assert cr.severity == "error"
    assert not cr.passed


def test_finite_metrics_inf_fires():
    result = ExperimentResult(
        claim_id="claim_x",
        model_key="m",
        paper_id="test",
        metrics={"loss": float("inf")},
    )
    claim = _make_claim(success_metric="probe_accuracy")
    cr = check_finite_metrics(result, claim)
    assert cr.severity == "error"


# ---------------------------------------------------------------------------
# #8: negative control contamination
# ---------------------------------------------------------------------------

def test_negative_control_contamination_fires():
    result = ExperimentResult(
        claim_id="claim_x",
        model_key="m",
        paper_id="test",
        metrics={"probe_accuracy": 0.85, "negative_control_contamination_ratio": 0.7},
    )
    claim = _make_claim()
    cr = check_negative_control_contamination(result, claim)
    assert cr.severity == "error"
    assert not cr.passed


def test_negative_control_contamination_clean():
    result = ExperimentResult(
        claim_id="claim_x",
        model_key="m",
        paper_id="test",
        metrics={"probe_accuracy": 0.85, "negative_control_contamination_ratio": 0.1},
    )
    claim = _make_claim()
    cr = check_negative_control_contamination(result, claim)
    assert cr.severity == "info"
    assert cr.passed


# ---------------------------------------------------------------------------
# Clean result passes everything important
# ---------------------------------------------------------------------------

def test_clean_result_no_errors(tmp_path: Path):
    n = 15
    cm = _zero_cm(n)
    # 30 per class, 26 correct -> no artifact
    for i in range(n):
        cm[i][i] = 26
        cm[i][(i + 1) % n] = 4
    result = ExperimentResult(
        claim_id="claim_x",
        model_key="m",
        paper_id="test",
        metrics={
            "probe_accuracy": 0.866,
            "confusion_matrix": cm,
            "per_concept_accuracy": {f"c{i}": 0.85 + 0.01 * (i % 3) for i in range(15)},
            "n_stimuli_total": 450,
        },
        success=True,
    )
    claim = _make_claim(
        success_metric="probe_accuracy",
        threshold=0.7,
        concept_set=[f"c{i}" for i in range(15)],
        n_stimuli_per_concept=30,
    )
    report = run_sanity_checks(result, claim, tmp_path / "sanity.json")
    assert report["n_errors"] == 0
    assert (tmp_path / "sanity.json").exists()
    loaded = json.loads((tmp_path / "sanity.json").read_text())
    assert loaded["claim_id"] == "claim_x"
    assert loaded["n_errors"] == 0


# ---------------------------------------------------------------------------
# Aggregator: atomic write & surfaces all checks
# ---------------------------------------------------------------------------

def test_run_sanity_checks_writes_report(tmp_path: Path):
    result = ExperimentResult(
        claim_id="claim_x",
        model_key="m",
        paper_id="test",
        metrics={"probe_accuracy": 0.07},
    )
    claim = _make_claim(
        success_metric="probe_accuracy",
        concept_set=[f"c{i}" for i in range(15)],
    )
    out = tmp_path / "sub" / "sanity.json"
    report = run_sanity_checks(result, claim, out)
    assert out.exists()
    assert report["n_warnings"] >= 1
    names = [c["name"] for c in report["checks"]]
    assert "chance_level_accuracy" in names
    assert "finite_metrics" in names


def test_run_sanity_checks_missing_keys_are_info(tmp_path: Path):
    result = ExperimentResult(
        claim_id="claim_x",
        model_key="m",
        paper_id="test",
        metrics={},
    )
    claim = _make_claim()
    report = run_sanity_checks(result, claim, tmp_path / "sanity.json")
    # Missing metric should not raise errors across checks (except
    # metric_in_range/finite are info since value is None and not present)
    assert report["n_errors"] == 0
