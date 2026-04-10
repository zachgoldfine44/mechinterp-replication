"""Post-experiment sanity checks for ExperimentResult artifacts.

Runs after every experiment finishes, writes a report to sanity.json next to
result.json, and surfaces warnings in the pipeline log. Catches known
artifacts: resolution quantization, chance-level probes, NaN metrics,
negative-control contamination, and stimulus-count drift.

Every check is defensive: if the expected metric key is missing, it returns
passed=True, severity="info", message="N/A". Registration is explicit via
the CHECKS list at the bottom of the file.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal

from src.core.claim import ClaimConfig, ExperimentResult

Severity = Literal["info", "warn", "error"]


@dataclass
class CheckResult:
    name: str
    severity: Severity
    passed: bool
    message: str
    details: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _na(name: str, message: str = "N/A") -> CheckResult:
    return CheckResult(name=name, severity="info", passed=True, message=message)


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _is_finite(x: Any) -> bool:
    return _is_number(x) and math.isfinite(float(x))


def _total_samples_from_confusion(cm: Any) -> int | None:
    try:
        total = 0
        for row in cm:
            for v in row:
                total += int(v)
        return total
    except Exception:
        return None


def _shape(cm: Any) -> tuple[int, int] | None:
    try:
        rows = len(cm)
        cols = len(cm[0]) if rows else 0
        return (rows, cols)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def check_confusion_matrix_resolution_artifact(
    result: ExperimentResult, claim: ClaimConfig,
) -> CheckResult:
    name = "confusion_matrix_resolution_artifact"
    cm = result.metrics.get("confusion_matrix")
    if cm is None:
        return _na(name, "no confusion_matrix")

    shape = _shape(cm)
    total = _total_samples_from_confusion(cm)
    if shape is None or total is None or total == 0:
        return _na(name, "confusion_matrix unreadable")

    n_concepts = shape[0]
    per_class = total / n_concepts if n_concepts else 0

    metric_val = result.metrics.get(claim.success_metric)
    details: dict[str, Any] = {
        "n_concepts": n_concepts,
        "total_samples": total,
        "per_class_samples": per_class,
        "metric_value": metric_val,
    }

    # Find k/n_concepts match
    matched_fraction: str | None = None
    if _is_finite(metric_val) and n_concepts > 0:
        for k in range(n_concepts + 1):
            if abs(float(metric_val) - k / n_concepts) < 1e-6:
                matched_fraction = f"{k}/{n_concepts}"
                details["matched_fraction"] = matched_fraction
                break

    severity: Severity = "info"
    passed = True
    msg = "ok"

    if per_class < 3:
        severity = "error"
        passed = False
        msg = (
            f"per-class sample size is {per_class:.2f} (<3); metric is effectively "
            f"pinned to a small set of discrete values"
        )
    elif total < 30:
        severity = "warn"
        passed = False
        msg = (
            f"total samples = {total} (<30); resolution artifact risk. "
            + (f"metric exactly {matched_fraction}." if matched_fraction else "")
        )
    elif matched_fraction is not None and per_class < 5:
        severity = "warn"
        passed = False
        msg = (
            f"metric exactly {matched_fraction} with only {per_class:.2f} per class; "
            "likely resolution artifact"
        )

    return CheckResult(name=name, severity=severity, passed=passed, message=msg, details=details)


def check_chance_level_accuracy(
    result: ExperimentResult, claim: ClaimConfig,
) -> CheckResult:
    name = "chance_level_accuracy"
    acc_key = None
    acc_val = None
    for k, v in result.metrics.items():
        if k.endswith("accuracy") and _is_finite(v):
            acc_key = k
            acc_val = float(v)
            break
    if acc_val is None:
        return _na(name, "no accuracy metric")

    # Determine n_concepts
    n_concepts: int | None = None
    cm = result.metrics.get("confusion_matrix")
    shape = _shape(cm) if cm is not None else None
    if shape:
        n_concepts = shape[0]
    else:
        concept_set = claim.params.get("concept_set")
        if isinstance(concept_set, list):
            n_concepts = len(concept_set)

    if not n_concepts or n_concepts < 2:
        return _na(name, "n_concepts unknown")

    chance = 1.0 / n_concepts
    details = {
        "accuracy_key": acc_key, "accuracy_value": acc_val,
        "n_concepts": n_concepts, "chance_level": chance,
    }
    if abs(acc_val - chance) < 0.05:
        return CheckResult(
            name=name, severity="warn", passed=False,
            message=(
                f"{acc_key}={acc_val:.3f} is within 0.05 of chance "
                f"({chance:.3f}) for {n_concepts} concepts; probe may be at chance"
            ),
            details=details,
        )
    return CheckResult(
        name=name, severity="info", passed=True,
        message=f"{acc_key}={acc_val:.3f} above chance", details=details,
    )


def check_round_number_suspicion(
    result: ExperimentResult, claim: ClaimConfig,
) -> CheckResult:
    name = "round_number_suspicion"
    val = result.metrics.get(claim.success_metric)
    if not _is_finite(val):
        return _na(name, "no metric")
    v = float(val)

    round_set = {0.0, 0.25, 0.5, 0.75, 1.0}
    for r in round_set:
        if abs(v - r) < 1e-9:
            return CheckResult(
                name=name, severity="info", passed=True,
                message=f"metric is exactly {r}",
                details={"metric_value": v, "matched": str(r)},
            )

    for y in range(1, 16):
        for x in range(y + 1):
            if abs(v - x / y) < 1e-9:
                return CheckResult(
                    name=name, severity="info", passed=True,
                    message=f"metric is exactly {x}/{y}",
                    details={"metric_value": v, "matched": f"{x}/{y}"},
                )
    return CheckResult(
        name=name, severity="info", passed=True,
        message="no round-number match", details={"metric_value": v},
    )


def check_per_concept_uniformity(
    result: ExperimentResult, claim: ClaimConfig,
) -> CheckResult:
    name = "per_concept_uniformity"
    per = result.metrics.get("per_concept_accuracy")
    if not isinstance(per, dict) or len(per) < 2:
        return _na(name, "no per_concept_accuracy")
    vals = [float(v) for v in per.values() if _is_finite(v)]
    if len(vals) < 2:
        return _na(name, "insufficient per-concept values")
    mean = sum(vals) / len(vals)
    var = sum((v - mean) ** 2 for v in vals) / len(vals)
    std = math.sqrt(var)
    details = {"mean": mean, "std": std, "n": len(vals)}
    if std < 0.01:
        return CheckResult(
            name=name, severity="warn", passed=False,
            message=f"per-concept std={std:.4f} <0.01; probe may have collapsed",
            details=details,
        )
    return CheckResult(
        name=name, severity="info", passed=True,
        message=f"per-concept std={std:.4f}", details=details,
    )


def check_metric_in_range(
    result: ExperimentResult, claim: ClaimConfig,
) -> CheckResult:
    name = "metric_in_range"
    val = result.metrics.get(claim.success_metric)
    if val is None:
        return _na(name, f"no value for {claim.success_metric}")
    if not _is_number(val) or not math.isfinite(float(val)):
        return CheckResult(
            name=name, severity="error", passed=False,
            message=f"{claim.success_metric} is NaN/inf/non-numeric: {val!r}",
            details={"metric_value": val},
        )
    v = float(val)
    bounded_metrics = ("accuracy", "dominance", "correlation", "f1")
    if any(b in claim.success_metric for b in bounded_metrics):
        if v < -1.0 - 1e-9 or v > 1.0 + 1e-9:
            return CheckResult(
                name=name, severity="error", passed=False,
                message=f"{claim.success_metric}={v} outside [-1, 1]",
                details={"metric_value": v},
            )
    return CheckResult(
        name=name, severity="info", passed=True,
        message=f"{claim.success_metric}={v} in range",
        details={"metric_value": v},
    )


def check_success_threshold_consistency(
    result: ExperimentResult, claim: ClaimConfig,
) -> CheckResult:
    name = "success_threshold_consistency"
    val = result.metrics.get(claim.success_metric)
    if not _is_finite(val):
        return _na(name, "no finite metric")
    v = float(val)
    threshold = float(claim.success_threshold)
    details = {"metric_value": v, "threshold": threshold, "success_flag": result.success}
    if result.success is True and v < threshold - 1e-9:
        return CheckResult(
            name=name, severity="error", passed=False,
            message=(
                f"result.success=True but {claim.success_metric}={v} < "
                f"threshold={threshold}"
            ),
            details=details,
        )
    if result.success is False and v >= threshold - 1e-9 and v >= threshold:
        # Only info — might be intentional (e.g., other criteria failed).
        return CheckResult(
            name=name, severity="info", passed=True,
            message=(
                f"result.success=False but {claim.success_metric}={v} >= "
                f"threshold={threshold}"
            ),
            details=details,
        )
    return CheckResult(
        name=name, severity="info", passed=True,
        message="threshold consistent with success flag", details=details,
    )


def check_n_stimuli_matches_config(
    result: ExperimentResult, claim: ClaimConfig,
) -> CheckResult:
    name = "n_stimuli_matches_config"
    actual = result.metrics.get("n_stimuli_total")
    per_concept = claim.params.get("n_stimuli_per_concept")
    concept_set = claim.params.get("concept_set")
    if actual is None or per_concept is None or not isinstance(concept_set, list):
        return _na(name, "missing n_stimuli or params")
    expected = per_concept * len(concept_set)
    if expected == 0:
        return _na(name, "expected=0")
    drift = abs(float(actual) - expected) / expected
    details = {"actual": actual, "expected": expected, "drift": drift}
    if drift > 0.1:
        return CheckResult(
            name=name, severity="warn", passed=False,
            message=f"n_stimuli_total={actual} vs expected {expected} (drift {drift:.1%})",
            details=details,
        )
    return CheckResult(
        name=name, severity="info", passed=True,
        message=f"n_stimuli matches config (drift {drift:.1%})", details=details,
    )


def check_negative_control_contamination(
    result: ExperimentResult, claim: ClaimConfig,
) -> CheckResult:
    name = "negative_control_contamination"
    ratio = result.metrics.get("negative_control_contamination_ratio")
    if not _is_finite(ratio):
        return _na(name, "no contamination metric")
    r = float(ratio)
    if r > 0.5:
        return CheckResult(
            name=name, severity="error", passed=False,
            message=f"negative_control_contamination_ratio={r:.3f} > 0.5",
            details={"ratio": r},
        )
    return CheckResult(
        name=name, severity="info", passed=True,
        message=f"contamination ratio {r:.3f} acceptable", details={"ratio": r},
    )


def check_finite_metrics(
    result: ExperimentResult, claim: ClaimConfig,
) -> CheckResult:
    name = "finite_metrics"
    bad: list[str] = []

    def walk(prefix: str, value: Any) -> None:
        if isinstance(value, dict):
            for k, v in value.items():
                walk(f"{prefix}.{k}" if prefix else str(k), v)
        elif isinstance(value, list):
            for i, v in enumerate(value):
                walk(f"{prefix}[{i}]", v)
        elif _is_number(value):
            if not math.isfinite(float(value)):
                bad.append(prefix)
        elif value is None and prefix in (claim.success_metric,):
            bad.append(prefix)

    walk("", result.metrics)

    if bad:
        return CheckResult(
            name=name, severity="error", passed=False,
            message=f"non-finite values at: {bad[:5]}{'...' if len(bad) > 5 else ''}",
            details={"bad_keys": bad},
        )
    return CheckResult(
        name=name, severity="info", passed=True,
        message="all metrics finite", details={},
    )


def check_layer_in_valid_range(
    result: ExperimentResult, claim: ClaimConfig,
) -> CheckResult:
    name = "layer_in_valid_range"
    layer = result.metrics.get("best_layer")
    if layer is None:
        layer = result.metadata.get("best_layer")
    if layer is None:
        return _na(name, "no best_layer reported")
    if not _is_number(layer):
        return _na(name, "best_layer not numeric")
    li = int(layer)
    n_layers = (
        result.metrics.get("n_layers")
        or result.metadata.get("n_layers")
    )
    details: dict[str, Any] = {"best_layer": li, "n_layers": n_layers}
    if li == -1:
        return CheckResult(
            name=name, severity="warn", passed=False,
            message="best_layer=-1 (suspicious sentinel)", details=details,
        )
    if _is_number(n_layers) and li == int(n_layers) - 1:
        return CheckResult(
            name=name, severity="warn", passed=False,
            message=f"best_layer={li} is the maximum layer; may be arbitrary",
            details=details,
        )
    return CheckResult(
        name=name, severity="info", passed=True,
        message=f"best_layer={li}", details=details,
    )


# ---------------------------------------------------------------------------
# Registry + runner
# ---------------------------------------------------------------------------

CheckFn = Callable[[ExperimentResult, ClaimConfig], CheckResult]

CHECKS: list[CheckFn] = [
    check_confusion_matrix_resolution_artifact,
    check_chance_level_accuracy,
    check_round_number_suspicion,
    check_per_concept_uniformity,
    check_metric_in_range,
    check_success_threshold_consistency,
    check_n_stimuli_matches_config,
    check_negative_control_contamination,
    check_finite_metrics,
    check_layer_in_valid_range,
]


def run_sanity_checks(
    result: ExperimentResult,
    claim: ClaimConfig,
    output_path: Path,
) -> dict[str, Any]:
    """Run all checks on a single experiment result.

    Writes the report to output_path (atomic). Returns the dict.
    Caller is responsible for surfacing the warnings to stdout.
    """
    check_results: list[CheckResult] = []
    for fn in CHECKS:
        try:
            cr = fn(result, claim)
        except Exception as exc:
            cr = CheckResult(
                name=getattr(fn, "__name__", "unknown"),
                severity="error",
                passed=False,
                message=f"check raised: {exc!r}",
                details={},
            )
        check_results.append(cr)

    n_warnings = sum(1 for c in check_results if c.severity == "warn")
    n_errors = sum(1 for c in check_results if c.severity == "error")

    summary_lines: list[str] = []
    for c in check_results:
        if c.severity in ("warn", "error"):
            summary_lines.append(f"[{c.severity.upper()}] {c.name}: {c.message}")
    summary = "; ".join(summary_lines) if summary_lines else "all checks passed"

    report: dict[str, Any] = {
        "claim_id": claim.claim_id,
        "model_key": result.model_key,
        "paper_id": result.paper_id,
        "n_warnings": n_warnings,
        "n_errors": n_errors,
        "summary": summary,
        "checks": [asdict(c) for c in check_results],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(report, f, indent=2, default=str)
    tmp.rename(output_path)

    return report
