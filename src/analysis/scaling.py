"""Scaling analysis: how do findings change with model parameters.

For each claim metric, groups results by model family, computes Spearman
correlation with model parameter count, and fits a log-linear trend.

Usage:
    from src.analysis.scaling import analyze_scaling

    report = analyze_scaling("emotions", data_root)
    for cid, family_trends in report.trends.items():
        for fam, trend in family_trends.items():
            print(f"{cid} / {fam}: r={trend.spearman_r:.3f}, p={trend.p_value:.4f}")

CLI:
    python -m src.analysis.scaling --paper emotions
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats as sp_stats

from src.core.config_loader import load_model_config, load_paper_config
from src.analysis.cross_model import compare_across_models, CrossModelReport

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ScalingTrend:
    """Scaling trend for one claim within one model family.

    Attributes:
        family: Model family name (e.g., 'llama').
        claim_id: Which claim this trend is for.
        spearman_r: Spearman rank correlation between log(params_b) and metric.
        p_value: Two-sided p-value for the Spearman correlation.
        points: List of (params_b, metric_value) tuples, sorted by params_b.
        log_linear_slope: Slope of linear fit on (log(params_b), metric).
            None if fewer than 2 data points.
        log_linear_intercept: Intercept of the log-linear fit.
            None if fewer than 2 data points.
    """

    family: str
    claim_id: str
    spearman_r: float
    p_value: float
    points: list[tuple[float, float]]
    log_linear_slope: float | None = None
    log_linear_intercept: float | None = None


@dataclass
class ScalingReport:
    """Aggregated scaling analysis for a paper.

    Attributes:
        paper_id: Paper identifier.
        trends: Nested dict -- claim_id -> family -> ScalingTrend.
        overall_correlation: Per-claim Spearman r across ALL models (ignoring family).
    """

    paper_id: str
    trends: dict[str, dict[str, ScalingTrend]]
    overall_correlation: dict[str, tuple[float, float]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def _fit_trend(
    family: str,
    claim_id: str,
    points: list[tuple[float, float]],
) -> ScalingTrend:
    """Compute Spearman r and log-linear fit for a set of (params_b, metric) points.

    Args:
        family: Model family name.
        claim_id: Claim identifier.
        points: List of (params_b, metric_value) tuples.

    Returns:
        Populated ScalingTrend.
    """
    points = sorted(points, key=lambda p: p[0])
    params = np.array([p[0] for p in points])
    metrics = np.array([p[1] for p in points])

    # Spearman correlation
    if len(points) >= 3:
        r, p = sp_stats.spearmanr(params, metrics)
    elif len(points) == 2:
        # With 2 points Spearman is +/-1 or 0; p-value is not meaningful
        r, p = sp_stats.spearmanr(params, metrics)
    else:
        r, p = 0.0, 1.0

    # Handle NaN from constant arrays
    if np.isnan(r):
        r = 0.0
    if np.isnan(p):
        p = 1.0

    # Log-linear fit: metric = slope * log(params_b) + intercept
    slope: float | None = None
    intercept: float | None = None
    if len(points) >= 2:
        log_params = np.log(params.clip(min=1e-6))
        try:
            slope_val, intercept_val, _, _, _ = sp_stats.linregress(log_params, metrics)
            slope = float(slope_val)
            intercept = float(intercept_val)
        except Exception:
            pass

    return ScalingTrend(
        family=family,
        claim_id=claim_id,
        spearman_r=float(r),
        p_value=float(p),
        points=points,
        log_linear_slope=slope,
        log_linear_intercept=intercept,
    )


def analyze_scaling(
    paper_id: str,
    data_root: Path,
    cross_model_report: CrossModelReport | None = None,
    replication_id: str | None = None,
) -> ScalingReport:
    """Analyze how experiment metrics scale with model size.

    For each claim and model family, computes Spearman correlation between
    model parameter count and the claim's primary metric, and fits a
    log-linear trend.

    Args:
        paper_id: Paper identifier.
        data_root: Root data directory.
        cross_model_report: Pre-computed CrossModelReport. If None, one is
            built from disk.
        replication_id: Optional replication identifier. Forwarded to
            ``compare_across_models`` and config loading.

    Returns:
        ScalingReport with per-claim, per-family trends.
    """
    if cross_model_report is None:
        cross_model_report = compare_across_models(
            paper_id, data_root, replication_id=replication_id,
        )

    report = cross_model_report
    paper_config = load_paper_config(paper_id, replication_id=replication_id)
    claim_metrics = {c.claim_id: c.success_metric for c in paper_config.claims}

    trends: dict[str, dict[str, ScalingTrend]] = {}
    overall_corr: dict[str, tuple[float, float]] = {}

    for claim in paper_config.claims:
        cid = claim.claim_id
        metric_key = claim_metrics[cid]
        model_results = report.claim_results.get(cid, {})

        # Gather (params_b, metric) per family
        family_points: dict[str, list[tuple[float, float]]] = {}
        all_points: list[tuple[float, float]] = []

        for mk, result in model_results.items():
            meta = report.model_metadata.get(mk, {})
            params_b = meta.get("params_b", 0.0)
            val = result.metrics.get(metric_key)
            if val is None:
                continue
            try:
                val_f = float(val)
            except (TypeError, ValueError):
                continue

            family = meta.get("family", "unknown")
            family_points.setdefault(family, []).append((params_b, val_f))
            all_points.append((params_b, val_f))

        # Per-family trends
        trends[cid] = {}
        for fam, pts in family_points.items():
            trends[cid][fam] = _fit_trend(fam, cid, pts)

        # Overall correlation across all models
        if len(all_points) >= 2:
            params_all = np.array([p[0] for p in all_points])
            metrics_all = np.array([p[1] for p in all_points])
            r, p = sp_stats.spearmanr(params_all, metrics_all)
            if np.isnan(r):
                r = 0.0
            if np.isnan(p):
                p = 1.0
            overall_corr[cid] = (float(r), float(p))
        else:
            overall_corr[cid] = (0.0, 1.0)

    return ScalingReport(
        paper_id=paper_id,
        trends=trends,
        overall_correlation=overall_corr,
    )


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def print_scaling_report(report: ScalingReport) -> str:
    """Format scaling analysis as a readable string.

    Args:
        report: ScalingReport from analyze_scaling().

    Returns:
        Multi-line formatted string.
    """
    lines: list[str] = []
    lines.append(f"Scaling analysis: {report.paper_id}")
    lines.append("=" * 72)

    for cid in sorted(report.trends.keys()):
        r_all, p_all = report.overall_correlation.get(cid, (0.0, 1.0))
        lines.append(f"\n{cid}  (overall r={r_all:.3f}, p={p_all:.4f})")
        lines.append("-" * 40)

        family_trends = report.trends[cid]
        for fam in sorted(family_trends.keys()):
            t = family_trends[fam]
            pts_str = ", ".join(f"{p:.1f}B->{m:.3f}" for p, m in t.points)
            slope_str = f"slope={t.log_linear_slope:.4f}" if t.log_linear_slope is not None else "slope=N/A"
            lines.append(
                f"  {fam:<8}  r={t.spearman_r:+.3f}  p={t.p_value:.4f}  "
                f"{slope_str}  [{pts_str}]"
            )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_scaling_report(report: ScalingReport, output_dir: Path) -> None:
    """Save scaling report to JSON.

    Writes output_dir/scaling_report.json.

    Args:
        report: ScalingReport to save.
        output_dir: Directory for output (created if needed).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    data: dict[str, Any] = {
        "paper_id": report.paper_id,
        "overall_correlation": {
            cid: {"spearman_r": r, "p_value": p}
            for cid, (r, p) in report.overall_correlation.items()
        },
        "trends": {},
    }

    for cid, family_trends in report.trends.items():
        data["trends"][cid] = {}
        for fam, t in family_trends.items():
            data["trends"][cid][fam] = {
                "spearman_r": t.spearman_r,
                "p_value": t.p_value,
                "log_linear_slope": t.log_linear_slope,
                "log_linear_intercept": t.log_linear_intercept,
                "points": [{"params_b": p, "metric": m} for p, m in t.points],
            }

    json_path = output_dir / "scaling_report.json"
    tmp_path = json_path.with_suffix(".tmp")
    with open(tmp_path, "w") as f:
        json.dump(data, f, indent=2)
    tmp_path.rename(json_path)
    logger.info("Saved scaling report: %s", json_path)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI: python -m src.analysis.scaling --paper <paper_id>"""
    import argparse

    from src.utils.env import get_data_root

    parser = argparse.ArgumentParser(description="Scaling analysis")
    parser.add_argument("--paper", required=True, help="Paper ID")
    parser.add_argument(
        "--replication", default=None,
        help="Replication identifier (e.g. emotions-zachgoldfine44-6models).",
    )
    parser.add_argument("--output", default=None, help="Output directory override")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    data_root = get_data_root()
    report = analyze_scaling(
        args.paper, data_root, replication_id=args.replication,
    )
    print(print_scaling_report(report))

    from src.core.experiment import results_root_for
    out = (
        Path(args.output) if args.output
        else results_root_for(data_root, args.paper, args.replication)
    )
    save_scaling_report(report, out)


if __name__ == "__main__":
    main()
