"""Compare results across the model matrix.

Loads results from results/{paper_id}/{model_key}/ for all models,
builds comparison tables, and identifies universal vs. model-specific findings.

Usage:
    from src.analysis.cross_model import compare_across_models, print_comparison_table

    report = compare_across_models("emotions", data_root)
    print(print_comparison_table(report))
    save_comparison(report, output_dir)

CLI:
    python -m src.analysis.cross_model --paper emotions
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.core.claim import ExperimentResult
from src.core.config_loader import load_model_config, load_paper_config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class CrossModelReport:
    """Aggregated cross-model comparison for a single paper.

    Attributes:
        paper_id: Identifier for the paper.
        claim_results: Nested dict -- claim_id -> model_key -> ExperimentResult.
        summary_table: DataFrame with rows=models, cols=claims, values=primary metric.
        universality: Per-claim classification of replication pattern.
            Values: "universal", "family_specific", "scale_dependent", "null".
        model_metadata: Per-model metadata (family, size_tier, params_b).
    """

    paper_id: str
    claim_results: dict[str, dict[str, ExperimentResult]]
    summary_table: pd.DataFrame
    universality: dict[str, str]
    model_metadata: dict[str, dict[str, Any]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def _load_result(result_dir: Path) -> ExperimentResult | None:
    """Try to load a result.json from a claim directory. Returns None on failure."""
    result_path = result_dir / "result.json"
    if not result_path.exists():
        return None
    try:
        return ExperimentResult.load(result_path)
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        logger.warning("Failed to load %s: %s", result_path, exc)
        return None


def _classify_universality(
    results: dict[str, ExperimentResult],
    success_metric: str,
    success_threshold: float,
    model_metadata: dict[str, dict[str, Any]],
) -> str:
    """Classify a claim's replication pattern across models.

    Categories:
        universal        -- passes on ALL models with results.
        family_specific  -- passes on all models in at least one family,
                            but fails in at least one other family.
        scale_dependent  -- passes only on medium/large but not small,
                            or vice-versa (within at least one family).
        null             -- fails on all models.

    Args:
        results: model_key -> ExperimentResult for one claim.
        success_metric: Which metric key to evaluate.
        success_threshold: Minimum value for success.
        model_metadata: model_key -> {family, size_tier, params_b, ...}.

    Returns:
        One of the category strings above.
    """
    if not results:
        return "null"

    passes: dict[str, bool] = {}
    for mk, res in results.items():
        val = res.metrics.get(success_metric)
        if val is None:
            passes[mk] = False
        else:
            try:
                passes[mk] = float(val) >= success_threshold
            except (TypeError, ValueError):
                passes[mk] = False

    if all(passes.values()):
        return "universal"
    if not any(passes.values()):
        return "null"

    # Check family-specific: group by family, see if any family is all-pass
    family_groups: dict[str, list[bool]] = {}
    for mk, passed in passes.items():
        fam = model_metadata.get(mk, {}).get("family", "unknown")
        family_groups.setdefault(fam, []).append(passed)

    any_family_all_pass = any(all(vals) for vals in family_groups.values())
    any_family_all_fail = any(not any(vals) for vals in family_groups.values())

    if any_family_all_pass and any_family_all_fail:
        return "family_specific"

    # Check scale-dependent: within a family, passes at some tiers but not others
    tier_order = {"small": 0, "medium": 1, "large": 2}
    for fam, vals in family_groups.items():
        family_models = [
            (mk, passes[mk])
            for mk in passes
            if model_metadata.get(mk, {}).get("family") == fam
        ]
        tiers_that_pass = {
            model_metadata.get(mk, {}).get("size_tier")
            for mk, p in family_models
            if p
        }
        tiers_that_fail = {
            model_metadata.get(mk, {}).get("size_tier")
            for mk, p in family_models
            if not p
        }
        if tiers_that_pass and tiers_that_fail:
            return "scale_dependent"

    # Mixed pattern that doesn't fit neatly -- default to scale_dependent
    return "scale_dependent"


def compare_across_models(
    paper_id: str,
    data_root: Path,
    replication_id: str | None = None,
) -> CrossModelReport:
    """Build a cross-model comparison report for a paper.

    Scans ``results/{paper_id}/{model_key}/`` (or
    ``results/{paper_id}/{replication_id}/{model_key}/`` when a
    replication is given) for all available model results, collects
    metric values for each claim, and classifies replication patterns.

    Args:
        paper_id: Paper identifier (e.g., 'emotions').
        data_root: Root data directory (from get_data_root()).
        replication_id: Optional replication identifier. When set, only
            results under that replication's namespace are included.

    Returns:
        CrossModelReport with all available results and summary.
    """
    paper_config = load_paper_config(paper_id, replication_id=replication_id)
    model_config = load_model_config()
    models = model_config.get("models", {})

    from src.core.experiment import results_root_for
    results_root = results_root_for(
        data_root, paper_id, paper_config.replication_id,
    )

    # Build model metadata lookup
    model_meta: dict[str, dict[str, Any]] = {}
    for mk, minfo in models.items():
        model_meta[mk] = {
            "family": minfo.get("family", "unknown"),
            "size_tier": minfo.get("size_tier", "unknown"),
            "params_b": minfo.get("params_b", 0.0),
            "hf_id": minfo.get("hf_id", ""),
        }

    # Discover which model_keys have result directories
    available_models: list[str] = []
    if results_root.is_dir():
        for child in sorted(results_root.iterdir()):
            if child.is_dir() and child.name in models:
                available_models.append(child.name)

    if not available_models:
        logger.warning("No model results found in %s", results_root)

    # Collect results: claim_id -> model_key -> ExperimentResult
    claim_results: dict[str, dict[str, ExperimentResult]] = {}
    claim_metrics: dict[str, str] = {}  # claim_id -> success_metric

    for claim in paper_config.claims:
        claim_results[claim.claim_id] = {}
        claim_metrics[claim.claim_id] = claim.success_metric

        for mk in available_models:
            claim_dir = results_root / mk / claim.claim_id
            result = _load_result(claim_dir)
            if result is not None:
                claim_results[claim.claim_id][mk] = result

    # Build summary table: rows=models, cols=claims, values=primary metric
    claim_ids = [c.claim_id for c in paper_config.claims]
    table_data: dict[str, dict[str, float | None]] = {}

    for mk in available_models:
        row: dict[str, float | None] = {}
        for cid in claim_ids:
            res = claim_results.get(cid, {}).get(mk)
            if res is not None:
                metric_key = claim_metrics[cid]
                val = res.metrics.get(metric_key)
                try:
                    row[cid] = float(val) if val is not None else None
                except (TypeError, ValueError):
                    row[cid] = None
            else:
                row[cid] = None
        table_data[mk] = row

    summary_table = pd.DataFrame.from_dict(table_data, orient="index")
    summary_table.index.name = "model"
    if claim_ids:
        summary_table = summary_table.reindex(columns=claim_ids)

    # Classify universality per claim
    universality: dict[str, str] = {}
    for claim in paper_config.claims:
        cid = claim.claim_id
        universality[cid] = _classify_universality(
            claim_results.get(cid, {}),
            claim.success_metric,
            claim.success_threshold,
            model_meta,
        )

    return CrossModelReport(
        paper_id=paper_id,
        claim_results=claim_results,
        summary_table=summary_table,
        universality=universality,
        model_metadata=model_meta,
    )


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def print_comparison_table(report: CrossModelReport) -> str:
    """Format the cross-model comparison as an ASCII table.

    Each cell shows the metric value with a PASS/FAIL tag.

    Args:
        report: CrossModelReport from compare_across_models().

    Returns:
        Formatted multi-line string.
    """
    if report.summary_table.empty:
        return f"[{report.paper_id}] No results to compare."

    paper_config = load_paper_config(report.paper_id)
    thresholds = {c.claim_id: c.success_threshold for c in paper_config.claims}

    lines: list[str] = []
    lines.append(f"Cross-model comparison: {report.paper_id}")
    lines.append("=" * 80)

    # Header
    claim_ids = list(report.summary_table.columns)
    header = f"{'Model':<16}"
    for cid in claim_ids:
        header += f" | {cid:<16}"
    lines.append(header)
    lines.append("-" * len(header))

    # Rows
    for model_key in report.summary_table.index:
        meta = report.model_metadata.get(model_key, {})
        label = f"{model_key:<16}"
        for cid in claim_ids:
            val = report.summary_table.at[model_key, cid]
            thresh = thresholds.get(cid, 0.0)
            if val is None or (isinstance(val, float) and np.isnan(val)):
                cell = "  --  "
            else:
                tag = "PASS" if val >= thresh else "FAIL"
                cell = f"{val:.3f} {tag}"
            label += f" | {cell:<16}"
        lines.append(label)

    lines.append("-" * len(header))

    # Universality summary
    lines.append("")
    lines.append("Universality:")
    for cid in claim_ids:
        pattern = report.universality.get(cid, "unknown")
        lines.append(f"  {cid}: {pattern}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_comparison(report: CrossModelReport, output_dir: Path) -> None:
    """Save the comparison report as JSON and CSV.

    Writes:
        output_dir/cross_model_summary.csv  -- the summary table
        output_dir/cross_model_report.json  -- full report (claim results + universality)

    Args:
        report: CrossModelReport to save.
        output_dir: Directory for output files (created if needed).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # CSV: summary table
    csv_path = output_dir / "cross_model_summary.csv"
    report.summary_table.to_csv(csv_path)
    logger.info("Saved summary CSV: %s", csv_path)

    # JSON: full report
    json_data: dict[str, Any] = {
        "paper_id": report.paper_id,
        "universality": report.universality,
        "model_metadata": report.model_metadata,
        "claim_results": {},
    }

    for cid, model_results in report.claim_results.items():
        json_data["claim_results"][cid] = {}
        for mk, result in model_results.items():
            json_data["claim_results"][cid][mk] = {
                "metrics": result.metrics,
                "success": result.success,
                "metadata": result.metadata,
            }

    json_path = output_dir / "cross_model_report.json"
    tmp_path = json_path.with_suffix(".tmp")
    with open(tmp_path, "w") as f:
        json.dump(json_data, f, indent=2, default=str)
    tmp_path.rename(json_path)
    logger.info("Saved report JSON: %s", json_path)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI: python -m src.analysis.cross_model --paper <paper_id>"""
    import argparse

    from src.utils.env import get_data_root

    parser = argparse.ArgumentParser(description="Cross-model comparison")
    parser.add_argument("--paper", required=True, help="Paper ID")
    parser.add_argument(
        "--replication", default=None,
        help="Replication identifier (e.g. emotions-zachgoldfine44-6models). "
             "When set, only that replication's results are compared.",
    )
    parser.add_argument("--output", default=None, help="Output directory override")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    data_root = get_data_root()
    report = compare_across_models(
        args.paper, data_root, replication_id=args.replication,
    )
    print(print_comparison_table(report))

    from src.core.experiment import results_root_for
    out = (
        Path(args.output) if args.output
        else results_root_for(data_root, args.paper, args.replication)
    )
    save_comparison(report, out)


if __name__ == "__main__":
    main()
