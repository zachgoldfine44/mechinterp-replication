"""Pipeline orchestrator: load config -> run experiments -> analyze.

Ties together config loading, model loading, stimulus preparation,
experiment execution, and result logging into a single CLI-driven flow.

Usage:
    python -m src.core.pipeline --paper emotions --model llama_1b
    python -m src.core.pipeline --paper emotions --tier small
    python -m src.core.pipeline --paper emotions --all
    python -m src.core.pipeline --paper emotions --validate-only
    python -m src.core.pipeline --paper emotions --model llama_1b --fast

The pipeline follows the replication protocol:
  1. Parse CLI arguments
  2. Load paper config and model config
  3. Resolve which models to run
  4. For each model:
     a. Load model (reuse if same as previous)
     b. Topologically sort claims by depends_on
     c. For each claim (in dependency order):
        - Resolve experiment_type to Experiment class
        - Call experiment.load_or_run(model, tokenizer, cache)
        - Log result (pass/fail, metric value)
     d. Unload model
  5. Print summary table
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from src.core.claim import ClaimConfig, ExperimentResult
from src.core.config_loader import (
    PaperConfig,
    load_model_config,
    load_paper_config,
    load_stimuli_config,
)
from src.core.experiment import Experiment
from src.models.loader import load_model, unload_model
from src.models.registry import ModelInfo, ModelRegistry
from src.utils.env import get_data_root, get_device

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Experiment class registry
# ---------------------------------------------------------------------------

# Maps experiment_type strings (from paper_config.yaml) to Experiment subclasses.
# Populated by register_experiment() and auto-discovery in _discover_experiments().
_EXPERIMENT_REGISTRY: dict[str, type[Experiment]] = {}


def register_experiment(experiment_type: str, cls: type[Experiment]) -> None:
    """Register an Experiment subclass for a given experiment_type string.

    Args:
        experiment_type: The string used in paper_config.yaml (e.g.,
            'probe_classification').
        cls: The Experiment subclass to instantiate for this type.
    """
    _EXPERIMENT_REGISTRY[experiment_type] = cls
    logger.debug("Registered experiment type: %s -> %s", experiment_type, cls.__name__)


def get_experiment_class(experiment_type: str) -> type[Experiment]:
    """Look up the Experiment subclass for an experiment_type string.

    Falls back to auto-discovery if the registry is empty.

    Args:
        experiment_type: The string from paper_config.yaml.

    Returns:
        The registered Experiment subclass.

    Raises:
        KeyError: If no class is registered for this type.
    """
    if not _EXPERIMENT_REGISTRY:
        _discover_experiments()

    if experiment_type not in _EXPERIMENT_REGISTRY:
        available = list(_EXPERIMENT_REGISTRY.keys())
        raise KeyError(
            f"Unknown experiment type: {experiment_type!r}. "
            f"Registered types: {available}. "
            f"Register new types with register_experiment() or add them to "
            f"src/experiments/."
        )
    return _EXPERIMENT_REGISTRY[experiment_type]


def _discover_experiments() -> None:
    """Auto-discover experiment classes from src.experiments.

    Imports the experiments package and merges its EXPERIMENT_REGISTRY
    into the pipeline's local registry.
    """
    try:
        from src.experiments import EXPERIMENT_REGISTRY as pkg_registry
        _EXPERIMENT_REGISTRY.update(pkg_registry)
        logger.debug("Discovered %d experiment types: %s", len(pkg_registry), list(pkg_registry.keys()))
    except Exception as exc:
        logger.debug("Experiment auto-discovery failed (expected early in dev): %s", exc)


# ---------------------------------------------------------------------------
# Topological sort for claim dependencies
# ---------------------------------------------------------------------------

def _topo_sort_claims(claims: list[ClaimConfig]) -> list[ClaimConfig]:
    """Topologically sort claims so that dependencies run first.

    Args:
        claims: List of ClaimConfig objects, potentially with depends_on edges.

    Returns:
        Sorted list where every claim appears after its dependency.

    Raises:
        ValueError: If there is a cycle or a depends_on references a
            non-existent claim.
    """
    by_id: dict[str, ClaimConfig] = {c.claim_id: c for c in claims}

    # Validate dependency references
    for c in claims:
        if c.depends_on and c.depends_on not in by_id:
            raise ValueError(
                f"Claim '{c.claim_id}' depends_on '{c.depends_on}', "
                f"which does not exist. Available claims: {list(by_id.keys())}"
            )

    # Kahn's algorithm
    in_degree: dict[str, int] = {c.claim_id: 0 for c in claims}
    dependents: dict[str, list[str]] = defaultdict(list)

    for c in claims:
        if c.depends_on:
            in_degree[c.claim_id] += 1
            dependents[c.depends_on].append(c.claim_id)

    queue = [cid for cid, deg in in_degree.items() if deg == 0]
    sorted_ids: list[str] = []

    while queue:
        cid = queue.pop(0)
        sorted_ids.append(cid)
        for dep in dependents[cid]:
            in_degree[dep] -= 1
            if in_degree[dep] == 0:
                queue.append(dep)

    if len(sorted_ids) != len(claims):
        remaining = set(by_id.keys()) - set(sorted_ids)
        raise ValueError(
            f"Cycle detected in claim dependencies. "
            f"Claims involved: {remaining}"
        )

    return [by_id[cid] for cid in sorted_ids]


# ---------------------------------------------------------------------------
# Model resolution
# ---------------------------------------------------------------------------

def _resolve_models(
    args: argparse.Namespace,
    registry: ModelRegistry,
) -> list[ModelInfo]:
    """Determine which models to run based on CLI arguments.

    Priority: --model > --tier > --all.  If none specified, defaults to
    the first small-tier model.

    Args:
        args: Parsed CLI arguments.
        registry: Model registry loaded from config/models.yaml.

    Returns:
        List of ModelInfo objects to run experiments on.
    """
    if args.model:
        return [registry.get(args.model)]

    if args.tier:
        models = registry.get_tier(args.tier)
        if not models:
            logger.error("No models found for tier: %s", args.tier)
            sys.exit(1)
        return models

    if args.all:
        return registry.all_models()

    # Default: first small model
    small = registry.get_tier("small")
    if small:
        logger.info(
            "No --model/--tier/--all specified; defaulting to first small model: %s",
            small[0].key,
        )
        return [small[0]]

    logger.error("No models found in registry")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Validation mode
# ---------------------------------------------------------------------------

def _validate_config(
    paper_config: PaperConfig,
    stimuli_config: dict[str, Any],
    registry: ModelRegistry,
    models: list[ModelInfo],
) -> bool:
    """Validate configs without running experiments.

    Checks:
      - All claims have valid experiment_type (if experiments are registered)
      - Dependency graph is acyclic
      - Stimulus sets referenced in claims exist in stimuli_config
      - Requested models exist in the registry

    Args:
        paper_config: Loaded paper configuration.
        stimuli_config: Loaded stimuli configuration.
        registry: Model registry.
        models: Resolved list of models.

    Returns:
        True if all checks pass, False otherwise.
    """
    ok = True
    print(f"\n{'='*60}")
    print(f"Validating config for paper: {paper_config.id}")
    print(f"{'='*60}")

    # Paper metadata
    print(f"\n  Title:    {paper_config.title}")
    print(f"  Authors:  {paper_config.authors}")
    print(f"  Variant:  {paper_config.model_variant}")
    print(f"  Techniques: {paper_config.techniques_required}")
    print(f"  Claims:   {len(paper_config.claims)}")

    # Dependency sort
    try:
        sorted_claims = _topo_sort_claims(paper_config.claims)
        print(f"\n  Execution order (topological):")
        for i, c in enumerate(sorted_claims, 1):
            dep = f" (after {c.depends_on})" if c.depends_on else ""
            print(f"    {i}. {c.claim_id} [{c.experiment_type}]{dep}")
    except ValueError as exc:
        print(f"\n  ERROR: Dependency problem: {exc}")
        ok = False

    # Check experiment types against registry
    if not _EXPERIMENT_REGISTRY:
        _discover_experiments()
    if _EXPERIMENT_REGISTRY:
        for c in paper_config.claims:
            if c.experiment_type not in _EXPERIMENT_REGISTRY:
                print(f"  WARNING: No registered class for experiment_type '{c.experiment_type}'")
    else:
        print("\n  NOTE: No experiment classes registered yet (expected early in development)")

    # Stimuli config
    stim_sets = stimuli_config.get("stimulus_sets", {})
    print(f"\n  Stimulus sets defined: {list(stim_sets.keys())}")
    for c in paper_config.claims:
        stim_set = c.params.get("stimulus_set") or c.params.get("training_stimulus_set")
        if stim_set and stim_set not in stim_sets:
            print(f"  WARNING: Claim '{c.claim_id}' references stimulus set "
                  f"'{stim_set}' not found in stimuli_config.yaml")

    # Models
    print(f"\n  Models to run ({len(models)}):")
    for m in models:
        print(f"    - {m.key}: {m.hf_id} ({m.size_tier}, {m.loader})")

    print(f"\n{'='*60}")
    if ok:
        print("  Validation PASSED")
    else:
        print("  Validation FAILED (see errors above)")
    print(f"{'='*60}\n")

    return ok


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(args: argparse.Namespace) -> dict[str, dict[str, ExperimentResult]]:
    """Execute the full replication pipeline.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Nested dict: {model_key: {claim_id: ExperimentResult}}.
    """
    data_root = get_data_root()

    # Load configs
    paper_config = load_paper_config(args.paper)
    stimuli_config = load_stimuli_config(args.paper)
    registry = ModelRegistry()

    # Resolve models
    models = _resolve_models(args, registry)

    # Validate-only mode
    if args.validate_only:
        success = _validate_config(paper_config, stimuli_config, registry, models)
        sys.exit(0 if success else 1)

    # Sort claims by dependency order
    sorted_claims = _topo_sort_claims(paper_config.claims)

    # Apply --fast to claim params: limit to 2 concepts, 10 stimuli each
    if args.fast:
        logger.info("--fast mode: reducing stimuli to 2 concepts, 10 per concept")
        fast_claims = []
        for c in sorted_claims:
            new_params = dict(c.params)
            if "concept_set" in new_params:
                new_params["concept_set"] = new_params["concept_set"][:2]
            if "n_stimuli_per_concept" in new_params:
                new_params["n_stimuli_per_concept"] = min(new_params["n_stimuli_per_concept"], 10)
            fast_claims.append(ClaimConfig(
                paper_id=c.paper_id,
                claim_id=c.claim_id,
                description=c.description,
                experiment_type=c.experiment_type,
                params=new_params,
                success_metric=c.success_metric,
                success_threshold=c.success_threshold,
                depends_on=c.depends_on,
                notes=c.notes,
            ))
        sorted_claims = fast_claims

    # Run on each model
    all_results: dict[str, dict[str, ExperimentResult]] = {}
    prev_model_key: str | None = None
    current_model: Any = None
    current_tokenizer: Any = None

    try:
        for model_info in models:
            model_key = model_info.key
            logger.info(
                "\n%s\n  Starting model: %s (%s)\n%s",
                "=" * 60, model_key, model_info.hf_id, "=" * 60,
            )

            # Load model (reuse if same key)
            if model_key != prev_model_key:
                if current_model is not None:
                    unload_model(current_model)
                    current_model = None
                    current_tokenizer = None

                try:
                    current_model, current_tokenizer = load_model(model_info)
                except Exception as exc:
                    logger.error(
                        "ERROR: %s %s model_load -- %s",
                        args.paper, model_key, exc,
                    )
                    all_results[model_key] = {}
                    continue

                prev_model_key = model_key

            model_results: dict[str, ExperimentResult] = {}
            claim_results_cache: dict[str, ExperimentResult] = {}

            for claim in sorted_claims:
                # Check dependency met
                if claim.depends_on and claim.depends_on not in claim_results_cache:
                    dep_result = claim_results_cache.get(claim.depends_on)
                    if dep_result is None:
                        logger.warning(
                            "Skipping %s: dependency '%s' did not produce a result",
                            claim.claim_id, claim.depends_on,
                        )
                        continue

                # Resolve experiment class
                try:
                    exp_cls = get_experiment_class(claim.experiment_type)
                except KeyError as exc:
                    logger.error(
                        "ERROR: %s %s %s -- %s",
                        args.paper, model_key, claim.claim_id, exc,
                    )
                    continue

                # Instantiate and run
                experiment = exp_cls(
                    config=claim,
                    model_key=model_key,
                    data_root=data_root,
                )

                t0 = time.time()
                try:
                    result = experiment.load_or_run(
                        current_model, current_tokenizer, activations_cache=None,
                    )
                except Exception as exc:
                    logger.error(
                        "ERROR: %s %s %s -- %s",
                        args.paper, model_key, claim.claim_id, exc,
                    )
                    result = ExperimentResult(
                        claim_id=claim.claim_id,
                        model_key=model_key,
                        paper_id=args.paper,
                        metrics={},
                        success=False,
                        metadata={"error": str(exc)},
                    )
                elapsed = time.time() - t0

                # Log result
                metric_val = result.metrics.get(claim.success_metric, "N/A")
                status = "PASS" if result.success else "FAIL"
                logger.info(
                    "  %s | %s | %s=%s (threshold=%s) | %.1fs",
                    claim.claim_id, status, claim.success_metric,
                    metric_val, claim.success_threshold, elapsed,
                )

                model_results[claim.claim_id] = result
                claim_results_cache[claim.claim_id] = result

            all_results[model_key] = model_results

    finally:
        # Clean up last model
        if current_model is not None:
            unload_model(current_model)

    # Print summary
    _print_summary(paper_config, all_results)

    # Save summary JSON
    summary_path = data_root / "results" / args.paper / "pipeline_summary.json"
    _save_summary(summary_path, paper_config, all_results)

    return all_results


# ---------------------------------------------------------------------------
# Summary output
# ---------------------------------------------------------------------------

def _print_summary(
    paper_config: PaperConfig,
    all_results: dict[str, dict[str, ExperimentResult]],
) -> None:
    """Print a formatted summary table of all results.

    Args:
        paper_config: The paper configuration.
        all_results: Nested dict of {model_key: {claim_id: result}}.
    """
    claim_ids = [c.claim_id for c in paper_config.claims]
    model_keys = list(all_results.keys())

    if not model_keys:
        print("\nNo results to display.")
        return

    print(f"\n{'='*80}")
    print(f"  REPLICATION SUMMARY: {paper_config.title}")
    print(f"{'='*80}\n")

    # Header
    claim_col_width = max(len(cid) for cid in claim_ids) + 2
    header = f"{'Model':<15}"
    for cid in claim_ids:
        header += f" {cid:<{claim_col_width}}"
    print(header)
    print("-" * len(header))

    # Rows
    for model_key in model_keys:
        results = all_results[model_key]
        row = f"{model_key:<15}"
        for cid in claim_ids:
            if cid in results:
                r = results[cid]
                metric_val = r.metrics.get(
                    next(
                        (c.success_metric for c in paper_config.claims if c.claim_id == cid),
                        "",
                    ),
                    "?",
                )
                status = "PASS" if r.success else "FAIL"
                cell = f"{status}({metric_val})"
            else:
                cell = "SKIP"
            row += f" {cell:<{claim_col_width}}"
        print(row)

    print(f"\n{'='*80}")

    # Count passes
    total = 0
    passed = 0
    for model_results in all_results.values():
        for r in model_results.values():
            total += 1
            if r.success:
                passed += 1

    print(f"  Total: {passed}/{total} passed")
    print(f"{'='*80}\n")


def _save_summary(
    path: Path,
    paper_config: PaperConfig,
    all_results: dict[str, dict[str, ExperimentResult]],
) -> None:
    """Save a JSON summary of all pipeline results.

    Uses atomic write pattern.

    Args:
        path: Output path for the summary JSON.
        paper_config: Paper configuration.
        all_results: Nested dict of results.
    """
    summary: dict[str, Any] = {
        "paper_id": paper_config.id,
        "paper_title": paper_config.title,
        "models": {},
    }

    for model_key, results in all_results.items():
        model_summary: dict[str, Any] = {}
        for claim_id, result in results.items():
            claim_cfg = next(
                (c for c in paper_config.claims if c.claim_id == claim_id), None,
            )
            metric_name = claim_cfg.success_metric if claim_cfg else "unknown"
            model_summary[claim_id] = {
                "success": result.success,
                "metric": metric_name,
                "value": result.metrics.get(metric_name),
                "threshold": claim_cfg.success_threshold if claim_cfg else None,
            }
        summary["models"][model_key] = model_summary

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    tmp.rename(path)

    logger.info("Pipeline summary saved to %s", path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the pipeline CLI.

    Returns:
        Configured ArgumentParser.
    """
    parser = argparse.ArgumentParser(
        description="Mechinterp paper replication pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.core.pipeline --paper emotions --validate-only
  python -m src.core.pipeline --paper emotions --model llama_1b --fast
  python -m src.core.pipeline --paper emotions --tier small
  python -m src.core.pipeline --paper emotions --all
        """,
    )

    parser.add_argument(
        "--paper",
        required=True,
        help="Paper ID to replicate (must match a folder under config/papers/).",
    )

    # Model selection (mutually exclusive)
    model_group = parser.add_mutually_exclusive_group()
    model_group.add_argument(
        "--model",
        help="Run on a single model by key (e.g., llama_1b).",
    )
    model_group.add_argument(
        "--tier",
        choices=["small", "medium", "large"],
        help="Run on all models in a size tier.",
    )
    model_group.add_argument(
        "--all",
        action="store_true",
        help="Run on all models in the matrix.",
    )

    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Parse configs, validate consistency, and exit without running.",
    )

    parser.add_argument(
        "--fast",
        action="store_true",
        help="Fast mode: reduce to 2 concepts and 10 stimuli per concept.",
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose (DEBUG) logging.",
    )

    return parser


def main() -> None:
    """Entry point for ``python -m src.core.pipeline``."""
    parser = build_parser()
    args = parser.parse_args()

    # Configure logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    logger.info("Pipeline starting: paper=%s", args.paper)
    logger.info("Device: %s", get_device())
    logger.info("Data root: %s", get_data_root())

    run_pipeline(args)


if __name__ == "__main__":
    main()
