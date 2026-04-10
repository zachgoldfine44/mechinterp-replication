"""Abstract Experiment base class for all replication experiments.

Every experiment -- whether generic (probe_classification, causal_steering)
or paper-specific -- inherits from Experiment. This ensures a uniform
interface for the pipeline orchestrator.

Subclasses must implement:
    run()      -- execute the experiment and return an ExperimentResult
    evaluate() -- check whether the result meets the claim's success criterion

The base class provides:
    load_or_run() -- cache-aware wrapper that skips re-execution when a
                     result file already exists on disk.

Usage:
    class ProbeClassification(Experiment):
        def run(self, model, tokenizer, activations_cache) -> ExperimentResult:
            ...
        def evaluate(self, result: ExperimentResult) -> bool:
            metric = result.metrics.get(self.config.success_metric, 0.0)
            return metric >= self.config.success_threshold
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from src.core.claim import ClaimConfig, ExperimentResult

logger = logging.getLogger(__name__)


class Experiment(ABC):
    """Base class for all replication experiments.

    Attributes:
        config: The ClaimConfig describing what to test and how to evaluate.
        model_key: Identifier for the model being tested (e.g., 'llama_1b').
        data_root: Root directory for all persistent data (from get_data_root()).
        results_dir: Where this experiment's outputs are saved.
    """

    def __init__(self, config: ClaimConfig, model_key: str, data_root: Path) -> None:
        self.config = config
        self.model_key = model_key
        self.data_root = data_root
        self.results_dir = (
            data_root / "results" / config.paper_id / model_key / config.claim_id
        )

    @abstractmethod
    def run(self, model: Any, tokenizer: Any, activations_cache: Any) -> ExperimentResult:
        """Execute the experiment and return raw results.

        Args:
            model: The loaded model (HuggingFace, TransformerLens, or nnsight).
            tokenizer: The model's tokenizer.
            activations_cache: Shared activation cache for this model, or None.

        Returns:
            ExperimentResult with metrics populated but success possibly None
            (evaluation happens separately).
        """
        ...

    @abstractmethod
    def evaluate(self, result: ExperimentResult) -> bool:
        """Evaluate whether the result meets the claim's success criterion.

        Args:
            result: The ExperimentResult to evaluate.

        Returns:
            True if the claim is replicated (metric >= threshold), False otherwise.
        """
        ...

    def load_or_run(
        self, model: Any, tokenizer: Any, activations_cache: Any
    ) -> ExperimentResult:
        """Load cached result if available, otherwise run the experiment.

        This is the primary entry point used by the pipeline. It checks for
        an existing result.json on disk before doing any computation.

        Every freshly-computed result gets a `run_manifest` attached to its
        metadata field (git SHA, library versions, torch device, etc.), so
        downstream consumers can detect stale caches and version drift.

        Args:
            model: The loaded model.
            tokenizer: The model's tokenizer.
            activations_cache: Shared activation cache, or None.

        Returns:
            ExperimentResult with success field populated.
        """
        result_path = self.results_dir / "result.json"

        if result_path.exists():
            logger.info("Loading cached result: %s", result_path)
            result = ExperimentResult.load(result_path)
            result.success = self.evaluate(result)
            return result

        logger.info(
            "Running experiment: %s on %s", self.config.claim_id, self.model_key
        )
        result = self.run(model, tokenizer, activations_cache)
        result.success = self.evaluate(result)

        # Attach run manifest (git SHA, library versions, device, etc.)
        # so downstream consumers can detect stale caches and version drift.
        # Best-effort: never let manifest collection break a successful run.
        try:
            from src.utils.manifest import build_run_manifest, manifest_summary
            manifest = build_run_manifest()
            if not isinstance(result.metadata, dict):
                result.metadata = {}
            result.metadata["run_manifest"] = manifest
            logger.info("  manifest: %s", manifest_summary(manifest))
        except Exception as e:
            logger.debug("Could not build run manifest: %s", e)

        result.save(result_path)
        return result
