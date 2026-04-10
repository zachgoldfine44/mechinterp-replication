"""Claim configuration and experiment result dataclasses.

ClaimConfig represents a single falsifiable claim from a paper, paired with
the experiment type and success criteria needed to test it.

ExperimentResult holds the output of running one experiment on one model,
with atomic save/load for checkpoint resilience.

Usage:
    from src.core.claim import ClaimConfig, ExperimentResult

    claim = ClaimConfig(
        paper_id="emotions",
        claim_id="claim_1",
        description="Linear probes classify 12 emotions above chance",
        experiment_type="probe_classification",
        params={"concept_set": ["joy", "anger"], "probe_type": "logistic_regression"},
        success_metric="probe_accuracy",
        success_threshold=0.70,
    )

    result = ExperimentResult(
        claim_id="claim_1",
        model_key="llama_1b",
        paper_id="emotions",
        metrics={"probe_accuracy": 0.73, "per_concept": {"joy": 0.81, "anger": 0.65}},
    )
    result.save(Path("results/emotions/llama_1b/claim_1/result.json"))
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ClaimConfig:
    """A single testable claim extracted from a paper.

    Attributes:
        paper_id: Identifier for the paper (e.g., 'emotions').
        claim_id: Unique claim identifier within the paper (e.g., 'claim_1').
        description: Human-readable description of the claim.
        experiment_type: Which generic experiment class to use
            (e.g., 'probe_classification', 'causal_steering').
        params: Experiment-specific parameters passed to the experiment class.
        success_metric: Key into ExperimentResult.metrics to evaluate
            (e.g., 'probe_accuracy').
        success_threshold: Minimum value of success_metric for the claim
            to be considered replicated.
        depends_on: Optional claim_id that must run first (e.g., probes
            before steering).
        notes: Free-text notes for interpreting results or known caveats.
        paper_section: Optional pointer to the section of the paper this
            claim comes from (e.g., "Section 3.2", "Figure 4", "Appendix B").
            Used by the critique agents to verify replicated methodology
            against the paper. STRONGLY ENCOURAGED — every claim should
            cite where in the paper it came from.
    """

    paper_id: str
    claim_id: str
    description: str
    experiment_type: str
    params: dict[str, Any]
    success_metric: str
    success_threshold: float
    depends_on: str | None = None
    notes: str = ""
    paper_section: str = ""


@dataclass
class ExperimentResult:
    """Output of running one experiment on one model.

    Attributes:
        claim_id: Which claim this result addresses.
        model_key: Model identifier (e.g., 'llama_1b').
        paper_id: Paper identifier.
        metrics: Dict of metric_name -> value. Must include the key
            specified by the corresponding ClaimConfig.success_metric.
        success: Whether the claim's success criterion was met.
            None means not yet evaluated.
        metadata: Additional info -- hyperparameters, timing, seeds,
            layer selection, etc. Stored but not used for evaluation.
    """

    claim_id: str
    model_key: str
    paper_id: str
    metrics: dict[str, Any]
    success: bool | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def save(self, path: Path) -> None:
        """Atomically save result to JSON.

        Writes to a .tmp file first, then renames to avoid corruption
        if the process is killed mid-write.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(asdict(self), f, indent=2, default=str)
        tmp.rename(path)

    @classmethod
    def load(cls, path: Path) -> ExperimentResult:
        """Load a result from JSON."""
        with open(path) as f:
            data = json.load(f)
        return cls(**data)
