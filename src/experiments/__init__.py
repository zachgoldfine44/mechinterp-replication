"""Experiment registry: maps experiment_type strings to experiment classes.

Auto-discovers and registers all generic experiment types so the pipeline
can instantiate the right experiment class from a paper config's
``experiment_type`` field.

Usage:
    from src.experiments import get_experiment_class, EXPERIMENT_REGISTRY

    ExperimentClass = get_experiment_class("probe_classification")
    experiment = ExperimentClass(claim_config, "llama_1b", data_root)
    result = experiment.load_or_run(model, tokenizer, cache)

Registry:
    probe_classification     -> ProbeClassificationExperiment
    generalization_test      -> GeneralizationTestExperiment
    representation_geometry  -> RepresentationGeometryExperiment
    parametric_scaling       -> ParametricScalingExperiment
    causal_steering          -> CausalSteeringExperiment
    circuit_identification   -> CircuitIdentificationExperiment
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.experiments.causal_steering import CausalSteeringExperiment
from src.experiments.circuit_identification import CircuitIdentificationExperiment
from src.experiments.generalization_test import GeneralizationTestExperiment
from src.experiments.parametric_scaling import ParametricScalingExperiment
from src.experiments.probe_classification import ProbeClassificationExperiment
from src.experiments.representation_geometry import RepresentationGeometryExperiment

if TYPE_CHECKING:
    from src.core.experiment import Experiment

EXPERIMENT_REGISTRY: dict[str, type[Experiment]] = {
    "probe_classification": ProbeClassificationExperiment,
    "generalization_test": GeneralizationTestExperiment,
    "representation_geometry": RepresentationGeometryExperiment,
    "parametric_scaling": ParametricScalingExperiment,
    "causal_steering": CausalSteeringExperiment,
    "circuit_identification": CircuitIdentificationExperiment,
}


def get_experiment_class(experiment_type: str) -> type[Experiment]:
    """Look up an experiment class by its type string.

    Args:
        experiment_type: Key matching an entry in EXPERIMENT_REGISTRY
            (e.g., "probe_classification", "causal_steering").

    Returns:
        The experiment class (not an instance).

    Raises:
        KeyError: If experiment_type is not registered.
    """
    if experiment_type not in EXPERIMENT_REGISTRY:
        available = list(EXPERIMENT_REGISTRY.keys())
        raise KeyError(
            f"Unknown experiment type: {experiment_type!r}. "
            f"Available: {available}. "
            f"Add custom experiments to src/experiments/paper_specific/."
        )
    return EXPERIMENT_REGISTRY[experiment_type]
