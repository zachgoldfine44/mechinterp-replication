"""Quick sanity tests -- these are the ones that run with --fast.

Verify that all modules import, environment detection works, project root
is found, model registry loads, and configs parse. No model downloads,
no heavy computation.
"""

from __future__ import annotations

from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

class TestImports:
    """Verify all key modules import without error."""

    def test_import_core(self) -> None:
        from src.core.claim import ClaimConfig, ExperimentResult  # noqa: F401
        from src.core.experiment import Experiment  # noqa: F401
        from src.core.config_loader import load_paper_config  # noqa: F401

    def test_import_models(self) -> None:
        from src.models.registry import ModelRegistry, ModelInfo  # noqa: F401

    def test_import_techniques(self) -> None:
        from src.techniques.probes import train_probe, ProbeResult  # noqa: F401
        from src.techniques.contrastive import (  # noqa: F401
            compute_concept_vectors,
            compute_similarity_matrix,
            compute_pca,
        )
        from src.techniques.steering import (  # noqa: F401
            get_steering_layers,
            create_control_vector,
        )

    def test_import_experiments(self) -> None:
        from src.experiments import (  # noqa: F401
            EXPERIMENT_REGISTRY,
            get_experiment_class,
        )
        from src.experiments.probe_classification import ProbeClassificationExperiment  # noqa: F401
        from src.experiments.generalization_test import GeneralizationTestExperiment  # noqa: F401
        from src.experiments.representation_geometry import RepresentationGeometryExperiment  # noqa: F401
        from src.experiments.parametric_scaling import ParametricScalingExperiment  # noqa: F401
        from src.experiments.causal_steering import CausalSteeringExperiment  # noqa: F401

    def test_import_utils(self) -> None:
        from src.utils.env import (  # noqa: F401
            detect_environment,
            get_data_root,
            get_device,
            get_project_root,
        )
        from src.utils.metrics import (  # noqa: F401
            cosine_similarity,
            cosine_similarity_matrix,
            accuracy,
            bootstrap_ci,
            effect_size_cohens_d,
            diagonal_dominance,
        )


# ---------------------------------------------------------------------------
# Environment detection
# ---------------------------------------------------------------------------

class TestEnvDetection:
    """Verify environment detection returns valid values."""

    def test_detect_environment(self) -> None:
        from src.utils.env import detect_environment

        env = detect_environment()
        assert env in ("colab", "macbook", "other"), (
            f"detect_environment() returned unexpected value: {env!r}"
        )

    def test_get_device(self) -> None:
        from src.utils.env import get_device

        device = get_device()
        assert device in ("cuda", "mps", "cpu"), (
            f"get_device() returned unexpected value: {device!r}"
        )

    def test_get_data_root_returns_path(self) -> None:
        from src.utils.env import get_data_root

        root = get_data_root()
        assert isinstance(root, Path)


# ---------------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------------

class TestProjectRoot:
    """Verify get_project_root() finds the repo root."""

    def test_project_root_has_claude_md(self) -> None:
        from src.utils.env import get_project_root

        root = get_project_root()
        assert (root / "CLAUDE.md").exists(), (
            f"get_project_root() returned {root} but CLAUDE.md not found there"
        )

    def test_project_root_has_config(self) -> None:
        from src.utils.env import get_project_root

        root = get_project_root()
        assert (root / "config" / "models.yaml").exists()
        assert (root / "config" / "active_paper.txt").exists()


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

class TestModelRegistry:
    """Verify ModelRegistry loads and has expected models."""

    def test_registry_loads(self) -> None:
        from src.models.registry import ModelRegistry

        reg = ModelRegistry()
        assert len(reg) == 9

    def test_registry_all_families(self) -> None:
        from src.models.registry import ModelRegistry

        reg = ModelRegistry()
        families = {m.family for m in reg.all_models()}
        assert families == {"llama", "qwen", "gemma"}

    def test_registry_all_tiers(self) -> None:
        from src.models.registry import ModelRegistry

        reg = ModelRegistry()
        tiers = {m.size_tier for m in reg.all_models()}
        assert tiers == {"small", "medium", "large"}

    def test_registry_get_known_model(self) -> None:
        from src.models.registry import ModelRegistry

        reg = ModelRegistry()
        info = reg.get("llama_1b")
        assert info.hf_id == "meta-llama/Llama-3.2-1B-Instruct"
        assert info.family == "llama"
        assert info.size_tier == "small"
        assert info.layers == 16
        assert info.hidden_dim == 2048

    def test_registry_get_unknown_raises(self) -> None:
        from src.models.registry import ModelRegistry

        reg = ModelRegistry()
        with pytest.raises(KeyError, match="Unknown model"):
            reg.get("nonexistent_model")

    def test_registry_tier_counts(self) -> None:
        from src.models.registry import ModelRegistry

        reg = ModelRegistry()
        for tier in ("small", "medium", "large"):
            models = reg.get_tier(tier)
            assert len(models) == 3, f"Tier {tier!r} has {len(models)} models, expected 3"


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

class TestConfigLoads:
    """Verify paper and model configs load successfully."""

    def test_paper_config_loads(self) -> None:
        from src.core.config_loader import load_paper_config

        paper = load_paper_config("emotions")
        assert paper.id == "emotions"
        assert len(paper.claims) >= 5

    def test_model_config_loads(self) -> None:
        from src.core.config_loader import load_model_config

        config = load_model_config()
        assert "models" in config
        assert len(config["models"]) == 9

    def test_stimuli_config_loads(self) -> None:
        from src.core.config_loader import load_stimuli_config

        stim = load_stimuli_config("emotions")
        assert "stimulus_sets" in stim
