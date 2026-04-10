"""Tests for config loading and validation.

Verifies that YAML configs parse correctly, all required fields are present,
dependency graphs are valid, and the active paper file works.
"""

from __future__ import annotations

import pytest

from src.core.config_loader import (
    PaperConfig,
    get_active_paper,
    get_models_for_tier,
    load_model_config,
    load_paper_config,
    load_stimuli_config,
)


# ---------------------------------------------------------------------------
# Paper config
# ---------------------------------------------------------------------------

class TestLoadPaperConfig:
    """Test loading and validation of paper configs."""

    def test_load_paper_config(self) -> None:
        """Load the emotions paper config and verify all top-level fields."""
        paper = load_paper_config("emotions")

        assert isinstance(paper, PaperConfig)
        assert paper.id == "emotions"
        assert "Emotion" in paper.title or "emotion" in paper.title
        assert paper.model_variant == "instruct"
        assert paper.original_model  # non-empty
        assert paper.authors  # non-empty
        assert paper.url.startswith("http")
        assert len(paper.techniques_required) >= 2
        assert "probes" in paper.techniques_required
        assert len(paper.claims) >= 5

    def test_paper_config_missing_raises(self) -> None:
        """Non-existent paper_id should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_paper_config("nonexistent_paper_xyz")


# ---------------------------------------------------------------------------
# Model config
# ---------------------------------------------------------------------------

class TestLoadModelConfig:
    """Test loading the model matrix."""

    def test_load_model_config(self) -> None:
        """Load models.yaml and verify all 9 models are present."""
        config = load_model_config()

        assert "models" in config
        models = config["models"]
        assert len(models) == 9

        expected_keys = {
            "llama_1b", "llama_8b", "llama_70b",
            "qwen_1_5b", "qwen_7b", "qwen_72b",
            "gemma_2b", "gemma_9b", "gemma_27b",
        }
        assert set(models.keys()) == expected_keys

    def test_model_required_fields(self) -> None:
        """Every model entry should have the required metadata fields."""
        config = load_model_config()
        required_fields = {
            "hf_id", "family", "size_tier", "params_b",
            "layers", "hidden_dim", "num_heads",
            "two_thirds_layer", "loader", "dtype",
        }
        for key, model in config["models"].items():
            missing = required_fields - set(model.keys())
            assert not missing, f"Model {key!r} is missing fields: {missing}"

    def test_models_for_tier_small(self) -> None:
        """get_models_for_tier('small') should return 3 models."""
        small = get_models_for_tier("small")
        assert len(small) == 3
        families = {m["family"] for m in small}
        assert families == {"llama", "qwen", "gemma"}

    def test_models_for_tier_empty(self) -> None:
        """A non-existent tier should return an empty list."""
        result = get_models_for_tier("gigantic")
        assert result == []


# ---------------------------------------------------------------------------
# Stimuli config
# ---------------------------------------------------------------------------

class TestLoadStimuliConfig:
    """Test loading stimuli configs."""

    def test_load_stimuli_config(self) -> None:
        """Load the emotions stimuli config and check key stimulus sets."""
        stim = load_stimuli_config("emotions")

        assert "stimulus_sets" in stim
        sets = stim["stimulus_sets"]
        assert "training_stories" in sets
        assert "implicit_scenarios" in sets
        assert "parametric_dosage" in sets

    def test_training_stories_structure(self) -> None:
        """Training stories set should have generation_prompt and concepts."""
        stim = load_stimuli_config("emotions")
        ts = stim["stimulus_sets"]["training_stories"]
        assert ts["type"] == "generated"
        assert "generation_prompt" in ts
        assert len(ts["concepts"]) == 15
        # Aligned to actual stimuli on disk (25 hand-crafted per emotion).
        # Earlier value of 50 was aspirational and didn't match what was loaded.
        assert ts["per_concept"] == 25

    def test_stimuli_config_missing_raises(self) -> None:
        """Non-existent paper should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_stimuli_config("nonexistent_paper_xyz")


# ---------------------------------------------------------------------------
# Claim config fields
# ---------------------------------------------------------------------------

class TestClaimConfigFields:
    """Verify all claims in the emotions paper have required fields."""

    def test_claim_config_fields(self) -> None:
        """Each claim must have id, description, experiment_type, metric, threshold."""
        paper = load_paper_config("emotions")

        for claim in paper.claims:
            assert claim.paper_id == "emotions"
            assert claim.claim_id, "claim_id must be non-empty"
            assert claim.description, "description must be non-empty"
            assert claim.experiment_type, "experiment_type must be non-empty"
            assert claim.success_metric, "success_metric must be non-empty"
            assert claim.success_threshold >= 0, "success_threshold must be >= 0"

    def test_claim_experiment_types_are_known(self) -> None:
        """Every claim's experiment_type should be in the registry."""
        from src.experiments import EXPERIMENT_REGISTRY

        paper = load_paper_config("emotions")
        for claim in paper.claims:
            assert claim.experiment_type in EXPERIMENT_REGISTRY, (
                f"Claim {claim.claim_id!r} has unknown experiment_type "
                f"{claim.experiment_type!r}"
            )


# ---------------------------------------------------------------------------
# Dependency graph
# ---------------------------------------------------------------------------

class TestDependencyGraph:
    """Verify that depends_on references point to valid claim IDs."""

    def test_dependency_graph(self) -> None:
        """Every depends_on should reference an existing claim_id in the paper."""
        paper = load_paper_config("emotions")
        claim_ids = {c.claim_id for c in paper.claims}

        for claim in paper.claims:
            if claim.depends_on is not None:
                assert claim.depends_on in claim_ids, (
                    f"Claim {claim.claim_id!r} depends_on "
                    f"{claim.depends_on!r} which is not a valid claim_id. "
                    f"Valid IDs: {claim_ids}"
                )

    def test_no_self_dependency(self) -> None:
        """No claim should depend on itself."""
        paper = load_paper_config("emotions")
        for claim in paper.claims:
            if claim.depends_on is not None:
                assert claim.depends_on != claim.claim_id, (
                    f"Claim {claim.claim_id!r} depends on itself"
                )


# ---------------------------------------------------------------------------
# Active paper
# ---------------------------------------------------------------------------

class TestActivePaper:
    """Test active_paper.txt reading."""

    def test_active_paper(self) -> None:
        """active_paper.txt should return a non-empty string."""
        paper_id = get_active_paper()
        assert isinstance(paper_id, str)
        assert len(paper_id) > 0
        # Should match one of the papers in config/papers/
        assert paper_id == "emotions"  # currently the only paper configured
