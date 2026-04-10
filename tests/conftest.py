"""Shared pytest fixtures and configuration for the test suite.

Provides:
- ``--fast`` flag to limit test scope (fewer concepts, samples, etc.)
- ``--integration`` flag to opt into integration tests that download a real
  tiny TransformerLens model. Integration tests are SKIPPED by default so the
  fast/dev suite stays fast.
- ``project_root`` fixture pointing to the repo directory
- ``data_root`` fixture using a temporary directory
- ``sample_activations`` and ``sample_labels`` for synthetic data
- ``sample_claim_config`` for a minimal ClaimConfig
- ``tiny_model`` fixture (module-scoped) that lazily loads pythia-14m for
  integration / regression tests; auto-skips if the download fails.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch


# ---------------------------------------------------------------------------
# Pytest option: --fast and --integration
# ---------------------------------------------------------------------------

def pytest_addoption(parser: Any) -> None:
    parser.addoption(
        "--fast",
        action="store_true",
        default=False,
        help="Run tests in fast mode: fewer concepts, samples, and models.",
    )
    parser.addoption(
        "--integration",
        action="store_true",
        default=False,
        help="Run integration tests (downloads pythia-14m, ~30MB).",
    )


def pytest_configure(config: Any) -> None:
    config.addinivalue_line(
        "markers",
        "integration: requires loading a real model (pythia-14m); "
        "skipped unless --integration is passed.",
    )


def pytest_collection_modifyitems(config: Any, items: list[Any]) -> None:
    """Skip integration-marked tests unless --integration was passed."""
    if config.getoption("--integration"):
        return
    skip_integration = pytest.mark.skip(reason="needs --integration to run")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip_integration)


@pytest.fixture
def fast(request: Any) -> bool:
    """Whether --fast was passed on the command line."""
    return request.config.getoption("--fast")


# ---------------------------------------------------------------------------
# Path fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def project_root() -> Path:
    """Absolute path to the project root (parent of tests/)."""
    return Path(__file__).resolve().parent.parent


@pytest.fixture
def data_root(tmp_path: Path) -> Path:
    """Temporary data root for test isolation (no Google Drive dependency)."""
    root = tmp_path / "test_data_root"
    root.mkdir()
    return root


# ---------------------------------------------------------------------------
# Synthetic data fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_activations() -> torch.Tensor:
    """Random activations: 50 samples x 2048 hidden dims.

    Seeded for determinism.
    """
    gen = torch.Generator().manual_seed(42)
    return torch.randn(50, 2048, generator=gen)


@pytest.fixture
def sample_labels() -> list[str]:
    """Labels for 5 concepts, 10 samples each (50 total)."""
    concepts = ["happy", "sad", "afraid", "angry", "calm"]
    return [c for c in concepts for _ in range(10)]


@pytest.fixture
def separable_activations() -> tuple[torch.Tensor, list[str]]:
    """Linearly separable synthetic activations for reliable probe testing.

    Creates 5 concepts with well-separated cluster centres in 64-D space
    so that a logistic regression probe should achieve > 0.8 accuracy.
    """
    rng = np.random.RandomState(42)
    n_per_concept = 20
    dim = 64
    concepts = ["happy", "sad", "afraid", "angry", "calm"]

    all_vecs: list[np.ndarray] = []
    all_labels: list[str] = []

    for i, concept in enumerate(concepts):
        centre = np.zeros(dim)
        centre[i * 10 : (i + 1) * 10] = 3.0  # strong cluster signal
        samples = centre + rng.randn(n_per_concept, dim) * 0.5
        all_vecs.append(samples)
        all_labels.extend([concept] * n_per_concept)

    X = torch.tensor(np.vstack(all_vecs), dtype=torch.float32)
    return X, all_labels


@pytest.fixture
def sample_claim_config():
    """A minimal ClaimConfig for testing."""
    from src.core.claim import ClaimConfig

    return ClaimConfig(
        paper_id="emotions",
        claim_id="test_claim",
        description="Test claim for unit tests",
        experiment_type="probe_classification",
        params={
            "concept_set": ["happy", "sad"],
            "n_stimuli_per_concept": 10,
            "probe_type": "logistic_regression",
        },
        success_metric="probe_accuracy",
        success_threshold=0.50,
    )


# ---------------------------------------------------------------------------
# Tiny real model fixture (integration / regression tests)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tiny_model():
    """Load pythia-14m via TransformerLens.

    Module-scoped so we only pay the download/load cost once per test session.
    Auto-skips the test if transformer_lens is missing or the download fails
    (e.g. no network, HuggingFace rate-limit).

    Used by both ``tests/test_integration_tiny_model.py`` and
    ``tests/test_regression_activation_extraction.py``.
    """
    pytest.importorskip("transformer_lens")
    from transformer_lens import HookedTransformer

    try:
        model = HookedTransformer.from_pretrained(
            "pythia-14m",
            device="cpu",  # CPU is fine; this is tiny
            fold_ln=False,
            center_writing_weights=False,
            center_unembed=False,
        )
    except Exception as exc:  # network errors, missing files, etc.
        pytest.skip(f"Could not load pythia-14m (likely no network): {exc}")

    model.eval()
    return model
