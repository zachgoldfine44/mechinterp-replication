"""Tests for ``src/techniques/sae.py``.

These tests are fully self-contained: no models, no SAELens installation, no
network access.  Everything runs on synthetic tensors so the suite stays fast.
"""

from __future__ import annotations

import pytest
import torch

from src.techniques.sae import (
    SimpleSAE,
    extract_sae_features,
    feature_dictionary_summary,
    load_pretrained_sae,
    top_features_for_concept,
    train_simple_sae,
)


# ---------------------------------------------------------------------------
# SimpleSAE forward / loss tests
# ---------------------------------------------------------------------------

def test_simple_sae_forward_shapes() -> None:
    """Forward pass on random input returns correctly shaped tensors."""
    torch.manual_seed(0)
    d_model, d_features = 32, 128
    sae = SimpleSAE(d_model=d_model, d_features=d_features)

    x = torch.randn(16, d_model)
    x_hat, features = sae(x)

    assert x_hat.shape == (16, d_model)
    assert features.shape == (16, d_features)


def test_simple_sae_encode_nonneg() -> None:
    """ReLU encoder should produce non-negative feature activations."""
    torch.manual_seed(1)
    sae = SimpleSAE(d_model=16, d_features=64)
    x = torch.randn(32, 16) * 5.0  # large magnitude so some pre-ReLU vals are negative
    features = sae.encode(x)

    assert (features >= 0).all(), "SimpleSAE.encode must return non-negative features"


def test_simple_sae_loss_components() -> None:
    """Loss dict should expose the documented keys and all be tensors."""
    torch.manual_seed(2)
    sae = SimpleSAE(d_model=8, d_features=32)
    x = torch.randn(10, 8)

    losses = sae.loss(x)

    assert set(losses.keys()) == {"loss", "recon", "l1", "n_active"}
    for k, v in losses.items():
        assert isinstance(v, torch.Tensor), f"{k} must be a tensor"
        assert v.ndim == 0, f"{k} must be a scalar tensor"

    # Sanity: total loss equals recon + l1_coefficient * l1
    expected = losses["recon"] + sae.l1_coefficient * losses["l1"]
    assert torch.allclose(losses["loss"], expected, atol=1e-6)


def test_simple_sae_tied_weights_roundtrip() -> None:
    """Tied-weight variant should forward without error and share W."""
    torch.manual_seed(3)
    sae = SimpleSAE(d_model=8, d_features=16, tied_weights=True)
    assert sae.W_dec is None  # tied: no separate decoder matrix

    x = torch.randn(4, 8)
    x_hat, features = sae(x)
    assert x_hat.shape == (4, 8)
    assert features.shape == (4, 16)


# ---------------------------------------------------------------------------
# train_simple_sae
# ---------------------------------------------------------------------------

def test_train_simple_sae_runs() -> None:
    """Five epochs of training on synthetic data should reduce loss."""
    torch.manual_seed(4)
    activations = torch.randn(64, 16)

    sae, history = train_simple_sae(
        activations,
        d_features=32,
        n_epochs=5,
        batch_size=16,
        lr=1e-2,
        l1_coefficient=1e-3,
        seed=4,
    )

    assert isinstance(sae, SimpleSAE)
    assert set(history.keys()) == {"loss", "recon", "l1", "n_active"}
    for v in history.values():
        assert len(v) == 5

    # Loss should generally decrease across 5 epochs on stationary data.
    assert history["loss"][-1] < history["loss"][0], (
        f"Loss did not decrease: start={history['loss'][0]:.4f} "
        f"end={history['loss'][-1]:.4f}"
    )


def test_train_simple_sae_rejects_non_2d() -> None:
    """Training should reject non-2D activation tensors."""
    bad = torch.randn(4, 8, 16)
    with pytest.raises(ValueError, match="2-D"):
        train_simple_sae(bad, d_features=16, n_epochs=1)


# ---------------------------------------------------------------------------
# extract_sae_features
# ---------------------------------------------------------------------------

def test_extract_sae_features_shape() -> None:
    """After training, extracted features should be (n_samples, d_features)."""
    torch.manual_seed(5)
    activations = torch.randn(40, 16)
    sae, _ = train_simple_sae(activations, d_features=32, n_epochs=3, batch_size=8, seed=5)

    features = extract_sae_features(sae, activations)

    assert features.shape == (40, 32)
    assert (features >= 0).all()


def test_extract_sae_features_top_k() -> None:
    """return_top_k should zero out everything except the top-k features."""
    torch.manual_seed(6)
    activations = torch.randn(20, 16)
    sae, _ = train_simple_sae(activations, d_features=32, n_epochs=3, batch_size=8, seed=6)

    features = extract_sae_features(sae, activations, return_top_k=4)

    # Each row should have at most 4 non-zero entries
    nonzero_per_row = (features > 0).sum(dim=-1)
    assert (nonzero_per_row <= 4).all()


def test_extract_sae_features_rejects_bad_top_k() -> None:
    """return_top_k must be positive if provided."""
    sae = SimpleSAE(d_model=8, d_features=16)
    acts = torch.randn(4, 8)
    with pytest.raises(ValueError, match="positive"):
        extract_sae_features(sae, acts, return_top_k=0)


# ---------------------------------------------------------------------------
# top_features_for_concept
# ---------------------------------------------------------------------------

def _make_structured_features() -> tuple[torch.Tensor, list[str]]:
    """Synthetic features where feature 0 fires for 'happy' and nothing else."""
    torch.manual_seed(7)
    n_per_concept = 10
    d_features = 20
    labels: list[str] = []
    rows: list[torch.Tensor] = []

    for concept in ("happy", "sad", "angry"):
        base = torch.rand(n_per_concept, d_features) * 0.01  # tiny noise floor
        if concept == "happy":
            base[:, 0] = 5.0  # distinctive feature
        rows.append(base)
        labels.extend([concept] * n_per_concept)

    features = torch.cat(rows, dim=0)
    return features, labels


def test_top_features_for_concept_returns_k() -> None:
    """Top-k for a known-structure dataset should return exactly k tuples."""
    features, labels = _make_structured_features()

    top = top_features_for_concept(features, labels, "happy", top_k=5)

    assert isinstance(top, list)
    assert len(top) == 5
    for idx, score in top:
        assert isinstance(idx, int)
        assert isinstance(score, float)

    # The distinctive feature at index 0 should be ranked #1
    assert top[0][0] == 0


def test_top_features_for_concept_log_odds() -> None:
    """log_odds method should also surface the distinctive feature first."""
    features, labels = _make_structured_features()

    top = top_features_for_concept(features, labels, "happy", top_k=3, method="log_odds")

    assert len(top) == 3
    assert top[0][0] == 0


def test_top_features_for_concept_unknown_method() -> None:
    """Unknown methods should raise ValueError."""
    features, labels = _make_structured_features()

    with pytest.raises(ValueError, match="Unknown method"):
        top_features_for_concept(features, labels, "happy", top_k=3, method="bogus")  # type: ignore[arg-type]


def test_top_features_for_concept_missing_concept() -> None:
    """Asking for a concept that isn't in the labels should raise."""
    features, labels = _make_structured_features()

    with pytest.raises(ValueError, match="not present"):
        top_features_for_concept(features, labels, "nonexistent", top_k=3)


# ---------------------------------------------------------------------------
# load_pretrained_sae (SAELens fallback behaviour)
# ---------------------------------------------------------------------------

def test_load_pretrained_sae_raises_without_saelens() -> None:
    """In an env without sae_lens, load_pretrained_sae should raise ImportError.

    This test assumes sae_lens is NOT installed (the default for this harness).
    If a developer has installed it locally the test is skipped.
    """
    try:
        import sae_lens  # noqa: F401
        pytest.skip("sae_lens is installed; skipping the fallback-error test")
    except ImportError:
        pass

    with pytest.raises(ImportError, match="sae_lens"):
        load_pretrained_sae("any-release", "any-sae-id")


# ---------------------------------------------------------------------------
# feature_dictionary_summary
# ---------------------------------------------------------------------------

def test_feature_dictionary_summary_keys() -> None:
    """Summary dict should expose all documented keys with correct types."""
    features = torch.rand(10, 32)
    # Force a few entries to zero so the stats aren't degenerate
    features[:, 0] = 0.0

    summary = feature_dictionary_summary(features)

    expected_keys = {
        "n_features",
        "n_samples",
        "mean_active_features_per_sample",
        "l0_per_sample",
        "fraction_of_dead_features",
        "mean_feature_magnitude",
    }
    assert set(summary.keys()) == expected_keys

    assert summary["n_features"] == 32
    assert summary["n_samples"] == 10
    assert summary["mean_active_features_per_sample"] == summary["l0_per_sample"]


def test_simple_sae_dead_features_detection() -> None:
    """A feature that is 0 for all samples should be flagged as dead."""
    features = torch.rand(8, 16)
    features[:, 3] = 0.0  # feature 3 is dead
    features[:, 7] = 0.0  # feature 7 is dead

    summary = feature_dictionary_summary(features)

    # 2 dead features out of 16 -> fraction = 0.125
    assert summary["fraction_of_dead_features"] == pytest.approx(2 / 16)


def test_feature_dictionary_summary_all_dead() -> None:
    """If every feature is zero, fraction_dead should be 1.0."""
    features = torch.zeros(5, 10)
    summary = feature_dictionary_summary(features)
    assert summary["fraction_of_dead_features"] == 1.0
    assert summary["mean_active_features_per_sample"] == 0.0
    assert summary["mean_feature_magnitude"] == 0.0
