"""Tests for activation patching primitives.

These tests use synthetic tensors and ``unittest.mock`` — no real model
is loaded. The goal is to exercise the metric helper, the
``patch_residual_stream`` dispatch, and the ``causal_trace`` shape/control
flow without paying for a model load.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest
import torch


# ---------------------------------------------------------------------------
# compute_patch_metric
# ---------------------------------------------------------------------------

class TestComputePatchMetric:
    """Pure-function tests for the restoration metric."""

    def test_compute_patch_metric_logit_diff_perfect_recovery(self) -> None:
        """patched == clean should give effect 1.0 for logit_diff."""
        from src.techniques.patching import compute_patch_metric

        vocab = 16
        clean = torch.zeros(vocab)
        clean[5] = 10.0
        corrupted = torch.zeros(vocab)
        corrupted[5] = 2.0
        patched = clean.clone()

        eff = compute_patch_metric(
            clean_logits=clean,
            corrupted_logits=corrupted,
            patched_logits=patched,
            answer_token_id=5,
            metric="logit_diff",
        )
        assert eff == pytest.approx(1.0)

    def test_compute_patch_metric_no_recovery(self) -> None:
        """patched == corrupted should give effect 0.0 for logit_diff."""
        from src.techniques.patching import compute_patch_metric

        vocab = 16
        clean = torch.zeros(vocab)
        clean[5] = 10.0
        corrupted = torch.zeros(vocab)
        corrupted[5] = 2.0
        patched = corrupted.clone()

        eff = compute_patch_metric(
            clean_logits=clean,
            corrupted_logits=corrupted,
            patched_logits=patched,
            answer_token_id=5,
            metric="logit_diff",
        )
        assert eff == pytest.approx(0.0)

    def test_compute_patch_metric_partial(self) -> None:
        """Halfway between corrupted and clean should give effect 0.5."""
        from src.techniques.patching import compute_patch_metric

        vocab = 16
        clean = torch.zeros(vocab)
        clean[5] = 10.0
        corrupted = torch.zeros(vocab)
        corrupted[5] = 2.0
        patched = torch.zeros(vocab)
        patched[5] = 6.0  # midpoint

        eff = compute_patch_metric(
            clean_logits=clean,
            corrupted_logits=corrupted,
            patched_logits=patched,
            answer_token_id=5,
            metric="logit_diff",
        )
        assert eff == pytest.approx(0.5, abs=1e-6)

    def test_compute_patch_metric_zero_difference(self) -> None:
        """clean == corrupted should not crash; returns a finite number."""
        from src.techniques.patching import compute_patch_metric

        vocab = 16
        clean = torch.zeros(vocab)
        clean[3] = 4.0
        corrupted = clean.clone()
        patched = clean.clone()

        eff = compute_patch_metric(
            clean_logits=clean,
            corrupted_logits=corrupted,
            patched_logits=patched,
            answer_token_id=3,
            metric="logit_diff",
        )
        # Returns 0 (graceful) rather than NaN/inf.
        assert np.isfinite(eff)
        assert eff == pytest.approx(0.0)

    def test_compute_patch_metric_prob_diff_perfect_recovery(self) -> None:
        """prob_diff metric: perfect recovery should give 1.0."""
        from src.techniques.patching import compute_patch_metric

        vocab = 8
        clean = torch.zeros(vocab)
        clean[2] = 8.0
        corrupted = torch.zeros(vocab)
        corrupted[2] = 1.0
        patched = clean.clone()

        eff = compute_patch_metric(
            clean_logits=clean,
            corrupted_logits=corrupted,
            patched_logits=patched,
            answer_token_id=2,
            metric="prob_diff",
        )
        assert eff == pytest.approx(1.0, abs=1e-5)

    def test_compute_patch_metric_kl_divergence_perfect_recovery(self) -> None:
        """kl_divergence metric: identical distributions => 0 (returned as -0)."""
        from src.techniques.patching import compute_patch_metric

        vocab = 8
        clean = torch.randn(vocab)
        corrupted = torch.randn(vocab)
        patched = clean.clone()

        eff = compute_patch_metric(
            clean_logits=clean,
            corrupted_logits=corrupted,
            patched_logits=patched,
            answer_token_id=0,
            metric="kl_divergence",
        )
        assert eff == pytest.approx(0.0, abs=1e-5)

    def test_compute_patch_metric_unknown(self) -> None:
        """Unknown metric should raise ValueError."""
        from src.techniques.patching import compute_patch_metric

        clean = torch.zeros(8)
        with pytest.raises(ValueError, match="Unknown metric"):
            compute_patch_metric(
                clean_logits=clean,
                corrupted_logits=clean,
                patched_logits=clean,
                answer_token_id=0,
                metric="not_a_metric",  # type: ignore[arg-type]
            )


# ---------------------------------------------------------------------------
# patch_residual_stream — TransformerLens path with a mocked model
# ---------------------------------------------------------------------------

def _make_mock_tl_model(n_layers: int, vocab: int, seq: int) -> MagicMock:
    """Build a mock that quacks like a TL HookedTransformer."""
    model = MagicMock()
    # _is_transformerlens checks for run_with_hooks AND cfg.
    model.cfg = MagicMock()
    model.cfg.n_layers = n_layers

    # run_with_hooks should return a fake (batch, seq, vocab) logits tensor.
    def fake_run_with_hooks(tokens, fwd_hooks=None, **kwargs):
        return torch.zeros(1, seq, vocab)

    model.run_with_hooks = MagicMock(side_effect=fake_run_with_hooks)

    # run_with_cache returns (logits, cache_dict).
    def fake_run_with_cache(tokens, **kwargs):
        cache = {
            f"blocks.{i}.hook_resid_post": torch.randn(1, seq, 8)
            for i in range(n_layers)
        }
        logits = torch.zeros(1, seq, vocab)
        return logits, cache

    model.run_with_cache = MagicMock(side_effect=fake_run_with_cache)

    # Plain call (used by _run_model for the corrupted forward pass).
    model.side_effect = lambda tokens, **kwargs: torch.zeros(1, seq, vocab)
    # MagicMock instances are callable; the side_effect above wires that up.

    return model


class TestPatchResidualStreamMock:
    """Mocked-model tests for patch_residual_stream."""

    def test_patch_residual_stream_with_mock(self) -> None:
        """run_with_hooks should be called with the right hook name."""
        from src.techniques.patching import patch_residual_stream

        n_layers, vocab, seq = 4, 32, 6
        model = _make_mock_tl_model(n_layers, vocab, seq)

        # Build a clean cache that has the layer-2 resid_post entry.
        clean_cache = {
            f"blocks.{i}.hook_resid_post": torch.randn(1, seq, 8)
            for i in range(n_layers)
        }
        tokens = torch.zeros(1, seq, dtype=torch.long)

        out = patch_residual_stream(
            model,
            tokens=tokens,
            clean_cache=clean_cache,
            patch_layer=2,
            patch_position=3,
            hook_name="resid_post",
        )

        # Returned a tensor with the expected logits shape.
        assert isinstance(out, torch.Tensor)
        assert out.shape == (1, seq, vocab)

        # And run_with_hooks was invoked with the right hook name.
        assert model.run_with_hooks.call_count == 1
        _, kwargs = model.run_with_hooks.call_args
        fwd_hooks = kwargs["fwd_hooks"]
        assert len(fwd_hooks) == 1
        hook_name, _hook_fn = fwd_hooks[0]
        assert hook_name == "blocks.2.hook_resid_post"

    def test_patch_residual_stream_missing_hook_raises(self) -> None:
        """Missing hook entries should raise KeyError."""
        from src.techniques.patching import patch_residual_stream

        model = _make_mock_tl_model(n_layers=4, vocab=8, seq=4)
        tokens = torch.zeros(1, 4, dtype=torch.long)
        # Empty cache — nothing to find.
        with pytest.raises(KeyError):
            patch_residual_stream(
                model,
                tokens=tokens,
                clean_cache={},
                patch_layer=1,
                patch_position=0,
                hook_name="resid_post",
            )


# ---------------------------------------------------------------------------
# causal_trace — sweep over a (layer, position) grid
# ---------------------------------------------------------------------------

class TestCausalTraceMock:
    """Mocked-model tests for causal_trace."""

    def test_causal_trace_grid_shape(self) -> None:
        """effect_grid shape should match (n_layers, n_positions)."""
        from src.techniques.patching import causal_trace

        n_layers, vocab, seq = 4, 16, 5
        model = _make_mock_tl_model(n_layers, vocab, seq)

        clean_tokens = torch.zeros(1, seq, dtype=torch.long)
        corrupted_tokens = torch.zeros(1, seq, dtype=torch.long)

        out = causal_trace(
            model=model,
            clean_tokens=clean_tokens,
            corrupted_tokens=corrupted_tokens,
            answer_token_id=0,
            metric="logit_diff",
        )

        assert "effect_grid" in out
        assert isinstance(out["effect_grid"], np.ndarray)
        assert out["effect_grid"].shape == (n_layers, seq)

        assert "best_layer" in out
        assert "best_position" in out
        assert 0 <= out["best_layer"] < n_layers
        assert 0 <= out["best_position"] < seq

        assert out["metric"] == "logit_diff"
        assert out["layers"] == list(range(n_layers))
        assert out["positions"] == list(range(seq))

    def test_causal_trace_subset_grid_shape(self) -> None:
        """Passing explicit layers/positions should size the grid accordingly."""
        from src.techniques.patching import causal_trace

        n_layers, vocab, seq = 6, 16, 8
        model = _make_mock_tl_model(n_layers, vocab, seq)

        clean_tokens = torch.zeros(1, seq, dtype=torch.long)
        corrupted_tokens = torch.zeros(1, seq, dtype=torch.long)

        layers_subset = [1, 3, 5]
        positions_subset = [0, 2, 4, 6]

        out = causal_trace(
            model=model,
            clean_tokens=clean_tokens,
            corrupted_tokens=corrupted_tokens,
            answer_token_id=0,
            layers=layers_subset,
            positions=positions_subset,
            metric="logit_diff",
        )
        assert out["effect_grid"].shape == (
            len(layers_subset),
            len(positions_subset),
        )

    def test_causal_trace_shape_mismatch_raises(self) -> None:
        """Different-length clean/corrupted should raise ValueError."""
        from src.techniques.patching import causal_trace

        model = _make_mock_tl_model(n_layers=2, vocab=8, seq=4)
        clean_tokens = torch.zeros(1, 4, dtype=torch.long)
        corrupted_tokens = torch.zeros(1, 5, dtype=torch.long)

        with pytest.raises(ValueError, match="shape"):
            causal_trace(
                model=model,
                clean_tokens=clean_tokens,
                corrupted_tokens=corrupted_tokens,
                answer_token_id=0,
            )


# ---------------------------------------------------------------------------
# PatchResult dataclass smoke
# ---------------------------------------------------------------------------

class TestPatchResultDataclass:
    """Smoke test that the dataclass instantiates with the right fields."""

    def test_patch_result_fields(self) -> None:
        from src.techniques.patching import PatchResult

        clean = torch.zeros(8)
        corrupted = torch.zeros(8)
        patched = torch.zeros(8)

        pr = PatchResult(
            clean_logits=clean,
            corrupted_logits=corrupted,
            patched_logits=patched,
            patch_layer=3,
            patch_position=5,
            effect_size=0.42,
            metric_used="logit_diff",
            metadata={"note": "smoke"},
        )
        assert pr.patch_layer == 3
        assert pr.patch_position == 5
        assert pr.effect_size == pytest.approx(0.42)
        assert pr.metric_used == "logit_diff"
        assert pr.metadata == {"note": "smoke"}
