"""Tests for src.techniques.logit_lens.

These tests use synthetic data and a tiny mock TransformerLens-style model;
no real transformers are loaded.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from src.techniques.logit_lens import (
    LogitLensResult,
    project_to_logits,
    target_token_trajectory,
    top_tokens_per_layer,
    tuned_lens_project,
)


# ---------------------------------------------------------------------------
# Test fixtures: tiny mock model
# ---------------------------------------------------------------------------

class TinyMockTL:
    """Minimal TransformerLens-shaped model for unit tests.

    Exposes ``unembed`` (an nn.Linear), ``ln_final`` (an nn.LayerNorm), and a
    ``cfg`` namespace with ``n_layers``, ``d_model``, ``d_vocab``. That's
    enough surface area for ``project_to_logits`` and the
    ``_resolve_unembed_and_ln`` helper.
    """

    def __init__(self, d_model: int = 8, vocab: int = 100, n_layers: int = 4):
        torch.manual_seed(0)
        self.unembed = nn.Linear(d_model, vocab, bias=False)
        self.ln_final = nn.LayerNorm(d_model)
        self.cfg = type(
            "cfg", (), {"n_layers": n_layers, "d_model": d_model, "d_vocab": vocab}
        )()

    def parameters(self):
        yield from self.unembed.parameters()
        yield from self.ln_final.parameters()

    # Required by _is_transformerlens duck check
    def run_with_cache(self, *args, **kwargs):  # pragma: no cover - not used here
        raise NotImplementedError


def _make_synthetic_sweep(with_target: bool = True) -> dict[int, LogitLensResult]:
    """Build a 4-layer synthetic sweep with known top tokens / target stats."""
    sweep: dict[int, LogitLensResult] = {}
    for layer in range(4):
        sweep[layer] = LogitLensResult(
            layer=layer,
            position=3,
            top_tokens=[
                ("apple", 0.5 - 0.05 * layer),
                ("banana", 0.3 - 0.02 * layer),
                ("cherry", 0.1),
                ("date", 0.05),
                ("egg", 0.04),
                ("fig", 0.01),
            ],
            target_token_id=42 if with_target else None,
            target_token_prob=(0.05 + 0.1 * layer) if with_target else None,
            target_token_rank=(5 - layer) if with_target else None,
        )
    return sweep


# ---------------------------------------------------------------------------
# 1. LogitLensResult dataclass
# ---------------------------------------------------------------------------

class TestLogitLensResultDataclass:
    def test_logit_lens_result_dataclass(self) -> None:
        """Instantiate, set fields, and verify defaults."""
        r = LogitLensResult(
            layer=2,
            position=7,
            top_tokens=[("hello", 0.6), ("world", 0.2)],
        )
        assert r.layer == 2
        assert r.position == 7
        assert r.top_tokens[0] == ("hello", 0.6)
        # Optional fields default to None
        assert r.target_token_id is None
        assert r.target_token_prob is None
        assert r.target_token_rank is None
        assert r.full_logits is None

        # Set the optional target_* fields
        r.target_token_id = 1234
        r.target_token_prob = 0.42
        r.target_token_rank = 3
        assert r.target_token_id == 1234
        assert r.target_token_prob == pytest.approx(0.42)
        assert r.target_token_rank == 3


# ---------------------------------------------------------------------------
# 2. top_tokens_per_layer sort order
# ---------------------------------------------------------------------------

class TestTopTokensPerLayer:
    def test_top_tokens_per_layer_sort(self) -> None:
        """Layers must come back in ascending order regardless of input order."""
        # Build dict in NON-sorted order to make sure we sort.
        sweep_unsorted = {
            3: LogitLensResult(layer=3, position=0, top_tokens=[("c", 0.9)]),
            0: LogitLensResult(layer=0, position=0, top_tokens=[("a", 0.9)]),
            2: LogitLensResult(layer=2, position=0, top_tokens=[("b", 0.9)]),
        }
        out = top_tokens_per_layer(sweep_unsorted, top_k=5)
        layers = [layer for layer, _ in out]
        assert layers == [0, 2, 3]
        # First layer's top token should be 'a'
        assert out[0][1][0][0] == "a"

    def test_top_tokens_per_layer_top_k_limit(self) -> None:
        """top_k must clip per-layer token lists to at most k entries."""
        sweep = _make_synthetic_sweep(with_target=False)
        # Each synthetic layer has 6 tokens; clip to 3.
        out = top_tokens_per_layer(sweep, top_k=3)
        for _, tokens in out:
            assert len(tokens) == 3

        # Asking for more than exist should keep all of them.
        out_big = top_tokens_per_layer(sweep, top_k=100)
        for _, tokens in out_big:
            assert len(tokens) == 6


# ---------------------------------------------------------------------------
# 3. target_token_trajectory
# ---------------------------------------------------------------------------

class TestTargetTokenTrajectory:
    def test_target_token_trajectory_keys(self) -> None:
        """Trajectory must contain layers/probs/ranks of equal length."""
        sweep = _make_synthetic_sweep(with_target=True)
        traj = target_token_trajectory(sweep)
        assert set(traj.keys()) == {"layers", "probs", "ranks"}
        assert len(traj["layers"]) == len(sweep)
        assert len(traj["probs"]) == len(sweep)
        assert len(traj["ranks"]) == len(sweep)
        # Layers come back sorted ascending.
        assert traj["layers"] == sorted(traj["layers"])
        # Sanity-check synthetic content.
        assert traj["probs"][0] == pytest.approx(0.05)
        assert traj["ranks"][0] == 5

    def test_target_token_trajectory_no_target_raises(self) -> None:
        """Trajectory without a target token must raise ValueError."""
        sweep = _make_synthetic_sweep(with_target=False)
        with pytest.raises(ValueError, match="target_token"):
            target_token_trajectory(sweep)


# ---------------------------------------------------------------------------
# 4. project_to_logits with mock model
# ---------------------------------------------------------------------------

class TestProjectToLogits:
    def test_project_to_logits_shape_with_mock(self) -> None:
        """project_to_logits should accept 1D / 2D / 3D residuals."""
        model = TinyMockTL(d_model=8, vocab=100)

        # 3D: (batch, seq, d_model) -> (batch, seq, vocab)
        residual_3d = torch.randn(1, 5, 8)
        logits_3d = project_to_logits(model, residual_3d, apply_final_ln=True)
        assert logits_3d.shape == (1, 5, 100)

        # 2D: (seq, d_model) -> (seq, vocab)
        residual_2d = torch.randn(5, 8)
        logits_2d = project_to_logits(model, residual_2d, apply_final_ln=True)
        assert logits_2d.shape == (5, 100)

        # 1D: (d_model,) -> (vocab,)
        residual_1d = torch.randn(8)
        logits_1d = project_to_logits(model, residual_1d, apply_final_ln=False)
        assert logits_1d.shape == (100,)

    def test_project_to_logits_apply_ln_changes_output(self) -> None:
        """Toggling apply_final_ln should produce different logits."""
        model = TinyMockTL(d_model=8, vocab=100)
        residual = torch.randn(1, 3, 8)
        with_ln = project_to_logits(model, residual, apply_final_ln=True)
        no_ln = project_to_logits(model, residual, apply_final_ln=False)
        assert not torch.allclose(with_ln, no_ln)


# ---------------------------------------------------------------------------
# 5. tuned_lens_project stub
# ---------------------------------------------------------------------------

class TestTunedLensProject:
    def test_tuned_lens_no_checkpoint_raises(self) -> None:
        """No checkpoint -> NotImplementedError pointing at the tuned-lens package."""
        model = TinyMockTL(d_model=8, vocab=100)
        residual = torch.randn(8)
        with pytest.raises(NotImplementedError, match="tuned-lens"):
            tuned_lens_project(model, residual, layer=0, tuned_lens_checkpoint=None)

    def test_tuned_lens_dict_checkpoint(self) -> None:
        """A simple dict-style checkpoint should run end-to-end."""
        model = TinyMockTL(d_model=8, vocab=100)
        residual = torch.randn(8)
        checkpoint = {
            0: {
                "weight": torch.eye(8),     # identity affine
                "bias": torch.zeros(8),
            }
        }
        logits = tuned_lens_project(model, residual, layer=0, tuned_lens_checkpoint=checkpoint)
        assert logits.shape == (100,)
