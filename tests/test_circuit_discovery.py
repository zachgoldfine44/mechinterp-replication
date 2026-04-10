"""Tests for src.techniques.circuit_discovery.

These tests use synthetic data and mocks — no real language models are loaded.
They cover the dataclass, the pure subgraph helper, the ACDC stub, and the
"plumbing" of edge_attribution_patching / path_patch via a MagicMock that
emulates a TransformerLens HookedTransformer.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

class TestEAPResultDataclass:
    """EAPResult should hold the expected fields and accept default metadata."""

    def test_eap_result_dataclass(self) -> None:
        from src.techniques.circuit_discovery import EAPResult

        result = EAPResult(
            edge_scores={"L0H0": 0.5, "L0H1": -0.25},
            top_edges=[("L0H0", 0.5), ("L0H1", -0.25)],
            metric_value_clean=2.0,
            metric_value_corrupted=0.5,
            n_edges_scored=2,
        )

        assert result.edge_scores == {"L0H0": 0.5, "L0H1": -0.25}
        assert result.top_edges[0] == ("L0H0", 0.5)
        assert result.metric_value_clean == 2.0
        assert result.metric_value_corrupted == 0.5
        assert result.n_edges_scored == 2
        assert result.metadata == {}  # default factory

        # metadata is mutable and can be populated by callers.
        result.metadata["nodes"] = "heads"
        assert result.metadata["nodes"] == "heads"


# ---------------------------------------------------------------------------
# extract_subgraph
# ---------------------------------------------------------------------------

class TestExtractSubgraph:
    """Pure helper: filter an edge-score dict by threshold and/or top-k."""

    def test_extract_subgraph_threshold(self) -> None:
        from src.techniques.circuit_discovery import extract_subgraph

        scores = {"a": 0.9, "b": 0.5, "c": 0.1, "d": -0.6}
        # Threshold of 0.5 keeps |score| >= 0.5: a, b, d.
        keep = extract_subgraph(scores, threshold=0.5)
        assert keep == {"a", "b", "d"}

    def test_extract_subgraph_top_k(self) -> None:
        from src.techniques.circuit_discovery import extract_subgraph

        scores = {"a": 0.9, "b": 0.5, "c": 0.1, "d": -0.6}
        # Top-2 by |score|: a (0.9), d (0.6).
        keep = extract_subgraph(scores, top_k=2)
        assert keep == {"a", "d"}

    def test_extract_subgraph_combined(self) -> None:
        from src.techniques.circuit_discovery import extract_subgraph

        scores = {"a": 0.9, "b": 0.5, "c": 0.1, "d": -0.6}
        # threshold=0.5 -> {a, b, d}; top_k=2 -> {a, d}; intersection -> {a, d}.
        keep = extract_subgraph(scores, threshold=0.5, top_k=2)
        assert keep == {"a", "d"}

        # threshold=0.7 -> {a}; top_k=3 -> {a, d, b}; intersection -> {a}.
        keep2 = extract_subgraph(scores, threshold=0.7, top_k=3)
        assert keep2 == {"a"}

    def test_extract_subgraph_empty(self) -> None:
        from src.techniques.circuit_discovery import extract_subgraph

        assert extract_subgraph({}) == set()
        assert extract_subgraph({}, threshold=0.5) == set()
        assert extract_subgraph({}, top_k=10) == set()
        assert extract_subgraph({}, threshold=0.5, top_k=10) == set()

    def test_extract_subgraph_no_filters_returns_all(self) -> None:
        from src.techniques.circuit_discovery import extract_subgraph

        scores = {"a": 0.9, "b": 0.5}
        assert extract_subgraph(scores) == {"a", "b"}

    def test_extract_subgraph_negative_top_k_raises(self) -> None:
        from src.techniques.circuit_discovery import extract_subgraph

        with pytest.raises(ValueError, match="top_k"):
            extract_subgraph({"a": 1.0}, top_k=-1)


# ---------------------------------------------------------------------------
# ACDC stub
# ---------------------------------------------------------------------------

class TestAcdcStub:
    """The ACDC entry point should raise a clear NotImplementedError."""

    def test_acdc_raises_not_implemented(self) -> None:
        from src.techniques.circuit_discovery import acdc

        with pytest.raises(NotImplementedError) as exc_info:
            acdc(model=None)

        msg = str(exc_info.value)
        assert "ACDC" in msg
        # Helpful pointers to the alternatives we DO implement.
        assert "edge_attribution_patching" in msg
        assert "path_patch" in msg


# ---------------------------------------------------------------------------
# Mock TransformerLens model
# ---------------------------------------------------------------------------

def _make_mock_tl_model(
    n_layers: int = 2,
    n_heads: int = 2,
    d_head: int = 4,
    d_model: int = 8,
    seq_len: int = 3,
    vocab: int = 16,
) -> MagicMock:
    """Build a MagicMock that quacks like a TransformerLens HookedTransformer.

    The mock supports:
      - ``model.cfg.n_layers`` / ``model.cfg.n_heads`` / ``model.cfg.d_head``
      - ``model.run_with_cache(tokens, names_filter=...)`` returning
        ``(logits, cache)`` with hook_z entries for each layer.
      - ``model.run_with_hooks(tokens, fwd_hooks=...)`` which actually CALLS
        each hook on a fake activation tensor (so capture_hooks see real data)
        and returns logits.
      - Plain ``model(tokens)`` returning logits.
    """
    model = MagicMock()
    model.cfg = SimpleNamespace(
        n_layers=n_layers,
        n_heads=n_heads,
        d_head=d_head,
        d_model=d_model,
    )

    def _fake_logits() -> torch.Tensor:
        # Use ones so metric_fn(logits) is a non-trivial scalar.
        return torch.ones(1, seq_len, vocab, requires_grad=True)

    def _fake_z_act() -> torch.Tensor:
        return torch.zeros(1, seq_len, n_heads, d_head)

    def _fake_resid_act() -> torch.Tensor:
        return torch.zeros(1, seq_len, d_model)

    def _build_full_cache() -> dict[str, torch.Tensor]:
        cache: dict[str, torch.Tensor] = {}
        for L in range(n_layers):
            cache[f"blocks.{L}.attn.hook_z"] = _fake_z_act() + 0.1 * (L + 1)
            cache[f"blocks.{L}.hook_resid_post"] = _fake_resid_act() + 0.1 * (L + 1)
        return cache

    def run_with_cache(tokens, names_filter=None, **kwargs):
        cache = _build_full_cache()
        if names_filter is not None:
            cache = {k: v for k, v in cache.items() if names_filter(k)}
        return _fake_logits(), cache

    def run_with_hooks(tokens, fwd_hooks=None, **kwargs):
        # Actually invoke each hook on a matching fake activation so capture
        # hooks populate their dicts (this is what we want to verify).
        if fwd_hooks:
            for name, hook_fn in fwd_hooks:
                if name.endswith("attn.hook_z"):
                    act = _fake_z_act().requires_grad_(True)
                else:
                    act = _fake_resid_act().requires_grad_(True)
                # Some hooks return tensors, some return None — both are OK
                # for TL semantics; the harness only cares that we called them.
                hook_fn(act, hook=SimpleNamespace(name=name))
        return _fake_logits()

    model.run_with_cache = MagicMock(side_effect=run_with_cache)
    model.run_with_hooks = MagicMock(side_effect=run_with_hooks)
    model.side_effect = lambda tokens, **kw: _fake_logits()
    # Plain call: ``model(tokens)``
    model.__call__ = lambda *a, **kw: _fake_logits()
    model.return_value = _fake_logits()
    return model


# ---------------------------------------------------------------------------
# edge_attribution_patching plumbing
# ---------------------------------------------------------------------------

class TestEdgeAttributionPatching:
    """Verify the EAP plumbing returns an EAPResult with the expected fields."""

    def test_edge_attribution_patching_with_mock(self) -> None:
        from src.techniques.circuit_discovery import (
            EAPResult,
            edge_attribution_patching,
        )

        model = _make_mock_tl_model(n_layers=2, n_heads=2, seq_len=3, vocab=16)
        clean = torch.zeros(1, 3, dtype=torch.long)
        corrupted = torch.ones(1, 3, dtype=torch.long)

        # A trivial scalar metric over the logits — exercises the backward
        # path. The mock returns a tensor with requires_grad=True so the
        # metric is differentiable.
        metric = lambda logits: logits.sum()

        result = edge_attribution_patching(
            model=model,
            clean_tokens=clean,
            corrupted_tokens=corrupted,
            metric_fn=metric,
            top_k=4,
            nodes="heads",
        )

        assert isinstance(result, EAPResult)
        assert result.metadata["nodes"] == "heads"
        assert result.metadata["n_layers"] == 2
        # 2 layers × 2 heads = 4 head nodes scored.
        assert result.n_edges_scored == 4
        assert set(result.edge_scores.keys()) == {
            "L0H0", "L0H1", "L1H0", "L1H1",
        }
        # top_edges respects top_k.
        assert len(result.top_edges) <= 4
        for name, score in result.top_edges:
            assert name in result.edge_scores
            assert isinstance(score, float)
        # Clean / corrupted metric values should be plain floats.
        assert isinstance(result.metric_value_clean, float)
        assert isinstance(result.metric_value_corrupted, float)

    def test_edge_attribution_patching_layers_mode(self) -> None:
        from src.techniques.circuit_discovery import edge_attribution_patching

        model = _make_mock_tl_model(n_layers=3, n_heads=2, seq_len=2, vocab=8)
        clean = torch.zeros(1, 2, dtype=torch.long)
        corrupted = torch.ones(1, 2, dtype=torch.long)

        result = edge_attribution_patching(
            model=model,
            clean_tokens=clean,
            corrupted_tokens=corrupted,
            metric_fn=lambda logits: logits.sum(),
            top_k=2,
            nodes="layers",
        )

        # 3 layers × 1 resid hook = 3 entries.
        assert result.n_edges_scored == 3
        assert all(
            name.startswith("blocks.") and name.endswith("hook_resid_post")
            for name in result.edge_scores
        )
        assert len(result.top_edges) <= 2

    def test_edge_attribution_patching_shape_mismatch_raises(self) -> None:
        from src.techniques.circuit_discovery import edge_attribution_patching

        model = _make_mock_tl_model()
        clean = torch.zeros(1, 3, dtype=torch.long)
        corrupted = torch.zeros(1, 5, dtype=torch.long)
        with pytest.raises(ValueError, match="shape"):
            edge_attribution_patching(
                model=model,
                clean_tokens=clean,
                corrupted_tokens=corrupted,
                metric_fn=lambda L: L.sum(),
            )

    def test_edge_attribution_patching_requires_transformerlens(self) -> None:
        from src.techniques.circuit_discovery import edge_attribution_patching

        not_tl = MagicMock(spec=[])  # no run_with_hooks, no cfg
        with pytest.raises(NotImplementedError, match="TransformerLens"):
            edge_attribution_patching(
                model=not_tl,
                clean_tokens=torch.zeros(1, 2, dtype=torch.long),
                corrupted_tokens=torch.zeros(1, 2, dtype=torch.long),
                metric_fn=lambda L: L.sum(),
            )


# ---------------------------------------------------------------------------
# path_patch plumbing
# ---------------------------------------------------------------------------

class TestPathPatch:
    """Verify path_patch returns a dict with the expected keys."""

    def test_path_patch_returns_dict_with_expected_keys(self) -> None:
        from src.techniques.circuit_discovery import path_patch

        model = _make_mock_tl_model(n_layers=4, seq_len=3, vocab=8)
        clean = torch.zeros(1, 3, dtype=torch.long)
        corrupted = torch.ones(1, 3, dtype=torch.long)

        out = path_patch(
            model=model,
            clean_tokens=clean,
            corrupted_tokens=corrupted,
            sender=("resid", 1, 0),
            receiver=("resid", 3, 0),
            metric_fn=lambda L: L.sum(),
        )

        assert isinstance(out, dict)
        assert set(out.keys()) == {
            "effect",
            "clean_metric",
            "corrupted_metric",
            "patched_metric",
        }
        for v in out.values():
            assert isinstance(v, float)

    def test_path_patch_invalid_layer_order_raises(self) -> None:
        from src.techniques.circuit_discovery import path_patch

        model = _make_mock_tl_model(n_layers=3)
        with pytest.raises(ValueError, match="receiver_layer"):
            path_patch(
                model=model,
                clean_tokens=torch.zeros(1, 2, dtype=torch.long),
                corrupted_tokens=torch.zeros(1, 2, dtype=torch.long),
                sender=("resid", 2, 0),
                receiver=("resid", 1, 0),  # receiver before sender
                metric_fn=lambda L: L.sum(),
            )

    def test_path_patch_unsupported_component_type_raises(self) -> None:
        from src.techniques.circuit_discovery import path_patch

        model = _make_mock_tl_model()
        with pytest.raises(NotImplementedError, match="component_type"):
            path_patch(
                model=model,
                clean_tokens=torch.zeros(1, 2, dtype=torch.long),
                corrupted_tokens=torch.zeros(1, 2, dtype=torch.long),
                sender=("attn", 0, 0),  # not yet supported
                receiver=("resid", 1, 0),
                metric_fn=lambda L: L.sum(),
            )

    def test_path_patch_requires_transformerlens(self) -> None:
        from src.techniques.circuit_discovery import path_patch

        not_tl = MagicMock(spec=[])
        with pytest.raises(NotImplementedError, match="TransformerLens"):
            path_patch(
                model=not_tl,
                clean_tokens=torch.zeros(1, 2, dtype=torch.long),
                corrupted_tokens=torch.zeros(1, 2, dtype=torch.long),
                sender=("resid", 0, 0),
                receiver=("resid", 1, 0),
                metric_fn=lambda L: L.sum(),
            )
