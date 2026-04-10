"""Unit tests for src/techniques/attention.py.

Uses synthetic tensors and mock models (no real LM loaded) to verify the
attention pattern dataclass, entropy computation, head ranking, and the
basic shape/keying contract of head attribution and pattern extraction.
"""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from src.techniques.attention import (
    AttentionPatterns,
    attention_entropy,
    compute_head_attribution,
    extract_attention_patterns,
    induction_head_score,
    top_attention_heads,
)


# ---------------------------------------------------------------------------
# AttentionPatterns dataclass
# ---------------------------------------------------------------------------

class TestAttentionPatternsDataclass:
    """Sanity checks for the AttentionPatterns dataclass."""

    def test_attention_patterns_dataclass(self) -> None:
        """Instantiate AttentionPatterns and verify all fields are set."""
        patterns = {
            0: torch.zeros(4, 5, 5),
            1: torch.ones(4, 5, 5) / 5.0,
        }
        ap = AttentionPatterns(
            patterns=patterns,
            tokens=["a", "b", "c", "d", "e"],
            n_layers=12,
            n_heads=4,
        )

        assert ap.n_layers == 12
        assert ap.n_heads == 4
        assert ap.tokens == ["a", "b", "c", "d", "e"]
        assert set(ap.patterns.keys()) == {0, 1}
        assert ap.patterns[0].shape == (4, 5, 5)
        assert ap.patterns[1].shape == (4, 5, 5)


# ---------------------------------------------------------------------------
# attention_entropy
# ---------------------------------------------------------------------------

class TestAttentionEntropy:
    """Tests for the attention_entropy helper."""

    def test_attention_entropy_uniform(self) -> None:
        """Uniform attention over n keys should give entropy log(n)."""
        n_heads, seq = 3, 8
        # Uniform: each key gets 1/seq mass.
        uniform = torch.full((n_heads, seq, seq), 1.0 / seq)
        out = attention_entropy({0: uniform})

        assert 0 in out
        ent = out[0]
        assert ent.shape == (n_heads, seq)

        expected = math.log(seq)
        assert torch.allclose(ent, torch.full_like(ent, expected), atol=1e-5)

    def test_attention_entropy_peaked(self) -> None:
        """One-hot attention (all mass on a single key) should give entropy 0."""
        n_heads, seq = 2, 6
        peaked = torch.zeros(n_heads, seq, seq)
        # Every query attends to key 0 with probability 1.
        peaked[..., 0] = 1.0
        out = attention_entropy({3: peaked})

        ent = out[3]
        assert ent.shape == (n_heads, seq)
        assert torch.allclose(ent, torch.zeros_like(ent), atol=1e-5)

    def test_attention_entropy_batched_shape(self) -> None:
        """Batched input (batch, heads, seq, seq) should yield (batch, heads, seq)."""
        attn = torch.full((2, 3, 4, 4), 0.25)  # uniform over 4 keys
        out = attention_entropy({7: attn})
        assert out[7].shape == (2, 3, 4)


# ---------------------------------------------------------------------------
# top_attention_heads
# ---------------------------------------------------------------------------

class TestTopAttentionHeads:
    """Tests for the top_attention_heads ranking helper."""

    def test_top_attention_heads_sort(self) -> None:
        """Heads should be sorted by score descending and limited to k."""
        attribution = {
            (0, 0): 0.1,
            (0, 1): 0.5,
            (1, 0): 0.9,
            (1, 1): 0.3,
            (2, 0): 0.7,
        }
        top = top_attention_heads(attribution, k=3)
        assert len(top) == 3
        assert top[0] == (1, 0, 0.9)
        assert top[1] == (2, 0, 0.7)
        assert top[2] == (0, 1, 0.5)

        # k larger than available -> return everything
        all_top = top_attention_heads(attribution, k=100)
        assert len(all_top) == len(attribution)
        # Verify descending order
        scores = [s for _, _, s in all_top]
        assert scores == sorted(scores, reverse=True)

    def test_top_attention_heads_empty(self) -> None:
        """Empty attribution dict should return an empty list."""
        assert top_attention_heads({}, k=5) == []


# ---------------------------------------------------------------------------
# compute_head_attribution (mock TL model)
# ---------------------------------------------------------------------------

class _MockTLModel:
    """Minimal TL-shaped mock with deterministic ablation behaviour.

    For each (layer, head) ablated, the mock returns logits whose first
    element along the vocab axis equals ``baseline - delta_lh`` where
    ``delta_lh`` is taken from a fixed table. The baseline pass returns
    ``baseline`` unchanged. This lets the tests verify the attribution
    keys and the sign convention without running real attention.
    """

    def __init__(self, n_layers: int = 2, n_heads: int = 3) -> None:
        self.cfg = SimpleNamespace(n_layers=n_layers, n_heads=n_heads, d_vocab=8)
        # Deterministic per-(layer, head) "importance"
        self._deltas = {
            (L, H): float(L * 10 + H + 1)
            for L in range(n_layers)
            for H in range(n_heads)
        }
        self._baseline = 100.0

    def __call__(self, tokens: torch.Tensor) -> torch.Tensor:
        # Baseline forward pass: produce logits whose [0,-1,0] element is baseline
        batch, seq = tokens.shape
        logits = torch.zeros(batch, seq, self.cfg.d_vocab)
        logits[:, -1, 0] = self._baseline
        return logits

    def run_with_cache(self, *args: Any, **kwargs: Any) -> Any:
        # Required only so _is_transformerlens(self) returns True. Not used.
        raise NotImplementedError("Mock TL model: run_with_cache is unused")

    def run_with_hooks(
        self,
        tokens: torch.Tensor,
        fwd_hooks: list[tuple[str, Any]],
    ) -> torch.Tensor:
        # The hook is registered on a single (layer, head). Inspect the hook
        # name to recover the layer, then call the hook with a fake z tensor
        # whose mutation we read back to determine which head was zeroed.
        assert len(fwd_hooks) == 1
        hook_name, hook_fn = fwd_hooks[0]
        # hook_name shape: blocks.{L}.attn.hook_z
        layer_idx = int(hook_name.split(".")[1])

        # Build a z tensor with each head slot tagged with a unique value.
        # Shape: (batch, seq, n_heads, d_head)
        d_head = 4
        z = torch.zeros(1, 1, self.cfg.n_heads, d_head)
        for H in range(self.cfg.n_heads):
            z[:, :, H, :] = float(H + 1)

        z_after = hook_fn(z, hook=None)

        # Detect which head was ablated to zero.
        ablated_head: int | None = None
        for H in range(self.cfg.n_heads):
            if torch.all(z_after[:, :, H, :] == 0):
                ablated_head = H
                break
        assert ablated_head is not None, "mock hook did not zero any head"

        delta = self._deltas[(layer_idx, ablated_head)]
        batch, seq = tokens.shape
        logits = torch.zeros(batch, seq, self.cfg.d_vocab)
        logits[:, -1, 0] = self._baseline - delta
        return logits


class TestComputeHeadAttribution:
    """Tests for compute_head_attribution against a mock TL model."""

    def test_compute_head_attribution_with_mock(self) -> None:
        """Verify keys, sign convention, and delta values for the mock."""
        model = _MockTLModel(n_layers=2, n_heads=3)
        tokens = torch.zeros(1, 5, dtype=torch.long)

        def metric_fn(logits: torch.Tensor) -> torch.Tensor:
            # Score is the first vocab logit at the last position
            return logits[0, -1, 0]

        attribution = compute_head_attribution(model, tokens, metric_fn)

        # 2 layers x 3 heads = 6 entries
        expected_keys = {(L, H) for L in range(2) for H in range(3)}
        assert set(attribution.keys()) == expected_keys

        # Score = baseline - ablated = (baseline) - (baseline - delta) = delta
        for (L, H), score in attribution.items():
            expected = float(L * 10 + H + 1)
            assert score == pytest.approx(expected, abs=1e-5)

    def test_compute_head_attribution_layer_subset(self) -> None:
        """layers/heads filters should restrict the sweep."""
        model = _MockTLModel(n_layers=4, n_heads=4)
        tokens = torch.zeros(1, 3, dtype=torch.long)

        def metric_fn(logits: torch.Tensor) -> torch.Tensor:
            return logits[0, -1, 0]

        attribution = compute_head_attribution(
            model, tokens, metric_fn, layers=[1, 2], heads=[0, 1],
        )
        assert set(attribution.keys()) == {(1, 0), (1, 1), (2, 0), (2, 1)}

    def test_compute_head_attribution_rejects_non_tl(self) -> None:
        """Non-TL model should raise RuntimeError."""
        # An object lacking run_with_cache/cfg fails the TL check.
        bad_model = SimpleNamespace()

        def metric_fn(logits: torch.Tensor) -> torch.Tensor:
            return logits.sum()

        with pytest.raises(RuntimeError, match="TransformerLens"):
            compute_head_attribution(
                bad_model, torch.zeros(1, 2, dtype=torch.long), metric_fn,
            )


# ---------------------------------------------------------------------------
# extract_attention_patterns (mock TL model)
# ---------------------------------------------------------------------------

class _MockTLExtractModel:
    """TL-shaped mock that returns canned attention patterns from run_with_cache."""

    def __init__(self, n_layers: int = 3, n_heads: int = 2, seq_len: int = 4) -> None:
        self.cfg = SimpleNamespace(
            n_layers=n_layers, n_heads=n_heads, d_vocab=10,
        )
        self._seq_len = seq_len
        self._n_layers = n_layers
        self._n_heads = n_heads

    def to_tokens(self, prompt: str | list[str]) -> torch.Tensor:
        if isinstance(prompt, list):
            return torch.zeros(len(prompt), self._seq_len, dtype=torch.long)
        return torch.zeros(1, self._seq_len, dtype=torch.long)

    def to_str_tokens(self, prompt: str) -> list[str]:
        return [f"tok{i}" for i in range(self._seq_len)]

    def run_with_cache(
        self,
        tokens: torch.Tensor,
        names_filter: Any,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        batch = tokens.shape[0]
        cache: dict[str, torch.Tensor] = {}
        for L in range(self._n_layers):
            name = f"blocks.{L}.attn.hook_pattern"
            if names_filter(name):
                # (batch, n_heads, seq, seq) — uniform attention
                attn = torch.full(
                    (batch, self._n_heads, self._seq_len, self._seq_len),
                    1.0 / self._seq_len,
                )
                cache[name] = attn
        # Logits are not used by extract_attention_patterns
        return torch.zeros(batch, self._seq_len, self.cfg.d_vocab), cache


class TestExtractAttentionPatterns:
    """Tests for extract_attention_patterns against a mock TL model."""

    def test_extract_attention_patterns_with_mock(self) -> None:
        """Single-prompt extraction returns (n_heads, seq, seq) per requested layer."""
        model = _MockTLExtractModel(n_layers=4, n_heads=2, seq_len=5)

        result = extract_attention_patterns(model, "hello", layers=[0, 2])

        assert isinstance(result, AttentionPatterns)
        assert result.n_layers == 4
        assert result.n_heads == 2
        assert set(result.patterns.keys()) == {0, 2}
        # Single prompt -> batch dim collapsed
        assert result.patterns[0].shape == (2, 5, 5)
        assert result.patterns[2].shape == (2, 5, 5)
        assert result.tokens == [f"tok{i}" for i in range(5)]

    def test_extract_attention_patterns_all_layers(self) -> None:
        """layers=None should fetch every layer."""
        model = _MockTLExtractModel(n_layers=3, n_heads=2, seq_len=4)
        result = extract_attention_patterns(model, "hello")
        assert set(result.patterns.keys()) == {0, 1, 2}

    def test_extract_attention_patterns_batched(self) -> None:
        """List input keeps the batch axis on each pattern tensor."""
        model = _MockTLExtractModel(n_layers=2, n_heads=2, seq_len=3)
        result = extract_attention_patterns(model, ["hi", "yo"], layers=[1])
        assert result.patterns[1].shape == (2, 2, 3, 3)


# ---------------------------------------------------------------------------
# induction_head_score (just verify the TL guard for unit-test scope)
# ---------------------------------------------------------------------------

class TestInductionHeadScore:
    """Cheap structural checks; the full test needs a real model."""

    def test_induction_head_score_rejects_non_tl(self) -> None:
        with pytest.raises(RuntimeError, match="TransformerLens"):
            induction_head_score(SimpleNamespace())

    def test_induction_head_score_validates_lengths(self) -> None:
        # Pass an object that satisfies the TL check, then trigger validation
        fake = SimpleNamespace(
            cfg=SimpleNamespace(n_layers=1, n_heads=1, d_vocab=10),
            run_with_cache=lambda *a, **k: (None, {}),
        )
        # context_length too small
        with pytest.raises(ValueError, match="context_length"):
            induction_head_score(fake, context_length=1)
        # seq_len too short
        with pytest.raises(ValueError, match="seq_len"):
            induction_head_score(fake, seq_len=10, context_length=8)
