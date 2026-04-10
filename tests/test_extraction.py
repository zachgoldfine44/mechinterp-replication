"""Unit tests for src.utils.extraction.extract_for_experiment.

Uses a fake HuggingFace-shaped model so no real model needs to load. The
real-model end-to-end behavior is covered by tests/test_integration_tiny_model.py.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from torch import nn

from src.utils.activation_cache import ActivationCache
from src.utils.extraction import extract_for_experiment


class _FakeLayer(nn.Module):
    """A trivial layer that emits a deterministic per-token signal.

    The hook on this module fires with output shape (batch, seq, hidden).
    The output encodes (layer_idx, token_position) so tests can prove that
    the right layer was extracted and the right token position was pooled.
    """

    def __init__(self, layer_idx: int, hidden_dim: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_dim = hidden_dim
        # A no-op linear so register_forward_hook fires.
        self.identity = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq, hidden). Replace with deterministic content.
        batch, seq, _ = x.shape
        out = torch.zeros(batch, seq, self.hidden_dim)
        # Position 0 of hidden carries the layer index, position 1 the seq pos.
        for b in range(batch):
            for t in range(seq):
                out[b, t, 0] = float(self.layer_idx)
                out[b, t, 1] = float(t)
        return self.identity(out)


class _FakeInner(nn.Module):
    def __init__(self, n_layers: int, hidden_dim: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [_FakeLayer(i, hidden_dim) for i in range(n_layers)]
        )


class _FakeModel(nn.Module):
    """Mimics a HuggingFace causal LM enough for extract_for_experiment.

    Critically:
      - exposes ``.model.layers`` for ``get_hf_layer_modules``
      - has no ``run_with_cache`` (so we go down the HF path)
      - forward takes ``input_ids`` + ``attention_mask`` and runs every layer
    """

    def __init__(self, n_layers: int = 4, hidden_dim: int = 8) -> None:
        super().__init__()
        self.model = _FakeInner(n_layers, hidden_dim)
        self.hidden_dim = hidden_dim

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        x = torch.zeros(
            input_ids.shape[0], input_ids.shape[1], self.hidden_dim
        )
        for layer in self.model.layers:
            x = layer(x)
        return x


class _FakeTokenizer:
    """Tokenizer that splits on whitespace and assigns sequential ids."""

    pad_token_id = 0

    def __call__(
        self,
        text: str,
        return_tensors: str = "pt",
        truncation: bool = True,
        max_length: int = 512,
    ) -> dict[str, torch.Tensor]:
        tokens = text.split()
        ids = list(range(1, len(tokens) + 1))
        return {
            "input_ids": torch.tensor([ids], dtype=torch.long),
            "attention_mask": torch.ones(1, len(ids), dtype=torch.long),
        }


# ──────────────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────────────


def test_extract_returns_correct_shapes() -> None:
    model = _FakeModel(n_layers=4, hidden_dim=8)
    tok = _FakeTokenizer()
    texts = ["hello world", "the quick brown fox", "a"]

    out = extract_for_experiment(
        model=model,
        tokenizer=tok,
        texts=texts,
        layers=[0, 2, 3],
        aggregation="last_token",
    )

    assert set(out.keys()) == {0, 2, 3}
    for layer in (0, 2, 3):
        assert out[layer].shape == (3, 8)
        assert out[layer].dtype == np.float32


def test_extract_last_token_picks_last_position() -> None:
    """The fake layer encodes seq position in hidden[1], so last_token of
    'a b c' (3 tokens) should yield hidden[1] == 2."""
    model = _FakeModel(n_layers=2, hidden_dim=8)
    tok = _FakeTokenizer()

    out = extract_for_experiment(
        model=model,
        tokenizer=tok,
        texts=["a b c"],
        layers=[1],
        aggregation="last_token",
    )

    arr = out[1][0]  # (hidden_dim,)
    assert arr[0] == 1.0, "layer index encoded in hidden[0]"
    assert arr[1] == 2.0, "last token of 3-token sequence is position 2"


def test_extract_first_token_picks_first_position() -> None:
    model = _FakeModel(n_layers=2, hidden_dim=8)
    tok = _FakeTokenizer()

    out = extract_for_experiment(
        model=model,
        tokenizer=tok,
        texts=["a b c d"],
        layers=[0],
        aggregation="first_token",
    )

    assert out[0][0, 1] == 0.0, "first token is position 0"


def test_extract_mean_averages_over_positions() -> None:
    """Mean of positions [0,1,2,3] is 1.5."""
    model = _FakeModel(n_layers=1, hidden_dim=8)
    tok = _FakeTokenizer()

    out = extract_for_experiment(
        model=model,
        tokenizer=tok,
        texts=["a b c d"],
        layers=[0],
        aggregation="mean",
    )

    assert out[0][0, 1] == pytest.approx(1.5, abs=1e-5)


def test_extract_uses_in_memory_cache(tmp_path: Path) -> None:
    """A second call with the same in-memory cache should not re-run the model."""
    model = _FakeModel(n_layers=2, hidden_dim=8)
    tok = _FakeTokenizer()
    cache = ActivationCache(model_key="fake")

    texts = ["one two", "three four"]

    out1 = extract_for_experiment(
        model=model, tokenizer=tok, texts=texts,
        layers=[0, 1], aggregation="last_token",
        activations_cache=cache,
    )
    stats1 = cache.stats()
    # The first pass short-circuits after the first miss per text, so we
    # only see 2 misses (one per text), but the cache then gets fully
    # populated by put_batch from the live extraction.
    assert stats1["misses"] >= 2
    assert stats1["size"] == 4  # 2 texts × 2 layers in the cache

    out2 = extract_for_experiment(
        model=model, tokenizer=tok, texts=texts,
        layers=[0, 1], aggregation="last_token",
        activations_cache=cache,
    )
    stats2 = cache.stats()
    # Second pass: every (text, layer) pair is a hit.
    assert stats2["hits"] - stats1["hits"] == 4
    assert stats2["misses"] == stats1["misses"]  # no new misses

    for layer in (0, 1):
        np.testing.assert_array_equal(out1[layer], out2[layer])


def test_extract_uses_disk_cache(tmp_path: Path) -> None:
    """A second call without an in-memory cache should hit the disk cache."""
    model = _FakeModel(n_layers=1, hidden_dim=8)
    tok = _FakeTokenizer()
    cache_dir = tmp_path / "acts"

    out1 = extract_for_experiment(
        model=model, tokenizer=tok, texts=["alpha beta"],
        layers=[0], aggregation="last_token",
        cache_dir=cache_dir,
    )

    # Check the per-stimulus file exists.
    assert (cache_dir / "stimulus_0000.pt").exists()

    # Replace the model so a fresh extraction would emit zeros.
    out2 = extract_for_experiment(
        model=_FakeModel(n_layers=1, hidden_dim=8),
        tokenizer=tok,
        texts=["alpha beta"],
        layers=[0],
        aggregation="last_token",
        cache_dir=cache_dir,
    )

    np.testing.assert_array_equal(out1[0], out2[0])


def test_extract_passes_aggregation_through_to_helper() -> None:
    """The aggregation kwarg should affect the result."""
    model = _FakeModel(n_layers=1, hidden_dim=8)
    tok = _FakeTokenizer()

    last = extract_for_experiment(
        model=model, tokenizer=tok, texts=["a b c d e"],
        layers=[0], aggregation="last_token",
    )[0]
    first = extract_for_experiment(
        model=model, tokenizer=tok, texts=["a b c d e"],
        layers=[0], aggregation="first_token",
    )[0]

    # Last position is 4, first is 0 → encoded in hidden[1].
    assert last[0, 1] == 4.0
    assert first[0, 1] == 0.0


def test_extract_raises_on_unknown_aggregation() -> None:
    model = _FakeModel(n_layers=1, hidden_dim=8)
    tok = _FakeTokenizer()

    with pytest.raises(ValueError):
        extract_for_experiment(
            model=model, tokenizer=tok, texts=["hello"],
            layers=[0], aggregation="not_a_real_strategy",
        )
