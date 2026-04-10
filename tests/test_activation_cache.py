"""Unit tests for src.utils.activation_cache.ActivationCache."""

from __future__ import annotations

import numpy as np
import pytest

from src.utils.activation_cache import ActivationCache


def test_put_get_roundtrip() -> None:
    cache = ActivationCache("test_model")
    vec = np.arange(8, dtype=np.float32)

    cache.put(layer=5, aggregation="last_token", text="hello", vec=vec)

    got = cache.get(layer=5, aggregation="last_token", text="hello")
    assert got is not None
    np.testing.assert_array_equal(got, vec)


def test_miss_on_different_layer() -> None:
    cache = ActivationCache("test_model")
    vec = np.arange(8, dtype=np.float32)
    cache.put(layer=5, aggregation="last_token", text="hello", vec=vec)

    assert cache.get(layer=6, aggregation="last_token", text="hello") is None


def test_miss_on_different_text() -> None:
    cache = ActivationCache("test_model")
    vec = np.arange(8, dtype=np.float32)
    cache.put(layer=5, aggregation="last_token", text="hello", vec=vec)

    assert cache.get(layer=5, aggregation="last_token", text="world") is None


def test_miss_on_different_aggregation() -> None:
    cache = ActivationCache("test_model")
    vec = np.arange(8, dtype=np.float32)
    cache.put(layer=5, aggregation="last_token", text="hello", vec=vec)

    assert cache.get(layer=5, aggregation="mean", text="hello") is None


def test_stats_counts() -> None:
    cache = ActivationCache("test_model")
    vec = np.arange(8, dtype=np.float32)
    cache.put(layer=5, aggregation="last_token", text="hello", vec=vec)

    # 1 hit
    cache.get(layer=5, aggregation="last_token", text="hello")
    # 2 misses (different layer, different text)
    cache.get(layer=6, aggregation="last_token", text="hello")
    cache.get(layer=5, aggregation="last_token", text="goodbye")

    stats = cache.stats()
    assert stats["hits"] == 1
    assert stats["misses"] == 2
    assert stats["size"] == 1
    assert stats["model_key"] == "test_model"


def test_batch_roundtrip() -> None:
    cache = ActivationCache("test_model")
    texts = ["alpha", "beta", "gamma"]
    vecs = np.stack(
        [
            np.arange(4, dtype=np.float32),
            np.arange(4, 8, dtype=np.float32),
            np.arange(8, 12, dtype=np.float32),
        ]
    )

    cache.put_batch(layer=3, aggregation="mean", texts=texts, vecs=vecs)

    stacked, misses = cache.get_batch(
        layer=3, aggregation="mean", texts=texts
    )
    assert misses == []
    assert stacked is not None
    np.testing.assert_array_equal(stacked, vecs)


def test_batch_partial_hits() -> None:
    cache = ActivationCache("test_model")
    cache.put(layer=0, aggregation="mean", text="a", vec=np.ones(4, dtype=np.float32))
    cache.put(layer=0, aggregation="mean", text="c", vec=np.full(4, 3.0, dtype=np.float32))

    stacked, misses = cache.get_batch(
        layer=0, aggregation="mean", texts=["a", "b", "c"]
    )
    assert misses == [1]
    assert stacked is not None
    np.testing.assert_array_equal(stacked[0], np.ones(4, dtype=np.float32))
    np.testing.assert_array_equal(stacked[2], np.full(4, 3.0, dtype=np.float32))


def test_batch_all_misses() -> None:
    cache = ActivationCache("test_model")
    stacked, misses = cache.get_batch(
        layer=0, aggregation="mean", texts=["a", "b"]
    )
    assert stacked is None
    assert misses == [0, 1]


def test_put_batch_shape_mismatch() -> None:
    cache = ActivationCache("test_model")
    with pytest.raises(ValueError):
        cache.put_batch(
            layer=0,
            aggregation="mean",
            texts=["a", "b"],
            vecs=np.zeros((3, 4), dtype=np.float32),
        )


def test_disk_persistence_roundtrip(tmp_path) -> None:
    cache = ActivationCache("test_model", data_root=tmp_path, persist=True)
    vec = np.arange(6, dtype=np.float32)
    cache.put(layer=2, aggregation="last_token", text="persist_me", vec=vec)

    # New cache, same disk location — should re-hydrate from disk.
    cache2 = ActivationCache("test_model", data_root=tmp_path, persist=True)
    got = cache2.get(layer=2, aggregation="last_token", text="persist_me")
    assert got is not None
    np.testing.assert_array_equal(got, vec)
