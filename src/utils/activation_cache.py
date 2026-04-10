"""In-memory (and optional disk-backed) activation cache.

Keyed by (model_key, layer, aggregation, text_hash). Used by experiments
to share activation extraction across claims that run on the same model
and the same texts — a probe_classification claim and a causal_steering
claim using the same stimuli should only pay the extraction cost once.

Text is hashed with sha1 truncated to 16 hex chars; full texts are not
stored in the cache keys. Collisions at this length are astronomically
unlikely for realistic stimulus-set sizes.

Usage:
    cache = ActivationCache("llama_1b")
    vec = cache.get(layer=5, aggregation="last_token", text="hello")
    if vec is None:
        vec = extract(...)
        cache.put(layer=5, aggregation="last_token", text="hello", vec=vec)
    print(cache.stats())
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def _hash_text(text: str) -> str:
    """Return a stable 16-hex-char hash of the input text."""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


class ActivationCache:
    """In-memory cache keyed by (layer, aggregation, text_hash).

    The model_key is fixed at construction time since one cache instance
    is scoped to a single model. If persist=True and a data_root is given,
    get/put also read/write one .npy file per entry under
    data_root/cache/activations/{model_key}/.
    """

    def __init__(
        self,
        model_key: str,
        data_root: Path | None = None,
        persist: bool = False,
    ) -> None:
        self.model_key = model_key
        self.data_root = data_root
        self.persist = persist
        self._store: dict[tuple[int, str, str], np.ndarray] = {}
        self._hits = 0
        self._misses = 0

        if self.persist:
            if self.data_root is None:
                raise ValueError(
                    "ActivationCache(persist=True) requires a data_root"
                )
            self._disk_dir = self.data_root / "cache" / "activations" / model_key
            self._disk_dir.mkdir(parents=True, exist_ok=True)
        else:
            self._disk_dir = None

    def _disk_path(self, layer: int, aggregation: str, text_hash: str) -> Path:
        assert self._disk_dir is not None
        return self._disk_dir / f"L{layer}_{aggregation}_{text_hash}.npy"

    def get(
        self, layer: int, aggregation: str, text: str
    ) -> np.ndarray | None:
        """Return the cached activation vector, or None on miss."""
        text_hash = _hash_text(text)
        key = (layer, aggregation, text_hash)

        vec = self._store.get(key)
        if vec is not None:
            self._hits += 1
            return vec

        # Try disk
        if self.persist and self._disk_dir is not None:
            path = self._disk_path(layer, aggregation, text_hash)
            if path.exists():
                try:
                    vec = np.load(path)
                    self._store[key] = vec
                    self._hits += 1
                    return vec
                except Exception as exc:
                    logger.warning("Failed to load cached activation %s: %s", path, exc)

        self._misses += 1
        return None

    def put(
        self,
        layer: int,
        aggregation: str,
        text: str,
        vec: np.ndarray,
    ) -> None:
        """Store an activation vector in the cache."""
        text_hash = _hash_text(text)
        key = (layer, aggregation, text_hash)
        self._store[key] = vec

        if self.persist and self._disk_dir is not None:
            path = self._disk_path(layer, aggregation, text_hash)
            tmp = path.with_suffix(".tmp.npy")
            try:
                np.save(tmp, vec)
                tmp.rename(path)
            except Exception as exc:
                logger.warning("Failed to persist activation %s: %s", path, exc)

    def get_batch(
        self,
        layer: int,
        aggregation: str,
        texts: list[str],
    ) -> tuple[np.ndarray | None, list[int]]:
        """Batch lookup.

        Returns:
            (stacked_array_or_None, miss_indices)

            If at least one text is cached, stacked_array is a numpy array
            of shape (len(texts), hidden_dim) where rows corresponding to
            misses are zero-filled (callers must not read those rows).
            If no hits, stacked_array is None.

            miss_indices lists the positions in `texts` that were not
            found in the cache.
        """
        hits: list[tuple[int, np.ndarray]] = []
        misses: list[int] = []
        for i, text in enumerate(texts):
            vec = self.get(layer, aggregation, text)
            if vec is None:
                misses.append(i)
            else:
                hits.append((i, vec))

        if not hits:
            return None, misses

        hidden_dim = hits[0][1].shape[-1]
        stacked = np.zeros((len(texts), hidden_dim), dtype=hits[0][1].dtype)
        for i, vec in hits:
            stacked[i] = vec
        return stacked, misses

    def put_batch(
        self,
        layer: int,
        aggregation: str,
        texts: list[str],
        vecs: np.ndarray,
    ) -> None:
        """Store a batch of activation vectors.

        Args:
            layer: Layer index.
            aggregation: Aggregation strategy string.
            texts: List of input texts.
            vecs: Array of shape (len(texts), hidden_dim).
        """
        if len(texts) != vecs.shape[0]:
            raise ValueError(
                f"put_batch: len(texts)={len(texts)} != vecs.shape[0]={vecs.shape[0]}"
            )
        for i, text in enumerate(texts):
            self.put(layer, aggregation, text, vecs[i])

    def stats(self) -> dict:
        """Return hit/miss/size statistics."""
        return {
            "model_key": self.model_key,
            "hits": self._hits,
            "misses": self._misses,
            "size": len(self._store),
            "persist": self.persist,
        }

    def __repr__(self) -> str:
        return (
            f"ActivationCache(model_key={self.model_key!r}, "
            f"size={len(self._store)}, hits={self._hits}, misses={self._misses})"
        )
