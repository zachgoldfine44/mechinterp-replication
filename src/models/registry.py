"""Model registry: metadata lookup for all models in the matrix.

Loads model definitions from config/models.yaml and provides typed access
to model metadata (layer counts, hidden dims, HuggingFace IDs, etc.).

Usage:
    from src.models.registry import ModelRegistry

    registry = ModelRegistry()                  # loads config/models.yaml
    info = registry.get("llama_1b")             # -> ModelInfo
    small_models = registry.get_tier("small")   # -> [ModelInfo, ...]
    llama_family = registry.get_family("llama") # -> [ModelInfo, ...]
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ModelInfo:
    """Immutable metadata for a single model in the replication matrix.

    Fields mirror the structure in config/models.yaml. The ``key`` field
    (e.g. "llama_1b") is added during loading and serves as the primary
    identifier throughout the codebase.

    Attributes:
        key: Short identifier used in paths and logs (e.g. "llama_1b").
        hf_id: HuggingFace model ID (e.g. "meta-llama/Llama-3.2-1B-Instruct").
        family: Model family ("llama", "qwen", "gemma").
        size_tier: Execution tier ("small", "medium", "large").
        params_b: Approximate parameter count in billions.
        layers: Number of transformer layers.
        hidden_dim: Hidden / residual stream dimension.
        num_heads: Number of attention heads.
        two_thirds_layer: Layer index at roughly 2/3 depth -- a common
            default for probe extraction (many mechinterp findings peak here).
        loader: Which loading backend to use:
            "transformerlens", "nnsight", or "huggingface".
        dtype: Weight precision: "float16", "float32", "4bit", etc.
        notes: Free-text notes about model-specific quirks.
    """

    key: str
    hf_id: str
    family: str
    size_tier: str
    params_b: float
    layers: int
    hidden_dim: int
    num_heads: int
    two_thirds_layer: int
    loader: str
    dtype: str
    notes: str = ""


class ModelRegistry:
    """Registry of all models in the replication matrix.

    Loads model definitions from ``config/models.yaml`` (or a custom path)
    and exposes typed lookup methods.

    Examples:
        >>> reg = ModelRegistry()
        >>> reg.get("llama_1b").hf_id
        'meta-llama/Llama-3.2-1B-Instruct'
        >>> [m.key for m in reg.get_tier("small")]
        ['llama_1b', 'qwen_1_5b', 'gemma_2b']
    """

    def __init__(self, config_path: Path | None = None) -> None:
        if config_path is None:
            config_path = (
                Path(__file__).resolve().parent.parent.parent
                / "config"
                / "models.yaml"
            )
        with open(config_path) as f:
            raw: dict[str, Any] = yaml.safe_load(f)

        self._models: dict[str, ModelInfo] = {}
        for key, info in raw["models"].items():
            self._models[key] = ModelInfo(key=key, **info)

        self._execution_order: list[dict[str, Any]] = raw.get(
            "execution_order", []
        )

    # ── Lookups ──────────────────────────────────────────────────────────

    def get(self, model_key: str) -> ModelInfo:
        """Return metadata for a single model by its key.

        Raises:
            KeyError: If *model_key* is not in the registry.
        """
        if model_key not in self._models:
            available = list(self._models.keys())
            raise KeyError(
                f"Unknown model: {model_key!r}. Available: {available}"
            )
        return self._models[model_key]

    def get_tier(self, tier: str) -> list[ModelInfo]:
        """Return all models belonging to a size tier (small / medium / large)."""
        return [m for m in self._models.values() if m.size_tier == tier]

    def get_family(self, family: str) -> list[ModelInfo]:
        """Return all models belonging to a model family (llama / qwen / gemma)."""
        return [m for m in self._models.values() if m.family == family]

    def all_models(self) -> list[ModelInfo]:
        """Return every model in the registry."""
        return list(self._models.values())

    def keys(self) -> list[str]:
        """Return all model keys in insertion order."""
        return list(self._models.keys())

    @property
    def execution_order(self) -> list[dict[str, Any]]:
        """Return the tier execution order metadata from the config."""
        return list(self._execution_order)

    def __len__(self) -> int:
        return len(self._models)

    def __contains__(self, model_key: str) -> bool:
        return model_key in self._models

    def __repr__(self) -> str:
        return f"ModelRegistry({len(self._models)} models)"
