"""Unified model loading across TransformerLens, nnsight, and HuggingFace.

Priority order (determined by ``ModelInfo.loader``):
  1. TransformerLens ("transformerlens") -- cleanest hook API for activation
     extraction.  Used for all small/medium models that TL supports.
  2. nnsight ("nnsight") -- proxy-based hooks for models TL doesn't cover.
  3. Raw HuggingFace ("huggingface") -- fallback with ``register_forward_hook``.
     Also used for quantized (4-bit) large models via bitsandbytes.

Usage:
    from src.models.registry import ModelRegistry
    from src.models.loader import load_model, unload_model

    registry = ModelRegistry()
    info = registry.get("llama_1b")
    model, tokenizer = load_model(info)

    # ... run experiments ...

    unload_model(model)
"""

from __future__ import annotations

import gc
import logging
import os
from typing import Any

import torch

from src.models.registry import ModelInfo
from src.utils.env import get_device

logger = logging.getLogger(__name__)


def _ensure_hf_token_env() -> None:
    """Ensure the HF_TOKEN env var is set from the cached login token.

    TransformerLens reads the token from the environment or HF's cache.
    Some versions don't properly read the cached token, so we explicitly
    set HF_TOKEN if it's not already present.
    """
    if os.environ.get("HF_TOKEN"):
        return
    try:
        from huggingface_hub import HfFolder
        token = HfFolder.get_token()
        if token:
            os.environ["HF_TOKEN"] = token
            logger.debug("Set HF_TOKEN from cached login token")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_model(
    model_info: ModelInfo,
    device: str | None = None,
) -> tuple[Any, Any]:
    """Load a model and its tokenizer based on :pyclass:`ModelInfo`.

    Args:
        model_info: Metadata for the model to load (from :class:`ModelRegistry`).
        device: Target device string ("cuda", "mps", "cpu").  If *None*,
            auto-detected via :func:`src.utils.env.get_device`.

    Returns:
        ``(model, tokenizer)`` tuple.

        * **TransformerLens**: *model* is a ``HookedTransformer``;
          *tokenizer* is ``model.tokenizer``.
        * **nnsight**: *model* is an ``NNsight`` wrapper; *tokenizer* is the
          underlying HF tokenizer.
        * **HuggingFace**: *model* is an ``AutoModelForCausalLM``; *tokenizer*
          is an ``AutoTokenizer``.

    Raises:
        ValueError: If ``model_info.loader`` is not a recognised backend.
    """
    if device is None:
        device = get_device()

    logger.info(
        "Loading %s (%s) via %s on %s",
        model_info.key,
        model_info.hf_id,
        model_info.loader,
        device,
    )

    if model_info.loader == "transformerlens":
        return _load_transformerlens(model_info, device)
    elif model_info.loader == "nnsight":
        return _load_nnsight(model_info, device)
    elif model_info.loader == "huggingface":
        return _load_huggingface(model_info, device)
    else:
        raise ValueError(
            f"Unknown loader: {model_info.loader!r}. "
            f"Expected one of: transformerlens, nnsight, huggingface"
        )


def unload_model(model: Any) -> None:
    """Free GPU/CPU memory held by *model*.

    Deletes the model reference, runs garbage collection, and empties the
    CUDA cache if applicable.  Callers should also ``del`` their own
    reference to the model object after calling this.
    """
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("Model unloaded; memory freed.")


# ---------------------------------------------------------------------------
# Backend: TransformerLens
# ---------------------------------------------------------------------------

def _resolve_dtype(dtype_str: str) -> torch.dtype:
    """Map a config dtype string to a ``torch.dtype``."""
    mapping: dict[str, torch.dtype] = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }
    return mapping.get(dtype_str, torch.float32)


def _load_transformerlens(
    model_info: ModelInfo,
    device: str,
) -> tuple[Any, Any]:
    """Load via TransformerLens ``HookedTransformer``.

    On MPS, some TransformerLens operations can fail.  If that happens we
    automatically retry on CPU with float32, which is fast enough for
    small-tier models on a MacBook.
    """
    from transformer_lens import HookedTransformer

    dtype = _resolve_dtype(model_info.dtype)

    # Ensure HF token is available for gated models (Llama, Gemma)
    _ensure_hf_token_env()

    try:
        model = HookedTransformer.from_pretrained(
            model_info.hf_id,
            device=device,
            dtype=dtype,
            fold_ln=False,
            center_writing_weights=False,
            center_unembed=False,
        )
    except Exception as exc:
        if device == "mps":
            logger.warning(
                "MPS failed for %s, falling back to CPU: %s",
                model_info.key,
                exc,
            )
            model = HookedTransformer.from_pretrained(
                model_info.hf_id,
                device="cpu",
                dtype=torch.float32,
                fold_ln=False,
                center_writing_weights=False,
                center_unembed=False,
            )
        else:
            raise

    model.eval()
    tokenizer = model.tokenizer
    logger.info("Loaded %s via TransformerLens (%d layers)", model_info.key, model_info.layers)
    return model, tokenizer


# ---------------------------------------------------------------------------
# Backend: nnsight
# ---------------------------------------------------------------------------

def _load_nnsight(
    model_info: ModelInfo,
    device: str,
) -> tuple[Any, Any]:
    """Load via nnsight's ``LanguageModel`` wrapper.

    nnsight wraps a HuggingFace model and exposes a proxy-based tracing API
    for activation extraction.  Useful for models that TransformerLens does
    not support.
    """
    from nnsight import LanguageModel

    dtype = _resolve_dtype(model_info.dtype)

    model = LanguageModel(
        model_info.hf_id,
        device_map=device if device != "cpu" else None,
        torch_dtype=dtype,
        trust_remote_code=True,
    )

    tokenizer = model.tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Loaded %s via nnsight (%d layers)", model_info.key, model_info.layers)
    return model, tokenizer


# ---------------------------------------------------------------------------
# Backend: raw HuggingFace
# ---------------------------------------------------------------------------

def _load_huggingface(
    model_info: ModelInfo,
    device: str,
) -> tuple[Any, Any]:
    """Load via ``transformers.AutoModelForCausalLM``.

    Supports optional 4-bit quantization via bitsandbytes for large models
    (70B+).  When ``dtype == "4bit"`` the model is loaded with
    ``BitsAndBytesConfig`` and ``device_map="auto"``; otherwise it is loaded
    at the requested precision and moved to *device* explicitly.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # -- Tokenizer --------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(
        model_info.hf_id,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # -- Model ------------------------------------------------------------
    load_kwargs: dict[str, Any] = {"trust_remote_code": True}

    if model_info.dtype == "4bit":
        from transformers import BitsAndBytesConfig

        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        load_kwargs["device_map"] = "auto"
    else:
        dtype = _resolve_dtype(model_info.dtype)
        load_kwargs["torch_dtype"] = dtype
        # device_map=None lets us call .to(device) manually afterwards,
        # which is more reliable for single-GPU or CPU setups.
        load_kwargs["device_map"] = device if device != "cpu" else None

    model = AutoModelForCausalLM.from_pretrained(
        model_info.hf_id,
        **load_kwargs,
    )
    model.eval()

    # If no device_map was used (CPU path), move explicitly.
    if load_kwargs.get("device_map") is None:
        model = model.to(device)

    logger.info(
        "Loaded %s via HuggingFace (dtype=%s, %d layers)",
        model_info.key,
        model_info.dtype,
        model_info.layers,
    )
    return model, tokenizer
