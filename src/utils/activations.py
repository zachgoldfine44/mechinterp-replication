"""Activation extraction with per-stimulus caching.

Extracts residual stream activations from models and caches to disk.
Supports TransformerLens HookedTransformer and HuggingFace models.
Resume-aware: skips stimuli that already have cached activations.

Cache format:
    {cache_dir}/layer_{L:03d}/stimulus_{id}.pt
    Each file contains a 1-D tensor of shape (hidden_dim,).

Usage:
    from src.utils.activations import extract_activations, load_cached_activations

    acts = extract_activations(
        model, tokenizer, texts,
        layers=[10, 15, 20],
        aggregation="last_token",
        cache_dir=Path("results/emotions/llama_1b/activations"),
        stimulus_ids=["stim_0000", "stim_0001", ...],
    )
    # acts: {10: Tensor(n_texts, hidden_dim), 15: ..., 20: ...}

Known limitations:
    - MPS: some TransformerLens ops may fail; callers should catch and
      retry on CPU (the model loader already does this).
    - HuggingFace extraction hooks target ``model.model.layers[L]`` which
      works for Llama, Qwen, Gemma, and most decoder-only LLMs.  Models
      with a different internal structure will need a custom layer accessor.
    - Mean aggregation uses attention_mask to exclude padding tokens.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Literal

import torch
from jaxtyping import Float
from torch import Tensor

from src.utils.env import get_device

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_activations(
    model: Any,
    tokenizer: Any,
    texts: list[str],
    layers: list[int] | None = None,
    aggregation: Literal["last_token", "mean"] = "last_token",
    batch_size: int = 4,
    cache_dir: Path | None = None,
    stimulus_ids: list[str] | None = None,
    device: str | None = None,
) -> dict[int, Float[Tensor, "n_texts hidden_dim"]]:
    """Extract residual stream activations from *model* for each text.

    Args:
        model: A TransformerLens ``HookedTransformer`` **or** a HuggingFace
            ``AutoModelForCausalLM`` (or compatible).
        tokenizer: The tokenizer paired with *model*.  For TransformerLens
            models this is typically ``model.tokenizer``; for HuggingFace
            models it is the ``AutoTokenizer`` instance.
        texts: List of input strings to extract activations for.
        layers: Layer indices to extract.  *None* means all layers.
        aggregation: How to reduce the sequence dimension:
            ``"last_token"`` -- use the final non-padding token's activation.
            ``"mean"`` -- mean over all non-padding tokens.
        batch_size: Number of texts processed per forward pass.
        cache_dir: If provided, per-stimulus activations are saved here and
            previously cached results are loaded instead of recomputed.
            Directory structure: ``{cache_dir}/layer_{L:03d}/stimulus_{id}.pt``
        stimulus_ids: Identifiers for each text, used for cache filenames.
            Must have the same length as *texts*.  If *None* and *cache_dir*
            is set, zero-padded indices are used (``stim_0000``, etc.).
        device: Torch device string.  Auto-detected if *None*.

    Returns:
        Dictionary mapping each requested layer index to a tensor of shape
        ``(len(texts), hidden_dim)``, moved to CPU.
    """
    if device is None:
        device = get_device()

    n_layers = _get_num_layers(model)
    if layers is None:
        layers = list(range(n_layers))
    else:
        for layer in layers:
            if layer < 0 or layer >= n_layers:
                raise ValueError(
                    f"Layer {layer} out of range for model with {n_layers} layers"
                )

    # Default stimulus IDs
    if stimulus_ids is None:
        stimulus_ids = [f"stim_{i:04d}" for i in range(len(texts))]
    if len(stimulus_ids) != len(texts):
        raise ValueError(
            f"stimulus_ids length ({len(stimulus_ids)}) != texts length ({len(texts)})"
        )

    # Try loading everything from cache first
    if cache_dir is not None:
        cached = load_cached_activations(cache_dir, stimulus_ids, layers)
        if cached is not None:
            logger.info(
                "All %d stimuli loaded from cache (%s)",
                len(texts),
                cache_dir,
            )
            return cached

    # Figure out which stimuli still need computation
    if cache_dir is not None:
        missing_mask = _find_missing_stimuli(cache_dir, stimulus_ids, layers)
    else:
        missing_mask = [True] * len(texts)

    n_missing = sum(missing_mask)
    n_cached = len(texts) - n_missing

    if n_cached > 0:
        logger.info(
            "Resuming: %d/%d stimuli cached, %d to compute",
            n_cached,
            len(texts),
            n_missing,
        )

    # Extract activations for missing stimuli
    if n_missing > 0:
        missing_texts = [t for t, m in zip(texts, missing_mask) if m]
        missing_ids = [s for s, m in zip(stimulus_ids, missing_mask) if m]

        if _is_transformerlens(model):
            raw = _extract_transformerlens(
                model, missing_texts, layers, aggregation, batch_size, device
            )
        else:
            raw = _extract_huggingface(
                model, tokenizer, missing_texts, layers, aggregation,
                batch_size, device,
            )

        # Save to cache
        if cache_dir is not None:
            _save_to_cache(cache_dir, raw, missing_ids, layers)

    # Load the full set from cache (now complete) or assemble from raw
    if cache_dir is not None:
        result = load_cached_activations(cache_dir, stimulus_ids, layers)
        assert result is not None, "Cache should be complete after extraction"
        return result
    else:
        # No caching -- raw already has everything (missing_mask was all True)
        return raw  # type: ignore[possibly-undefined]


def load_cached_activations(
    cache_dir: Path,
    stimulus_ids: list[str],
    layers: list[int],
) -> dict[int, Tensor] | None:
    """Load cached activations if all stimuli are present for all layers.

    Args:
        cache_dir: Root cache directory.
        stimulus_ids: Identifiers for each stimulus.
        layers: Layer indices to load.

    Returns:
        Dictionary mapping layer -> tensor of shape ``(n_stimuli, hidden_dim)``
        with all tensors on CPU.  Returns *None* if any stimulus file is
        missing for any layer.
    """
    cache_dir = Path(cache_dir)
    result: dict[int, list[Tensor]] = {layer: [] for layer in layers}

    for layer in layers:
        layer_dir = cache_dir / f"layer_{layer:03d}"
        for stim_id in stimulus_ids:
            path = layer_dir / f"stimulus_{stim_id}.pt"
            if not path.exists():
                return None
            tensor = torch.load(path, map_location="cpu", weights_only=True)
            result[layer].append(tensor)

    return {
        layer: torch.stack(tensors, dim=0)
        for layer, tensors in result.items()
    }


# ---------------------------------------------------------------------------
# TransformerLens backend
# ---------------------------------------------------------------------------


def _extract_transformerlens(
    model: Any,
    texts: list[str],
    layers: list[int],
    aggregation: Literal["last_token", "mean"],
    batch_size: int,
    device: str,
) -> dict[int, Float[Tensor, "n_texts hidden_dim"]]:
    """Extract residual stream activations using TransformerLens hooks.

    Uses ``model.run_with_cache()`` with a ``names_filter`` that selects
    only the ``blocks.{L}.hook_resid_post`` hooks for the requested layers.
    """
    hook_names = [f"blocks.{layer}.hook_resid_post" for layer in layers]
    names_filter = lambda name: name in hook_names  # noqa: E731

    # Accumulate per-layer results across batches
    per_layer: dict[int, list[Tensor]] = {layer: [] for layer in layers}

    for batch_start in range(0, len(texts), batch_size):
        batch_texts = texts[batch_start : batch_start + batch_size]

        # Tokenize -- TransformerLens models accept strings directly but
        # we need token IDs to compute sequence lengths for aggregation.
        tokens = model.to_tokens(batch_texts, prepend_bos=True)
        tokens = tokens.to(device)

        with torch.no_grad():
            _, cache = model.run_with_cache(
                tokens, names_filter=names_filter
            )

        for layer in layers:
            hook_name = f"blocks.{layer}.hook_resid_post"
            # cache[hook_name] shape: (batch, seq_len, hidden_dim)
            acts = cache[hook_name].detach().cpu()
            aggregated = _aggregate(acts, tokens, aggregation, tokenizer=model.tokenizer)
            per_layer[layer].append(aggregated)

        # Free cache memory
        del cache

    return {
        layer: torch.cat(tensors, dim=0)
        for layer, tensors in per_layer.items()
    }


# ---------------------------------------------------------------------------
# HuggingFace backend
# ---------------------------------------------------------------------------


def _extract_huggingface(
    model: Any,
    tokenizer: Any,
    texts: list[str],
    layers: list[int],
    aggregation: Literal["last_token", "mean"],
    batch_size: int,
    device: str,
) -> dict[int, Float[Tensor, "n_texts hidden_dim"]]:
    """Extract residual stream activations using HuggingFace forward hooks.

    Registers ``register_forward_hook`` on the transformer layer modules.
    Compatible with Llama, Qwen, Gemma, and most decoder-only models that
    expose layers via ``model.model.layers``.
    """
    # Ensure left-padding for causal LMs
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Locate the layer modules
    layer_modules = _get_hf_layer_modules(model)

    per_layer: dict[int, list[Tensor]] = {layer: [] for layer in layers}

    for batch_start in range(0, len(texts), batch_size):
        batch_texts = texts[batch_start : batch_start + batch_size]

        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(device)

        # Storage for hook outputs
        hook_outputs: dict[int, Tensor] = {}

        def _make_hook(layer_idx: int):
            def hook_fn(module, input, output):
                # output is typically a tuple; first element is the hidden state
                if isinstance(output, tuple):
                    hidden = output[0]
                else:
                    hidden = output
                hook_outputs[layer_idx] = hidden.detach().cpu()
            return hook_fn

        # Register hooks
        handles = []
        for layer_idx in layers:
            handle = layer_modules[layer_idx].register_forward_hook(
                _make_hook(layer_idx)
            )
            handles.append(handle)

        try:
            with torch.no_grad():
                model(**inputs)
        finally:
            for handle in handles:
                handle.remove()

        # Aggregate and store
        attention_mask = inputs["attention_mask"].cpu()
        for layer_idx in layers:
            acts = hook_outputs[layer_idx]
            aggregated = _aggregate(
                acts, None, aggregation,
                attention_mask=attention_mask,
            )
            per_layer[layer_idx].append(aggregated)

        del hook_outputs

    # Restore original padding side
    tokenizer.padding_side = original_padding_side

    return {
        layer: torch.cat(tensors, dim=0)
        for layer, tensors in per_layer.items()
    }


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _aggregate(
    acts: Float[Tensor, "batch seq_len hidden_dim"],
    tokens: Tensor | None,
    aggregation: Literal["last_token", "mean"],
    *,
    tokenizer: Any | None = None,
    attention_mask: Tensor | None = None,
) -> Float[Tensor, "batch hidden_dim"]:
    """Reduce the sequence dimension of activation tensors.

    For ``"last_token"``, uses the last non-padding position.
    For ``"mean"``, averages over all non-padding positions.

    Padding is determined from *attention_mask* (HuggingFace path) or by
    finding the pad_token_id in *tokens* (TransformerLens path).

    Args:
        acts: Activations of shape ``(batch, seq_len, hidden_dim)``.
        tokens: Token IDs of shape ``(batch, seq_len)``.  Used to derive
            padding mask for TransformerLens models.
        aggregation: ``"last_token"`` or ``"mean"``.
        tokenizer: TransformerLens tokenizer, used to get pad_token_id.
        attention_mask: Explicit attention mask of shape ``(batch, seq_len)``
            with 1 for real tokens and 0 for padding.

    Returns:
        Tensor of shape ``(batch, hidden_dim)``.
    """
    batch_size, seq_len, hidden_dim = acts.shape

    # Build attention mask if not provided
    if attention_mask is None:
        if tokens is not None and tokenizer is not None:
            pad_id = getattr(tokenizer, "pad_token_id", None)
            if pad_id is not None:
                attention_mask = (tokens != pad_id).long().cpu()
            else:
                attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
        else:
            attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)

    if aggregation == "last_token":
        # Find the index of the last non-padding token per sequence
        # Sum of mask gives the count of real tokens; subtract 1 for 0-index
        seq_lengths = attention_mask.sum(dim=1) - 1  # (batch,)
        seq_lengths = seq_lengths.clamp(min=0)

        # Gather the last-token activation
        indices = seq_lengths.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, hidden_dim)
        result = acts.gather(1, indices).squeeze(1)  # (batch, hidden_dim)
        return result

    elif aggregation == "mean":
        # Masked mean: sum over real tokens, divide by count
        mask_expanded = attention_mask.unsqueeze(-1).float()  # (batch, seq, 1)
        masked_acts = acts * mask_expanded
        summed = masked_acts.sum(dim=1)  # (batch, hidden_dim)
        counts = attention_mask.sum(dim=1, keepdim=True).float().clamp(min=1)
        return summed / counts

    else:
        raise ValueError(f"Unknown aggregation: {aggregation!r}")


# ---------------------------------------------------------------------------
# Caching helpers
# ---------------------------------------------------------------------------


def _find_missing_stimuli(
    cache_dir: Path,
    stimulus_ids: list[str],
    layers: list[int],
) -> list[bool]:
    """Return a boolean mask where True means the stimulus needs computation.

    A stimulus is considered cached only if ALL requested layers have a
    cached file for it.
    """
    cache_dir = Path(cache_dir)
    missing = []
    for stim_id in stimulus_ids:
        is_missing = False
        for layer in layers:
            path = cache_dir / f"layer_{layer:03d}" / f"stimulus_{stim_id}.pt"
            if not path.exists():
                is_missing = True
                break
        missing.append(is_missing)
    return missing


def _save_to_cache(
    cache_dir: Path,
    activations: dict[int, Tensor],
    stimulus_ids: list[str],
    layers: list[int],
) -> None:
    """Save per-stimulus activations to disk with atomic writes.

    Each stimulus gets its own file per layer:
        ``{cache_dir}/layer_{L:03d}/stimulus_{id}.pt``

    Uses a ``.tmp`` suffix + ``os.rename()`` for atomicity so that
    interrupted saves don't leave corrupt files.
    """
    cache_dir = Path(cache_dir)

    for layer in layers:
        layer_dir = cache_dir / f"layer_{layer:03d}"
        layer_dir.mkdir(parents=True, exist_ok=True)

        layer_acts = activations[layer]  # (n_stimuli, hidden_dim)

        for i, stim_id in enumerate(stimulus_ids):
            final_path = layer_dir / f"stimulus_{stim_id}.pt"
            if final_path.exists():
                continue  # already cached (race-condition safe)

            tmp_path = final_path.with_suffix(".tmp")
            torch.save(layer_acts[i].cpu(), tmp_path)
            os.rename(tmp_path, final_path)

    logger.info(
        "Cached %d stimuli x %d layers to %s",
        len(stimulus_ids),
        len(layers),
        cache_dir,
    )


# ---------------------------------------------------------------------------
# Model introspection helpers
# ---------------------------------------------------------------------------


def _is_transformerlens(model: Any) -> bool:
    """Check if *model* is a TransformerLens ``HookedTransformer``."""
    cls_name = type(model).__name__
    # Check class name to avoid importing transformer_lens at module level
    if cls_name == "HookedTransformer":
        return True
    # Also check via isinstance if transformer_lens is already imported
    try:
        from transformer_lens import HookedTransformer
        return isinstance(model, HookedTransformer)
    except ImportError:
        return False


def _get_num_layers(model: Any) -> int:
    """Get the number of transformer layers from a model.

    Supports TransformerLens and HuggingFace models.
    """
    if _is_transformerlens(model):
        return model.cfg.n_layers

    # HuggingFace: try common layer accessor patterns
    layer_modules = _get_hf_layer_modules(model)
    return len(layer_modules)


def _get_hf_layer_modules(model: Any) -> list[Any]:
    """Return the list of transformer layer modules from a HuggingFace model.

    Tries several common attribute paths used by different model families:
        - model.model.layers      (Llama, Qwen, Gemma, Mistral)
        - model.transformer.h     (GPT-2, GPT-Neo)
        - model.gpt_neox.layers   (GPT-NeoX, Pythia)

    Raises:
        AttributeError: If none of the known paths exist.
    """
    # Most common: Llama / Qwen / Gemma / Mistral
    inner = getattr(model, "model", None)
    if inner is not None:
        layers = getattr(inner, "layers", None)
        if layers is not None:
            return list(layers)

    # GPT-2 / GPT-Neo
    transformer = getattr(model, "transformer", None)
    if transformer is not None:
        h = getattr(transformer, "h", None)
        if h is not None:
            return list(h)

    # GPT-NeoX / Pythia
    neox = getattr(model, "gpt_neox", None)
    if neox is not None:
        layers = getattr(neox, "layers", None)
        if layers is not None:
            return list(layers)

    raise AttributeError(
        f"Cannot find transformer layers in model of type {type(model).__name__}. "
        f"Known patterns: model.model.layers, model.transformer.h, model.gpt_neox.layers. "
        f"Add support for this architecture in _get_hf_layer_modules()."
    )
