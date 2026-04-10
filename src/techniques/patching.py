"""Activation patching for causal interpretability.

Implements the Meng et al. (ROME) style causal trace and supporting
primitives:

- ``patch_residual_stream``: low-level — run a model while replacing the
  residual stream at a chosen ``(layer, position)`` with a precomputed
  activation from another (clean) run.
- ``causal_trace`` / ``denoising_patch_sweep``: corrupted run + restore
  one ``(layer, position)`` at a time, measure how much the clean answer
  is recovered.
- ``noising_patch_sweep``: the dual — clean run with patches replaced by
  corrupted activations.
- ``compute_patch_metric``: pure helper for the normalized restoration
  effect (logit_diff / prob_diff / kl_divergence).

Works with both TransformerLens (``run_with_hooks``) and HuggingFace
(``register_forward_hook``) models. Backend is detected via
``hasattr(model, "run_with_hooks")``.

Usage:
    from src.techniques.patching import (
        patch_residual_stream,
        causal_trace,
        compute_patch_metric,
        PatchResult,
    )

    # Get a cache of clean activations first (TL: model.run_with_cache).
    _, clean_cache = model.run_with_cache(clean_tokens)

    # Patch the residual stream at layer 10, position 5.
    new_logits = patch_residual_stream(
        model, corrupted_tokens, clean_cache,
        patch_layer=10, patch_position=5,
    )

    # Or sweep over (layer, position) pairs:
    out = causal_trace(
        model, clean_tokens, corrupted_tokens,
        answer_token_id=answer_id, metric="logit_diff",
    )
    print(out["effect_grid"].shape)  # (n_layers, n_positions)
    print(out["best_layer"], out["best_position"])
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import torch
from torch import Tensor

from src.utils.activations import get_hf_layer_modules

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class PatchResult:
    """Result of a single patching intervention.

    Attributes:
        clean_logits: Logits from the clean (un-corrupted) run. Either the
            full ``(seq_len, vocab)`` tensor or just the answer-token logit
            depending on what the caller passes through.
        corrupted_logits: Logits from the corrupted/baseline run.
        patched_logits: Logits after patching the corrupted run with clean
            activations at ``patch_layer, patch_position``.
        patch_layer: Which transformer layer was patched.
        patch_position: Which token position was patched (``None`` = all
            positions; ``int`` = single position; ``slice`` = a range).
        effect_size: Restoration effect in ``[0, 1]`` for ``logit_diff`` and
            ``prob_diff`` metrics. For ``kl_divergence`` it is ``-KL`` so
            "higher is better" stays consistent.
        metric_used: Name of the metric used to compute ``effect_size``.
        metadata: Free-form dict for any extras (timestamps, prompt info,
            answer token id, etc.).
    """

    clean_logits: Tensor
    corrupted_logits: Tensor
    patched_logits: Tensor
    patch_layer: int
    patch_position: int | slice | None
    effect_size: float
    metric_used: str
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Backend detection
# ---------------------------------------------------------------------------

def _is_transformerlens(model: Any) -> bool:
    """Return True if ``model`` looks like a TransformerLens HookedTransformer."""
    return hasattr(model, "run_with_hooks") and hasattr(model, "cfg")


# Layer-module accessor lives in src.utils.activations.get_hf_layer_modules
# (imported at the top of this file). This used to be a duplicated local
# helper; consolidated to one canonical implementation.


def _get_n_layers(model: Any) -> int:
    """Best-effort layer count for either backend."""
    if _is_transformerlens(model):
        return int(model.cfg.n_layers)
    return len(get_hf_layer_modules(model))


# ---------------------------------------------------------------------------
# Public API: metric helper
# ---------------------------------------------------------------------------

def compute_patch_metric(
    clean_logits: Tensor,
    corrupted_logits: Tensor,
    patched_logits: Tensor,
    answer_token_id: int,
    metric: Literal["logit_diff", "prob_diff", "kl_divergence"] = "logit_diff",
) -> float:
    """Compute the normalized restoration effect for a single patch.

    Each input is the model output at the position whose logits we care
    about (typically the last position). Inputs may be either ``(vocab,)``
    or ``(seq_len, vocab)``; if 2-D, the last position is used.

    Metrics
    -------
    - ``logit_diff``::

          (patched - corrupted) / (clean - corrupted)

      Equals 1.0 when patching fully restores the clean answer logit, 0.0
      when it has no effect, negative if patching pushes *away* from clean.

    - ``prob_diff``: same formula but on softmax probabilities.

    - ``kl_divergence``: ``-KL(patched || clean)`` so higher = closer to
      clean (sign-flipped for consistency with the other two metrics).

    The denominator is clamped to avoid divide-by-zero. When ``clean ==
    corrupted`` (no signal to recover), the function returns ``0.0``
    rather than NaN.

    Args:
        clean_logits: Logits from the clean run (``(vocab,)`` or
            ``(seq, vocab)``).
        corrupted_logits: Logits from the corrupted run.
        patched_logits: Logits from the patched run.
        answer_token_id: Vocab id of the answer token to score.
        metric: Which restoration metric to compute.

    Returns:
        Float restoration effect (see metric descriptions above).

    Raises:
        ValueError: If ``metric`` is unrecognized.
    """
    clean = _last_position(clean_logits).detach().float()
    corrupted = _last_position(corrupted_logits).detach().float()
    patched = _last_position(patched_logits).detach().float()

    if metric == "logit_diff":
        c = clean[answer_token_id].item()
        x = corrupted[answer_token_id].item()
        p = patched[answer_token_id].item()
        denom = c - x
        if abs(denom) < 1e-10:
            return 0.0
        return float((p - x) / denom)

    elif metric == "prob_diff":
        clean_probs = torch.softmax(clean, dim=-1)
        corrupted_probs = torch.softmax(corrupted, dim=-1)
        patched_probs = torch.softmax(patched, dim=-1)
        c = clean_probs[answer_token_id].item()
        x = corrupted_probs[answer_token_id].item()
        p = patched_probs[answer_token_id].item()
        denom = c - x
        if abs(denom) < 1e-10:
            return 0.0
        return float((p - x) / denom)

    elif metric == "kl_divergence":
        # KL(patched || clean): how much info is lost when using patched
        # as an approximation to clean. Lower = closer; we return -KL so
        # higher is still better.
        log_patched = torch.log_softmax(patched, dim=-1)
        log_clean = torch.log_softmax(clean, dim=-1)
        patched_probs = log_patched.exp()
        kl = (patched_probs * (log_patched - log_clean)).sum(dim=-1)
        return float(-kl.item())

    else:
        raise ValueError(
            f"Unknown metric: {metric!r}. "
            f"Expected 'logit_diff', 'prob_diff', or 'kl_divergence'."
        )


def _last_position(logits: Tensor) -> Tensor:
    """Return the last-position logits, accepting (vocab,) or (seq, vocab)."""
    if logits.dim() == 1:
        return logits
    if logits.dim() == 2:
        return logits[-1]
    if logits.dim() == 3:
        # (batch, seq, vocab) — take batch 0, last position.
        return logits[0, -1]
    raise ValueError(
        f"Unexpected logits shape {tuple(logits.shape)}; "
        f"expected (vocab,), (seq, vocab), or (batch, seq, vocab)."
    )


# ---------------------------------------------------------------------------
# Public API: low-level residual-stream patching
# ---------------------------------------------------------------------------

@torch.no_grad()
def patch_residual_stream(
    model: Any,
    tokens: Tensor,
    clean_cache: dict[str, Tensor],
    patch_layer: int,
    patch_position: int | slice | None = None,
    hook_name: str = "resid_post",
) -> Tensor:
    """Run ``model`` on ``tokens`` while patching the residual stream.

    At layer ``patch_layer``, replace the residual-stream activation at
    ``patch_position`` with the corresponding slice of ``clean_cache``.
    Returns the resulting logits.

    Args:
        model: TransformerLens HookedTransformer or HuggingFace model.
        tokens: Input token ids of shape ``(batch, seq_len)`` to feed
            through the model (typically the corrupted run).
        clean_cache: Dict of cached activations from a clean run. For TL,
            this is the dict produced by ``model.run_with_cache``. For HF,
            it should be a dict whose key is the layer index and whose
            value is the layer's hidden-state tensor with shape
            ``(batch, seq_len, hidden_dim)`` (e.g. from ``output_hidden_states``).
        patch_layer: Layer index to patch.
        patch_position: ``None`` to patch every position; ``int`` for a
            single position; ``slice`` for a range of positions.
        hook_name: Which residual-stream point to patch when using TL.
            Common values: ``"resid_post"``, ``"resid_pre"``, ``"resid_mid"``.

    Returns:
        Logits tensor with shape ``(batch, seq_len, vocab)``.
    """
    if _is_transformerlens(model):
        return _tl_patch_residual_stream(
            model, tokens, clean_cache, patch_layer, patch_position, hook_name,
        )
    else:
        return _hf_patch_residual_stream(
            model, tokens, clean_cache, patch_layer, patch_position,
        )


def _tl_patch_residual_stream(
    model: Any,
    tokens: Tensor,
    clean_cache: dict[str, Tensor],
    patch_layer: int,
    patch_position: int | slice | None,
    hook_name: str,
) -> Tensor:
    """TransformerLens patching via run_with_hooks."""
    full_hook_name = f"blocks.{patch_layer}.hook_{hook_name}"

    if full_hook_name not in clean_cache:
        # Try common alternates that TL uses.
        alternates = [
            f"blocks.{patch_layer}.hook_resid_post",
            f"blocks.{patch_layer}.hook_resid_pre",
            f"blocks.{patch_layer}.hook_resid_mid",
        ]
        found = None
        for alt in alternates:
            if alt in clean_cache:
                found = alt
                break
        if found is None:
            raise KeyError(
                f"Hook {full_hook_name!r} not found in clean_cache. "
                f"Available keys (first 5): {list(clean_cache.keys())[:5]}"
            )
        full_hook_name = found

    clean_act = clean_cache[full_hook_name]

    def hook_fn(activation: Tensor, hook: Any) -> Tensor:
        # activation shape: (batch, seq_len, hidden_dim)
        if patch_position is None:
            activation[...] = clean_act.to(activation.device)
        else:
            activation[:, patch_position, :] = clean_act[:, patch_position, :].to(
                activation.device
            )
        return activation

    logits = model.run_with_hooks(
        tokens,
        fwd_hooks=[(full_hook_name, hook_fn)],
    )
    return logits


def _hf_patch_residual_stream(
    model: Any,
    tokens: Tensor,
    clean_cache: dict[str, Tensor],
    patch_layer: int,
    patch_position: int | slice | None,
) -> Tensor:
    """HuggingFace patching via register_forward_hook on a layer module."""
    layer_modules = get_hf_layer_modules(model)
    if patch_layer >= len(layer_modules):
        raise IndexError(
            f"patch_layer={patch_layer} out of range for model with "
            f"{len(layer_modules)} layers."
        )

    # Look up clean activation. Accept either int key (preferred) or
    # the TL-style string key as a courtesy.
    if patch_layer in clean_cache:
        clean_act = clean_cache[patch_layer]
    elif str(patch_layer) in clean_cache:
        clean_act = clean_cache[str(patch_layer)]
    else:
        tl_key = f"blocks.{patch_layer}.hook_resid_post"
        if tl_key in clean_cache:
            clean_act = clean_cache[tl_key]
        else:
            raise KeyError(
                f"No clean activation for layer {patch_layer} in cache. "
                f"Available keys (first 5): {list(clean_cache.keys())[:5]}"
            )

    def hook_fn(module: Any, inputs: Any, output: Any) -> Any:
        if isinstance(output, tuple):
            hidden = output[0]
        else:
            hidden = output

        clean = clean_act.to(hidden.device).to(hidden.dtype)
        if patch_position is None:
            hidden = clean
        else:
            hidden = hidden.clone()
            hidden[:, patch_position, :] = clean[:, patch_position, :]

        if isinstance(output, tuple):
            return (hidden,) + output[1:]
        return hidden

    handle = layer_modules[patch_layer].register_forward_hook(hook_fn)
    try:
        outputs = model(tokens)
        # HF outputs are usually ModelOutput with .logits, but be permissive.
        if hasattr(outputs, "logits"):
            return outputs.logits
        if isinstance(outputs, tuple):
            return outputs[0]
        return outputs
    finally:
        handle.remove()


# ---------------------------------------------------------------------------
# Public API: causal trace / patch sweeps
# ---------------------------------------------------------------------------

def causal_trace(
    model: Any,
    clean_tokens: Tensor,
    corrupted_tokens: Tensor,
    answer_token_id: int,
    layers: list[int] | None = None,
    positions: list[int] | None = None,
    metric: Literal["logit_diff", "prob_diff", "kl_divergence"] = "logit_diff",
    hook_name: str = "resid_post",
) -> dict[str, Any]:
    """Meng-et-al-style denoising causal trace.

    Runs the model on ``corrupted_tokens`` while sweeping over
    ``(layer, position)`` pairs, restoring the residual stream from a
    clean run one slot at a time. Returns a grid of restoration effects.

    Args:
        model: TransformerLens or HuggingFace model.
        clean_tokens: Token ids for the clean prompt, shape
            ``(batch, seq_len)`` (typically batch=1).
        corrupted_tokens: Token ids for the corrupted prompt, same shape.
            Must be the same length as ``clean_tokens``.
        answer_token_id: Vocab id of the correct answer token.
        layers: Layers to sweep over (default: all layers).
        positions: Positions to sweep over (default: all positions in
            ``clean_tokens``).
        metric: Restoration metric — passed to ``compute_patch_metric``.
        hook_name: Residual-stream point to patch (TL only).

    Returns:
        Dict with::

            effect_grid: np.ndarray, shape (n_layers, n_positions)
            clean_logit: float — clean run's answer-token logit
            corrupted_logit: float — corrupted run's answer-token logit
            best_layer: int
            best_position: int
            layers: list[int]
            positions: list[int]
            metric: str
    """
    if clean_tokens.shape != corrupted_tokens.shape:
        raise ValueError(
            f"clean_tokens shape {tuple(clean_tokens.shape)} != "
            f"corrupted_tokens shape {tuple(corrupted_tokens.shape)}. "
            f"Pad to equal length before calling causal_trace."
        )

    n_layers = _get_n_layers(model)
    seq_len = clean_tokens.shape[-1]

    if layers is None:
        layers = list(range(n_layers))
    if positions is None:
        positions = list(range(seq_len))

    # Get clean cache + clean logits.
    clean_cache, clean_logits = _get_clean_cache_and_logits(
        model, clean_tokens, hook_name,
    )

    # Get corrupted logits (no patching).
    corrupted_logits = _run_model(model, corrupted_tokens)

    last_clean = _last_position(clean_logits)
    last_corrupted = _last_position(corrupted_logits)
    clean_logit_val = float(last_clean[answer_token_id].item())
    corrupted_logit_val = float(last_corrupted[answer_token_id].item())

    effect_grid = np.zeros((len(layers), len(positions)), dtype=np.float32)

    for li, layer in enumerate(layers):
        for pi, pos in enumerate(positions):
            patched_logits = patch_residual_stream(
                model,
                corrupted_tokens,
                clean_cache,
                patch_layer=layer,
                patch_position=pos,
                hook_name=hook_name,
            )
            effect = compute_patch_metric(
                clean_logits=clean_logits,
                corrupted_logits=corrupted_logits,
                patched_logits=patched_logits,
                answer_token_id=answer_token_id,
                metric=metric,
            )
            effect_grid[li, pi] = effect

    flat_idx = int(np.argmax(effect_grid))
    best_li, best_pi = np.unravel_index(flat_idx, effect_grid.shape)

    logger.info(
        "causal_trace: best (layer=%d, position=%d) effect=%.4f",
        layers[best_li], positions[best_pi], float(effect_grid[best_li, best_pi]),
    )

    return {
        "effect_grid": effect_grid,
        "clean_logit": clean_logit_val,
        "corrupted_logit": corrupted_logit_val,
        "best_layer": int(layers[best_li]),
        "best_position": int(positions[best_pi]),
        "layers": list(layers),
        "positions": list(positions),
        "metric": metric,
    }


def denoising_patch_sweep(
    model: Any,
    clean_tokens: Tensor,
    corrupted_tokens: Tensor,
    answer_token_id: int,
    layers: list[int] | None = None,
    positions: list[int] | None = None,
    metric: Literal["logit_diff", "prob_diff", "kl_divergence"] = "logit_diff",
    hook_name: str = "resid_post",
) -> dict[str, Any]:
    """Denoising sweep: corrupted run with clean activations restored.

    Alias of :func:`causal_trace` — kept for symmetry with
    :func:`noising_patch_sweep`.
    """
    return causal_trace(
        model=model,
        clean_tokens=clean_tokens,
        corrupted_tokens=corrupted_tokens,
        answer_token_id=answer_token_id,
        layers=layers,
        positions=positions,
        metric=metric,
        hook_name=hook_name,
    )


def noising_patch_sweep(
    model: Any,
    clean_tokens: Tensor,
    corrupted_tokens: Tensor,
    answer_token_id: int,
    layers: list[int] | None = None,
    positions: list[int] | None = None,
    metric: Literal["logit_diff", "prob_diff", "kl_divergence"] = "logit_diff",
    hook_name: str = "resid_post",
) -> dict[str, Any]:
    """Noising sweep: clean run with corrupted activations injected (the dual).

    Same shape of result as :func:`causal_trace`. Here ``effect_grid[l, p]``
    measures how much *destroying* a single ``(layer, position)`` damages
    the clean answer — under ``logit_diff``, lower (more negative) means
    that slot was more important to the clean answer.
    """
    if clean_tokens.shape != corrupted_tokens.shape:
        raise ValueError(
            f"clean_tokens shape {tuple(clean_tokens.shape)} != "
            f"corrupted_tokens shape {tuple(corrupted_tokens.shape)}."
        )

    n_layers = _get_n_layers(model)
    seq_len = clean_tokens.shape[-1]

    if layers is None:
        layers = list(range(n_layers))
    if positions is None:
        positions = list(range(seq_len))

    # Cache the corrupted activations, run the clean prompt patched with them.
    corrupted_cache, corrupted_logits = _get_clean_cache_and_logits(
        model, corrupted_tokens, hook_name,
    )
    clean_logits = _run_model(model, clean_tokens)

    last_clean = _last_position(clean_logits)
    last_corrupted = _last_position(corrupted_logits)
    clean_logit_val = float(last_clean[answer_token_id].item())
    corrupted_logit_val = float(last_corrupted[answer_token_id].item())

    effect_grid = np.zeros((len(layers), len(positions)), dtype=np.float32)

    for li, layer in enumerate(layers):
        for pi, pos in enumerate(positions):
            patched_logits = patch_residual_stream(
                model,
                clean_tokens,
                corrupted_cache,
                patch_layer=layer,
                patch_position=pos,
                hook_name=hook_name,
            )
            # Same metric formula; here a value near 0 means "still clean"
            # and a value near 1 means "fully corrupted by this patch".
            effect = compute_patch_metric(
                clean_logits=clean_logits,
                corrupted_logits=corrupted_logits,
                patched_logits=patched_logits,
                answer_token_id=answer_token_id,
                metric=metric,
            )
            effect_grid[li, pi] = effect

    flat_idx = int(np.argmax(effect_grid))
    best_li, best_pi = np.unravel_index(flat_idx, effect_grid.shape)

    return {
        "effect_grid": effect_grid,
        "clean_logit": clean_logit_val,
        "corrupted_logit": corrupted_logit_val,
        "best_layer": int(layers[best_li]),
        "best_position": int(positions[best_pi]),
        "layers": list(layers),
        "positions": list(positions),
        "metric": metric,
    }


# ---------------------------------------------------------------------------
# Internal: cache + plain forward
# ---------------------------------------------------------------------------

@torch.no_grad()
def _get_clean_cache_and_logits(
    model: Any,
    tokens: Tensor,
    hook_name: str,
) -> tuple[dict[str, Tensor], Tensor]:
    """Get a residual-stream cache plus logits for ``tokens``.

    For TransformerLens, returns ``model.run_with_cache``'s dict directly.
    For HuggingFace, runs with ``output_hidden_states=True`` and packages
    the hidden states into a ``{layer_index: tensor}`` dict.
    """
    if _is_transformerlens(model):
        logits, cache = model.run_with_cache(tokens)
        # Cast to a plain dict — TL caches are dict-like already.
        return dict(cache), logits

    # HF path
    outputs = model(tokens, output_hidden_states=True)
    logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
    hidden_states = outputs.hidden_states  # tuple, len = n_layers + 1

    # hidden_states[0] is embeddings; hidden_states[i+1] is post-layer-i.
    # We want post-layer-i for layer index i.
    cache: dict[str, Tensor] = {}
    for layer_idx in range(len(hidden_states) - 1):
        cache[layer_idx] = hidden_states[layer_idx + 1]
    return cache, logits


@torch.no_grad()
def _run_model(model: Any, tokens: Tensor) -> Tensor:
    """Run a forward pass and return logits, for either backend."""
    if _is_transformerlens(model):
        return model(tokens)
    outputs = model(tokens)
    if hasattr(outputs, "logits"):
        return outputs.logits
    if isinstance(outputs, tuple):
        return outputs[0]
    return outputs
