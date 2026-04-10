"""Logit lens: project residual stream activations through the unembedding.

The logit lens (nostalgebraist, 2020) asks: "What would the model predict if
it stopped thinking at layer L?" Conceptually, we take the residual stream
after layer L, apply the model's final layer norm, and project through the
unembedding matrix ``W_U`` to get a distribution over the vocabulary. This
gives a per-layer view of how the model's prediction crystallizes as it goes
deeper.

Works with both TransformerLens HookedTransformers (primary, via
``run_with_cache`` and ``model.unembed`` / ``model.ln_final``) and
HuggingFace causal LMs (fallback, via ``output_hidden_states=True`` and
``model.lm_head`` / the architecture's final layer norm).

Usage:
    from src.techniques.logit_lens import (
        logit_lens_sweep,
        top_tokens_per_layer,
        target_token_trajectory,
        project_to_logits,
    )

    sweep = logit_lens_sweep(
        model, tokenizer,
        prompt="The Eiffel Tower is in",
        target_token=" Paris",
        top_k=10,
    )
    for layer, tokens in top_tokens_per_layer(sweep, top_k=5):
        print(layer, tokens)

    traj = target_token_trajectory(sweep)
    print(traj["layers"], traj["probs"], traj["ranks"])

The tuned lens (Belrose et al., 2023) improves on the plain logit lens via a
learned per-layer affine transform. Only the inference stub is provided here;
pass a loaded ``tuned-lens`` checkpoint via ``tuned_lens_project`` or install
the ``tuned-lens`` package for the full version.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from torch import Tensor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class LogitLensResult:
    """Result of a single logit-lens readout at one layer and one position.

    Attributes:
        layer: Zero-indexed layer this readout is from (-1 for embeddings,
            0..n_layers-1 for post-block residuals).
        position: Token position the readout was taken at (resolved from
            negative indices to absolute positions).
        top_tokens: List of ``(token_string, probability)`` tuples, sorted
            by probability descending.
        target_token_id: Token id of the user-specified target, if any.
        target_token_prob: Probability of the target token under this layer's
            logit-lens distribution.
        target_token_rank: Rank of the target token (0 = top prediction).
        full_logits: Optional full vocabulary logits tensor, kept only when
            ``keep_full_logits=True`` was passed to ``logit_lens_sweep``.
    """

    layer: int
    position: int
    top_tokens: list[tuple[str, float]]
    target_token_id: int | None = None
    target_token_prob: float | None = None
    target_token_rank: int | None = None
    full_logits: torch.Tensor | None = None


# ---------------------------------------------------------------------------
# Public API: project residual through unembedding
# ---------------------------------------------------------------------------

def project_to_logits(
    model: Any,
    residual: Tensor,
    apply_final_ln: bool = True,
) -> Tensor:
    """Project residual stream activations through the unembedding.

    For TransformerLens: uses ``model.unembed`` and (optionally)
    ``model.ln_final``.
    For HuggingFace: uses ``model.lm_head`` and (optionally) the final layer
    norm located at ``model.transformer.ln_f``, ``model.model.norm``,
    ``model.gpt_neox.final_layer_norm``, or ``model.model.decoder.final_layer_norm``.

    Accepts residuals of shape ``(d_model,)``, ``(seq, d_model)``, or
    ``(batch, seq, d_model)``; returns ``(..., vocab_size)`` with the leading
    dimensions preserved.

    Args:
        model: A TransformerLens HookedTransformer, a HuggingFace causal LM,
            or any object exposing the duck-typed attributes described above.
        residual: Residual stream activations.
        apply_final_ln: Whether to apply the model's final layer norm before
            the unembed. Matches how the real forward pass consumes these
            activations, so defaults to True.

    Returns:
        Logits tensor with the same leading dims as ``residual`` and a trailing
        vocabulary dimension.

    Raises:
        RuntimeError: If the unembedding (or final layer norm, when requested)
            cannot be located on the provided model.
    """
    if residual.ndim == 0:
        raise ValueError("residual must be at least 1-D (d_model,)")

    # -- Resolve unembedding + final layer norm ----------------------------
    unembed_module, ln_final = _resolve_unembed_and_ln(model)

    # Match dtype/device of the unembedding parameters to avoid silent casts.
    param = next(unembed_module.parameters(), None)
    if param is not None:
        residual = residual.to(device=param.device, dtype=param.dtype)

    x = residual
    if apply_final_ln:
        if ln_final is None:
            logger.warning(
                "apply_final_ln=True but no final layer norm found on model; "
                "skipping ln.",
            )
        else:
            x = ln_final(x)

    logits = unembed_module(x)
    return logits


# ---------------------------------------------------------------------------
# Public API: full per-layer sweep
# ---------------------------------------------------------------------------

def logit_lens_sweep(
    model: Any,
    tokenizer: Any,
    prompt: str,
    target_position: int = -1,
    target_token: str | int | None = None,
    layers: list[int] | None = None,
    top_k: int = 10,
    keep_full_logits: bool = False,
    device: str | None = None,
) -> dict[int, LogitLensResult]:
    """Run a model on ``prompt`` and take a logit-lens readout at every layer.

    Extracts the residual stream after every transformer block, projects it
    through the unembedding, and collects the top-k next-token predictions at
    ``target_position``. If a ``target_token`` is provided, also records its
    rank and probability at each layer.

    Args:
        model: A TransformerLens HookedTransformer or a HuggingFace causal LM.
        tokenizer: The model's tokenizer.
        prompt: Input prompt to run through the model.
        target_position: Sequence position to read out. Negative indices are
            resolved from the end (default ``-1`` = last prompt token).
        target_token: Optional target token. May be a string (tokenized with
            ``tokenizer.encode(..., add_special_tokens=False)`` — the first
            token is used) or an int token id.
        layers: Which block indices to read out. ``None`` means all blocks.
        top_k: How many top tokens to store per layer.
        keep_full_logits: If True, stash the full vocabulary logits tensor on
            each ``LogitLensResult``. Memory-heavy; off by default.
        device: Device to run the model on. If ``None``, inferred from the
            model's parameters.

    Returns:
        Dict mapping ``layer_index -> LogitLensResult``. Layer indices run
        from 0 to ``n_layers - 1`` (post-block residual streams).
    """
    if device is None:
        device = _infer_device(model)

    # Resolve target token id (if any) up front so we can use it in every layer.
    target_token_id: int | None = None
    if target_token is not None:
        target_token_id = _resolve_target_token_id(tokenizer, target_token)

    # Extract per-layer residual streams.
    residuals = _collect_residuals(model, tokenizer, prompt, device)
    n_layers = len(residuals)

    if layers is None:
        layers_list = list(range(n_layers))
    else:
        layers_list = [int(l) for l in layers if 0 <= int(l) < n_layers]

    # Resolve the target position (may be negative) against the actual
    # sequence length. All residuals share the same seq length, so use [0].
    seq_len = residuals[0].shape[-2]
    abs_position = target_position if target_position >= 0 else seq_len + target_position
    if not (0 <= abs_position < seq_len):
        raise IndexError(
            f"target_position {target_position} out of range for seq_len {seq_len}"
        )

    results: dict[int, LogitLensResult] = {}
    with torch.no_grad():
        for layer_idx in layers_list:
            resid = residuals[layer_idx]  # (batch, seq, d_model) or (seq, d_model)

            # Select the target position's residual. Keep trailing d_model.
            if resid.ndim == 3:
                pos_residual = resid[0, abs_position, :]
            elif resid.ndim == 2:
                pos_residual = resid[abs_position, :]
            else:
                raise ValueError(
                    f"Unexpected residual shape at layer {layer_idx}: {tuple(resid.shape)}"
                )

            logits = project_to_logits(model, pos_residual, apply_final_ln=True)
            # logits shape: (vocab,)
            probs = torch.softmax(logits.float(), dim=-1)

            effective_k = min(top_k, probs.shape[-1])
            topk = torch.topk(probs, k=effective_k)
            top_ids = topk.indices.tolist()
            top_ps = topk.values.tolist()

            top_tokens: list[tuple[str, float]] = []
            for tid, p in zip(top_ids, top_ps):
                tok_str = _decode_single_token(tokenizer, int(tid))
                top_tokens.append((tok_str, float(p)))

            # Target-token stats
            target_prob: float | None = None
            target_rank: int | None = None
            if target_token_id is not None:
                target_prob = float(probs[target_token_id].item())
                # Rank = number of tokens with strictly higher prob.
                target_rank = int((probs > probs[target_token_id]).sum().item())

            results[layer_idx] = LogitLensResult(
                layer=layer_idx,
                position=abs_position,
                top_tokens=top_tokens,
                target_token_id=target_token_id,
                target_token_prob=target_prob,
                target_token_rank=target_rank,
                full_logits=logits.detach().cpu() if keep_full_logits else None,
            )

    logger.info(
        "logit_lens_sweep: %d layers at position %d (target=%s)",
        len(results), abs_position, target_token_id,
    )
    return results


# ---------------------------------------------------------------------------
# Public API: summary helpers
# ---------------------------------------------------------------------------

def top_tokens_per_layer(
    sweep: dict[int, LogitLensResult],
    top_k: int = 5,
) -> list[tuple[int, list[tuple[str, float]]]]:
    """Compact summary: per-layer top-k tokens, sorted by layer index.

    Args:
        sweep: Output of ``logit_lens_sweep``.
        top_k: How many tokens to keep per layer (clamped to what each
            ``LogitLensResult.top_tokens`` already contains).

    Returns:
        List of ``(layer_index, [(token, prob), ...])`` tuples, sorted by
        layer index ascending.
    """
    out: list[tuple[int, list[tuple[str, float]]]] = []
    for layer in sorted(sweep.keys()):
        tokens = sweep[layer].top_tokens[:top_k]
        out.append((layer, tokens))
    return out


def target_token_trajectory(
    sweep: dict[int, LogitLensResult],
) -> dict[str, list]:
    """Probability / rank of the target token as a function of layer.

    Shows how a specific token's prediction strength evolves across depth,
    which is often the key quantity for logit-lens analyses.

    Args:
        sweep: Output of ``logit_lens_sweep`` run with a non-None
            ``target_token``.

    Returns:
        Dict with three parallel lists of the same length:

        - ``"layers"``: layer indices (sorted ascending)
        - ``"probs"``: target-token probability at each layer
        - ``"ranks"``: target-token rank at each layer (0 = top prediction)

    Raises:
        ValueError: If the sweep was not run with a target token set.
    """
    layers_sorted = sorted(sweep.keys())
    if not layers_sorted:
        return {"layers": [], "probs": [], "ranks": []}

    first = sweep[layers_sorted[0]]
    if first.target_token_id is None:
        raise ValueError(
            "target_token_trajectory requires a sweep run with target_token set; "
            "none of the LogitLensResults carry a target_token_id."
        )

    layers_out: list[int] = []
    probs_out: list[float] = []
    ranks_out: list[int] = []
    for layer in layers_sorted:
        r = sweep[layer]
        if r.target_token_prob is None or r.target_token_rank is None:
            # Should not happen if first had a target, but be defensive.
            continue
        layers_out.append(layer)
        probs_out.append(float(r.target_token_prob))
        ranks_out.append(int(r.target_token_rank))

    return {"layers": layers_out, "probs": probs_out, "ranks": ranks_out}


# ---------------------------------------------------------------------------
# Public API: tuned lens (stub / inference)
# ---------------------------------------------------------------------------

def tuned_lens_project(
    model: Any,
    residual: Tensor,
    layer: int,
    tuned_lens_checkpoint: Any | None = None,
) -> Tensor:
    """Tuned lens projection (Belrose et al. 2023).

    The tuned lens applies a learned per-layer affine transform to the
    residual before the unembed, which fixes some of the "late-layer bias"
    of the plain logit lens. This function delegates to a user-provided
    checkpoint: either an object exposing a ``forward(residual, layer)`` /
    callable interface, or a dict with per-layer ``weight`` and ``bias``
    tensors keyed by layer index.

    Args:
        model: The model (used only to find the unembedding if the checkpoint
            produces a transformed residual rather than logits directly).
        residual: Residual stream activations of shape ``(..., d_model)``.
        layer: Zero-indexed layer the residual came from.
        tuned_lens_checkpoint: A loaded tuned-lens checkpoint. If ``None``,
            raises ``NotImplementedError`` pointing at the ``tuned-lens``
            package.

    Returns:
        Logits tensor of shape ``(..., vocab_size)``.

    Raises:
        NotImplementedError: If no checkpoint is provided.
    """
    if tuned_lens_checkpoint is None:
        raise NotImplementedError(
            "tuned_lens_project: no checkpoint provided. Install the "
            "`tuned-lens` package (https://github.com/AlignmentResearch/tuned-lens) "
            "and pass a loaded TunedLens object as `tuned_lens_checkpoint`, or "
            "supply a dict of {layer: {'weight': Tensor, 'bias': Tensor}}."
        )

    # Callable / nn.Module checkpoint: let it do the full projection.
    if callable(tuned_lens_checkpoint):
        try:
            return tuned_lens_checkpoint(residual, layer)
        except TypeError:
            # Fall through to dict-style handling.
            pass

    # Dict checkpoint: apply learned affine, then plain unembed.
    if isinstance(tuned_lens_checkpoint, dict) and layer in tuned_lens_checkpoint:
        params = tuned_lens_checkpoint[layer]
        weight: Tensor = params["weight"]
        bias: Tensor | None = params.get("bias", None)
        transformed = residual @ weight.T
        if bias is not None:
            transformed = transformed + bias
        return project_to_logits(model, transformed, apply_final_ln=True)

    raise NotImplementedError(
        f"tuned_lens_project: unrecognized checkpoint type "
        f"{type(tuned_lens_checkpoint).__name__}. Provide a callable lens, "
        f"or a dict mapping layer index -> {{'weight','bias'}}."
    )


# ---------------------------------------------------------------------------
# Internals: model introspection
# ---------------------------------------------------------------------------

def _is_transformerlens(model: Any) -> bool:
    """Duck-type check for a TransformerLens HookedTransformer."""
    return hasattr(model, "run_with_cache") and hasattr(model, "cfg") and hasattr(model, "unembed")


def _resolve_unembed_and_ln(model: Any) -> tuple[Any, Any | None]:
    """Find the unembedding module and (optional) final layer norm on a model.

    Tries TransformerLens conventions first, then common HuggingFace
    architectures (Llama/Qwen/Gemma/GPT-2/NeoX/OPT), then a duck-typed fallback
    that searches for a ``lm_head`` / ``unembed`` attribute anywhere on the
    object.
    """
    # TransformerLens
    if hasattr(model, "unembed"):
        ln_final = getattr(model, "ln_final", None)
        return model.unembed, ln_final

    # HuggingFace: lm_head is the unembedding
    lm_head = getattr(model, "lm_head", None)
    if lm_head is None:
        raise RuntimeError(
            "project_to_logits: could not find an unembedding on the model "
            "(expected `unembed` or `lm_head`)."
        )

    # Try common locations for the final layer norm.
    ln_final: Any | None = None
    for attr_path in (
        "transformer.ln_f",           # GPT-2, GPT-Neo
        "model.norm",                 # Llama, Qwen, Gemma
        "gpt_neox.final_layer_norm",  # GPT-NeoX, Pythia
        "model.decoder.final_layer_norm",  # OPT
    ):
        obj = model
        ok = True
        for part in attr_path.split("."):
            if not hasattr(obj, part):
                ok = False
                break
            obj = getattr(obj, part)
        if ok:
            ln_final = obj
            break

    return lm_head, ln_final


def _infer_device(model: Any) -> str:
    """Pick a device string by inspecting model parameters."""
    try:
        p = next(model.parameters())
        return str(p.device)
    except (StopIteration, AttributeError):
        return "cpu"


def _collect_residuals(
    model: Any,
    tokenizer: Any,
    prompt: str,
    device: str,
) -> list[Tensor]:
    """Return a list of per-layer residual streams (len = n_layers).

    Each entry is shaped ``(batch, seq, d_model)`` or ``(seq, d_model)``.
    Uses TransformerLens ``run_with_cache`` when available, else
    HuggingFace ``output_hidden_states=True``.
    """
    if _is_transformerlens(model):
        tokens = model.to_tokens(prompt) if hasattr(model, "to_tokens") else (
            tokenizer.encode(prompt, return_tensors="pt").to(device)
        )
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens)
        n_layers = int(model.cfg.n_layers)
        residuals: list[Tensor] = []
        for layer in range(n_layers):
            key = f"blocks.{layer}.hook_resid_post"
            if key not in cache:
                # Fall back to pre-attn for layer 0 if resid_post missing.
                alt = f"blocks.{layer}.hook_resid_pre"
                if alt not in cache:
                    raise RuntimeError(
                        f"Neither {key} nor {alt} found in TL cache."
                    )
                residuals.append(cache[alt])
            else:
                residuals.append(cache[key])
        return residuals

    # HuggingFace path
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, return_dict=True)
    hidden_states = outputs.hidden_states  # tuple: (embeds, block_0, ..., block_{L-1})
    if hidden_states is None:
        raise RuntimeError(
            "HuggingFace model did not return hidden_states. "
            "Make sure `output_hidden_states=True` is supported."
        )
    # Drop the embedding layer; keep post-block states only.
    return [hs for hs in hidden_states[1:]]


def _resolve_target_token_id(tokenizer: Any, target_token: str | int) -> int:
    """Resolve a target token (string or id) to a single vocab id."""
    if isinstance(target_token, int):
        return int(target_token)
    if not isinstance(target_token, str):
        raise TypeError(
            f"target_token must be str or int, got {type(target_token).__name__}"
        )
    # Prefer add_special_tokens=False so leading BOS tokens don't sneak in.
    try:
        ids = tokenizer.encode(target_token, add_special_tokens=False)
    except TypeError:
        ids = tokenizer.encode(target_token)
    if not ids:
        raise ValueError(f"Tokenizer returned no ids for target token {target_token!r}")
    if len(ids) > 1:
        logger.debug(
            "target_token %r tokenizes to %d ids; using the first (%d).",
            target_token, len(ids), ids[0],
        )
    return int(ids[0])


def _decode_single_token(tokenizer: Any, token_id: int) -> str:
    """Decode a single token id to a display string."""
    # Try convert_ids_to_tokens first (keeps BPE markers like Ġ visible),
    # falling back to decode for tokenizers that don't implement it.
    fn = getattr(tokenizer, "convert_ids_to_tokens", None)
    if fn is not None:
        try:
            tok = fn(token_id)
            if isinstance(tok, list):
                tok = tok[0]
            return str(tok)
        except Exception:
            pass
    try:
        return tokenizer.decode([token_id])
    except Exception:
        return f"<id:{token_id}>"
