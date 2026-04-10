"""Attention pattern analysis and head attribution.

Provides utilities for extracting attention patterns from a model, computing
per-head attribution to a target metric, ranking heads, measuring per-head
attention entropy, and detecting induction heads via the standard repeated
random sequence test.

Most extraction works with both TransformerLens (preferred) and HuggingFace
models. Causal head ablation and the induction head test are TransformerLens
only because they require named hooks (``hook_z`` / ``hook_pattern``).

Usage:
    from src.techniques.attention import (
        extract_attention_patterns,
        compute_head_attribution,
        top_attention_heads,
        attention_entropy,
        induction_head_score,
    )

    # Extract attention patterns
    patterns = extract_attention_patterns(model, "Hello world", layers=[0, 1, 2])
    print(patterns.n_layers, patterns.n_heads, patterns.tokens)

    # Per-head attribution to a logit-difference metric (TL only)
    def logit_diff(logits):
        return logits[0, -1, target_id] - logits[0, -1, distractor_id]

    attribution = compute_head_attribution(model, tokens, logit_diff)
    top = top_attention_heads(attribution, k=5)

    # Entropy of attention distributions
    entropies = attention_entropy(patterns.patterns)

    # Induction head detection (TL only)
    induction_scores = induction_head_score(model, seq_len=50, context_length=25)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

import numpy as np
import torch
from torch import Tensor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class AttentionPatterns:
    """Attention patterns extracted from a model for one or more layers.

    Attributes:
        patterns: Mapping from layer index to an attention tensor of shape
            ``(n_heads, seq_len, seq_len)`` for a single prompt, or
            ``(batch, n_heads, seq_len, seq_len)`` for a batch of prompts.
        tokens: Decoded tokens corresponding to the (last) prompt's
            ``seq_len`` axis. Used for downstream interpretability.
        n_layers: Total number of layers in the source model.
        n_heads: Number of attention heads per layer.
    """

    patterns: dict[int, Tensor]
    tokens: list[str] = field(default_factory=list)
    n_layers: int = 0
    n_heads: int = 0


# ---------------------------------------------------------------------------
# Public API: pattern extraction
# ---------------------------------------------------------------------------

def extract_attention_patterns(
    model: Any,
    prompt: str | list[str],
    layers: list[int] | None = None,
) -> AttentionPatterns:
    """Run the model and return attention patterns at the specified layers.

    For TransformerLens models, this uses ``run_with_cache`` with a
    ``names_filter`` for ``blocks.{L}.attn.hook_pattern`` keys.

    For HuggingFace models, this calls the model with
    ``output_attentions=True`` and extracts ``outputs.attentions[layer]``.

    Args:
        model: A TransformerLens HookedTransformer or HuggingFace causal LM.
            Tokenization is handled internally: TL uses ``model.to_tokens`` and
            ``model.to_str_tokens``; HF uses ``model.config`` plus a tokenizer
            attribute (``model.tokenizer``) if present.
        prompt: A single prompt string or a list of prompt strings (batch).
        layers: Which layer indices to extract. ``None`` extracts every layer.

    Returns:
        AttentionPatterns. For a single prompt, each tensor has shape
        ``(n_heads, seq_len, seq_len)``. For a batched prompt, each tensor
        has shape ``(batch, n_heads, seq_len, seq_len)``.

    Raises:
        RuntimeError: If a HuggingFace model is supplied without a usable
            tokenizer attribute.
    """
    is_batched = isinstance(prompt, list)

    if _is_transformerlens(model):
        return _tl_extract_patterns(model, prompt, layers, is_batched)
    return _hf_extract_patterns(model, prompt, layers, is_batched)


def _tl_extract_patterns(
    model: Any,
    prompt: str | list[str],
    layers: list[int] | None,
    is_batched: bool,
) -> AttentionPatterns:
    """Extract attention patterns via TransformerLens hook_pattern."""
    cfg = model.cfg
    n_layers = cfg.n_layers
    n_heads = cfg.n_heads

    if layers is None:
        layers = list(range(n_layers))

    hook_names = {f"blocks.{L}.attn.hook_pattern" for L in layers}

    def names_filter(name: str) -> bool:
        return name in hook_names

    # Tokenize via model utilities
    if is_batched:
        tokens = model.to_tokens(prompt)
    else:
        tokens = model.to_tokens(prompt)

    with torch.no_grad():
        _, cache = model.run_with_cache(tokens, names_filter=names_filter)

    patterns: dict[int, Tensor] = {}
    for L in layers:
        # cache returns shape (batch, n_heads, seq, seq)
        raw = cache[f"blocks.{L}.attn.hook_pattern"].detach()
        if is_batched:
            patterns[L] = raw
        else:
            patterns[L] = raw[0]  # collapse the batch dim

    # Decode tokens for the (first) prompt for interpretability
    try:
        if is_batched:
            decoded = model.to_str_tokens(prompt[0])
        else:
            decoded = model.to_str_tokens(prompt)
        decoded = [str(t) for t in decoded]
    except Exception as exc:  # pragma: no cover - best-effort decoding
        logger.warning("Could not decode tokens via to_str_tokens: %s", exc)
        decoded = []

    logger.info(
        "Extracted attention patterns from %d/%d layers (n_heads=%d, seq_len=%d)",
        len(layers), n_layers, n_heads,
        patterns[layers[0]].shape[-1] if layers else 0,
    )

    return AttentionPatterns(
        patterns=patterns,
        tokens=decoded,
        n_layers=n_layers,
        n_heads=n_heads,
    )


def _hf_extract_patterns(
    model: Any,
    prompt: str | list[str],
    layers: list[int] | None,
    is_batched: bool,
) -> AttentionPatterns:
    """Extract attention patterns via HuggingFace output_attentions=True."""
    tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is None:
        raise RuntimeError(
            "HuggingFace attention extraction requires model.tokenizer to be set. "
            "Attach a tokenizer to the model object before calling this function."
        )

    inputs = tokenizer(prompt, return_tensors="pt", padding=is_batched)
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True, return_dict=True)

    # outputs.attentions: tuple of length n_layers, each (batch, n_heads, seq, seq)
    attentions = outputs.attentions
    if attentions is None:
        raise RuntimeError(
            "HuggingFace model returned no attentions; ensure the model supports "
            "output_attentions and is not using FlashAttention."
        )

    n_layers = len(attentions)
    n_heads = attentions[0].shape[1]

    if layers is None:
        layers = list(range(n_layers))

    patterns: dict[int, Tensor] = {}
    for L in layers:
        raw = attentions[L].detach()
        if is_batched:
            patterns[L] = raw
        else:
            patterns[L] = raw[0]

    # Decode tokens for the first prompt
    first_ids = inputs["input_ids"][0]
    decoded = tokenizer.convert_ids_to_tokens(first_ids.tolist())
    decoded = [str(t) for t in decoded]

    logger.info(
        "Extracted attention patterns from %d/%d HF layers (n_heads=%d)",
        len(layers), n_layers, n_heads,
    )

    return AttentionPatterns(
        patterns=patterns,
        tokens=decoded,
        n_layers=n_layers,
        n_heads=n_heads,
    )


# ---------------------------------------------------------------------------
# Public API: head attribution
# ---------------------------------------------------------------------------

def compute_head_attribution(
    model: Any,
    tokens: Tensor,
    metric_fn: Callable[[Tensor], Tensor | float],
    ablation: Literal["zero", "mean"] = "zero",
    layers: list[int] | None = None,
    heads: list[int] | None = None,
) -> dict[tuple[int, int], float]:
    """Compute per-head attribution by ablating each head and measuring the
    metric drop.

    For each (layer, head) in the sweep, this runs the model twice: once with
    no intervention (baseline), and once with the head's ``hook_z`` output
    ablated. The attribution score is ``baseline_metric - ablated_metric``;
    higher means the head is more important for the metric.

    Args:
        model: A TransformerLens HookedTransformer. Required because the
            ablation uses ``blocks.{L}.attn.hook_z``.
        tokens: Input token IDs of shape ``(batch, seq_len)`` ready to be
            passed to ``model.run_with_hooks``.
        metric_fn: Callable that takes a logits tensor of shape
            ``(batch, seq_len, vocab)`` and returns a scalar (Tensor or float).
        ablation: ``"zero"`` replaces the head's z output with zeros.
            ``"mean"`` replaces it with the mean across the batch and
            sequence axes (a less destructive baseline).
        layers: Which layer indices to ablate. ``None`` = all layers.
        heads: Which head indices to ablate. ``None`` = all heads.

    Returns:
        Dict mapping ``(layer, head)`` to attribution score.

    Raises:
        RuntimeError: If the model is not a TransformerLens HookedTransformer.
    """
    if not _is_transformerlens(model):
        raise RuntimeError(
            "compute_head_attribution requires a TransformerLens HookedTransformer. "
            "HuggingFace models are not supported because the function uses "
            "blocks.{L}.attn.hook_z to ablate per-head outputs."
        )

    cfg = model.cfg
    n_layers = cfg.n_layers
    n_heads = cfg.n_heads

    if layers is None:
        layers = list(range(n_layers))
    if heads is None:
        heads = list(range(n_heads))

    # -- Baseline pass ----------------------------------------------------
    with torch.no_grad():
        baseline_logits = model(tokens)
    baseline_metric = _to_float(metric_fn(baseline_logits))

    attribution: dict[tuple[int, int], float] = {}

    for L in layers:
        hook_name = f"blocks.{L}.attn.hook_z"

        for H in heads:
            def make_hook(head_idx: int):
                def hook_fn(z: Tensor, hook: Any) -> Tensor:
                    # z shape: (batch, seq, n_heads, d_head)
                    z = z.clone()
                    if ablation == "zero":
                        z[:, :, head_idx, :] = 0.0
                    elif ablation == "mean":
                        # Replace with mean across batch and sequence
                        mean_val = z[:, :, head_idx, :].mean(
                            dim=(0, 1), keepdim=True
                        )
                        z[:, :, head_idx, :] = mean_val
                    else:
                        raise ValueError(
                            f"Unknown ablation: {ablation!r}. "
                            f"Expected 'zero' or 'mean'."
                        )
                    return z
                return hook_fn

            with torch.no_grad():
                ablated_logits = model.run_with_hooks(
                    tokens,
                    fwd_hooks=[(hook_name, make_hook(H))],
                )
            ablated_metric = _to_float(metric_fn(ablated_logits))
            attribution[(L, H)] = baseline_metric - ablated_metric

    logger.info(
        "Computed head attribution for %d (layer, head) pairs (ablation=%s)",
        len(attribution), ablation,
    )
    return attribution


def top_attention_heads(
    attribution: dict[tuple[int, int], float],
    k: int = 10,
) -> list[tuple[int, int, float]]:
    """Return the top-k heads sorted by attribution score (descending).

    Args:
        attribution: Dict mapping ``(layer, head)`` to a numeric attribution
            score (typically from :func:`compute_head_attribution`).
        k: Maximum number of heads to return.

    Returns:
        List of ``(layer, head, score)`` tuples, sorted by score descending.
        If ``attribution`` is empty, returns an empty list.
    """
    if not attribution:
        return []

    items = [(L, H, float(s)) for (L, H), s in attribution.items()]
    items.sort(key=lambda x: x[2], reverse=True)
    return items[:k]


# ---------------------------------------------------------------------------
# Public API: attention entropy
# ---------------------------------------------------------------------------

def attention_entropy(
    patterns: dict[int, Tensor],
) -> dict[int, Tensor]:
    """Compute Shannon entropy of attention distributions per (head, query).

    For each layer, computes ``-sum_j p_ij * log(p_ij)`` along the key axis
    for every (head, query position) pair. Higher entropy means a more
    diffuse / uniform attention pattern; lower entropy means more focused.

    Supports both single-prompt patterns of shape ``(n_heads, seq, seq)`` and
    batched patterns of shape ``(batch, n_heads, seq, seq)``.

    Args:
        patterns: Dict mapping layer index to an attention tensor.

    Returns:
        Dict mapping layer index to a per-query entropy tensor of shape
        ``(n_heads, seq)`` (single prompt) or ``(batch, n_heads, seq)``
        (batched).
    """
    eps = 1e-12
    out: dict[int, Tensor] = {}
    for layer, attn in patterns.items():
        # Sum over the last (key) axis
        p = attn.detach().float()
        log_p = torch.log(p.clamp(min=eps))
        entropy = -(p * log_p).sum(dim=-1)
        out[layer] = entropy
    return out


# ---------------------------------------------------------------------------
# Public API: induction head detection
# ---------------------------------------------------------------------------

def induction_head_score(
    model: Any,
    seq_len: int = 50,
    context_length: int = 25,
    n_seqs: int = 10,
    seed: int = 42,
) -> dict[tuple[int, int], float]:
    """Detect induction heads via the standard repeated random sequence test.

    Builds ``n_seqs`` sequences of the form ``[r_1, r_2, ..., r_K, r_1,
    r_2, ..., r_K]`` where ``K = context_length`` and ``r_i`` are uniformly
    random token IDs. An induction head's attention from position ``K + i``
    should peak on position ``i + 1`` (the token that *followed* ``r_i`` in
    the first half).

    The score for a head is the average attention weight from each query
    position ``q`` in the second half to the corresponding induction key
    position ``q - K + 1``, averaged over (sequences, query positions). The
    score lies in ``[0, 1]``; values near 1 indicate a strong induction head.

    Args:
        model: A TransformerLens HookedTransformer. Required.
        seq_len: Total sequence length passed to the model. Must be at least
            ``2 * context_length``.
        context_length: Period of the repeating block (the ``K`` above).
        n_seqs: Number of independent random sequences to average over.
        seed: Random seed for reproducibility.

    Returns:
        Dict mapping ``(layer, head)`` to induction score in ``[0, 1]``.

    Raises:
        RuntimeError: If the model is not a TransformerLens HookedTransformer.
        ValueError: If ``seq_len < 2 * context_length`` or ``context_length < 2``.
    """
    if not _is_transformerlens(model):
        raise RuntimeError(
            "induction_head_score requires a TransformerLens HookedTransformer."
        )
    if context_length < 2:
        raise ValueError(f"context_length must be >= 2, got {context_length}")
    if seq_len < 2 * context_length:
        raise ValueError(
            f"seq_len ({seq_len}) must be >= 2 * context_length "
            f"({2 * context_length})"
        )

    cfg = model.cfg
    n_layers = cfg.n_layers
    n_heads = cfg.n_heads
    vocab_size = cfg.d_vocab

    gen = torch.Generator().manual_seed(seed)

    # Build n_seqs sequences of repeated random tokens
    K = context_length
    # Each row: [r_1..r_K, r_1..r_K, padding to seq_len with random tokens]
    half = torch.randint(0, vocab_size, (n_seqs, K), generator=gen)
    repeated = torch.cat([half, half], dim=1)  # (n_seqs, 2K)
    if seq_len > 2 * K:
        pad = torch.randint(
            0, vocab_size, (n_seqs, seq_len - 2 * K), generator=gen
        )
        tokens = torch.cat([repeated, pad], dim=1)
    else:
        tokens = repeated  # exactly 2K

    device = next(model.parameters()).device
    tokens = tokens.to(device)

    # Run with cache to grab all hook_pattern outputs
    pattern_names = {f"blocks.{L}.attn.hook_pattern" for L in range(n_layers)}

    def names_filter(name: str) -> bool:
        return name in pattern_names

    with torch.no_grad():
        _, cache = model.run_with_cache(tokens, names_filter=names_filter)

    # For each head, average the attention weight from query position
    # K + i (for i in 1..K-1) to key position i. We index q in [K+1, 2K-1]
    # so the induction key i = q - K is in [1, K-1] (skipping i=0 which has
    # no preceding token to "induct").
    query_positions = torch.arange(K + 1, 2 * K, device=device)
    key_positions = query_positions - K

    scores: dict[tuple[int, int], float] = {}
    for L in range(n_layers):
        attn = cache[f"blocks.{L}.attn.hook_pattern"]  # (batch, heads, seq, seq)
        # Gather attention[batch, head, q, k] for our (q, k) pairs
        sub = attn[:, :, query_positions, :][..., key_positions]
        # ``sub`` shape: (batch, heads, len(q), len(k)) — we want only the
        # diagonal (i.e. q_i ↔ k_i), so use take_along_dim or just slice
        # via arange.
        diag_idx = torch.arange(query_positions.numel(), device=device)
        induction_attn = sub[:, :, diag_idx, diag_idx]  # (batch, heads, len(q))

        # Average over batch and query positions
        head_scores = induction_attn.mean(dim=(0, 2))  # (heads,)
        for H in range(n_heads):
            scores[(L, H)] = float(head_scores[H].item())

    logger.info(
        "Computed induction scores for %d (layer, head) pairs over %d seqs",
        len(scores), n_seqs,
    )
    return scores


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_transformerlens(model: Any) -> bool:
    """Best-effort check that ``model`` is a TransformerLens HookedTransformer."""
    return hasattr(model, "run_with_cache") and hasattr(model, "cfg")


def _to_float(value: Tensor | float | int | np.floating | np.integer) -> float:
    """Coerce a metric output to a Python float."""
    if isinstance(value, Tensor):
        return float(value.detach().cpu().item())
    return float(value)
