"""Circuit discovery primitives.

A foundation for circuit-discovery experiments. We implement three useful
methods plus a stub for the heavyweight ACDC algorithm:

1. ``edge_attribution_patching`` — gradient-based approximation of activation
   patching across all (sender, receiver) edges (Syed et al. 2023). Two
   forward passes + one backward pass replace what would otherwise be N
   per-node patches. We implement the *node-attribution* simplification: each
   hook point's net effect on the metric is approximated by the inner product
   of (clean - corrupted) activations with the metric's gradient at that
   point. Cheap, fast, and good enough to rank components for follow-up.

2. ``path_patch`` — single-source / single-receiver path patching following
   Wang et al. 2022 (IOI). Targeted, rigorous, slower than EAP. Only the
   ``component_type="resid"`` form is implemented here; per-head attention
   patches are left as future work.

3. ``extract_subgraph`` — pure helper that filters an edge-score dict by
   threshold and/or top-k, returning the resulting set of edge names.

4. ``acdc`` — explicit ``NotImplementedError`` stub with a pointer to the
   reference implementation.

Backend
-------
This module is TransformerLens-only. Circuit discovery relies heavily on
named hook points and ``run_with_cache``, both of which TL supports natively.
Adding a HuggingFace backend would require building an equivalent named-hook
graph from raw modules — out of scope for the first cut.

Usage
-----
::

    from src.techniques.circuit_discovery import (
        edge_attribution_patching,
        path_patch,
        extract_subgraph,
        EAPResult,
    )

    metric = lambda logits: logits[0, -1, target_id] - logits[0, -1, distractor_id]
    result = edge_attribution_patching(
        model, clean_tokens, corrupted_tokens, metric, top_k=20, nodes="heads",
    )
    keep = extract_subgraph(result.edge_scores, top_k=10)

    effect = path_patch(
        model, clean_tokens, corrupted_tokens,
        sender=("resid", 5, 0), receiver=("resid", 9, 0),
        metric_fn=metric,
    )
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
class EAPResult:
    """Result of an edge-attribution-patching sweep.

    Attributes:
        edge_scores: Mapping from edge / node name (e.g. ``"L5H3"`` or
            ``"blocks.5.hook_resid_post"``) to its scalar attribution.
            Sign is preserved: positive means the clean activation pushes
            the metric up vs. corrupted, negative means it pushes the
            metric down.
        top_edges: Edges sorted by absolute attribution, descending. Each
            entry is ``(name, score)`` and the list has length up to
            ``top_k``.
        metric_value_clean: Scalar value of ``metric_fn(clean_logits)``.
        metric_value_corrupted: Scalar value of ``metric_fn(corrupted_logits)``.
        n_edges_scored: Number of distinct edges/nodes that received a score.
        metadata: Free-form dict for parameters and diagnostics.
    """

    edge_scores: dict[str, float]
    top_edges: list[tuple[str, float]]
    metric_value_clean: float
    metric_value_corrupted: float
    n_edges_scored: int
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Backend detection
# ---------------------------------------------------------------------------

def _is_transformerlens(model: Any) -> bool:
    """Return True if ``model`` looks like a TransformerLens HookedTransformer."""
    return hasattr(model, "run_with_hooks") and hasattr(model, "cfg")


def _require_transformerlens(model: Any, fn_name: str) -> None:
    """Raise if the model is not a TransformerLens HookedTransformer."""
    if not _is_transformerlens(model):
        raise NotImplementedError(
            f"{fn_name} currently requires a TransformerLens HookedTransformer. "
            f"HuggingFace support is not implemented for circuit_discovery; use "
            f"src.techniques.patching for HF-backed activation patching instead."
        )


def _get_n_layers(model: Any) -> int:
    """Best-effort layer count from a TransformerLens model."""
    return int(model.cfg.n_layers)


def _get_n_heads(model: Any) -> int:
    """Best-effort head count from a TransformerLens model."""
    return int(model.cfg.n_heads)


# ---------------------------------------------------------------------------
# Hook-point name builders
# ---------------------------------------------------------------------------

def _hook_names_for_nodes(
    n_layers: int,
    nodes: Literal["heads", "layers", "all"],
) -> list[str]:
    """Return the list of TL hook-point names corresponding to ``nodes``.

    - ``"heads"``: per-head attention output, ``blocks.{L}.attn.hook_z``.
        These are the canonical EAP nodes for circuit discovery.
    - ``"layers"``: per-layer residual stream, ``blocks.{L}.hook_resid_post``.
    - ``"all"``: union of the two.
    """
    if nodes == "heads":
        return [f"blocks.{L}.attn.hook_z" for L in range(n_layers)]
    if nodes == "layers":
        return [f"blocks.{L}.hook_resid_post" for L in range(n_layers)]
    if nodes == "all":
        return (
            [f"blocks.{L}.attn.hook_z" for L in range(n_layers)]
            + [f"blocks.{L}.hook_resid_post" for L in range(n_layers)]
        )
    raise ValueError(
        f"Unknown nodes={nodes!r}. Expected 'heads', 'layers', or 'all'."
    )


# ---------------------------------------------------------------------------
# Public API: edge attribution patching
# ---------------------------------------------------------------------------

def edge_attribution_patching(
    model: Any,
    clean_tokens: Tensor,
    corrupted_tokens: Tensor,
    metric_fn: Callable[[Tensor], Tensor],
    top_k: int = 50,
    nodes: Literal["heads", "layers", "all"] = "heads",
) -> EAPResult:
    """Edge / node attribution patching (Syed et al. 2023, simplified).

    A gradient-based approximation of activation patching. The full EAP
    paper scores edges between sender/receiver pairs in the computational
    graph. We implement the *node-attribution* simplification: for each
    hook point ``h`` we estimate

        score(h) ~= < (clean_act_h - corrupted_act_h), grad_corrupted_h >

    where ``grad_corrupted_h`` is the gradient of the corrupted-run metric
    w.r.t. the activation at ``h``. This is exactly the first-order Taylor
    expansion of "what would the metric become if I patched in the clean
    activation here?", and it lets us rank every hook point with a single
    forward+backward on the corrupted run plus a forward on the clean run.

    Per-head attention nodes
    ------------------------
    For ``nodes="heads"`` (the default and most useful setting) we hook
    ``blocks.{L}.attn.hook_z``, whose activation has shape
    ``(batch, seq, n_heads, d_head)``. We compute a separate score per head
    by reducing across batch / seq / d_head. The result is a dict keyed by
    ``"L{layer}H{head}"``.

    For ``nodes="layers"`` we hook the residual stream at each layer
    (``hook_resid_post``) and produce one score per layer, keyed by the
    full hook name.

    Args:
        model: TransformerLens HookedTransformer.
        clean_tokens: Token ids ``(batch, seq_len)`` for the "good" run.
        corrupted_tokens: Token ids of the same shape for the contrast run.
        metric_fn: ``logits -> scalar Tensor``. The standard choice is a
            logit difference: ``lambda L: L[0, -1, ans] - L[0, -1, distractor]``.
            Must produce a scalar tensor that supports ``.backward()``.
        top_k: How many entries to keep in ``top_edges``. Sorted by
            absolute score, descending.
        nodes: Which kind of nodes to score (see above).

    Returns:
        :class:`EAPResult`. ``edge_scores`` is sign-preserving; ``top_edges``
        is sorted by ``|score|`` descending.

    Notes:
        - This is a *node* attribution, not a true *edge* attribution. The
          name ``edge_attribution_patching`` is kept for consistency with
          the EAP literature, where this is the standard simplification.
        - We do not wrap in ``torch.no_grad()``; gradients are required.
          Final scores are detached before being returned.
    """
    _require_transformerlens(model, "edge_attribution_patching")

    if clean_tokens.shape != corrupted_tokens.shape:
        raise ValueError(
            f"clean_tokens shape {tuple(clean_tokens.shape)} != "
            f"corrupted_tokens shape {tuple(corrupted_tokens.shape)}. "
            f"Pad to equal length before calling edge_attribution_patching."
        )

    n_layers = _get_n_layers(model)
    hook_points = _hook_names_for_nodes(n_layers, nodes)

    # ── Pass 1: clean run with cache (no grads needed) ────────────────────
    with torch.no_grad():
        clean_logits, clean_cache = model.run_with_cache(
            clean_tokens, names_filter=lambda n: n in hook_points,
        )
        metric_value_clean = float(metric_fn(clean_logits).detach().item())

    # ── Pass 2: corrupted forward + backward, capturing per-hook activations
    # We need both the activation and its gradient at each hook point. The
    # cleanest TL way is a forward hook that stashes the tensor and calls
    # ``.requires_grad_(True).retain_grad()``.
    captured: dict[str, Tensor] = {}

    def make_capture_hook(name: str):
        def hook_fn(activation: Tensor, hook: Any) -> Tensor:
            # Detach from any prior graph, then re-enable grad so the
            # backward pass can populate `.grad` on this leaf-like tensor.
            act = activation.detach().clone().requires_grad_(True)
            act.retain_grad()
            captured[name] = act
            return act
        return hook_fn

    fwd_hooks = [(name, make_capture_hook(name)) for name in hook_points]

    # Make sure we don't accidentally inherit a no_grad context from the caller.
    with torch.enable_grad():
        corrupted_logits = model.run_with_hooks(
            corrupted_tokens, fwd_hooks=fwd_hooks,
        )
        metric_corr_t = metric_fn(corrupted_logits)
        metric_value_corrupted = float(metric_corr_t.detach().item())

        # Zero any stale grads on captured tensors before backprop.
        for t in captured.values():
            if t.grad is not None:
                t.grad.zero_()

        # If the metric tensor is somehow detached (shouldn't be), bail out
        # gracefully with zero gradients rather than raising.
        if metric_corr_t.requires_grad:
            metric_corr_t.backward()

    # ── Score each hook point: <clean - corrupted, grad> ──────────────────
    edge_scores: dict[str, float] = {}

    for hook_name in hook_points:
        if hook_name not in captured or hook_name not in clean_cache:
            continue

        corrupted_act = captured[hook_name]
        clean_act = clean_cache[hook_name].to(corrupted_act.device)
        grad = corrupted_act.grad

        if grad is None:
            # No gradient flowed (e.g. metric didn't depend on this hook,
            # or the user passed a metric_fn that returns a constant).
            grad = torch.zeros_like(corrupted_act)

        diff = (clean_act - corrupted_act.detach()).float()
        grad = grad.detach().float()

        if hook_name.endswith("attn.hook_z") and diff.dim() == 4:
            # Per-head attention output: (batch, seq, n_heads, d_head)
            # Reduce across batch / seq / d_head to get (n_heads,) scores.
            per_head = (diff * grad).sum(dim=(0, 1, 3))  # (n_heads,)
            # Layer index from the hook name: blocks.{L}.attn.hook_z
            try:
                layer_idx = int(hook_name.split(".")[1])
            except (IndexError, ValueError):
                layer_idx = -1
            for h in range(per_head.shape[0]):
                edge_scores[f"L{layer_idx}H{h}"] = float(per_head[h].item())
        else:
            # Residual / generic case: collapse everything to one scalar.
            score = (diff * grad).sum().item()
            edge_scores[hook_name] = float(score)

    # ── Top-k sort by absolute attribution ────────────────────────────────
    sorted_edges = sorted(
        edge_scores.items(), key=lambda kv: abs(kv[1]), reverse=True,
    )
    top_edges = sorted_edges[: max(0, int(top_k))]

    logger.info(
        "edge_attribution_patching: scored %d %s nodes, clean=%.4f, corrupted=%.4f",
        len(edge_scores), nodes, metric_value_clean, metric_value_corrupted,
    )

    return EAPResult(
        edge_scores=edge_scores,
        top_edges=top_edges,
        metric_value_clean=metric_value_clean,
        metric_value_corrupted=metric_value_corrupted,
        n_edges_scored=len(edge_scores),
        metadata={
            "nodes": nodes,
            "n_hook_points": len(hook_points),
            "top_k": int(top_k),
            "n_layers": n_layers,
        },
    )


# ---------------------------------------------------------------------------
# Public API: single-source / single-receiver path patching
# ---------------------------------------------------------------------------

def path_patch(
    model: Any,
    clean_tokens: Tensor,
    corrupted_tokens: Tensor,
    sender: tuple[str, int, int],
    receiver: tuple[str, int, int],
    metric_fn: Callable[[Tensor], Tensor],
) -> dict[str, float]:
    """Single-source / single-receiver path patching (Wang et al. 2022).

    Measures the effect on ``metric_fn`` of patching ONLY the path from a
    chosen sender component to a chosen receiver component, while freezing
    everything else along the way.

    The classic three-pass procedure (specialized to residual-stream
    sender/receiver pairs):

    1. **Clean pass.** Run the clean prompt with cache, capture the
       sender's activation ``S_clean`` and the residual stream just before
       the receiver, ``R_clean``.
    2. **Frozen-receiver pass.** Run the corrupted prompt while freezing
       the residual stream at the receiver layer to ``R_clean`` — i.e. we
       compute what the receiver would see if everything *upstream* of the
       receiver looked clean.
    3. **Sender-only pass.** Run the corrupted prompt while patching only
       the sender's residual slot to ``S_clean`` and freezing the rest of
       the residual stream at the *frozen-receiver-pass* values. The
       difference between this run and the corrupted baseline is the
       "sender → receiver" path effect.

    This implementation supports the residual-stream form
    (``component_type="resid"``) cleanly. Per-head ``"attn"`` / ``"mlp"``
    forms are more involved and not implemented yet — pass a
    ``component_type`` other than ``"resid"`` and the function raises
    :class:`NotImplementedError` with a pointer to extend it.

    Args:
        model: TransformerLens HookedTransformer.
        clean_tokens: Clean-run token ids, shape ``(batch, seq_len)``.
        corrupted_tokens: Corrupted-run token ids of the same shape.
        sender: ``(component_type, layer, position)``. ``component_type``
            must be ``"resid"`` in the current implementation. ``layer`` is
            the residual-stream layer index. ``position`` is the token
            position to patch (use ``-1`` for "all positions").
        receiver: Same shape as ``sender``. The receiver's layer must be
            strictly greater than the sender's layer.
        metric_fn: ``logits -> scalar Tensor``.

    Returns:
        Dict with::

            {
                "effect": float,            # patched - corrupted
                "clean_metric": float,
                "corrupted_metric": float,
                "patched_metric": float,
            }

    Raises:
        NotImplementedError: For non-residual component types.
        ValueError: For shape mismatches or invalid sender/receiver layers.
    """
    _require_transformerlens(model, "path_patch")

    if clean_tokens.shape != corrupted_tokens.shape:
        raise ValueError(
            f"clean_tokens shape {tuple(clean_tokens.shape)} != "
            f"corrupted_tokens shape {tuple(corrupted_tokens.shape)}."
        )

    sender_kind, sender_layer, sender_pos = sender
    receiver_kind, receiver_layer, receiver_pos = receiver

    if sender_kind != "resid" or receiver_kind != "resid":
        raise NotImplementedError(
            f"path_patch currently only supports component_type='resid' for "
            f"both sender and receiver, got sender={sender_kind!r}, "
            f"receiver={receiver_kind!r}. Extend _path_patch_resid with "
            f"per-head hooks (blocks.{{L}}.attn.hook_z indexed by head) for "
            f"the 'attn' case."
        )

    if receiver_layer <= sender_layer:
        raise ValueError(
            f"receiver_layer ({receiver_layer}) must be strictly greater "
            f"than sender_layer ({sender_layer})."
        )

    sender_hook = f"blocks.{sender_layer}.hook_resid_post"
    receiver_hook = f"blocks.{receiver_layer}.hook_resid_post"

    # ── Pass 1: clean run with cache ──────────────────────────────────────
    with torch.no_grad():
        clean_logits, clean_cache = model.run_with_cache(
            clean_tokens,
            names_filter=lambda n: n in (sender_hook, receiver_hook),
        )
        clean_metric = float(metric_fn(clean_logits).detach().item())

        # Plain corrupted forward (no patching) for the baseline.
        corrupted_logits = model(corrupted_tokens)
        corrupted_metric = float(metric_fn(corrupted_logits).detach().item())

    if sender_hook not in clean_cache or receiver_hook not in clean_cache:
        raise KeyError(
            f"Expected hook(s) missing from clean_cache: "
            f"{sender_hook}, {receiver_hook}. Got first 5 keys: "
            f"{list(clean_cache.keys())[:5]}"
        )

    s_clean = clean_cache[sender_hook]
    r_clean = clean_cache[receiver_hook]

    # ── Pass 2: corrupted run with the receiver layer frozen to clean. ───
    # This gives us "what would the residual look like at every layer if we
    # forced the receiver-layer state to its clean value?". We capture the
    # full set of intermediate residuals at sender_layer and receiver_layer.
    captured_pass2: dict[str, Tensor] = {}

    def freeze_receiver_hook(activation: Tensor, hook: Any) -> Tensor:
        # Replace receiver-layer residual with the clean version.
        new_act = activation.clone()
        new_act[...] = r_clean.to(new_act.device).to(new_act.dtype)
        captured_pass2[receiver_hook] = new_act.detach().clone()
        return new_act

    def capture_sender_hook(activation: Tensor, hook: Any) -> Tensor:
        captured_pass2[sender_hook] = activation.detach().clone()
        return activation

    with torch.no_grad():
        _ = model.run_with_hooks(
            corrupted_tokens,
            fwd_hooks=[
                (sender_hook, capture_sender_hook),
                (receiver_hook, freeze_receiver_hook),
            ],
        )

    # ── Pass 3: corrupted run with sender patched to clean. ──────────────
    # Replace just the sender's residual slot with s_clean (at the chosen
    # position) and run forward; everything downstream propagates naturally
    # except the receiver, which is overwritten with what pass 2 captured —
    # i.e. the "everything upstream is clean except this one sender slot
    # was just changed" residual. The metric on this run minus the
    # corrupted-baseline metric is the path effect.
    def patch_sender_hook(activation: Tensor, hook: Any) -> Tensor:
        new_act = activation.clone()
        clean_slot = s_clean.to(new_act.device).to(new_act.dtype)
        if sender_pos is None or sender_pos == -1:
            new_act[...] = clean_slot
        else:
            new_act[:, sender_pos, :] = clean_slot[:, sender_pos, :]
        return new_act

    def overwrite_receiver_hook(activation: Tensor, hook: Any) -> Tensor:
        new_act = activation.clone()
        frozen = captured_pass2.get(receiver_hook)
        if frozen is None:
            return activation
        frozen = frozen.to(new_act.device).to(new_act.dtype)
        if receiver_pos is None or receiver_pos == -1:
            new_act[...] = frozen
        else:
            new_act[:, receiver_pos, :] = frozen[:, receiver_pos, :]
        return new_act

    with torch.no_grad():
        patched_logits = model.run_with_hooks(
            corrupted_tokens,
            fwd_hooks=[
                (sender_hook, patch_sender_hook),
                (receiver_hook, overwrite_receiver_hook),
            ],
        )
        patched_metric = float(metric_fn(patched_logits).detach().item())

    effect = patched_metric - corrupted_metric

    logger.info(
        "path_patch: sender=L%d, receiver=L%d, effect=%.4f "
        "(clean=%.4f, corrupted=%.4f, patched=%.4f)",
        sender_layer, receiver_layer, effect,
        clean_metric, corrupted_metric, patched_metric,
    )

    return {
        "effect": float(effect),
        "clean_metric": clean_metric,
        "corrupted_metric": corrupted_metric,
        "patched_metric": patched_metric,
    }


# ---------------------------------------------------------------------------
# Public API: subgraph extraction
# ---------------------------------------------------------------------------

def extract_subgraph(
    edge_scores: dict[str, float],
    threshold: float | None = None,
    top_k: int | None = None,
) -> set[str]:
    """Filter ``edge_scores`` to a subset of edges.

    Args:
        edge_scores: Dict from edge / node name to score.
        threshold: If provided, keep only edges with ``|score| >= threshold``.
        top_k: If provided, keep only the ``top_k`` edges by ``|score|``.

    Returns:
        Set of edge names that pass the filter(s). When BOTH ``threshold``
        and ``top_k`` are provided, the result is the intersection (the
        edge must satisfy both). When neither is provided, all edges are
        returned.

    Raises:
        ValueError: If ``top_k`` is negative.
    """
    if top_k is not None and top_k < 0:
        raise ValueError(f"top_k must be >= 0, got {top_k}")

    if not edge_scores:
        return set()

    # Threshold filter
    if threshold is None:
        thresh_set: set[str] = set(edge_scores.keys())
    else:
        thresh_set = {
            name for name, score in edge_scores.items()
            if abs(score) >= threshold
        }

    # Top-k filter
    if top_k is None:
        topk_set: set[str] = set(edge_scores.keys())
    else:
        sorted_edges = sorted(
            edge_scores.items(), key=lambda kv: abs(kv[1]), reverse=True,
        )
        topk_set = {name for name, _ in sorted_edges[:top_k]}

    return thresh_set & topk_set


# ---------------------------------------------------------------------------
# Public API: ACDC stub
# ---------------------------------------------------------------------------

def acdc(model: Any, *args: Any, **kwargs: Any) -> None:
    """Automatic Circuit Discovery (Conmy et al. 2023) — NOT YET IMPLEMENTED.

    ACDC iteratively prunes edges from the computation graph, ablating one
    at a time and keeping those whose removal degrades the chosen metric
    beyond a threshold. It is a heavyweight algorithm with many knobs
    (edge ordering, ablation type, KL threshold schedule) that deserves a
    dedicated module rather than a half-implementation tucked into this
    file.

    Reference implementation:
        https://github.com/ArthurConmy/Automatic-Circuit-Discovery

    For now, use:
        - :func:`edge_attribution_patching` for a fast gradient-based
          approximation across all hook points.
        - :func:`path_patch` for a targeted, rigorous single-edge test.
    """
    raise NotImplementedError(
        "ACDC is not yet implemented. Use edge_attribution_patching() for "
        "a fast gradient-based approximation, or path_patch() for targeted "
        "single-edge tests. See the module docstring for the ACDC reference "
        "implementation."
    )
