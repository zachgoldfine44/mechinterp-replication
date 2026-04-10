"""Layered activation extraction used by every experiment.

This is the canonical extraction pipeline. It is a free function (not a
method) so every experiment can call it directly without inheriting from
`ProbeClassificationExperiment` or doing the `__new__(...)` shim trick the
old code did.

Layered cache (in priority order):
    1. ``activations_cache`` — in-memory ``ActivationCache`` shared across
       all experiments running against the same (paper, model). Cheapest.
    2. ``cache_dir`` — per-stimulus, per-layer disk cache. Survives process
       restart, atomic writes (tmp + rename), resume-safe.
    3. Live extraction from the model.

The aggregation parameter is passed through to
``src.utils.aggregation.aggregate_hidden_states`` and supports any of:
``last_token``, ``first_token``, ``mean``, ``max``, ``last_k:N``.

Returned shape: ``dict[layer, np.ndarray (n_texts, hidden_dim)]``.
Numpy is used (rather than torch) because every downstream consumer
(sklearn probes, scipy stats, numpy linalg) wants numpy arrays anyway.

Usage::

    from src.utils.extraction import extract_for_experiment

    acts_by_layer = extract_for_experiment(
        model=model,
        tokenizer=tokenizer,
        texts=texts,
        layers=[10, 15, 20],
        aggregation="last_token",
        cache_dir=results_dir / "activations",
        activations_cache=activations_cache,
    )

Known limitations:
    - The TransformerLens path uses single-example forwards (one
      ``run_with_cache`` call per text) so masking is trivial. Batching is
      done on the HuggingFace path only, where padding makes the
      attention-mask path actually matter.
    - HuggingFace layer access goes through ``get_hf_layer_modules`` from
      ``src.utils.activations``; new model architectures should be added
      there.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.utils.activations import get_hf_layer_modules
from src.utils.aggregation import aggregate_hidden_states

logger = logging.getLogger(__name__)


__all__ = ["extract_for_experiment"]


@torch.no_grad()
def extract_for_experiment(
    model: Any,
    tokenizer: Any,
    texts: list[str],
    layers: list[int],
    aggregation: str,
    cache_dir: Path | None = None,
    activations_cache: Any | None = None,
    batch_size: int = 8,
) -> dict[int, np.ndarray]:
    """Extract residual stream activations for a list of texts at given layers.

    Args:
        model: TransformerLens ``HookedTransformer``, HuggingFace
            ``AutoModelForCausalLM``, or anything wrapping one of those.
        tokenizer: The model's tokenizer (HF). For TransformerLens models
            ``model.tokenizer`` is used internally; this argument may still
            be passed (and is ignored on the TL path).
        texts: Input strings.
        layers: Which transformer layer indices to extract.
        aggregation: Aggregation strategy understood by
            ``aggregate_hidden_states`` — ``last_token``, ``first_token``,
            ``mean``, ``max``, or ``last_k:N``.
        cache_dir: If provided, a per-stimulus disk cache lives here as
            ``stimulus_NNNN.pt`` files. Resume-safe. If ``None``, no disk
            cache is consulted or written.
        activations_cache: Optional ``ActivationCache`` instance for the
            shared in-memory cache. If ``None``, falls through to the disk
            cache and live extraction.
        batch_size: Batch size for live extraction (HuggingFace path only;
            TransformerLens path runs one text at a time).

    Returns:
        ``dict[int, np.ndarray]`` mapping each requested layer to an array
        of shape ``(len(texts), hidden_dim)``.

    Raises:
        RuntimeError: If after extraction any (layer, stimulus) slot is
            still empty — typically a sign that hooks failed to fire.
    """
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

    n = len(texts)
    slots: dict[int, list[np.ndarray | None]] = {
        layer: [None] * n for layer in layers
    }
    needs_extraction: list[int] = []

    for i, text in enumerate(texts):
        # 1. Try the shared in-memory cache first.
        if activations_cache is not None:
            all_hit = True
            for layer in layers:
                vec = activations_cache.get(layer, aggregation, text)
                if vec is None:
                    all_hit = False
                    break
                slots[layer][i] = vec
            if all_hit:
                continue
            # Partial hit: clear so the disk/live path fills everything
            # consistently for this stimulus.
            for layer in layers:
                slots[layer][i] = None

        # 2. Per-stimulus disk cache.
        if cache_dir is not None:
            cache_path = cache_dir / f"stimulus_{i:04d}.pt"
            if cache_path.exists():
                cached = torch.load(
                    cache_path, map_location="cpu", weights_only=True
                )
                missing_layer = False
                for layer in layers:
                    if layer not in cached:
                        missing_layer = True
                        break
                    arr = cached[layer].numpy()
                    slots[layer][i] = arr
                    if activations_cache is not None:
                        activations_cache.put(layer, aggregation, text, arr)
                if not missing_layer:
                    continue
                # Fall through to live extraction if disk lacks some layers.

        needs_extraction.append(i)

    if needs_extraction:
        logger.info(
            "Extracting activations for %d/%d stimuli (aggregation=%s)",
            len(needs_extraction), n, aggregation,
        )

    # Live extraction in batches.
    for batch_start in range(0, len(needs_extraction), batch_size):
        batch_indices = needs_extraction[batch_start : batch_start + batch_size]
        batch_texts = [texts[idx] for idx in batch_indices]

        batch_acts = _extract_batch(
            model, tokenizer, batch_texts, layers, aggregation
        )

        for local_idx, global_idx in enumerate(batch_indices):
            text = texts[global_idx]
            per_stimulus: dict[int, torch.Tensor] = {}
            for layer in layers:
                act_vec = batch_acts[layer][local_idx]
                arr = act_vec.numpy()
                slots[layer][global_idx] = arr
                per_stimulus[layer] = act_vec
                if activations_cache is not None:
                    activations_cache.put(layer, aggregation, text, arr)

            if cache_dir is not None:
                cache_path = cache_dir / f"stimulus_{global_idx:04d}.pt"
                tmp = cache_path.with_suffix(".tmp")
                torch.save(per_stimulus, tmp)
                tmp.rename(cache_path)

    # Stack into arrays. Every slot should now be filled.
    out: dict[int, np.ndarray] = {}
    for layer in layers:
        filled = slots[layer]
        if any(v is None for v in filled):
            missing = [i for i, v in enumerate(filled) if v is None]
            raise RuntimeError(
                f"Activation extraction incomplete at layer {layer}: "
                f"{len(missing)} stimuli missing (indices {missing[:5]}...)"
            )
        out[layer] = np.stack(filled)  # type: ignore[arg-type]
    return out


def _extract_batch(
    model: Any,
    tokenizer: Any,
    texts: list[str],
    layers: list[int],
    aggregation: str,
) -> dict[int, list[torch.Tensor]]:
    """Run the model on a batch of texts, return pooled activations per layer.

    Returns dict[layer, list[Tensor]] where each tensor is shape (hidden_dim,)
    on CPU.
    """
    result: dict[int, list[torch.Tensor]] = {layer: [] for layer in layers}

    # ── TransformerLens path ──────────────────────────────────────────────
    if hasattr(model, "run_with_cache"):
        for text in texts:
            tokens = model.to_tokens(text)
            _, cache = model.run_with_cache(
                tokens, names_filter=lambda name: "resid_post" in name
            )
            # TL single-example forward: no padding, mask is all ones.
            tl_mask = torch.ones(
                tokens.shape[0], tokens.shape[1], device=tokens.device
            )

            for layer in layers:
                hook_name = f"blocks.{layer}.hook_resid_post"
                acts = cache[hook_name]  # (1, seq_len, hidden_dim)
                pooled = aggregate_hidden_states(acts, tl_mask, aggregation)
                result[layer].append(pooled[0].float().cpu())
        return result

    # ── HuggingFace / nnsight path ────────────────────────────────────────
    activations_store: dict[int, list[torch.Tensor]] = {layer: [] for layer in layers}
    hooks = []

    # ``get_hf_layer_modules`` already handles the LlamaForCausalLM →
    # LlamaModel.layers walk itself, so pass the top-level model in unmodified.
    # ``hf_model`` here is just a reference for device lookup and forward
    # pass; if the caller wrapped a HuggingFace model in something else
    # (e.g. an nnsight wrapper) we still want to call the wrapper's forward.
    layer_modules = get_hf_layer_modules(model)
    hf_model = model

    def make_hook(layer_idx: int):
        def hook_fn(module, inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            activations_store[layer_idx].append(hidden.detach().float().cpu())
        return hook_fn

    for layer_idx in layers:
        if layer_idx < len(layer_modules):
            h = layer_modules[layer_idx].register_forward_hook(make_hook(layer_idx))
            hooks.append(h)

    try:
        for text in texts:
            for layer in layers:
                activations_store[layer] = []

            inputs = tokenizer(
                text, return_tensors="pt", truncation=True, max_length=512
            )
            try:
                device = next(hf_model.parameters()).device
            except StopIteration:
                device = torch.device("cpu")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                hf_model(**inputs)

            attn_mask_cpu = inputs["attention_mask"].detach().cpu()
            for layer in layers:
                if activations_store[layer]:
                    acts = activations_store[layer][0]  # (1, seq_len, hidden)
                    pooled = aggregate_hidden_states(
                        acts, attn_mask_cpu, aggregation
                    )
                    result[layer].append(pooled[0])
                else:
                    logger.warning(
                        "No activations captured for layer %d", layer
                    )
    finally:
        for h in hooks:
            h.remove()

    return result
