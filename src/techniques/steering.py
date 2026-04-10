"""Activation steering: add scaled vectors to residual stream activations.

Supports:
- Single-layer or multi-layer steering
- Configurable alpha (steering strength)
- Control conditions: random vector (null baseline), negated vector (reversal)

Works with both TransformerLens (via run_with_hooks / run_with_cache) and
HuggingFace models (via register_forward_hook).

Usage:
    from src.techniques.steering import (
        steer_and_generate,
        steer_and_score,
        get_steering_layers,
        create_control_vector,
    )

    # Generate text under steering
    text = steer_and_generate(
        model, tokenizer, "I think that",
        vector=joy_vector, layers=[10, 11, 12], alpha=1.0,
    )

    # Score multiple-choice completions
    scores = steer_and_score(
        model, tokenizer, "I feel",
        vector=joy_vector, layers=[10, 11, 12], alpha=1.0,
        choices=["happy", "sad", "neutral"],
    )

    # Control vectors for null baselines
    random_vec = create_control_vector(joy_vector, control_type="random")
"""

from __future__ import annotations

import logging
from typing import Any, Literal

import torch
from torch import Tensor

from src.utils.activations import get_hf_layer_modules

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API: steering layer selection
# ---------------------------------------------------------------------------

def get_steering_layers(
    n_layers: int,
    strategy: Literal["middle_third", "two_thirds", "all"] = "middle_third",
) -> list[int]:
    """Select which layers to steer based on a named strategy.

    Args:
        n_layers: Total number of layers in the model.
        strategy:
            - "middle_third": layers in the middle third (most common for CAA).
            - "two_thirds": single layer at approximately 2/3 depth.
            - "all": every layer (expensive, mainly for ablation studies).

    Returns:
        Sorted list of zero-indexed layer indices.

    Raises:
        ValueError: If strategy is unrecognized or n_layers < 1.
    """
    if n_layers < 1:
        raise ValueError(f"n_layers must be >= 1, got {n_layers}")

    if strategy == "middle_third":
        start = n_layers // 3
        end = 2 * n_layers // 3
        return list(range(start, max(end, start + 1)))

    elif strategy == "two_thirds":
        layer = int(round(2 * n_layers / 3))
        layer = min(layer, n_layers - 1)
        return [layer]

    elif strategy == "all":
        return list(range(n_layers))

    else:
        raise ValueError(
            f"Unknown strategy: {strategy!r}. "
            f"Expected 'middle_third', 'two_thirds', or 'all'."
        )


# ---------------------------------------------------------------------------
# Public API: control vectors
# ---------------------------------------------------------------------------

def create_control_vector(
    vector: Tensor,
    control_type: Literal["random", "negated", "zero"] = "random",
    seed: int = 42,
) -> Tensor:
    """Create a control vector for baseline comparisons.

    Args:
        vector: The original steering vector of shape (hidden_dim,).
        control_type:
            - "random": Random direction with same L2 norm as *vector*.
            - "negated": -vector (tests directionality).
            - "zero": Zero vector (no-steering baseline).
        seed: Random seed for the "random" control type.

    Returns:
        Control vector with the same shape and dtype as *vector*.
    """
    if control_type == "negated":
        return -vector.clone()

    elif control_type == "zero":
        return torch.zeros_like(vector)

    elif control_type == "random":
        gen = torch.Generator()
        gen.manual_seed(seed)
        random_vec = torch.randn(vector.shape, generator=gen, dtype=vector.dtype)
        # Match the original vector's norm
        original_norm = vector.norm()
        random_norm = random_vec.norm()
        if random_norm > 1e-8:
            random_vec = random_vec * (original_norm / random_norm)
        return random_vec

    else:
        raise ValueError(
            f"Unknown control_type: {control_type!r}. "
            f"Expected 'random', 'negated', or 'zero'."
        )


# ---------------------------------------------------------------------------
# Public API: steer and generate
# ---------------------------------------------------------------------------

def steer_and_generate(
    model: Any,
    tokenizer: Any,
    prompt: str,
    vector: Tensor,
    layers: list[int],
    alpha: float = 0.5,
    max_new_tokens: int = 200,
    device: str = "cpu",
) -> str:
    """Generate text with activation steering applied.

    Adds ``alpha * vector`` to the residual stream at each specified layer
    during generation.

    Args:
        model: A TransformerLens HookedTransformer or HuggingFace model.
        tokenizer: The model's tokenizer.
        prompt: Input prompt to continue.
        vector: Steering vector of shape (hidden_dim,).
        layers: Which layers to steer (zero-indexed).
        alpha: Steering strength multiplier.
        max_new_tokens: Maximum number of tokens to generate.
        device: Device string ("cpu", "cuda", "mps").

    Returns:
        Generated text (prompt + new tokens).
    """
    if _is_transformerlens(model):
        return _tl_steer_generate(
            model, tokenizer, prompt, vector, layers, alpha, max_new_tokens, device,
        )
    else:
        return _hf_steer_generate(
            model, tokenizer, prompt, vector, layers, alpha, max_new_tokens, device,
        )


# ---------------------------------------------------------------------------
# Public API: steer and score
# ---------------------------------------------------------------------------

def steer_and_score(
    model: Any,
    tokenizer: Any,
    prompt: str,
    vector: Tensor,
    layers: list[int],
    alpha: float,
    choices: list[str],
    device: str = "cpu",
) -> dict[str, float]:
    """Score multiple-choice completions under activation steering.

    For each choice, computes the log-probability of the choice tokens
    following the prompt, with steering applied.

    Args:
        model: A TransformerLens HookedTransformer or HuggingFace model.
        tokenizer: The model's tokenizer.
        prompt: Context prompt.
        vector: Steering vector of shape (hidden_dim,).
        layers: Which layers to steer.
        alpha: Steering strength.
        choices: List of completion strings to score.
        device: Device string.

    Returns:
        Dict mapping each choice string to its total log-probability
        under steering.
    """
    if _is_transformerlens(model):
        return _tl_steer_score(
            model, tokenizer, prompt, vector, layers, alpha, choices, device,
        )
    else:
        return _hf_steer_score(
            model, tokenizer, prompt, vector, layers, alpha, choices, device,
        )


# ---------------------------------------------------------------------------
# TransformerLens backend
# ---------------------------------------------------------------------------

def _is_transformerlens(model: Any) -> bool:
    """Check if model is a TransformerLens HookedTransformer."""
    return hasattr(model, "run_with_hooks") and hasattr(model, "cfg")


def _make_tl_hook(vector: Tensor, alpha: float, device: str):
    """Create a TransformerLens hook function that adds alpha * vector to resid."""
    steering_vec = (alpha * vector).to(device)

    def hook_fn(activation: Tensor, hook: Any) -> Tensor:
        # activation shape: (batch, seq_len, hidden_dim)
        # Add steering vector to all sequence positions
        return activation + steering_vec.unsqueeze(0).unsqueeze(0)

    return hook_fn


def _tl_steer_generate(
    model: Any,
    tokenizer: Any,
    prompt: str,
    vector: Tensor,
    layers: list[int],
    alpha: float,
    max_new_tokens: int,
    device: str,
) -> str:
    """Generate with TransformerLens using run_with_hooks for each token."""
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    generated = input_ids.clone()

    for _ in range(max_new_tokens):
        # Build hook list for this forward pass
        fwd_hooks = []
        for layer in layers:
            hook_name = f"blocks.{layer}.hook_resid_post"
            fwd_hooks.append((hook_name, _make_tl_hook(vector, alpha, device)))

        with torch.no_grad():
            logits = model.run_with_hooks(
                generated,
                fwd_hooks=fwd_hooks,
            )

        # Sample next token (greedy)
        next_token = logits[0, -1, :].argmax(dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token.unsqueeze(0)], dim=-1)

        # Stop on EOS
        if tokenizer.eos_token_id is not None and next_token.item() == tokenizer.eos_token_id:
            break

    return tokenizer.decode(generated[0], skip_special_tokens=True)


def _tl_steer_score(
    model: Any,
    tokenizer: Any,
    prompt: str,
    vector: Tensor,
    layers: list[int],
    alpha: float,
    choices: list[str],
    device: str,
) -> dict[str, float]:
    """Score choices with TransformerLens steering."""
    scores: dict[str, float] = {}

    for choice in choices:
        full_text = prompt + choice
        tokens = tokenizer.encode(full_text, return_tensors="pt").to(device)
        prompt_tokens = tokenizer.encode(prompt, return_tensors="pt").to(device)
        prompt_len = prompt_tokens.shape[1]

        fwd_hooks = []
        for layer in layers:
            hook_name = f"blocks.{layer}.hook_resid_post"
            fwd_hooks.append((hook_name, _make_tl_hook(vector, alpha, device)))

        with torch.no_grad():
            logits = model.run_with_hooks(tokens, fwd_hooks=fwd_hooks)

        # Sum log-probs of choice tokens
        log_probs = torch.nn.functional.log_softmax(logits[0], dim=-1)
        total_log_prob = 0.0
        for i in range(prompt_len - 1, tokens.shape[1] - 1):
            next_token_id = tokens[0, i + 1]
            total_log_prob += log_probs[i, next_token_id].item()

        scores[choice] = total_log_prob

    return scores


# ---------------------------------------------------------------------------
# HuggingFace backend
# ---------------------------------------------------------------------------

# Layer-module accessor lives in src.utils.activations.get_hf_layer_modules
# (imported at the top of this file). This used to be a duplicated local
# helper; consolidated to one canonical implementation.


def _hf_steer_generate(
    model: Any,
    tokenizer: Any,
    prompt: str,
    vector: Tensor,
    layers: list[int],
    alpha: float,
    max_new_tokens: int,
    device: str,
) -> str:
    """Generate with HuggingFace model using register_forward_hook."""
    layer_modules = get_hf_layer_modules(model)
    steering_vec = (alpha * vector).to(device)
    handles = []

    def make_hook():
        def hook_fn(module: Any, input: Any, output: Any) -> Any:
            # output is typically a tuple; first element is the hidden states
            if isinstance(output, tuple):
                hidden = output[0]
                modified = hidden + steering_vec.unsqueeze(0).unsqueeze(0)
                return (modified,) + output[1:]
            else:
                return output + steering_vec.unsqueeze(0).unsqueeze(0)
        return hook_fn

    try:
        # Register hooks
        for layer_idx in layers:
            if layer_idx < len(layer_modules):
                handle = layer_modules[layer_idx].register_forward_hook(make_hook())
                handles.append(handle)

        # Generate
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        return tokenizer.decode(output_ids[0], skip_special_tokens=True)

    finally:
        # Always remove hooks
        for handle in handles:
            handle.remove()


def _hf_steer_score(
    model: Any,
    tokenizer: Any,
    prompt: str,
    vector: Tensor,
    layers: list[int],
    alpha: float,
    choices: list[str],
    device: str,
) -> dict[str, float]:
    """Score choices with HuggingFace model steering."""
    layer_modules = get_hf_layer_modules(model)
    steering_vec = (alpha * vector).to(device)
    scores: dict[str, float] = {}

    def make_hook():
        def hook_fn(module: Any, input: Any, output: Any) -> Any:
            if isinstance(output, tuple):
                hidden = output[0]
                modified = hidden + steering_vec.unsqueeze(0).unsqueeze(0)
                return (modified,) + output[1:]
            else:
                return output + steering_vec.unsqueeze(0).unsqueeze(0)
        return hook_fn

    for choice in choices:
        handles = []
        try:
            for layer_idx in layers:
                if layer_idx < len(layer_modules):
                    handle = layer_modules[layer_idx].register_forward_hook(make_hook())
                    handles.append(handle)

            full_text = prompt + choice
            tokens = tokenizer(full_text, return_tensors="pt").to(device)
            prompt_tokens = tokenizer(prompt, return_tensors="pt")
            prompt_len = prompt_tokens["input_ids"].shape[1]

            with torch.no_grad():
                outputs = model(**tokens)
                logits = outputs.logits

            log_probs = torch.nn.functional.log_softmax(logits[0], dim=-1)
            total_log_prob = 0.0
            for i in range(prompt_len - 1, tokens["input_ids"].shape[1] - 1):
                next_token_id = tokens["input_ids"][0, i + 1]
                total_log_prob += log_probs[i, next_token_id].item()

            scores[choice] = total_log_prob

        finally:
            for handle in handles:
                handle.remove()

    return scores
