"""Regression tests with pinned outputs for activation extraction.

These tests load a tiny real model, extract activations on a FIXED prompt,
and assert specific properties of the resulting tensors. If the activation
extraction code regresses (wrong layer indexed, wrong aggregation, dtype
mismatch, hook misconfigured, etc.) these tests will catch it.

Note: we don't pin EXACT floating point values because numerical noise
across torch versions / hardware would make the test flaky. Instead we pin:
- shapes
- approximate norms (within wide tolerances)
- monotonicity / sign properties
- relative differences between layers and prompts
- determinism across repeat runs

The ``tiny_model`` fixture is shared with ``tests/test_integration_tiny_model.py``
via ``tests/conftest.py``.

These tests are SKIPPED by default. Run with::

    pytest tests/test_regression_activation_extraction.py --integration -v
"""

from __future__ import annotations

import pytest
import torch

from src.utils.activations import extract_activations

pytestmark = pytest.mark.integration


# A fixed prompt used by most tests so any future regression bisects cleanly.
FIXED_PROMPT = "The cat sat on the mat."


# ---------------------------------------------------------------------------
# 1. Finite, non-zero norms at every layer
# ---------------------------------------------------------------------------

def test_extract_residual_stream_norms_are_finite(tiny_model) -> None:
    """Every layer's activation has finite, non-zero norm."""
    n_layers = tiny_model.cfg.n_layers
    layers = list(range(n_layers))

    acts = extract_activations(
        tiny_model,
        tiny_model.tokenizer,
        [FIXED_PROMPT],
        layers=layers,
        aggregation="last_token",
        device="cpu",
    )

    assert set(acts.keys()) == set(layers)
    for layer in layers:
        tensor = acts[layer]
        assert torch.isfinite(tensor).all(), f"layer {layer}: non-finite values"
        norm = float(tensor.norm())
        assert norm > 0.0, f"layer {layer}: zero norm"


# ---------------------------------------------------------------------------
# 2. Aggregations differ
# ---------------------------------------------------------------------------

def test_extract_last_token_vs_mean_aggregation_differ(tiny_model) -> None:
    """last_token and mean aggregations should produce different vectors."""
    layer = tiny_model.cfg.n_layers // 2

    last_acts = extract_activations(
        tiny_model,
        tiny_model.tokenizer,
        [FIXED_PROMPT],
        layers=[layer],
        aggregation="last_token",
        device="cpu",
    )[layer]

    mean_acts = extract_activations(
        tiny_model,
        tiny_model.tokenizer,
        [FIXED_PROMPT],
        layers=[layer],
        aggregation="mean",
        device="cpu",
    )[layer]

    assert last_acts.shape == mean_acts.shape
    diff = (last_acts - mean_acts).abs().max().item()
    assert diff > 1e-4, (
        f"last_token and mean aggregations are nearly identical (max diff {diff:.2e}); "
        f"one of the aggregation paths is broken"
    )


# ---------------------------------------------------------------------------
# 3. Determinism: same input -> same output across runs
# ---------------------------------------------------------------------------

def test_extract_consistent_across_runs(tiny_model) -> None:
    """Two extraction runs on the same prompt produce identical activations."""
    layers = [0, tiny_model.cfg.n_layers // 2, tiny_model.cfg.n_layers - 1]

    run1 = extract_activations(
        tiny_model,
        tiny_model.tokenizer,
        [FIXED_PROMPT],
        layers=layers,
        device="cpu",
    )
    run2 = extract_activations(
        tiny_model,
        tiny_model.tokenizer,
        [FIXED_PROMPT],
        layers=layers,
        device="cpu",
    )

    for layer in layers:
        a, b = run1[layer], run2[layer]
        assert a.shape == b.shape
        assert torch.allclose(a, b, atol=1e-6), (
            f"layer {layer}: extraction not deterministic; "
            f"max diff = {(a - b).abs().max().item():.2e}"
        )


# ---------------------------------------------------------------------------
# 4. Layer norms grow with depth (residual stream property)
# ---------------------------------------------------------------------------

def test_extract_layer_norms_increase_with_depth(tiny_model) -> None:
    """Last-layer activation norm should be noticeably larger than first-layer.

    For a transformer LM's residual stream, each block adds its output to the
    running residual, so norms typically increase with depth. We require a 1.5x
    ratio between the last and first layer — a very loose floor that pythia-14m
    clears comfortably (~1.86x in practice). If even this fails, something is
    very wrong with the extraction (e.g. wrong hook point, layer order
    reversed, off-by-one).
    """
    n_layers = tiny_model.cfg.n_layers
    layers = list(range(n_layers))

    acts = extract_activations(
        tiny_model,
        tiny_model.tokenizer,
        [FIXED_PROMPT],
        layers=layers,
        device="cpu",
    )

    norms = [float(acts[layer].norm()) for layer in layers]
    first, last = norms[0], norms[-1]
    assert last >= 1.5 * first, (
        f"residual norms do not grow as expected: layer 0 norm = {first:.3f}, "
        f"layer {n_layers - 1} norm = {last:.3f}; full sequence = {norms}"
    )


# ---------------------------------------------------------------------------
# 5. Different prompts -> different activations
# ---------------------------------------------------------------------------

def test_extract_different_prompts_produce_different_activations(tiny_model) -> None:
    """Two genuinely different prompts should not produce nearly identical vectors."""
    layer = tiny_model.cfg.n_layers // 2

    acts = extract_activations(
        tiny_model,
        tiny_model.tokenizer,
        ["The cat sat on the mat.", "Quantum mechanics is hard."],
        layers=[layer],
        device="cpu",
    )[layer]

    a, b = acts[0], acts[1]
    cos = torch.nn.functional.cosine_similarity(
        a.unsqueeze(0), b.unsqueeze(0), dim=-1
    ).item()
    assert cos < 0.99, (
        f"two unrelated prompts have cosine similarity {cos:.4f} >= 0.99; "
        f"extraction is collapsing different inputs to the same vector"
    )


# ---------------------------------------------------------------------------
# 6. Single-token edge case
# ---------------------------------------------------------------------------

def test_extract_handles_single_token_prompt(tiny_model) -> None:
    """A trivially short prompt should not crash."""
    layer = 0
    # A single character is the most likely case to trip up aggregation logic.
    acts = extract_activations(
        tiny_model,
        tiny_model.tokenizer,
        ["a"],
        layers=[layer],
        aggregation="last_token",
        device="cpu",
    )

    assert layer in acts
    tensor = acts[layer]
    assert tensor.shape == (1, tiny_model.cfg.d_model)
    assert torch.isfinite(tensor).all()


# ---------------------------------------------------------------------------
# 7. Long prompt edge case
# ---------------------------------------------------------------------------

def test_extract_handles_long_prompt(tiny_model) -> None:
    """A ~50-token prompt should produce shape (1, d_model)."""
    long_prompt = " ".join(
        [
            "The quick brown fox jumps over the lazy dog,",
            "and then it runs across the open field looking for",
            "a place to rest under the warm afternoon sun while",
            "the birds sing and the wind rustles through the leaves",
            "and somewhere in the distance a church bell rings softly.",
        ]
    )

    # Verify it's actually long-ish (well over 30 tokens for any tokenizer).
    n_tokens = tiny_model.to_tokens(long_prompt).shape[1]
    assert n_tokens >= 30, f"prompt only tokenized to {n_tokens} tokens"

    layer = tiny_model.cfg.n_layers // 2
    acts = extract_activations(
        tiny_model,
        tiny_model.tokenizer,
        [long_prompt],
        layers=[layer],
        device="cpu",
    )[layer]

    assert acts.shape == (1, tiny_model.cfg.d_model)
    assert torch.isfinite(acts).all()


# ---------------------------------------------------------------------------
# 8. Dtype is preserved
# ---------------------------------------------------------------------------

def test_extract_dtype_preserved(tiny_model) -> None:
    """Extracting from a fp32 model returns fp32 tensors."""
    # Confirm the model is fp32 (pythia-14m loaded on CPU defaults to fp32).
    p = next(tiny_model.parameters())
    assert p.dtype == torch.float32, (
        f"sanity check: model param dtype is {p.dtype}, not float32; "
        f"this test assumes a fp32 model"
    )

    acts = extract_activations(
        tiny_model,
        tiny_model.tokenizer,
        [FIXED_PROMPT],
        layers=[0],
        device="cpu",
    )[0]

    assert acts.dtype == torch.float32, (
        f"extraction silently downgraded dtype to {acts.dtype}"
    )


# ---------------------------------------------------------------------------
# 9. Subset of layers
# ---------------------------------------------------------------------------

def test_extract_subset_of_layers(tiny_model) -> None:
    """Asking for only a subset of layers returns only those layers."""
    n_layers = tiny_model.cfg.n_layers
    # Pick two layers that exist for any pythia-14m variant (it has >= 6 layers).
    layers = [0, min(2, n_layers - 1)]
    layers = sorted(set(layers))  # de-duplicate if n_layers happens to be 1

    acts = extract_activations(
        tiny_model,
        tiny_model.tokenizer,
        [FIXED_PROMPT],
        layers=layers,
        device="cpu",
    )

    assert set(acts.keys()) == set(layers), (
        f"got layers {sorted(acts.keys())}, expected {layers}"
    )
    for layer in layers:
        assert acts[layer].shape == (1, tiny_model.cfg.d_model)


# ---------------------------------------------------------------------------
# 10. Concept vectors are unit-norm when normalize=True
# ---------------------------------------------------------------------------

def test_concept_vectors_are_normalized(tiny_model) -> None:
    """compute_concept_vectors with normalize=True returns L2-norm-1 vectors."""
    from src.techniques.contrastive import compute_concept_vectors

    texts = [
        "The dog ran across the yard chasing a ball.",
        "The dog ran across the yard chasing a ball.",
        "The dog ran across the yard chasing a ball.",
        "Mathematicians prove theorems using rigorous logic.",
        "Mathematicians prove theorems using rigorous logic.",
        "Mathematicians prove theorems using rigorous logic.",
    ]
    labels = ["A", "A", "A", "B", "B", "B"]

    layer = tiny_model.cfg.n_layers // 2
    acts = extract_activations(
        tiny_model,
        tiny_model.tokenizer,
        texts,
        layers=[layer],
        device="cpu",
    )[layer]

    vectors = compute_concept_vectors(acts, labels, normalize=True)
    assert set(vectors.keys()) == {"A", "B"}
    for name, vec in vectors.items():
        norm = float(vec.norm())
        assert abs(norm - 1.0) < 1e-5, (
            f"concept vector {name!r} has norm {norm}, expected 1.0"
        )
