"""End-to-end integration tests against a real tiny TransformerLens model.

These tests load ``pythia-14m`` (the smallest TL-supported HuggingFace LM, ~30MB)
and exercise the harness pipeline against the real model. Unlike the unit
tests in ``tests/test_techniques.py``, no mocks are used: every code path
exercised here is exactly what runs in production.

The fixture ``tiny_model`` lives in ``tests/conftest.py`` and is module-scoped
so the model is only downloaded once per test session.

These tests are SKIPPED by default. Run them with::

    pytest tests/test_integration_tiny_model.py --integration -v

If pythia-14m fails to load (no network, HF rate-limit, etc.) the fixture
``pytest.skip()``s gracefully — the suite will never error out.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

# All tests in this module are integration tests.
pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# 1. Basic forward pass
# ---------------------------------------------------------------------------

def test_load_model_and_basic_forward(tiny_model) -> None:
    """Sanity-check the fixture: model loads, forward pass returns logits."""
    n_layers = tiny_model.cfg.n_layers
    d_model = tiny_model.cfg.d_model
    vocab_size = tiny_model.cfg.d_vocab

    assert n_layers > 0
    assert d_model > 0
    assert vocab_size > 0

    tokens = tiny_model.to_tokens("Hello world")
    assert tokens.dim() == 2
    assert tokens.shape[0] == 1
    seq_len = tokens.shape[1]

    with torch.no_grad():
        logits = tiny_model(tokens)

    assert logits.shape == (1, seq_len, vocab_size)
    assert torch.isfinite(logits).all()


# ---------------------------------------------------------------------------
# 2. Activation extraction
# ---------------------------------------------------------------------------

def test_extract_activations_shape(tiny_model) -> None:
    """extract_activations() returns dict[layer -> (n_texts, d_model)]."""
    from src.utils.activations import extract_activations

    texts = [
        "The cat sat on the mat.",
        "Hello world.",
        "Machine learning is fun.",
        "I went to the store today.",
        "Pythia is a tiny language model.",
    ]
    layers = [0, 1]
    d_model = tiny_model.cfg.d_model

    acts = extract_activations(
        tiny_model,
        tiny_model.tokenizer,
        texts,
        layers=layers,
        aggregation="last_token",
        batch_size=2,
        device="cpu",
    )

    assert isinstance(acts, dict)
    assert set(acts.keys()) == set(layers)
    for layer, tensor in acts.items():
        assert tensor.shape == (len(texts), d_model), (
            f"layer {layer}: got {tuple(tensor.shape)}, "
            f"expected ({len(texts)}, {d_model})"
        )
        assert torch.isfinite(tensor).all()


def test_extract_activations_caching(tiny_model, tmp_path: Path) -> None:
    """Cache files are written on first call and reused on the second."""
    import time

    from src.utils.activations import extract_activations

    texts = ["First.", "Second.", "Third."]
    layers = [0, 1]
    cache_dir = tmp_path / "act_cache"

    # First call: cache miss -> compute and write files.
    t0 = time.perf_counter()
    acts1 = extract_activations(
        tiny_model,
        tiny_model.tokenizer,
        texts,
        layers=layers,
        cache_dir=cache_dir,
        device="cpu",
    )
    first_call_time = time.perf_counter() - t0

    # Cache files should exist.
    for layer in layers:
        layer_dir = cache_dir / f"layer_{layer:03d}"
        assert layer_dir.is_dir(), f"missing cache layer dir: {layer_dir}"
        files = sorted(layer_dir.glob("stimulus_*.pt"))
        assert len(files) == len(texts), (
            f"layer {layer}: {len(files)} cache files, expected {len(texts)}"
        )

    # Second call: should hit the cache and return identical tensors.
    t0 = time.perf_counter()
    acts2 = extract_activations(
        tiny_model,
        tiny_model.tokenizer,
        texts,
        layers=layers,
        cache_dir=cache_dir,
        device="cpu",
    )
    second_call_time = time.perf_counter() - t0

    for layer in layers:
        assert torch.allclose(acts1[layer], acts2[layer], atol=1e-6), (
            f"layer {layer}: cached read does not match original"
        )

    # Cached read is at least somewhat faster than the model run. We don't
    # require a hard ratio (CI noise), just that we didn't accidentally
    # rerun the full model.
    assert second_call_time < first_call_time * 1.5 + 0.5, (
        f"cached call took {second_call_time:.3f}s vs first {first_call_time:.3f}s"
    )


# ---------------------------------------------------------------------------
# 3. Probe classification end-to-end
# ---------------------------------------------------------------------------

def test_probe_classification_e2e(tiny_model) -> None:
    """Full pipeline: extract activations -> train probe -> get a ProbeResult.

    Uses 3 very-different concepts with several stimuli each. We pick the best
    layer across the model and check that the resulting probe beats chance
    (1/3 ≈ 0.33) by a clear margin. pythia-14m is too small to expect strong
    accuracy, so we set the threshold conservatively above chance.
    """
    from src.techniques.probes import ProbeResult, train_probe
    from src.utils.activations import extract_activations

    # Three concepts, six stimuli each — they need to be very distinct
    # so a 14M-param model can separate them in residual space.
    stimuli = [
        # Animals
        ("animal", "The cat chased the mouse around the kitchen."),
        ("animal", "Dogs bark loudly at strangers walking by."),
        ("animal", "Birds build nests high in the tall trees."),
        ("animal", "Horses gallop across the open green field."),
        ("animal", "The lion roared as it stalked through the savanna."),
        ("animal", "Elephants drank water from the muddy river."),
        # Math
        ("math", "Two plus two equals four in basic arithmetic."),
        ("math", "The integral of x squared is x cubed over three."),
        ("math", "Prime numbers are divisible only by one and themselves."),
        ("math", "A right triangle has one ninety degree angle."),
        ("math", "The derivative of sine is cosine in calculus."),
        ("math", "Pi is approximately three point one four one five nine."),
        # Cooking
        ("cooking", "Boil the pasta in salted water for ten minutes."),
        ("cooking", "Whisk the eggs and pour them into a hot pan."),
        ("cooking", "Slice the onions thinly and saute them in butter."),
        ("cooking", "Bake the bread at three hundred and fifty degrees."),
        ("cooking", "Marinate the chicken in lemon juice and garlic."),
        ("cooking", "Stir the soup gently while adding the cream."),
    ]
    labels = [c for c, _ in stimuli]
    texts = [t for _, t in stimuli]

    # Extract from all layers, then probe each and take the best.
    n_layers = tiny_model.cfg.n_layers
    acts_dict = extract_activations(
        tiny_model,
        tiny_model.tokenizer,
        texts,
        layers=list(range(n_layers)),
        aggregation="last_token",
        device="cpu",
    )

    best_acc = 0.0
    best_result: ProbeResult | None = None
    for layer in range(n_layers):
        activations = acts_dict[layer]
        assert activations.shape == (len(texts), tiny_model.cfg.d_model)
        result = train_probe(
            activations,
            labels,
            probe_type="logistic_regression",
            n_folds=3,
            seed=42,
        )
        if result.mean_accuracy > best_acc:
            best_acc = result.mean_accuracy
            best_result = result

    assert best_result is not None
    assert isinstance(best_result, ProbeResult)
    # Chance for 3 classes is 1/3 ≈ 0.33. We require a clear margin above
    # chance to verify the probe is actually learning something — not full
    # mastery, since pythia-14m is tiny.
    assert best_result.mean_accuracy > 0.45, (
        f"best layer probe accuracy {best_result.mean_accuracy:.3f} <= 0.45 "
        f"(chance for 3 classes ~0.33)"
    )
    assert set(best_result.label_names) == {"animal", "math", "cooking"}


# ---------------------------------------------------------------------------
# 4. Concept vectors
# ---------------------------------------------------------------------------

def test_concept_vector_extraction_e2e(tiny_model) -> None:
    """Concept vectors have unit norm and shape (d_model,)."""
    from src.techniques.contrastive import compute_concept_vectors
    from src.utils.activations import extract_activations

    texts = [
        "I love sunny weather and warm beaches.",
        "I love sunny weather and warm beaches.",
        "I love sunny weather and warm beaches.",
        "Heavy rain is pouring down outside today.",
        "Heavy rain is pouring down outside today.",
        "Heavy rain is pouring down outside today.",
    ]
    labels = ["sunny", "sunny", "sunny", "rainy", "rainy", "rainy"]

    layer = tiny_model.cfg.n_layers // 2
    acts = extract_activations(
        tiny_model,
        tiny_model.tokenizer,
        texts,
        layers=[layer],
        device="cpu",
    )[layer]

    vectors = compute_concept_vectors(acts, labels, normalize=True)
    assert set(vectors.keys()) == {"sunny", "rainy"}
    for name, vec in vectors.items():
        assert vec.shape == (tiny_model.cfg.d_model,)
        norm = float(vec.norm())
        assert abs(norm - 1.0) < 1e-5, f"{name}: norm {norm} != 1.0"


# ---------------------------------------------------------------------------
# 5. Steering changes logits
# ---------------------------------------------------------------------------

def test_steering_changes_logits(tiny_model) -> None:
    """A large random steering vector should perturb the logits.

    Tests the *plumbing* of the TL steering hook, not its semantic effect.
    """
    from src.techniques.steering import _make_tl_hook  # noqa: PLC2701

    prompt = "The capital of France is"
    tokens = tiny_model.to_tokens(prompt)

    # Baseline logits.
    with torch.no_grad():
        baseline_logits = tiny_model(tokens)

    # Random unit vector with a large alpha so we definitely shift things.
    d_model = tiny_model.cfg.d_model
    gen = torch.Generator().manual_seed(0)
    vector = torch.randn(d_model, generator=gen)
    vector = vector / vector.norm()
    alpha = 5.0

    # Apply hook at a middle layer.
    layer = tiny_model.cfg.n_layers // 2
    hook_name = f"blocks.{layer}.hook_resid_post"
    hook_fn = _make_tl_hook(vector, alpha, "cpu")

    with torch.no_grad():
        steered_logits = tiny_model.run_with_hooks(
            tokens,
            fwd_hooks=[(hook_name, hook_fn)],
        )

    assert steered_logits.shape == baseline_logits.shape
    diff = (steered_logits - baseline_logits).abs().max().item()
    assert diff > 1e-3, (
        f"steering vector did not perturb logits (max abs diff = {diff:.2e})"
    )
    # Both should still be finite.
    assert torch.isfinite(steered_logits).all()


# ---------------------------------------------------------------------------
# 6. Logit lens sweep
# ---------------------------------------------------------------------------

def test_logit_lens_sweep_e2e(tiny_model) -> None:
    """logit_lens_sweep returns one entry per layer with valid top_tokens."""
    from src.techniques.logit_lens import LogitLensResult, logit_lens_sweep

    sweep = logit_lens_sweep(
        tiny_model,
        tiny_model.tokenizer,
        prompt="The Eiffel Tower is in",
        top_k=5,
    )

    n_layers = tiny_model.cfg.n_layers
    assert isinstance(sweep, dict)
    assert set(sweep.keys()) == set(range(n_layers)), (
        f"sweep keys {sorted(sweep.keys())} != expected layers 0..{n_layers - 1}"
    )

    for layer_idx, result in sweep.items():
        assert isinstance(result, LogitLensResult)
        assert result.layer == layer_idx
        assert isinstance(result.top_tokens, list)
        assert len(result.top_tokens) > 0, f"layer {layer_idx}: empty top_tokens"
        assert len(result.top_tokens) <= 5

        for token, prob in result.top_tokens:
            assert isinstance(token, str)
            assert isinstance(prob, float)
            assert 0.0 <= prob <= 1.0, f"layer {layer_idx}: prob {prob} out of [0,1]"

        # Top-5 sum is bounded above by 1 (full distribution sums to 1).
        top_sum = sum(p for _, p in result.top_tokens)
        assert top_sum <= 1.0 + 1e-5, (
            f"layer {layer_idx}: top-5 prob sum {top_sum} > 1"
        )


# ---------------------------------------------------------------------------
# 7. Attention pattern extraction
# ---------------------------------------------------------------------------

def test_attention_extraction_e2e(tiny_model) -> None:
    """Attention patterns have correct shape and rows sum to 1 (softmaxed)."""
    from src.techniques.attention import (
        AttentionPatterns,
        extract_attention_patterns,
    )

    patterns = extract_attention_patterns(
        tiny_model,
        prompt="The quick brown fox jumps over the lazy dog.",
        layers=[0, 1],
    )

    assert isinstance(patterns, AttentionPatterns)
    assert patterns.n_layers == tiny_model.cfg.n_layers
    assert patterns.n_heads == tiny_model.cfg.n_heads
    assert set(patterns.patterns.keys()) == {0, 1}

    for layer, attn in patterns.patterns.items():
        # Single (un-batched) prompt: (n_heads, seq, seq).
        assert attn.dim() == 3, f"layer {layer}: dim {attn.dim()} != 3"
        n_heads, seq_q, seq_k = attn.shape
        assert n_heads == tiny_model.cfg.n_heads
        assert seq_q == seq_k

        # Each query row is a softmaxed distribution -> sums to ~1 along key.
        row_sums = attn.float().sum(dim=-1)
        assert torch.allclose(
            row_sums, torch.ones_like(row_sums), atol=1e-4
        ), (
            f"layer {layer}: attention rows do not sum to 1; "
            f"min={row_sums.min().item():.4f}, max={row_sums.max().item():.4f}"
        )


# ---------------------------------------------------------------------------
# 8. Residual stream patching
# ---------------------------------------------------------------------------

def test_patching_residual_stream_e2e(tiny_model) -> None:
    """Patching A's residual stream at one position changes the logits.

    We patch a single position (not all positions) at an early layer. The
    patched run should differ from both the clean A run and the clean B run:
    - it differs from B because we injected A's residual at that position;
    - it differs from A because the *other* positions are still B's.
    """
    from src.techniques.patching import patch_residual_stream

    prompt_a = "The capital of France is"
    prompt_b = "The largest desert is the"

    tokens_a = tiny_model.to_tokens(prompt_a)
    tokens_b = tiny_model.to_tokens(prompt_b)

    # Pad / trim to the same length so the patch shapes line up cleanly.
    seq_len = min(tokens_a.shape[1], tokens_b.shape[1])
    tokens_a = tokens_a[:, :seq_len]
    tokens_b = tokens_b[:, :seq_len]

    with torch.no_grad():
        logits_a, cache_a = tiny_model.run_with_cache(tokens_a)
        logits_b = tiny_model(tokens_b)

    # Patch a single middle position at an early layer. Patching ALL positions
    # at any layer would make the patched run downstream-identical to A.
    patch_layer = 1
    patch_position = seq_len // 2

    patched_logits = patch_residual_stream(
        tiny_model,
        tokens_b,
        dict(cache_a),
        patch_layer=patch_layer,
        patch_position=patch_position,
        hook_name="resid_post",
    )

    assert patched_logits.shape == logits_b.shape
    assert torch.isfinite(patched_logits).all()

    # Patched should differ from both clean A and clean B.
    diff_to_a = (patched_logits - logits_a).abs().max().item()
    diff_to_b = (patched_logits - logits_b).abs().max().item()
    assert diff_to_a > 1e-4, (
        f"patched run is nearly identical to A (max diff {diff_to_a:.2e})"
    )
    assert diff_to_b > 1e-4, (
        f"patched run is nearly identical to B (max diff {diff_to_b:.2e}); "
        f"patch had no effect"
    )


# ---------------------------------------------------------------------------
# 9. Edge attribution patching
# ---------------------------------------------------------------------------

def test_eap_e2e(tiny_model) -> None:
    """edge_attribution_patching returns an EAPResult with edge_scores."""
    from src.techniques.circuit_discovery import (
        EAPResult,
        edge_attribution_patching,
    )

    clean_tokens = tiny_model.to_tokens("The capital of France is")
    corrupted_tokens = tiny_model.to_tokens("The capital of Germany is")

    # Same length is required.
    seq_len = min(clean_tokens.shape[1], corrupted_tokens.shape[1])
    clean_tokens = clean_tokens[:, :seq_len]
    corrupted_tokens = corrupted_tokens[:, :seq_len]

    # Pick two arbitrary token ids for a logit-difference metric.
    target_id = int(clean_tokens[0, -1].item())
    distractor_id = int(corrupted_tokens[0, -1].item())
    if target_id == distractor_id:
        distractor_id = (distractor_id + 1) % tiny_model.cfg.d_vocab

    def metric(logits):
        return logits[0, -1, target_id] - logits[0, -1, distractor_id]

    result = edge_attribution_patching(
        tiny_model,
        clean_tokens,
        corrupted_tokens,
        metric_fn=metric,
        top_k=5,
        nodes="heads",
    )

    assert isinstance(result, EAPResult)
    assert isinstance(result.edge_scores, dict)
    assert len(result.edge_scores) > 0, "no edge scores returned"
    assert result.n_edges_scored == len(result.edge_scores)
    assert isinstance(result.top_edges, list)
    assert len(result.top_edges) > 0
    # All scores should be finite numbers.
    for name, score in result.edge_scores.items():
        assert isinstance(name, str)
        assert isinstance(score, float)
        import math
        assert math.isfinite(score), f"non-finite score for {name}: {score}"
