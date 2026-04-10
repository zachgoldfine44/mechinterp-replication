"""Unit tests for src.utils.aggregation.aggregate_hidden_states."""

from __future__ import annotations

import pytest
import torch

from src.utils.aggregation import aggregate_hidden_states


@pytest.fixture
def fake_batch() -> tuple[torch.Tensor, torch.Tensor]:
    """Return a small (batch=2, seq=4, hidden=8) tensor with a known mask.

    Example 0: length 3 (positions 0,1,2 are real; position 3 is pad).
    Example 1: length 2 (positions 0,1 are real; positions 2,3 are pad).
    Hidden values are constructed so each aggregation has a clear answer.
    """
    torch.manual_seed(0)
    hidden = torch.zeros(2, 4, 8)
    # Example 0: position t gets value (t + 1) in all hidden dims.
    for t in range(4):
        hidden[0, t, :] = float(t + 1)
    # Example 1: position t gets value (10 * (t + 1)) in all hidden dims.
    for t in range(4):
        hidden[1, t, :] = float(10 * (t + 1))

    mask = torch.tensor(
        [
            [1, 1, 1, 0],
            [1, 1, 0, 0],
        ],
        dtype=torch.long,
    )
    return hidden, mask


def test_last_token_respects_mask(fake_batch):
    hidden, mask = fake_batch
    out = aggregate_hidden_states(hidden, mask, "last_token")
    assert out.shape == (2, 8)
    # Example 0: last real token is index 2 -> value 3.
    assert torch.allclose(out[0], torch.full((8,), 3.0))
    # Example 1: last real token is index 1 -> value 20.
    assert torch.allclose(out[1], torch.full((8,), 20.0))


def test_first_token(fake_batch):
    hidden, mask = fake_batch
    out = aggregate_hidden_states(hidden, mask, "first_token")
    assert out.shape == (2, 8)
    assert torch.allclose(out[0], torch.full((8,), 1.0))
    assert torch.allclose(out[1], torch.full((8,), 10.0))


def test_mean_excludes_padding(fake_batch):
    hidden, mask = fake_batch
    out = aggregate_hidden_states(hidden, mask, "mean")
    assert out.shape == (2, 8)
    # Example 0: mean of [1, 2, 3] = 2.0 across all hidden dims.
    assert torch.allclose(out[0], torch.full((8,), 2.0))
    # Example 1: mean of [10, 20] = 15.0 across all hidden dims.
    assert torch.allclose(out[1], torch.full((8,), 15.0))


def test_max_excludes_padding(fake_batch):
    hidden, mask = fake_batch
    out = aggregate_hidden_states(hidden, mask, "max")
    assert out.shape == (2, 8)
    # Example 0: max of [1,2,3] = 3.0.
    assert torch.allclose(out[0], torch.full((8,), 3.0))
    # Example 1: max of [10,20] = 20.0; pad value 30 must be excluded.
    assert torch.allclose(out[1], torch.full((8,), 20.0))


def test_last_k_2_mean_of_last_two(fake_batch):
    hidden, mask = fake_batch
    out = aggregate_hidden_states(hidden, mask, "last_k:2")
    assert out.shape == (2, 8)
    # Example 0: last 2 real tokens = [2, 3], mean = 2.5.
    assert torch.allclose(out[0], torch.full((8,), 2.5))
    # Example 1: only 2 real tokens = [10, 20], mean = 15.0.
    assert torch.allclose(out[1], torch.full((8,), 15.0))


def test_last_k_larger_than_length_falls_back_to_all_real(fake_batch):
    hidden, mask = fake_batch
    out = aggregate_hidden_states(hidden, mask, "last_k:10")
    # Should equal the masked mean.
    mean_out = aggregate_hidden_states(hidden, mask, "mean")
    assert torch.allclose(out, mean_out)


def test_unknown_aggregation_raises(fake_batch):
    hidden, mask = fake_batch
    with pytest.raises(ValueError, match="Unknown aggregation"):
        aggregate_hidden_states(hidden, mask, "bogus")


def test_bad_last_k_raises(fake_batch):
    hidden, mask = fake_batch
    with pytest.raises(ValueError):
        aggregate_hidden_states(hidden, mask, "last_k:notanint")
    with pytest.raises(ValueError):
        aggregate_hidden_states(hidden, mask, "last_k:0")
