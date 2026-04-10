"""Activation aggregation strategies.

Provides a single function, `aggregate_hidden_states`, that collapses a
(batch, seq, hidden) tensor of per-token hidden states into a (batch, hidden)
tensor of per-example activations, respecting an attention mask so that
padding positions are never included in means or maxes.

Supported strategies:
    - "last_token"  : the last non-pad token
    - "first_token" : the first token (usually BOS)
    - "mean"        : mean over all non-pad tokens
    - "max"         : element-wise max over all non-pad tokens
    - "last_k:N"    : mean over the last N non-pad tokens (N parsed from string)

All strategies accept tensors on any device and return a tensor of shape
(batch, hidden) with the same dtype as the input.
"""

from __future__ import annotations

import torch


def aggregate_hidden_states(
    hidden: torch.Tensor,
    attention_mask: torch.Tensor,
    aggregation: str,
) -> torch.Tensor:
    """Aggregate per-token hidden states into a single vector per example.

    Args:
        hidden: Tensor of shape (batch, seq, hidden).
        attention_mask: Tensor of shape (batch, seq); 1 for real tokens, 0 for pad.
        aggregation: One of "last_token", "first_token", "mean", "max",
            or "last_k:N" where N is a positive integer.

    Returns:
        Tensor of shape (batch, hidden).
    """
    if hidden.dim() != 3:
        raise ValueError(
            f"hidden must be (batch, seq, hidden); got shape {tuple(hidden.shape)}"
        )
    if attention_mask.dim() != 2:
        raise ValueError(
            f"attention_mask must be (batch, seq); got shape {tuple(attention_mask.shape)}"
        )
    if hidden.shape[:2] != attention_mask.shape:
        raise ValueError(
            f"hidden batch/seq {tuple(hidden.shape[:2])} does not match "
            f"attention_mask {tuple(attention_mask.shape)}"
        )

    mask = attention_mask.to(hidden.device).to(hidden.dtype)  # (B, S)
    batch_size, seq_len, hidden_dim = hidden.shape

    if aggregation == "last_token":
        # Index of the last non-pad token per example.
        lengths = attention_mask.to(torch.long).sum(dim=1)  # (B,)
        # Guard against empty sequences; clamp to at least 1 real token.
        last_idx = (lengths - 1).clamp(min=0)
        batch_idx = torch.arange(batch_size, device=hidden.device)
        return hidden[batch_idx, last_idx, :]

    if aggregation == "first_token":
        return hidden[:, 0, :]

    if aggregation == "mean":
        mask_expanded = mask.unsqueeze(-1)  # (B, S, 1)
        summed = (hidden * mask_expanded).sum(dim=1)  # (B, H)
        counts = mask.sum(dim=1).clamp(min=1).unsqueeze(-1)  # (B, 1)
        return summed / counts

    if aggregation == "max":
        # Replace pad positions with -inf so they never win the max.
        neg_inf = torch.finfo(hidden.dtype).min
        mask_expanded = mask.unsqueeze(-1).bool()  # (B, S, 1)
        masked = hidden.masked_fill(~mask_expanded, neg_inf)
        return masked.max(dim=1).values

    if aggregation.startswith("last_k:"):
        try:
            k = int(aggregation.split(":", 1)[1])
        except (ValueError, IndexError) as e:
            raise ValueError(
                f"Invalid last_k aggregation {aggregation!r}; expected 'last_k:N'"
            ) from e
        if k <= 0:
            raise ValueError(f"last_k:N requires N > 0; got N={k}")

        lengths = attention_mask.to(torch.long).sum(dim=1)  # (B,)
        # Build a per-example mask selecting the last min(k, length) non-pad tokens.
        # Positions [length - min(k, length), length) are included.
        positions = torch.arange(seq_len, device=hidden.device).unsqueeze(0)  # (1, S)
        k_per_row = lengths.clamp(max=k)  # (B,)
        start = (lengths - k_per_row).unsqueeze(1)  # (B, 1)
        end = lengths.unsqueeze(1)  # (B, 1)
        sel_mask = (positions >= start) & (positions < end)  # (B, S)
        sel_mask = sel_mask.to(hidden.dtype)

        mask_expanded = sel_mask.unsqueeze(-1)  # (B, S, 1)
        summed = (hidden * mask_expanded).sum(dim=1)  # (B, H)
        counts = sel_mask.sum(dim=1).clamp(min=1).unsqueeze(-1)  # (B, 1)
        return summed / counts

    raise ValueError(
        f"Unknown aggregation strategy: {aggregation!r}. "
        f"Expected one of: last_token, first_token, mean, max, last_k:N"
    )
