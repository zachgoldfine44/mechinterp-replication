"""Sparse Autoencoder (SAE) feature extraction.

Two paths, one API:

1. ``SimpleSAE`` -- a minimal, self-contained sparse autoencoder that can be
   trained from scratch on a tensor of cached activations.  Always available;
   has no dependencies beyond torch.

2. ``load_pretrained_sae`` -- thin wrapper around SAELens for loading
   community-released SAEs (e.g. Joseph Bloom's GPT-2 residual stream SAEs).
   Requires ``pip install sae-lens``.  Lazy-imported so the rest of this module
   works without SAELens installed.

Both SAE types expose an ``encode`` method returning (n, d_features), so the
downstream utilities (``extract_sae_features``, ``top_features_for_concept``,
``feature_dictionary_summary``) work uniformly.

Usage:
    from src.techniques.sae import (
        SimpleSAE,
        train_simple_sae,
        load_pretrained_sae,
        extract_sae_features,
        top_features_for_concept,
        feature_dictionary_summary,
    )

    # Train from scratch
    sae, history = train_simple_sae(activations, d_features=512, n_epochs=50)
    features = extract_sae_features(sae, activations)

    # Or load a pretrained one (requires sae-lens)
    sae = load_pretrained_sae("gpt2-small-res-jb", "blocks.8.hook_resid_pre")

    # Diagnostics
    summary = feature_dictionary_summary(features)
    top = top_features_for_concept(features, labels, "happy", top_k=20)
"""

from __future__ import annotations

import logging
from typing import Any, Literal

import torch
import torch.nn as nn
from torch import Tensor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SimpleSAE: self-contained torch module
# ---------------------------------------------------------------------------

class SimpleSAE(nn.Module):
    """Minimal sparse autoencoder for residual stream activations.

    Architecture:
        encode: (x - b_dec) @ W_enc + b_enc, then ReLU
        decode: f @ W_dec + b_dec   (W_dec is W_enc.T when ``tied_weights``)

    Loss = reconstruction MSE (summed over hidden dim) + L1 sparsity penalty
    on the feature activations.

    This is intentionally minimal -- no resampling of dead features, no
    learning-rate warmup, no geometric median initialisation.  It is meant for
    quick cross-model experiments and for tests.  For serious work, load a
    pretrained SAE with ``load_pretrained_sae`` or use SAELens directly.
    """

    def __init__(
        self,
        d_model: int,
        d_features: int,
        l1_coefficient: float = 1e-3,
        tied_weights: bool = False,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_features = d_features
        self.l1_coefficient = l1_coefficient
        self.tied_weights = tied_weights

        # Small init -- helps avoid dead features at step 0 on small data
        self.W_enc = nn.Parameter(torch.randn(d_model, d_features) * 0.1)
        self.b_enc = nn.Parameter(torch.zeros(d_features))
        if tied_weights:
            self.W_dec = None  # decoder reuses W_enc.T
        else:
            self.W_dec = nn.Parameter(torch.randn(d_features, d_model) * 0.1)
        self.b_dec = nn.Parameter(torch.zeros(d_model))

    # ------------------------------------------------------------------
    # Forward pieces
    # ------------------------------------------------------------------

    def encode(self, x: Tensor) -> Tensor:
        """Encode activations to sparse features.

        Args:
            x: Tensor of shape (..., d_model).

        Returns:
            Non-negative feature activations of shape (..., d_features).
        """
        return torch.relu((x - self.b_dec) @ self.W_enc + self.b_enc)

    def decode(self, f: Tensor) -> Tensor:
        """Decode features back to the activation space.

        Args:
            f: Tensor of shape (..., d_features).

        Returns:
            Reconstructed activations of shape (..., d_model).
        """
        W = self.W_enc.T if self.W_dec is None else self.W_dec
        return f @ W + self.b_dec

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Returns ``(reconstruction, features)``."""
        f = self.encode(x)
        x_hat = self.decode(f)
        return x_hat, f

    def loss(self, x: Tensor) -> dict[str, Tensor]:
        """Compute SAE loss terms.

        Returns a dict with:
            - ``loss``: total loss (recon + l1_coefficient * l1)
            - ``recon``: reconstruction MSE (summed over d_model, averaged over batch)
            - ``l1``: sum-of-absolute-features penalty (averaged over batch)
            - ``n_active``: fraction of feature units that are > 0 (L0 proxy)
        """
        x_hat, f = self.forward(x)
        recon = ((x - x_hat) ** 2).sum(dim=-1).mean()
        l1 = f.abs().sum(dim=-1).mean()
        total = recon + self.l1_coefficient * l1
        n_active = (f > 0).float().mean()
        return {"loss": total, "recon": recon, "l1": l1, "n_active": n_active}


# ---------------------------------------------------------------------------
# Training loop for SimpleSAE
# ---------------------------------------------------------------------------

def train_simple_sae(
    activations: Tensor,
    d_features: int,
    n_epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    l1_coefficient: float = 1e-3,
    tied_weights: bool = False,
    device: str = "cpu",
    seed: int = 42,
) -> tuple[SimpleSAE, dict[str, list[float]]]:
    """Train a :class:`SimpleSAE` on a fixed tensor of cached activations.

    Args:
        activations: Tensor of shape ``(n_samples, d_model)``.
        d_features: Dictionary size (number of SAE features).
        n_epochs: Number of epochs to train for.
        batch_size: Minibatch size for SGD.
        lr: Adam learning rate.
        l1_coefficient: L1 sparsity penalty coefficient.
        tied_weights: If True, ``W_dec = W_enc.T``.
        device: Torch device for training ("cpu", "cuda", "mps").
        seed: Random seed for reproducibility.

    Returns:
        A tuple ``(trained_sae, history)``.  ``history`` contains lists of
        per-epoch means for ``"loss"``, ``"recon"``, ``"l1"``, and ``"n_active"``.

    Raises:
        ValueError: If ``activations`` is not 2-D.
    """
    if activations.ndim != 2:
        raise ValueError(
            f"Expected 2-D activations (n_samples, d_model); got shape {tuple(activations.shape)}"
        )

    torch.manual_seed(seed)
    d_model = activations.shape[1]
    n_samples = activations.shape[0]

    sae = SimpleSAE(
        d_model=d_model,
        d_features=d_features,
        l1_coefficient=l1_coefficient,
        tied_weights=tied_weights,
    ).to(device)

    X = activations.detach().float().to(device)
    optimizer = torch.optim.Adam(sae.parameters(), lr=lr)

    history: dict[str, list[float]] = {
        "loss": [],
        "recon": [],
        "l1": [],
        "n_active": [],
    }

    effective_batch = max(1, min(batch_size, n_samples))
    n_batches = max(1, n_samples // effective_batch)

    sae.train()
    for epoch in range(n_epochs):
        perm = torch.randperm(n_samples, device=device)
        X_shuffled = X[perm]

        epoch_totals: dict[str, float] = {"loss": 0.0, "recon": 0.0, "l1": 0.0, "n_active": 0.0}
        for b in range(n_batches):
            start = b * effective_batch
            end = min(start + effective_batch, n_samples)
            x_batch = X_shuffled[start:end]

            optimizer.zero_grad()
            losses = sae.loss(x_batch)
            losses["loss"].backward()
            optimizer.step()

            for k, v in losses.items():
                epoch_totals[k] += float(v.detach().cpu().item())

        for k, v in epoch_totals.items():
            history[k].append(v / n_batches)

        if (epoch + 1) % max(1, n_epochs // 5) == 0:
            logger.info(
                "SAE epoch %d/%d: loss=%.4f recon=%.4f l1=%.4f n_active=%.3f",
                epoch + 1, n_epochs,
                history["loss"][-1],
                history["recon"][-1],
                history["l1"][-1],
                history["n_active"][-1],
            )

    sae.eval()
    return sae, history


# ---------------------------------------------------------------------------
# Pretrained SAE loader (SAELens path)
# ---------------------------------------------------------------------------

def load_pretrained_sae(
    release: str,
    sae_id: str,
    device: str = "cpu",
) -> Any:
    """Load a pretrained SAE via SAELens.

    Thin wrapper around :meth:`sae_lens.SAE.from_pretrained`.  Discards the
    returned config dict and sparsity tensor for a simpler API; if you need
    those, call SAELens directly.

    Args:
        release: SAELens release identifier (e.g. ``"gpt2-small-res-jb"``).
        sae_id: Specific SAE hook within the release
            (e.g. ``"blocks.8.hook_resid_pre"``).
        device: Torch device for the loaded SAE.

    Returns:
        The SAELens ``SAE`` instance.  It exposes an ``encode`` method, so it
        can be passed to :func:`extract_sae_features` without adaptation.

    Raises:
        ImportError: If ``sae_lens`` is not installed.  The harness ships
            without SAELens on purpose; ``SimpleSAE`` is the no-dependency
            fallback.
    """
    try:
        from sae_lens import SAE  # type: ignore[import-not-found]
    except ImportError as e:
        raise ImportError(
            "sae_lens is not installed. Install with `pip install sae-lens` "
            "to load pretrained SAEs, or use SimpleSAE for training from scratch."
        ) from e

    sae, _cfg_dict, _sparsity = SAE.from_pretrained(
        release=release,
        sae_id=sae_id,
        device=device,
    )
    logger.info("Loaded pretrained SAE: release=%s sae_id=%s", release, sae_id)
    return sae


# ---------------------------------------------------------------------------
# Feature extraction + analysis
# ---------------------------------------------------------------------------

def extract_sae_features(
    sae: Any,
    activations: Tensor,
    return_top_k: int | None = None,
    device: str = "cpu",
) -> Tensor:
    """Encode activations into SAE feature activations.

    Works with both :class:`SimpleSAE` and :class:`sae_lens.SAE` -- both
    expose an ``encode(x) -> (..., d_features)`` method.

    Args:
        sae: A SimpleSAE or SAELens SAE instance.
        activations: Tensor of shape ``(n_samples, d_model)``.
        return_top_k: If provided, zero out all but the top-k features per
            sample before returning (emulates a "keep most-active-k" sparsity
            constraint at inference time).
        device: Torch device for the encoding pass.

    Returns:
        Feature tensor of shape ``(n_samples, d_features)``.

    Raises:
        ValueError: If ``activations`` is not 2-D, or if ``return_top_k`` is
            non-positive.
    """
    if activations.ndim != 2:
        raise ValueError(
            f"Expected 2-D activations (n_samples, d_model); got shape {tuple(activations.shape)}"
        )
    if return_top_k is not None and return_top_k <= 0:
        raise ValueError(f"return_top_k must be positive; got {return_top_k}")

    x = activations.detach().float().to(device)

    # Put the SAE in eval mode if it's an nn.Module (both SimpleSAE and
    # sae_lens.SAE are nn.Modules).
    was_training = False
    if isinstance(sae, nn.Module):
        was_training = sae.training
        sae.eval()

    with torch.no_grad():
        features = sae.encode(x)

    if isinstance(sae, nn.Module) and was_training:
        sae.train()

    features = features.detach()

    if return_top_k is not None:
        k = min(return_top_k, features.shape[-1])
        # Zero out everything except the top-k per sample
        topk_vals, topk_idx = features.topk(k, dim=-1)
        masked = torch.zeros_like(features)
        masked.scatter_(-1, topk_idx, topk_vals)
        features = masked

    return features.cpu()


def top_features_for_concept(
    features: Tensor,
    labels: list[str],
    target_concept: str,
    top_k: int = 20,
    method: Literal["mean_diff", "log_odds"] = "mean_diff",
) -> list[tuple[int, float]]:
    """Find SAE features that differentially activate for a target concept.

    Args:
        features: Feature tensor of shape ``(n_samples, d_features)``, as
            returned by :func:`extract_sae_features`.
        labels: Per-sample concept labels.  Length must match
            ``features.shape[0]``.
        target_concept: The concept of interest.  Features that activate
            more strongly for this concept than for the others are scored
            highest.
        top_k: Number of top features to return.
        method: Scoring method.

            - ``"mean_diff"``: ``mean(f | target) - mean(f | other)``.  Simple,
              symmetric, and preferred when feature magnitudes carry signal.
            - ``"log_odds"``: uses activation *rates* (fraction of samples
              where the feature fires).  Computes
              ``log((p_target + eps) / (p_other + eps))``.  More robust when
              magnitudes are noisy but firing patterns are reliable.

    Returns:
        List of ``(feature_idx, score)`` tuples sorted by descending score,
        length ``min(top_k, d_features)``.

    Raises:
        ValueError: If ``target_concept`` is absent from ``labels``, if
            lengths mismatch, or if ``method`` is unknown.
    """
    if features.ndim != 2:
        raise ValueError(
            f"Expected 2-D features (n_samples, d_features); got shape {tuple(features.shape)}"
        )
    if features.shape[0] != len(labels):
        raise ValueError(
            f"feature count ({features.shape[0]}) != label count ({len(labels)})"
        )
    if method not in ("mean_diff", "log_odds"):
        raise ValueError(
            f"Unknown method: {method!r}. Expected 'mean_diff' or 'log_odds'."
        )

    target_mask = torch.tensor(
        [lab == target_concept for lab in labels], dtype=torch.bool
    )
    other_mask = ~target_mask

    if target_mask.sum().item() == 0:
        raise ValueError(
            f"target_concept {target_concept!r} not present in labels"
        )
    if other_mask.sum().item() == 0:
        raise ValueError(
            f"All samples have label {target_concept!r}; no 'other' class to contrast against"
        )

    f = features.detach().float()

    if method == "mean_diff":
        scores = f[target_mask].mean(dim=0) - f[other_mask].mean(dim=0)
    else:  # log_odds
        eps = 1e-6
        p_target = (f[target_mask] > 0).float().mean(dim=0)
        p_other = (f[other_mask] > 0).float().mean(dim=0)
        scores = torch.log((p_target + eps) / (p_other + eps))

    k = min(top_k, scores.shape[0])
    topk_vals, topk_idx = scores.topk(k)

    return [(int(i), float(v)) for i, v in zip(topk_idx.tolist(), topk_vals.tolist())]


def feature_dictionary_summary(features: Tensor) -> dict[str, float | int]:
    """Diagnostic summary of an SAE feature dictionary's activity.

    Args:
        features: Feature tensor of shape ``(n_samples, d_features)``.

    Returns:
        Dict with:

        - ``n_features``: dictionary size (``d_features``)
        - ``n_samples``: number of samples
        - ``mean_active_features_per_sample``: average number of features > 0
          per sample (L0 norm averaged)
        - ``l0_per_sample``: alias of ``mean_active_features_per_sample``
          (standard SAE terminology)
        - ``fraction_of_dead_features``: fraction of features that are zero
          for every sample in the input
        - ``mean_feature_magnitude``: mean absolute activation across all
          samples and features

    Raises:
        ValueError: If ``features`` is not 2-D.
    """
    if features.ndim != 2:
        raise ValueError(
            f"Expected 2-D features (n_samples, d_features); got shape {tuple(features.shape)}"
        )

    f = features.detach().float()
    n_samples, n_features = f.shape

    active_mask = f > 0  # (n_samples, n_features)
    l0_per_sample = float(active_mask.float().sum(dim=-1).mean().item())

    # A feature is "dead" if it is never > 0 across any sample
    never_active = ~active_mask.any(dim=0)  # (n_features,)
    fraction_dead = float(never_active.float().mean().item())

    mean_magnitude = float(f.abs().mean().item())

    return {
        "n_features": int(n_features),
        "n_samples": int(n_samples),
        "mean_active_features_per_sample": l0_per_sample,
        "l0_per_sample": l0_per_sample,
        "fraction_of_dead_features": fraction_dead,
        "mean_feature_magnitude": mean_magnitude,
    }
