"""Contrastive activation extraction.

Computes mean-difference vectors (CAA-style) for each concept:
  v_concept = mean(activations | concept) - mean(all activations)
  v_concept = v_concept / ||v_concept||

Also supports paired contrastive:
  v = mean(activations | A) - mean(activations | B)

And analysis utilities: cosine similarity matrices and PCA projection.

Usage:
    from src.techniques.contrastive import (
        compute_concept_vectors,
        compute_paired_vectors,
        compute_similarity_matrix,
        compute_pca,
    )

    vectors = compute_concept_vectors(activations, labels)
    sim_matrix, names = compute_similarity_matrix(vectors)
    pca = compute_pca(vectors, n_components=3)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import torch
from torch import Tensor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class PCAResult:
    """Result of PCA on concept vectors.

    Attributes:
        components: Principal component directions, shape (n_components, hidden_dim).
        explained_variance_ratio: Fraction of variance explained per component.
        projected: Per-concept 2D (or n_components-D) coordinates.
            Dict mapping concept name to numpy array of shape (n_components,).
        concept_names: Ordered list of concept names.
    """

    components: np.ndarray
    explained_variance_ratio: np.ndarray
    projected: dict[str, np.ndarray]
    concept_names: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_concept_vectors(
    activations: Tensor,
    labels: list[str] | list[int] | np.ndarray,
    normalize: bool = True,
) -> dict[str, Tensor]:
    """Compute mean-difference concept vectors (CAA-style).

    For each concept c:
        v_c = mean(activations where label == c) - mean(all activations)
        if normalize: v_c = v_c / ||v_c||

    Args:
        activations: Tensor of shape (n_samples, hidden_dim).
        labels: Per-sample concept labels (strings or ints).
        normalize: Whether to L2-normalize the resulting vectors.

    Returns:
        Dict mapping concept label (as string) to vector of shape (hidden_dim,).

    Raises:
        ValueError: If activations and labels have mismatched lengths.
    """
    if activations.shape[0] != len(labels):
        raise ValueError(
            f"Activation count ({activations.shape[0]}) != label count ({len(labels)})"
        )

    labels_list = _to_string_list(labels)
    unique_labels = sorted(set(labels_list))
    acts = activations.detach().float()

    global_mean = acts.mean(dim=0)
    vectors: dict[str, Tensor] = {}

    for concept in unique_labels:
        mask = torch.tensor(
            [lab == concept for lab in labels_list], dtype=torch.bool
        )
        concept_mean = acts[mask].mean(dim=0)
        vec = concept_mean - global_mean

        if normalize:
            norm = vec.norm()
            if norm > 1e-8:
                vec = vec / norm
            else:
                logger.warning(
                    "Concept %r has near-zero vector norm (%.2e); "
                    "returning unnormalized zero-ish vector.",
                    concept, norm.item(),
                )

        vectors[concept] = vec

    logger.info(
        "Computed %d concept vectors (dim=%d, normalized=%s)",
        len(vectors), activations.shape[1], normalize,
    )
    return vectors


def compute_paired_vectors(
    activations_a: Tensor,
    activations_b: Tensor,
    normalize: bool = True,
) -> Tensor:
    """Compute a paired contrastive vector: mean(A) - mean(B).

    Useful when you have two specific groups (e.g., true vs. false statements)
    rather than many concepts against a global mean.

    Args:
        activations_a: Tensor of shape (n_a, hidden_dim) for group A.
        activations_b: Tensor of shape (n_b, hidden_dim) for group B.
        normalize: Whether to L2-normalize the result.

    Returns:
        Contrastive vector of shape (hidden_dim,).

    Raises:
        ValueError: If the hidden dimensions don't match.
    """
    if activations_a.shape[1] != activations_b.shape[1]:
        raise ValueError(
            f"Hidden dim mismatch: A has {activations_a.shape[1]}, "
            f"B has {activations_b.shape[1]}"
        )

    mean_a = activations_a.detach().float().mean(dim=0)
    mean_b = activations_b.detach().float().mean(dim=0)
    vec = mean_a - mean_b

    if normalize:
        norm = vec.norm()
        if norm > 1e-8:
            vec = vec / norm
        else:
            logger.warning(
                "Paired vector has near-zero norm (%.2e); "
                "returning unnormalized.",
                norm.item(),
            )

    return vec


def compute_similarity_matrix(
    vectors: dict[str, Tensor],
) -> tuple[np.ndarray, list[str]]:
    """Cosine similarity matrix between all pairs of concept vectors.

    Args:
        vectors: Dict mapping concept name to vector tensor.

    Returns:
        (similarity_matrix, concept_names) where similarity_matrix has shape
        (n_concepts, n_concepts) and concept_names[i] labels row/col i.
    """
    names = sorted(vectors.keys())
    n = len(names)

    if n == 0:
        return np.empty((0, 0)), []

    # Stack into (n, d) matrix
    stacked = torch.stack([vectors[name].detach().float() for name in names])

    # Normalize rows (handles already-normalized vectors gracefully)
    norms = stacked.norm(dim=1, keepdim=True).clamp(min=1e-8)
    normalized = stacked / norms

    sim = (normalized @ normalized.T).numpy()

    return sim, names


def compute_pca(
    vectors: dict[str, Tensor],
    n_components: int = 5,
) -> PCAResult:
    """PCA on concept vectors to visualize their geometry.

    Projects concept vectors into a low-dimensional space to see clustering,
    opposition, and other geometric structure.

    Args:
        vectors: Dict mapping concept name to vector tensor.
        n_components: Number of principal components. Clamped to
            min(n_concepts, hidden_dim).

    Returns:
        PCAResult with components, explained variance, and per-concept projections.
    """
    from sklearn.decomposition import PCA

    names = sorted(vectors.keys())
    if len(names) == 0:
        return PCAResult(
            components=np.empty((0, 0)),
            explained_variance_ratio=np.empty(0),
            projected={},
            concept_names=[],
        )

    stacked = torch.stack([vectors[name].detach().float() for name in names]).numpy()

    # Clamp n_components to valid range
    max_components = min(stacked.shape[0], stacked.shape[1])
    n_components = min(n_components, max_components)

    pca = PCA(n_components=n_components)
    coords = pca.fit_transform(stacked)

    projected = {name: coords[i] for i, name in enumerate(names)}

    logger.info(
        "PCA: %d components explain %.1f%% of variance",
        n_components,
        100.0 * pca.explained_variance_ratio_.sum(),
    )

    return PCAResult(
        components=pca.components_,
        explained_variance_ratio=pca.explained_variance_ratio_,
        projected=projected,
        concept_names=names,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_string_list(labels: list[str] | list[int] | np.ndarray | Sequence) -> list[str]:
    """Convert labels to a list of strings."""
    if isinstance(labels, np.ndarray):
        return [str(x) for x in labels.tolist()]
    return [str(x) for x in labels]
