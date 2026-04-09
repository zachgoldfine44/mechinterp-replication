"""Metrics utilities: cosine similarity, accuracy, effect sizes, confidence intervals.

Reusable statistical and geometric measures used across experiments.
All functions are pure (no side effects) and work with both numpy arrays
and torch tensors as documented.

Usage:
    from src.utils.metrics import cosine_similarity, bootstrap_ci, effect_size_cohens_d

    sim = cosine_similarity(vec_a, vec_b)
    mean, lo, hi = bootstrap_ci(accuracy_scores, ci=0.95)
    d = effect_size_cohens_d(group_a, group_b)
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from jaxtyping import Float
from torch import Tensor


def cosine_similarity(a: Float[Tensor, " d"], b: Float[Tensor, " d"]) -> float:
    """Cosine similarity between two vectors.

    Args:
        a: 1-D tensor of shape (d,).
        b: 1-D tensor of shape (d,).

    Returns:
        Scalar cosine similarity in [-1, 1].
    """
    return (torch.dot(a, b) / (a.norm() * b.norm()).clamp(min=1e-8)).item()


def cosine_similarity_matrix(
    vectors: Float[Tensor, "n d"],
) -> Float[Tensor, "n n"]:
    """Pairwise cosine similarity matrix for a set of vectors.

    Args:
        vectors: Tensor of shape (n, d).

    Returns:
        Symmetric (n, n) tensor where entry (i, j) is the cosine
        similarity between vectors[i] and vectors[j].
    """
    norms = vectors.norm(dim=1, keepdim=True)
    normalized = vectors / norms.clamp(min=1e-8)
    return normalized @ normalized.T


def accuracy(predictions: Sequence, labels: Sequence) -> float:
    """Simple classification accuracy.

    Args:
        predictions: Predicted labels (any comparable type).
        labels: Ground-truth labels (same length as predictions).

    Returns:
        Fraction of correct predictions in [0, 1].

    Raises:
        ValueError: If inputs have different lengths or are empty.
    """
    if len(predictions) != len(labels):
        raise ValueError(
            f"Length mismatch: {len(predictions)} predictions vs {len(labels)} labels"
        )
    if len(labels) == 0:
        raise ValueError("Cannot compute accuracy on empty inputs")

    correct = sum(p == l for p, l in zip(predictions, labels))
    return correct / len(labels)


def bootstrap_ci(
    data: np.ndarray,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Bootstrap confidence interval for the mean.

    Args:
        data: 1-D array of observations.
        n_bootstrap: Number of bootstrap resamples.
        ci: Confidence level (e.g., 0.95 for 95% CI).
        seed: Random seed for reproducibility.

    Returns:
        (mean, lower_bound, upper_bound) tuple.
    """
    rng = np.random.RandomState(seed)
    boot_means = np.empty(n_bootstrap)

    for i in range(n_bootstrap):
        sample = rng.choice(data, size=len(data), replace=True)
        boot_means[i] = np.mean(sample)

    boot_means.sort()
    alpha = (1 - ci) / 2
    lower = float(boot_means[int(alpha * n_bootstrap)])
    upper = float(boot_means[int((1 - alpha) * n_bootstrap)])
    return float(np.mean(data)), lower, upper


def effect_size_cohens_d(group_a: np.ndarray, group_b: np.ndarray) -> float:
    """Cohen's d effect size between two groups.

    Uses pooled standard deviation. Returns 0.0 if both groups have
    near-zero variance (avoids division by zero).

    Args:
        group_a: 1-D array of observations for group A.
        group_b: 1-D array of observations for group B.

    Returns:
        Cohen's d (positive means group_a > group_b).
    """
    na, nb = len(group_a), len(group_b)
    var_a = np.var(group_a, ddof=1) if na > 1 else 0.0
    var_b = np.var(group_b, ddof=1) if nb > 1 else 0.0
    pooled_std = np.sqrt(((na - 1) * var_a + (nb - 1) * var_b) / max(na + nb - 2, 1))

    if pooled_std < 1e-8:
        return 0.0

    return float((np.mean(group_a) - np.mean(group_b)) / pooled_std)


def diagonal_dominance(confusion_matrix: np.ndarray) -> float:
    """Fraction of classes where the diagonal is the row maximum.

    Used to evaluate generalization: if a probe trained on concept X
    fires most strongly on concept X stimuli, the diagonal dominates.

    Args:
        confusion_matrix: Square (n, n) array where entry (i, j) is
            some score for true-class i, predicted/activated-class j.

    Returns:
        Fraction in [0, 1]. 1.0 means every class's diagonal entry
        is the max in its row.
    """
    n = confusion_matrix.shape[0]
    if n == 0:
        return 0.0

    dominant = 0
    for i in range(n):
        row_max = confusion_matrix[i].max()
        if confusion_matrix[i, i] == row_max and row_max > 0:
            dominant += 1

    return dominant / n
