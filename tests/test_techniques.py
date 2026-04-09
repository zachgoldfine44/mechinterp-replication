"""Tests for technique modules with synthetic data (no real models required).

Each test uses random tensors or linearly separable synthetic data to verify
that probes, contrastive vectors, PCA, similarity matrices, and steering
utilities work correctly.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch


# ---------------------------------------------------------------------------
# Probes
# ---------------------------------------------------------------------------

class TestProbeLogisticRegression:
    """Test probe training with logistic regression."""

    def test_probe_logistic_regression(
        self, separable_activations: tuple[torch.Tensor, list[str]]
    ) -> None:
        """Logistic regression probe on separable data should achieve > 0.8 accuracy."""
        from src.techniques.probes import train_probe

        X, labels = separable_activations
        result = train_probe(X, labels, probe_type="logistic_regression", n_folds=3)

        assert result.mean_accuracy > 0.8, (
            f"Expected > 0.8 on separable data, got {result.mean_accuracy:.3f}"
        )
        assert result.probe_type == "logistic_regression"
        assert len(result.label_names) == 5

    def test_probe_mismatched_lengths(self) -> None:
        """Mismatched activation/label counts should raise ValueError."""
        from src.techniques.probes import train_probe

        X = torch.randn(10, 32)
        labels = ["a"] * 5  # wrong length
        with pytest.raises(ValueError, match="label count"):
            train_probe(X, labels)


class TestProbeCrossValidation:
    """Test k-fold cross-validation in probes."""

    @pytest.mark.parametrize("n_folds", [3, 5])
    def test_probe_cross_validation(
        self,
        separable_activations: tuple[torch.Tensor, list[str]],
        n_folds: int,
    ) -> None:
        """k-fold CV should produce exactly k fold results."""
        from src.techniques.probes import train_probe

        X, labels = separable_activations
        result = train_probe(X, labels, probe_type="logistic_regression", n_folds=n_folds)

        assert result.n_folds == n_folds
        assert len(result.fold_results) == n_folds
        for fr in result.fold_results:
            assert 0.0 <= fr.accuracy <= 1.0
            assert fr.train_size > 0
            assert fr.val_size > 0


class TestProbeCheckpointResume:
    """Test that probe training can resume from checkpoints."""

    def test_probe_checkpoint_resume(
        self,
        separable_activations: tuple[torch.Tensor, list[str]],
        tmp_path,
    ) -> None:
        """Running train_probe twice with same checkpoint_dir should be idempotent."""
        from src.techniques.probes import train_probe

        X, labels = separable_activations
        ckpt = tmp_path / "probe_ckpt"

        r1 = train_probe(X, labels, n_folds=3, checkpoint_dir=ckpt)
        r2 = train_probe(X, labels, n_folds=3, checkpoint_dir=ckpt)

        assert abs(r1.mean_accuracy - r2.mean_accuracy) < 1e-6


# ---------------------------------------------------------------------------
# Contrastive vectors
# ---------------------------------------------------------------------------

class TestConceptVectors:
    """Test contrastive concept vector computation."""

    def test_concept_vectors(
        self, sample_activations: torch.Tensor, sample_labels: list[str]
    ) -> None:
        """Concept vectors should be unit-norm and one per concept."""
        from src.techniques.contrastive import compute_concept_vectors

        vectors = compute_concept_vectors(sample_activations, sample_labels, normalize=True)

        assert len(vectors) == 5
        for name, vec in vectors.items():
            assert vec.shape == (2048,)
            # Should be approximately unit norm
            assert abs(vec.norm().item() - 1.0) < 0.01, (
                f"Concept {name!r} norm = {vec.norm().item():.4f}, expected ~1.0"
            )

    def test_concept_vectors_unnormalized(
        self, sample_activations: torch.Tensor, sample_labels: list[str]
    ) -> None:
        """Unnormalized concept vectors should NOT be unit norm in general."""
        from src.techniques.contrastive import compute_concept_vectors

        vectors = compute_concept_vectors(
            sample_activations, sample_labels, normalize=False
        )
        norms = [v.norm().item() for v in vectors.values()]
        # At least one should differ from 1.0 significantly (random data)
        assert any(abs(n - 1.0) > 0.01 for n in norms)


class TestPCA:
    """Test PCA on concept vectors."""

    def test_pca_explained_variance(
        self, sample_activations: torch.Tensor, sample_labels: list[str]
    ) -> None:
        """Explained variance ratios should sum to approximately 1.0 when all
        components are used (n_components = n_concepts)."""
        from src.techniques.contrastive import compute_concept_vectors, compute_pca

        vectors = compute_concept_vectors(sample_activations, sample_labels)
        pca_result = compute_pca(vectors, n_components=5)

        total = float(pca_result.explained_variance_ratio.sum())
        assert abs(total - 1.0) < 0.01, (
            f"Explained variance sums to {total:.4f}, expected ~1.0"
        )
        assert len(pca_result.projected) == 5
        assert len(pca_result.concept_names) == 5

    def test_pca_components_shape(
        self, sample_activations: torch.Tensor, sample_labels: list[str]
    ) -> None:
        """PCA components should have shape (n_components, hidden_dim)."""
        from src.techniques.contrastive import compute_concept_vectors, compute_pca

        vectors = compute_concept_vectors(sample_activations, sample_labels)
        pca_result = compute_pca(vectors, n_components=3)

        assert pca_result.components.shape == (3, 2048)


# ---------------------------------------------------------------------------
# Similarity matrix
# ---------------------------------------------------------------------------

class TestSimilarityMatrix:
    """Test cosine similarity matrix computation."""

    def test_similarity_matrix_properties(
        self, sample_activations: torch.Tensor, sample_labels: list[str]
    ) -> None:
        """Similarity matrix should be symmetric with 1s on the diagonal."""
        from src.techniques.contrastive import (
            compute_concept_vectors,
            compute_similarity_matrix,
        )

        vectors = compute_concept_vectors(sample_activations, sample_labels)
        sim_matrix, names = compute_similarity_matrix(vectors)

        n = len(names)
        assert sim_matrix.shape == (n, n)

        # Symmetry
        assert np.allclose(sim_matrix, sim_matrix.T, atol=1e-6), (
            "Similarity matrix is not symmetric"
        )

        # Diagonal should be 1.0 (cosine self-similarity)
        for i in range(n):
            assert abs(sim_matrix[i, i] - 1.0) < 1e-5, (
                f"Diagonal element [{i},{i}] = {sim_matrix[i, i]:.6f}, expected 1.0"
            )

    def test_similarity_values_in_range(
        self, sample_activations: torch.Tensor, sample_labels: list[str]
    ) -> None:
        """All similarity values should be in [-1, 1]."""
        from src.techniques.contrastive import (
            compute_concept_vectors,
            compute_similarity_matrix,
        )

        vectors = compute_concept_vectors(sample_activations, sample_labels)
        sim_matrix, _ = compute_similarity_matrix(vectors)

        assert sim_matrix.min() >= -1.0 - 1e-6
        assert sim_matrix.max() <= 1.0 + 1e-6


# ---------------------------------------------------------------------------
# Steering utilities
# ---------------------------------------------------------------------------

class TestControlVector:
    """Test control vector creation for baseline comparisons."""

    def test_control_vector_random_same_norm(self) -> None:
        """Random control vector should have same L2 norm as original."""
        from src.techniques.steering import create_control_vector

        original = torch.randn(512)
        random_vec = create_control_vector(original, control_type="random")

        assert random_vec.shape == original.shape
        assert abs(random_vec.norm().item() - original.norm().item()) < 1e-4

    def test_control_vector_negated(self) -> None:
        """Negated control vector should be -original."""
        from src.techniques.steering import create_control_vector

        original = torch.randn(512)
        negated = create_control_vector(original, control_type="negated")

        assert torch.allclose(negated, -original, atol=1e-6)

    def test_control_vector_zero(self) -> None:
        """Zero control vector should be all zeros."""
        from src.techniques.steering import create_control_vector

        original = torch.randn(512)
        zero_vec = create_control_vector(original, control_type="zero")

        assert torch.allclose(zero_vec, torch.zeros_like(original))

    def test_control_vector_unknown_type(self) -> None:
        """Unknown control_type should raise ValueError."""
        from src.techniques.steering import create_control_vector

        with pytest.raises(ValueError, match="Unknown control_type"):
            create_control_vector(torch.randn(64), control_type="bogus")


class TestSteeringLayers:
    """Test steering layer selection strategies."""

    @pytest.mark.parametrize(
        "n_layers, strategy, expected",
        [
            (32, "middle_third", list(range(10, 21))),
            (32, "two_thirds", [21]),
            (32, "all", list(range(32))),
            (16, "middle_third", list(range(5, 10))),
            (16, "two_thirds", [11]),
            (1, "middle_third", [0]),
            (1, "two_thirds", [0]),
        ],
    )
    def test_steering_layers(
        self, n_layers: int, strategy: str, expected: list[int]
    ) -> None:
        """get_steering_layers should return correct layer indices."""
        from src.techniques.steering import get_steering_layers

        result = get_steering_layers(n_layers, strategy=strategy)
        assert result == expected

    def test_steering_layers_invalid_strategy(self) -> None:
        """Unknown strategy should raise ValueError."""
        from src.techniques.steering import get_steering_layers

        with pytest.raises(ValueError, match="Unknown strategy"):
            get_steering_layers(32, strategy="bottom_quarter")

    def test_steering_layers_zero_layers(self) -> None:
        """n_layers < 1 should raise ValueError."""
        from src.techniques.steering import get_steering_layers

        with pytest.raises(ValueError, match="n_layers must be >= 1"):
            get_steering_layers(0)


# ---------------------------------------------------------------------------
# Metrics utilities
# ---------------------------------------------------------------------------

class TestMetrics:
    """Test utility metric functions."""

    def test_cosine_similarity_identical(self) -> None:
        """Cosine similarity of a vector with itself should be 1.0."""
        from src.utils.metrics import cosine_similarity

        v = torch.randn(128)
        assert abs(cosine_similarity(v, v) - 1.0) < 1e-5

    def test_cosine_similarity_opposite(self) -> None:
        """Cosine similarity of a vector with its negative should be -1.0."""
        from src.utils.metrics import cosine_similarity

        v = torch.randn(128)
        assert abs(cosine_similarity(v, -v) + 1.0) < 1e-5

    def test_cosine_similarity_matrix_diagonal(self) -> None:
        """Diagonal of the cosine similarity matrix should be all 1s."""
        from src.utils.metrics import cosine_similarity_matrix

        vecs = torch.randn(10, 64)
        sim = cosine_similarity_matrix(vecs)
        diag = torch.diag(sim)
        assert torch.allclose(diag, torch.ones(10), atol=1e-5)

    def test_diagonal_dominance(self) -> None:
        """A perfect confusion matrix should have dominance = 1.0."""
        from src.utils.metrics import diagonal_dominance

        cm = np.diag([10, 10, 10]).astype(float)
        assert diagonal_dominance(cm) == 1.0

    def test_diagonal_dominance_partial(self) -> None:
        """A confusion matrix with some off-diagonal maxima should be < 1.0."""
        from src.utils.metrics import diagonal_dominance

        cm = np.array([[10, 2, 1], [3, 5, 8], [1, 2, 10]], dtype=float)
        dd = diagonal_dominance(cm)
        # Row 1 has max at col 2 (8 > 5), so 2/3 rows dominate
        assert abs(dd - 2.0 / 3.0) < 1e-6

    def test_bootstrap_ci(self) -> None:
        """Bootstrap CI should return (mean, lower, upper) with lower <= mean <= upper."""
        from src.utils.metrics import bootstrap_ci

        data = np.array([0.7, 0.75, 0.72, 0.68, 0.73])
        mean, lo, hi = bootstrap_ci(data, n_bootstrap=500)
        assert lo <= mean <= hi

    def test_effect_size_cohens_d(self) -> None:
        """Cohen's d should be positive when group_a > group_b."""
        from src.utils.metrics import effect_size_cohens_d

        a = np.array([10.0, 11.0, 12.0, 10.5, 11.5])
        b = np.array([5.0, 6.0, 5.5, 6.5, 5.0])
        d = effect_size_cohens_d(a, b)
        assert d > 0
