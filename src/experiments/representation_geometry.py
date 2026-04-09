"""Representation geometry experiment: PCA and clustering of concept vectors.

Analyzes the geometric structure of concept vectors in activation space,
testing whether learned representations align with known psychological
dimensions (e.g., valence-arousal for emotions).

Expected params in ClaimConfig.params:
    n_components: int                   -- PCA components to compute
    valence_labels: dict[str, float]    -- human valence ratings per concept
    arousal_labels: dict[str, float]    -- human arousal ratings per concept
    analysis_type: str                  -- "pca" (default)
    correlation_target: str             -- "pc1_valence" (default)

Outputs:
    metrics["valence_correlation"]      -- |Pearson r| between PC1 and valence
    metrics["arousal_correlation"]      -- |Pearson r| between PC2 and arousal
    metrics["explained_variance"]       -- list of variance ratios per PC
    metrics["n_clusters"]               -- optimal k from k-means (silhouette)
    metrics["cluster_labels"]           -- dict[concept, cluster_id]

Depends on:
    A prior probe_classification run (uses its concept_vectors.pt).

Usage:
    from src.experiments.representation_geometry import RepresentationGeometryExperiment
    exp = RepresentationGeometryExperiment(claim_config, "llama_1b", data_root)
    result = exp.load_or_run(model, tokenizer, activations_cache)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from src.core.claim import ClaimConfig, ExperimentResult
from src.core.experiment import Experiment

logger = logging.getLogger(__name__)


class RepresentationGeometryExperiment(Experiment):
    """Analyze geometric structure of concept vectors in activation space.

    Loads concept vectors computed by a prior probe_classification experiment,
    runs PCA to find principal dimensions, and tests whether those dimensions
    correlate with known human-rated scales (valence, arousal). Also runs
    k-means clustering to identify natural groupings.
    """

    def __init__(self, config: ClaimConfig, model_key: str, data_root: Path) -> None:
        super().__init__(config, model_key, data_root)
        self.n_components: int = config.params.get("n_components", 5)
        self.valence_labels: dict[str, float] = config.params.get("valence_labels", {})
        self.arousal_labels: dict[str, float] = config.params.get("arousal_labels", {})
        self.dependency_claim: str = config.depends_on or ""
        if not self.dependency_claim:
            raise ValueError(
                "RepresentationGeometryExperiment requires 'depends_on' to locate concept vectors."
            )

    def run(
        self, model: Any, tokenizer: Any, activations_cache: Any
    ) -> ExperimentResult:
        """Run representation geometry analysis.

        Args:
            model: Loaded language model (may not be needed if vectors are cached).
            tokenizer: The model's tokenizer.
            activations_cache: Shared activation cache, or None.

        Returns:
            ExperimentResult with valence_correlation and geometry metrics.
        """
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Step 1: Load concept vectors from dependency
        dep_results_dir = (
            self.data_root / "results" / self.config.paper_id
            / self.model_key / self.dependency_claim
        )
        cv_path = dep_results_dir / "concept_vectors.pt"
        if not cv_path.exists():
            raise FileNotFoundError(
                f"Concept vectors not found at {cv_path}. "
                f"Run claim '{self.dependency_claim}' first."
            )

        concept_vectors_all = torch.load(cv_path, map_location="cpu", weights_only=False)

        # Step 2: Find the best layer from dependency results
        dep_result_path = dep_results_dir / "result.json"
        if dep_result_path.exists():
            with open(dep_result_path) as f:
                dep_result = json.load(f)
            best_layer = int(dep_result["metrics"]["best_layer"])
        else:
            # Use the first available layer
            best_layer = min(concept_vectors_all.keys())

        concept_vectors = concept_vectors_all[best_layer]
        concepts = sorted(concept_vectors.keys())
        logger.info(
            "Loaded %d concept vectors from layer %d", len(concepts), best_layer
        )

        if len(concepts) < 3:
            logger.warning("Only %d concepts -- geometry analysis needs >= 3", len(concepts))
            return ExperimentResult(
                claim_id=self.config.claim_id,
                model_key=self.model_key,
                paper_id=self.config.paper_id,
                metrics={
                    "valence_correlation": 0.0,
                    "arousal_correlation": 0.0,
                    "error": "Too few concepts for geometry analysis",
                },
            )

        # Step 3: Stack vectors into matrix
        vectors_matrix = torch.stack([concept_vectors[c] for c in concepts])
        X = vectors_matrix.numpy()  # (n_concepts, hidden_dim)

        # Step 4: PCA
        n_components = min(self.n_components, len(concepts))
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X)  # (n_concepts, n_components)
        explained_variance = pca.explained_variance_ratio_.tolist()

        logger.info(
            "PCA explained variance: %s",
            [f"PC{i+1}={v:.3f}" for i, v in enumerate(explained_variance)],
        )

        # Step 5: Correlate PCs with valence and arousal
        valence_corr, arousal_corr, pc_correlations = self._correlate_with_ratings(
            X_pca, concepts
        )

        # Step 6: K-means clustering with silhouette analysis
        n_clusters, cluster_labels, silhouette = self._find_clusters(X_pca, concepts)

        # Step 7: Compute pairwise cosine similarity matrix
        norms = vectors_matrix.norm(dim=1, keepdim=True).clamp(min=1e-8)
        cos_sim = (vectors_matrix / norms) @ (vectors_matrix / norms).T
        cos_sim_dict = {
            f"{concepts[i]}_{concepts[j]}": float(cos_sim[i, j])
            for i in range(len(concepts))
            for j in range(i + 1, len(concepts))
        }

        # Save PCA results for visualization
        pca_results = {
            "concepts": concepts,
            "pca_coords": X_pca.tolist(),
            "explained_variance": explained_variance,
            "cluster_labels": cluster_labels,
        }
        pca_path = self.results_dir / "pca_results.json"
        tmp = pca_path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(pca_results, f, indent=2)
        tmp.rename(pca_path)

        metrics = {
            "valence_correlation": valence_corr,
            "arousal_correlation": arousal_corr,
            "pc_correlations": pc_correlations,
            "explained_variance": explained_variance,
            "total_variance_explained": sum(explained_variance),
            "n_clusters": n_clusters,
            "cluster_labels": cluster_labels,
            "silhouette_score": silhouette,
            "best_layer": best_layer,
            "n_concepts": len(concepts),
            "top_cosine_similarities": dict(
                sorted(cos_sim_dict.items(), key=lambda x: -x[1])[:10]
            ),
        }

        metadata = {
            "n_components": n_components,
            "dependency_claim": self.dependency_claim,
            "valence_labels_provided": bool(self.valence_labels),
            "arousal_labels_provided": bool(self.arousal_labels),
        }

        return ExperimentResult(
            claim_id=self.config.claim_id,
            model_key=self.model_key,
            paper_id=self.config.paper_id,
            metrics=metrics,
            metadata=metadata,
        )

    def evaluate(self, result: ExperimentResult) -> bool:
        """Check if the target correlation meets the success threshold."""
        metric_val = result.metrics.get(self.config.success_metric, 0.0)
        return float(metric_val) >= self.config.success_threshold

    # ── Private helpers ─────────────────────────────────────────────────────

    def _correlate_with_ratings(
        self,
        X_pca: np.ndarray,
        concepts: list[str],
    ) -> tuple[float, float, dict[str, Any]]:
        """Correlate each PC with valence and arousal ratings.

        Tests all PCs against both scales and reports the best match.
        Standard approach: PC1 should correlate with valence, PC2 with arousal,
        but we check all combinations.

        Returns:
            (best_valence_corr, best_arousal_corr, detailed_pc_correlations)
        """
        pc_correlations: dict[str, Any] = {}

        # Build rating vectors (only for concepts that have ratings)
        valence_concepts = [c for c in concepts if c in self.valence_labels]
        arousal_concepts = [c for c in concepts if c in self.arousal_labels]

        best_valence_corr = 0.0
        best_arousal_corr = 0.0

        n_pcs = X_pca.shape[1]

        for pc_idx in range(n_pcs):
            pc_key = f"PC{pc_idx + 1}"
            pc_correlations[pc_key] = {}

            # Valence correlation
            if len(valence_concepts) >= 3:
                pc_vals = [X_pca[concepts.index(c), pc_idx] for c in valence_concepts]
                val_ratings = [self.valence_labels[c] for c in valence_concepts]
                r_val, p_val = stats.pearsonr(pc_vals, val_ratings)
                pc_correlations[pc_key]["valence_r"] = float(r_val)
                pc_correlations[pc_key]["valence_p"] = float(p_val)

                if abs(r_val) > abs(best_valence_corr):
                    best_valence_corr = abs(float(r_val))

            # Arousal correlation
            if len(arousal_concepts) >= 3:
                pc_vals = [X_pca[concepts.index(c), pc_idx] for c in arousal_concepts]
                ar_ratings = [self.arousal_labels[c] for c in arousal_concepts]
                r_ar, p_ar = stats.pearsonr(pc_vals, ar_ratings)
                pc_correlations[pc_key]["arousal_r"] = float(r_ar)
                pc_correlations[pc_key]["arousal_p"] = float(p_ar)

                if abs(r_ar) > abs(best_arousal_corr):
                    best_arousal_corr = abs(float(r_ar))

        if not valence_concepts:
            logger.warning("No valence ratings provided; skipping valence correlation")
        if not arousal_concepts:
            logger.warning("No arousal ratings provided; skipping arousal correlation")

        return best_valence_corr, best_arousal_corr, pc_correlations

    def _find_clusters(
        self,
        X_pca: np.ndarray,
        concepts: list[str],
    ) -> tuple[int, dict[str, int], float]:
        """Find optimal number of clusters via silhouette analysis.

        Tests k=2 through min(10, n_concepts-1) and picks the k with
        the highest silhouette score.

        Returns:
            (best_k, concept_to_cluster_dict, best_silhouette_score)
        """
        n = len(concepts)
        if n < 3:
            # Cannot cluster fewer than 3 points meaningfully
            return 1, {c: 0 for c in concepts}, 0.0

        max_k = min(10, n - 1)
        best_k = 2
        best_sil = -1.0
        best_labels: np.ndarray | None = None

        for k in range(2, max_k + 1):
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = km.fit_predict(X_pca)

            # Need at least 2 distinct labels for silhouette
            if len(set(cluster_labels)) < 2:
                continue

            sil = float(silhouette_score(X_pca, cluster_labels))
            if sil > best_sil:
                best_sil = sil
                best_k = k
                best_labels = cluster_labels

        if best_labels is None:
            best_labels = np.zeros(n, dtype=int)

        cluster_dict = {concepts[i]: int(best_labels[i]) for i in range(n)}
        return best_k, cluster_dict, best_sil
