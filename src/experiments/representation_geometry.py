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
        from src.core.experiment import _results_dir_for
        dep_results_dir = _results_dir_for(
            data_root=self.data_root,
            paper_id=self.config.paper_id,
            replication_id=self.config.replication_id,
            model_key=self.model_key,
            claim_id=self.dependency_claim,
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
        """Correlate PC1 with valence and PC2 with arousal.

        The original Sofroniew et al. (2026) claim is specifically that
        PC1 of the emotion-vector PCA correlates with valence and PC2
        with arousal. We test that specific claim, not max-over-all-PCs.

        We report |Pearson r| because the sign of a PC is arbitrary
        (PCA components can flip sign under repeated runs / different
        seeds without changing the geometry).

        We also compute a 95% bootstrap CI on the |r| value, since
        with N=15 emotions the point estimate has substantial noise.

        For exploratory completeness, the returned `pc_correlations`
        dict also includes raw r and p for every PC, but those values
        are NOT what `valence_correlation` / `arousal_correlation`
        report.

        Returns:
            (pc1_valence_abs_r, pc2_arousal_abs_r, detailed_pc_correlations)
        """
        pc_correlations: dict[str, Any] = {}

        valence_concepts = [c for c in concepts if c in self.valence_labels]
        arousal_concepts = [c for c in concepts if c in self.arousal_labels]

        n_pcs = X_pca.shape[1]

        # Exploratory: compute correlation for ALL PCs (kept in metadata only).
        for pc_idx in range(n_pcs):
            pc_key = f"PC{pc_idx + 1}"
            pc_correlations[pc_key] = {}

            if len(valence_concepts) >= 3:
                pc_vals = [X_pca[concepts.index(c), pc_idx] for c in valence_concepts]
                val_ratings = [self.valence_labels[c] for c in valence_concepts]
                r_val, p_val = stats.pearsonr(pc_vals, val_ratings)
                pc_correlations[pc_key]["valence_r"] = float(r_val)
                pc_correlations[pc_key]["valence_abs_r"] = float(abs(r_val))
                pc_correlations[pc_key]["valence_p"] = float(p_val)

            if len(arousal_concepts) >= 3:
                pc_vals = [X_pca[concepts.index(c), pc_idx] for c in arousal_concepts]
                ar_ratings = [self.arousal_labels[c] for c in arousal_concepts]
                r_ar, p_ar = stats.pearsonr(pc_vals, ar_ratings)
                pc_correlations[pc_key]["arousal_r"] = float(r_ar)
                pc_correlations[pc_key]["arousal_abs_r"] = float(abs(r_ar))
                pc_correlations[pc_key]["arousal_p"] = float(p_ar)

        if not valence_concepts:
            logger.warning("No valence ratings provided; skipping valence correlation")
        if not arousal_concepts:
            logger.warning("No arousal ratings provided; skipping arousal correlation")

        # PRIMARY METRIC: PC1 vs valence and PC2 vs arousal (the paper's claim).
        pc1_valence = float(pc_correlations.get("PC1", {}).get("valence_abs_r", 0.0))
        pc2_arousal = float(pc_correlations.get("PC2", {}).get("arousal_abs_r", 0.0))

        # Compute bootstrap CIs and p-values for the primary metric.
        if len(valence_concepts) >= 3:
            pc1_vals = np.array([X_pca[concepts.index(c), 0] for c in valence_concepts])
            val_arr = np.array([self.valence_labels[c] for c in valence_concepts])
            r_pc1, p_pc1 = stats.pearsonr(pc1_vals, val_arr)
            pc_correlations["PC1_valence_signed_r"] = float(r_pc1)
            pc_correlations["PC1_valence_p"] = float(p_pc1)
            pc_correlations["PC1_valence_n"] = int(len(valence_concepts))
            pc_correlations["PC1_valence_abs_ci95"] = self._bootstrap_abs_pearson_ci(
                pc1_vals, val_arr, n_boot=2000, ci=0.95
            )

        if len(arousal_concepts) >= 3:
            pc2_vals = np.array([X_pca[concepts.index(c), 1] for c in arousal_concepts])
            ar_arr = np.array([self.arousal_labels[c] for c in arousal_concepts])
            r_pc2, p_pc2 = stats.pearsonr(pc2_vals, ar_arr)
            pc_correlations["PC2_arousal_signed_r"] = float(r_pc2)
            pc_correlations["PC2_arousal_p"] = float(p_pc2)
            pc_correlations["PC2_arousal_n"] = int(len(arousal_concepts))
            pc_correlations["PC2_arousal_abs_ci95"] = self._bootstrap_abs_pearson_ci(
                pc2_vals, ar_arr, n_boot=2000, ci=0.95
            )

        # Also compute the "max over all PCs" (the OLD metric) so consumers
        # can see the gap between the claim-faithful PC1 metric and the
        # multiple-comparisons-inflated max metric. This is useful for
        # transparently documenting the methodological correction.
        # Only iterate over per-PC dict entries (PC1, PC2, ...), not over
        # the scalar summary fields like PC1_valence_signed_r.
        per_pc_keys = [
            k for k in pc_correlations
            if k.startswith("PC")
            and isinstance(pc_correlations[k], dict)
        ]
        all_valence_abs = [
            pc_correlations[k].get("valence_abs_r", 0.0)
            for k in per_pc_keys
            if "valence_abs_r" in pc_correlations[k]
        ]
        all_arousal_abs = [
            pc_correlations[k].get("arousal_abs_r", 0.0)
            for k in per_pc_keys
            if "arousal_abs_r" in pc_correlations[k]
        ]
        pc_correlations["max_valence_abs_r_any_pc"] = max(all_valence_abs) if all_valence_abs else 0.0
        pc_correlations["max_arousal_abs_r_any_pc"] = max(all_arousal_abs) if all_arousal_abs else 0.0

        return pc1_valence, pc2_arousal, pc_correlations

    @staticmethod
    def _bootstrap_abs_pearson_ci(
        x: np.ndarray, y: np.ndarray, n_boot: int = 2000, ci: float = 0.95, seed: int = 42
    ) -> tuple[float, float]:
        """Bootstrap CI on |Pearson r| of (x, y).

        Returns (lower, upper) for the central CI. With N as small as 15
        the point estimate is noisy enough that this CI is essential for
        honest reporting.
        """
        rng = np.random.default_rng(seed)
        n = len(x)
        if n < 3:
            return (0.0, 0.0)
        rs = np.zeros(n_boot)
        for i in range(n_boot):
            idx = rng.integers(0, n, size=n)
            xs = x[idx]
            ys = y[idx]
            if np.std(xs) < 1e-9 or np.std(ys) < 1e-9:
                rs[i] = 0.0
                continue
            r, _ = stats.pearsonr(xs, ys)
            rs[i] = abs(r)
        alpha = (1 - ci) / 2
        lo = float(np.quantile(rs, alpha))
        hi = float(np.quantile(rs, 1 - alpha))
        return (lo, hi)

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
