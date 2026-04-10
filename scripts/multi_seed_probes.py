"""Multi-seed probe re-training on cached activations.

The harness's reported probe accuracies are single-seed point estimates
(`random_state=42` for both StratifiedKFold and LogisticRegression).
External critique #2-5 flagged this as a methodology gap. v3.2 adds
error bars by re-training probes with 5 different seeds on the
*same* cached activation tensors.

Key observation: model forward passes are deterministic given the input,
so seed variation in activation extraction is meaningless. The only
source of randomness in our reportable metrics is probe training, and
that runs entirely on CPU.

For each model:
  1. Load cached per-stimulus activations (375 files × N layers each)
  2. Reconstruct the (n_stimuli, d_model) activation matrix at each
     of the top-3 probe-accuracy layers from the cached result.json
  3. Train a 5-fold CV LogisticRegression with 5 different
     `random_state` values
  4. Report mean ± std of the 5 mean-fold-accuracies

Usage:
    MECHINTERP_DATA_ROOT=./drive_data python3 scripts/multi_seed_probes.py

Output:
    drive_data/results/emotions/multi_seed_probes.json
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

CONCEPTS = [
    "happy", "sad", "afraid", "angry", "calm", "guilty", "proud",
    "nervous", "loving", "hostile", "desperate", "enthusiastic",
    "vulnerable", "stubborn", "blissful",
]
N_PER_CONCEPT = 25
SEEDS = [42, 123, 7, 99, 2024]


def load_cached_activations(
    model_key: str, data_root: Path, layer: int
) -> tuple[np.ndarray, list[str]]:
    """Load cached per-stimulus activations into a (n_stimuli, d_model) matrix.

    The harness saves one .pt file per stimulus, each containing a dict
    {layer_idx: tensor(d_model)}. We reconstruct the matrix in the same
    stimulus order the original probe extraction used: outer loop over
    concept (in config order), inner loop over the first N_PER_CONCEPT
    items from the concept's training file.
    """
    activations_dir = data_root / "results" / "emotions" / model_key / "emotion_probe_classification" / "activations"

    vectors: list[np.ndarray] = []
    labels: list[str] = []
    stimulus_idx = 0

    for concept in CONCEPTS:
        # The probe loader reads the concept's training file but only
        # iterates over the first N_PER_CONCEPT items. We just need to
        # know how many cached files belong to each concept; the order
        # is concept-major.
        for _ in range(N_PER_CONCEPT):
            f = activations_dir / f"stimulus_{stimulus_idx:04d}.pt"
            if not f.exists():
                raise FileNotFoundError(f"Missing cached activation: {f}")
            d = torch.load(f, weights_only=False, map_location="cpu")
            if layer not in d:
                raise KeyError(f"Layer {layer} not in {f.name}; available: {sorted(d.keys())}")
            vectors.append(d[layer].float().cpu().numpy())
            labels.append(concept)
            stimulus_idx += 1

    X = np.stack(vectors, axis=0)
    return X, labels


def train_probe_one_seed(
    X: np.ndarray, y: np.ndarray, seed: int, n_folds: int = 5
) -> dict:
    """Train one 5-fold CV logistic regression with a fixed seed.

    The seed controls BOTH the StratifiedKFold split and the
    LogisticRegression random_state. Returns mean accuracy and per-fold list.
    """
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_accs: list[float] = []
    for train_idx, test_idx in skf.split(X, y):
        clf = LogisticRegression(
            max_iter=1000, solver="lbfgs", random_state=seed, C=1.0,
        )
        clf.fit(X[train_idx], y[train_idx])
        pred = clf.predict(X[test_idx])
        fold_accs.append(float(accuracy_score(y[test_idx], pred)))
    return {
        "seed": seed,
        "mean_accuracy": float(np.mean(fold_accs)),
        "std_accuracy": float(np.std(fold_accs)),
        "fold_accuracies": fold_accs,
    }


def get_top_layers(model_key: str, data_root: Path, k: int = 3) -> list[int]:
    """Return the top-k layers by cached probe accuracy."""
    result_path = data_root / "results" / "emotions" / model_key / "emotion_probe_classification" / "result.json"
    with open(result_path) as f:
        d = json.load(f)
    per_layer = d["metrics"]["per_layer_accuracy"]
    sorted_layers = sorted(per_layer.items(), key=lambda kv: kv[1], reverse=True)
    return [int(L) for L, _ in sorted_layers[:k]]


def run_model(model_key: str, data_root: Path) -> dict:
    """Run multi-seed probes for one model at the top-3 layers."""
    logger.info("=== %s ===", model_key)
    top_layers = get_top_layers(model_key, data_root, k=3)
    logger.info("  top-3 probe layers: %s", top_layers)

    le = LabelEncoder()
    y = le.fit_transform(CONCEPTS * N_PER_CONCEPT)  # placeholder; actual labels loaded with each layer

    per_layer_results: dict[int, dict] = {}
    overall_seeds_at_best: list[float] = []  # for headline error bar

    for layer in top_layers:
        t0 = time.time()
        try:
            X, labels = load_cached_activations(model_key, data_root, layer)
        except (FileNotFoundError, KeyError) as e:
            logger.warning("  L%d: %s", layer, e)
            continue
        y_layer = le.fit_transform(labels)
        load_time = time.time() - t0

        seed_results = []
        for seed in SEEDS:
            r = train_probe_one_seed(X, y_layer, seed=seed)
            seed_results.append(r)
        means = [r["mean_accuracy"] for r in seed_results]
        per_layer_results[layer] = {
            "n_stimuli": int(X.shape[0]),
            "d_model": int(X.shape[1]),
            "load_time_s": load_time,
            "seed_results": seed_results,
            "mean_across_seeds": float(np.mean(means)),
            "std_across_seeds": float(np.std(means)),
            "min_seed_mean": float(np.min(means)),
            "max_seed_mean": float(np.max(means)),
        }
        logger.info(
            "  L%d: %.4f ± %.4f  (range %.4f - %.4f, n_stimuli=%d, d=%d, load %.1fs)",
            layer,
            float(np.mean(means)),
            float(np.std(means)),
            float(np.min(means)),
            float(np.max(means)),
            X.shape[0], X.shape[1], load_time,
        )
        # Use the first (= highest probe accuracy) layer as the headline
        if layer == top_layers[0]:
            overall_seeds_at_best = means

    return {
        "model_key": model_key,
        "top_3_layers": top_layers,
        "per_layer": per_layer_results,
        "headline_layer": top_layers[0] if top_layers else None,
        "headline_mean": float(np.mean(overall_seeds_at_best)) if overall_seeds_at_best else None,
        "headline_std": float(np.std(overall_seeds_at_best)) if overall_seeds_at_best else None,
        "headline_seeds": overall_seeds_at_best,
    }


def main() -> int:
    import argparse, os
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=os.environ.get("MECHINTERP_DATA_ROOT", "drive_data"))
    parser.add_argument("--models", nargs="+", default=["llama_1b", "qwen_1_5b", "gemma_2b", "llama_8b", "qwen_7b", "gemma_9b"])
    args = parser.parse_args()

    data_root = Path(args.data_root)
    all_results = {}
    overall_t0 = time.time()
    for m in args.models:
        try:
            all_results[m] = run_model(m, data_root)
        except Exception as e:
            logger.exception("Failed for %s: %s", m, e)
            all_results[m] = {"error": str(e)}

    out_path = data_root / "results" / "emotions" / "multi_seed_probes.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path.with_suffix(".tmp"), "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    out_path.with_suffix(".tmp").rename(out_path)

    elapsed = time.time() - overall_t0
    logger.info("\n=== ALL DONE in %.1fs ===", elapsed)
    logger.info("Saved %s", out_path)

    # Print summary table
    print()
    print("=" * 78)
    print("MULTI-SEED PROBE SUMMARY (5 seeds per layer, top-3 layers per model)")
    print("=" * 78)
    print(f"  {'Model':14s}  {'best layer':>10s}  {'mean ± std':>14s}  {'range':>16s}")
    print("-" * 78)
    for m, r in all_results.items():
        if "error" in r:
            print(f"  {m:14s}  ERROR")
            continue
        bl = r.get("headline_layer", "?")
        mean = r.get("headline_mean", 0)
        std = r.get("headline_std", 0)
        seeds = r.get("headline_seeds", [])
        rng = f"{min(seeds):.4f}-{max(seeds):.4f}" if seeds else "?"
        print(f"  {m:14s}  {bl:>10}  {mean:.4f} ± {std:.4f}  {rng:>16s}")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
