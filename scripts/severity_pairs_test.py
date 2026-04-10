"""Severity pairs test: does the fear vector distinguish danger from number magnitude?

The v3 negative-control finding: the parametric scaling result is heavily
contaminated by numerical-magnitude artifacts (a "blueberries" template
with the same numerical scaling as Tylenol dosage gives the same fear
vector response). v3 left open the question: is there ANY real
"severity" signal underneath the contamination, or is the whole effect
a number-magnitude artifact?

This script answers that question with a paired contrast:

  For each pair, both prompts contain the SAME literal number but
  DIFFERENT danger context (e.g. "I drank 500ml of water" vs "I drank
  500ml of bleach"). If the fear vector responds to severity (not just
  number magnitude), the dangerous prompt should give a substantially
  larger fear projection than the neutral prompt.

  Paired test:
    delta_pair = projection(dangerous) - projection(neutral)
  Aggregate across pairs: mean delta, std, fraction of pairs where
  dangerous > neutral, paired t-test.

This is a much cleaner test than the original parametric scaling
because the literal number is held constant per pair.

Usage:
    HF_TOKEN=... MECHINTERP_DATA_ROOT=./drive_data \\
        python3 scripts/severity_pairs_test.py --models llama_1b qwen_1_5b gemma_2b

For medium tier on RunPod:
    python3 scripts/severity_pairs_test.py --models llama_8b qwen_7b gemma_9b
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy import stats

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.registry import ModelRegistry
from src.models.loader import load_model, unload_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


@torch.no_grad()
def extract_one(model: Any, text: str, layer: int) -> torch.Tensor:
    """Extract residual stream at last token, single prompt, single layer."""
    hook_name = f"blocks.{layer}.hook_resid_post"
    tokens = model.to_tokens(text)
    _, cache = model.run_with_cache(tokens, names_filter=[hook_name])
    return cache[hook_name][0, -1, :].float().cpu()


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def paired_test(deltas: np.ndarray) -> dict:
    """Paired one-sample t-test against 0."""
    if len(deltas) < 2:
        return {"t": 0.0, "p": 1.0, "n": len(deltas)}
    t, p = stats.ttest_1samp(deltas, 0.0)
    return {"t": float(t), "p": float(p), "n": int(len(deltas))}


def run_one_model(model: Any, model_key: str, data_root: Path, pairs: list[dict]) -> dict:
    """For one model, project the severity pairs onto each emotion's fear/calm vector."""
    logger.info("=== %s ===", model_key)

    # Load best probe layer + concept vectors
    probe_path = data_root / "results" / "emotions" / model_key / "emotion_probe_classification" / "result.json"
    cv_path = data_root / "results" / "emotions" / model_key / "emotion_probe_classification" / "concept_vectors.pt"
    if not probe_path.exists() or not cv_path.exists():
        return {"error": f"Missing probe data for {model_key}"}
    with open(probe_path) as f:
        probe_result = json.load(f)
    best_layer = int(probe_result["metrics"]["best_layer"])
    cv_all = torch.load(cv_path, weights_only=False, map_location="cpu")
    if best_layer not in cv_all:
        return {"error": f"Best layer {best_layer} not in concept vectors"}
    concept_vectors = cv_all[best_layer]

    # We'll project on multiple concepts to be thorough
    target_concepts = ["afraid", "calm", "nervous", "desperate", "vulnerable", "happy"]
    available = [c for c in target_concepts if c in concept_vectors]
    logger.info("  best layer: %d, concepts: %s", best_layer, available)

    # Pre-normalize concept vectors
    vecs = {}
    for c in available:
        v = concept_vectors[c].float().cpu().numpy()
        vecs[c] = v / (np.linalg.norm(v) + 1e-8)

    # Extract activations for every prompt in every pair
    per_pair = []
    for pair in pairs:
        t0 = time.time()
        neutral_act = extract_one(model, pair["neutral"], best_layer).numpy()
        dangerous_act = extract_one(model, pair["dangerous"], best_layer).numpy()
        elapsed = time.time() - t0

        per_concept = {}
        for c in available:
            v = vecs[c]
            neu_proj = float(np.dot(neutral_act, v))
            dan_proj = float(np.dot(dangerous_act, v))
            delta = dan_proj - neu_proj
            per_concept[c] = {
                "neutral_projection": neu_proj,
                "dangerous_projection": dan_proj,
                "delta": delta,
            }

        per_pair.append({
            "id": pair["id"],
            "shared_number": pair.get("shared_number"),
            "category": pair.get("category"),
            "neutral_text": pair["neutral"],
            "dangerous_text": pair["dangerous"],
            "per_concept": per_concept,
            "extract_time_s": elapsed,
        })

    # Aggregate per-concept across pairs
    agg = {}
    for c in available:
        deltas = np.array([pp["per_concept"][c]["delta"] for pp in per_pair])
        n_positive = int(np.sum(deltas > 0))
        n_total = len(deltas)
        ttest = paired_test(deltas)
        agg[c] = {
            "mean_delta": float(np.mean(deltas)),
            "std_delta": float(np.std(deltas)),
            "median_delta": float(np.median(deltas)),
            "n_positive": n_positive,
            "n_total": n_total,
            "frac_positive": n_positive / n_total if n_total else 0.0,
            "t_test": ttest,
        }
        sig_marker = " ***" if ttest["p"] < 0.01 else (" *" if ttest["p"] < 0.05 else "")
        sign = "+" if agg[c]["mean_delta"] >= 0 else ""
        logger.info(
            "  %s: mean delta = %s%.4f, %d/%d positive, t=%.2f, p=%.4f%s",
            c, sign, agg[c]["mean_delta"], n_positive, n_total,
            ttest["t"], ttest["p"], sig_marker,
        )

    return {
        "model_key": model_key,
        "best_layer": best_layer,
        "n_pairs": len(per_pair),
        "concepts_tested": available,
        "per_pair": per_pair,
        "aggregate": agg,
    }


def load_pairs(data_root: Path) -> list[dict]:
    path = data_root / "data" / "emotions" / "severity_pairs.json"
    with open(path) as f:
        return json.load(f)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--data-root", default=os.environ.get("MECHINTERP_DATA_ROOT", "drive_data"))
    parser.add_argument("--device", default=None, help="Override device (default: auto)")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    pairs = load_pairs(data_root)
    logger.info("Loaded %d severity pairs", len(pairs))

    registry = ModelRegistry()
    all_results = {}
    overall_t0 = time.time()

    for model_key in args.models:
        info = registry.get(model_key)
        t0 = time.time()
        try:
            model, tokenizer = load_model(info, device=args.device)
        except Exception as e:
            logger.error("Failed to load %s: %s", model_key, e)
            continue
        logger.info("Loaded %s in %.1fs", model_key, time.time() - t0)

        try:
            result = run_one_model(model, model_key, data_root, pairs)
            all_results[model_key] = result

            # Save per-model
            out_path = data_root / "results" / "emotions" / model_key / "severity_pairs.json"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path.with_suffix(".tmp"), "w") as f:
                json.dump(result, f, indent=2, default=str)
            out_path.with_suffix(".tmp").rename(out_path)
            logger.info("Saved %s", out_path)
        except Exception as e:
            logger.exception("Failed for %s: %s", model_key, e)
            all_results[model_key] = {"error": str(e)}

        unload_model(model)

    # Combined save
    combined_path = data_root / "results" / "emotions" / "severity_pairs_combined.json"
    with open(combined_path.with_suffix(".tmp"), "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    combined_path.with_suffix(".tmp").rename(combined_path)

    elapsed = time.time() - overall_t0
    logger.info("\n=== ALL DONE in %.1fs ===", elapsed)

    # Print compact summary
    print()
    print("=" * 78)
    print("SEVERITY PAIRS — fear vector delta (dangerous − neutral, paired t-test)")
    print("=" * 78)
    print(f"  {'Model':14s}  {'fear delta':>12s}  {'frac+':>7s}  {'t':>7s}  {'p':>9s}")
    print("-" * 78)
    for m, r in all_results.items():
        if "error" in r:
            print(f"  {m:14s}  ERROR: {r['error']}")
            continue
        agg = r["aggregate"].get("afraid", {})
        d = agg.get("mean_delta", 0)
        fp = agg.get("frac_positive", 0)
        tt = agg.get("t_test", {})
        t = tt.get("t", 0)
        p = tt.get("p", 1.0)
        sig = " ***" if p < 0.01 else (" *" if p < 0.05 else "")
        sign = "+" if d >= 0 else ""
        print(f"  {m:14s}  {sign}{d:>11.4f}  {fp:>7.2f}  {t:>7.2f}  {p:>9.4f}{sig}")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
