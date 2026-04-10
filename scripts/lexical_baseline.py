"""Lexical baseline for the emotion probe classification task.

This script trains a *text-only* classifier (TF-IDF + logistic regression)
on the same training stimuli the activation probes use, with the same
stratified k-fold cross-validation. If a text-only baseline achieves
similar accuracy to the activation-based probes, then the activation
probe results are NOT primarily about internal representations — they
may largely reflect surface lexical features that survived the
"don't use the emotion word" generation prompt.

This is a critical validity check raised by external critique:
> If "afraid" stories consistently mention "heart pounding" or "dark alley"
> while "happy" stories mention "sunshine," the probe may be learning
> surface-level lexical features, not abstract emotion representations.

Usage:
    python scripts/lexical_baseline.py --paper emotions

Reads stimuli from drive_data/data/{paper_id}/training/{concept}.json
and writes results to drive_data/results/{paper_id}/lexical_baseline.json
plus a per-concept breakdown.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_emotions_stimuli(data_root: Path, paper_id: str = "emotions") -> tuple[list[str], list[str]]:
    """Load training stimuli for the emotions paper.

    Returns:
        (texts, labels) where texts[i] has concept labels[i].
    """
    training_dir = data_root / "data" / paper_id / "training"
    if not training_dir.exists():
        raise FileNotFoundError(f"Training stimuli not found at {training_dir}")

    texts: list[str] = []
    labels: list[str] = []

    json_files = sorted(training_dir.glob("*.json"))
    # Skip combined files like all_stimuli.json
    concept_files = [f for f in json_files if f.stem != "all_stimuli"]

    for f in concept_files:
        concept = f.stem
        with open(f) as fh:
            items = json.load(fh)
        for item in items:
            texts.append(item["text"])
            labels.append(concept)

    return texts, labels


def run_text_baseline(
    texts: list[str],
    labels: list[str],
    n_folds: int = 5,
    seed: int = 42,
    feature_type: str = "word_tfidf",
) -> dict:
    """Train a text-only classifier with k-fold CV.

    feature_type:
      - "word_tfidf": TF-IDF over word unigrams + bigrams
      - "char_tfidf": TF-IDF over character 3-5 grams
      - "bow": plain bag-of-words counts
    """
    le = LabelEncoder()
    y = le.fit_transform(labels)
    classes = le.classes_

    if feature_type == "word_tfidf":
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95,
            lowercase=True,
            stop_words=None,  # KEY: do NOT remove stop words; we want to see what's there
        )
    elif feature_type == "char_tfidf":
        vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=2,
            max_df=0.95,
            lowercase=True,
        )
    elif feature_type == "bow":
        from sklearn.feature_extraction.text import CountVectorizer
        vectorizer = CountVectorizer(
            ngram_range=(1, 1),
            min_df=1,
            lowercase=True,
        )
    else:
        raise ValueError(f"Unknown feature_type: {feature_type}")

    pipe = Pipeline([
        ("vec", vectorizer),
        ("clf", LogisticRegression(max_iter=2000, solver="lbfgs", random_state=seed)),
    ])

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_accuracies: list[float] = []
    all_y_true: list[int] = []
    all_y_pred: list[int] = []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(np.arange(len(texts)), y)):
        train_texts = [texts[i] for i in train_idx]
        test_texts = [texts[i] for i in test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]

        pipe.fit(train_texts, y_train)
        y_pred = pipe.predict(test_texts)

        acc = accuracy_score(y_test, y_pred)
        fold_accuracies.append(float(acc))
        all_y_true.extend(y_test.tolist())
        all_y_pred.extend(y_pred.tolist())

        logger.info("  fold %d: %s acc = %.4f", fold_idx, feature_type, acc)

    mean_acc = float(np.mean(fold_accuracies))
    std_acc = float(np.std(fold_accuracies))

    cm = confusion_matrix(all_y_true, all_y_pred, labels=list(range(len(classes))))
    per_concept = {}
    for i, c in enumerate(classes):
        row_total = cm[i].sum()
        per_concept[str(c)] = float(cm[i, i] / row_total) if row_total > 0 else 0.0

    return {
        "feature_type": feature_type,
        "mean_accuracy": mean_acc,
        "std_accuracy": std_acc,
        "fold_accuracies": fold_accuracies,
        "n_stimuli": len(texts),
        "n_concepts": int(len(classes)),
        "concepts": list(map(str, classes)),
        "per_concept_accuracy": per_concept,
        "confusion_matrix": cm.tolist(),
        "n_folds": n_folds,
        "seed": seed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper", default="emotions")
    parser.add_argument("--data-root", default="drive_data")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    texts, labels = load_emotions_stimuli(data_root, args.paper)
    logger.info("Loaded %d stimuli across %d concepts", len(texts), len(set(labels)))

    out_dir = data_root / "results" / args.paper
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}
    for feature_type in ["word_tfidf", "char_tfidf", "bow"]:
        logger.info("=== Running %s baseline ===", feature_type)
        result = run_text_baseline(
            texts, labels,
            n_folds=args.n_folds,
            seed=args.seed,
            feature_type=feature_type,
        )
        logger.info(
            "%s: mean acc = %.4f ± %.4f (n_stimuli=%d, n_concepts=%d)",
            feature_type,
            result["mean_accuracy"],
            result["std_accuracy"],
            result["n_stimuli"],
            result["n_concepts"],
        )
        all_results[feature_type] = result

    out_path = out_dir / "lexical_baseline.json"
    tmp = out_path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(all_results, f, indent=2)
    tmp.rename(out_path)
    logger.info("Saved %s", out_path)

    # Print summary
    print("\n" + "=" * 60)
    print("LEXICAL BASELINE SUMMARY (k-fold CV on training stimuli)")
    print("=" * 60)
    for ft, r in all_results.items():
        print(f"  {ft:14s}: {r['mean_accuracy']:.4f} ± {r['std_accuracy']:.4f}")
    print()
    print("Compare to activation probe accuracies (per-model):")
    print("  Llama-3.2-1B :  0.773")
    print("  Qwen2.5-1.5B :  0.731")
    print("  Gemma-2-2B   :  0.776")
    print("  Llama-3.1-8B :  0.819")
    print("  Qwen2.5-7B   :  0.784")
    print("  Gemma-2-9B   :  0.840")
    print()
    print("If text-only baseline approaches activation probe accuracy, the")
    print("activation probe is partly capturing surface lexical features")
    print("rather than (or in addition to) abstract emotion representations.")


if __name__ == "__main__":
    main()
