"""Linear probes for classifying concepts from activations.

Supports logistic regression (sklearn) and simple MLP probes (torch).
k-fold cross-validation with per-fold checkpointing.

Usage:
    from src.techniques.probes import train_probe, ProbeResult

    result = train_probe(activations, labels, probe_type="logistic_regression")
    print(result.mean_accuracy, result.per_concept_accuracy)

    # With per-fold checkpointing (resume-safe):
    result = train_probe(
        activations, labels,
        probe_type="mlp",
        checkpoint_dir=Path("checkpoints/probe_fold"),
    )
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch import Tensor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class FoldResult:
    """Result of a single cross-validation fold."""

    fold: int
    accuracy: float
    train_size: int
    val_size: int
    val_predictions: list[int] = field(default_factory=list)
    val_labels: list[int] = field(default_factory=list)


@dataclass
class ProbeResult:
    """Aggregated result of probe training with k-fold cross-validation.

    Attributes:
        mean_accuracy: Mean accuracy across all folds.
        std_accuracy: Standard deviation of accuracy across folds.
        per_concept_accuracy: Per-concept accuracy (concept label -> accuracy).
        fold_results: Detailed per-fold results.
        confusion_matrix: (n_concepts, n_concepts) confusion matrix as nested list.
        probe_type: Type of probe used ("logistic_regression" or "mlp").
        n_folds: Number of CV folds.
        label_names: Ordered list of concept names corresponding to matrix indices.
    """

    mean_accuracy: float
    std_accuracy: float
    per_concept_accuracy: dict[str, float]
    fold_results: list[FoldResult]
    confusion_matrix: list[list[int]]
    probe_type: str
    n_folds: int
    label_names: list[str] = field(default_factory=list)

    def save(self, path: Path) -> None:
        """Save probe result to JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> ProbeResult:
        """Load probe result from JSON."""
        with open(path) as f:
            data = json.load(f)
        data["fold_results"] = [FoldResult(**fr) for fr in data["fold_results"]]
        return cls(**data)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def train_probe(
    activations: Tensor,
    labels: list[str] | list[int] | np.ndarray,
    probe_type: Literal["logistic_regression", "mlp"] = "logistic_regression",
    n_folds: int = 5,
    seed: int = 42,
    regularization: float = 1.0,
    checkpoint_dir: Path | None = None,
    mlp_hidden_dim: int = 128,
    mlp_epochs: int = 100,
    mlp_lr: float = 1e-3,
) -> ProbeResult:
    """Train a classification probe with stratified k-fold cross-validation.

    Args:
        activations: Tensor of shape (n_samples, hidden_dim).
        labels: Per-sample labels -- strings (concept names) or ints.
        probe_type: "logistic_regression" (sklearn, L2 regularized) or
            "mlp" (2-layer torch MLP).
        n_folds: Number of stratified CV folds.
        seed: Random seed for reproducibility.
        regularization: Inverse regularization strength for logistic regression
            (sklearn's C parameter).  Ignored for MLP.
        checkpoint_dir: If provided, save/load per-fold results here for
            resume-safety.  Each fold is stored as ``fold_{i}.json``.
        mlp_hidden_dim: Hidden dimension for the MLP probe.
        mlp_epochs: Training epochs for MLP.
        mlp_lr: Learning rate for MLP.

    Returns:
        ProbeResult with aggregated metrics and confusion matrix.

    Raises:
        ValueError: If activations and labels have mismatched lengths,
            or if probe_type is unrecognized.
    """
    from sklearn.model_selection import StratifiedKFold

    # -- Validate inputs ---------------------------------------------------
    if activations.shape[0] != len(labels):
        raise ValueError(
            f"Activation count ({activations.shape[0]}) != label count ({len(labels)})"
        )
    if probe_type not in ("logistic_regression", "mlp"):
        raise ValueError(f"Unknown probe_type: {probe_type!r}")

    # -- Encode labels to ints if strings ----------------------------------
    label_list: list[str]
    if isinstance(labels, np.ndarray):
        labels = labels.tolist()
    labels = list(labels)

    if isinstance(labels[0], str):
        unique_labels = sorted(set(labels))
        label_to_idx = {lab: i for i, lab in enumerate(unique_labels)}
        encoded = np.array([label_to_idx[lab] for lab in labels])
        label_list = unique_labels
    else:
        encoded = np.array(labels, dtype=int)
        label_list = [str(i) for i in sorted(set(encoded.tolist()))]

    n_classes = len(label_list)
    X = activations.detach().cpu().float().numpy()
    y = encoded

    # -- Stratified K-Fold -------------------------------------------------
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_results: list[FoldResult] = []
    all_val_preds: list[int] = []
    all_val_labels: list[int] = []

    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        # -- Check for cached fold -----------------------------------------
        if checkpoint_dir is not None:
            fold_path = checkpoint_dir / f"fold_{fold_idx}.json"
            if fold_path.exists():
                with open(fold_path) as f:
                    fr_data = json.load(f)
                fr = FoldResult(**fr_data)
                fold_results.append(fr)
                all_val_preds.extend(fr.val_predictions)
                all_val_labels.extend(fr.val_labels)
                logger.info("Fold %d/%d: loaded from checkpoint", fold_idx + 1, n_folds)
                continue

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # -- Train -----------------------------------------------------------
        if probe_type == "logistic_regression":
            preds = _train_logistic(X_train, y_train, X_val, regularization, seed)
        else:
            preds = _train_mlp(
                X_train, y_train, X_val, n_classes,
                hidden_dim=mlp_hidden_dim, epochs=mlp_epochs, lr=mlp_lr, seed=seed,
            )

        acc = float(np.mean(preds == y_val))
        fr = FoldResult(
            fold=fold_idx,
            accuracy=acc,
            train_size=len(train_idx),
            val_size=len(val_idx),
            val_predictions=preds.tolist(),
            val_labels=y_val.tolist(),
        )
        fold_results.append(fr)
        all_val_preds.extend(preds.tolist())
        all_val_labels.extend(y_val.tolist())

        logger.info("Fold %d/%d: accuracy=%.4f", fold_idx + 1, n_folds, acc)

        # -- Save fold checkpoint -------------------------------------------
        if checkpoint_dir is not None:
            fold_path = checkpoint_dir / f"fold_{fold_idx}.json"
            tmp_path = fold_path.with_suffix(".tmp")
            with open(tmp_path, "w") as f:
                json.dump(asdict(fr), f)
            tmp_path.rename(fold_path)

    # -- Aggregate ---------------------------------------------------------
    accuracies = [fr.accuracy for fr in fold_results]
    mean_accuracy = float(np.mean(accuracies))
    std_accuracy = float(np.std(accuracies, ddof=1)) if len(accuracies) > 1 else 0.0

    # Confusion matrix
    cm = _build_confusion_matrix(all_val_labels, all_val_preds, n_classes)

    # Per-concept accuracy
    per_concept_accuracy = _per_concept_accuracy(all_val_labels, all_val_preds, label_list)

    return ProbeResult(
        mean_accuracy=mean_accuracy,
        std_accuracy=std_accuracy,
        per_concept_accuracy=per_concept_accuracy,
        fold_results=fold_results,
        confusion_matrix=cm.tolist(),
        probe_type=probe_type,
        n_folds=n_folds,
        label_names=label_list,
    )


# ---------------------------------------------------------------------------
# Logistic Regression
# ---------------------------------------------------------------------------

def _train_logistic(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    C: float,
    seed: int,
) -> np.ndarray:
    """Train sklearn LogisticRegression with L2 penalty and predict on val."""
    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression(
        C=C,
        l1_ratio=0,  # equivalent to L2 penalty (penalty= deprecated in sklearn 1.8)
        solver="lbfgs",
        max_iter=2000,
        random_state=seed,
    )
    clf.fit(X_train, y_train)
    return clf.predict(X_val)


# ---------------------------------------------------------------------------
# MLP Probe
# ---------------------------------------------------------------------------

class _MLPProbe(torch.nn.Module):
    """Simple 2-layer MLP probe for classification."""

    def __init__(self, input_dim: int, hidden_dim: int, n_classes: int) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


def _train_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    n_classes: int,
    hidden_dim: int = 128,
    epochs: int = 100,
    lr: float = 1e-3,
    seed: int = 42,
) -> np.ndarray:
    """Train a 2-layer MLP probe and predict on validation set.

    Training runs on CPU to avoid MPS compatibility issues with small probes.
    """
    torch.manual_seed(seed)

    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.long)
    X_v = torch.tensor(X_val, dtype=torch.float32)

    input_dim = X_t.shape[1]
    model = _MLPProbe(input_dim, hidden_dim, n_classes)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = torch.nn.CrossEntropyLoss()

    # Mini-batch training
    batch_size = min(256, len(X_t))
    n_batches = max(1, len(X_t) // batch_size)

    for epoch in range(epochs):
        # Shuffle each epoch
        perm = torch.randperm(len(X_t))
        X_t_shuffled = X_t[perm]
        y_t_shuffled = y_t[perm]

        epoch_loss = 0.0
        for b in range(n_batches):
            start = b * batch_size
            end = min(start + batch_size, len(X_t))
            X_batch = X_t_shuffled[start:end]
            y_batch = y_t_shuffled[start:end]

            optimizer.zero_grad()
            logits = model(X_batch)
            loss = loss_fn(logits, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

    model.eval()
    with torch.no_grad():
        logits = model(X_v)
        preds = logits.argmax(dim=1).numpy()

    return preds


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_confusion_matrix(
    true_labels: list[int],
    pred_labels: list[int],
    n_classes: int,
) -> np.ndarray:
    """Build a confusion matrix: cm[true][pred] = count."""
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(true_labels, pred_labels):
        cm[t, p] += 1
    return cm


def _per_concept_accuracy(
    true_labels: list[int],
    pred_labels: list[int],
    label_names: list[str],
) -> dict[str, float]:
    """Compute accuracy for each concept class."""
    true_arr = np.array(true_labels)
    pred_arr = np.array(pred_labels)
    result: dict[str, float] = {}

    for idx, name in enumerate(label_names):
        mask = true_arr == idx
        if mask.sum() == 0:
            result[name] = 0.0
        else:
            result[name] = float(np.mean(pred_arr[mask] == idx))

    return result
