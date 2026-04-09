"""Visualization: publication-ready plots for cross-model results.

Generates plots to figures/{paper_id}/:
  - Comparison heatmaps (metric values across models and claims)
  - Scaling curves (metric vs. model size per family)
  - PCA scatter plots (concept vectors in PC1-PC2 space)
  - Confusion matrices
  - Steering effect bar charts

All plots are saved as both PNG (300 dpi) and PDF.

Usage:
    from src.analysis.visualize import plot_comparison_heatmap, plot_scaling_curves

    plot_comparison_heatmap(cross_model_report, output_dir)
    plot_scaling_curves(scaling_report, output_dir)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src.analysis.cross_model import CrossModelReport
from src.analysis.scaling import ScalingReport

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Style configuration
# ---------------------------------------------------------------------------

# Use non-interactive backend when saving to files (safe on headless machines)
matplotlib.use("Agg")

_STYLE_APPLIED = False


def _apply_style() -> None:
    """Set publication-ready matplotlib/seaborn style once."""
    global _STYLE_APPLIED
    if _STYLE_APPLIED:
        return

    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        # Fallback for older matplotlib versions
        try:
            plt.style.use("seaborn-whitegrid")
        except OSError:
            pass

    sns.set_context("paper", font_scale=1.2)
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.family": "sans-serif",
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 16,
    })
    _STYLE_APPLIED = True


def _save_fig(fig: matplotlib.figure.Figure, output_dir: Path, name: str) -> None:
    """Save figure as PNG and PDF, then close it.

    Args:
        fig: Matplotlib figure.
        output_dir: Target directory (created if needed).
        name: Base filename without extension.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        path = output_dir / f"{name}.{ext}"
        fig.savefig(path)
        logger.info("Saved: %s", path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Family color palette (consistent across all plots)
# ---------------------------------------------------------------------------

FAMILY_COLORS: dict[str, str] = {
    "llama": "#1f77b4",   # blue
    "qwen": "#ff7f0e",    # orange
    "gemma": "#2ca02c",   # green
}

FAMILY_MARKERS: dict[str, str] = {
    "llama": "o",
    "qwen": "s",
    "gemma": "^",
}


def _family_color(family: str) -> str:
    return FAMILY_COLORS.get(family, "#7f7f7f")


def _family_marker(family: str) -> str:
    return FAMILY_MARKERS.get(family, "D")


# ---------------------------------------------------------------------------
# Plot functions
# ---------------------------------------------------------------------------

def plot_comparison_heatmap(
    report: CrossModelReport,
    output_dir: Path,
    *,
    title: str | None = None,
) -> None:
    """Heatmap of metric values: rows=models, columns=claims.

    Cells are color-coded by value. Models are grouped by family.
    Cells meeting the success threshold get a check mark overlay.

    Args:
        report: CrossModelReport from compare_across_models().
        output_dir: Directory for saved figures.
        title: Optional custom title.
    """
    _apply_style()

    df = report.summary_table.copy()
    if df.empty:
        logger.warning("Empty summary table -- skipping heatmap.")
        return

    # Sort rows by family then size
    def _sort_key(model_key: str) -> tuple[str, float]:
        meta = report.model_metadata.get(model_key, {})
        return (meta.get("family", "z"), meta.get("params_b", 0.0))

    sorted_models = sorted(df.index, key=_sort_key)
    df = df.loc[sorted_models]

    # Prepare annotation text (value + PASS/FAIL)
    from src.core.config_loader import load_paper_config
    paper_config = load_paper_config(report.paper_id)
    thresholds = {c.claim_id: c.success_threshold for c in paper_config.claims}

    annot = df.copy().astype(str)
    for cid in df.columns:
        thresh = thresholds.get(cid, 0.0)
        for mk in df.index:
            val = df.at[mk, cid]
            if val is None or (isinstance(val, float) and np.isnan(val)):
                annot.at[mk, cid] = "--"
            else:
                tag = "+" if val >= thresh else "-"
                annot.at[mk, cid] = f"{val:.2f}{tag}"

    # Replace None/NaN with 0 for coloring (masked later)
    df_numeric = df.fillna(0.0).astype(float)

    fig_height = max(3, 0.6 * len(sorted_models) + 1.5)
    fig_width = max(5, 1.2 * len(df.columns) + 2.5)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # Mask cells with no data
    mask = df.isnull()

    sns.heatmap(
        df_numeric,
        annot=annot.values,
        fmt="",
        mask=mask.values,
        cmap="RdYlGn",
        vmin=0.0,
        vmax=1.0,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Metric value", "shrink": 0.8},
        ax=ax,
    )

    # Add family labels on y-axis
    ylabels = []
    for mk in sorted_models:
        meta = report.model_metadata.get(mk, {})
        fam = meta.get("family", "?")
        params = meta.get("params_b", "?")
        ylabels.append(f"{mk} ({fam} {params}B)")
    ax.set_yticklabels(ylabels, rotation=0)

    ax.set_xlabel("Claim")
    ax.set_ylabel("Model")
    ax.set_title(title or f"Cross-Model Comparison: {report.paper_id}")

    _save_fig(fig, output_dir, f"comparison_heatmap_{report.paper_id}")


def plot_scaling_curves(
    scaling: ScalingReport,
    output_dir: Path,
    *,
    title: str | None = None,
) -> None:
    """Scaling curves: metric vs. model parameters per family.

    One subplot per claim. X-axis is log(params_b), Y-axis is metric value.
    Each family gets its own color and marker. Log-linear trend lines are shown.

    Args:
        scaling: ScalingReport from analyze_scaling().
        output_dir: Directory for saved figures.
        title: Optional suptitle override.
    """
    _apply_style()

    claim_ids = sorted(scaling.trends.keys())
    if not claim_ids:
        logger.warning("No scaling trends to plot.")
        return

    n_claims = len(claim_ids)
    n_cols = min(n_claims, 3)
    n_rows = (n_claims + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(5 * n_cols, 4 * n_rows),
        squeeze=False,
    )

    for idx, cid in enumerate(claim_ids):
        row, col = divmod(idx, n_cols)
        ax = axes[row][col]
        family_trends = scaling.trends[cid]

        for fam in sorted(family_trends.keys()):
            trend = family_trends[fam]
            if not trend.points:
                continue

            xs = [p[0] for p in trend.points]
            ys = [p[1] for p in trend.points]

            ax.scatter(
                xs, ys,
                color=_family_color(fam),
                marker=_family_marker(fam),
                s=80,
                label=f"{fam} (r={trend.spearman_r:.2f})",
                zorder=3,
            )

            # Plot log-linear trend line
            if trend.log_linear_slope is not None and len(xs) >= 2:
                x_range = np.linspace(min(xs) * 0.8, max(xs) * 1.2, 50)
                y_fit = trend.log_linear_slope * np.log(x_range) + trend.log_linear_intercept
                ax.plot(
                    x_range, y_fit,
                    color=_family_color(fam),
                    linestyle="--",
                    alpha=0.5,
                    linewidth=1.5,
                )

        ax.set_xscale("log")
        ax.set_xlabel("Parameters (B)")
        ax.set_ylabel("Metric")
        ax.set_title(cid)
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(n_claims, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row][col].set_visible(False)

    fig.suptitle(title or f"Scaling Analysis: {scaling.paper_id}", y=1.02)
    fig.tight_layout()
    _save_fig(fig, output_dir, f"scaling_curves_{scaling.paper_id}")


def plot_pca_scatter(
    pca_coords: dict[str, tuple[float, float]],
    valence: dict[str, float],
    output_dir: Path,
    *,
    title: str = "Concept Vectors in PC1-PC2 Space",
) -> None:
    """Scatter plot of concept vectors projected into PC1-PC2 space.

    Each point is a concept (e.g., an emotion), colored by a continuous
    valence/attribute value.

    Args:
        pca_coords: concept_name -> (pc1, pc2) coordinates.
        valence: concept_name -> scalar attribute value for coloring
            (e.g., valence from -1 to +1).
        output_dir: Directory for saved figures.
        title: Plot title.
    """
    _apply_style()

    if not pca_coords:
        logger.warning("No PCA coordinates to plot.")
        return

    concepts = sorted(pca_coords.keys())
    xs = [pca_coords[c][0] for c in concepts]
    ys = [pca_coords[c][1] for c in concepts]
    colors = [valence.get(c, 0.0) for c in concepts]

    fig, ax = plt.subplots(figsize=(8, 6))

    scatter = ax.scatter(
        xs, ys,
        c=colors,
        cmap="RdBu_r",
        s=120,
        edgecolors="black",
        linewidths=0.5,
        zorder=3,
    )

    # Label each point
    for i, concept in enumerate(concepts):
        ax.annotate(
            concept,
            (xs[i], ys[i]),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=9,
            ha="left",
        )

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Valence")

    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--", alpha=0.5)
    ax.axvline(0, color="gray", linewidth=0.5, linestyle="--", alpha=0.5)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    _save_fig(fig, output_dir, "pca_scatter")


def plot_confusion_matrix(
    cm: np.ndarray,
    labels: list[str],
    title: str,
    output_dir: Path,
    *,
    filename: str = "confusion_matrix",
    normalize: bool = True,
) -> None:
    """Plot a labeled confusion matrix as a heatmap.

    Args:
        cm: Square confusion matrix of shape (n, n). Raw counts or
            pre-normalized values.
        labels: Class labels, length n.
        title: Plot title.
        output_dir: Directory for saved figures.
        filename: Base filename (without extension).
        normalize: If True, normalize rows to sum to 1.
    """
    _apply_style()

    cm_plot = cm.astype(float).copy()
    fmt = ".2f"
    if normalize:
        row_sums = cm_plot.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1, row_sums)  # avoid div-by-zero
        cm_plot = cm_plot / row_sums
    else:
        fmt = "d"

    n = len(labels)
    fig_size = max(5, 0.5 * n + 2)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))

    sns.heatmap(
        cm_plot,
        annot=True,
        fmt=fmt,
        xticklabels=labels,
        yticklabels=labels,
        cmap="Blues",
        vmin=0.0,
        vmax=1.0 if normalize else None,
        linewidths=0.5,
        linecolor="white",
        square=True,
        cbar_kws={"shrink": 0.8},
        ax=ax,
    )

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    # Rotate labels for readability
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    _save_fig(fig, output_dir, filename)


def plot_steering_effects(
    effects: dict[str, float],
    output_dir: Path,
    *,
    title: str = "Steering Effects by Concept",
    filename: str = "steering_effects",
    error_bars: dict[str, tuple[float, float]] | None = None,
) -> None:
    """Bar chart of steering effect sizes per concept.

    Args:
        effects: concept_name -> effect size (e.g., Cohen's d or
            preference shift magnitude).
        output_dir: Directory for saved figures.
        title: Plot title.
        filename: Base filename (without extension).
        error_bars: Optional concept_name -> (lower_ci, upper_ci) for
            95% confidence interval error bars.
    """
    _apply_style()

    if not effects:
        logger.warning("No steering effects to plot.")
        return

    # Sort by effect size descending
    concepts = sorted(effects.keys(), key=lambda c: effects[c], reverse=True)
    values = [effects[c] for c in concepts]

    fig, ax = plt.subplots(figsize=(max(6, 0.6 * len(concepts) + 2), 5))

    colors = ["#2ca02c" if v > 0 else "#d62728" for v in values]

    bars = ax.bar(
        range(len(concepts)),
        values,
        color=colors,
        edgecolor="black",
        linewidth=0.5,
        alpha=0.85,
    )

    # Error bars
    if error_bars is not None:
        for i, concept in enumerate(concepts):
            if concept in error_bars:
                lo, hi = error_bars[concept]
                ax.errorbar(
                    i, values[i],
                    yerr=[[values[i] - lo], [hi - values[i]]],
                    fmt="none",
                    color="black",
                    capsize=3,
                    linewidth=1.5,
                )

    ax.set_xticks(range(len(concepts)))
    ax.set_xticklabels(concepts, rotation=45, ha="right")
    ax.set_ylabel("Effect Size")
    ax.set_title(title)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="-")
    ax.grid(True, axis="y", alpha=0.3)

    _save_fig(fig, output_dir, filename)


def plot_layer_sweep(
    layer_metrics: dict[str, list[float]],
    output_dir: Path,
    *,
    title: str = "Metric by Layer",
    ylabel: str = "Accuracy",
    filename: str = "layer_sweep",
) -> None:
    """Line plot of a metric value across layers for multiple models.

    Useful for showing where in the network a representation emerges.

    Args:
        layer_metrics: model_key -> list of metric values (one per layer).
        output_dir: Directory for saved figures.
        title: Plot title.
        ylabel: Y-axis label.
        filename: Base filename (without extension).
    """
    _apply_style()

    if not layer_metrics:
        logger.warning("No layer metrics to plot.")
        return

    fig, ax = plt.subplots(figsize=(10, 5))

    for mk, values in sorted(layer_metrics.items()):
        layers = list(range(len(values)))
        ax.plot(layers, values, marker=".", markersize=4, label=mk, linewidth=1.5)

    ax.set_xlabel("Layer")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)

    _save_fig(fig, output_dir, filename)
