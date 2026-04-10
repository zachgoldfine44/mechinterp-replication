#!/usr/bin/env python3
"""Generate publication-quality figures for the emotions replication paper.

Reads JSON data files from drive_data/results/emotions/ and produces
4 figures in figures/emotions/.

Usage:
    python scripts/generate_paper_figures.py
"""

import json
import sys
from pathlib import Path
import numpy as np

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
import seaborn as sns

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO = Path(__file__).resolve().parent.parent
DATA_DIR = REPO / "drive_data" / "results" / "emotions"
FIG_DIR = REPO / "figures" / "emotions"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "font.family": "serif",
    "axes.grid": False,
    "axes.facecolor": "#f7f7f7",
    "figure.facecolor": "white",
})

# ---------------------------------------------------------------------------
# Model ordering (by param count)
# ---------------------------------------------------------------------------
MODEL_ORDER = ["qwen_1_5b", "llama_1b", "gemma_2b", "qwen_7b", "llama_8b", "gemma_9b"]
MODEL_LABELS = {
    "qwen_1_5b": "Qwen 1.5B",
    "llama_1b": "Llama 1B",
    "gemma_2b": "Gemma 2B",
    "qwen_7b": "Qwen 7B",
    "llama_8b": "Llama 8B",
    "gemma_9b": "Gemma 9B",
}
MODEL_PARAMS = {
    "qwen_1_5b": 1.5, "llama_1b": 1.0, "gemma_2b": 2.0,
    "qwen_7b": 7.0, "llama_8b": 8.0, "gemma_9b": 9.0,
}
FAMILY_COLORS = {
    "llama": "#4878CF",   # blue
    "qwen": "#EE854A",    # orange
    "gemma": "#6ACC64",   # green
}


def family_of(model_key: str) -> str:
    if model_key.startswith("llama"):
        return "llama"
    if model_key.startswith("qwen"):
        return "qwen"
    return "gemma"


def load_json(name: str) -> dict:
    p = DATA_DIR / name
    if not p.exists():
        print(f"WARNING: {p} not found, using fallback data")
        return {}
    with open(p) as f:
        return json.load(f)


def save_fig(fig: plt.Figure, stem: str) -> None:
    for ext in ("png", "pdf"):
        out = FIG_DIR / f"{stem}.{ext}"
        fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved {FIG_DIR / stem}.png  and  .pdf")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 1: Probe accuracy vs lexical baseline
# ═══════════════════════════════════════════════════════════════════════════
def fig1_probe_vs_baseline():
    print("Figure 1: Probe accuracy vs lexical baseline ...")

    # Hard-coded values from the user spec (these match cross_model_report)
    probe_acc = {
        "llama_1b": 0.773, "llama_8b": 0.819,
        "qwen_1_5b": 0.731, "qwen_7b": 0.784,
        "gemma_2b": 0.776, "gemma_9b": 0.840,
    }
    probe_std = {
        "llama_1b": 0.008, "llama_8b": 0.004,
        "qwen_1_5b": 0.012, "qwen_7b": 0.007,
        "gemma_2b": 0.011, "gemma_9b": 0.004,
    }
    best_lexical = 0.400  # bag-of-words baseline (same for all models)
    chance = 1.0 / 15.0   # 0.0667
    original_paper = 0.713  # Claude Sonnet 4.5

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(MODEL_ORDER))
    bar_w = 0.35

    # Probe bars (green)
    probe_vals = [probe_acc[m] for m in MODEL_ORDER]
    probe_errs = [probe_std[m] for m in MODEL_ORDER]
    ax.bar(x - bar_w / 2, probe_vals, bar_w, color="#6ACC64", edgecolor="white",
           linewidth=0.5, label="Residual-stream probe", yerr=probe_errs,
           capsize=3, error_kw={"linewidth": 1.0, "color": "#333333"})

    # Lexical baseline bars (gray)
    ax.bar(x + bar_w / 2, [best_lexical] * len(MODEL_ORDER), bar_w,
           color="#AAAAAA", edgecolor="white", linewidth=0.5,
           label="Best lexical baseline (BoW)")

    # Reference lines
    ax.axhline(chance, color="red", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.text(len(MODEL_ORDER) - 0.5, chance + 0.015, "Chance (1/15)",
            color="red", fontsize=8, ha="right", va="bottom")

    ax.axhline(original_paper, color="#4878CF", linestyle=":", linewidth=1.2, alpha=0.8)
    ax.text(len(MODEL_ORDER) - 0.5, original_paper + 0.015,
            "Original paper (Claude Sonnet 4.5)", color="#4878CF",
            fontsize=8, ha="right", va="bottom")

    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS[m] for m in MODEL_ORDER])
    ax.set_ylabel("15-way Classification Accuracy")
    ax.set_ylim(0, 1.0)
    ax.legend(loc="upper left", frameon=True, framealpha=0.9, edgecolor="#cccccc")

    # Clean up spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    save_fig(fig, "fig1_probe_vs_baseline")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 2: Geometry (valence correlation) + Severity pairs
# ═══════════════════════════════════════════════════════════════════════════
def fig2_geometry_and_severity():
    print("Figure 2: Geometry and severity pairs ...")

    valence_r = {
        "llama_1b": 0.666, "llama_8b": 0.738,
        "qwen_1_5b": 0.810, "qwen_7b": 0.828,
        "gemma_2b": 0.811, "gemma_9b": 0.790,
    }
    original_r = 0.81

    # Load severity pairs for Panel B
    sev_data = load_json("severity_pairs_combined.json")

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(12, 5))

    # ── Panel A: Valence correlation bars ──
    x = np.arange(len(MODEL_ORDER))
    colors = [FAMILY_COLORS[family_of(m)] for m in MODEL_ORDER]
    vals = [valence_r[m] for m in MODEL_ORDER]

    ax_a.bar(x, vals, color=colors, edgecolor="white", linewidth=0.5, width=0.65)
    ax_a.axhline(original_r, color="#888888", linestyle="--", linewidth=1.0, alpha=0.7)
    ax_a.text(len(MODEL_ORDER) - 0.5, original_r + 0.015, "Original paper (r = 0.81)",
              color="#555555", fontsize=8, ha="right", va="bottom")

    ax_a.set_xticks(x)
    ax_a.set_xticklabels([MODEL_LABELS[m] for m in MODEL_ORDER], rotation=25, ha="right")
    ax_a.set_ylabel("|r|  (PC1 vs human valence)")
    ax_a.set_ylim(0, 1.0)
    ax_a.spines["top"].set_visible(False)
    ax_a.spines["right"].set_visible(False)

    # Annotate values on bars
    for i, v in enumerate(vals):
        ax_a.text(i, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=8)

    # Family legend
    legend_patches = [Patch(facecolor=FAMILY_COLORS[f], label=f.capitalize())
                      for f in ("llama", "qwen", "gemma")]
    ax_a.legend(handles=legend_patches, loc="lower right", frameon=True,
                framealpha=0.9, edgecolor="#cccccc")
    ax_a.set_title("A", fontsize=13, fontweight="bold", loc="left", pad=8)

    # ── Panel B: Severity-pair dot plot for Llama 8B "afraid" ──
    if sev_data and "llama_8b" in sev_data:
        pairs = sev_data["llama_8b"]["per_pair"]
        concept = "afraid"

        # Extract projections for each pair
        pair_labels = []
        neutral_vals = []
        danger_vals = []
        for pair in pairs:
            label = pair["id"].replace("_", " ")
            # Shorten labels: take last two meaningful words
            parts = label.split()
            short = " ".join(parts[-2:]) if len(parts) > 2 else label
            pair_labels.append(short)
            neutral_vals.append(pair["per_concept"][concept]["neutral_projection"])
            danger_vals.append(pair["per_concept"][concept]["dangerous_projection"])

        y_pos = np.arange(len(pairs))

        for i in range(len(pairs)):
            delta = danger_vals[i] - neutral_vals[i]
            line_color = "#6ACC64" if delta > 0 else "#E15759"
            ax_b.plot([neutral_vals[i], danger_vals[i]], [y_pos[i], y_pos[i]],
                      color=line_color, linewidth=1.8, zorder=1)

        ax_b.scatter(neutral_vals, y_pos, marker="o", color="#4878CF", s=50,
                     zorder=2, label="Neutral")
        ax_b.scatter(danger_vals, y_pos, marker="^", color="#E15759", s=50,
                     zorder=2, label="Dangerous")

        ax_b.set_yticks(y_pos)
        ax_b.set_yticklabels(pair_labels, fontsize=8)
        ax_b.set_xlabel("Projection onto 'afraid' vector")
        ax_b.legend(loc="lower right", frameon=True, framealpha=0.9,
                    edgecolor="#cccccc", fontsize=8)
        ax_b.spines["top"].set_visible(False)
        ax_b.spines["right"].set_visible(False)

        # Count positive deltas
        n_positive = sum(1 for n, d in zip(neutral_vals, danger_vals) if d > n)
        ax_b.text(0.02, 0.98, f"{n_positive}/{len(pairs)} pairs shift toward 'afraid'",
                  transform=ax_b.transAxes, fontsize=8, va="top",
                  bbox=dict(boxstyle="round,pad=0.3", facecolor="#f0f0f0",
                            edgecolor="#cccccc"))
    else:
        ax_b.text(0.5, 0.5, "Severity data not available", ha="center",
                  va="center", transform=ax_b.transAxes)

    ax_b.set_title("B", fontsize=13, fontweight="bold", loc="left", pad=8)

    fig.tight_layout(w_pad=3)
    save_fig(fig, "fig2_geometry_and_severity")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 3: Steering null result
# ═══════════════════════════════════════════════════════════════════════════
def fig3_steering_null():
    print("Figure 3: Steering null result ...")

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(10, 5), sharey=True)

    # ── Left panel: Original paper ──
    categories = ["Baseline", "Steered\n(+desperate)"]
    original_vals = [22, 72]
    bar_colors_l = ["#FCBF74", "#C44E52"]  # light orange, dark red

    bars_l = ax_l.bar(categories, original_vals, color=bar_colors_l,
                      edgecolor="white", linewidth=0.5, width=0.55)
    for bar, val in zip(bars_l, original_vals):
        ax_l.text(bar.get_x() + bar.get_width() / 2, val + 1.5,
                  f"{val}%", ha="center", va="bottom", fontsize=11,
                  fontweight="bold")

    ax_l.set_ylabel("Unethical Response Rate (%)")
    ax_l.set_ylim(0, 85)
    ax_l.set_title("Original Paper\n(Claude Sonnet 4.5)", fontsize=10,
                   fontweight="bold", pad=10)
    ax_l.spines["top"].set_visible(False)
    ax_l.spines["right"].set_visible(False)

    # ── Right panel: This replication ──
    medium_models = ["Llama 8B", "Qwen 7B", "Gemma 9B"]
    x_r = np.arange(len(medium_models))
    bar_w = 0.35

    # All values are 0%
    baseline_vals = [0, 0, 0]
    steered_vals = [0, 0, 0]

    ax_r.bar(x_r - bar_w / 2, baseline_vals, bar_w, color="#FCBF74",
             edgecolor="#AAAAAA", linewidth=0.5, label="Baseline")
    ax_r.bar(x_r + bar_w / 2, steered_vals, bar_w, color="#C44E52",
             edgecolor="#AAAAAA", linewidth=0.5, label="Steered (+desperate)")

    # Add tiny markers at 0 so the chart isn't empty
    for i in range(len(medium_models)):
        ax_r.plot(x_r[i] - bar_w / 2, 0.3, marker="_", color="#FCBF74",
                  markersize=15, markeredgewidth=2)
        ax_r.plot(x_r[i] + bar_w / 2, 0.3, marker="_", color="#C44E52",
                  markersize=15, markeredgewidth=2)

    ax_r.set_xticks(x_r)
    ax_r.set_xticklabels(medium_models)
    ax_r.set_ylim(0, 85)
    ax_r.set_title("This Replication\n(Open-Source Models)", fontsize=10,
                   fontweight="bold", pad=10)
    ax_r.legend(loc="upper right", frameon=True, framealpha=0.9,
                edgecolor="#cccccc", fontsize=8)
    ax_r.spines["top"].set_visible(False)
    ax_r.spines["right"].set_visible(False)

    # Annotation spanning the right panel
    ax_r.text(0.5, 0.55,
              "0% unethical responses\nacross all 135 conditions\n"
              "(3 models x 5 concepts x\n3 alphas x 3 scenarios)",
              transform=ax_r.transAxes, ha="center", va="center",
              fontsize=9, color="#555555",
              bbox=dict(boxstyle="round,pad=0.5", facecolor="#fff3e0",
                        edgecolor="#FCBF74", alpha=0.9))

    fig.tight_layout(w_pad=3)
    save_fig(fig, "fig3_steering_null")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 4: Universality scorecard + scaling inset
# ═══════════════════════════════════════════════════════════════════════════
def fig4_universality_scorecard():
    print("Figure 4: Universality scorecard ...")

    report = load_json("cross_model_report.json")
    scaling = load_json("scaling_bootstrap_ci.json")

    claims_ids = [
        "emotion_probe_classification",
        "emotion_generalization",
        "representation_geometry_valence",
        "parametric_intensity",
        "causal_steering_behavior",
        "preference_steering",
    ]
    claim_labels = [
        "Probe\nClassification",
        "Generalization",
        "Valence\nGeometry",
        "Parametric\nIntensity",
        "Causal\nSteering",
        "Preference\nSteering",
    ]
    metric_keys = [
        "probe_accuracy",
        "diagonal_dominance",
        "valence_correlation",
        "rank_correlation",
        "causal_effect_count",
        "preference_correlation",
    ]
    thresholds = [0.50, 0.50, 0.50, 0.50, 3, 0.40]

    n_models = len(MODEL_ORDER)
    n_claims = len(claims_ids)

    # Build the data matrix
    data_matrix = np.zeros((n_models, n_claims))
    for j, (cid, mk) in enumerate(zip(claims_ids, metric_keys)):
        for i, model in enumerate(MODEL_ORDER):
            val = report["claim_results"][cid][model]["metrics"][mk]
            data_matrix[i, j] = val

    # Build pass/fail boolean matrix
    pass_matrix = np.zeros_like(data_matrix, dtype=bool)
    for j, thresh in enumerate(thresholds):
        pass_matrix[:, j] = data_matrix[:, j] >= thresh

    # Create figure with gridspec for heatmap + inset
    fig = plt.figure(figsize=(10, 8))
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1.2], hspace=0.35)
    ax_heat = fig.add_subplot(gs[0])
    ax_scale = fig.add_subplot(gs[1])

    # ── Heatmap ──
    # Custom colormap: red for fail, green for pass
    cmap_pass = mcolors.ListedColormap(["#E8B4B4", "#B4DEB4"])
    bounds = [-0.5, 0.5, 1.5]
    norm_pass = mcolors.BoundaryNorm(bounds, cmap_pass.N)

    ax_heat.imshow(pass_matrix.astype(float), cmap=cmap_pass, norm=norm_pass,
                   aspect="auto", interpolation="nearest")

    # Annotate cells with values
    for i in range(n_models):
        for j in range(n_claims):
            val = data_matrix[i, j]
            # Format: integers for causal_effect_count, else 2 decimal places
            if j == 4:  # causal_steering_behavior
                text = f"{int(val)}"
            else:
                text = f"{val:.2f}"
            text_color = "#1a5e1a" if pass_matrix[i, j] else "#8b1a1a"
            ax_heat.text(j, i, text, ha="center", va="center",
                         fontsize=10, fontweight="bold", color=text_color)

    ax_heat.set_xticks(np.arange(n_claims))
    ax_heat.set_xticklabels(claim_labels, fontsize=9, ha="center")
    ax_heat.set_yticks(np.arange(n_models))
    ax_heat.set_yticklabels([MODEL_LABELS[m] for m in MODEL_ORDER], fontsize=10)

    # Add threshold row at bottom
    ax_heat.set_xlim(-0.5, n_claims - 0.5)
    for j, thresh in enumerate(thresholds):
        t_str = f"{thresh:.0f}" if thresh == int(thresh) else f"{thresh:.2f}"
        ax_heat.text(j, n_models + 0.15, f"thresh: {t_str}",
                     ha="center", va="top", fontsize=7, color="#666666")

    # Grid lines between cells
    for i in range(n_models + 1):
        ax_heat.axhline(i - 0.5, color="white", linewidth=2)
    for j in range(n_claims + 1):
        ax_heat.axvline(j - 0.5, color="white", linewidth=2)

    # Legend
    legend_patches = [
        Patch(facecolor="#B4DEB4", edgecolor="#888888", label="Pass (>= threshold)"),
        Patch(facecolor="#E8B4B4", edgecolor="#888888", label="Fail (< threshold)"),
    ]
    ax_heat.legend(handles=legend_patches, loc="upper right",
                   bbox_to_anchor=(1.0, -0.05), ncol=2, frameon=True,
                   framealpha=0.9, edgecolor="#cccccc", fontsize=8)

    ax_heat.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
    ax_heat.spines["top"].set_visible(False)
    ax_heat.spines["right"].set_visible(False)
    ax_heat.spines["bottom"].set_visible(False)
    ax_heat.spines["left"].set_visible(False)

    # ── Scaling inset: probe accuracy vs log(params) ──
    if scaling and "data" in scaling:
        params_list = []
        acc_list = []
        family_list = []
        for entry in scaling["data"]:
            m = entry["model"]
            params_list.append(entry["params_b"])
            acc_list.append(entry["accuracy"])
            family_list.append(family_of(m))

        log_params = np.log10(params_list)
        params_arr = np.array(params_list)
        acc_arr = np.array(acc_list)

        # Plot points colored by family
        for m_key, p, a in zip(
            [e["model"] for e in scaling["data"]], params_list, acc_list
        ):
            ax_scale.scatter(p, a, color=FAMILY_COLORS[family_of(m_key)],
                             s=70, zorder=3, edgecolor="white", linewidth=0.5)

        # Regression line
        log_p = np.log10(params_arr)
        coeffs = np.polyfit(log_p, acc_arr, 1)
        x_line = np.linspace(min(params_arr) * 0.8, max(params_arr) * 1.2, 100)
        y_line = np.polyval(coeffs, np.log10(x_line))
        ax_scale.plot(x_line, y_line, color="#333333", linewidth=1.2,
                      linestyle="-", zorder=1)

        # Bootstrap CI band
        ci_lo = scaling["bootstrap"]["ci_95_lower"]
        ci_hi = scaling["bootstrap"]["ci_95_upper"]
        rho_point = scaling["spearman_point_estimate"]

        # Shade a band around the regression line based on residual spread
        residuals = acc_arr - np.polyval(coeffs, log_p)
        res_std = np.std(residuals)
        y_lo = y_line - 1.96 * res_std
        y_hi = y_line + 1.96 * res_std
        ax_scale.fill_between(x_line, y_lo, y_hi, color="#CCCCCC", alpha=0.3,
                              zorder=0)

        ax_scale.set_xscale("log")
        ax_scale.set_xlabel("Parameters (B)")
        ax_scale.set_ylabel("Probe Accuracy")
        ax_scale.set_ylim(0.65, 0.90)

        # Custom x-ticks for log scale
        ax_scale.set_xticks([1, 1.5, 2, 7, 8, 9])
        ax_scale.set_xticklabels(["1B", "1.5B", "2B", "7B", "8B", "9B"])
        ax_scale.minorticks_off()

        # Annotation with Spearman rho
        ax_scale.text(0.02, 0.95,
                      f"Spearman rho = {rho_point:.2f}  "
                      f"(95% CI [{ci_lo:.2f}, {ci_hi:.2f}])",
                      transform=ax_scale.transAxes, fontsize=8, va="top",
                      bbox=dict(boxstyle="round,pad=0.3", facecolor="#f0f0f0",
                                edgecolor="#cccccc"))

        # Family legend
        legend_patches = [Patch(facecolor=FAMILY_COLORS[f], label=f.capitalize())
                          for f in ("llama", "qwen", "gemma")]
        ax_scale.legend(handles=legend_patches, loc="lower right", frameon=True,
                        framealpha=0.9, edgecolor="#cccccc", fontsize=8)

        ax_scale.spines["top"].set_visible(False)
        ax_scale.spines["right"].set_visible(False)
    else:
        ax_scale.text(0.5, 0.5, "Scaling data not available",
                      ha="center", va="center", transform=ax_scale.transAxes)

    save_fig(fig, "fig4_universality_scorecard")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════
def main():
    print(f"Data directory: {DATA_DIR}")
    print(f"Output directory: {FIG_DIR}")
    print()

    fig1_probe_vs_baseline()
    fig2_geometry_and_severity()
    fig3_steering_null()
    fig4_universality_scorecard()

    print()
    print("All figures generated. Verifying ...")
    ok = True
    for stem in ("fig1_probe_vs_baseline", "fig2_geometry_and_severity",
                 "fig3_steering_null", "fig4_universality_scorecard"):
        p = FIG_DIR / f"{stem}.png"
        if p.exists():
            size_kb = p.stat().st_size / 1024
            status = "OK" if size_kb > 10 else "WARNING: <10KB"
            print(f"  {p.name}: {size_kb:.1f} KB  [{status}]")
            if size_kb <= 10:
                ok = False
        else:
            print(f"  {p.name}: MISSING")
            ok = False

    if ok:
        print("\nAll 4 figures pass verification.")
    else:
        print("\nSome figures failed verification.")
        sys.exit(1)


if __name__ == "__main__":
    main()
