#!/usr/bin/env python3
"""Generate publication-quality figures for the emotions replication paper.

Reads JSON data files from results/emotions/ (git-tracked, committed
alongside the writeup) and produces 5 figures in figures/emotions/.

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
DATA_DIR = REPO / "results" / "emotions"
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
MODEL_ORDER = ["qwen_1_5b", "llama_1b", "gemma_2b", "qwen_7b", "llama_8b", "gemma_9b", "llama_70b"]
MODEL_LABELS = {
    "qwen_1_5b": "Qwen 1.5B",
    "llama_1b": "Llama 1B",
    "gemma_2b": "Gemma 2B",
    "qwen_7b": "Qwen 7B",
    "llama_8b": "Llama 8B",
    "gemma_9b": "Gemma 9B",
    "llama_70b": "Llama 70B",
}
MODEL_PARAMS = {
    "qwen_1_5b": 1.5, "llama_1b": 1.0, "gemma_2b": 2.0,
    "qwen_7b": 7.0, "llama_8b": 8.0, "gemma_9b": 9.0,
    "llama_70b": 70.0,
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
        "llama_1b": 0.773, "llama_8b": 0.819, "llama_70b": 0.845,
        "qwen_1_5b": 0.731, "qwen_7b": 0.784,
        "gemma_2b": 0.776, "gemma_9b": 0.840,
    }
    probe_std = {
        "llama_1b": 0.008, "llama_8b": 0.004, "llama_70b": None,
        "qwen_1_5b": 0.012, "qwen_7b": 0.007,
        "gemma_2b": 0.011, "gemma_9b": 0.004,
    }
    best_lexical = 0.400  # bag-of-words baseline (same for all models)
    chance = 1.0 / 15.0   # 0.0667
    original_paper = 0.713  # Claude Sonnet 4.5

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(MODEL_ORDER))
    bar_w = 0.35

    # Probe bars (green) — handle None std (single-seed estimates)
    probe_vals = [probe_acc[m] for m in MODEL_ORDER]
    probe_errs = [probe_std[m] if probe_std[m] is not None else 0.0
                  for m in MODEL_ORDER]
    ax.bar(x - bar_w / 2, probe_vals, bar_w, color="#6ACC64", edgecolor="white",
           linewidth=0.5, label="Residual-stream probe", yerr=probe_errs,
           capsize=3, error_kw={"linewidth": 1.0, "color": "#333333"})

    # For models with None std, draw a minimal hairline instead and add dagger
    for i, m in enumerate(MODEL_ORDER):
        if probe_std[m] is None:
            # Minimal hairline (1px) to indicate "no error bar"
            ax.plot([x[i] - bar_w / 2, x[i] - bar_w / 2],
                    [probe_vals[i] - 0.002, probe_vals[i] + 0.002],
                    color="#333333", linewidth=0.5)
            # Dagger symbol below the x-label
            ax.text(x[i], -0.04, "\u2020", ha="center", va="top",
                    fontsize=11, fontweight="bold", color="#555555")

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

    # Footnote for dagger symbol
    fig.text(0.02, -0.02, "\u2020Single-seed estimate (no multi-seed data)",
             fontsize=7, color="#555555", ha="left", va="top", style="italic")

    save_fig(fig, "fig1_probe_vs_baseline")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 2: Geometry (valence correlation) + Severity pairs
# ═══════════════════════════════════════════════════════════════════════════
def fig2_geometry_and_severity():
    print("Figure 2: Geometry and severity pairs ...")

    valence_r = {
        "llama_1b": 0.666, "llama_8b": 0.738, "llama_70b": 0.754,
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

    # Load Clopper-Pearson CIs
    cp_data = load_json("figure3_clopper_pearson.json")
    cp_cis = cp_data.get("clopper_pearson_CIs", {})

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

    # ── Right panel: This replication (medium + large) ──
    repl_models_keys = ["llama_8b", "qwen_7b", "gemma_9b", "llama_70b"]
    repl_models = ["Llama 8B", "Qwen 7B", "Gemma 9B", "Llama 70B"]
    x_r = np.arange(len(repl_models))
    bar_w = 0.35

    # Medium models all 0%; 70B has non-zero baselines on some scenarios
    # but 0/45 significant effects. Show pooled rates across all conditions.
    # 70B: load from result.json to get mean baseline/steered rates
    baseline_vals = [0, 0, 0, 0]
    steered_vals = [0, 0, 0, 0]

    # Try to load 70B detailed steering results for pooled rates
    steer_70b_path = DATA_DIR / "llama_70b" / "causal_steering_behavior" / "result.json"
    if steer_70b_path.exists():
        with open(steer_70b_path) as f:
            steer_70b = json.load(f)
        effects = steer_70b.get("metrics", {}).get("per_scenario_effects", [])
        if effects:
            base_rates = [e["baseline_unethical_rate"] * 100 for e in effects]
            steer_rates = [e["steered_unethical_rate"] * 100 for e in effects]
            baseline_vals[3] = float(np.mean(base_rates))
            steered_vals[3] = float(np.mean(steer_rates))

    # Compute Clopper-Pearson CI upper bounds (as %) for error bars
    ci_uppers = []
    for mk in repl_models_keys:
        if mk in cp_cis:
            ci_upper = cp_cis[mk]["ci95_per_condition"][1] * 100
        else:
            ci_upper = 30.85  # fallback
        ci_uppers.append(ci_upper)

    ax_r.bar(x_r - bar_w / 2, baseline_vals, bar_w, color="#FCBF74",
             edgecolor="#AAAAAA", linewidth=0.5, label="Baseline")
    steered_bars = ax_r.bar(x_r + bar_w / 2, steered_vals, bar_w, color="#C44E52",
                            edgecolor="#AAAAAA", linewidth=0.5,
                            label="Steered (+desperate)")

    # Add Clopper-Pearson 95% CI error bars on the steered bars
    ax_r.errorbar(x_r + bar_w / 2, steered_vals,
                  yerr=[[0] * len(repl_models), ci_uppers],
                  fmt="none", ecolor="#333333", elinewidth=1.0, capsize=4,
                  capthick=1.0, zorder=5)

    # Add tiny markers at 0 for medium models so the chart isn't empty
    for i in range(3):  # medium models only
        ax_r.plot(x_r[i] - bar_w / 2, 0.3, marker="_", color="#FCBF74",
                  markersize=12, markeredgewidth=2)
        ax_r.plot(x_r[i] + bar_w / 2, 0.3, marker="_", color="#C44E52",
                  markersize=12, markeredgewidth=2)

    ax_r.set_xticks(x_r)
    ax_r.set_xticklabels(repl_models)
    ax_r.set_ylim(0, 85)
    ax_r.set_title("This Replication\n(Open-Source Models)", fontsize=10,
                   fontweight="bold", pad=10)
    ax_r.legend(loc="upper right", frameon=True, framealpha=0.9,
                edgecolor="#cccccc", fontsize=8)
    ax_r.spines["top"].set_visible(False)
    ax_r.spines["right"].set_visible(False)

    # Annotation
    ax_r.text(0.5, 0.6,
              "0/45 significant effects\nacross 4 models\n"
              "(Medium: 0% baseline floor;\n"
              " 70B: non-zero baselines\n"
              " but no significant shift)\n"
              "CP 95% CI: [0%, 30.8%] per cond.",
              transform=ax_r.transAxes, ha="center", va="center",
              fontsize=8, color="#555555",
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

    # ── Reference row: Claude Sonnet 4.5 (original paper values) ──
    # Values: probe=0.713, generalization=0.760, valence=0.810,
    #         parametric=N/A, steering="significant", preference=0.850
    # "significant" for causal_steering maps to threshold-passing;
    # we use the threshold value (3) as a display stand-in.
    ref_values = [0.713, 0.760, 0.810, None, 3, 0.850]  # None = N/A
    ref_label = "Claude Sonnet 4.5"

    # Build the data matrix (from cross_model_report if available, else from result.json)
    data_matrix = np.zeros((n_models, n_claims))
    for j, (cid, mk) in enumerate(zip(claims_ids, metric_keys)):
        for i, model in enumerate(MODEL_ORDER):
            # Try cross_model_report first
            if (report and "claim_results" in report
                    and cid in report.get("claim_results", {})
                    and model in report["claim_results"][cid]):
                val = report["claim_results"][cid][model]["metrics"][mk]
            else:
                # Fallback: read directly from result.json
                rj_path = DATA_DIR / model / cid / "result.json"
                if rj_path.exists():
                    with open(rj_path) as f:
                        rj = json.load(f)
                    val = rj.get("metrics", {}).get(mk, 0)
                else:
                    val = 0
            data_matrix[i, j] = val

    # Build pass/fail boolean matrix
    pass_matrix = np.zeros_like(data_matrix, dtype=bool)
    for j, thresh in enumerate(thresholds):
        pass_matrix[:, j] = data_matrix[:, j] >= thresh

    # Total rows = 1 (reference) + n_models (replication)
    n_total_rows = 1 + n_models

    # Create figure with gridspec for heatmap + inset
    fig = plt.figure(figsize=(10, 8.5))
    gs = fig.add_gridspec(2, 1, height_ratios=[3.2, 1.2], hspace=0.35)
    ax_heat = fig.add_subplot(gs[0])
    ax_scale = fig.add_subplot(gs[1])

    # ── Heatmap ──
    # Intensity-graded colors: stronger greens for higher passing values,
    # stronger reds for values further below threshold.  Each column is
    # normalised independently so the colour gradient is meaningful within
    # each claim (different metrics have different natural ranges).
    from matplotlib.colors import LinearSegmentedColormap

    green_cmap = LinearSegmentedColormap.from_list(
        "greens", ["#daf2da", "#4caf50", "#1b5e20"], N=256
    )
    red_cmap = LinearSegmentedColormap.from_list(
        "reds", ["#fce4e4", "#e57373", "#b71c1c"], N=256
    )

    # Reference row color: light blue/gray
    ref_bg = np.array([0.85, 0.91, 0.97, 1.0])    # light blue
    ref_na_bg = np.array([0.88, 0.88, 0.88, 1.0])  # gray for N/A cells

    # Build an RGBA image: row 0 = reference, rows 1..n_models = replication
    cell_colors = np.zeros((n_total_rows, n_claims, 4))

    # Fill reference row (row 0)
    for j in range(n_claims):
        if ref_values[j] is None:
            cell_colors[0, j] = ref_na_bg
        else:
            cell_colors[0, j] = ref_bg

    # Fill replication rows (rows 1..n_total_rows)
    for j in range(n_claims):
        col_vals = data_matrix[:, j]
        thresh = thresholds[j]

        # Passing cells: normalise to [0, 1] within the range
        # [threshold, column_max].  Higher values -> darker green.
        pass_mask = col_vals >= thresh
        col_max = col_vals[pass_mask].max() if pass_mask.any() else thresh + 1e-9
        col_min_pass = thresh
        span = max(col_max - col_min_pass, 1e-9)

        for i in range(n_models):
            row = i + 1  # shift down by 1 for reference row
            if pass_matrix[i, j]:
                normed = np.clip((col_vals[i] - col_min_pass) / span, 0, 1)
                cell_colors[row, j] = green_cmap(normed)
            else:
                # Failing cells: normalise within [0, threshold].
                # Lower values -> darker red.
                normed = np.clip(1.0 - col_vals[i] / max(thresh, 1e-9), 0, 1)
                cell_colors[row, j] = red_cmap(normed)

    ax_heat.imshow(cell_colors, aspect="auto", interpolation="nearest")

    # Annotate reference row (row 0)
    for j in range(n_claims):
        if ref_values[j] is None:
            text = "N/A"
            text_color = "#666666"
        elif j == 4:  # causal_steering — show "sig."
            text = "sig."
            text_color = "#1a4a7a"
        else:
            text = f"{ref_values[j]:.3f}"
            text_color = "#1a4a7a"
        ax_heat.text(j, 0, text, ha="center", va="center",
                     fontsize=10, fontweight="bold", color=text_color,
                     fontstyle="italic")

    # Annotate replication rows (rows 1..n_total_rows)
    for i in range(n_models):
        row = i + 1
        for j in range(n_claims):
            val = data_matrix[i, j]
            if j == 4:  # causal_steering_behavior (integer count)
                text = f"{int(val)}"
            else:
                text = f"{val:.2f}"
            # Dark text for readability against the gradient backgrounds
            text_color = "#0d3b0d" if pass_matrix[i, j] else "#5a0000"
            ax_heat.text(j, row, text, ha="center", va="center",
                         fontsize=10, fontweight="bold", color=text_color)

    ax_heat.set_xticks(np.arange(n_claims))
    ax_heat.set_xticklabels(claim_labels, fontsize=9, ha="center")
    ax_heat.set_yticks(np.arange(n_total_rows))
    y_labels = [ref_label] + [MODEL_LABELS[m] for m in MODEL_ORDER]
    ax_heat.set_yticklabels(y_labels, fontsize=10)

    # Italicize the reference row label
    ytick_labels = ax_heat.get_yticklabels()
    ytick_labels[0].set_fontstyle("italic")
    ytick_labels[0].set_color("#1a4a7a")

    # Add threshold row at bottom
    ax_heat.set_xlim(-0.5, n_claims - 0.5)
    for j, thresh in enumerate(thresholds):
        t_str = f"{thresh:.0f}" if thresh == int(thresh) else f"{thresh:.2f}"
        ax_heat.text(j, n_total_rows + 0.15, f"thresh: {t_str}",
                     ha="center", va="top", fontsize=7, color="#666666")

    # Grid lines between cells
    for i in range(n_total_rows + 1):
        lw = 3 if i == 1 else 2  # thicker line between reference and replication
        ax_heat.axhline(i - 0.5, color="white", linewidth=lw)
    for j in range(n_claims + 1):
        ax_heat.axvline(j - 0.5, color="white", linewidth=2)

    # Legend
    legend_patches = [
        Patch(facecolor="#B4DEB4", edgecolor="#888888", label="Pass (>= threshold)"),
        Patch(facecolor="#E8B4B4", edgecolor="#888888", label="Fail (< threshold)"),
        Patch(facecolor="#d9e8f5", edgecolor="#888888", label="Original paper (reference)"),
    ]
    ax_heat.legend(handles=legend_patches, loc="upper right",
                   bbox_to_anchor=(1.0, -0.05), ncol=3, frameon=True,
                   framealpha=0.9, edgecolor="#cccccc", fontsize=8)

    ax_heat.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
    ax_heat.spines["top"].set_visible(False)
    ax_heat.spines["right"].set_visible(False)
    ax_heat.spines["bottom"].set_visible(False)
    ax_heat.spines["left"].set_visible(False)

    # ── Scaling inset: probe accuracy vs log(params) ──
    # Build scaling data: use scaling_bootstrap_ci.json if available, then add 70B
    scaling_data = []
    if scaling and "data" in scaling:
        scaling_data = list(scaling["data"])

    # Add 70B if not already present
    llama_70b_present = any(e["model"] == "llama_70b" for e in scaling_data)
    if not llama_70b_present:
        rj_70b = DATA_DIR / "llama_70b" / "emotion_probe_classification" / "result.json"
        if rj_70b.exists():
            with open(rj_70b) as f:
                rj = json.load(f)
            scaling_data.append({
                "model": "llama_70b",
                "params_b": 70.0,
                "accuracy": rj["metrics"]["probe_accuracy"],
            })

    if scaling_data:
        params_list = []
        acc_list = []
        family_list = []
        for entry in scaling_data:
            m = entry["model"]
            params_list.append(entry["params_b"])
            acc_list.append(entry["accuracy"])
            family_list.append(family_of(m))

        log_params = np.log10(params_list)
        params_arr = np.array(params_list)
        acc_arr = np.array(acc_list)

        # Plot points colored by family
        for m_key, p, a in zip(
            [e["model"] for e in scaling_data], params_list, acc_list
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

        # Custom x-ticks for log scale (now including 70B)
        ax_scale.set_xticks([1, 1.5, 2, 7, 8, 9, 70])
        ax_scale.set_xticklabels(["1B", "1.5B", "2B", "7B", "8B", "9B", "70B"])
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
    else:  # no scaling_data at all
        ax_scale.text(0.5, 0.5, "Scaling data not available",
                      ha="center", va="center", transform=ax_scale.transAxes)

    save_fig(fig, "fig4_universality_scorecard")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 5: Sentiment positive control (v3.4)
# ═══════════════════════════════════════════════════════════════════════════
def fig5_sentiment_positive_control():
    print("Figure 5: Sentiment positive control ...")

    # Load happy@alpha=5.0 shift from each model's sentiment_control.json
    # Models ordered by parameter count
    ordered_models = ["llama_1b", "qwen_1_5b", "gemma_2b", "llama_8b", "qwen_7b", "gemma_9b", "llama_70b"]
    ordered_labels = ["Llama 1B", "Qwen 1.5B", "Gemma 2B", "Llama 8B", "Qwen 7B", "Gemma 9B", "Llama 70B"]

    shifts = []
    for model_key in ordered_models:
        shift = 0.0
        # Try sentiment_control.json first (medium/small models)
        sc_path = DATA_DIR / model_key / "critique_followups" / "sentiment_control.json"
        if sc_path.exists():
            with open(sc_path) as f:
                sc = json.load(f)
            shift = sc.get("summary", {}).get("happy", {}).get("shifts", {}).get("5.0", 0.0)
        else:
            # Try combined.json (70B format)
            comb_path = DATA_DIR / model_key / "critique_followups" / "combined.json"
            if comb_path.exists():
                with open(comb_path) as f:
                    comb = json.load(f)
                # Find happy at alpha=5.0 from sentiment_control list
                sc_data = comb.get("sentiment_control", [])
                baseline = None
                alpha5 = None
                for entry in sc_data:
                    if entry.get("concept") == "happy":
                        if entry.get("alpha") == 0.0:
                            baseline = entry.get("mean_sentiment", 0)
                        elif entry.get("alpha") == 5.0:
                            alpha5 = entry.get("mean_sentiment", 0)
                if baseline is not None and alpha5 is not None:
                    shift = alpha5 - baseline
            else:
                print(f"  WARNING: no sentiment data for {model_key}, using 0.0")
        shifts.append(shift)

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(ordered_models))
    colors = [FAMILY_COLORS[family_of(m)] for m in ordered_models]

    bars = ax.bar(x, shifts, color=colors, edgecolor="white", linewidth=0.5,
                  width=0.6)

    # Annotate bar values
    for i, (bar, val) in enumerate(zip(bars, shifts)):
        y_pos = val + 0.01 if val >= 0 else val - 0.03
        va = "bottom" if val >= 0 else "top"
        ax.text(bar.get_x() + bar.get_width() / 2, y_pos,
                f"+{val:.3f}" if val >= 0 else f"{val:.3f}",
                ha="center", va=va, fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(ordered_labels)
    ax.set_ylabel("Sentiment Shift (delta from baseline)")
    ax.axhline(0, color="#333333", linewidth=0.8, zorder=0)

    # Family legend
    legend_patches = [Patch(facecolor=FAMILY_COLORS[f], label=f.capitalize())
                      for f in ("llama", "qwen", "gemma")]
    ax.legend(handles=legend_patches, loc="upper left", frameon=True,
              framealpha=0.9, edgecolor="#cccccc")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Add annotation box
    ax.text(0.98, 0.95,
            "Positive control: 'happy' vector\n"
            "at alpha=5.0 shifts sentiment\n"
            "in the expected direction\n"
            "6 of 7 models shift positive\n"
            "(70B degrades at high alpha)",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=8, color="#555555",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#e8f5e9",
                      edgecolor="#6ACC64", alpha=0.9))

    save_fig(fig, "fig5_sentiment_positive_control")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 6: Sycophancy pushback capitulation rates
# ═══════════════════════════════════════════════════════════════════════════
def fig6_sycophancy_pushback():
    print("Figure 6: Sycophancy pushback capitulation rates ...")

    syc_data = load_json("sycophancy_v2_cross_model_summary.json")
    pushback = syc_data.get("pushback_fisher_tests", {})

    if not pushback:
        print("  WARNING: no sycophancy pushback data found, skipping fig6")
        return

    # 6 models ordered by parameter count (no 70B data for sycophancy)
    syc_model_order = ["llama_1b", "qwen_1_5b", "gemma_2b", "qwen_7b", "llama_8b", "gemma_9b"]
    syc_model_labels = {
        "llama_1b": "Llama 1B", "qwen_1_5b": "Qwen 1.5B", "gemma_2b": "Gemma 2B",
        "qwen_7b": "Qwen 7B", "llama_8b": "Llama 8B", "gemma_9b": "Gemma 9B",
    }

    baseline_pcts = []
    steered_pcts = []
    p_values = []
    labels = []

    for mk in syc_model_order:
        entry = pushback.get(mk, {})
        baseline_pcts.append(entry.get("baseline_rate", 0) * 100)
        steered_pcts.append(entry.get("steered_rate", 0) * 100)
        p_values.append(entry.get("p_value_one_sided", 1.0))
        labels.append(syc_model_labels[mk])

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(syc_model_order))
    bar_w = 0.35

    # Baseline bars (light orange)
    ax.bar(x - bar_w / 2, baseline_pcts, bar_w, color="#FCBF74",
           edgecolor="white", linewidth=0.5, label="Baseline")

    # Steered bars (dark red)
    ax.bar(x + bar_w / 2, steered_pcts, bar_w, color="#C44E52",
           edgecolor="white", linewidth=0.5, label="Steered (+emotion)")

    # Annotate significance markers above steered bars
    for i, p in enumerate(p_values):
        steered_y = steered_pcts[i]
        if p < 0.05:
            marker = "*"
        elif p < 0.10:
            marker = "\u2020"  # dagger for p<0.10
        else:
            marker = ""
        if marker:
            ax.text(x[i] + bar_w / 2, steered_y + 1.0, marker,
                    ha="center", va="bottom", fontsize=14, fontweight="bold",
                    color="#333333")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Capitulation Rate (%)")
    ax.set_ylim(0, max(max(baseline_pcts), max(steered_pcts)) * 1.3 + 5)

    ax.legend(loc="upper right", frameon=True, framealpha=0.9,
              edgecolor="#cccccc", fontsize=9)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Annotation box
    ax.text(0.02, 0.95,
            "Emotion steering increases\n"
            "pushback capitulation in\n"
            "1 of 6 models (p<0.05)",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=8, color="#555555",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#fff3e0",
                      edgecolor="#FCBF74", alpha=0.9))

    # Significance legend footnote
    fig.text(0.02, -0.02,
             "* p<0.05 (Fisher exact, one-sided)    "
             "\u2020 p<0.10 (Fisher exact, one-sided)",
             fontsize=7, color="#555555", ha="left", va="top", style="italic")

    save_fig(fig, "fig6_sycophancy_pushback")


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
    fig5_sentiment_positive_control()
    fig6_sycophancy_pushback()

    print()
    print("All figures generated. Verifying ...")
    ok = True
    for stem in ("fig1_probe_vs_baseline", "fig2_geometry_and_severity",
                 "fig3_steering_null", "fig4_universality_scorecard",
                 "fig5_sentiment_positive_control", "fig6_sycophancy_pushback"):
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
        print("\nAll 6 figures pass verification.")
    else:
        print("\nSome figures failed verification.")
        sys.exit(1)


if __name__ == "__main__":
    main()
