#!/usr/bin/env python3
"""
Generate publication-quality plots from ablation analysis results.

Reads:
    data/ablation_metrics.json  -- output from ablation.py

Outputs (all under paper/plots/):
    feature_ablation.pdf        -- progressive feature subset performance
    hyperparam_heatmap.pdf      -- XGBoost depth x estimators heatmap
    model_family_comparison.pdf -- comparison across model families
    bootstrap_ci.pdf            -- confidence interval forest plot
    feature_correlation.pdf     -- Spearman correlation heatmap
    prevalence_robustness.pdf   -- metrics at different class prevalence
    threshold_prevalence.pdf    -- threshold sweep across prevalence levels

Usage:
    python code/ablation_plots.py
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

LOG = logging.getLogger("ablation_plots")

_LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
_LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)-5s  %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(_LOG_DIR / "ablation_plots.log"),
    ],
)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_METRICS = ROOT / "data" / "ablation_metrics.json"
DEFAULT_PLOTS_DIR = ROOT / "paper" / "plots"

# ── Theme (matches evaluate.py) ────────────────────────────────────────────

C_PRIMARY = "#911a4b"
C_SECONDARY = "#0d1c24"
C_TERTIARY = "#5a7d8c"
C_NEUTRAL = "#BBBBBB"
C_HIGHLIGHT = "#a89882"
C_DARK = "#222222"

PALETTE = [C_PRIMARY, C_SECONDARY, C_TERTIARY, C_HIGHLIGHT, "#7a9a82", "#66CCEE"]

_FEATURE_LABELS = {
    "file_count": "File count",
    "total_size_bytes": "Total size",
    "avg_file_size_bytes": "Avg file size",
    "min_file_size_bytes": "Min file size",
    "max_file_size_bytes": "Max file size",
    "stddev_file_size_bytes": "Stddev file size",
    "total_records": "Total records",
    "avg_records_per_file": "Avg records/file",
    "num_partitions_actual": "Partition count",
    "avg_files_per_partition": "Avg files/part.",
    "max_files_per_partition": "Max files/part.",
    "min_files_per_partition": "Min files/part.",
    "stddev_files_per_partition": "Stddev files/part.",
    "num_snapshots": "Snapshot count",
    "small_file_ratio": "Small-file ratio",
    "file_size_cv": "File size CV",
    "files_per_partition_cv": "Files/part. CV",
}

COL_W = 3.5
FULL_W = 7.16


def _fl(name: str) -> str:
    return _FEATURE_LABELS.get(name, name.replace("_", " ").title())


def _apply_theme() -> None:
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["STIX", "STIXGeneral", "Times", "Nimbus Roman"],
        "mathtext.fontset": "stix",
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "legend.title_fontsize": 8,
        "axes.edgecolor": C_DARK,
        "axes.linewidth": 0.6,
        "axes.labelcolor": C_DARK,
        "axes.facecolor": "#FAFAFA",
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": "#E0E0E0",
        "grid.linewidth": 0.4,
        "grid.linestyle": "-",
        "xtick.color": C_DARK,
        "ytick.color": C_DARK,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "figure.facecolor": "white",
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "savefig.transparent": False,
        "legend.frameon": True,
        "legend.framealpha": 0.9,
        "legend.edgecolor": "#CCCCCC",
        "legend.fancybox": False,
    })


def _despine(ax, left=False):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if left:
        ax.spines["left"].set_visible(False)


# ── Plot functions ──────────────────────────────────────────────────────────


def plot_feature_ablation(data: dict, out: Path) -> None:
    """Progressive feature addition: R2 vs number of features (single panel)."""
    prog = data["feature_ablation"]["progressive"]
    ks = [e["k"] for e in prog]
    r2s = [e["reg"]["r2"] for e in prog]

    fig, ax = plt.subplots(figsize=(COL_W, COL_W * 0.55))

    ax.plot(ks, r2s, "s-", color=C_PRIMARY, markersize=4, linewidth=1.2)
    ax.set_xlabel("Number of features (by importance rank)")
    ax.set_ylabel(r"Regression $R^2$")
    ax.set_ylim(0.95, 1.005)
    ax.set_xticks(ks)
    _despine(ax)

    # Annotate feature names for first few points
    order = data["feature_ablation"]["importance_order"]
    for i, k in enumerate(ks[:3]):
        label = _fl(order[k - 1]) if k <= len(order) else ""
        ax.annotate(
            f"+{label}",
            (k, r2s[i]),
            textcoords="offset points",
            xytext=(5, -10),
            fontsize=5.5,
            color=C_DARK,
        )

    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    LOG.info("Saved %s", out)


def plot_leave_one_out(data: dict, out: Path) -> None:
    """Leave-one-out feature ablation bar chart."""
    loo = data["feature_ablation"]["leave_one_out"]
    # Full model baselines
    full = data["feature_ablation"]["progressive"][-1]
    full_r2 = full["reg"]["r2"]

    # Sort by R2 drop
    loo_sorted = sorted(loo, key=lambda x: x["reg"]["r2"] - full_r2)

    features = [_fl(e["dropped_feature"]) for e in loo_sorted]
    deltas = [e["reg"]["r2"] - full_r2 for e in loo_sorted]

    fig, ax = plt.subplots(figsize=(COL_W, COL_W * 0.85))

    colors = [C_SECONDARY if d < -0.001 else C_NEUTRAL for d in deltas]
    ax.barh(features, deltas, color=colors, edgecolor="white", linewidth=0.3, height=0.65)
    ax.axvline(0, color=C_DARK, linewidth=0.5)
    ax.set_xlabel(r"$\Delta R^2$ when feature dropped")
    _despine(ax)

    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    LOG.info("Saved %s", out)


def plot_hyperparam_heatmap(data: dict, out: Path) -> None:
    """Heatmap of R2 across max_depth x n_estimators."""
    sweep = data["hyperparam_sensitivity"]["xgb_sweep"]

    depths = sorted(set(e["max_depth"] for e in sweep))
    n_ests = sorted(set(e["n_estimators"] for e in sweep))

    r2_grid = np.zeros((len(depths), len(n_ests)))
    for e in sweep:
        i = depths.index(e["max_depth"])
        j = n_ests.index(e["n_estimators"])
        r2_grid[i, j] = e["reg"]["r2"]

    fig, ax = plt.subplots(figsize=(COL_W, COL_W * 0.6))

    im = ax.imshow(r2_grid, cmap="YlGnBu", aspect="auto", vmin=0.84, vmax=1.0)
    ax.set_xticks(range(len(n_ests)))
    ax.set_xticklabels(n_ests)
    ax.set_yticks(range(len(depths)))
    ax.set_yticklabels(depths)
    ax.set_xlabel(r"\texttt{n\_estimators}")
    ax.set_ylabel(r"\texttt{max\_depth}")

    # Annotate cells
    for i in range(len(depths)):
        for j in range(len(n_ests)):
            val = r2_grid[i, j]
            color = "white" if val > 0.97 else C_DARK
            ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                    fontsize=6, color=color)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label(r"$R^2$", fontsize=7)
    cbar.ax.tick_params(labelsize=6)

    fig.savefig(out)
    plt.close(fig)
    LOG.info("Saved %s", out)


def plot_model_family_comparison(data: dict, out: Path) -> None:
    """Bar chart comparing model families."""
    families = data["hyperparam_sensitivity"]["model_families"]

    # Add XGBoost (default config: depth=6, n_est=200)
    xgb_default = None
    for e in data["hyperparam_sensitivity"]["xgb_sweep"]:
        if e["max_depth"] == 6 and e["n_estimators"] == 200:
            xgb_default = e
            break

    models = []
    if xgb_default:
        models.append({"model": "XGBoost", "clf": xgb_default["clf"], "reg": xgb_default["reg"]})
    for f in families:
        name = f["model"].replace("_", " ").title()
        models.append({"model": name, "clf": f["clf"], "reg": f["reg"]})

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FULL_W, FULL_W * 0.28))

    names = [m["model"] for m in models]
    x = np.arange(len(names))

    # Classification F1
    f1s = [m["clf"]["f1"] for m in models]
    bars1 = ax1.bar(x, f1s, color=PALETTE[:len(models)], edgecolor="white",
                    linewidth=0.4, width=0.6, zorder=3)
    for bar in bars1:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                 f"{h:.4f}", ha="center", va="bottom", fontsize=5.5, color=C_DARK)
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=25, ha="right", fontsize=6)
    ax1.set_ylabel("Classification F1")
    ax1.set_ylim(0.85, 1.02)
    _despine(ax1)

    # Regression R2
    r2s = [m["reg"]["r2"] for m in models]
    bars2 = ax2.bar(x, r2s, color=PALETTE[:len(models)], edgecolor="white",
                    linewidth=0.4, width=0.6, zorder=3)
    for bar in bars2:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                 f"{h:.4f}", ha="center", va="bottom", fontsize=5.5, color=C_DARK)
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=25, ha="right", fontsize=6)
    ax2.set_ylabel(r"Regression $R^2$")
    ax2.set_ylim(0.0, 1.1)
    _despine(ax2)

    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    LOG.info("Saved %s", out)


def plot_bootstrap_ci(data: dict, out: Path) -> None:
    """Forest plot of metrics with 95% confidence intervals."""
    ci = data["bootstrap_ci"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FULL_W, FULL_W * 0.28))

    # Classifier metrics
    clf_metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    clf_labels = ["Accuracy", "Precision", "Recall", "F1", "ROC AUC"]
    y_pos = np.arange(len(clf_metrics))

    for i, (m, label) in enumerate(zip(clf_metrics, clf_labels)):
        s = ci["classifier"][m]
        ax1.errorbar(
            s["mean"], i,
            xerr=[[s["mean"] - s["ci_lo"]], [s["ci_hi"] - s["mean"]]],
            fmt="o", color=C_PRIMARY, markersize=4, capsize=3, capthick=0.8,
            linewidth=0.8,
        )
        ax1.text(s["ci_lo"] - 0.002, i, f'{s["mean"]:.4f}', va="center",
                 ha="right", fontsize=5.5, color=C_DARK)

    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(clf_labels)
    ax1.set_xlabel("Score (95\\% CI)")
    ax1.set_title("Classification")
    ax1.invert_yaxis()
    _despine(ax1)

    # Regressor metrics
    reg_metrics = ["rmse", "mae", "r2"]
    reg_labels = ["RMSE", "MAE", r"$R^2$"]
    y_pos2 = np.arange(len(reg_metrics))

    for i, (m, label) in enumerate(zip(reg_metrics, reg_labels)):
        s = ci["regressor"][m]
        ax2.errorbar(
            s["mean"], i,
            xerr=[[s["mean"] - s["ci_lo"]], [s["ci_hi"] - s["mean"]]],
            fmt="s", color=C_SECONDARY, markersize=4, capsize=3, capthick=0.8,
            linewidth=0.8,
        )
        ax2.text(
            s["ci_hi"] + 0.001, i,
            f'{s["mean"]:.4f} [{s["ci_lo"]:.4f}, {s["ci_hi"]:.4f}]',
            va="center", ha="left", fontsize=5.5, color=C_DARK,
        )

    ax2.set_yticks(y_pos2)
    ax2.set_yticklabels(reg_labels)
    ax2.set_xlabel("Score (95\\% CI)")
    ax2.set_title("Regression")
    ax2.invert_yaxis()
    _despine(ax2)

    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    LOG.info("Saved %s", out)


def plot_feature_correlation(data: dict, out: Path) -> None:
    """Spearman correlation heatmap of top features."""
    corr = data["correlation"]["spearman"]
    features = list(corr.keys())

    # Select top features by label correlation
    label_corr = data["correlation"]["feature_label_correlations"]
    fr_corr = label_corr["file_reduction_ratio"]["spearman"]
    top_features = sorted(features, key=lambda f: abs(fr_corr[f]), reverse=True)[:10]

    n = len(top_features)
    mat = np.zeros((n, n))
    for i, f1 in enumerate(top_features):
        for j, f2 in enumerate(top_features):
            mat[i, j] = corr[f1][f2]

    fig, ax = plt.subplots(figsize=(COL_W + 0.5, COL_W + 0.3))

    im = ax.imshow(mat, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    labels = [_fl(f) for f in top_features]
    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=5.5)
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=5.5)

    for i in range(n):
        for j in range(n):
            val = mat[i, j]
            color = "white" if abs(val) > 0.6 else C_DARK
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=4.5, color=color)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Spearman $\\rho$", fontsize=7)
    cbar.ax.tick_params(labelsize=6)

    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    LOG.info("Saved %s", out)


def plot_prevalence_robustness(data: dict, out: Path) -> None:
    """Metrics at different class prevalence levels."""
    levels = data["prevalence_robustness"]["levels"]

    prevs = [e["actual_prevalence"] for e in levels]
    f1s = [e["clf"]["f1"] for e in levels]
    r2s = [e["reg"]["r2"] for e in levels]
    accs = [e["clf"]["accuracy"] for e in levels]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FULL_W, FULL_W * 0.28))

    # Classification
    ax1.plot(prevs, f1s, "o-", color=C_PRIMARY, markersize=5, linewidth=1.2, label="F1")
    ax1.plot(prevs, accs, "s--", color=C_TERTIARY, markersize=4, linewidth=1.0, label="Accuracy")
    ax1.set_xlabel("Positive class prevalence")
    ax1.set_ylabel("Score")
    ax1.set_title("Classification")
    ax1.legend(loc="lower left")
    ax1.set_xlim(0.25, 0.95)
    _despine(ax1)

    # Regression
    ax2.plot(prevs, r2s, "s-", color=C_SECONDARY, markersize=5, linewidth=1.2)
    ax2.set_xlabel("Positive class prevalence")
    ax2.set_ylabel(r"$R^2$")
    ax2.set_title("Regression")
    ax2.set_xlim(0.25, 0.95)
    _despine(ax2)

    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    LOG.info("Saved %s", out)


def plot_threshold_prevalence(data: dict, out: Path) -> None:
    """Threshold k accuracy across prevalence levels."""
    levels = data["prevalence_robustness"]["levels"]

    fig, ax = plt.subplots(figsize=(COL_W, COL_W * 0.6))

    for i, level in enumerate(levels):
        prev = level["actual_prevalence"]
        sweep = level["threshold_sweep"]
        ks = [s["k"] for s in sweep]
        accs = [s["accuracy"] for s in sweep]
        ax.plot(ks, accs, "o-", color=PALETTE[i], markersize=3, linewidth=1.0,
                label=f"prev={prev:.2f}")

    ax.axvline(4, color=C_NEUTRAL, linestyle="--", linewidth=0.8, label="$k=4$")
    ax.set_xlabel(r"Threshold $k$")
    ax.set_ylabel("Accuracy")
    ax.legend(fontsize=6)
    ax.set_xlim(0.5, 20.5)
    _despine(ax)

    fig.savefig(out)
    plt.close(fig)
    LOG.info("Saved %s", out)


# ── CLI ─────────────────────────────────────────────────────────────────────


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Generate ablation analysis plots")
    p.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    p.add_argument("--plots-dir", type=Path, default=DEFAULT_PLOTS_DIR)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    _apply_theme()

    data = json.loads(args.metrics.read_text())
    out = args.plots_dir
    out.mkdir(parents=True, exist_ok=True)

    plot_feature_ablation(data, out / "feature_ablation.pdf")
    plot_leave_one_out(data, out / "leave_one_out.pdf")
    plot_hyperparam_heatmap(data, out / "hyperparam_heatmap.pdf")
    plot_model_family_comparison(data, out / "model_family_comparison.pdf")
    plot_bootstrap_ci(data, out / "bootstrap_ci.pdf")
    plot_feature_correlation(data, out / "feature_correlation.pdf")
    plot_prevalence_robustness(data, out / "prevalence_robustness.pdf")
    plot_threshold_prevalence(data, out / "threshold_prevalence.pdf")

    LOG.info("All plots saved to %s/", out)


if __name__ == "__main__":
    main()
