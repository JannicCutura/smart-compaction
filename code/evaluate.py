#!/usr/bin/env python3
"""
Evaluate trained models and generate publication-quality plots.

Reads:
    data/dataset.csv          -- full labelled dataset
    data/train_metrics.json   -- metrics from train.py
    data/model_clf.json       -- trained classifier
    data/model_reg.json       -- trained regressor

Outputs (all under paper/plots/):
    roc_curve.pdf             -- classifier ROC curve
    feature_importance_clf.pdf-- classifier feature importance (top 10)
    feature_importance_reg.pdf-- regressor feature importance (top 10)
    regression_scatter.pdf    -- predicted vs actual file_reduction_ratio
    label_distribution.pdf    -- histogram of file_reduction_ratio
    model_comparison_clf.pdf  -- grouped bar: classifier metric comparison
    model_comparison_reg.pdf  -- grouped bar: regressor metric comparison

Usage:
    python code/evaluate.py
    python code/evaluate.py --dataset data/dataset.csv --metrics data/train_metrics.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, XGBRegressor

LOG = logging.getLogger("evaluate")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)-5s  %(message)s",
)

# ── Defaults ────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = ROOT / "data" / "dataset.csv"
DEFAULT_METRICS = ROOT / "data" / "train_metrics.json"
DEFAULT_CLF_MODEL = ROOT / "data" / "model_clf.json"
DEFAULT_REG_MODEL = ROOT / "data" / "model_reg.json"
DEFAULT_PLOTS_DIR = ROOT / "paper" / "plots"

# Must match train.py
FEATURE_COLUMNS = [
    "file_count",
    "total_size_bytes",
    "avg_file_size_bytes",
    "min_file_size_bytes",
    "max_file_size_bytes",
    "stddev_file_size_bytes",
    "total_records",
    "avg_records_per_file",
    "num_partitions_actual",
    "avg_files_per_partition",
    "max_files_per_partition",
    "min_files_per_partition",
    "stddev_files_per_partition",
    "num_snapshots",
    "small_file_ratio",
    "file_size_cv",
    "files_per_partition_cv",
]

# ── Publication Theme ───────────────────────────────────────────────────────
# IEEE style guide: "Use 8 point Times New Roman for Figure labels."
# We use LaTeX rendering with STIX fonts (Times-compatible) for consistency
# with the paper body.

# Colorblind-safe palette (Tol bright, reordered for aesthetics)
C_PRIMARY = "#911a4b"  # burgundy   -- main model / positive class
C_SECONDARY = "#0d1c24"  # dark navy -- baseline / negative class
C_TERTIARY = "#5a7d8c"  # slate blue  -- ideal lines / accents
C_NEUTRAL = "#BBBBBB"  # light grey  -- gridlines, reference
C_HIGHLIGHT = "#a89882"  # warm taupe  -- emphasis annotations
C_DARK = "#222222"  # near-black  -- text, spines

PALETTE = [C_PRIMARY, C_SECONDARY, C_TERTIARY, C_HIGHLIGHT, "#7a9a82", "#66CCEE"]

# Human-readable feature labels for plots
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
    "avg_files_per_partition": "Avg files/partition",
    "max_files_per_partition": "Max files/partition",
    "min_files_per_partition": "Min files/partition",
    "stddev_files_per_partition": "Stddev files/partition",
    "num_snapshots": "Snapshot count",
    "small_file_ratio": "Small-file ratio",
    "file_size_cv": "File size CV",
    "files_per_partition_cv": "Files/partition CV",
}


def _feature_label(name: str) -> str:
    return _FEATURE_LABELS.get(name, name.replace("_", " ").title())


def _apply_theme() -> None:
    """Set matplotlib rcParams for IEEE-compatible publication plots."""
    plt.rcParams.update(
        {
            # -- TeX rendering (matches paper fonts) --
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["STIX", "STIXGeneral", "Times", "Nimbus Roman"],
            "mathtext.fontset": "stix",
            # -- Sizing (8pt labels per IEEE guide) --
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "legend.title_fontsize": 8,
            # -- Colours & spines --
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
            # -- Figure --
            "figure.facecolor": "white",
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
            "savefig.transparent": False,
            # -- Legend --
            "legend.frameon": True,
            "legend.framealpha": 0.9,
            "legend.edgecolor": "#CCCCCC",
            "legend.fancybox": False,
        }
    )


def _despine(ax: plt.Axes, left: bool = False) -> None:
    """Remove top and right spines for a cleaner look."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if left:
        ax.spines["left"].set_visible(False)


def _annotate_bars(
    ax: plt.Axes, bars, fmt: str = "{:.3f}", fontsize: int = 6, offset: float = 0.01
) -> None:
    """Add value labels above bars."""
    for bar in bars:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + offset,
            fmt.format(h),
            ha="center",
            va="bottom",
            fontsize=fontsize,
            color=C_DARK,
        )


# Column width for IEEE 2-column: ~3.5 in.  Full width: ~7.16 in.
COL_W = 3.5  # single-column figure width (inches)
FULL_W = 7.16  # double-column figure width (inches)


# ── Plot functions ──────────────────────────────────────────────────────────


def plot_roc_curve(
    clf: XGBClassifier, X_test: pd.DataFrame, y_test: pd.Series, out: Path
) -> None:
    y_score = clf.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(COL_W, COL_W * 0.6))

    # Shaded area under curve
    ax.fill_between(fpr, tpr, alpha=0.15, color=C_PRIMARY)
    ax.plot(fpr, tpr, color=C_PRIMARY, linewidth=1.5,
            label=f"XGBoost (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], color=C_NEUTRAL, linewidth=0.8,
            linestyle="--", label="Random classifier")

    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend(loc="lower right")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.05)
    _despine(ax)

    fig.savefig(out)
    plt.close(fig)
    LOG.info("Saved %s", out)


def plot_feature_importance(
    importances: list[dict], out: Path, top_n: int = 10
) -> None:
    top = importances[:top_n]
    features = [_feature_label(d["feature"]) for d in reversed(top)]
    values = [d["importance"] for d in reversed(top)]

    fig, ax = plt.subplots(figsize=(COL_W, COL_W * 0.6))

    bars = ax.barh(
        features, values,
        color=C_PRIMARY, edgecolor="white", linewidth=0.3, height=0.65,
    )
    # Subtle value labels at bar tips
    for bar in bars:
        w = bar.get_width()
        ax.text(
            w + max(values) * 0.02, bar.get_y() + bar.get_height() / 2,
            f"{w:.3f}", va="center", fontsize=6, color=C_DARK,
        )

    ax.set_xlabel("Importance (gain)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.invert_yaxis()  # highest at top (already reversed)

    fig.savefig(out)
    plt.close(fig)
    LOG.info("Saved %s", out)


def plot_regression_scatter(
    reg: XGBRegressor, X_test: pd.DataFrame, y_test: pd.Series, out: Path
) -> None:
    y_pred = reg.predict(X_test)

    fig, ax = plt.subplots(figsize=(COL_W, COL_W * 0.6))

    ax.scatter(
        y_test, y_pred,
        s=14, alpha=0.65, color=C_PRIMARY, edgecolors=C_DARK,
        linewidths=0.2, zorder=3,
    )

    lo = min(y_test.min(), y_pred.min()) - 0.03
    hi = max(y_test.max(), y_pred.max()) + 0.03
    ax.plot([lo, hi], [lo, hi], color=C_SECONDARY, linewidth=1.0,
            linestyle="--", label="Ideal", zorder=2)

    ax.set_xlabel("Actual file reduction ratio")
    ax.set_ylabel("Predicted file reduction ratio")
    ax.legend(loc="upper left", framealpha=0.9)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    _despine(ax)

    fig.savefig(out)
    plt.close(fig)
    LOG.info("Saved %s", out)


def plot_label_distribution(df: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(COL_W, COL_W * 0.6))

    n, bins, patches = ax.hist(
        df["file_reduction_ratio"],
        bins=30, edgecolor="white", linewidth=0.4,
        color=C_PRIMARY, alpha=0.85, zorder=3,
    )

    ax.set_xlabel("File reduction ratio")
    ax.set_ylabel("Number of tables")
    _despine(ax)

    fig.savefig(out)
    plt.close(fig)
    LOG.info("Saved %s", out)


def plot_model_comparison_clf(metrics: dict, out: Path) -> None:
    """Grouped bar chart comparing classifier metrics."""
    models = {}
    for model_name, model_data in metrics["classifier"].items():
        models[model_name] = model_data["metrics"]

    compare_metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1", "ROC AUC"]
    x = np.arange(len(compare_metrics))
    model_names = list(models.keys())
    n_models = len(model_names)
    width = 0.7 / n_models

    fig, ax = plt.subplots(figsize=(FULL_W, FULL_W * 0.3))

    for i, name in enumerate(model_names):
        vals = [models[name].get(m, 0) for m in compare_metrics]
        offset = (i - (n_models - 1) / 2) * width
        bars = ax.bar(
            x + offset, vals, width * 0.92,
            label=name.replace("_", " ").title(),
            color=PALETTE[i], edgecolor="white", linewidth=0.4, zorder=3,
        )
        _annotate_bars(ax, bars, fmt="{:.3f}", fontsize=5.5, offset=0.008)

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score")
    ax.legend(loc="lower right")
    _despine(ax)

    fig.savefig(out)
    plt.close(fig)
    LOG.info("Saved %s", out)


def plot_model_comparison_reg(metrics: dict, out: Path) -> None:
    """Side-by-side panels comparing regressor metrics."""
    models = {}
    for model_name, model_data in metrics["regressor"].items():
        models[model_name] = model_data["metrics"]

    model_names = list(models.keys())
    metric_names = ["rmse", "mae", "r2"]
    titles = [r"RMSE $\downarrow$", r"MAE $\downarrow$", r"$R^2$ $\uparrow$"]

    fig, axes = plt.subplots(1, 3, figsize=(FULL_W, FULL_W * 0.3))

    for ax, metric, title in zip(axes, metric_names, titles):
        vals = [models[m][metric] for m in model_names]
        pretty_names = [n.replace("_", " ").title() for n in model_names]
        bars = ax.bar(
            pretty_names, vals,
            color=PALETTE[: len(model_names)],
            edgecolor="white", linewidth=0.4, width=0.55, zorder=3,
        )
        _annotate_bars(ax, bars, fmt="{:.4f}", fontsize=5.5, offset=max(vals) * 0.02)
        ax.set_title(title)
        _despine(ax)

    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    LOG.info("Saved %s", out)


def plot_cost_scatter(df: pd.DataFrame, out: Path) -> None:
    """Scatter plot of file reduction ratio vs compaction wall-clock time.

    Colours points by whether compaction rewrote any files, providing a
    visual cost-benefit analysis: high duration + low reduction = wasted
    compute.
    """
    fig, ax = plt.subplots(figsize=(COL_W, COL_W * 0.6))

    rewrote = df["rewritten_data_files_count"] > 0
    # Plot non-compacted (no rewrite) first so compacted dots are on top
    ax.scatter(
        df.loc[~rewrote, "compaction_duration_s"],
        df.loc[~rewrote, "file_reduction_ratio"],
        s=12, alpha=0.55, color=C_SECONDARY, edgecolors=C_DARK,
        linewidths=0.2, zorder=2, label="No files rewritten",
    )
    ax.scatter(
        df.loc[rewrote, "compaction_duration_s"],
        df.loc[rewrote, "file_reduction_ratio"],
        s=12, alpha=0.55, color=C_PRIMARY, edgecolors=C_DARK,
        linewidths=0.2, zorder=3, label="Files rewritten",
    )

    ax.set_xlabel("Compaction wall-clock time (s)")
    ax.set_ylabel("File reduction ratio")
    ax.legend(loc="center right", framealpha=0.9)
    _despine(ax)

    fig.savefig(out)
    plt.close(fig)
    LOG.info("Saved %s", out)


def plot_feature_violins(df: pd.DataFrame, out: Path) -> None:
    """Ridge plot of the 17 input features, min-max scaled to [0,1]."""
    from scipy.stats import gaussian_kde

    feats = FEATURE_COLUMNS
    log_feats = {
        "file_count", "total_size_bytes", "avg_file_size_bytes",
        "min_file_size_bytes", "max_file_size_bytes", "stddev_file_size_bytes",
        "total_records", "avg_records_per_file",
        "avg_files_per_partition", "max_files_per_partition",
    }

    n = len(feats)
    row_h = 0.40
    overlap = 0.50

    fig, ax = plt.subplots(figsize=(COL_W, n * row_h * (1 - overlap) + 0.7))

    x_grid = np.linspace(0, 1, 300)  # shared normalised x-axis

    for i, feat in enumerate(feats):
        vals = df[feat].dropna().values.astype(float)
        use_log = feat in log_feats
        if use_log:
            vals = np.log10(vals + 1)

        if vals.std() == 0 or len(vals) < 3:
            continue

        # Min-max normalise to [0, 1]
        vmin, vmax = vals.min(), vals.max()
        vals_norm = (vals - vmin) / (vmax - vmin)

        kde = gaussian_kde(vals_norm, bw_method=0.25)
        density = kde(x_grid)
        density = density / density.max() * row_h  # peak = row_h

        baseline = i * row_h * (1 - overlap)
        ax.fill_between(x_grid, baseline, baseline + density,
                         alpha=0.6, color=PALETTE[i % len(PALETTE)],
                         edgecolor="white", linewidth=0.3, zorder=n - i)
        ax.plot(x_grid, baseline + density,
                color=C_DARK, linewidth=0.4, zorder=n - i + 1)

    # Y-axis: feature labels
    yticks = [i * row_h * (1 - overlap) + row_h * 0.20 for i in range(n)]
    ylabels = []
    for feat in feats:
        lbl = _feature_label(feat)
        if feat in log_feats:
            lbl += r"*"
        ylabels.append(lbl)

    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=5)
    ax.set_ylim(-0.03, n * row_h * (1 - overlap) + row_h * 0.4)
    ax.set_xlim(-0.02, 1.02)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0", "0.25", "0.5", "0.75", "1"], fontsize=6)
    ax.set_xlabel(r"Min--max scaled value \quad (* = $\log_{10}$ transformed)",
                  fontsize=6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)

    fig.tight_layout(pad=0.3)
    fig.savefig(out)
    plt.close(fig)
    LOG.info("Saved %s", out)


# ── CLI ─────────────────────────────────────────────────────────────────────


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate models and generate plots")
    p.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    p.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    p.add_argument("--clf-model", type=Path, default=DEFAULT_CLF_MODEL)
    p.add_argument("--reg-model", type=Path, default=DEFAULT_REG_MODEL)
    p.add_argument("--plots-dir", type=Path, default=DEFAULT_PLOTS_DIR)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--test-size", type=float, default=0.2)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    _apply_theme()

    # Load data
    df = pd.read_csv(args.dataset)
    metrics = json.loads(args.metrics.read_text())
    LOG.info("Dataset: %d rows, Metrics seed=%d", len(df), metrics["seed"])

    # Reproduce the same train/test split
    X = df[FEATURE_COLUMNS]
    y_clf = df["needs_compaction"]
    y_reg = df["file_reduction_ratio"]

    _, X_test, _, y_clf_test, _, y_reg_test = train_test_split(
        X, y_clf, y_reg,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=y_clf,
    )

    # Load models
    clf = XGBClassifier()
    clf.load_model(str(args.clf_model))

    reg = XGBRegressor()
    reg.load_model(str(args.reg_model))

    # Generate plots
    out = args.plots_dir
    out.mkdir(parents=True, exist_ok=True)

    plot_roc_curve(clf, X_test, y_clf_test, out / "roc_curve.pdf")
    plot_feature_importance(
        metrics["classifier"]["xgboost"]["feature_importance"],
        out / "feature_importance_clf.pdf",
    )
    plot_feature_importance(
        metrics["regressor"]["xgboost"]["feature_importance"],
        out / "feature_importance_reg.pdf",
    )
    plot_regression_scatter(reg, X_test, y_reg_test, out / "regression_scatter.pdf")
    plot_label_distribution(df, out / "label_distribution.pdf")
    plot_model_comparison_clf(metrics, out / "model_comparison_clf.pdf")
    plot_model_comparison_reg(metrics, out / "model_comparison_reg.pdf")
    plot_feature_violins(df, out / "feature_violins.pdf")
    plot_cost_scatter(df, out / "cost_scatter.pdf")

    # Print summary
    LOG.info("=== Summary ===")
    for model_name, model_data in metrics["classifier"].items():
        m = model_data["metrics"]
        LOG.info(
            "Classifier [%s]: accuracy=%.4f, f1=%.4f, AUC=%.4f",
            model_name, m["accuracy"], m["f1"], m["roc_auc"],
        )
    for model_name, model_data in metrics["regressor"].items():
        m = model_data["metrics"]
        LOG.info(
            "Regressor [%s]: RMSE=%.4f, MAE=%.4f, R2=%.4f",
            model_name, m["rmse"], m["mae"], m["r2"],
        )
    LOG.info("Plots saved to %s/", out)


if __name__ == "__main__":
    main()
