#!/usr/bin/env python3
"""
W4 response: deeper regression analysis.

1. Per-quantile error analysis  -- RMSE/MAE across r_fc bins
2. Bootstrap CI on marginal ΔR² -- k=2 vs k=17 with 1000 bootstraps
3. Cost-benefit threshold       -- I/O break-even analysis

Outputs:
    data/w4_results.json
    paper/tables/quantile_errors.tex
    paper/plots/quantile_errors.pdf

Usage:
    python code/w4_analysis.py
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
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

LOG = logging.getLogger("w4_analysis")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)-5s  %(message)s",
)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = ROOT / "data" / "dataset.csv"
DEFAULT_OUT_DIR = ROOT / "data"
DEFAULT_PLOTS_DIR = ROOT / "paper" / "plots"
DEFAULT_TABLES_DIR = ROOT / "paper" / "tables"

FEATURE_COLUMNS = [
    "file_count", "total_size_bytes", "avg_file_size_bytes",
    "min_file_size_bytes", "max_file_size_bytes", "stddev_file_size_bytes",
    "total_records", "avg_records_per_file", "num_partitions_actual",
    "avg_files_per_partition", "max_files_per_partition",
    "min_files_per_partition", "stddev_files_per_partition",
    "num_snapshots", "small_file_ratio", "file_size_cv",
    "files_per_partition_cv",
]

# ── Theme (matches evaluate.py / ablation_plots.py) ────────────────────────

C_PRIMARY = "#911a4b"
C_SECONDARY = "#0d1c24"
C_TERTIARY = "#5a7d8c"
C_NEUTRAL = "#BBBBBB"
C_DARK = "#222222"
COL_W = 3.5


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


def _make_xgb_reg(seed: int, **overrides) -> XGBRegressor:
    params = dict(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
        random_state=seed, eval_metric="rmse",
    )
    params.update(overrides)
    return XGBRegressor(**params)


# ── 1. Per-quantile error analysis ─────────────────────────────────────────


def quantile_error_analysis(
    X_test: pd.DataFrame, y_test: np.ndarray, y_pred: np.ndarray,
) -> list[dict]:
    """Compute RMSE, MAE, and max|error| in bins of r_fc."""
    # Bins: [0,0], (0,0.5], (0.5,0.8], (0.8,0.95], (0.95,1.0]
    bins = [(-0.001, 0.001), (0.001, 0.5), (0.5, 0.8), (0.8, 0.95), (0.95, 1.001)]
    labels = ["$r_{fc}=0$", "$(0,\\,0.5]$", "$(0.5,\\,0.8]$",
              "$(0.8,\\,0.95]$", "$(0.95,\\,1]$"]

    results = []
    for (lo, hi), label in zip(bins, labels):
        mask = (y_test >= lo) & (y_test <= hi)
        n = int(mask.sum())
        if n == 0:
            continue
        yt = y_test[mask]
        yp = y_pred[mask]
        rmse = float(np.sqrt(mean_squared_error(yt, yp)))
        mae = float(mean_absolute_error(yt, yp))
        max_err = float(np.max(np.abs(yt - yp)))
        # Relative error (only meaningful when true > 0)
        pos_mask = yt > 0.01
        if pos_mask.sum() > 0:
            mape = float(np.mean(np.abs(yt[pos_mask] - yp[pos_mask]) / yt[pos_mask]))
        else:
            mape = None
        results.append({
            "bin": label,
            "lo": lo, "hi": hi, "n": n,
            "rmse": rmse, "mae": mae, "max_abs_error": max_err,
            "mape": mape,
        })
        LOG.info("  Bin %s (n=%d): RMSE=%.4f, MAE=%.4f, max|e|=%.4f",
                 label, n, rmse, mae, max_err)
    return results


# ── 2. Bootstrap CI on marginal ΔR² ────────────────────────────────────────


def bootstrap_marginal_r2(
    df: pd.DataFrame, seed: int, test_size: float,
    n_boot: int = 1000,
) -> dict:
    """Bootstrap CI on R²(k=17) - R²(k=2)."""
    LOG.info("=== Bootstrap marginal ΔR² (n=%d) ===", n_boot)

    # Load importance order from ablation_metrics.json
    ablation_path = ROOT / "data" / "ablation_metrics.json"
    ablation = json.loads(ablation_path.read_text())
    importance_order = ablation["feature_ablation"]["importance_order"]
    top2 = importance_order[:2]

    y_reg = df["file_reduction_ratio"]
    y_clf = df["needs_compaction"]
    X_full = df[FEATURE_COLUMNS]
    X_top2 = df[top2]

    delta_r2s = []
    r2_full_vals = []
    r2_top2_vals = []

    rng = np.random.RandomState(seed)

    for i in range(n_boot):
        s = rng.randint(0, 2**31)
        X_tr_f, X_te_f, yr_tr, yr_te = train_test_split(
            X_full, y_reg, test_size=test_size, random_state=s,
        )
        X_tr_2 = X_tr_f[top2]
        X_te_2 = X_te_f[top2]

        # Full model (use fewer trees for bootstrap speed)
        reg_full = _make_xgb_reg(s, n_estimators=100)
        reg_full.fit(X_tr_f, yr_tr, verbose=False)
        r2_full = r2_score(yr_te, reg_full.predict(X_te_f))

        # Top-2 model
        reg_top2 = _make_xgb_reg(s, n_estimators=100)
        reg_top2.fit(X_tr_2, yr_tr, verbose=False)
        r2_top2 = r2_score(yr_te, reg_top2.predict(X_te_2))

        delta = r2_full - r2_top2
        delta_r2s.append(delta)
        r2_full_vals.append(r2_full)
        r2_top2_vals.append(r2_top2)

        if (i + 1) % 50 == 0:
            LOG.info("  Completed %d / %d bootstraps", i + 1, n_boot)

    delta_arr = np.array(delta_r2s)
    result = {
        "top2_features": top2,
        "n_boot": n_boot,
        "delta_r2": {
            "mean": float(delta_arr.mean()),
            "std": float(delta_arr.std()),
            "ci_lo": float(np.percentile(delta_arr, 2.5)),
            "ci_hi": float(np.percentile(delta_arr, 97.5)),
            "median": float(np.median(delta_arr)),
            "frac_positive": float((delta_arr > 0).mean()),
        },
        "r2_full": {
            "mean": float(np.mean(r2_full_vals)),
            "ci_lo": float(np.percentile(r2_full_vals, 2.5)),
            "ci_hi": float(np.percentile(r2_full_vals, 97.5)),
        },
        "r2_top2": {
            "mean": float(np.mean(r2_top2_vals)),
            "ci_lo": float(np.percentile(r2_top2_vals, 2.5)),
            "ci_hi": float(np.percentile(r2_top2_vals, 97.5)),
        },
    }
    LOG.info("  ΔR² = %.4f [%.4f, %.4f]",
             result["delta_r2"]["mean"],
             result["delta_r2"]["ci_lo"],
             result["delta_r2"]["ci_hi"])
    return result


# ── 3. Cost-benefit threshold ──────────────────────────────────────────────


def cost_benefit_analysis(df: pd.DataFrame) -> dict:
    """Analyse I/O cost vs. reduction benefit."""
    LOG.info("=== Cost-benefit analysis ===")

    # Only consider tables that were compacted (needs_compaction=1)
    compacted = df[df["needs_compaction"] == 1].copy()
    not_compacted = df[df["needs_compaction"] == 0].copy()

    # I/O cost of compaction: bytes read (original files) + bytes written (compacted files)
    # bytes_read ≈ total_size_bytes (before), bytes_written ≈ total_size_bytes * (1 - r_fc)
    # But after compaction, size ≈ total_size_bytes (Parquet doesn't change much)
    # Actually: compaction rewrites files but doesn't change data volume much.
    # The cost is rewriting: rewritten_data_files_count files read + added_data_files_count written
    # We have these columns!

    has_cost_cols = all(c in df.columns for c in [
        "rewritten_data_files_count", "added_data_files_count",
        "after_total_size_bytes", "compaction_duration_s"
    ])

    rfc = compacted["file_reduction_ratio"].values
    total_size = compacted["total_size_bytes"].values
    file_count = compacted["file_count"].values

    # Files eliminated
    files_eliminated = file_count * rfc

    # The "benefit" is proportional to r_fc (more files eliminated = better)
    # The "cost" is proportional to total_size_bytes (data rewritten)
    # Cost-benefit ratio: r_fc / (total_size_bytes / median_total_size)
    # But simpler: at what r_fc is compaction "not worth it"?

    # Compute bytes saved per byte of I/O
    # Approximation: compaction reads all files and writes fewer files,
    # but total bytes are ~unchanged (Parquet data is preserved).
    # The I/O cost ≈ 2 * total_size_bytes (read + write).
    # The benefit is reducing file_count by factor (1 - r_fc).
    # Benefit in metadata terms: eliminated files = file_count * r_fc.

    # For scheduling: a table with r_fc < threshold is not worth compacting
    # because the overhead of reading/writing all data exceeds the marginal
    # metadata improvement.

    # Analyse: for low r_fc tables, what is the absolute file reduction?
    thresholds = [0.05, 0.10, 0.15, 0.20, 0.30, 0.50]
    threshold_analysis = []
    for t in thresholds:
        below = compacted[compacted["file_reduction_ratio"] <= t]
        if len(below) == 0:
            threshold_analysis.append({
                "threshold": t, "n_tables": 0,
            })
            continue

        entry = {
            "threshold": t,
            "n_tables": int(len(below)),
            "pct_of_compacted": float(len(below) / len(compacted) * 100),
            "mean_files_eliminated": float((below["file_count"] * below["file_reduction_ratio"]).mean()),
            "median_files_eliminated": float((below["file_count"] * below["file_reduction_ratio"]).median()),
            "mean_total_size_gb": float(below["total_size_bytes"].mean() / 1e9),
            "mean_rfc": float(below["file_reduction_ratio"].mean()),
        }
        if has_cost_cols:
            entry["mean_duration_s"] = float(below["compaction_duration_s"].mean())
        threshold_analysis.append(entry)
        LOG.info("  r_fc <= %.2f: %d tables (%.1f%%), mean files eliminated=%.1f",
                 t, entry["n_tables"], entry["pct_of_compacted"],
                 entry["mean_files_eliminated"])

    result = {
        "n_compacted": int(len(compacted)),
        "n_not_compacted": int(len(not_compacted)),
        "overall_stats": {
            "mean_rfc": float(rfc.mean()),
            "median_rfc": float(np.median(rfc)),
            "mean_files_eliminated": float(files_eliminated.mean()),
            "median_files_eliminated": float(np.median(files_eliminated)),
        },
        "threshold_analysis": threshold_analysis,
    }
    return result


# ── Outputs ────────────────────────────────────────────────────────────────


def generate_quantile_table(quantiles: list[dict], out: Path) -> None:
    """Write LaTeX table for per-quantile errors."""
    lines = [
        r"\begin{tabular}{lcrrr}",
        r"\toprule",
        r"$r_{\mathrm{fc}}$ bin & $n$ & RMSE & MAE & Max$|e|$ \\",
        r"\midrule",
    ]
    for q in quantiles:
        lines.append(
            f"{q['bin']} & {q['n']} & {q['rmse']:.4f} & {q['mae']:.4f} & {q['max_abs_error']:.3f} \\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    LOG.info("Saved %s", out)


def plot_quantile_errors(quantiles: list[dict], out: Path) -> None:
    """Bar chart of RMSE and MAE per r_fc bin."""
    bins = [q["bin"] for q in quantiles]
    rmses = [q["rmse"] for q in quantiles]
    maes = [q["mae"] for q in quantiles]

    x = np.arange(len(bins))
    w = 0.35

    fig, ax = plt.subplots(figsize=(COL_W, COL_W * 0.55))

    bars1 = ax.bar(x - w/2, rmses, w, color=C_PRIMARY, edgecolor="white",
                   linewidth=0.3, label="RMSE", zorder=3)
    bars2 = ax.bar(x + w/2, maes, w, color=C_SECONDARY, edgecolor="white",
                   linewidth=0.3, label="MAE", zorder=3)

    # Value labels
    for bar in list(bars1) + list(bars2):
        h = bar.get_height()
        if h > 0.001:
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.001,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=5.5,
                    color=C_DARK)

    ax.set_xticks(x)
    ax.set_xticklabels(bins, fontsize=6)
    ax.set_ylabel("Error")
    ax.set_xlabel(r"$r_{\mathrm{fc}}$ bin")
    ax.legend(loc="upper right", fontsize=6)
    _despine(ax)

    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    LOG.info("Saved %s", out)


# ── CLI ────────────────────────────────────────────────────────────────────


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="W4 regression analysis")
    p.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--plots-dir", type=Path, default=DEFAULT_PLOTS_DIR)
    p.add_argument("--tables-dir", type=Path, default=DEFAULT_TABLES_DIR)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--n-boot", type=int, default=50)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    _apply_theme()

    df = pd.read_csv(args.dataset)
    LOG.info("Dataset: %d rows", len(df))

    # Reproduce train/test split
    X = df[FEATURE_COLUMNS]
    y_clf = df["needs_compaction"]
    y_reg = df["file_reduction_ratio"]

    X_train, X_test, _, _, yr_train, yr_test = train_test_split(
        X, y_clf, y_reg,
        test_size=args.test_size, random_state=args.seed, stratify=y_clf,
    )

    # Train regressor
    reg = _make_xgb_reg(args.seed)
    reg.fit(X_train, yr_train, eval_set=[(X_test, yr_test)], verbose=False)
    yr_pred = reg.predict(X_test)

    results = {}

    # 1. Per-quantile error
    LOG.info("=== Per-quantile error analysis ===")
    quantiles = quantile_error_analysis(X_test, yr_test.values, yr_pred)
    results["quantile_errors"] = quantiles

    # 2. Bootstrap marginal ΔR²
    marginal = bootstrap_marginal_r2(df, args.seed, args.test_size, args.n_boot)
    results["marginal_r2"] = marginal

    # 3. Cost-benefit
    cost = cost_benefit_analysis(df)
    results["cost_benefit"] = cost

    # Save results
    out_path = args.out_dir / "w4_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    LOG.info("Saved %s", out_path)

    # Generate outputs
    generate_quantile_table(quantiles, args.tables_dir / "quantile_errors.tex")
    plot_quantile_errors(quantiles, args.plots_dir / "quantile_errors.pdf")

    LOG.info("Done.")


if __name__ == "__main__":
    main()
