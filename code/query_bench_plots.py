#!/usr/bin/env python3
"""Generate query benchmark plots for the paper."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── Theme (shared with ablation_plots.py) ───────────────────────────────────
C_PRIMARY = "#911a4b"
C_SECONDARY = "#0d1c24"
C_TERTIARY = "#5a7d8c"
C_NEUTRAL = "#BBBBBB"
C_HIGHLIGHT = "#a89882"
C_DARK = "#222222"

PALETTE = [C_PRIMARY, C_SECONDARY, C_TERTIARY, C_HIGHLIGHT, "#7a9a82", "#66CCEE"]

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

COL_W = 3.5  # IEEE single column width in inches
FULL_W = 7.16

PLOT_DIR = Path(__file__).resolve().parent.parent / "paper" / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)


def _despine(ax, left=False):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if left:
        ax.spines["left"].set_visible(False)


def plot_speedup_by_query(df: pd.DataFrame) -> None:
    """Bar chart of median speedup per query type."""
    query_order = ["count_star", "orders_agg", "tpch_q1",
                   "cust_nation_agg", "part_type_agg"]
    labels = ["COUNT(*)", "orders_agg", "tpch_q1",
              "cust_nation", "part_type"]

    medians = []
    for q in query_order:
        sub = df[df["query"] == q]
        medians.append(sub["speedup"].median() if len(sub) > 0 else 0)

    fig, ax = plt.subplots(figsize=(COL_W, 2.2))
    x = np.arange(len(query_order))
    colors = [C_PRIMARY if m >= 1.0 else C_SECONDARY for m in medians]
    bars = ax.bar(x, medians, color=colors, edgecolor="white", linewidth=0.5, width=0.6)

    ax.axhline(y=1.0, color=C_DARK, linestyle="--", linewidth=0.8, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("Median Speedup")
    ax.set_ylim(0, max(medians) * 1.25)

    # Annotate bars
    for bar, m in zip(bars, medians):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.03,
                f"{m:.2f}x", ha="center", va="bottom", fontsize=7,
                color=C_DARK)

    _despine(ax)
    fig.tight_layout()

    out = PLOT_DIR / "query_speedup.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def plot_speedup_vs_files(df: pd.DataFrame) -> None:
    """Scatter plot of speedup vs pre-compaction file count, colored by query."""
    fig, ax = plt.subplots(figsize=(COL_W, 2.5))

    query_colors = {
        "count_star": C_PRIMARY,
        "tpch_q1": C_SECONDARY,
        "orders_agg": C_TERTIARY,
        "cust_nation_agg": C_HIGHLIGHT,
        "part_type_agg": "#AA3377",
    }
    query_labels = {
        "count_star": "COUNT(*)",
        "tpch_q1": "tpch_q1",
        "orders_agg": "orders_agg",
        "cust_nation_agg": "cust_nation",
        "part_type_agg": "part_type",
    }

    for q, color in query_colors.items():
        sub = df[df["query"] == q]
        if len(sub) == 0:
            continue
        ax.scatter(sub["file_count_pre"], sub["speedup"],
                   c=color, s=18, alpha=0.8, label=query_labels[q],
                   edgecolors=C_DARK, linewidths=0.3)

    ax.axhline(y=1.0, color=C_DARK, linestyle="--", linewidth=0.8, alpha=0.7)
    ax.set_xlabel("Pre-compaction file count")
    ax.set_ylabel("Speedup")
    ax.set_xscale("log")
    ax.legend(loc="upper left", framealpha=0.9, fontsize=6)
    _despine(ax)
    fig.tight_layout()

    out = PLOT_DIR / "query_speedup_scatter.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def main() -> None:
    csv_path = Path("data/tpch/tpch_query_bench.csv")
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} measurements from {csv_path}")

    plot_speedup_by_query(df)
    plot_speedup_vs_files(df)

    print("Done.")


if __name__ == "__main__":
    main()
