#!/usr/bin/env python3
"""
Join pre-compaction features with post-compaction metrics and derive
training labels for the ML model.

Inputs:
    data/features.csv     – 24 columns of pre-compaction metadata
    data/compaction.csv   – 7 columns of post-compaction metrics

Output:
    data/dataset.csv      – all feature columns + derived label columns

Label columns:
    needs_compaction (int 0/1)
        1 if Iceberg actually rewrote files, 0 otherwise.
        This is the primary classification target.

    file_reduction_ratio (float 0–1)
        (before_file_count − after_file_count) / before_file_count.
        Regression target: fraction of files eliminated.

    size_change_ratio (float, can be negative)
        (after_total_size − before_total_size) / before_total_size.
        Positive = bloat, negative = savings.

    files_rewritten_ratio (float ≥ 0)
        rewritten_data_files_count / before_file_count.
        Measures compaction effort relative to table size.

Usage:
    python code/label.py
    python code/label.py --features data/features.csv --compaction data/compaction.csv --out data/dataset.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

LOG = logging.getLogger("label")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)-5s  %(message)s",
)

# ── Defaults ────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_FEATURES = ROOT / "data" / "features.csv"
DEFAULT_COMPACTION = ROOT / "data" / "compaction.csv"
DEFAULT_OUT = ROOT / "data" / "dataset.csv"

# Columns from features.csv that are generation parameters
# (not observable in production) vs. actual metadata features.
# We keep both in the dataset file but mark which are which so
# train.py can drop the config params from X.
CONFIG_PARAM_COLUMNS = [
    "num_rows",
    "num_columns",
    "num_partitions_cfg",
    "num_writers",
    "num_write_batches",
    "file_size_target_kb",
    "partition_skew",
]

LABEL_COLUMNS = [
    "needs_compaction",
    "file_reduction_ratio",
    "size_change_ratio",
    "files_rewritten_ratio",
]


# ── Core logic ──────────────────────────────────────────────────────────────


def build_dataset(
    features_path: Path,
    compaction_path: Path,
) -> pd.DataFrame:
    """Join features + compaction and derive labels."""
    feat = pd.read_csv(features_path)
    comp = pd.read_csv(compaction_path)

    LOG.info("Features: %d rows, %d cols", len(feat), len(feat.columns))
    LOG.info("Compaction: %d rows, %d cols", len(comp), len(comp.columns))

    df = feat.merge(comp, on="config_id", how="inner")
    if len(df) != len(feat):
        LOG.warning(
            "Inner join dropped rows: features=%d, compaction=%d, joined=%d",
            len(feat),
            len(comp),
            len(df),
        )

    # ── Derive labels ───────────────────────────────────────────────────
    # Binary: did compaction do anything?
    df["needs_compaction"] = (df["rewritten_data_files_count"] > 0).astype(int)

    # File reduction ratio (0 = no change, 1 = maximally reduced)
    df["file_reduction_ratio"] = (
        (df["file_count"] - df["after_file_count"]) / df["file_count"]
    ).clip(lower=0.0)

    # Size change ratio (negative = savings)
    df["size_change_ratio"] = (
        (df["after_total_size_bytes"] - df["total_size_bytes"])
        / df["total_size_bytes"]
    )

    # Effort ratio: how many files rewritten per original file
    df["files_rewritten_ratio"] = (
        df["rewritten_data_files_count"] / df["file_count"]
    )

    # ── Summary stats ───────────────────────────────────────────────────
    n_pos = df["needs_compaction"].sum()
    n_neg = len(df) - n_pos
    LOG.info(
        "Label distribution: needs_compaction=1: %d (%.1f%%), 0: %d (%.1f%%)",
        n_pos,
        100 * n_pos / len(df),
        n_neg,
        100 * n_neg / len(df),
    )
    LOG.info(
        "file_reduction_ratio: mean=%.3f, median=%.3f, max=%.3f",
        df["file_reduction_ratio"].mean(),
        df["file_reduction_ratio"].median(),
        df["file_reduction_ratio"].max(),
    )
    LOG.info(
        "size_change_ratio: mean=%.4f, median=%.4f",
        df["size_change_ratio"].mean(),
        df["size_change_ratio"].median(),
    )

    return df


# ── CLI ─────────────────────────────────────────────────────────────────────


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build labelled dataset")
    p.add_argument(
        "--features",
        type=Path,
        default=DEFAULT_FEATURES,
        help="Path to features.csv",
    )
    p.add_argument(
        "--compaction",
        type=Path,
        default=DEFAULT_COMPACTION,
        help="Path to compaction.csv",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help="Output dataset CSV path",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    df = build_dataset(args.features, args.compaction)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    LOG.info("Wrote %d rows × %d cols → %s", len(df), len(df.columns), args.out)


if __name__ == "__main__":
    main()
