#!/usr/bin/env python3
"""
Run Iceberg compaction on generated tables and record before/after metrics.

Calls ``rewrite_data_files`` on each table and captures:
  - before / after file count, total size, avg file size
  - number of files rewritten, added
  - wall-clock compaction duration

The output CSV is later joined with features.csv to produce training
labels (was compaction worthwhile?).

Usage:
    python code/compact.py                                  # all configs
    python code/compact.py --sample 5                       # 5 random
    python code/compact.py --config-id abc12345             # single
    python code/compact.py --out data/compaction.csv        # output path
"""

from __future__ import annotations

import argparse
import csv
import logging
import random as rand_mod
import sys
import time
from pathlib import Path
from typing import Any

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from params import SimConfig, full_grid  # noqa: E402

LOG = logging.getLogger("compact")

# ── Defaults ────────────────────────────────────────────────────────────────

DEFAULT_WAREHOUSE = "/mnt/data/warehouse"
CATALOG = "local"
DATABASE = "compaction"

# Iceberg default target file size (128 MB).  We use this as the target
# for rewrite_data_files so it mirrors what a production system would do.
DEFAULT_TARGET_MB = 128
REWRITE_TARGET_BYTES = DEFAULT_TARGET_MB * 1024 * 1024  # 128 MB

OUTPUT_COLUMNS = [
    "config_id",
    # After compaction
    "after_file_count",
    "after_total_size_bytes",
    "after_avg_file_size_bytes",
    # Compaction procedure output
    "rewritten_data_files_count",
    "added_data_files_count",
    "compaction_duration_s",
]


# ── Spark session ───────────────────────────────────────────────────────────


def get_spark(warehouse: str) -> SparkSession:
    """Build a local SparkSession with Iceberg hadoop catalog."""
    return (
        SparkSession.builder.master("local[1]")
        .appName("iceberg-compact")
        .config("spark.driver.memory", "8g")
        .config(
            "spark.sql.extensions",
            "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions",
        )
        .config(
            f"spark.sql.catalog.{CATALOG}",
            "org.apache.iceberg.spark.SparkCatalog",
        )
        .config(f"spark.sql.catalog.{CATALOG}.type", "hadoop")
        .config(f"spark.sql.catalog.{CATALOG}.warehouse", warehouse)
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.sql.iceberg.vectorization.enabled", "false")
        .getOrCreate()
    )


# ── Helpers ─────────────────────────────────────────────────────────────────


def _fqn(table_name: str) -> str:
    return f"{CATALOG}.{DATABASE}.{table_name}"


def _file_stats(spark: SparkSession, fqn: str) -> dict[str, Any]:
    """Collect file-level aggregates from Iceberg metadata."""
    row = (
        spark.sql(f"SELECT * FROM {fqn}.files")
        .agg(
            F.count("*").alias("file_count"),
            F.sum("file_size_in_bytes").alias("total_size_bytes"),
            F.avg("file_size_in_bytes").alias("avg_file_size_bytes"),
        )
        .collect()[0]
    )
    return {
        "file_count": row["file_count"],
        "total_size_bytes": int(row["total_size_bytes"] or 0),
        "avg_file_size_bytes": round(row["avg_file_size_bytes"] or 0, 2),
    }


def table_exists(spark: SparkSession, cfg: SimConfig) -> bool:
    """Check if a table exists."""
    fqn = _fqn(f"t_{cfg.config_id}")
    try:
        spark.sql(f"SELECT 1 FROM {fqn} LIMIT 1").collect()
        return True
    except Exception:
        return False


# ── Compaction ──────────────────────────────────────────────────────────────


def compact_table(
    spark: SparkSession, cfg: SimConfig,
    rewrite_target_bytes: int = REWRITE_TARGET_BYTES,
) -> dict[str, Any]:
    """Run rewrite_data_files on one table and return metrics."""
    table_name = f"t_{cfg.config_id}"
    fqn = _fqn(table_name)

    # ── Run compaction ──────────────────────────────────────────────────
    t0 = time.time()
    result_df = spark.sql(f"""
        CALL {CATALOG}.system.rewrite_data_files(
            table => '{DATABASE}.{table_name}',
            options => map(
                'target-file-size-bytes', '{rewrite_target_bytes}',
                'min-file-size-bytes',    '{int(rewrite_target_bytes * 0.75)}',
                'max-file-size-bytes',    '{int(rewrite_target_bytes * 1.8)}'
            )
        )
    """)
    result_row = result_df.collect()[0]
    duration = time.time() - t0

    rewritten = result_row["rewritten_data_files_count"]
    added = result_row["added_data_files_count"]

    # ── After snapshot ──────────────────────────────────────────────────
    after = _file_stats(spark, fqn)

    LOG.info(
        "  %s: %d files, %.1f MB, rewritten=%d, added=%d  (%.1fs)",
        table_name,
        after["file_count"],
        after["total_size_bytes"] / 1e6,
        rewritten,
        added,
        duration,
    )

    return {
        "config_id": cfg.config_id,
        "after_file_count": after["file_count"],
        "after_total_size_bytes": after["total_size_bytes"],
        "after_avg_file_size_bytes": after["avg_file_size_bytes"],
        "rewritten_data_files_count": rewritten,
        "added_data_files_count": added,
        "compaction_duration_s": round(duration, 2),
    }


# ── Config loading ──────────────────────────────────────────────────────────


def load_configs_from_csv(csv_path: Path) -> list[SimConfig]:
    """Read SimConfig objects from the grid CSV produced by params.py."""
    configs: list[SimConfig] = []
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            configs.append(
                SimConfig(
                    num_rows=int(row["num_rows"]),
                    num_columns=int(row["num_columns"]),
                    num_partitions=int(row["num_partitions"]),
                    num_writers=int(row["num_writers"]),
                    num_write_batches=int(row["num_write_batches"]),
                    file_size_target_kb=int(row["file_size_target_kb"]),
                    partition_skew=row["partition_skew"],
                )
            )
    return configs


# ── CLI ─────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run Iceberg compaction and record before/after metrics"
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="code/grid.csv",
        help="Path to grid CSV (default: code/grid.csv)",
    )
    parser.add_argument(
        "--warehouse",
        type=str,
        default=DEFAULT_WAREHOUSE,
        help=f"Iceberg warehouse path (default: {DEFAULT_WAREHOUSE})",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="data/compaction.csv",
        help="Output CSV path (default: data/compaction.csv)",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=0,
        help="Compact only N random configs (0 = all)",
    )
    parser.add_argument(
        "--config-id",
        type=str,
        default=None,
        help="Compact only this config ID",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for sampling"
    )
    parser.add_argument(
        "--target-mb", type=int, default=DEFAULT_TARGET_MB,
        help=f"Compaction target file size in MB (default: {DEFAULT_TARGET_MB})",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip config IDs already present in --out CSV and append new results",
    )
    parser.add_argument(
        "--skip",
        type=str,
        default="",
        help="Comma-separated config IDs to skip (known-crashing tables)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s  %(message)s",
    )

    # ── Load configs ────────────────────────────────────────────────────
    csv_path = Path(args.csv)
    if csv_path.exists():
        configs = load_configs_from_csv(csv_path)
        LOG.info("Loaded %d configs from %s", len(configs), csv_path)
    else:
        configs = full_grid()
        LOG.info(
            "CSV not found at %s; using full_grid() (%d configs)",
            csv_path,
            len(configs),
        )

    # ── Filter ──────────────────────────────────────────────────────────
    if args.config_id:
        configs = [c for c in configs if c.config_id == args.config_id]
        if not configs:
            LOG.error("Config ID %s not found in grid", args.config_id)
            return 1
    elif args.sample > 0:
        rng = rand_mod.Random(args.seed)
        configs = rng.sample(configs, min(args.sample, len(configs)))

    # ── Resume: skip already-done configs ────────────────────────────
    done_ids: set[str] = set()
    out_path = Path(args.out)
    if args.resume and out_path.exists():
        with out_path.open() as f:
            for row in csv.DictReader(f):
                done_ids.add(row["config_id"])
        before = len(configs)
        configs = [c for c in configs if c.config_id not in done_ids]
        LOG.info(
            "Resume: %d already done, %d remaining (was %d)",
            len(done_ids), len(configs), before,
        )

    # ── Skip known-crashing configs ─────────────────────────────────
    if args.skip:
        skip_ids = {s.strip() for s in args.skip.split(",") if s.strip()}
        before = len(configs)
        configs = [c for c in configs if c.config_id not in skip_ids]
        LOG.info("Skipping %d known-crashing configs, %d remaining",
                 before - len(configs), len(configs))

    LOG.info("Selected %d configs for compaction", len(configs))

    # ── Compact ─────────────────────────────────────────────────────────
    rewrite_target = args.target_mb * 1024 * 1024
    LOG.info("Warehouse: %s", args.warehouse)
    LOG.info("Rewrite target: %d bytes (%.0f MB)", rewrite_target,
             rewrite_target / 1e6)
    spark = get_spark(args.warehouse)

    compacted = 0
    skipped = 0
    failed = 0

    def _write_header_if_needed() -> None:
        """Write CSV header if starting fresh (not resuming)."""
        if not (args.resume and done_ids):
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with out_path.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
                writer.writeheader()

    def _append_result(row: dict[str, Any]) -> None:
        """Append a single result row to CSV immediately."""
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
            writer.writerow(row)

    _write_header_if_needed()

    try:
        for i, cfg in enumerate(configs, 1):
            LOG.info("=== [%d/%d] config %s ===", i, len(configs), cfg.config_id)

            if not table_exists(spark, cfg):
                LOG.warning("  SKIP (table does not exist)")
                skipped += 1
                continue

            try:
                metrics = compact_table(spark, cfg, rewrite_target_bytes=rewrite_target)
                _append_result(metrics)
                compacted += 1
            except Exception:
                LOG.exception("FAILED config %s", cfg.config_id)
                failed += 1

        # ── Summary ─────────────────────────────────────────────────────
        LOG.info("=== Summary ===")
        LOG.info(
            "Compacted: %d  |  Skipped: %d  |  Failed: %d  |  Total: %d",
            compacted,
            skipped,
            failed,
            len(configs),
        )

        return 1 if failed else 0
    finally:
        try:
            spark.stop()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
