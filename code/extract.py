#!/usr/bin/env python3
"""
Extract metadata features from generated Iceberg tables.

Queries Iceberg metadata tables (.files, .snapshots, .partitions) to
build a feature vector per table. These features are what the ML model
sees when deciding whether compaction is worthwhile.

Usage:
    python code/extract.py                                 # all configs
    python code/extract.py --sample 5                      # 5 random
    python code/extract.py --config-id abc12345            # single
    python code/extract.py --out code/features.csv         # output path
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

LOG = logging.getLogger("extract")

# ── Defaults ────────────────────────────────────────────────────────────────

DEFAULT_WAREHOUSE = "/mnt/data/warehouse"
CATALOG = "local"
DATABASE = "compaction"

FEATURE_COLUMNS = [
    "config_id",
    # Table-level
    "num_rows",
    "num_columns",
    "num_partitions_cfg",
    "num_writers",
    "num_write_batches",
    "file_size_target_kb",
    "partition_skew",
    # File-level aggregates
    "file_count",
    "total_size_bytes",
    "avg_file_size_bytes",
    "min_file_size_bytes",
    "max_file_size_bytes",
    "stddev_file_size_bytes",
    "total_records",
    "avg_records_per_file",
    # Partition-level
    "num_partitions_actual",
    "avg_files_per_partition",
    "max_files_per_partition",
    "min_files_per_partition",
    "stddev_files_per_partition",
    # Snapshot-level
    "num_snapshots",
    # Derived ratios
    "small_file_ratio",
    "file_size_cv",
    "files_per_partition_cv",
]


# ── Spark session ───────────────────────────────────────────────────────────


def get_spark(warehouse: str) -> SparkSession:
    """Build a local SparkSession with Iceberg hadoop catalog."""
    return (
        SparkSession.builder.master("local[1]")
        .appName("iceberg-feature-extract")
        .config("spark.driver.memory", "4g")
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
        .getOrCreate()
    )


# ── Feature extraction ──────────────────────────────────────────────────────


def _fqn(table_name: str) -> str:
    return f"{CATALOG}.{DATABASE}.{table_name}"


def extract_features(spark: SparkSession, cfg: SimConfig) -> dict[str, Any]:
    """Extract metadata features from one Iceberg table.

    Returns a flat dict suitable for CSV output.
    """
    table_name = f"t_{cfg.config_id}"
    fqn = _fqn(table_name)

    t0 = time.time()

    # ── File-level features ─────────────────────────────────────────────
    files_df = spark.sql(f"SELECT * FROM {fqn}.files")

    file_stats = files_df.agg(
        F.count("*").alias("file_count"),
        F.sum("file_size_in_bytes").alias("total_size_bytes"),
        F.avg("file_size_in_bytes").alias("avg_file_size_bytes"),
        F.min("file_size_in_bytes").alias("min_file_size_bytes"),
        F.max("file_size_in_bytes").alias("max_file_size_bytes"),
        F.stddev("file_size_in_bytes").alias("stddev_file_size_bytes"),
        F.sum("record_count").alias("total_records"),
        F.avg("record_count").alias("avg_records_per_file"),
    ).collect()[0]

    file_count = file_stats["file_count"]
    total_size = file_stats["total_size_bytes"] or 0
    avg_file_size = file_stats["avg_file_size_bytes"] or 0
    min_file_size = file_stats["min_file_size_bytes"] or 0
    max_file_size = file_stats["max_file_size_bytes"] or 0
    stddev_file_size = file_stats["stddev_file_size_bytes"] or 0
    total_records = file_stats["total_records"] or 0
    avg_records = file_stats["avg_records_per_file"] or 0

    # ── Partition-level features ────────────────────────────────────────
    if cfg.num_partitions > 1:
        # Count files per partition value
        part_stats_df = files_df.select(
            F.expr("partition.part_key").alias("part_key")
        ).groupBy("part_key").agg(
            F.count("*").alias("files_in_part")
        )

        part_agg = part_stats_df.agg(
            F.count("*").alias("num_partitions_actual"),
            F.avg("files_in_part").alias("avg_files_per_partition"),
            F.max("files_in_part").alias("max_files_per_partition"),
            F.min("files_in_part").alias("min_files_per_partition"),
            F.stddev("files_in_part").alias("stddev_files_per_partition"),
        ).collect()[0]

        num_parts_actual = part_agg["num_partitions_actual"]
        avg_files_per_part = part_agg["avg_files_per_partition"] or 0
        max_files_per_part = part_agg["max_files_per_partition"] or 0
        min_files_per_part = part_agg["min_files_per_partition"] or 0
        stddev_files_per_part = part_agg["stddev_files_per_partition"] or 0
    else:
        # Unpartitioned — all files belong to one implicit partition
        num_parts_actual = 1
        avg_files_per_part = float(file_count)
        max_files_per_part = file_count
        min_files_per_part = file_count
        stddev_files_per_part = 0.0

    # ── Snapshot features ───────────────────────────────────────────────
    num_snapshots = spark.sql(f"SELECT * FROM {fqn}.snapshots").count()

    # ── Derived features ────────────────────────────────────────────────
    # The compaction filter skips files >= min-file-size-bytes (96 MB).
    # Files below this threshold are "small files" that will be rewritten.
    # We use this fixed threshold rather than the per-table
    # file_size_target_kb to avoid leaking generation parameters into
    # observable features.
    COMPACTION_MIN_BYTES = 96 * 1024 * 1024  # 96 MB = 75% of 128 MB target

    # Ratio of files below the compaction threshold (these are "small files")
    if file_count > 0:
        small_file_count = files_df.filter(
            F.col("file_size_in_bytes") < COMPACTION_MIN_BYTES
        ).count()
        small_file_ratio = small_file_count / file_count
    else:
        small_file_ratio = 0.0

    # Coefficient of variation for file sizes (stddev / mean)
    file_size_cv = (stddev_file_size / avg_file_size) if avg_file_size > 0 else 0.0

    # Coefficient of variation for files-per-partition
    files_per_part_cv = (
        (stddev_files_per_part / avg_files_per_part)
        if avg_files_per_part > 0
        else 0.0
    )

    elapsed = time.time() - t0

    features = {
        "config_id": cfg.config_id,
        # Config params (included as features — the model can learn
        # whether schema width or batch count affect compaction value)
        "num_rows": cfg.num_rows,
        "num_columns": cfg.num_columns,
        "num_partitions_cfg": cfg.num_partitions,
        "num_writers": cfg.num_writers,
        "num_write_batches": cfg.num_write_batches,
        "file_size_target_kb": cfg.file_size_target_kb,
        "partition_skew": cfg.partition_skew,
        # File-level
        "file_count": file_count,
        "total_size_bytes": int(total_size),
        "avg_file_size_bytes": round(avg_file_size, 2),
        "min_file_size_bytes": int(min_file_size),
        "max_file_size_bytes": int(max_file_size),
        "stddev_file_size_bytes": round(stddev_file_size, 2),
        "total_records": int(total_records),
        "avg_records_per_file": round(avg_records, 2),
        # Partition-level
        "num_partitions_actual": num_parts_actual,
        "avg_files_per_partition": round(avg_files_per_part, 2),
        "max_files_per_partition": int(max_files_per_part),
        "min_files_per_partition": int(min_files_per_part),
        "stddev_files_per_partition": round(stddev_files_per_part, 2),
        # Snapshot-level
        "num_snapshots": num_snapshots,
        # Derived
        "small_file_ratio": round(small_file_ratio, 4),
        "file_size_cv": round(file_size_cv, 4),
        "files_per_partition_cv": round(files_per_part_cv, 4),
    }

    LOG.info(
        "  %s: %d files, %.1f MB, cv=%.2f, small_ratio=%.2f  (%.1fs)",
        table_name,
        file_count,
        total_size / 1e6,
        file_size_cv,
        small_file_ratio,
        elapsed,
    )

    return features


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


def table_exists(spark: SparkSession, cfg: SimConfig) -> bool:
    """Check if a table exists (even partially)."""
    fqn = _fqn(f"t_{cfg.config_id}")
    try:
        spark.sql(f"SELECT 1 FROM {fqn} LIMIT 1").collect()
        return True
    except Exception:
        return False


# ── CLI ─────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract metadata features from Iceberg tables"
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
        default="code/features.csv",
        help="Output CSV path (default: code/features.csv)",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=0,
        help="Extract only N random configs (0 = all)",
    )
    parser.add_argument(
        "--config-id",
        type=str,
        default=None,
        help="Extract only this config ID",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for sampling"
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

    LOG.info("Selected %d configs", len(configs))

    # ── Extract ─────────────────────────────────────────────────────────
    LOG.info("Warehouse: %s", args.warehouse)
    spark = get_spark(args.warehouse)

    try:
        results: list[dict[str, Any]] = []
        skipped = 0
        failed = 0

        for i, cfg in enumerate(configs, 1):
            LOG.info("=== [%d/%d] config %s ===", i, len(configs), cfg.config_id)

            if not table_exists(spark, cfg):
                LOG.warning("  SKIP (table does not exist yet)")
                skipped += 1
                continue

            try:
                features = extract_features(spark, cfg)
                results.append(features)
            except Exception:
                LOG.exception("FAILED config %s", cfg.config_id)
                failed += 1

        # ── Write output ────────────────────────────────────────────────
        if results:
            out_path = Path(args.out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with out_path.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=FEATURE_COLUMNS)
                writer.writeheader()
                for r in results:
                    writer.writerow(r)
            LOG.info("Features written to %s (%d rows)", out_path, len(results))

        # ── Summary ─────────────────────────────────────────────────────
        LOG.info("=== Summary ===")
        LOG.info(
            "Extracted: %d  |  Skipped: %d  |  Failed: %d  |  Total: %d",
            len(results),
            skipped,
            failed,
            len(configs),
        )

        return 1 if failed else 0
    finally:
        spark.stop()


if __name__ == "__main__":
    raise SystemExit(main())
