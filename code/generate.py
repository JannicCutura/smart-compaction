#!/usr/bin/env python3
"""
Generate synthetic Iceberg tables from the simulation parameter grid.

For each SimConfig, creates an Iceberg table whose data layout (number of
small files, partitioning, skew) is controlled by the config parameters.

Key design choices (v2)
-----------------------
* **Realistic file sizes.**  Payload columns are 36-char UUID strings
  (~30 bytes/col in Parquet).  With 50 columns, each row is ~1.5 KB.
  Combined with ``file_size_target_kb`` up to 128 MB, this produces files
  that span from single-digit KB to hundreds of MB, straddling the
  compaction filter thresholds (96 MB min, 230 MB max).

* **Zipfian partition skew.**  When ``partition_skew == "zipf"``, row
  assignment follows Zipf(s=1).  Partition 0 is hottest; cold partitions
  receive very few rows, matching empirical data-lake patterns.

* **Heterogeneous batches.**  When there are multiple write batches:
  - Batch 0 gets 50 % of total rows (big initial load).
  - Subsequent batches split the remaining 50 % equally (increments).
  - Even-indexed batches (1, 3, …) use a file-size target 8× smaller
    than the configured value, creating within-table file-size
    heterogeneity that mimics mixed-pipeline production tables.
  Between batches the table property ``write.target-file-size-bytes``
  is ALTER-ed so that Spark's Parquet writer respects the new target.

Usage:
    python code/generate.py                              # all configs
    python code/generate.py --sample 5                   # 5 random
    python code/generate.py --config-id abc12345         # single
    python code/generate.py --warehouse /mnt/data/wh     # custom path
    python code/generate.py --dry-run                    # preview only
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import random as rand_mod
import sys
import time
from pathlib import Path
from typing import Any

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType

# Resolve imports from the same package
sys.path.insert(0, str(Path(__file__).resolve().parent))
from params import SimConfig, full_grid, sampled_grid  # noqa: E402

LOG = logging.getLogger("generate")

# ── Defaults ────────────────────────────────────────────────────────────────

DEFAULT_WAREHOUSE = "/mnt/data/warehouse"
CATALOG = "local"
DATABASE = "compaction"


# ── Spark session ───────────────────────────────────────────────────────────


def get_spark(warehouse: str) -> SparkSession:
    """Build a local SparkSession with Iceberg hadoop catalog."""
    return (
        SparkSession.builder.master("local[*]")
        .appName("iceberg-table-gen")
        .config("spark.driver.memory", "12g")
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
        # Keep shuffle partitions low for local mode
        .config("spark.sql.shuffle.partitions", "4")
        .getOrCreate()
    )


# ── Zipfian partition assignment ────────────────────────────────────────────


def _build_zipf_expr(num_partitions: int, seed: int = 42) -> F.Column:
    """Build a Spark SQL expression that maps rand() → Zipf(s=1) partition.

    Uses inverse-CDF via a chained WHEN expression.  For K partitions the
    weights are w_k = 1/k (k = 1..K), normalised.  The CDF is evaluated
    once at code-gen time; the runtime cost is a single rand() + O(K)
    comparisons, which Spark JIT-compiles efficiently.
    """
    K = num_partitions
    weights = [1.0 / (k + 1) for k in range(K)]  # 0-indexed
    total = sum(weights)
    cdf = []
    running = 0.0
    for w in weights:
        running += w / total
        cdf.append(running)
    cdf[-1] = 1.0  # ensure no floating-point gap

    rand_col = F.rand(seed=seed)
    # Build from last partition backward so the WHEN chain is correct
    expr = F.lit(K - 1).cast("int")
    for i in range(K - 2, -1, -1):
        expr = F.when(rand_col < F.lit(cdf[i]), F.lit(i).cast("int")).otherwise(expr)
    return expr


# ── Data generation ─────────────────────────────────────────────────────────


def generate_batch_df(
    spark: SparkSession,
    cfg: SimConfig,
    batch_rows: int,
    batch_offset: int,
) -> DataFrame:
    """Create a DataFrame of synthetic rows for one write batch.

    Columns: id (BIGINT), ts (TIMESTAMP), part_key (INT), col_0..col_N (STRING).
    Partition key assignment respects the skew profile.
    Payload columns are UUIDs (~36 chars, low compressibility in Parquet).
    """
    df = spark.range(batch_offset, batch_offset + batch_rows).withColumnRenamed(
        "id", "row_id"
    )

    # Assign partition keys
    if cfg.num_partitions == 1:
        df = df.withColumn("part_key", F.lit(0).cast("int"))
    elif cfg.partition_skew == "uniform":
        df = df.withColumn(
            "part_key", (F.col("row_id") % cfg.num_partitions).cast("int")
        )
    elif cfg.partition_skew == "zipf":
        df = df.withColumn("part_key", _build_zipf_expr(cfg.num_partitions))
    else:
        # Fallback: uniform
        df = df.withColumn(
            "part_key", (F.col("row_id") % cfg.num_partitions).cast("int")
        )

    # Timestamp & id
    df = df.withColumn("ts", F.current_timestamp())
    df = df.withColumn("id", F.col("row_id")).drop("row_id")

    # Payload columns – UUIDs give predictable ~36 bytes/col/row
    for i in range(cfg.num_columns):
        df = df.withColumn(f"col_{i}", F.expr("uuid()"))

    # Final column order
    cols = ["id", "ts", "part_key"] + [f"col_{i}" for i in range(cfg.num_columns)]
    return df.select(*cols)


# ── Table DDL ───────────────────────────────────────────────────────────────


def _fqn(table_name: str) -> str:
    return f"{CATALOG}.{DATABASE}.{table_name}"


def create_table_ddl(spark: SparkSession, cfg: SimConfig, table_name: str) -> None:
    """Issue CREATE TABLE for the given config."""
    col_defs = [
        "id BIGINT NOT NULL",
        "ts TIMESTAMP NOT NULL",
        "part_key INT NOT NULL",
    ]
    for i in range(cfg.num_columns):
        col_defs.append(f"col_{i} STRING")
    cols_sql = ", ".join(col_defs)

    partition_clause = ""
    if cfg.num_partitions > 1:
        partition_clause = "PARTITIONED BY (part_key)"

    target_bytes = cfg.file_size_target_kb * 1024

    spark.sql(
        f"""
        CREATE TABLE IF NOT EXISTS {_fqn(table_name)} (
            {cols_sql}
        )
        USING iceberg
        {partition_clause}
        TBLPROPERTIES (
            'write.target-file-size-bytes' = '{target_bytes}',
            'write.distribution-mode' = 'none'
        )
        """
    )


def _alter_file_target(spark: SparkSession, table_name: str, target_kb: int) -> None:
    """ALTER TABLE to change the write target file size for the next batch."""
    fqn = _fqn(table_name)
    target_bytes = target_kb * 1024
    spark.sql(
        f"ALTER TABLE {fqn} SET TBLPROPERTIES "
        f"('write.target-file-size-bytes' = '{target_bytes}')"
    )


# ── Per-table generation ────────────────────────────────────────────────────


def table_exists(spark: SparkSession, cfg: SimConfig) -> bool:
    """Check if a table already exists with the expected row count."""
    table_name = f"t_{cfg.config_id}"
    fqn = _fqn(table_name)
    try:
        actual = spark.sql(f"SELECT COUNT(*) FROM {fqn}").collect()[0][0]
        if actual == cfg.num_rows:
            return True
        LOG.info(
            "Table %s exists but has %d rows (expected %d) — regenerating",
            table_name, actual, cfg.num_rows,
        )
        return False
    except Exception:
        return False


def generate_table(spark: SparkSession, cfg: SimConfig) -> dict[str, Any]:
    """Generate one Iceberg table for *cfg*. Returns a result dict."""
    table_name = f"t_{cfg.config_id}"
    fqn = _fqn(table_name)

    LOG.info(
        "Generating %s  rows=%d cols=%d parts=%d writers=%d "
        "batches=%d file_kb=%d skew=%s",
        table_name,
        cfg.num_rows,
        cfg.num_columns,
        cfg.num_partitions,
        cfg.num_writers,
        cfg.num_write_batches,
        cfg.file_size_target_kb,
        cfg.partition_skew,
    )

    t0 = time.time()

    # Drop and recreate to ensure clean state
    spark.sql(f"DROP TABLE IF EXISTS {fqn}")
    create_table_ddl(spark, cfg, table_name)

    row_offset = 0
    for batch_idx in range(cfg.num_write_batches):
        batch_rows = cfg.batch_rows(batch_idx)
        batch_target = cfg.batch_target_kb(batch_idx)

        # ALTER file target if it differs from the initial setting
        if batch_idx > 0:
            _alter_file_target(spark, table_name, batch_target)

        df = generate_batch_df(spark, cfg, batch_rows, row_offset)

        # Repartition controls how many Spark tasks write (≈ one file each
        # per Iceberg partition with distribution-mode=none).
        df = df.repartition(cfg.num_writers)

        df.writeTo(fqn).append()

        LOG.info(
            "  batch %d/%d: %d rows appended (target %d KB)",
            batch_idx + 1,
            cfg.num_write_batches,
            batch_rows,
            batch_target,
        )
        row_offset += batch_rows

    elapsed = time.time() - t0

    # Verify row count
    actual_rows = spark.sql(f"SELECT COUNT(*) FROM {fqn}").collect()[0][0]

    # Count data files via Iceberg metadata table
    file_count = spark.sql(f"SELECT * FROM {fqn}.files").count()

    result = {
        "config_id": cfg.config_id,
        "table_name": table_name,
        "row_count": actual_rows,
        "file_count": file_count,
        "expected_file_count": cfg.expected_file_count,
        "elapsed_s": round(elapsed, 2),
    }

    LOG.info(
        "  => %d rows, %d files (expected %d), %.1fs",
        actual_rows,
        file_count,
        cfg.expected_file_count,
        elapsed,
    )
    return result


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
        description="Generate synthetic Iceberg tables from the parameter grid"
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
        "--sample",
        type=int,
        default=0,
        help="Generate only N random configs (0 = all)",
    )
    parser.add_argument(
        "--config-id",
        type=str,
        default=None,
        help="Generate only this config ID",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for sampling"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print configs that would be generated, then exit",
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
        LOG.info("CSV not found at %s; using full_grid() (%d configs)", csv_path, len(configs))

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

    # ── Dry-run ─────────────────────────────────────────────────────────
    if args.dry_run:
        for c in configs:
            print(
                f"  {c.config_id}  rows={c.num_rows:>10,}  cols={c.num_columns:>2}  "
                f"parts={c.num_partitions:>3}  writers={c.num_writers:>2}  "
                f"batches={c.num_write_batches:>2}  file_kb={c.file_size_target_kb:>7,}  "
                f"skew={c.partition_skew:<8}  => ~{c.expected_file_count} files"
            )
        total_files = sum(c.expected_file_count for c in configs)
        print(f"\nTotal: {len(configs)} tables, ~{total_files} expected files")
        return 0

    # ── Generate ────────────────────────────────────────────────────────
    LOG.info("Warehouse: %s", args.warehouse)
    spark = get_spark(args.warehouse)

    try:
        spark.sql(f"CREATE NAMESPACE IF NOT EXISTS {CATALOG}.{DATABASE}")

        results: list[dict[str, Any]] = []
        failed = 0

        skipped = 0
        for i, cfg in enumerate(configs, 1):
            LOG.info("=== [%d/%d] config %s ===", i, len(configs), cfg.config_id)
            if table_exists(spark, cfg):
                LOG.info("  SKIP (already exists with correct row count)")
                skipped += 1
                continue
            try:
                result = generate_table(spark, cfg)
                results.append(result)
            except Exception:
                LOG.exception("FAILED config %s", cfg.config_id)
                failed += 1

        # ── Summary ─────────────────────────────────────────────────────
        LOG.info("=== Summary ===")
        LOG.info(
            "Generated: %d  |  Skipped: %d  |  Failed: %d  |  Total: %d",
            len(results),
            skipped,
            failed,
            len(configs),
        )
        if results:
            total_files = sum(r["file_count"] for r in results)
            total_rows = sum(r["row_count"] for r in results)
            total_time = sum(r["elapsed_s"] for r in results)
            LOG.info(
                "Files: %d  |  Rows: %d  |  Time: %.1fs",
                total_files,
                total_rows,
                total_time,
            )

        return 1 if failed else 0
    finally:
        spark.stop()


if __name__ == "__main__":
    raise SystemExit(main())
