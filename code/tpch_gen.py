#!/usr/bin/env python3
"""
Generate TPC-H-inspired Iceberg tables for cross-schema validation.

Creates tables with realistic TPC-H data types (INT, DECIMAL, DATE,
variable-length STRING) instead of the UUID-string payload used in the
synthetic generator.  Write patterns (file-size target, batch count,
writer count) are varied by a parameter grid to produce a range of
file-fragmentation states, so the pre-trained XGBoost model can be
tested on data it has never seen.

Design rationale
----------------
The *only* difference from the synthetic pipeline is the data content:
column types and value distributions.  The Iceberg write mechanics
(file-size targets, multi-batch heterogeneous writes, repartition by
writer count) are identical, so any accuracy drop on these tables
isolates the question "does the model generalise across schemas?"

Tables generated (inspired by TPC-H ~SF10):
    lineitem   16 cols   10M rows  3 parts  ~240 MB
    orders      9 cols    6M rows  3 parts  ~90 MB
    customer    8 cols    3M rows  1 part   ~60 MB
    part        9 cols    5M rows  1 part   ~100 MB
    partsupp    5 cols    5M rows  1 part   ~50 MB
    supplier    7 cols  300K rows  1 part   ~6 MB

Usage:
    python code/tpch_gen.py                         # full grid (96 tables)
    python code/tpch_gen.py --sample 10             # 10 random tables
    python code/tpch_gen.py --dry-run               # preview only
    python code/tpch_gen.py --warehouse /mnt/data/tpch_warehouse
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import logging
import random as rand_mod
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

LOG = logging.getLogger("tpch_gen")

# ── Defaults ────────────────────────────────────────────────────────────────

DEFAULT_WAREHOUSE = "/mnt/data/tpch_warehouse"
CATALOG = "local"
DATABASE = "tpch"

# ── TPC-H table definitions ────────────────────────────────────────────────

TPCH_TABLE_DEFS: dict[str, dict[str, Any]] = {
    "lineitem":  {"num_rows": 10_000_000, "num_partitions": 3, "num_columns": 16},  # ~240 MB
    "orders":    {"num_rows":  6_000_000, "num_partitions": 3, "num_columns": 9},   # ~90 MB
    "customer":  {"num_rows":  3_000_000, "num_partitions": 1, "num_columns": 8},   # ~60 MB
    "part":      {"num_rows":  5_000_000, "num_partitions": 1, "num_columns": 9},   # ~100 MB
    "partsupp":  {"num_rows":  5_000_000, "num_partitions": 1, "num_columns": 5},   # ~50 MB
    "supplier":  {"num_rows":    300_000, "num_partitions": 1, "num_columns": 7},   # ~6 MB
}

# Write-pattern grid axes
WRITE_AXES: dict[str, list[Any]] = {
    "file_size_target_kb": [512, 4_096, 32_768, 131_072],
    "num_write_batches":   [1, 5],
    "num_writers":         [1, 3],
}


# ── Config dataclass ────────────────────────────────────────────────────────

@dataclass(frozen=True)
class TpchConfig:
    """One point in the TPC-H validation grid."""
    tpch_table: str
    num_rows: int
    num_columns: int
    num_partitions: int
    file_size_target_kb: int
    num_write_batches: int
    num_writers: int

    @property
    def config_id(self) -> str:
        raw = json.dumps(
            {
                "table": self.tpch_table,
                "rows": self.num_rows,
                "parts": self.num_partitions,
                "target_kb": self.file_size_target_kb,
                "batches": self.num_write_batches,
                "writers": self.num_writers,
            },
            sort_keys=True,
        )
        return hashlib.sha256(raw.encode()).hexdigest()[:8]

    @property
    def iceberg_table_name(self) -> str:
        return f"tpch_{self.tpch_table}_{self.config_id}"

    def batch_target_kb(self, batch_idx: int) -> int:
        """Heterogeneous file-size target (mirrors SimConfig)."""
        if self.num_write_batches <= 1 or batch_idx == 0:
            return self.file_size_target_kb
        if batch_idx % 2 == 0:
            return self.file_size_target_kb
        return max(8, self.file_size_target_kb // 8)

    def batch_rows(self, batch_idx: int) -> int:
        """Row count per batch (mirrors SimConfig)."""
        if self.num_write_batches <= 1:
            return self.num_rows
        if batch_idx == 0:
            return self.num_rows // 2
        remaining = self.num_rows - self.num_rows // 2
        per_batch = max(1, remaining // (self.num_write_batches - 1))
        if batch_idx == self.num_write_batches - 1:
            used = self.num_rows // 2 + per_batch * (self.num_write_batches - 2)
            return self.num_rows - used
        return per_batch


# ── Grid generation ─────────────────────────────────────────────────────────

def full_tpch_grid() -> list[TpchConfig]:
    """Cartesian product: TPC-H tables × write-pattern axes."""
    configs: list[TpchConfig] = []
    write_keys = list(WRITE_AXES.keys())
    for table_name, table_def in TPCH_TABLE_DEFS.items():
        for combo in itertools.product(*(WRITE_AXES[k] for k in write_keys)):
            write_params = dict(zip(write_keys, combo))
            configs.append(
                TpchConfig(
                    tpch_table=table_name,
                    num_rows=table_def["num_rows"],
                    num_columns=table_def["num_columns"],
                    num_partitions=table_def["num_partitions"],
                    **write_params,
                )
            )
    return configs


# ── Spark session ───────────────────────────────────────────────────────────

def get_spark(warehouse: str) -> SparkSession:
    return (
        SparkSession.builder.master("local[*]")
        .appName("tpch-gen")
        .config("spark.driver.memory", "12g")
        .config(
            "spark.sql.extensions",
            "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions",
        )
        .config(f"spark.sql.catalog.{CATALOG}", "org.apache.iceberg.spark.SparkCatalog")
        .config(f"spark.sql.catalog.{CATALOG}.type", "hadoop")
        .config(f"spark.sql.catalog.{CATALOG}.warehouse", warehouse)
        .config("spark.sql.shuffle.partitions", "4")
        .getOrCreate()
    )


# ── TPC-H data generators ──────────────────────────────────────────────────
# Each function returns a DataFrame with realistic column types.
# A `part_key` column is added for partitioned tables so that the
# Iceberg PARTITIONED BY clause can reference it uniformly.

def _add_part_key(df: DataFrame, num_partitions: int, seed: int) -> DataFrame:
    """Assign an integer partition key using uniform modular distribution."""
    if num_partitions <= 1:
        return df.withColumn("part_key", F.lit(0).cast("int"))
    return df.withColumn(
        "part_key",
        (F.abs(F.hash(F.col("_row_id"), F.lit(seed))) % num_partitions).cast("int"),
    )


def _gen_lineitem(spark: SparkSession, num_rows: int, offset: int,
                  num_partitions: int, batch_seed: int) -> DataFrame:
    """16-column lineitem table with mixed types."""
    df = spark.range(offset, offset + num_rows).withColumnRenamed("id", "_row_id")
    s = batch_seed  # shift seeds per batch to avoid identical values
    df = (
        df
        .withColumn("l_orderkey", F.col("_row_id").cast("bigint"))
        .withColumn("l_partkey", (F.rand(seed=s+1) * 200_000 + 1).cast("bigint"))
        .withColumn("l_suppkey", (F.rand(seed=s+2) * 10_000 + 1).cast("bigint"))
        .withColumn("l_linenumber", (F.rand(seed=s+3) * 7 + 1).cast("int"))
        .withColumn("l_quantity", F.round(F.rand(seed=s+4) * 50 + 1, 2).cast("decimal(15,2)"))
        .withColumn("l_extendedprice", F.round(F.rand(seed=s+5) * 100_000, 2).cast("decimal(15,2)"))
        .withColumn("l_discount", F.round(F.rand(seed=s+6) * 0.10, 2).cast("decimal(15,2)"))
        .withColumn("l_tax", F.round(F.rand(seed=s+7) * 0.08, 2).cast("decimal(15,2)"))
        .withColumn("l_returnflag",
                    F.array(F.lit("R"), F.lit("A"), F.lit("N"))
                    .getItem((F.rand(seed=s+8) * 3).cast("int")))
        .withColumn("l_linestatus",
                    F.array(F.lit("O"), F.lit("F"))
                    .getItem((F.rand(seed=s+9) * 2).cast("int")))
        .withColumn("l_shipdate",
                    F.date_add(F.lit("1992-01-01"), (F.rand(seed=s+10) * 2526).cast("int")))
        .withColumn("l_commitdate",
                    F.date_add(F.lit("1992-01-01"), (F.rand(seed=s+11) * 2526).cast("int")))
        .withColumn("l_receiptdate",
                    F.date_add(F.lit("1992-01-01"), (F.rand(seed=s+12) * 2526).cast("int")))
        .withColumn("l_shipinstruct",
                    F.array(F.lit("DELIVER IN PERSON"), F.lit("COLLECT COD"),
                            F.lit("NONE"), F.lit("TAKE BACK RETURN"))
                    .getItem((F.rand(seed=s+13) * 4).cast("int")))
        .withColumn("l_shipmode",
                    F.array(F.lit("REG AIR"), F.lit("AIR"), F.lit("RAIL"),
                            F.lit("SHIP"), F.lit("TRUCK"), F.lit("MAIL"), F.lit("FOB"))
                    .getItem((F.rand(seed=s+14) * 7).cast("int")))
        .withColumn("l_comment", F.expr("substr(uuid(), 1, 27)"))
    )
    df = _add_part_key(df, num_partitions, seed=s + 99)
    cols = [c for c in df.columns if c != "_row_id"]
    return df.select(*cols)


def _gen_orders(spark: SparkSession, num_rows: int, offset: int,
                num_partitions: int, batch_seed: int) -> DataFrame:
    """9-column orders table."""
    df = spark.range(offset, offset + num_rows).withColumnRenamed("id", "_row_id")
    s = batch_seed
    df = (
        df
        .withColumn("o_orderkey", F.col("_row_id").cast("bigint"))
        .withColumn("o_custkey", (F.rand(seed=s+1) * 150_000 + 1).cast("bigint"))
        .withColumn("o_orderstatus",
                    F.array(F.lit("O"), F.lit("F"), F.lit("P"))
                    .getItem((F.rand(seed=s+2) * 3).cast("int")))
        .withColumn("o_totalprice", F.round(F.rand(seed=s+3) * 500_000, 2).cast("decimal(15,2)"))
        .withColumn("o_orderdate",
                    F.date_add(F.lit("1992-01-01"), (F.rand(seed=s+4) * 2526).cast("int")))
        .withColumn("o_orderpriority",
                    F.array(F.lit("1-URGENT"), F.lit("2-HIGH"), F.lit("3-MEDIUM"),
                            F.lit("4-NOT SPECIFIED"), F.lit("5-LOW"))
                    .getItem((F.rand(seed=s+5) * 5).cast("int")))
        .withColumn("o_clerk",
                    F.concat(F.lit("Clerk#"), F.lpad((F.rand(seed=s+6) * 1000 + 1).cast("int").cast("string"), 9, "0")))
        .withColumn("o_shippriority", F.lit(0).cast("int"))
        .withColumn("o_comment", F.expr("substr(uuid(), 1, 48)"))
    )
    df = _add_part_key(df, num_partitions, seed=s + 99)
    cols = [c for c in df.columns if c != "_row_id"]
    return df.select(*cols)


def _gen_customer(spark: SparkSession, num_rows: int, offset: int,
                  num_partitions: int, batch_seed: int) -> DataFrame:
    """8-column customer table."""
    df = spark.range(offset, offset + num_rows).withColumnRenamed("id", "_row_id")
    s = batch_seed
    df = (
        df
        .withColumn("c_custkey", F.col("_row_id").cast("bigint"))
        .withColumn("c_name",
                    F.concat(F.lit("Customer#"), F.lpad(F.col("_row_id").cast("string"), 9, "0")))
        .withColumn("c_address", F.expr("substr(uuid(), 1, 25)"))
        .withColumn("c_nationkey", (F.rand(seed=s+1) * 25).cast("int"))
        .withColumn("c_phone",
                    F.concat(
                        F.lpad((F.rand(seed=s+2) * 25 + 10).cast("int").cast("string"), 2, "0"),
                        F.lit("-"),
                        F.lpad((F.rand(seed=s+3) * 900 + 100).cast("int").cast("string"), 3, "0"),
                        F.lit("-"),
                        F.lpad((F.rand(seed=s+4) * 900 + 100).cast("int").cast("string"), 3, "0"),
                        F.lit("-"),
                        F.lpad((F.rand(seed=s+5) * 9000 + 1000).cast("int").cast("string"), 4, "0"),
                    ))
        .withColumn("c_acctbal", F.round(F.rand(seed=s+6) * 10_000 - 1_000, 2).cast("decimal(15,2)"))
        .withColumn("c_mktsegment",
                    F.array(F.lit("AUTOMOBILE"), F.lit("BUILDING"), F.lit("FURNITURE"),
                            F.lit("HOUSEHOLD"), F.lit("MACHINERY"))
                    .getItem((F.rand(seed=s+7) * 5).cast("int")))
        .withColumn("c_comment", F.expr("substr(uuid(), 1, 72)"))
    )
    df = _add_part_key(df, num_partitions, seed=s + 99)
    cols = [c for c in df.columns if c != "_row_id"]
    return df.select(*cols)


def _gen_part(spark: SparkSession, num_rows: int, offset: int,
              num_partitions: int, batch_seed: int) -> DataFrame:
    """9-column part table."""
    df = spark.range(offset, offset + num_rows).withColumnRenamed("id", "_row_id")
    s = batch_seed
    types = ["STANDARD", "SMALL", "MEDIUM", "LARGE", "ECONOMY", "PROMO"]
    metals = ["TIN", "NICKEL", "BRASS", "STEEL", "COPPER"]
    df = (
        df
        .withColumn("p_partkey", F.col("_row_id").cast("bigint"))
        .withColumn("p_name", F.concat(
            F.array(*[F.lit(t) for t in types]).getItem((F.rand(seed=s+1) * len(types)).cast("int")),
            F.lit(" "),
            F.array(*[F.lit(m) for m in metals]).getItem((F.rand(seed=s+2) * len(metals)).cast("int")),
        ))
        .withColumn("p_mfgr",
                    F.concat(F.lit("Manufacturer#"), (F.rand(seed=s+3) * 5 + 1).cast("int").cast("string")))
        .withColumn("p_brand",
                    F.concat(F.lit("Brand#"), (F.rand(seed=s+4) * 40 + 11).cast("int").cast("string")))
        .withColumn("p_type", F.concat(
            F.array(*[F.lit(t) for t in types]).getItem((F.rand(seed=s+5) * len(types)).cast("int")),
            F.lit(" POLISHED "),
            F.array(*[F.lit(m) for m in metals]).getItem((F.rand(seed=s+6) * len(metals)).cast("int")),
        ))
        .withColumn("p_size", (F.rand(seed=s+7) * 50 + 1).cast("int"))
        .withColumn("p_container",
                    F.array(F.lit("SM CASE"), F.lit("SM BOX"), F.lit("SM PACK"), F.lit("SM PKG"),
                            F.lit("MED BAG"), F.lit("MED BOX"), F.lit("MED PKG"),
                            F.lit("LG CASE"), F.lit("LG BOX"), F.lit("LG PACK"))
                    .getItem((F.rand(seed=s+8) * 10).cast("int")))
        .withColumn("p_retailprice", F.round(F.rand(seed=s+9) * 2000 + 1, 2).cast("decimal(15,2)"))
        .withColumn("p_comment", F.expr("substr(uuid(), 1, 14)"))
    )
    df = _add_part_key(df, num_partitions, seed=s + 99)
    cols = [c for c in df.columns if c != "_row_id"]
    return df.select(*cols)


def _gen_partsupp(spark: SparkSession, num_rows: int, offset: int,
                  num_partitions: int, batch_seed: int) -> DataFrame:
    """5-column partsupp table."""
    df = spark.range(offset, offset + num_rows).withColumnRenamed("id", "_row_id")
    s = batch_seed
    df = (
        df
        .withColumn("ps_partkey", (F.rand(seed=s+1) * 200_000 + 1).cast("bigint"))
        .withColumn("ps_suppkey", (F.rand(seed=s+2) * 10_000 + 1).cast("bigint"))
        .withColumn("ps_availqty", (F.rand(seed=s+3) * 10_000).cast("int"))
        .withColumn("ps_supplycost", F.round(F.rand(seed=s+4) * 1_000, 2).cast("decimal(15,2)"))
        .withColumn("ps_comment", F.expr("substr(uuid(), 1, 124)"))
    )
    df = _add_part_key(df, num_partitions, seed=s + 99)
    cols = [c for c in df.columns if c != "_row_id"]
    return df.select(*cols)


def _gen_supplier(spark: SparkSession, num_rows: int, offset: int,
                  num_partitions: int, batch_seed: int) -> DataFrame:
    """7-column supplier table."""
    df = spark.range(offset, offset + num_rows).withColumnRenamed("id", "_row_id")
    s = batch_seed
    df = (
        df
        .withColumn("s_suppkey", F.col("_row_id").cast("bigint"))
        .withColumn("s_name",
                    F.concat(F.lit("Supplier#"), F.lpad(F.col("_row_id").cast("string"), 9, "0")))
        .withColumn("s_address", F.expr("substr(uuid(), 1, 25)"))
        .withColumn("s_nationkey", (F.rand(seed=s+1) * 25).cast("int"))
        .withColumn("s_phone",
                    F.concat(
                        F.lpad((F.rand(seed=s+2) * 25 + 10).cast("int").cast("string"), 2, "0"),
                        F.lit("-"),
                        F.lpad((F.rand(seed=s+3) * 900 + 100).cast("int").cast("string"), 3, "0"),
                        F.lit("-"),
                        F.lpad((F.rand(seed=s+4) * 900 + 100).cast("int").cast("string"), 3, "0"),
                        F.lit("-"),
                        F.lpad((F.rand(seed=s+5) * 9000 + 1000).cast("int").cast("string"), 4, "0"),
                    ))
        .withColumn("s_acctbal", F.round(F.rand(seed=s+6) * 10_000 - 1_000, 2).cast("decimal(15,2)"))
        .withColumn("s_comment", F.expr("substr(uuid(), 1, 63)"))
    )
    df = _add_part_key(df, num_partitions, seed=s + 99)
    cols = [c for c in df.columns if c != "_row_id"]
    return df.select(*cols)


TABLE_GENERATORS = {
    "lineitem": _gen_lineitem,
    "orders":   _gen_orders,
    "customer": _gen_customer,
    "part":     _gen_part,
    "partsupp": _gen_partsupp,
    "supplier": _gen_supplier,
}


# ── Table DDL & write logic ────────────────────────────────────────────────

def _fqn(table_name: str) -> str:
    return f"{CATALOG}.{DATABASE}.{table_name}"


def _create_table(spark: SparkSession, cfg: TpchConfig,
                  sample_df: DataFrame) -> None:
    """Create Iceberg table from a sample DataFrame's schema."""
    table_name = cfg.iceberg_table_name
    fqn = _fqn(table_name)

    # Build DDL from the DataFrame schema
    col_defs = []
    for field in sample_df.schema.fields:
        col_defs.append(f"{field.name} {field.dataType.simpleString()}")
    cols_sql = ", ".join(col_defs)

    partition_clause = ""
    if cfg.num_partitions > 1:
        partition_clause = "PARTITIONED BY (part_key)"

    target_bytes = cfg.file_size_target_kb * 1024
    spark.sql(f"DROP TABLE IF EXISTS {fqn}")
    spark.sql(f"""
        CREATE TABLE {fqn} (
            {cols_sql}
        )
        USING iceberg
        {partition_clause}
        TBLPROPERTIES (
            'write.target-file-size-bytes' = '{target_bytes}',
            'write.distribution-mode' = 'none'
        )
    """)


def _alter_file_target(spark: SparkSession, table_name: str, target_kb: int) -> None:
    fqn = _fqn(table_name)
    target_bytes = target_kb * 1024
    spark.sql(
        f"ALTER TABLE {fqn} SET TBLPROPERTIES "
        f"('write.target-file-size-bytes' = '{target_bytes}')"
    )


def table_exists(spark: SparkSession, cfg: TpchConfig) -> bool:
    table_name = cfg.iceberg_table_name
    fqn = _fqn(table_name)
    try:
        actual = spark.sql(f"SELECT COUNT(*) FROM {fqn}").collect()[0][0]
        if actual == cfg.num_rows:
            return True
        LOG.info("Table %s has %d rows (expected %d) — regenerating",
                 table_name, actual, cfg.num_rows)
        return False
    except Exception:
        return False


def generate_table(spark: SparkSession, cfg: TpchConfig) -> dict[str, Any]:
    """Generate one TPC-H Iceberg table. Returns a result dict."""
    gen_fn = TABLE_GENERATORS[cfg.tpch_table]
    table_name = cfg.iceberg_table_name
    fqn = _fqn(table_name)

    LOG.info(
        "Generating %s  tpch=%s rows=%d cols=%d parts=%d "
        "writers=%d batches=%d file_kb=%d",
        table_name, cfg.tpch_table, cfg.num_rows, cfg.num_columns,
        cfg.num_partitions, cfg.num_writers, cfg.num_write_batches,
        cfg.file_size_target_kb,
    )

    t0 = time.time()

    # Generate a small sample to infer schema for DDL
    sample_df = gen_fn(spark, 1, 0, cfg.num_partitions, batch_seed=0)
    _create_table(spark, cfg, sample_df)

    row_offset = 0
    for batch_idx in range(cfg.num_write_batches):
        batch_rows = cfg.batch_rows(batch_idx)
        batch_target = cfg.batch_target_kb(batch_idx)

        if batch_idx > 0:
            _alter_file_target(spark, table_name, batch_target)

        batch_seed = batch_idx * 100
        df = gen_fn(spark, batch_rows, row_offset, cfg.num_partitions, batch_seed)
        df = df.repartition(cfg.num_writers)
        df.writeTo(fqn).append()

        LOG.info(
            "  batch %d/%d: %d rows appended (target %d KB)",
            batch_idx + 1, cfg.num_write_batches, batch_rows, batch_target,
        )
        row_offset += batch_rows

    elapsed = time.time() - t0

    actual_rows = spark.sql(f"SELECT COUNT(*) FROM {fqn}").collect()[0][0]
    file_count = spark.sql(f"SELECT * FROM {fqn}.files").count()

    result = {
        "config_id": cfg.config_id,
        "tpch_table": cfg.tpch_table,
        "table_name": table_name,
        "row_count": actual_rows,
        "file_count": file_count,
        "elapsed_s": round(elapsed, 2),
    }

    LOG.info("  => %d rows, %d files, %.1fs", actual_rows, file_count, elapsed)
    return result


# ── CSV export ──────────────────────────────────────────────────────────────

def export_grid_csv(configs: list[TpchConfig], outpath: Path) -> None:
    """Dump the TPC-H config grid to CSV."""
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "config_id", "tpch_table", "num_rows", "num_columns",
        "num_partitions", "file_size_target_kb", "num_write_batches",
        "num_writers", "iceberg_table_name",
    ]
    with outpath.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for c in configs:
            writer.writerow({
                "config_id": c.config_id,
                "tpch_table": c.tpch_table,
                "num_rows": c.num_rows,
                "num_columns": c.num_columns,
                "num_partitions": c.num_partitions,
                "file_size_target_kb": c.file_size_target_kb,
                "num_write_batches": c.num_write_batches,
                "num_writers": c.num_writers,
                "iceberg_table_name": c.iceberg_table_name,
            })
    LOG.info("Grid CSV written to %s (%d configs)", outpath, len(configs))


# ── CLI ─────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate TPC-H Iceberg tables for cross-schema validation"
    )
    parser.add_argument("--warehouse", type=str, default=DEFAULT_WAREHOUSE)
    parser.add_argument("--sample", type=int, default=0,
                        help="Generate only N random tables (0 = all)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--grid-csv", type=str, default="code/tpch_grid.csv",
                        help="Path to write grid CSV")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s  %(message)s",
    )

    configs = full_tpch_grid()
    LOG.info("Full TPC-H grid: %d configs", len(configs))

    if args.sample > 0:
        rng = rand_mod.Random(args.seed)
        configs = rng.sample(configs, min(args.sample, len(configs)))
        LOG.info("Sampled %d configs", len(configs))

    # Always write grid CSV
    export_grid_csv(configs, Path(args.grid_csv))

    if args.dry_run:
        for c in configs:
            print(
                f"  {c.config_id}  tpch={c.tpch_table:<10}  "
                f"rows={c.num_rows:>10,}  cols={c.num_columns:>2}  "
                f"parts={c.num_partitions:>2}  writers={c.num_writers}  "
                f"batches={c.num_write_batches}  file_kb={c.file_size_target_kb:>7,}"
            )
        print(f"\nTotal: {len(configs)} tables")
        return 0

    LOG.info("Warehouse: %s", args.warehouse)
    spark = get_spark(args.warehouse)

    try:
        spark.sql(f"CREATE NAMESPACE IF NOT EXISTS {CATALOG}.{DATABASE}")

        results: list[dict[str, Any]] = []
        skipped = 0
        failed = 0

        for i, cfg in enumerate(configs, 1):
            LOG.info("=== [%d/%d] %s ===", i, len(configs), cfg.iceberg_table_name)
            if table_exists(spark, cfg):
                LOG.info("  SKIP (already exists with correct row count)")
                skipped += 1
                continue
            try:
                result = generate_table(spark, cfg)
                results.append(result)
            except Exception:
                LOG.exception("FAILED %s", cfg.iceberg_table_name)
                failed += 1

        LOG.info("=== Summary ===")
        LOG.info(
            "Generated: %d  |  Skipped: %d  |  Failed: %d  |  Total: %d",
            len(results), skipped, failed, len(configs),
        )
        if results:
            total_files = sum(r["file_count"] for r in results)
            total_rows = sum(r["row_count"] for r in results)
            total_time = sum(r["elapsed_s"] for r in results)
            LOG.info("Files: %d  |  Rows: %d  |  Time: %.1fs",
                     total_files, total_rows, total_time)

        return 1 if failed else 0
    finally:
        spark.stop()


if __name__ == "__main__":
    raise SystemExit(main())
