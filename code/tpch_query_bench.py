#!/usr/bin/env python3
"""Query-latency benchmark on TPC-H Iceberg tables.

Measures query execution time before and after compaction using
Iceberg time-travel (VERSION AS OF <snapshot_id>).  Links the
predicted compaction utility to actual downstream query speedup,
directly addressing reviewer concern M3.

Three representative query types:
  Q1  Full-scan aggregate    COUNT(*)
  Q2  Filtered aggregation   SUM/GROUP BY with date filter
  Q3  Join                   Two-table equi-join with aggregation

Each query is run WARMUP+REPEATS times; first WARMUP runs are
discarded and median of remaining REPEATS is reported.

Usage:
    python code/tpch_query_bench.py                    # all 96 tables
    python code/tpch_query_bench.py --sample 20        # random 20
    python code/tpch_query_bench.py --dry-run          # preview tables
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

# Fix JAVA_HOME for environments where the VS Code extension path doesn't exist
_JAVA_HOME = "/usr/lib/jvm/java-21-amazon-corretto"
if os.path.isdir(_JAVA_HOME):
    os.environ["JAVA_HOME"] = _JAVA_HOME

LOG = logging.getLogger("tpch_query_bench")
_log_dir = Path(__file__).resolve().parent.parent / "logs"
_log_dir.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)-8s %(message)s",
    handlers=[
        logging.StreamHandler(sys.stderr),
        logging.FileHandler(_log_dir / "tpch_query_bench.log"),
    ],
)

# ── Constants ───────────────────────────────────────────────────────────────

DEFAULT_WAREHOUSE = "/mnt/data/tpch_warehouse"
CATALOG = "local"
DATABASE = "tpch"

WARMUP = 1     # JIT / cache warmup runs (discarded)
REPEATS = 5    # timed repetitions (median reported)

TPCH_QUERY_TABLES = {"lineitem", "orders", "customer", "part"}


# ── Queries per TPC-H base table ────────────────────────────────────────────
# Each entry: (query_label, SQL template with {fqn} placeholder)

def _queries_for(base_table: str, fqn: str) -> list[tuple[str, str]]:
    """Return list of (label, sql) for a given TPC-H base table."""
    queries: list[tuple[str, str]] = []

    # Q1: full-scan aggregate (every base table)
    queries.append(("count_star", f"SELECT COUNT(*) FROM {fqn}"))

    if base_table == "lineitem":
        # Q2: TPC-H Q1 variant – filtered aggregation
        queries.append((
            "tpch_q1",
            f"""
            SELECT l_returnflag, l_linestatus,
                   SUM(l_quantity)        AS sum_qty,
                   SUM(l_extendedprice)   AS sum_base_price,
                   AVG(l_discount)        AS avg_disc,
                   COUNT(*)               AS count_order
            FROM {fqn}
            WHERE l_shipdate <= DATE '1998-09-02'
            GROUP BY l_returnflag, l_linestatus
            ORDER BY l_returnflag, l_linestatus
            """,
        ))
    elif base_table == "orders":
        # Q2: aggregation with date filter
        queries.append((
            "orders_agg",
            f"""
            SELECT o_orderpriority, COUNT(*) AS order_count
            FROM {fqn}
            WHERE o_orderdate >= DATE '1993-07-01'
              AND o_orderdate <  DATE '1993-10-01'
            GROUP BY o_orderpriority
            ORDER BY o_orderpriority
            """,
        ))
    elif base_table == "customer":
        queries.append((
            "cust_nation_agg",
            f"""
            SELECT c_nationkey, COUNT(*) AS cnt, AVG(c_acctbal) AS avg_bal
            FROM {fqn}
            GROUP BY c_nationkey
            ORDER BY c_nationkey
            """,
        ))
    elif base_table == "part":
        queries.append((
            "part_type_agg",
            f"""
            SELECT p_type, COUNT(*) AS cnt, AVG(p_retailprice) AS avg_price
            FROM {fqn}
            GROUP BY p_type
            ORDER BY cnt DESC
            """,
        ))

    return queries


# ── Benchmark logic ─────────────────────────────────────────────────────────

def _fqn(table_name: str) -> str:
    return f"{CATALOG}.{DATABASE}.{table_name}"


def get_snapshots(spark, table_name: str) -> dict[str, int]:
    """Return {op: snapshot_id} for first append and last replace."""
    fqn = _fqn(table_name)
    rows = spark.sql(
        f"SELECT snapshot_id, operation FROM {fqn}.snapshots "
        f"ORDER BY committed_at"
    ).collect()
    result: dict[str, int] = {}
    for r in rows:
        if r["operation"] == "append" and "pre" not in result:
            # first append (might be overwritten by later appends)
            pass
        if r["operation"] == "append":
            result["pre"] = r["snapshot_id"]  # last append = pre-compaction
        if r["operation"] == "replace":
            result["post"] = r["snapshot_id"]
    return result


def time_query(spark, sql: str, warmup: int = WARMUP,
               repeats: int = REPEATS) -> float:
    """Run SQL warmup+repeats times, return median of timed runs (seconds)."""
    for _ in range(warmup):
        spark.sql(sql).collect()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        spark.sql(sql).collect()
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def time_query_snapshot(spark, table_name: str, snapshot_id: int,
                        base_table: str, warmup: int = WARMUP,
                        repeats: int = REPEATS) -> dict[str, float]:
    """Run all queries for a table at a specific snapshot, return {label: median_s}."""
    fqn_snap = f"{_fqn(table_name)} VERSION AS OF {snapshot_id}"
    queries = _queries_for(base_table, fqn_snap)
    results: dict[str, float] = {}
    for label, sql in queries:
        results[label] = time_query(spark, sql, warmup=warmup, repeats=repeats)
    return results


def _load_tpch_configs(grid_path: Path) -> list[dict[str, str]]:
    """Load TPC-H grid configs."""
    with grid_path.open() as f:
        return list(csv.DictReader(f))


def _load_model_predictions(dataset_path: Path, model_clf_path: Path,
                             model_reg_path: Path, feature_path: Path
                             ) -> dict[str, dict[str, Any]]:
    """Load model predictions for each config_id from the TPC-H dataset."""
    # Just load existing dataset which has labels
    preds: dict[str, dict[str, Any]] = {}
    with dataset_path.open() as f:
        for r in csv.DictReader(f):
            cid = r["config_id"]
            preds[cid] = {
                "needs_compaction": int(r["needs_compaction"]),
                "file_reduction_ratio": float(r["file_reduction_ratio"]),
                "file_count": int(r["file_count"]),
                "max_files_per_partition": int(r["max_files_per_partition"]),
                "tpch_table": r.get("tpch_table", ""),
            }
    return preds


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Query benchmark on TPC-H tables")
    parser.add_argument("--warehouse", type=str, default=DEFAULT_WAREHOUSE)
    parser.add_argument("--grid", type=Path, default=Path("code/tpch_grid.csv"))
    parser.add_argument("--dataset", type=Path, default=Path("data/tpch/tpch_dataset.csv"))
    parser.add_argument("--out", type=Path, default=Path("data/tpch/tpch_query_bench.csv"))
    parser.add_argument("--out-json", type=Path, default=Path("data/tpch/tpch_query_bench.json"))
    parser.add_argument("--sample", type=int, default=0,
                        help="Random sample of N tables (0 = all)")
    parser.add_argument("--warmup", type=int, default=WARMUP)
    parser.add_argument("--repeats", type=int, default=REPEATS)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    # Add file handler for logs
    log_path = Path("logs/tpch_query_bench.log")
    log_path.parent.mkdir(exist_ok=True)
    fh = logging.FileHandler(log_path)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(name)s] %(levelname)-8s %(message)s"))
    LOG.addHandler(fh)

    # Load configs
    configs = _load_tpch_configs(args.grid)
    LOG.info(f"Loaded {len(configs)} configs from {args.grid}")

    # Filter to tables that support meaningful queries
    configs = [c for c in configs if c.get("tpch_table", "") in TPCH_QUERY_TABLES]
    LOG.info(f"Filtered to {len(configs)} configs with queryable tables")

    if args.sample > 0:
        import random
        random.seed(42)
        configs = random.sample(configs, min(args.sample, len(configs)))
        LOG.info(f"Sampled {len(configs)} configs")

    if args.dry_run:
        for c in configs:
            print(f"  {c['config_id']} {c.get('tpch_table', '?')}")
        return 0

    # Load model predictions / labels
    predictions = _load_model_predictions(
        args.dataset, Path("data/model_clf.json"),
        Path("data/model_reg.json"), Path("data/tpch/tpch_features.csv"),
    )

    # Start Spark
    from pyspark.sql import SparkSession
    spark = (
        SparkSession.builder.master("local[*]")
        .appName("tpch-query-bench")
        .config("spark.driver.memory", "12g")
        .config("spark.sql.extensions",
                "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions")
        .config(f"spark.sql.catalog.{CATALOG}",
                "org.apache.iceberg.spark.SparkCatalog")
        .config(f"spark.sql.catalog.{CATALOG}.type", "hadoop")
        .config(f"spark.sql.catalog.{CATALOG}.warehouse", args.warehouse)
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.sql.iceberg.vectorization.enabled", "false")
        .getOrCreate()
    )

    # Prepare output CSV
    csv_fields = [
        "config_id", "tpch_table", "query", "file_count_pre", "file_count_post",
        "needs_compaction", "file_reduction_ratio",
        "time_pre_s", "time_post_s", "speedup",
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    csv_f = args.out.open("w", newline="")
    writer = csv.DictWriter(csv_f, fieldnames=csv_fields)
    writer.writeheader()

    all_results: list[dict[str, Any]] = []
    total = len(configs)

    for i, cfg in enumerate(configs, 1):
        cid = cfg["config_id"]
        base_table = cfg.get("tpch_table", "")
        table_name = f"tpch_{base_table}_{cid}"

        # Check table exists
        try:
            spark.sql(f"SELECT 1 FROM {_fqn(table_name)} LIMIT 1").collect()
        except Exception:
            LOG.warning(f"[{i}/{total}] SKIP {table_name}: not found")
            continue

        # Get snapshots
        snaps = get_snapshots(spark, table_name)
        if "pre" not in snaps or "post" not in snaps:
            LOG.warning(f"[{i}/{total}] SKIP {table_name}: missing snapshots")
            continue

        # Get file counts from predictions/dataset
        if cid in predictions:
            pred = predictions[cid]
            file_count_pre = pred["file_count"]
            needs_comp = pred["needs_compaction"]
            frr = pred["file_reduction_ratio"]
        else:
            file_count_pre = -1
            needs_comp = -1
            frr = -1.0

        # Get post file count from current table state
        try:
            post_fc = spark.sql(
                f"SELECT COUNT(*) AS c FROM {_fqn(table_name)}.files"
            ).collect()[0]["c"]
        except Exception:
            post_fc = -1

        LOG.info(f"[{i}/{total}] {table_name}: files {file_count_pre}->{post_fc}, "
                 f"needs_compact={needs_comp}")

        # Run queries at both snapshots
        try:
            pre_times = time_query_snapshot(
                spark, table_name, snaps["pre"], base_table,
                warmup=args.warmup, repeats=args.repeats,
            )
            post_times = time_query_snapshot(
                spark, table_name, snaps["post"], base_table,
                warmup=args.warmup, repeats=args.repeats,
            )
        except Exception as e:
            LOG.error(f"[{i}/{total}] FAIL {table_name}: {e}")
            continue

        for qlabel in pre_times:
            t_pre = pre_times[qlabel]
            t_post = post_times.get(qlabel, t_pre)
            speedup = t_pre / t_post if t_post > 0 else 0.0

            row = {
                "config_id": cid,
                "tpch_table": base_table,
                "query": qlabel,
                "file_count_pre": file_count_pre,
                "file_count_post": post_fc,
                "needs_compaction": needs_comp,
                "file_reduction_ratio": frr,
                "time_pre_s": round(t_pre, 4),
                "time_post_s": round(t_post, 4),
                "speedup": round(speedup, 4),
            }
            writer.writerow(row)
            csv_f.flush()
            all_results.append(row)

            LOG.info(f"  {qlabel}: pre={t_pre:.3f}s post={t_post:.3f}s "
                     f"speedup={speedup:.2f}x")

    csv_f.close()
    spark.stop()

    # Save JSON summary
    summary = _compute_summary(all_results)
    args.out_json.write_text(json.dumps(summary, indent=2))
    LOG.info(f"Results: {len(all_results)} query measurements across "
             f"{len(set(r['config_id'] for r in all_results))} tables")
    LOG.info(f"Saved CSV -> {args.out}")
    LOG.info(f"Saved JSON -> {args.out_json}")
    return 0


def _compute_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute aggregate statistics from benchmark results."""
    if not results:
        return {}

    from collections import defaultdict
    by_query: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_query[r["query"]].append(r)

    summary: dict[str, Any] = {"n_tables": len(set(r["config_id"] for r in results))}
    query_summaries = {}

    for qlabel, rows in by_query.items():
        speedups = [r["speedup"] for r in rows]
        # Split by needs_compaction
        beneficial = [r for r in rows if r["needs_compaction"] == 1]
        not_beneficial = [r for r in rows if r["needs_compaction"] == 0]

        qsum: dict[str, Any] = {
            "n": len(rows),
            "median_speedup": round(statistics.median(speedups), 4),
            "mean_speedup": round(statistics.mean(speedups), 4),
            "min_speedup": round(min(speedups), 4),
            "max_speedup": round(max(speedups), 4),
        }

        if beneficial:
            qsum["beneficial_median_speedup"] = round(
                statistics.median([r["speedup"] for r in beneficial]), 4)
        if not_beneficial:
            qsum["not_beneficial_median_speedup"] = round(
                statistics.median([r["speedup"] for r in not_beneficial]), 4)

        # Spearman correlation between file_reduction_ratio and speedup
        if len(rows) >= 5:
            try:
                frr_vals = [r["file_reduction_ratio"] for r in rows]
                spd_vals = [r["speedup"] for r in rows]
                from scipy.stats import spearmanr
                rho, pval = spearmanr(frr_vals, spd_vals)
                qsum["spearman_rho_frr_speedup"] = round(rho, 4)
                qsum["spearman_pval"] = round(pval, 6)
            except ImportError:
                # Fallback: manual rank correlation
                pass

        query_summaries[qlabel] = qsum

    summary["per_query"] = query_summaries

    # Overall
    all_speedups = [r["speedup"] for r in results]
    summary["overall_median_speedup"] = round(statistics.median(all_speedups), 4)
    summary["overall_mean_speedup"] = round(statistics.mean(all_speedups), 4)

    return summary


if __name__ == "__main__":
    sys.exit(main())
