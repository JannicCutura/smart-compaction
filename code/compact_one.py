#!/usr/bin/env python3
"""Compact a single Iceberg table in an isolated JVM.

Called by compact_runner.py via subprocess so JVM crashes
don't kill the orchestrator.
"""
import csv
import sys
import time
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

CATALOG = "local"
DATABASE = "compaction"
WAREHOUSE = "/mnt/data/warehouse"
DEFAULT_TARGET_MB = 128


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Compact one Iceberg table")
    parser.add_argument("config_id", help="Config ID of table to compact")
    parser.add_argument("--out", type=Path, default=Path("data/compaction.csv"),
                        help="Output CSV path")
    parser.add_argument("--target-mb", type=int, default=DEFAULT_TARGET_MB,
                        help="Compaction target file size in MB (default: 128)")
    parser.add_argument("--rollback-first", action="store_true",
                        help="Rollback to first (append) snapshot before compacting")
    args = parser.parse_args()

    config_id = args.config_id
    out_path = args.out
    rollback_first = args.rollback_first
    REWRITE_TARGET_BYTES = args.target_mb * 1024 * 1024

    table_name = f"t_{config_id}"
    fqn = f"{CATALOG}.{DATABASE}.{table_name}"

    spark = (
        SparkSession.builder.master("local[1]")
        .appName(f"compact-{config_id}")
        .config("spark.driver.memory", "8g")
        .config("spark.sql.extensions",
                "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions")
        .config(f"spark.sql.catalog.{CATALOG}",
                "org.apache.iceberg.spark.SparkCatalog")
        .config(f"spark.sql.catalog.{CATALOG}.type", "hadoop")
        .config(f"spark.sql.catalog.{CATALOG}.warehouse", WAREHOUSE)
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.sql.iceberg.vectorization.enabled", "false")
        .getOrCreate()
    )

    try:
        # Check table exists
        spark.sql(f"SELECT 1 FROM {fqn} LIMIT 1").collect()
    except Exception:
        print(f"SKIP:{config_id}:table_not_found", flush=True)
        spark.stop()
        return 2

    if rollback_first:
        try:
            # Find the first (append) snapshot
            snapshots = spark.sql(f"SELECT snapshot_id, operation FROM {fqn}.snapshots ORDER BY committed_at").collect()
            append_snap = None
            for s in snapshots:
                if s["operation"] == "append":
                    append_snap = s["snapshot_id"]
                    break
            if append_snap is None:
                print(f"SKIP:{config_id}:no_append_snapshot", flush=True)
                spark.stop()
                return 2
            spark.sql(f"CALL {CATALOG}.system.rollback_to_snapshot('{DATABASE}.{table_name}', {append_snap})")
            # Expire non-current snapshots and GC orphan files to reclaim disk
            import datetime
            now_ms = int(datetime.datetime.now().timestamp() * 1000)
            spark.sql(f"""
                CALL {CATALOG}.system.expire_snapshots(
                    table => '{DATABASE}.{table_name}',
                    older_than => TIMESTAMP '{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
                    retain_last => 1
                )
            """)
            spark.sql(f"""
                CALL {CATALOG}.system.remove_orphan_files(
                    table => '{DATABASE}.{table_name}'
                )
            """)
        except Exception as e:
            print(f"FAIL:{config_id}:rollback:{type(e).__name__}:{str(e)[:200]}", flush=True)
            spark.stop()
            return 1

    try:
        t0 = time.time()
        result = spark.sql(f"""
            CALL {CATALOG}.system.rewrite_data_files(
                table => '{DATABASE}.{table_name}',
                options => map(
                    'target-file-size-bytes', '{REWRITE_TARGET_BYTES}',
                    'min-file-size-bytes',    '{int(REWRITE_TARGET_BYTES * 0.75)}',
                    'max-file-size-bytes',    '{int(REWRITE_TARGET_BYTES * 1.8)}'
                )
            )
        """)
        row = result.collect()[0]
        duration = time.time() - t0
        rewritten = row["rewritten_data_files_count"]
        added = row["added_data_files_count"]

        # Get after stats
        after = (
            spark.sql(f"SELECT * FROM {fqn}.files")
            .agg(
                F.count("*").alias("fc"),
                F.sum("file_size_in_bytes").alias("ts"),
                F.avg("file_size_in_bytes").alias("avg"),
            )
            .collect()[0]
        )

        row_data = {
            "config_id": config_id,
            "after_file_count": after["fc"],
            "after_total_size_bytes": int(after["ts"] or 0),
            "after_avg_file_size_bytes": round(after["avg"] or 0, 2),
            "rewritten_data_files_count": rewritten,
            "added_data_files_count": added,
            "compaction_duration_s": round(duration, 2),
        }

        # Append to CSV
        cols = list(row_data.keys())
        write_header = not out_path.exists() or out_path.stat().st_size == 0
        with out_path.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            if write_header:
                w.writeheader()
            w.writerow(row_data)

        print(f"OK:{config_id}:rewritten={rewritten},added={added},time={duration:.1f}s", flush=True)
    except Exception as e:
        print(f"FAIL:{config_id}:{type(e).__name__}:{str(e)[:200]}", flush=True)
        spark.stop()
        return 1

    spark.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
