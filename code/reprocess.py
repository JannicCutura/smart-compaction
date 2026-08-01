#!/usr/bin/env python3
"""
Reprocess a subset of tables: delete, regenerate, extract, compact.

Takes a list of config IDs and runs the full pipeline for those tables
only, updating features.csv and compaction.csv in place. Useful for
fixing outliers or augmenting the parameter grid without regenerating
the entire dataset.

Steps per config ID:
  1. Remove existing rows from features.csv and compaction.csv
  2. DROP the Iceberg table
  3. Regenerate (reuses generate.py logic)
  4. Extract metadata features → append to features.csv
  5. Compact via isolated subprocess → append to compaction.csv

After all tables are reprocessed, optionally re-runs label → train →
evaluate to rebuild the dataset and models.

Usage:
    # Reprocess specific config IDs
    python code/reprocess.py fe4cf344 e6af09a3 c5fa7095

    # Read config IDs from a file (one per line)
    python code/reprocess.py --ids-file reprocess_ids.txt

    # Skip the label/train/evaluate rebuild
    python code/reprocess.py --no-rebuild fe4cf344

    # Dry run: show what would be done
    python code/reprocess.py --dry-run fe4cf344 e6af09a3
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from pyspark.sql import SparkSession

sys.path.insert(0, str(Path(__file__).resolve().parent))
from params import SimConfig  # noqa: E402

LOG = logging.getLogger("reprocess")

# ── Defaults ────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WAREHOUSE = "/mnt/data/warehouse"
CATALOG = "local"
DATABASE = "compaction"

FEATURES_CSV = ROOT / "data" / "features.csv"
COMPACTION_CSV = ROOT / "data" / "compaction.csv"
GRID_CSV = ROOT / "code" / "grid.csv"

PYTHON = str(ROOT / ".venv" / "bin" / "python")
COMPACT_ONE = str(ROOT / "code" / "compact_one.py")
LOGS = ROOT / "logs"

JAVA_HOME = "/usr/lib/jvm/java-21-amazon-corretto"
JAVA_OPTS = (
    f"-Xshare:off -XX:ErrorFile={LOGS}/hs_err_pid%p.log "
    "-XX:-CreateCoredumpOnCrash -XX:+UnlockDiagnosticVMOptions "
    "-XX:-DumpReplayDataOnError"
)
TIMEOUT_S = 600


# ── CSV surgery ─────────────────────────────────────────────────────────────


def remove_ids_from_csv(csv_path: Path, ids: set[str]) -> int:
    """Remove rows matching config IDs from a CSV. Returns count removed."""
    if not csv_path.exists():
        return 0

    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return 0
        fieldnames = list(reader.fieldnames)
        rows = list(reader)

    before = len(rows)
    rows = [r for r in rows if r["config_id"] not in ids]
    removed = before - len(rows)

    if removed > 0:
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        LOG.info("Removed %d rows from %s", removed, csv_path.name)

    return removed


# ── Config lookup ───────────────────────────────────────────────────────────


def load_configs_by_id(ids: set[str]) -> dict[str, SimConfig]:
    """Load SimConfig objects from grid.csv for the given config IDs."""
    configs: dict[str, SimConfig] = {}
    if not GRID_CSV.exists():
        LOG.error("Grid CSV not found at %s", GRID_CSV)
        return configs

    with GRID_CSV.open() as f:
        for row in csv.DictReader(f):
            cid = row["config_id"]
            if cid in ids:
                configs[cid] = SimConfig(
                    num_rows=int(row["num_rows"]),
                    num_columns=int(row["num_columns"]),
                    num_partitions=int(row["num_partitions"]),
                    num_writers=int(row["num_writers"]),
                    num_write_batches=int(row["num_write_batches"]),
                    file_size_target_kb=int(row["file_size_target_kb"]),
                    partition_skew=row["partition_skew"],
                )
    return configs


# ── Spark helpers ───────────────────────────────────────────────────────────


def get_spark(warehouse: str) -> SparkSession:
    """SparkSession for generate + extract (needs more memory than compact)."""
    return (
        SparkSession.builder.master("local[*]")
        .appName("iceberg-reprocess")
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
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.sql.iceberg.vectorization.enabled", "false")
        .getOrCreate()
    )


def _fqn(table_name: str) -> str:
    return f"{CATALOG}.{DATABASE}.{table_name}"


# ── Step 1: Delete ──────────────────────────────────────────────────────────


def drop_table(spark: SparkSession, config_id: str) -> None:
    """DROP TABLE IF EXISTS for the given config ID."""
    table_name = f"t_{config_id}"
    fqn = _fqn(table_name)
    spark.sql(f"DROP TABLE IF EXISTS {fqn}")
    LOG.info("Dropped %s", fqn)


# ── Step 2: Generate (reuses generate.py) ───────────────────────────────────


def regenerate_table(spark: SparkSession, cfg: SimConfig) -> dict[str, Any]:
    """Generate one Iceberg table. Imported from generate.py."""
    from generate import generate_table  # noqa: E402
    return generate_table(spark, cfg)


# ── Step 3: Extract (reuses extract.py) ─────────────────────────────────────


def extract_and_append(spark: SparkSession, cfg: SimConfig) -> dict[str, Any]:
    """Extract features for one table and append to features.csv."""
    from extract import FEATURE_COLUMNS, extract_features  # noqa: E402

    features = extract_features(spark, cfg)

    write_header = (
        not FEATURES_CSV.exists() or FEATURES_CSV.stat().st_size == 0
    )
    with FEATURES_CSV.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FEATURE_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(features)

    LOG.info("Appended features for %s", cfg.config_id)
    return features


# ── Step 4: Compact (subprocess for JVM isolation) ──────────────────────────


def compact_via_subprocess(config_id: str) -> str:
    """Run compact_one.py in an isolated subprocess. Returns status."""
    env = os.environ.copy()
    env["JAVA_HOME"] = JAVA_HOME
    env["JAVA_TOOL_OPTIONS"] = JAVA_OPTS

    LOGS.mkdir(exist_ok=True)
    stdout_path = LOGS / f"reprocess_{config_id}.out"
    stderr_path = LOGS / f"reprocess_{config_id}.err"

    try:
        with stdout_path.open("w") as fout, stderr_path.open("w") as ferr:
            proc = subprocess.run(
                [PYTHON, COMPACT_ONE, config_id, str(COMPACTION_CSV)],
                stdout=fout,
                stderr=ferr,
                timeout=TIMEOUT_S,
                env=env,
                cwd=str(ROOT),
            )
        stdout_text = stdout_path.read_text().strip()
        for line in reversed(stdout_text.splitlines()):
            if line.startswith(("OK:", "FAIL:", "SKIP:")):
                stdout_path.unlink(missing_ok=True)
                stderr_path.unlink(missing_ok=True)
                return line
        if proc.returncode != 0:
            return f"CRASH:{config_id}:exit={proc.returncode}"
        return f"UNKNOWN:{config_id}:no_status_line"
    except subprocess.TimeoutExpired:
        return f"TIMEOUT:{config_id}"
    except Exception as e:
        return f"ERROR:{config_id}:{e}"
    finally:
        subprocess.run(
            ["pkill", "-f", f"compact-{config_id}"],
            capture_output=True, timeout=5,
        )
        for d in Path("/tmp").glob("spark-*"):
            try:
                if d.is_dir():
                    shutil.rmtree(d, ignore_errors=True)
            except Exception:
                pass


# ── Rebuild downstream ──────────────────────────────────────────────────────


def rebuild_downstream() -> None:
    """Re-run label → train → evaluate to rebuild dataset and models."""
    for script, desc in [
        ("code/label.py", "label"),
        ("code/train.py", "train"),
        ("code/evaluate.py", "evaluate"),
    ]:
        LOG.info("Running %s ...", desc)
        result = subprocess.run(
            [PYTHON, str(ROOT / script)],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            LOG.error("%s failed:\n%s", desc, result.stderr[-500:])
        else:
            LOG.info("%s complete", desc)


# ── CLI ─────────────────────────────────────────────────────────────────────


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Reprocess specific tables: delete → generate → extract → compact",
    )
    p.add_argument(
        "config_ids",
        nargs="*",
        help="Config IDs to reprocess (space-separated)",
    )
    p.add_argument(
        "--ids-file",
        type=Path,
        default=None,
        help="File with one config ID per line (comments with # allowed)",
    )
    p.add_argument(
        "--warehouse",
        type=str,
        default=DEFAULT_WAREHOUSE,
        help=f"Iceberg warehouse path (default: {DEFAULT_WAREHOUSE})",
    )
    p.add_argument(
        "--no-rebuild",
        action="store_true",
        help="Skip label/train/evaluate rebuild after reprocessing",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)-5s  %(message)s",
    )

    # ── Collect config IDs ──────────────────────────────────────────────
    ids: set[str] = set(args.config_ids or [])
    if args.ids_file:
        if not args.ids_file.exists():
            LOG.error("IDs file not found: %s", args.ids_file)
            return 1
        for line in args.ids_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                ids.add(line.split()[0])  # first token on line

    if not ids:
        LOG.error("No config IDs supplied. Pass them as arguments or via --ids-file.")
        return 1

    # ── Look up configs in grid ─────────────────────────────────────────
    configs = load_configs_by_id(ids)
    missing = ids - set(configs.keys())
    if missing:
        LOG.warning("Config IDs not found in grid.csv: %s", ", ".join(sorted(missing)))
    if not configs:
        LOG.error("No valid config IDs to process")
        return 1

    LOG.info("Reprocessing %d tables: %s", len(configs), " ".join(sorted(configs)))

    # ── Dry run ─────────────────────────────────────────────────────────
    if args.dry_run:
        for cid, cfg in sorted(configs.items()):
            print(
                f"  {cid}  rows={cfg.num_rows:>10,}  cols={cfg.num_columns:>2}  "
                f"parts={cfg.num_partitions:>3}  writers={cfg.num_writers}  "
                f"batches={cfg.num_write_batches:>2}  file_kb={cfg.file_size_target_kb:>7,}  "
                f"skew={cfg.partition_skew}"
            )
        print(f"\nWould delete, regenerate, extract, and compact {len(configs)} tables.")
        if not args.no_rebuild:
            print("Would rebuild: label → train → evaluate")
        return 0

    # ── Step 0: Purge old rows from CSVs ────────────────────────────────
    id_set = set(configs.keys())
    remove_ids_from_csv(FEATURES_CSV, id_set)
    remove_ids_from_csv(COMPACTION_CSV, id_set)
    LOG.info("CSV cleanup complete")

    # ── Steps 1-3: Drop, generate, extract (single Spark session) ───────
    spark = get_spark(args.warehouse)
    spark.sql(f"CREATE NAMESPACE IF NOT EXISTS {CATALOG}.{DATABASE}")

    gen_ok = 0
    ext_ok = 0
    gen_fail: list[str] = []

    try:
        for i, (cid, cfg) in enumerate(sorted(configs.items()), 1):
            LOG.info("=== [%d/%d] %s ===", i, len(configs), cid)

            # Drop
            drop_table(spark, cid)

            # Generate
            try:
                result = regenerate_table(spark, cfg)
                gen_ok += 1
                LOG.info(
                    "  Generated: %d files, %d rows, %.1fs",
                    result["file_count"], result["row_count"], result["elapsed_s"],
                )
            except Exception:
                LOG.exception("  FAILED to generate %s", cid)
                gen_fail.append(cid)
                continue

            # Extract
            try:
                extract_and_append(spark, cfg)
                ext_ok += 1
            except Exception:
                LOG.exception("  FAILED to extract %s", cid)
    finally:
        spark.stop()

    # ── Step 4: Compact (subprocess per table for JVM isolation) ────────
    compact_ok = 0
    compact_fail: list[str] = []
    compactable = [cid for cid in sorted(configs) if cid not in gen_fail]

    for i, cid in enumerate(compactable, 1):
        LOG.info("=== Compact [%d/%d] %s ===", i, len(compactable), cid)
        t0 = time.time()
        status = compact_via_subprocess(cid)
        elapsed = time.time() - t0
        tag = status.split(":")[0]
        LOG.info("  %s  (%.1fs)", status, elapsed)
        if tag == "OK":
            compact_ok += 1
        else:
            compact_fail.append(cid)

    # ── Summary ─────────────────────────────────────────────────────────
    LOG.info("=== Reprocess Summary ===")
    LOG.info("Generated: %d/%d", gen_ok, len(configs))
    LOG.info("Extracted: %d/%d", ext_ok, len(configs))
    LOG.info("Compacted: %d/%d", compact_ok, len(compactable))
    if gen_fail:
        LOG.warning("Generate failures: %s", ", ".join(gen_fail))
    if compact_fail:
        LOG.warning("Compact failures: %s", ", ".join(compact_fail))

    # ── Rebuild downstream ──────────────────────────────────────────────
    if not args.no_rebuild and gen_ok > 0:
        LOG.info("Rebuilding downstream: label → train → evaluate")
        rebuild_downstream()
    elif args.no_rebuild:
        LOG.info("Skipping rebuild (--no-rebuild). Run manually:")
        LOG.info("  python code/label.py && python code/train.py && python code/evaluate.py")

    return 1 if (gen_fail or compact_fail) else 0


if __name__ == "__main__":
    raise SystemExit(main())
