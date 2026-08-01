#!/usr/bin/env python3
"""Orchestrator: compact remaining tables, one JVM per table.

Spawns a subprocess per table so JVM crashes are isolated.
Reads existing compaction.csv to skip already-done tables.
"""
import csv
import os
import subprocess
import sys
import time
from pathlib import Path

GRID = Path("code/grid.csv")
OUT = Path("data/compaction.csv")
PYTHON = str(Path(".venv/bin/python"))
WORKER = str(Path("code/compact_one.py"))
TIMEOUT_S = 600  # 10 min per table max
DEFAULT_TARGET_MB = 128

JAVA_HOME = "/usr/lib/jvm/java-21-amazon-corretto"
LOGS = Path("logs")
JAVA_OPTS = f"-Xshare:off -XX:ErrorFile={LOGS}/hs_err_pid%p.log -XX:-CreateCoredumpOnCrash -XX:+UnlockDiagnosticVMOptions -XX:-DumpReplayDataOnError"

SKIP_IDS: set[str] = set()  # vectorization fix resolved all JVM crashes


def load_done_ids() -> set[str]:
    if not OUT.exists():
        return set()
    with OUT.open() as f:
        return {r["config_id"] for r in csv.DictReader(f)}


def load_remaining() -> list[str]:
    done = load_done_ids()
    remaining = []
    with GRID.open() as f:
        for r in csv.DictReader(f):
            cid = r["config_id"]
            if cid not in done and cid not in SKIP_IDS:
                remaining.append(cid)
    return remaining


def compact_one(config_id: str, target_mb: int = DEFAULT_TARGET_MB,
               rollback_first: bool = False) -> str:
    """Run compact_one.py in subprocess. Returns status string."""
    env = os.environ.copy()
    env["JAVA_HOME"] = JAVA_HOME
    env["JAVA_TOOL_OPTIONS"] = JAVA_OPTS

    LOGS.mkdir(exist_ok=True)
    stdout_path = LOGS / f"compact_sub_{config_id}.out"
    stderr_path = LOGS / f"compact_sub_{config_id}.err"

    try:
        cmd = [PYTHON, WORKER, config_id, "--out", str(OUT),
               "--target-mb", str(target_mb)]
        if rollback_first:
            cmd.append("--rollback-first")
        with stdout_path.open("w") as fout, stderr_path.open("w") as ferr:
            proc = subprocess.run(
                cmd,
                stdout=fout,
                stderr=ferr,
                timeout=TIMEOUT_S,
                env=env,
                cwd=str(Path.cwd()),
            )
        # Parse last meaningful line from stdout
        stdout_text = stdout_path.read_text().strip()
        for line in reversed(stdout_text.splitlines()):
            if line.startswith(("OK:", "FAIL:", "SKIP:")):
                # Clean up output files on success
                stdout_path.unlink(missing_ok=True)
                stderr_path.unlink(missing_ok=True)
                return line
        # No status line -- check for JVM crash
        if proc.returncode != 0:
            return f"CRASH:{config_id}:exit={proc.returncode}"
        return f"UNKNOWN:{config_id}:no_status_line"
    except subprocess.TimeoutExpired:
        return f"TIMEOUT:{config_id}"
    except Exception as e:
        return f"ERROR:{config_id}:{e}"
    finally:
        # Clean up stale JVM processes
        subprocess.run(["pkill", "-f", f"compact-{config_id}"],
                       capture_output=True, timeout=5)
        # Clean up spark temp dirs
        for d in Path("/tmp").glob("spark-*"):
            try:
                if d.is_dir():
                    import shutil
                    shutil.rmtree(d, ignore_errors=True)
            except Exception:
                pass


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Orchestrate compaction runs")
    parser.add_argument("--target-mb", type=int, default=DEFAULT_TARGET_MB,
                        help="Compaction target file size in MB (default: 128)")
    parser.add_argument("--rollback-first", action="store_true",
                        help="Rollback tables to first snapshot before compacting")
    parser.add_argument("--out", type=Path, default=None,
                        help="Output CSV path (overrides default)")
    parser.add_argument("--grid", type=Path, default=None,
                        help="Grid CSV path (overrides default)")
    cli_args = parser.parse_args()

    global OUT, GRID
    if cli_args.out:
        OUT = cli_args.out
    if cli_args.grid:
        GRID = cli_args.grid

    target_mb = cli_args.target_mb
    rollback_first = cli_args.rollback_first

    remaining = load_remaining()
    total = len(remaining)
    done_before = len(load_done_ids())
    print(f"Already done: {done_before}, Remaining: {total}", flush=True)
    print(f"Compaction target: {target_mb} MB", flush=True)

    ok = 0
    skipped = 0
    failed = 0
    crashed = 0
    crash_ids: list[str] = []

    for i, cid in enumerate(remaining, 1):
        t0 = time.time()
        status = compact_one(cid, target_mb=target_mb,
                             rollback_first=rollback_first)
        elapsed = time.time() - t0

        tag = status.split(":")[0]
        print(f"[{i}/{total}] {status}  ({elapsed:.1f}s)", flush=True)

        if tag == "OK":
            ok += 1
        elif tag == "SKIP":
            skipped += 1
        elif tag == "FAIL":
            failed += 1
        elif tag in ("CRASH", "TIMEOUT", "ERROR"):
            crashed += 1
            crash_ids.append(cid)
        else:
            failed += 1

    done_after = len(load_done_ids())
    print(f"\n=== Summary ===", flush=True)
    print(f"New: {ok} | Skipped: {skipped} | Failed: {failed} | Crashed: {crashed}", flush=True)
    print(f"Total in CSV: {done_after}", flush=True)
    if crash_ids:
        print(f"Crashed IDs: {','.join(crash_ids)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
