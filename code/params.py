#!/usr/bin/env python3
"""
Simulation parameter grid for synthetic Iceberg table generation.

Defines the axes of variation for the compaction-utility experiment,
generates the full or sampled parameter grid, and exports a LaTeX table
describing the parameters for inclusion in the paper.

Design rationale (v3)
---------------------
The grid is designed so that resulting Iceberg data files span the full
range of the compaction filter:

  * well below the 128 MB target  →  always compacted
  * straddling the 96 MB min-threshold  →  borderline
  * at or above 128 MB  →  skipped by compaction

To achieve this we vary *row count × schema width* together with the
Parquet ``write.target-file-size-bytes`` so that per-writer-per-partition
file sizes range from single-digit KB to hundreds of MB.

v3 adds eight intermediate ``file_size_target_kb`` values (128 KB to
64 MB).  These place the 0.75 × 128 MB compaction threshold *within*
the natural file-size distribution of many row-count / column-count
combinations, producing partial-compaction outcomes (rfc in 0.1–0.7)
that were absent in v2's bimodal distribution.

Multi-batch tables alternate the target file size between batches
(``batch_target_alternation``) to mimic real tables that accumulate
data from heterogeneous pipelines.  When enabled, odd-indexed batches
use the configured target, while even-indexed batches use a target 8×
smaller (clamped to 8 KB min), creating within-table file-size
heterogeneity.

Partition skew follows a Zipf(s = 1) distribution when ``partition_skew``
is ``"zipf"``, which matches empirically observed access patterns better
than a crude 80/20 split.

Usage:
    python code/params.py                       # print grid stats + LaTeX
    python code/params.py --sample 50           # random sample of 50
    python code/params.py --csv code/grid.csv   # dump full grid to CSV
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import random
import sys
import textwrap
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# ── Parameter axes ──────────────────────────────────────────────────────────

PARAM_AXES: dict[str, list[Any]] = {
    # Table size (total rows across all partitions & batches)
    "num_rows":           [10_000, 100_000, 1_000_000],
    # Schema width (number of payload columns, excl. partition key).
    # 50 cols × ~30 B/col in Parquet ≈ 1.5 KB/row — needed to produce
    # files large enough to straddle the 128 MB compaction target.
    "num_columns":        [5, 50],
    # Partition count (1 = unpartitioned).
    # 100 partitions is realistic for date-based schemes (3+ years of days).
    "num_partitions":     [1, 10, 100],
    # Concurrent writers per batch (each produces 1 file per partition).
    "num_writers":        [1, 5],
    # Number of append rounds.
    "num_write_batches":  [1, 5, 20],
    # Target Parquet file size in KB.
    #     8 KB — extreme fragmentation (CDC / micro-batch)
    #   128 KB — small streaming batches
    #   512 KB — small-file accumulation
    #     2 MB — borderline micro-batch output
    #     4 MB — small-to-medium transition
    #     8 MB — medium files, partial compaction zone
    #    16 MB — medium files, partial compaction zone
    #    24 MB — straddles compaction threshold for small tables
    #    32 MB — below compaction threshold (32_768 KB)
    #    64 MB — near-threshold, partial compaction for large tables
    #   128 MB — production target, at compaction threshold (131_072 KB)
    "file_size_target_kb": [
        8, 128, 512,
        2_048, 4_096, 8_192, 16_384, 24_576,
        32_768, 65_536, 131_072,
    ],
    # Partition skew profile.
    #   "uniform" — equal row counts per partition.
    #   "zipf"    — Zipf(s=1); partition 0 is hottest.
    "partition_skew":     ["uniform", "zipf"],
}

# Human-readable descriptions for the LaTeX table
PARAM_DESCRIPTIONS: dict[str, str] = {
    "num_rows":           "Total rows per table",
    "num_columns":        "Payload columns (excl.\\ partition key)",
    "num_partitions":     "Number of partitions (1 = none)",
    "num_writers":        "Concurrent writers per batch",
    "num_write_batches":  "Number of append rounds",
    "file_size_target_kb": "Target file size (KB)",
    "partition_skew":     "Partition skew profile",
}


# ── Config dataclass ────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SimConfig:
    """One point in the parameter grid."""
    num_rows: int
    num_columns: int
    num_partitions: int
    num_writers: int
    num_write_batches: int
    file_size_target_kb: int
    partition_skew: str

    @property
    def config_id(self) -> str:
        """Deterministic short hash for this config (first 8 hex chars)."""
        raw = json.dumps(asdict(self), sort_keys=True)
        return hashlib.sha256(raw.encode()).hexdigest()[:8]

    @property
    def expected_file_count(self) -> int:
        """Upper bound on data files before compaction."""
        return self.num_writers * self.num_write_batches * max(self.num_partitions, 1)

    def batch_target_kb(self, batch_idx: int) -> int:
        """File-size target for a specific batch index.

        When there are multiple batches, even-indexed batches (0-based)
        use a target 8× smaller than the configured value (min 8 KB).
        This creates within-table file-size heterogeneity that mimics
        production tables fed by multiple pipelines.

        Batch 0 always uses the configured target (the "initial load").
        """
        if self.num_write_batches <= 1 or batch_idx == 0:
            return self.file_size_target_kb
        if batch_idx % 2 == 0:
            return self.file_size_target_kb  # odd batches (1-indexed): base
        else:
            return max(8, self.file_size_target_kb // 8)  # even: smaller

    def batch_rows(self, batch_idx: int) -> int:
        """Row count for a specific batch index.

        Batch 0 gets 50 % of total rows (big initial load).
        Remaining batches split the other 50 % equally, simulating
        smaller incremental ingestion.
        """
        if self.num_write_batches <= 1:
            return self.num_rows
        if batch_idx == 0:
            return self.num_rows // 2
        remaining = self.num_rows - self.num_rows // 2
        per_batch = max(1, remaining // (self.num_write_batches - 1))
        # Last batch absorbs rounding remainder
        if batch_idx == self.num_write_batches - 1:
            used = self.num_rows // 2 + per_batch * (self.num_write_batches - 2)
            return self.num_rows - used
        return per_batch

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["config_id"] = self.config_id
        d["expected_file_count"] = self.expected_file_count
        return d


# ── Grid generation ─────────────────────────────────────────────────────────

def full_grid() -> list[SimConfig]:
    """Cartesian product of all parameter axes."""
    keys = list(PARAM_AXES.keys())
    configs = []
    for combo in itertools.product(*(PARAM_AXES[k] for k in keys)):
        configs.append(SimConfig(**dict(zip(keys, combo))))
    return configs


def sampled_grid(n: int, seed: int = 42) -> list[SimConfig]:
    """Random sample of n configs from the full grid."""
    grid = full_grid()
    if n >= len(grid):
        return grid
    rng = random.Random(seed)
    return rng.sample(grid, n)


# ── LaTeX table export ──────────────────────────────────────────────────────

def generate_latex_table(outpath: Path) -> None:
    """Write a LaTeX table describing the parameter axes."""
    lines = [
        r"\begin{table}[t]",
        r"\caption{Simulation Parameter Grid}",
        r"\label{tab:params}",
        r"\begin{center}",
        r"\footnotesize",
        r"\begin{tabular}{lp{2.2cm}l}",
        r"\toprule",
        r"\textbf{Parameter} & \textbf{Description} & \textbf{Values} \\",
        r"\midrule",
    ]

    for param, values in PARAM_AXES.items():
        desc = PARAM_DESCRIPTIONS[param]
        # Format values
        formatted = []
        for v in values:
            if isinstance(v, int) and v >= 1_000_000:
                formatted.append(f"{v // 1_000_000}M")
            elif isinstance(v, int) and v >= 1_000:
                formatted.append(f"{v // 1_000}K")
            else:
                formatted.append(str(v))
        val_str = ", ".join(formatted)
        # Escape underscores for LaTeX
        param_tex = param.replace("_", r"\_")
        lines.append(
            rf"\texttt{{{param_tex}}} & {desc} & {{{val_str}}} \\"
        )

    total = len(full_grid())
    lines += [
        r"\midrule",
        rf"\multicolumn{{3}}{{l}}{{\textit{{Full grid: {total} configurations}}}} \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{center}",
        r"\end{table}",
    ]

    outpath.parent.mkdir(parents=True, exist_ok=True)
    outpath.write_text("\n".join(lines) + "\n")
    print(f"LaTeX table written to {outpath}")


# ── CSV export ──────────────────────────────────────────────────────────────

def export_csv(configs: list[SimConfig], outpath: Path) -> None:
    """Dump configs to CSV for downstream consumption."""
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fields = list(PARAM_AXES.keys()) + ["config_id", "expected_file_count"]
    with outpath.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for c in configs:
            writer.writerow(c.to_dict())
    print(f"CSV written to {outpath}  ({len(configs)} configs)")


# ── Main ────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Simulation parameter grid")
    parser.add_argument("--sample", type=int, default=0,
                        help="Random-sample N configs (0 = full grid)")
    parser.add_argument("--csv", type=str, default=None,
                        help="Path to write CSV grid")
    parser.add_argument("--latex", type=str,
                        default="paper/tables/params.tex",
                        help="Path to write LaTeX table")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for sampling")
    args = parser.parse_args()

    # Build grid
    if args.sample > 0:
        configs = sampled_grid(args.sample, seed=args.seed)
        print(f"Sampled {len(configs)} configs from full grid of {len(full_grid())}")
    else:
        configs = full_grid()
        print(f"Full grid: {len(configs)} configs")

    # Stats
    file_counts = [c.expected_file_count for c in configs]
    print(f"Expected file counts: min={min(file_counts)}, "
          f"max={max(file_counts)}, "
          f"median={sorted(file_counts)[len(file_counts)//2]}")

    # Disk-usage estimate
    avg_row_bytes = {5: 200, 50: 1500}  # approximate Parquet bytes/row
    total_bytes = 0
    for c in configs:
        total_bytes += c.num_rows * avg_row_bytes.get(c.num_columns, 800)
    print(f"Estimated total disk: {total_bytes / 1e9:.1f} GB")

    # LaTeX table (always write — describes axes, not individual configs)
    generate_latex_table(Path(args.latex))

    # Optional CSV
    if args.csv:
        export_csv(configs, Path(args.csv))


if __name__ == "__main__":
    main()
