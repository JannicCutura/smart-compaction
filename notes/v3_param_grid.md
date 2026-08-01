# v3 Parameter Grid (Future Work)

Based on deep research into real-world Iceberg/Delta/Hudi table characteristics.
Source: `agents/compass_artifact_wf-7c0ebda1-450c-4256-b04b-69bdef775739_text_markdown.md`

## Proposed Axes

| Axis | Values | Count | Rationale |
|------|--------|-------|-----------|
| `num_rows` | [100K, 1M, 10M, 50M] | 4 | Drops degenerate 10K; 50M rows ≈ 75 GB at 50 cols |
| `num_columns` | [10, 50, 150] | 3 | Narrow, standard, wide tables |
| `num_partitions` | [1, 10, 100, 1000] | 4 | Unpartitioned through heavily partitioned |
| `num_writers` | [1, 5] | 2 | Keep current |
| `num_write_batches` | [1, 5, 20] | 3 | Keep current |
| `file_size_target_kb` | [32, 8192, 131072, 524288] | 4 | 32 KB stress, 8 MB small, 128 MB streaming, 512 MB default |
| `partition_skew` | ["uniform", "zipf"] | 2 | Keep current |
| `data_types` | ["uuid_strings", "mixed"] | 2 | NEW: controls compression behavior |

**Raw total: 4,608 configs.** After pruning degenerate combos: ~2,500–3,000.

## Key Changes from v2

1. **Compaction target → 512 MB** (min=384 MB, max=922 MB) to match Iceberg default
2. **Row counts 10–50x larger** — current max 1M rows produces only 3–12 files at target size
3. **512 MB file size target** — the Iceberg default, currently missing
4. **1000 partitions** — daily-partitioned tables over 3 years
5. **Mixed data types** — real tables compress 2–3x better than UUID-only

## Pruning Rules

- Drop configs where total data < 2x compaction target (nothing to compact)
- Drop `num_rows=100K × num_columns=10 × file_size_target=512MB` (1 file total)
- Drop `num_partitions=1000 × num_rows=100K` (~100 rows/partition)

## Disk Estimate

10M-row tables at 50 cols ≈ 15 GB each. Full grid would need 12–25 TB.
Options: sample the grid, use `/mnt/data` (984 GB), or attach more EBS.

## Priority 3 (Future Future Work)

- Delete files axis: [0, "low", "high"]
- Compression codec: ["zstd", "snappy"]
- Zipf exponent variation: [s=0.5, s=1.0, s=1.5]
- Sort order: ["unsorted", "sorted_by_partition_key"]
