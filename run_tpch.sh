#!/usr/bin/env bash
# Run the full TPC-H cross-schema validation pipeline.
#
# Steps:
#   1. Generate TPC-H Iceberg tables (various write patterns)
#   2. Extract features, run compaction, apply model, report metrics
#
# Usage:
#   ./run_tpch.sh              # full grid (96 tables)
#   ./run_tpch.sh --sample 10  # quick smoke test (10 tables)
set -euo pipefail

export JAVA_HOME=/usr/lib/jvm/java-21-amazon-corretto
cd "$(dirname "$0")"

PY=".venv/bin/python"
LOG_DIR="/tmp"
WAREHOUSE="/mnt/data/tpch_warehouse"
SAMPLE_ARGS="${@}"

echo "============================================"
echo "  TPC-H Cross-Schema Validation Pipeline"
echo "============================================"
echo ""

echo "=== Step 1: Generate TPC-H tables ==="
$PY code/tpch_gen.py --warehouse "$WAREHOUSE" $SAMPLE_ARGS \
    2>&1 | tee "$LOG_DIR/tpch_gen.log"

echo ""
echo "=== Step 2: Validate (extract → compact → predict) ==="
$PY code/tpch_validate.py --warehouse "$WAREHOUSE" $SAMPLE_ARGS \
    2>&1 | tee "$LOG_DIR/tpch_validate.log"

echo ""
echo "============================================"
echo "  Pipeline complete"
echo "============================================"
echo "Results: data/tpch/"
echo "  tpch_features.csv           — 17 metadata features"
echo "  tpch_compaction.csv         — compaction outcomes"
echo "  tpch_dataset.csv            — labelled dataset"
echo "  tpch_validation_metrics.json — generalization metrics"
