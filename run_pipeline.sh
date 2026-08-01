#!/usr/bin/env bash
# Run the full downstream pipeline after generation completes.
# Usage: ./run_pipeline.sh
set -euo pipefail

export JAVA_HOME=/usr/lib/jvm/java-21-amazon-corretto
cd "$(dirname "$0")"

PY=".venv/bin/python"
LOG_DIR="/tmp"

echo "=== Step 1: Extract features ==="
$PY code/extract.py --csv code/grid.csv --out data/features.csv \
    2>&1 | tee "$LOG_DIR/extract.log"

echo ""
echo "=== Step 2: Snapshot warehouse ==="
if [ -d /mnt/data/warehouse_snapshot ]; then
    echo "Snapshot already exists; skipping."
else
    cp -a /mnt/data/warehouse /mnt/data/warehouse_snapshot
    echo "Snapshot saved."
fi

echo ""
echo "=== Step 3: Run compaction ==="
$PY code/compact.py --csv code/grid.csv --out data/compaction.csv \
    2>&1 | tee "$LOG_DIR/compact.log"

echo ""
echo "=== Step 4: Build labels ==="
$PY code/label.py 2>&1 | tee "$LOG_DIR/label.log"

echo ""
echo "=== Step 5: Train models ==="
$PY code/train.py 2>&1 | tee "$LOG_DIR/train.log"

echo ""
echo "=== Step 6: Evaluate ==="
$PY code/evaluate.py 2>&1 | tee "$LOG_DIR/evaluate.log"

echo ""
echo "=== Pipeline complete ==="
echo "Dataset: data/dataset.csv"
echo "Models: data/model_clf.json, data/model_reg.json"
echo "Plots: plots/"
