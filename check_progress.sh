#!/bin/bash
# Quick progress check for any running pipeline stage
echo "=== $(date) ==="

# Generation progress
if [[ -f logs/generate.log ]]; then
    ON_DISK=$(ls /mnt/data/warehouse/compaction/ 2>/dev/null | wc -l)
    LAST=$(tail -1 logs/generate.log 2>/dev/null)
    echo "Generate: $ON_DISK/2376 tables on disk"
    echo "  Last: $LAST"
fi

# Extraction progress
if [[ -f data/features.csv ]]; then
    ROWS=$(wc -l < data/features.csv)
    echo "Extract: $((ROWS - 1)) features extracted"
fi

# Compaction progress
if [[ -f data/compaction.csv ]]; then
    ROWS=$(wc -l < data/compaction.csv)
    echo "Compact: $((ROWS - 1)) tables compacted"
fi

# Resource usage
echo ""
free -h | head -2
df -h /mnt/data | tail -1

# Active Spark processes
SPARK_COUNT=$(pgrep -f SparkSubmit | wc -l)
echo "Spark JVMs running: $SPARK_COUNT"
