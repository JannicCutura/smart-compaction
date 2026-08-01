#!/bin/bash
# Retry loop for generate.py — keeps restarting until all 2376 tables exist.
set -u

cd "$(dirname "$0")"

export JAVA_HOME=/usr/lib/jvm/java-21-amazon-corretto
WAREHOUSE=/mnt/data/warehouse/compaction
TARGET=2376
MAX_RETRIES=50

for attempt in $(seq 1 $MAX_RETRIES); do
    COUNT=$(ls "$WAREHOUSE" 2>/dev/null | wc -l)
    echo "[$(date)] Attempt $attempt: $COUNT/$TARGET tables on disk"

    if [[ $COUNT -ge $TARGET ]]; then
        echo "[$(date)] All $TARGET tables generated. Done."
        exit 0
    fi

    .venv/bin/python code/generate.py --csv code/grid.csv >> logs/generate.log 2>&1
    EXIT_CODE=$?

    NEW_COUNT=$(ls "$WAREHOUSE" 2>/dev/null | wc -l)
    PROGRESS=$((NEW_COUNT - COUNT))
    echo "[$(date)] Run exited ($EXIT_CODE). Progress: +$PROGRESS tables ($NEW_COUNT/$TARGET)"

    if [[ $PROGRESS -eq 0 ]]; then
        echo "[$(date)] No progress made — may be stuck. Waiting 10s before retry."
        sleep 10
    fi
done

echo "[$(date)] Hit max retries ($MAX_RETRIES). $(ls "$WAREHOUSE" | wc -l)/$TARGET tables generated."
exit 1
