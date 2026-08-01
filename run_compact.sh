#!/usr/bin/env bash
# Retry loop: run compact.py --resume until all tables are done.
# When the JVM crashes (SIGSEGV), this loop restarts it automatically.
# The --resume flag skips already-compacted tables.
set -u

cd "$(dirname "$0")"

export JAVA_HOME=/usr/lib/jvm/java-21-amazon-corretto
export JAVA_TOOL_OPTIONS="-Xshare:off -XX:ErrorFile=logs/hs_err_pid%p.log -XX:-CreateCoredumpOnCrash -XX:+UnlockDiagnosticVMOptions -XX:-DumpReplayDataOnError"

LOGFILE="logs/compact_loop.log"
mkdir -p logs

MAX_ATTEMPTS=100
attempt=0
prev_count=0

while [ $attempt -lt $MAX_ATTEMPTS ]; do
    attempt=$((attempt + 1))
    count=$(tail -n+2 data/compaction.csv 2>/dev/null | wc -l)
    echo "[$(date)] Attempt $attempt — CSV has $count rows" | tee -a "$LOGFILE"

    # If no new rows were added in the last run, we're done
    if [ $attempt -gt 1 ] && [ "$count" -eq "$prev_count" ]; then
        echo "[$(date)] No progress since last attempt. Done." | tee -a "$LOGFILE"
        break
    fi
    prev_count=$count

    .venv/bin/python code/compact.py --resume 2>&1 | tee -a "$LOGFILE"
    exit_code=${PIPESTATUS[0]}

    echo "[$(date)] compact.py exited with code $exit_code" | tee -a "$LOGFILE"

    # Clean up stale JVM processes and spark temp dirs
    pkill -9 -f "compact-" 2>/dev/null
    rm -rf /tmp/spark-* 2>/dev/null

    sleep 2
done

final=$(tail -n+2 data/compaction.csv 2>/dev/null | wc -l)
echo "[$(date)] Finished after $attempt attempts. CSV has $final rows." | tee -a "$LOGFILE"
