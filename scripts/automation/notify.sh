#!/bin/bash
# Usage: notify.sh <run_id> <status> [message]
#   run_id: e.g. FFT-S2, CPT-S3, CHAIN
#   status: RUNNING | DONE | FAILED
#   message: optional descriptive text

RUN_ID="$1"
STATUS="$2"
MESSAGE="${3:-}"
TIMESTAMP=$(date -u +%Y-%m-%dT%H:%M:%SZ)
FLAG_DIR="/workspace/automation/flags"
CHAIN_LOG="/workspace/automation/logs/chain.log"

mkdir -p "$FLAG_DIR"

# Write flag file with timestamped status
{
    echo "status: $STATUS"
    echo "timestamp: $TIMESTAMP"
    echo "run_id: $RUN_ID"
    [ -n "$MESSAGE" ] && echo "message: $MESSAGE"
} > "$FLAG_DIR/${RUN_ID}.status"

# Also log to chain log
echo "[$TIMESTAMP] $RUN_ID: $STATUS ${MESSAGE:+— $MESSAGE}" >> "$CHAIN_LOG"
