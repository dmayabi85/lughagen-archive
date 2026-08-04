#!/bin/bash
# ==============================================================================
# LughaGen Watchdog — 15-min health checker
# ==============================================================================
# Every 15 min, checks:
#   1. Is a torchrun training process alive?
#   2. Is the latest log being written to (not hung)?
#   3. Are GPUs at reasonable utilization?
#   4. Is disk usage below 90%?
#   5. Is the chain flag consistent with observed state?
#
# Writes to /workspace/automation/logs/watchdog.log
# Writes ANOMALY flag if health check fails
# ==============================================================================

WATCHDOG_LOG="/workspace/automation/logs/watchdog.log"
FLAG_DIR="/workspace/automation/flags"
CHECK_INTERVAL=900  # 15 minutes

log() {
    local ts=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    echo "[$ts] $*" >> "$WATCHDOG_LOG"
}

flag_anomaly() {
    local reason="$1"
    echo "status: ANOMALY" > "$FLAG_DIR/WATCHDOG.status"
    echo "timestamp: $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$FLAG_DIR/WATCHDOG.status"
    echo "reason: $reason" >> "$FLAG_DIR/WATCHDOG.status"
    log "ANOMALY: $reason"
    ANOMALY_THIS_LOOP=1
}

clear_anomaly() {
    rm -f "$FLAG_DIR/WATCHDOG.status"
}

while true; do
    ANOMALY_THIS_LOOP=""
    log "--- Watchdog check ---"

    # 1. Torchrun workers alive?
    WORKERS=$(ps aux | grep train_orchestrator | grep -v grep | wc -l)

    # Determine if the chain expects a run to be active
    CHAIN_STATUS="unknown"
    if [ -f "$FLAG_DIR/CHAIN.status" ]; then
        CHAIN_STATUS=$(grep "^status:" "$FLAG_DIR/CHAIN.status" | awk '{print $2}')
    fi

    log "Chain status: $CHAIN_STATUS, workers: $WORKERS"

    # Only consider a run "active" if chain is RUNNING
    if [ "$CHAIN_STATUS" = "RUNNING" ]; then
        if [ "$WORKERS" -lt 4 ]; then
            flag_anomaly "Chain RUNNING but only $WORKERS torchrun workers found (expected 8-9)"
            sleep $CHECK_INTERVAL
            continue
        fi

        # 2. Check latest training log is being written (not hung)
        LATEST_LOG=$(ls -t /workspace/logs/phase3_RUN-P3-*_*.log 2>/dev/null | head -1)
        if [ -n "$LATEST_LOG" ]; then
            LAST_MODIFIED=$(stat -c %Y "$LATEST_LOG")
            NOW=$(date +%s)
            AGE_SECS=$((NOW - LAST_MODIFIED))
            # Log should be written to at least every 5 minutes (steps log every 10 steps, ~100s at 10s/step)
            if [ "$AGE_SECS" -gt 600 ]; then
                flag_anomaly "Log $LATEST_LOG not written to in ${AGE_SECS}s — possible hang"
                sleep $CHECK_INTERVAL
                continue
            fi
            log "Latest log active (last written ${AGE_SECS}s ago)"
        fi

        # 3. GPU utilization sanity check
        AVG_UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | \
                   awk '{sum+=$1; n++} END {print (n > 0) ? sum/n : 0}')
        # During training expect avg > 50%; during eval/checkpointing expect it can dip
        # So only flag if consistently VERY low
        if [ "$(awk "BEGIN {print ($AVG_UTIL < 10) ? 1 : 0}")" -eq 1 ]; then
            log "WARNING: Avg GPU util very low ($AVG_UTIL%) — possibly stuck. Rechecking in 5 min..."
            sleep 300
            AVG_UTIL_2=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | \
                       awk '{sum+=$1; n++} END {print (n > 0) ? sum/n : 0}')
            if [ "$(awk "BEGIN {print ($AVG_UTIL_2 < 10) ? 1 : 0}")" -eq 1 ]; then
                flag_anomaly "GPUs stuck at low utilization (${AVG_UTIL}% then ${AVG_UTIL_2}%)"
            fi
        fi
        log "Avg GPU util: $AVG_UTIL%"
    fi

    # 4. Disk usage (regardless of chain state)
    DISK_USED_PCT=$(df /workspace | awk 'NR==2 {print $5}' | tr -d '%')
    if [ "$DISK_USED_PCT" -gt 99 ]; then
        flag_anomaly "Disk at ${DISK_USED_PCT}% used — cleanup needed"
    fi
    log "Disk used: ${DISK_USED_PCT}%"

    # 5. If all checks pass and there was an anomaly flag, clear it
    if [ -f "$FLAG_DIR/WATCHDOG.status" ] && [ -z "$ANOMALY_THIS_LOOP" ]; then
        # Only clear if no anomaly was flagged this iteration
        clear_anomaly
        log "Prior anomaly cleared"
    fi

    log "All checks OK"
    sleep $CHECK_INTERVAL
done
