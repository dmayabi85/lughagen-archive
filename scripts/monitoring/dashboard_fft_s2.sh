#!/bin/bash
# Live dashboard for RUN-P3-FFT-S2 training run
# Refreshes every 30s. Ctrl-C to exit.

CELL_ID="RUN-P3-FFT-S2"
LOG=$(ls -t /workspace/logs/phase3_RUN-P3-FFT-S2_*.log 2>/dev/null | head -1)

while true; do
    clear
    echo "=========================================="
    echo "  LughaGen FFT Seed 2 Dashboard"
    echo "  $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "=========================================="
    echo ""

    # --- Process health ---
    WORKERS=$(ps aux | grep train_orchestrator | grep -v grep | wc -l)
    echo "▸ Torchrun workers: $WORKERS"
    if [ "$WORKERS" -eq 0 ]; then
        echo "  ⚠  NO WORKERS RUNNING — training stopped or crashed"
    fi
    echo ""

    # --- GPU state ---
    echo "▸ GPU state:"
    nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw \
               --format=csv,noheader,nounits | \
        awk -F',' '{printf "  GPU%s: %5s / %5s MiB  util=%3s%%  temp=%s°C  power=%3sW\n", $1, $2, $3, $4, $5, int($6)}'
    echo ""

    # --- Training progress ---
    echo "▸ Training progress:"
    if [ -f "$LOG" ]; then
        # Latest step
        LATEST_STEP=$(grep -oP "'step': \K[0-9]+" "$LOG" | tail -1)
        LATEST_LOSS=$(grep -oP "'loss': \K[0-9.]+" "$LOG" | tail -1)
        LATEST_LR=$(grep -oP "'learning_rate': \K[0-9.e+-]+" "$LOG" | tail -1)
        LATEST_GRAD=$(grep -oP "'grad_norm': \K[0-9.]+" "$LOG" | tail -1)

        if [ -n "$LATEST_STEP" ]; then
            PCT=$(awk "BEGIN {printf \"%.1f\", $LATEST_STEP/6000*100}")
            echo "  Step:      $LATEST_STEP / 6000  ($PCT%)"
            echo "  Loss:      $LATEST_LOSS"
            echo "  LR:        $LATEST_LR"
            echo "  Grad norm: $LATEST_GRAD"
        else
            echo "  (Still initializing — no training steps yet)"
        fi

        echo ""
        # Latest eval
        LATEST_EVAL_STEP=$(grep -oP "\"eval_loss\":.*?\"epoch\":" "$LOG" 2>/dev/null | tail -1)
        LATEST_EVAL_PPL=$(grep -oP "'eval_perplexity': \K[0-9.]+" "$LOG" | tail -1)
        LATEST_EVAL_LOSS=$(grep -oP "'eval_loss': \K[0-9.]+" "$LOG" | tail -1)
        echo "▸ Latest evaluation:"
        if [ -n "$LATEST_EVAL_PPL" ]; then
            echo "  eval_perplexity: $LATEST_EVAL_PPL"
            echo "  eval_loss:       $LATEST_EVAL_LOSS"
        else
            echo "  (No eval yet — first eval at step 200)"
        fi
    else
        echo "  ⚠  No log file found"
    fi
    echo ""

    # --- Time estimation ---
    if [ -f "$LOG" ]; then
        START_LINE=$(head -20 "$LOG" | grep -oP "Started: \K[0-9T:Z-]+" | head -1)
        if [ -z "$START_LINE" ]; then
            # Fall back to log file mtime
            START_LINE=$(stat -c %y "$LOG" 2>/dev/null | cut -d'.' -f1 | tr ' ' 'T')
        fi

        if [ -n "$LATEST_STEP" ] && [ "$LATEST_STEP" -gt 10 ]; then
            LOG_START_EPOCH=$(stat -c %Y "$LOG")
            NOW_EPOCH=$(date +%s)
            ELAPSED=$((NOW_EPOCH - LOG_START_EPOCH))
            SECS_PER_STEP=$(awk "BEGIN {printf \"%.2f\", $ELAPSED/$LATEST_STEP}")
            REMAINING_STEPS=$((6000 - LATEST_STEP))
            REMAINING_SECS=$(awk "BEGIN {printf \"%.0f\", $REMAINING_STEPS*$SECS_PER_STEP}")
            REMAINING_HRS=$(awk "BEGIN {printf \"%.1f\", $REMAINING_SECS/3600}")
            echo "▸ Timing:"
            echo "  Elapsed:        $(awk "BEGIN {printf \"%.1f\", $ELAPSED/3600}") hrs"
            echo "  Sec/step:       $SECS_PER_STEP"
            echo "  ETA remaining:  $REMAINING_HRS hrs"
            ETA_EPOCH=$((NOW_EPOCH + REMAINING_SECS))
            ETA_UTC=$(date -u -d "@$ETA_EPOCH" +%Y-%m-%dT%H:%M:%SZ)
            echo "  ETA completion: $ETA_UTC"
        fi
    fi
    echo ""

    # --- Disk ---
    DISK=$(df -h /workspace | awk 'NR==2 {print $4 " free (" $5 " used)"}')
    echo "▸ Disk: $DISK"
    echo ""

    # --- Log tail ---
    echo "▸ Log tail (last 5 lines):"
    if [ -f "$LOG" ]; then
        tail -5 "$LOG" | sed 's/^/  /'
    fi
    echo ""

    echo "=========================================="
    echo "  Refreshing every 30s. Ctrl-C to exit."

    sleep 30
done
