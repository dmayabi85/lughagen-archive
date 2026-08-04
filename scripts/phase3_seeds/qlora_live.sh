#!/bin/bash
# Live view of RUN-P3-CLEAN-QLORA with clear pph tracking.

LOG_FILE=$(ls -t /workspace/logs/phase3_RUN-P3-CLEAN-QLORA_*.log 2>/dev/null | head -1)
[ -z "$LOG_FILE" ] && { echo "no phase3 log found"; exit 1; }

CELL_ID="RUN-P3-CLEAN-QLORA"
BASELINE_PPH="23.704"

while true; do
    clear
    NOW=$(date -u +"%Y-%m-%d %H:%M:%S UTC")

    echo "+============================================================================+"
    echo "| LughaGen Phase 3: $CELL_ID  ·  $NOW"
    echo "+============================================================================+"

    # Status
    PROC=$(ps aux | grep train_orchestrator | grep -v grep | wc -l)
    if [ "$PROC" -eq "0" ]; then
        echo "| STATUS: STOPPED"
    else
        echo "| STATUS: RUNNING ($PROC processes)"
    fi

    # Progress
    PROGRESS=$(grep -oE "[0-9]+/6000 \[[0-9:<,\.]+ +[0-9\.]+s/it\]" "$LOG_FILE" | tail -1)
    if [ -n "$PROGRESS" ]; then
        STEP=$(echo "$PROGRESS" | grep -oE "^[0-9]+")
        PCT=$(python3 -c "print(f'{100*$STEP/6000:.1f}')")
        RATE=$(echo "$PROGRESS" | grep -oE "[0-9\.]+s/it" | grep -oE "[0-9\.]+")
        ELAPSED=$(echo "$PROGRESS" | grep -oE "\[[0-9:]+<" | tr -d "[<")
        ETA=$(echo "$PROGRESS" | grep -oE "<[0-9:]+," | tr -d "<,")
        if [ -n "$RATE" ]; then
            REM=$((6000 - STEP))
            REM_SEC=$(python3 -c "print(int($REM * $RATE))")
            FINISH=$(date -u -d "+${REM_SEC} seconds" +"%m-%d %H:%M UTC")
        fi
        echo "| Step:    $STEP / 6000  ($PCT%)  ·  ${RATE}s/step"
        echo "| Elapsed: $ELAPSED  ·  ETA: $ETA  ·  Finish: ~$FINISH"
    else
        echo "| Progress: (setup phase, no training steps yet)"
    fi

    # ================================================================
    # PPH TRACKING (the KPI you actually care about)
    # ================================================================
    echo "+============================================================================+"
    echo "| PERPLEXITY TRACKING                                                        |"
    echo "|                                                                            |"
    echo "|   Target (C05-S2 on unfiltered corpus):  $BASELINE_PPH                    |"
    echo "|   Expected range for RUN-P3-CLEAN-QLORA: 23.5 to 24.0 (within noise)      |"
    echo "+----------------------------------------------------------------------------+"

    # All eval points so far — compute pph from eval_loss and show trajectory
    EVAL_LINES=$(grep "'eval_loss':" "$LOG_FILE" 2>/dev/null)
    if [ -n "$EVAL_LINES" ]; then
        echo "| Eval trajectory (pph = exp(eval_loss), evaluated every 200 steps):        |"
        echo "|                                                                            |"
        echo "$EVAL_LINES" | while read line; do
            EL=$(echo "$line" | grep -oE "'eval_loss': '[0-9\.]+'" | grep -oE "[0-9\.]+")
            EP=$(echo "$line" | grep -oE "'epoch': '[0-9\.]+'" | grep -oE "[0-9\.]+")
            [ -z "$EL" ] && continue
            PPH=$(python3 -c "import math; print(f'{math.exp($EL):.2f}')" 2>/dev/null)
            DELTA=$(python3 -c "print(f'{$PPH - $BASELINE_PPH:+.2f}')" 2>/dev/null)
            printf "|   epoch=%-9s eval_loss=%-8s pph=%-10s  (vs baseline: %s)\n" "$EP" "$EL" "$PPH" "$DELTA"
        done
        echo "|                                                                            |"

        # Latest pph highlighted
        LATEST_LOSS=$(echo "$EVAL_LINES" | tail -1 | grep -oE "'eval_loss': '[0-9\.]+'" | grep -oE "[0-9\.]+")
        LATEST_PPH=$(python3 -c "import math; print(f'{math.exp($LATEST_LOSS):.2f}')" 2>/dev/null)
        echo "|   LATEST PPH: $LATEST_PPH   (baseline: $BASELINE_PPH)"
    else
        echo "|   No evals yet.  First eval fires at step 200.                             |"
        echo "|   Watch for '{'eval_loss': ...}' in the log.                               |"
    fi

    # ================================================================
    # Recent training loss
    # ================================================================
    echo "+============================================================================+"
    echo "| Training loss (last 5 logging points; not the same as eval pph)           |"
    grep "'loss':" "$LOG_FILE" 2>/dev/null | tail -5 | while read line; do
        L=$(echo "$line" | grep -oE "'loss': '[0-9\.]+'" | grep -oE "[0-9\.]+")
        G=$(echo "$line" | grep -oE "'grad_norm': '[0-9\.]+'" | grep -oE "[0-9\.]+")
        LR=$(echo "$line" | grep -oE "'learning_rate': '[0-9\.e\-]+'" | grep -oE "[0-9\.e\-]+" | head -1)
        printf "|   loss=%-8s grad_norm=%-8s lr=%s\n" "$L" "$G" "$LR"
    done

    # ================================================================
    # GPUs
    # ================================================================
    echo "+============================================================================+"
    echo "| GPUs                                                                       |"
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu,temperature.gpu --format=csv,noheader | \
        awk -F',' '{printf "|   GPU %s: mem=%s  util=%s  temp=%s\n", $1, $2, $3, $4}'

    # Disk + checkpoints
    DISK=$(df -h /workspace | tail -1 | awk '{printf "%s used / %s (%s free)", $3, $2, $4}')
    echo "+----------------------------------------------------------------------------+"
    echo "| Disk /workspace: $DISK"
    if [ -d "/workspace/checkpoints/$CELL_ID" ]; then
        CKPT_COUNT=$(ls -d /workspace/checkpoints/$CELL_ID/checkpoint-* 2>/dev/null | wc -l)
        CKPT_SIZE=$(du -sh /workspace/checkpoints/$CELL_ID 2>/dev/null | cut -f1)
        LATEST=$(ls -td /workspace/checkpoints/$CELL_ID/checkpoint-* 2>/dev/null | head -1 | xargs -I{} basename {} 2>/dev/null)
        echo "| Checkpoints: $CKPT_COUNT saved  ·  latest: $LATEST  ·  total: $CKPT_SIZE"
    fi

    echo "+============================================================================+"
    echo "  Log: $(basename $LOG_FILE)  ·  Refresh 15s  ·  Ctrl-C to exit"

    sleep 15
done
