#!/bin/bash
# Live view of RUN-P3-FFT training.
# Refreshes every 15 seconds. Ctrl-C to exit.

LOG_FILE=$(ls -t /workspace/logs/phase3_RUN-P3-FFT_*.log 2>/dev/null | head -1)
if [ -z "$LOG_FILE" ]; then
    echo "ERROR: no phase3 FFT log found"
    exit 1
fi

CELL_ID="RUN-P3-FFT"
QLORA_CLEAN_PPH="23.69"   # RUN-P3-CLEAN-QLORA (fair baseline)
C05_UNFILTERED_PPH="23.70" # Phase 2 C05-S2 baseline (reference)

while true; do
    clear
    NOW=$(date -u +"%Y-%m-%d %H:%M:%S UTC")

    echo "+============================================================================+"
    echo "| LughaGen Phase 3: $CELL_ID  ·  $NOW"
    echo "+============================================================================+"

    # ----- Process status -----
    PROC=$(ps aux | grep train_orchestrator | grep -v grep | wc -l)
    if [ "$PROC" -eq "0" ]; then
        echo "| STATUS: STOPPED"
    else
        echo "| STATUS: RUNNING ($PROC processes)"
    fi

    # ----- Progress -----
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
        echo "| Progress: (setup phase — no training steps yet)"
        # Show what setup phase we're in
        LAST_SETUP=$(grep -E "\[FFT\]|\[remap\]|\[FA2\]|\[data\]|\[batch\]|Loading weights|Generating|Map " "$LOG_FILE" | tail -3)
        if [ -n "$LAST_SETUP" ]; then
            echo "|"
            echo "| Recent setup markers:"
            echo "$LAST_SETUP" | while read line; do
                echo "|   ${line:0:70}"
            done
        fi
    fi

    # ----- PPH tracking -----
    echo "+============================================================================+"
    echo "| PERPLEXITY TRACKING (FFT vs QLoRA-clean baseline)                          |"
    echo "|                                                                            |"
    echo "|   Fair baseline (P3-CLEAN-QLORA):  $QLORA_CLEAN_PPH                              |"
    echo "|   Reference (C05-S2 unfiltered):   $C05_UNFILTERED_PPH                              |"
    echo "|                                                                            |"
    echo "|   Success criterion (per pre-registration):                                |"
    echo "|     Aggregate improvement >= 2 pph                                         |"
    echo "|     AND Kamba improvement >= 5 pph                                         |"
    echo "|     (See phase3_plan.yaml for full IES decision rule)                      |"
    echo "+----------------------------------------------------------------------------+"

    EVAL_LINES=$(grep "'eval_loss':" "$LOG_FILE" 2>/dev/null)
    if [ -n "$EVAL_LINES" ]; then
        echo "| Eval trajectory (pph = exp(eval_loss)):                                    |"
        echo "|                                                                            |"
        echo "$EVAL_LINES" | while read line; do
            EL=$(echo "$line" | grep -oE "'eval_loss': '[0-9\.]+'" | grep -oE "[0-9\.]+")
            EP=$(echo "$line" | grep -oE "'epoch': '[0-9\.]+'" | grep -oE "[0-9\.]+")
            [ -z "$EL" ] && continue
            PPH=$(python3 -c "import math; print(f'{math.exp($EL):.2f}')" 2>/dev/null)
            DELTA=$(python3 -c "print(f'{$PPH - $QLORA_CLEAN_PPH:+.2f}')" 2>/dev/null)
            printf "|   epoch=%-9s eval_loss=%-8s pph=%-10s  (vs P3-QLoRA: %s)\n" \
                "$EP" "$EL" "$PPH" "$DELTA"
        done
        echo "|                                                                            |"

        LATEST_LOSS=$(echo "$EVAL_LINES" | tail -1 | grep -oE "'eval_loss': '[0-9\.]+'" | grep -oE "[0-9\.]+")
        LATEST_PPH=$(python3 -c "import math; print(f'{math.exp($LATEST_LOSS):.2f}')" 2>/dev/null)
        LATEST_DELTA=$(python3 -c "print(f'{$LATEST_PPH - $QLORA_CLEAN_PPH:+.2f}')" 2>/dev/null)
        echo "|   LATEST PPH: $LATEST_PPH   (vs P3-QLoRA baseline: $LATEST_DELTA)"
    else
        echo "|   No evals yet.  First eval fires at step 200.                             |"
    fi

    # ----- Training loss -----
    echo "+============================================================================+"
    echo "| Training loss (last 5 log points)                                          |"
    grep "'loss':" "$LOG_FILE" 2>/dev/null | tail -5 | while read line; do
        L=$(echo "$line" | grep -oE "'loss': '[0-9\.]+'" | grep -oE "[0-9\.]+")
        G=$(echo "$line" | grep -oE "'grad_norm': '[0-9\.]+'" | grep -oE "[0-9\.]+")
        LR=$(echo "$line" | grep -oE "'learning_rate': '[0-9\.e\-]+'" | grep -oE "[0-9\.e\-]+" | head -1)
        printf "|   loss=%-8s grad_norm=%-8s lr=%s\n" "$L" "$G" "$LR"
    done

    # ----- GPU status -----
    echo "+============================================================================+"
    echo "| GPUs                                                                       |"
    nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu,temperature.gpu \
        --format=csv,noheader | \
        awk -F',' '{printf "|   GPU %s: mem=%s/%s  util=%s  temp=%s\n", $1, $2, $3, $4, $5}'

    # ----- Disk + checkpoints -----
    DISK=$(df -h /workspace | tail -1 | awk '{printf "%s used / %s (%s free)", $3, $2, $4}')
    echo "+----------------------------------------------------------------------------+"
    echo "| Disk /workspace: $DISK"

    if [ -d "/workspace/checkpoints/$CELL_ID" ]; then
        CKPT_COUNT=$(ls -d /workspace/checkpoints/$CELL_ID/checkpoint-* 2>/dev/null | wc -l)
        CKPT_SIZE=$(du -sh /workspace/checkpoints/$CELL_ID 2>/dev/null | cut -f1)
        LATEST=$(ls -td /workspace/checkpoints/$CELL_ID/checkpoint-* 2>/dev/null | head -1 | \
            xargs -I{} basename {} 2>/dev/null)
        echo "| Checkpoints: $CKPT_COUNT saved  ·  latest: $LATEST  ·  total: $CKPT_SIZE"
        # FFT checkpoints are large (~15-30 GB each), warn if disk usage is climbing fast
    fi

    echo "+============================================================================+"
    echo "  Log: $(basename $LOG_FILE)  ·  Refresh 15s  ·  Ctrl-C to exit"

    sleep 15
done
