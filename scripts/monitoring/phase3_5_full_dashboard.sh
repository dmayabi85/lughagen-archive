#!/bin/bash
# Live dashboard for Phase 3.5 Gemma 2 27B CPT full run.
# Refreshes every 30s. Ctrl-C to exit. Read-only.

RUN_ID="RUN-P3.5-GEMMA2-CPT-S137"
LOG_DIR="/workspace/logs"
OUTPUT_DIR="/workspace/checkpoints/$RUN_ID"
LOG_GLOB="/workspace/logs/phase3_5_${RUN_ID}_*.log"
MAX_STEPS=6000
EVAL_STEPS=200
SAVE_STEPS=1000
WARMUP_STEPS=500

GREEN=$'\033[0;32m'
YELLOW=$'\033[0;33m'
RED=$'\033[0;31m'
BLUE=$'\033[0;34m'
CYAN=$'\033[0;36m'
MAGENTA=$'\033[0;35m'
BOLD=$'\033[1m'
DIM=$'\033[2m'
RESET=$'\033[0m'

# Get run start time from first log line if available
get_run_start_epoch() {
    local log="$1"
    local start_iso=$(grep -E "^Started:" "$log" 2>/dev/null | head -1 | awk '{print $2}')
    if [ -z "$start_iso" ]; then
        # Fallback: use file mtime of log
        stat -c %Y "$log" 2>/dev/null
    else
        date -d "$start_iso" +%s 2>/dev/null
    fi
}

# Extract current training step from log
get_current_step() {
    local log="$1"
    # Look for "step / total_steps" patterns after "[train] starting"
    grep -oE " ([0-9]+)/${MAX_STEPS} \[" "$log" 2>/dev/null | tail -1 | grep -oE "^ [0-9]+" | tr -d ' '
}

# Extract loss values as JSON-like strings
get_loss_series() {
    local log="$1"
    grep -oE "'loss': '[0-9.]+'" "$log" 2>/dev/null
}

format_seconds() {
    local secs=$1
    if [ -z "$secs" ] || [ "$secs" -le 0 ] 2>/dev/null; then
        echo "?"
        return
    fi
    local days=$((secs / 86400))
    local hours=$(( (secs % 86400) / 3600 ))
    local minutes=$(( (secs % 3600) / 60 ))
    if [ "$days" -gt 0 ]; then
        printf "%dd %dh %dm" "$days" "$hours" "$minutes"
    elif [ "$hours" -gt 0 ]; then
        printf "%dh %dm" "$hours" "$minutes"
    else
        printf "%dm" "$minutes"
    fi
}

progress_bar() {
    local current=$1
    local total=$2
    local width=40
    if [ "$total" -le 0 ]; then
        return
    fi
    local filled=$(( current * width / total ))
    local empty=$(( width - filled ))
    local bar="["
    for ((i=0; i<filled; i++)); do bar+="#"; done
    for ((i=0; i<empty; i++)); do bar+="-"; done
    bar+="]"
    printf "%s" "$bar"
}

while true; do
    clear
    NOW_EPOCH=$(date -u +%s)
    NOW_UTC=$(date -u '+%Y-%m-%d %H:%M:%S UTC')

    echo "${BOLD}╔══════════════════════════════════════════════════════════════════════════╗${RESET}"
    echo "${BOLD}║          Phase 3.5 Gemma 2 27B CPT — Full Run                          ║${RESET}"
    echo "${BOLD}║          $NOW_UTC                                          ║${RESET}"
    echo "${BOLD}╚══════════════════════════════════════════════════════════════════════════╝${RESET}"
    echo ""

    LATEST_LOG=$(ls -t $LOG_GLOB 2>/dev/null | head -1)
    if [ -z "$LATEST_LOG" ]; then
        echo "${YELLOW}▸ No log yet — waiting for run to start${RESET}"
        sleep 30
        continue
    fi

    START_EPOCH=$(get_run_start_epoch "$LATEST_LOG")
    ELAPSED_SEC=$((NOW_EPOCH - START_EPOCH))

    # Pipeline markers
    HAS_ROUTING=$(grep -c "gemma-family: disabled attn_logit_softcapping" $LATEST_LOG)
    HAS_WEIGHTS=$(grep -c "Loading weights: 100%" $LATEST_LOG)
    HAS_REMAP_APPLIED=$(grep -c "\[remap\] applied" $LATEST_LOG)
    HAS_ATTN_ENGAGED=$(grep -c "\[attn\] 'sdpa' engaged\|\[attn\] 'flash_attention_2' engaged" $LATEST_LOG)
    HAS_TRAIN_START=$(grep -c "\[train\] starting" $LATEST_LOG)
    HAS_DONE=$(grep -c "\[done\] wall_clock" $LATEST_LOG)

    # Current step
    CURRENT_STEP=$(get_current_step "$LATEST_LOG")
    [ -z "$CURRENT_STEP" ] && CURRENT_STEP=0

    # Loss series (for trend + latest)
    LOSS_VALUES=$(get_loss_series "$LATEST_LOG" | grep -oE "[0-9.]+")
    LOSS_COUNT=$(echo "$LOSS_VALUES" | grep -c .)
    LATEST_LOSS=$(echo "$LOSS_VALUES" | tail -1)
    FIRST_LOSS=$(echo "$LOSS_VALUES" | head -1)
    LOSS_10AGO=$(echo "$LOSS_VALUES" | tail -11 | head -1)

    # Recent step/sec calculation (needs [train] start marker)
    STEPS_PER_SEC=""
    SEC_PER_STEP=""
    ETA_SEC=""
    ETA_ABSOLUTE=""
    if [ "$HAS_TRAIN_START" -gt 0 ] && [ "$CURRENT_STEP" -gt 10 ]; then
        # Timestamp of [train] starting
        TRAIN_START_LINE=$(grep -n "\[train\] starting" $LATEST_LOG | head -1 | cut -d: -f1)
        # Compute time since [train] start using log file mtime as proxy for now
        # Better: use tqdm rate estimate from log
        # Extract last "[time_elapsed<time_remaining, X.Xs/it]" from tqdm
        TQDM_RATE=$(grep -oE "[0-9]+:[0-9]+<[0-9]+:[0-9:]+, +[0-9.]+s/it" $LATEST_LOG | tail -1)
        if [ -n "$TQDM_RATE" ]; then
            SEC_PER_STEP=$(echo "$TQDM_RATE" | grep -oE "[0-9.]+s/it" | grep -oE "[0-9.]+")
            REMAINING=$((MAX_STEPS - CURRENT_STEP))
            if [ -n "$SEC_PER_STEP" ]; then
                ETA_SEC=$(awk -v r="$REMAINING" -v s="$SEC_PER_STEP" 'BEGIN{printf "%.0f", r*s}')
                ETA_UTC_EPOCH=$((NOW_EPOCH + ETA_SEC))
                ETA_ABSOLUTE=$(date -u -d "@${ETA_UTC_EPOCH}" '+%Y-%m-%d %H:%M UTC')
            fi
        fi
    fi

    # Warmup indicator
    WARMUP_STATUS=""
    if [ "$CURRENT_STEP" -lt "$WARMUP_STEPS" ]; then
        WARMUP_PCT=$(awk -v c="$CURRENT_STEP" -v w="$WARMUP_STEPS" 'BEGIN{printf "%.0f", 100*c/w}')
        WARMUP_STATUS="${YELLOW}IN WARMUP (${WARMUP_PCT}% through ${WARMUP_STEPS} warmup steps)${RESET}"
    else
        WARMUP_STATUS="${GREEN}POST-WARMUP (cosine decay to zero)${RESET}"
    fi

    # Overall status
    if [ "$HAS_DONE" -gt 0 ]; then
        echo "${GREEN}${BOLD}▸ STATUS: RUN COMPLETE${RESET}"
    elif grep -qE "Traceback|Error|refuse to train|OOM|CUDA out of memory" $LATEST_LOG 2>/dev/null; then
        echo "${RED}${BOLD}▸ STATUS: ERROR DETECTED${RESET}"
    elif [ "$HAS_TRAIN_START" -gt 0 ] && [ "$CURRENT_STEP" -gt 0 ]; then
        echo "${GREEN}${BOLD}▸ STATUS: TRAINING${RESET}"
    elif [ "$HAS_ATTN_ENGAGED" -gt 0 ]; then
        echo "${YELLOW}▸ STATUS: SETUP (dataset prep)${RESET}"
    else
        echo "${YELLOW}▸ STATUS: INITIALIZING${RESET}"
    fi
    echo ""

    # === KEY KPI PANEL ===
    echo "${BOLD}▸ Progress:${RESET}"
    PCT=$(awk -v c="$CURRENT_STEP" -v m="$MAX_STEPS" 'BEGIN{printf "%.1f", 100*c/m}')
    printf "  Step:      %s/%s  %s  %s%%\n" "${BOLD}${CURRENT_STEP}${RESET}" "$MAX_STEPS" "$(progress_bar $CURRENT_STEP $MAX_STEPS)" "$PCT"
    printf "  Phase:     %b\n" "$WARMUP_STATUS"
    echo ""

    echo "${BOLD}▸ Timing:${RESET}"
    printf "  Elapsed:   %s\n" "$(format_seconds $ELAPSED_SEC)"
    if [ -n "$SEC_PER_STEP" ]; then
        printf "  Pace:      %s sec/step\n" "$SEC_PER_STEP"
        printf "  ETA (rem): %s\n" "$(format_seconds $ETA_SEC)"
        printf "  ETA (abs): ${CYAN}${BOLD}%s${RESET}\n" "${ETA_ABSOLUTE}"
    else
        echo "  Pace:      (waiting for tqdm data...)"
    fi
    echo ""

    echo "${BOLD}▸ Loss trajectory:${RESET}"
    if [ -n "$LATEST_LOSS" ]; then
        printf "  Latest:    ${BOLD}%s${RESET}\n" "$LATEST_LOSS"
        if [ -n "$FIRST_LOSS" ]; then
            DELTA_FROM_START=$(awk -v f="$FIRST_LOSS" -v l="$LATEST_LOSS" 'BEGIN{printf "%.2f", f-l}')
            printf "  Δ start:   %s→%s  (dropped %s)\n" "$FIRST_LOSS" "$LATEST_LOSS" "$DELTA_FROM_START"
        fi
        if [ -n "$LOSS_10AGO" ] && [ "$LOSS_COUNT" -gt 10 ]; then
            RECENT_DELTA=$(awk -v o="$LOSS_10AGO" -v l="$LATEST_LOSS" 'BEGIN{printf "%.3f", o-l}')
            printf "  Δ last 10: %s\n" "$RECENT_DELTA"
        fi
        printf "  Samples:   %s loss values logged\n" "$LOSS_COUNT"
    else
        echo "  (no loss values yet)"
    fi
    # Latest eval
    LATEST_EVAL=$(grep -oE "'eval_loss': '[0-9.]+'" $LATEST_LOG | tail -1)
    LATEST_PPH=$(grep -oE "'val_pph': [0-9.]+" $LATEST_LOG | tail -1)
    if [ -n "$LATEST_EVAL" ]; then
        printf "  Eval:      %s\n" "$(echo $LATEST_EVAL | tr -d "'")"
    fi
    echo ""

    echo "${BOLD}▸ Pipeline validation:${RESET}"
    STATUS_LINE=""
    [ "$HAS_ROUTING" -gt 0 ] && STATUS_LINE+="${GREEN}✓${RESET} softcap disabled  " || STATUS_LINE+="${BLUE}○${RESET} softcap disabled  "
    [ "$HAS_WEIGHTS" -gt 0 ] && STATUS_LINE+="${GREEN}✓${RESET} weights loaded  " || STATUS_LINE+="${BLUE}○${RESET} weights  "
    [ "$HAS_REMAP_APPLIED" -gt 0 ] && STATUS_LINE+="${GREEN}✓${RESET} remap applied  " || STATUS_LINE+="${BLUE}○${RESET} remap  "
    [ "$HAS_ATTN_ENGAGED" -gt 0 ] && STATUS_LINE+="${GREEN}✓${RESET} SDPA engaged  " || STATUS_LINE+="${BLUE}○${RESET} SDPA  "
    [ "$HAS_TRAIN_START" -gt 0 ] && STATUS_LINE+="${GREEN}✓${RESET} training" || STATUS_LINE+="${BLUE}○${RESET} training"
    echo "  $STATUS_LINE"
    echo ""

    # Milestone tracking
    echo "${BOLD}▸ Upcoming milestones:${RESET}"
    NEXT_EVAL=$(( (CURRENT_STEP / EVAL_STEPS + 1) * EVAL_STEPS ))
    NEXT_SAVE=$(( (CURRENT_STEP / SAVE_STEPS + 1) * SAVE_STEPS ))
    [ "$NEXT_EVAL" -gt "$MAX_STEPS" ] && NEXT_EVAL="—"
    [ "$NEXT_SAVE" -gt "$MAX_STEPS" ] && NEXT_SAVE="—"
    if [ -n "$SEC_PER_STEP" ] && [ "$NEXT_EVAL" != "—" ]; then
        STEPS_TO_EVAL=$((NEXT_EVAL - CURRENT_STEP))
        ETA_EVAL_SEC=$(awk -v s="$STEPS_TO_EVAL" -v r="$SEC_PER_STEP" 'BEGIN{printf "%.0f", s*r}')
        printf "  Next eval:  step %s (in %s)\n" "$NEXT_EVAL" "$(format_seconds $ETA_EVAL_SEC)"
    else
        printf "  Next eval:  step %s\n" "$NEXT_EVAL"
    fi
    if [ -n "$SEC_PER_STEP" ] && [ "$NEXT_SAVE" != "—" ]; then
        STEPS_TO_SAVE=$((NEXT_SAVE - CURRENT_STEP))
        ETA_SAVE_SEC=$(awk -v s="$STEPS_TO_SAVE" -v r="$SEC_PER_STEP" 'BEGIN{printf "%.0f", s*r}')
        printf "  Next save:  step %s (in %s)  ${DIM}~341 GB write${RESET}\n" "$NEXT_SAVE" "$(format_seconds $ETA_SAVE_SEC)"
    else
        printf "  Next save:  step %s  ${DIM}~341 GB write${RESET}\n" "$NEXT_SAVE"
    fi
    if [ "$CURRENT_STEP" -lt "$WARMUP_STEPS" ]; then
        WARMUP_REMAIN=$((WARMUP_STEPS - CURRENT_STEP))
        if [ -n "$SEC_PER_STEP" ]; then
            WARMUP_ETA_SEC=$(awk -v s="$WARMUP_REMAIN" -v r="$SEC_PER_STEP" 'BEGIN{printf "%.0f", s*r}')
            printf "  Post-warmup: step %s (in %s)\n" "$WARMUP_STEPS" "$(format_seconds $WARMUP_ETA_SEC)"
        else
            printf "  Post-warmup: step %s\n" "$WARMUP_STEPS"
        fi
    fi
    echo ""

    # === RESOURCE PANEL ===
    echo "${BOLD}▸ GPUs (avg across 8):${RESET}"
    nvidia-smi --query-gpu=memory.used,utilization.gpu,temperature.gpu,power.draw --format=csv,noheader,nounits 2>/dev/null | \
        awk -F', ' 'BEGIN {ms=0; us=0; ts=0; ps=0; n=0}
        { ms+=$1; us+=$2; ts+=$3; ps+=$4; n++ }
        END { if (n>0) printf "  memory: %5d MiB used  util: %3.0f%%  temp: %d°C  power: %d W\n", ms/n, us/n, ts/n, ps/n }'
    # Show any GPU with util < 50% as warning
    LOW_UTIL_GPUS=$(nvidia-smi --query-gpu=index,utilization.gpu --format=csv,noheader,nounits 2>/dev/null | awk -F', ' '$2 < 50 {print $1}' | tr '\n' ',' | sed 's/,$//')
    if [ -n "$LOW_UTIL_GPUS" ] && [ "$HAS_TRAIN_START" -gt 0 ] && [ "$CURRENT_STEP" -gt 20 ]; then
        echo "  ${YELLOW}Warning: GPUs at low util: $LOW_UTIL_GPUS${RESET}"
    fi
    echo ""

    echo "${BOLD}▸ CPU / Disk:${RESET}"
    CPU_MEM_STR=$(free -g | awk 'NR==2 {printf "%d/%d GB used (%.0f%%)", $3, $2, 100*$3/$2}')
    DISK_STR=$(df -h /workspace 2>/dev/null | tail -1 | awk '{printf "%s free, %s used", $4, $5}')
    printf "  CPU memory:  %s  ${DIM}(offloaded optimizer state)${RESET}\n" "$CPU_MEM_STR"
    printf "  Disk:        %s\n" "$DISK_STR"
    if [ -d "$OUTPUT_DIR" ]; then
        CP_SIZE=$(du -sh "$OUTPUT_DIR" 2>/dev/null | awk '{print $1}')
        printf "  Checkpoint:  %s ${DIM}(caps at ~341 GB per save, top_k=1)${RESET}\n" "$CP_SIZE"
    fi
    echo ""

    echo "${BOLD}▸ Log tail:${RESET}"
    tail -2 "$LATEST_LOG" 2>/dev/null | while IFS= read -r line; do
        echo "  ${DIM}${line:0:110}${RESET}"
    done
    echo ""

    echo "${DIM}──────────────────────────────────────────────────────────────────────${RESET}"
    echo "${DIM}  Refreshing every 30s. Ctrl-C to exit. Read-only.${RESET}"

    sleep 30
done
