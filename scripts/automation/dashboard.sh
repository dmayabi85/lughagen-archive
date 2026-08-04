#!/bin/bash
# ==============================================================================
# LughaGen Tier 1 Automation Dashboard
# Uses tqdm-format progress bar in the log for step/timing
# ==============================================================================

FLAG_DIR="/workspace/automation/flags"
CHAIN_LOG="/workspace/automation/logs/chain.log"
WATCHDOG_LOG="/workspace/automation/logs/watchdog.log"
LOG_DIR="/workspace/logs"

# Colors
CYAN='\033[0;36m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
GRAY='\033[0;37m'
DIM='\033[2m'
BOLD='\033[1m'
NC='\033[0m'

CHAIN_RUNS=("RUN-P3-FFT-S2" "RUN-P3-CPT-S2" "RUN-P3-FFT-S3" "RUN-P3-CPT-S3")
TARGET_STEPS=6000

declare -A HISTORICAL
HISTORICAL["RUN-P3-FFT"]="19.55 pooled (seed 137)"
HISTORICAL["RUN-P3-CPT"]="22.20 pooled (seed 137)"
HISTORICAL["RUN-P3-CLEAN-QLORA"]="24.62 wrapped / 25.58 direct (seed 137)"

get_run_status() {
    local flag="$FLAG_DIR/$1.status"
    [ -f "$flag" ] && grep "^status:" "$flag" | awk '{print $2}' || echo "PENDING"
}

get_run_val_pph() {
    python3 -c "
import csv
for r in csv.DictReader(open('/workspace/registry/runs.csv')):
    if r.get('factorial_run_id') == '$1':
        v = r.get('metrics.validation_perplexity', '')
        if v: print(f'{float(v):.2f}')
        break
" 2>/dev/null
}

# Parse tqdm progress bar from log
# Format: " NN%|████|CURRENT/TOTAL [ELAPSED<REMAINING, SPEED s/it]"
parse_tqdm_from_log() {
    local log="$1"
    # tqdm writes progress lines like: "812/6000 [2:13:27<14:37:12, 10.15s/it]"
    # This regex works because tqdm's fancy block chars are in the DECORATIVE
    # portion before the step number; we skip straight to the numeric part.
    grep -oE "[0-9]+/6000 \[[0-9:]+<[0-9:]+, [0-9.]+s?/it\]" "$log" 2>/dev/null | tail -1
}

extract_tqdm_step() {
    # Input: "812/6000 [2:13:27<14:37:12, 10.15s/it]" -> 812
    echo "$1" | grep -oE "^[0-9]+" | head -1
}

extract_tqdm_elapsed() {
    # extracts 2:12:39 from the log line
    echo "$1" | grep -oE "\[[0-9:]+<" | tr -d '[<'
}

extract_tqdm_remaining() {
    # extracts 18:24:24 from the log line
    echo "$1" | grep -oE "<[0-9:]+," | tr -d '<,'
}

extract_tqdm_sec_per_it() {
    # extracts 12.76 from "12.76s/it"
    echo "$1" | grep -oE "[0-9.]+s/it" | grep -oE "[0-9.]+"
}

# Convert "HH:MM:SS" or "MM:SS" to seconds
hms_to_seconds() {
    local t="$1"
    local IFS=":"
    local parts=($t)
    local n=${#parts[@]}
    if [ $n -eq 3 ]; then
        echo $((10#${parts[0]}*3600 + 10#${parts[1]}*60 + 10#${parts[2]}))
    elif [ $n -eq 2 ]; then
        echo $((10#${parts[0]}*60 + 10#${parts[1]}))
    else
        echo 0
    fi
}

# Extract last eval_loss from log
get_latest_eval_loss() {
    local log="$1"
    grep -oP "eval_loss': '[0-9.]+" "$log" | tail -1 | grep -oP "[0-9.]+$"
}

get_latest_eval_epoch() {
    local log="$1"
    grep -oP "epoch': '[0-9.]+" "$log" | tail -1 | grep -oP "[0-9.]+$"
}

while true; do
    clear
    NOW_UTC=$(date -u +"%Y-%m-%d %H:%M:%S UTC")
    NOW_EPOCH=$(date +%s)

    # =====================================================================
    # HEADER
    # =====================================================================
    echo -e "${BOLD}${CYAN}╔══════════════════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BOLD}${CYAN}║             LughaGen Tier 1 Automation — Live Dashboard                     ║${NC}"
    echo -e "${BOLD}${CYAN}║             $(printf '%-64s' "$NOW_UTC")║${NC}"
    echo -e "${BOLD}${CYAN}╚══════════════════════════════════════════════════════════════════════════════╝${NC}"
    echo ""

    # =====================================================================
    # SECTION 1: CHAIN STATE
    # =====================================================================
    CHAIN_STATE=$(get_run_status "CHAIN")
    case "$CHAIN_STATE" in
        RUNNING) CC="$GREEN" ;;
        DONE)    CC="$BLUE"  ;;
        FAILED)  CC="$RED"   ;;
        *)       CC="$GRAY"  ;;
    esac
    echo -e "${BOLD}▸ CHAIN STATE: ${CC}${CHAIN_STATE}${NC}"
    echo ""

    ACTIVE_RUN=""
    echo -e "${BOLD}▸ Chain sequence:${NC}"
    for run_id in "${CHAIN_RUNS[@]}"; do
        s=$(get_run_status "$run_id")
        case "$s" in
            DONE)
                pph=$(get_run_val_pph "$run_id")
                echo -e "  ${GREEN}✓${NC} $run_id  ${GREEN}DONE${NC}  ${DIM}val_pph=${pph:-?}${NC}"
                ;;
            RUNNING)
                echo -e "  ${YELLOW}►${NC} $run_id  ${YELLOW}RUNNING${NC}  ${BOLD}(current)${NC}"
                ACTIVE_RUN="$run_id"
                ;;
            FAILED)
                echo -e "  ${RED}✗${NC} $run_id  ${RED}FAILED${NC}"
                ;;
            *)
                echo -e "  ${GRAY}○${NC} $run_id  ${DIM}PENDING${NC}"
                ;;
        esac
    done
    echo ""

    # =====================================================================
    # SECTION 2: ACTIVE RUN DETAILS
    # =====================================================================
    if [ -n "$ACTIVE_RUN" ]; then
        LOG=$(ls -t "$LOG_DIR"/phase3_${ACTIVE_RUN}_*.log 2>/dev/null | head -1)
        echo -e "${BOLD}▸ Active run: ${YELLOW}${ACTIVE_RUN}${NC}"

        if [ -f "$LOG" ]; then
            TQDM=$(parse_tqdm_from_log "$LOG")

            if [ -n "$TQDM" ]; then
                STEP=$(extract_tqdm_step "$TQDM")
                ELAPSED_HMS=$(extract_tqdm_elapsed "$TQDM")
                REMAINING_HMS=$(extract_tqdm_remaining "$TQDM")
                SEC_PER_STEP=$(extract_tqdm_sec_per_it "$TQDM")

                PCT=$(awk "BEGIN {printf \"%.1f\", $STEP/$TARGET_STEPS*100}")

                # Progress bar
                BAR_W=50
                FILLED=$(awk "BEGIN {printf \"%d\", $STEP/$TARGET_STEPS*$BAR_W}")
                BAR=""
                for ((i=0; i<BAR_W; i++)); do
                    [ $i -lt $FILLED ] && BAR="${BAR}█" || BAR="${BAR}░"
                done

                echo -e "  Step:      ${BOLD}${STEP}${NC} / ${TARGET_STEPS}  ${DIM}(${PCT}%)${NC}"
                echo -e "  Progress:  ${GREEN}${BAR}${NC}"
                echo -e "  Elapsed:   ${ELAPSED_HMS}   Remaining: ${REMAINING_HMS}   Speed: ${SEC_PER_STEP}s/step"

                # Compute UTC ETA
                REM_SECS=$(hms_to_seconds "$REMAINING_HMS")
                ETA_EPOCH=$((NOW_EPOCH + REM_SECS))
                ETA_UTC=$(date -u -d "@$ETA_EPOCH" +"%m-%d %H:%M UTC")
                echo -e "  ETA:       ${BOLD}${ETA_UTC}${NC}"

                echo ""
                # Latest eval
                EVAL_LOSS=$(get_latest_eval_loss "$LOG")
                EVAL_EPOCH=$(get_latest_eval_epoch "$LOG")
                if [ -n "$EVAL_LOSS" ]; then
                    EVAL_PPL=$(awk "BEGIN {printf \"%.2f\", exp($EVAL_LOSS)}")
                    echo -e "  ${BOLD}Latest eval:${NC}  loss=${EVAL_LOSS}  ${BOLD}ppl=${GREEN}${EVAL_PPL}${NC}  ${DIM}(epoch=${EVAL_EPOCH})${NC}"
                else
                    echo -e "  Latest eval:  ${DIM}(no eval yet — first at step 200)${NC}"
                fi
            else
                echo -e "  ${DIM}(No tqdm progress found yet — still initializing)${NC}"
            fi
        else
            echo -e "  ${DIM}(No log file found)${NC}"
        fi
    else
        if [ "$CHAIN_STATE" = "DONE" ]; then
            echo -e "${BOLD}${BLUE}▸ Chain complete — all 4 runs done${NC}"
        elif [ "$CHAIN_STATE" = "FAILED" ]; then
            echo -e "${BOLD}${RED}▸ Chain FAILED — check $CHAIN_LOG${NC}"
        else
            echo -e "${DIM}▸ Chain idle${NC}"
        fi
    fi
    echo ""

    # =====================================================================
    # SECTION 3: GPU STATE
    # =====================================================================
    echo -e "${BOLD}▸ GPU state:${NC}"
    nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw \
               --format=csv,noheader,nounits 2>/dev/null | \
        awk -F',' 'BEGIN {tm=0; tu=0; n=0}
        {
            m=$2+0; u=$4+0; t=$5+0; p=$6+0;
            tm+=m; tu+=u; n++;
            if (u >= 80) col="\033[0;32m"; else if (u >= 40) col="\033[1;33m"; else col="\033[0;31m";
            printf "  GPU%s: %5d/%d MiB  util=%s%3d%%\033[0m  temp=%d°C  power=%3dW\n",
                   $1, m, $3, col, u, t, p;
        }
        END {
            if (n > 0) printf "  \033[1mAverage: %d MiB used, %.1f%% util across %d GPUs\033[0m\n", tm/n, tu/n, n;
        }'
    echo ""

    # =====================================================================
    # SECTION 4: SYSTEM HEALTH
    # =====================================================================
    echo -e "${BOLD}▸ System health:${NC}"

    WORKERS=$(ps aux | grep train_orchestrator | grep -v grep | wc -l)
    if [ "$WORKERS" -ge 8 ]; then
        WC="$GREEN"; WM="healthy"
    elif [ "$WORKERS" -gt 0 ]; then
        WC="$YELLOW"; WM="starting/stopping"
    else
        WC="$GRAY"; WM="idle"
    fi
    echo -e "  Torchrun procs:   ${WC}${WORKERS}${NC} ${DIM}(${WM})${NC}"

    if [ -f /workspace/automation/chain.pid ]; then
        CP=$(cat /workspace/automation/chain.pid)
        if ps -p $CP > /dev/null 2>&1; then
            echo -e "  Chain runner:     ${GREEN}alive${NC} ${DIM}(PID $CP)${NC}"
        else
            echo -e "  Chain runner:     ${RED}DEAD${NC} ${DIM}(PID $CP)${NC}"
        fi
    fi

    if [ -f /workspace/automation/watchdog.pid ]; then
        WDP=$(cat /workspace/automation/watchdog.pid)
        if ps -p $WDP > /dev/null 2>&1; then
            echo -e "  Watchdog:         ${GREEN}alive${NC} ${DIM}(PID $WDP)${NC}"
        else
            echo -e "  Watchdog:         ${RED}DEAD${NC} ${DIM}(PID $WDP)${NC}"
        fi
    fi

    if [ -f "$FLAG_DIR/WATCHDOG.status" ]; then
        REASON=$(grep "^reason:" "$FLAG_DIR/WATCHDOG.status" | cut -d: -f2- | sed 's/^ *//')
        echo -e "  Watchdog anomaly: ${YELLOW}${REASON}${NC}"
    else
        echo -e "  Watchdog anomaly: ${GREEN}none${NC}"
    fi

    DISK_INFO=$(df -h /workspace | awk 'NR==2 {print $4 " free (" $5 " used)"}')
    DISK_PCT=$(df /workspace | awk 'NR==2 {print $5}' | tr -d '%')
    if [ "$DISK_PCT" -lt 95 ]; then
        DC="$GREEN"
    elif [ "$DISK_PCT" -lt 99 ]; then
        DC="$YELLOW"
    else
        DC="$RED"
    fi
    echo -e "  Disk:             ${DC}${DISK_INFO}${NC}"

    CKPT_SIZE=$(du -sh /workspace/checkpoints/ 2>/dev/null | cut -f1)
    echo -e "  Checkpoints:      ${CKPT_SIZE}"
    echo ""

    # =====================================================================
    # SECTION 5: REFERENCE VALUES
    # =====================================================================
    echo -e "${BOLD}▸ Reference values (Phase 3 seed 1):${NC}"
    for run_id in "${!HISTORICAL[@]}"; do
        echo -e "  ${DIM}${run_id}: ${HISTORICAL[$run_id]}${NC}"
    done
    echo ""

    # =====================================================================
    # SECTION 6: LATEST CHAIN LOG (no color leaks)
    # =====================================================================
    echo -e "${BOLD}▸ Latest chain events:${NC}"
    if [ -f "$CHAIN_LOG" ]; then
        tail -5 "$CHAIN_LOG" | while IFS= read -r line; do
            printf "  ${DIM}%s${NC}\n" "$line"
        done
    fi
    echo ""

    echo -e "${DIM}────────────────────────────────────────────────────────────────────────${NC}"
    echo -e "${DIM}  Refreshing every 30s. Ctrl-C to exit. Dashboard does not affect any run.${NC}"

    sleep 30
done
