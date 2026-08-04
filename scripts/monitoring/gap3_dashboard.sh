#!/bin/bash
# Live dashboard for Gap 3 baseline eval runs.
# Refreshes every 15 seconds. Ctrl-C to exit.

BASELINES=(
    "afrique_llama_8b:AfriqueLlama-8B"
    "lugha_llama_8b_wura:Lugha-Llama-8B-wura"
    "lugha_llama_8b_wura_edu:Lugha-Llama-8B-wura_edu"
    "gemma_2_27b:Gemma 2 27B untuned"
)

RESULTS_DIR="/workspace/results"
LOG_DIR="/workspace/logs"

# ANSI colors
GREEN=$'\033[0;32m'
YELLOW=$'\033[0;33m'
RED=$'\033[0;31m'
BLUE=$'\033[0;34m'
BOLD=$'\033[1m'
RESET=$'\033[0m'

while true; do
    clear
    echo "${BOLD}╔══════════════════════════════════════════════════════════════════════════╗${RESET}"
    echo "${BOLD}║          Gap 3 Baselines — Live Dashboard                              ║${RESET}"
    echo "${BOLD}║          $(date -u '+%Y-%m-%d %H:%M:%S UTC')                                          ║${RESET}"
    echo "${BOLD}╚══════════════════════════════════════════════════════════════════════════╝${RESET}"
    echo ""

    # Overall state
    RUNNING_PROC=$(ps aux | grep eval_baseline_bpc | grep -v grep | head -1)
    if [ -n "$RUNNING_PROC" ]; then
        RUNNING_MODEL=$(echo "$RUNNING_PROC" | grep -oE -- '--model-path[[:space:]]+[^[:space:]]+' | awk '{print $2}')
        echo "${GREEN}▸ ACTIVE:${RESET} eval running for ${BOLD}${RUNNING_MODEL}${RESET}"
    else
        # Check if all done
        DONE_COUNT=0
        for entry in "${BASELINES[@]}"; do
            SLUG="${entry%%:*}"
            [ -f "${RESULTS_DIR}/baseline_bpc_${SLUG}_per_language_eval.json" ] && DONE_COUNT=$((DONE_COUNT+1))
        done
        if [ "$DONE_COUNT" -eq "${#BASELINES[@]}" ]; then
            echo "${GREEN}▸ COMPLETE:${RESET} all ${#BASELINES[@]} baselines finished"
        else
            echo "${YELLOW}▸ IDLE:${RESET} no eval running (${DONE_COUNT}/${#BASELINES[@]} baselines complete)"
        fi
    fi
    echo ""

    # Per-baseline status
    echo "${BOLD}▸ Baseline sequence:${RESET}"
    for entry in "${BASELINES[@]}"; do
        SLUG="${entry%%:*}"
        NAME="${entry##*:}"
        RESULT_FILE="${RESULTS_DIR}/baseline_bpc_${SLUG}_per_language_eval.json"
        LOG_FILE="${LOG_DIR}/gap3_${SLUG}.log"

        if [ -f "$RESULT_FILE" ]; then
            # Extract pooled BPC + Swahili BPC
            POOLED_BPC=$(python3 -c "
import json
try:
    d = json.load(open('$RESULT_FILE'))
    p = d['results']['pooled']
    print(f\"{p['bpc']:.4f}\")
except Exception:
    print('?')
" 2>/dev/null)
            SWA_BPC=$(python3 -c "
import json
try:
    d = json.load(open('$RESULT_FILE'))
    print(f\"{d['results']['swahili_authentic']['bpc']:.4f}\")
except Exception:
    print('?')
" 2>/dev/null)
            echo "  ${GREEN}✓${RESET} ${NAME}  DONE   BPC(pooled)=${POOLED_BPC}  BPC(sw)=${SWA_BPC}"
        elif [ -f "$LOG_FILE" ]; then
            # Currently running or crashed — inspect log
            RUNNING=$(ps aux | grep eval_baseline_bpc | grep -v grep | grep -q "$SLUG\|${NAME//_/-}" && echo yes || echo no)
            LAST_LINE=$(tail -1 "$LOG_FILE" 2>/dev/null | head -c 60)
            if [ "$RUNNING" = "yes" ]; then
                echo "  ${YELLOW}⟳${RESET} ${NAME}  RUNNING"
                [ -n "$LAST_LINE" ] && echo "      last log: ${LAST_LINE}..."
            else
                # Log exists but no proc — check if error
                if grep -q "Traceback\|Error\|error" "$LOG_FILE" 2>/dev/null; then
                    echo "  ${RED}✗${RESET} ${NAME}  ERROR"
                    ERR_LINE=$(grep -E "Traceback|Error|error" "$LOG_FILE" 2>/dev/null | tail -1 | head -c 60)
                    echo "      ${RED}${ERR_LINE}${RESET}"
                else
                    echo "  ${YELLOW}○${RESET} ${NAME}  LOG_EXISTS_NO_RESULT"
                fi
            fi
        else
            echo "  ${BLUE}○${RESET} ${NAME}  PENDING"
        fi
    done
    echo ""

    # GPU state (only 1 GPU used for baseline eval, but show all in case)
    echo "${BOLD}▸ GPU state:${RESET}"
    nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader,nounits | \
        awk -F', ' 'BEGIN {mem_sum=0; util_sum=0; n=0}
        {
            printf "  GPU%s: %5s/%5s MiB  util=%3s%%  temp=%s°C\n", $1, $2, $3, $4, $5
            mem_sum += $2; util_sum += $4; n++
        }
        END { if (n>0) printf "  Average: %d MiB used, %.1f%% util across %d GPUs\n", mem_sum/n, util_sum/n, n }'
    echo ""

    # Disk
    echo "${BOLD}▸ Disk:${RESET}"
    df -h /workspace | tail -1 | awk '{ printf "  %-15s %s free (%s used)\n", "/workspace:", $4, $5 }'
    echo ""

    # Latest log line (whichever baseline is active)
    ACTIVE_LOG=$(ls -t /workspace/logs/gap3_*.log 2>/dev/null | head -1)
    if [ -n "$ACTIVE_LOG" ]; then
        echo "${BOLD}▸ Latest log line ($(basename $ACTIVE_LOG)):${RESET}"
        tail -3 "$ACTIVE_LOG" 2>/dev/null | sed 's/^/  /' | head -3
    fi
    echo ""

    echo "────────────────────────────────────────────────────────────────────"
    echo "  Refreshing every 15s. Ctrl-C to exit. Dashboard is read-only."

    sleep 15
done
