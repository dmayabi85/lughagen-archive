#!/bin/bash
# Live dashboard for LughaGen BPC runs (FFT, CPT, QLoRA).
# Refreshes every 15s. Ctrl-C to exit. Read-only.

RUNS=(
    "lughagen_fft_s137:LughaGen FFT seed 137"
    "lughagen_cpt_s137:LughaGen CPT seed 137"
    "lughagen_qlora_merged_s137:LughaGen QLoRA merged"
)

RESULTS_DIR="/workspace/results"
LOG_DIR="/workspace/logs"

GREEN=$'\033[0;32m'
YELLOW=$'\033[0;33m'
RED=$'\033[0;31m'
BLUE=$'\033[0;34m'
BOLD=$'\033[1m'
RESET=$'\033[0m'

while true; do
    clear
    echo "${BOLD}╔══════════════════════════════════════════════════════════════════════════╗${RESET}"
    echo "${BOLD}║          LughaGen BPC — Live Dashboard                                 ║${RESET}"
    echo "${BOLD}║          $(date -u '+%Y-%m-%d %H:%M:%S UTC')                                          ║${RESET}"
    echo "${BOLD}╚══════════════════════════════════════════════════════════════════════════╝${RESET}"
    echo ""

    RUNNING_PROC=$(ps aux | grep eval_baseline_bpc | grep -v grep | head -1)
    if [ -n "$RUNNING_PROC" ]; then
        RUNNING_MODEL=$(echo "$RUNNING_PROC" | grep -oE -- '--model-path[[:space:]]+[^[:space:]]+' | awk '{print $2}')
        RUNTIME=$(echo "$RUNNING_PROC" | awk '{print $10}')
        echo "${GREEN}▸ ACTIVE:${RESET} ${BOLD}${RUNNING_MODEL}${RESET} (runtime: ${RUNTIME})"
    else
        DONE_COUNT=0
        for entry in "${RUNS[@]}"; do
            SLUG="${entry%%:*}"
            [ -f "${RESULTS_DIR}/baseline_bpc_${SLUG}_per_language_eval.json" ] && DONE_COUNT=$((DONE_COUNT+1))
        done
        if [ "$DONE_COUNT" -eq "${#RUNS[@]}" ]; then
            echo "${GREEN}▸ COMPLETE:${RESET} all ${#RUNS[@]} LughaGen BPC runs finished"
        else
            echo "${YELLOW}▸ IDLE:${RESET} no eval running (${DONE_COUNT}/${#RUNS[@]} complete)"
        fi
    fi
    echo ""

    echo "${BOLD}▸ Run sequence:${RESET}"
    for entry in "${RUNS[@]}"; do
        SLUG="${entry%%:*}"
        NAME="${entry##*:}"
        RESULT_FILE="${RESULTS_DIR}/baseline_bpc_${SLUG}_per_language_eval.json"
        LOG_FILE="${LOG_DIR}/gap3_${SLUG%_s137}.log"
        # Try alternate log naming
        [ ! -f "$LOG_FILE" ] && LOG_FILE="${LOG_DIR}/gap3_${SLUG}.log"

        if [ -f "$RESULT_FILE" ]; then
            METRICS=$(python3 -c "
import json
try:
    d = json.load(open('$RESULT_FILE'))
    r = d['results']
    pool_bpc = r.get('pooled', {}).get('bpc', None)
    pool_ppl = r.get('pooled', {}).get('perplexity', None)
    swa = r.get('swahili_authentic', {}).get('bpc', None)
    kik = r.get('kikuyu', {}).get('bpc', None)
    luo = r.get('luo', {}).get('bpc', None)
    kam = r.get('kamba', {}).get('bpc', None)
    she = r.get('sheng', {}).get('bpc', None)
    def f(v): return f'{v:.3f}' if isinstance(v, (int, float)) else '?'
    print(f'{f(pool_bpc)}|{f(pool_ppl)}|{f(swa)}|{f(kik)}|{f(luo)}|{f(kam)}|{f(she)}')
except Exception as e:
    print(f'ERR|{str(e)[:30]}||||')
" 2>/dev/null)
            IFS='|' read -r POOL_BPC POOL_PPL SWA KIK LUO KAM SHE <<< "$METRICS"
            echo "  ${GREEN}✓${RESET} ${NAME}"
            echo "      BPC(pool)=${BOLD}${POOL_BPC}${RESET}  ppl(pool)=${POOL_PPL}"
            echo "      sw=${SWA}  kik=${KIK}  luo=${LUO}  kam=${KAM}  she=${SHE}"
        elif [ -f "$LOG_FILE" ] && [ -n "$RUNNING_PROC" ] && echo "$RUNNING_PROC" | grep -qE "${SLUG%_s137}|${SLUG}"; then
            LAST_EVAL=$(grep -oE '\[eval\][^:]+:[^n]+ppl=[0-9.]+[^ ]*' "$LOG_FILE" 2>/dev/null | tail -3)
            LAST_LINE=$(tail -1 "$LOG_FILE" 2>/dev/null | head -c 90)
            echo "  ${YELLOW}⟳${RESET} ${NAME}  RUNNING"
            if [ -n "$LAST_EVAL" ]; then
                echo "$LAST_EVAL" | while IFS= read -r line; do
                    echo "      ${line:0:90}"
                done
            elif [ -n "$LAST_LINE" ]; then
                echo "      ${LAST_LINE}"
            fi
        elif [ -f "$LOG_FILE" ]; then
            if grep -q "Traceback\|Error\|error" "$LOG_FILE" 2>/dev/null; then
                echo "  ${RED}✗${RESET} ${NAME}  ERROR"
                ERR_LINE=$(grep -E "Traceback|Error|error" "$LOG_FILE" 2>/dev/null | tail -1 | head -c 80)
                echo "      ${RED}${ERR_LINE}${RESET}"
            else
                echo "  ${YELLOW}○${RESET} ${NAME}  LOG_EXISTS_NO_RESULT"
            fi
        else
            echo "  ${BLUE}○${RESET} ${NAME}  PENDING"
        fi
    done
    echo ""

    echo "${BOLD}▸ GPU state:${RESET}"
    nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader,nounits 2>/dev/null | \
        awk -F', ' 'BEGIN {mem_sum=0; util_sum=0; n=0}
        {
            printf "  GPU%s: %5s/%5s MiB  util=%3s%%  temp=%s°C\n", $1, $2, $3, $4, $5
            mem_sum += $2; util_sum += $4; n++
        }
        END { if (n>0) printf "  Average: %d MiB used, %.1f%% util across %d GPUs\n", mem_sum/n, util_sum/n, n }'
    echo ""

    ACTIVE_LOG=$(ls -t /workspace/logs/gap3_lughagen_*.log 2>/dev/null | head -1)
    if [ -n "$ACTIVE_LOG" ]; then
        echo "${BOLD}▸ Latest log ($(basename $ACTIVE_LOG)):${RESET}"
        tail -4 "$ACTIVE_LOG" 2>/dev/null | while IFS= read -r line; do
            echo "  ${line:0:100}"
        done
    fi
    echo ""

    # Cross-baseline peek — show baselines already done for reference
    BASELINE_COUNT=$(ls /workspace/results/baseline_bpc_afrique_llama_8b_per_language_eval.json /workspace/results/baseline_bpc_lugha_llama_8b_wura_per_language_eval.json /workspace/results/baseline_bpc_lugha_llama_8b_wura_edu_per_language_eval.json /workspace/results/baseline_bpc_gemma_2_27b_per_language_eval.json 2>/dev/null | wc -l)
    if [ "$BASELINE_COUNT" -gt 0 ]; then
        echo "${BOLD}▸ External baselines already available for Chapter 6 comparison: ${BASELINE_COUNT}/4${RESET}"
    fi

    echo "────────────────────────────────────────────────────────────────────"
    echo "  Refreshing every 15s. Ctrl-C to exit. Dashboard is read-only."

    sleep 15
done
