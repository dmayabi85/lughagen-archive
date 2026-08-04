#!/bin/bash
# Live dashboard for Phase 3.5 Gemma 2 27B CPT smoke test.
# Refreshes every 20s. Ctrl-C to exit. Read-only.

LOG_DIR="/workspace/logs"
OUTPUT_DIR="/workspace/checkpoints/RUN-P3.5-GEMMA2-CPT-SMOKE"
LOG_GLOB="/workspace/logs/phase3_5_RUN-P3.5-GEMMA2-CPT-SMOKE_*.log"
MAX_STEPS=150
EVAL_STEPS=50
SAVE_STEPS=150

GREEN=$'\033[0;32m'
YELLOW=$'\033[0;33m'
RED=$'\033[0;31m'
BLUE=$'\033[0;34m'
CYAN=$'\033[0;36m'
BOLD=$'\033[1m'
RESET=$'\033[0m'

while true; do
    clear
    echo "${BOLD}╔══════════════════════════════════════════════════════════════════════════╗${RESET}"
    echo "${BOLD}║          Phase 3.5 Gemma 2 27B CPT — SMOKE Dashboard                    ║${RESET}"
    echo "${BOLD}║          $(date -u '+%Y-%m-%d %H:%M:%S UTC')                                          ║${RESET}"
    echo "${BOLD}╚══════════════════════════════════════════════════════════════════════════╝${RESET}"
    echo ""

    LATEST_LOG=$(ls -t $LOG_GLOB 2>/dev/null | head -1)
    if [ -z "$LATEST_LOG" ]; then
        echo "${YELLOW}▸ No smoke log found yet${RESET}"
        sleep 20
        continue
    fi
    LOG_NAME=$(basename $LATEST_LOG)
    LOG_SIZE=$(wc -l < $LATEST_LOG 2>/dev/null || echo 0)

    # Process state
    TORCHRUN_PROCS=$(ps aux | grep -E "torchrun|train_orchestrator" | grep -v grep | wc -l)
    TRAIN_PROCS=$(ps aux | grep "train_orchestrator.py" | grep -v grep | wc -l)

    # Extract pipeline state from log
    HAS_ROUTING=$(grep -c "gemma-family: disabled attn_logit_softcapping" $LATEST_LOG 2>/dev/null || echo 0)
    HAS_ATTN_REQ=$(grep -c "\[attn\] model_type=gemma" $LATEST_LOG 2>/dev/null || echo 0)
    HAS_WEIGHTS=$(grep -c "Loading weights: 100%" $LATEST_LOG 2>/dev/null || echo 0)
    HAS_REMAP_LOAD=$(grep -c "\[remap\] loading token-aware" $LATEST_LOG 2>/dev/null || echo 0)
    HAS_REMAP_APPLIED=$(grep -c "\[remap\] applied" $LATEST_LOG 2>/dev/null || echo 0)
    HAS_ATTN_ENGAGED=$(grep -c "\[attn\] 'sdpa' engaged\|\[attn\] 'flash_attention_2' engaged" $LATEST_LOG 2>/dev/null || echo 0)
    HAS_SMOKE_MODE=$(grep -c "\[smoke\] max_steps" $LATEST_LOG 2>/dev/null || echo 0)
    HAS_TRAIN_START=$(grep -c "\[train\] starting" $LATEST_LOG 2>/dev/null || echo 0)

    # Extract current step
    CURRENT_STEP=$(grep -oE "^\s*[0-9]+%\|[^|]+\| ([0-9]+)/[0-9]+" $LATEST_LOG 2>/dev/null | tail -1 | grep -oE "[0-9]+/[0-9]+" | head -1)
    if [ -z "$CURRENT_STEP" ]; then
        # Try alternate pattern from tqdm
        CURRENT_STEP=$(grep -oE "\| [0-9]+/[0-9]+" $LATEST_LOG 2>/dev/null | tail -1 | grep -oE "[0-9]+/[0-9]+")
    fi

    # Extract latest loss
    LATEST_LOSS=$(grep -oE "'loss': '[0-9.]+'" $LATEST_LOG 2>/dev/null | tail -1)
    LATEST_LR=$(grep -oE "'learning_rate': '[0-9.e-]+'" $LATEST_LOG 2>/dev/null | tail -1)
    LATEST_GRADNORM=$(grep -oE "'grad_norm': '[0-9.]+'" $LATEST_LOG 2>/dev/null | tail -1)
    LATEST_EVAL=$(grep -oE "'eval_loss': '[0-9.]+'" $LATEST_LOG 2>/dev/null | tail -1)

    # Overall status
    if grep -q "smoke.*complete\|COMPLETE\|Writing model shards.*100%\|\[done\]" $LATEST_LOG 2>/dev/null; then
        echo "${GREEN}▸ STATUS: SMOKE COMPLETE${RESET}"
    elif grep -q "Traceback\|Error\|error\|OOM\|refuse to train" $LATEST_LOG 2>/dev/null; then
        echo "${RED}▸ STATUS: ERROR DETECTED${RESET}"
    elif [ "$HAS_TRAIN_START" -gt 0 ]; then
        echo "${GREEN}▸ STATUS: TRAINING (max_steps=${MAX_STEPS})${RESET}"
    elif [ "$HAS_ATTN_ENGAGED" -gt 0 ]; then
        echo "${GREEN}▸ STATUS: PRE-TRAIN SETUP (smoke config)${RESET}"
    elif [ "$HAS_REMAP_APPLIED" -gt 0 ]; then
        echo "${YELLOW}▸ STATUS: REMAP APPLIED, VERIFYING ATTENTION${RESET}"
    elif [ "$HAS_WEIGHTS" -gt 0 ]; then
        echo "${YELLOW}▸ STATUS: WEIGHTS LOADED, DATASET PREP${RESET}"
    else
        echo "${YELLOW}▸ STATUS: LOADING${RESET}"
    fi
    echo ""

    # Pipeline checkpoint list
    echo "${BOLD}▸ Pipeline checkpoints:${RESET}"
    if [ "$HAS_ROUTING" -gt 0 ]; then
        echo "  ${GREEN}✓${RESET} Amendment 11 routing: softcap disabled, SDPA requested"
    else
        echo "  ${BLUE}○${RESET} Amendment 11 routing: pending"
    fi
    if [ "$HAS_WEIGHTS" -gt 0 ]; then
        echo "  ${GREEN}✓${RESET} Model weights loaded (508 shards × 8 ranks)"
    else
        echo "  ${BLUE}○${RESET} Model weights: pending"
    fi
    if [ "$HAS_REMAP_LOAD" -gt 0 ]; then
        echo "  ${GREEN}✓${RESET} Remap tensors loaded from LUGHAGEN_REMAP_DIR"
    else
        echo "  ${BLUE}○${RESET} Remap load: pending"
    fi
    if [ "$HAS_REMAP_APPLIED" -gt 0 ]; then
        echo "  ${GREEN}✓${RESET} Remap applied (token-aware initialization)"
    else
        echo "  ${BLUE}○${RESET} Remap applied: pending"
    fi
    if [ "$HAS_ATTN_ENGAGED" -gt 0 ]; then
        echo "  ${GREEN}✓${RESET} Attention verified (spec §12 passed)"
    else
        echo "  ${BLUE}○${RESET} Attention verification: pending (CRITICAL)"
    fi
    if [ "$HAS_SMOKE_MODE" -gt 0 ]; then
        echo "  ${GREEN}✓${RESET} Smoke config active (max_steps=${MAX_STEPS})"
    else
        echo "  ${BLUE}○${RESET} Smoke config: pending"
    fi
    if [ "$HAS_TRAIN_START" -gt 0 ]; then
        echo "  ${GREEN}✓${RESET} Training loop started"
    else
        echo "  ${BLUE}○${RESET} Training loop: pending"
    fi
    echo ""

    # Training progress
    if [ -n "$CURRENT_STEP" ] && [ "$HAS_TRAIN_START" -gt 0 ]; then
        echo "${BOLD}▸ Training progress:${RESET}"
        echo "  Step:          $CURRENT_STEP"
        [ -n "$LATEST_LOSS" ] && echo "  $LATEST_LOSS"
        [ -n "$LATEST_GRADNORM" ] && echo "  $LATEST_GRADNORM"
        [ -n "$LATEST_LR" ] && echo "  $LATEST_LR"
        [ -n "$LATEST_EVAL" ] && echo "  ${CYAN}$LATEST_EVAL${RESET}"
        echo ""
    fi

    # Process count
    echo "${BOLD}▸ Processes:${RESET}"
    echo "  torchrun/orchestrator: $TORCHRUN_PROCS"
    echo "  training workers:      $TRAIN_PROCS"
    echo ""

    # GPU state
    echo "${BOLD}▸ GPU state:${RESET}"
    nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw --format=csv,noheader,nounits 2>/dev/null | \
        awk -F', ' 'BEGIN {mem_sum=0; util_sum=0; n=0; pwr_sum=0}
        {
            printf "  GPU%s: %5s/%5s MiB  util=%3s%%  temp=%s°C  pwr=%sW\n", $1, $2, $3, $4, $5, $6
            mem_sum += $2; util_sum += $4; n++; pwr_sum += $6
        }
        END { if (n>0) printf "  Avg: %d MiB used, %.1f%% util, %d W across %d GPUs\n", mem_sum/n, util_sum/n, pwr_sum/n, n }'
    echo ""

    # CPU state (relevant for CPU offload)
    CPU_MEM=$(free -g | awk 'NR==2 {printf "%d/%d GB used", $3, $2}')
    echo "${BOLD}▸ CPU:${RESET}"
    echo "  memory:        $CPU_MEM"
    echo ""

    # Disk
    DISK=$(df -h /workspace 2>/dev/null | tail -1 | awk '{ printf "%s free (%s used)", $4, $5 }')
    echo "${BOLD}▸ Disk /workspace:${RESET} $DISK"
    echo ""

    # Checkpoint dir
    if [ -d "$OUTPUT_DIR" ]; then
        CP_SIZE=$(du -sh "$OUTPUT_DIR" 2>/dev/null | awk '{print $1}')
        echo "${BOLD}▸ Checkpoint dir:${RESET} $OUTPUT_DIR ($CP_SIZE)"
    fi
    echo ""

    # Latest 4 log lines
    echo "${BOLD}▸ Latest log ($LOG_NAME, $LOG_SIZE lines):${RESET}"
    tail -4 "$LATEST_LOG" 2>/dev/null | while IFS= read -r line; do
        echo "  ${line:0:120}"
    done
    echo ""

    echo "────────────────────────────────────────────────────────────────────"
    echo "  Refreshing every 20s. Ctrl-C to exit. Dashboard is read-only."

    sleep 20
done
