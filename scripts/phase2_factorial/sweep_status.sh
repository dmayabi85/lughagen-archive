#!/bin/bash
# Print a snapshot of sweep progress + current cell status

echo "════════════════════════════════════════════════════════════"
echo "  LughaGen Phase 2 Sweep Status — $(date -u +'%Y-%m-%d %H:%M:%S UTC')"
echo "════════════════════════════════════════════════════════════"
echo ""

# Sweep alive?
if tmux has-session -t lughagen-sweep 2>/dev/null; then
    echo "Sweep session:    ALIVE (lughagen-sweep)"
else
    echo "Sweep session:    DEAD or not yet launched"
fi

# Completed cells
COMPLETED=$(grep -c ",Phase2_Factorial," /workspace/registry/runs.csv 2>/dev/null || echo "0")
echo "Cells completed:  $COMPLETED / 32"

# Currently active cell (newest log)
LATEST_LOG=$(ls -t /workspace/logs/phase2_RUN-P2-C*.log 2>/dev/null | head -1)
if [ -n "$LATEST_LOG" ]; then
    CURRENT_CELL=$(basename "$LATEST_LOG" | grep -oE "RUN-P2-C[0-9]+-S[0-9]+")
    LOG_AGE_MIN=$(( ($(date +%s) - $(stat -c %Y "$LATEST_LOG")) / 60 ))
    echo "Currently active: $CURRENT_CELL"
    echo "  log file:       $LATEST_LOG"
    echo "  last modified:  $LOG_AGE_MIN minutes ago"
    echo ""
    echo "  Latest training output:"
    grep "{'loss':" "$LATEST_LOG" | tail -3 | sed 's/^/    /'
    echo ""
    echo "  Step progress:"
    tail -200 "$LATEST_LOG" | grep -oE "[0-9]+/6000" | tail -1 | sed 's/^/    /'
fi
echo ""

# Disk
echo "Disk /:  $(df -h / | tail -1 | awk '{print $3 " used, " $4 " free (" $5 ")"}')"

# GPUs
echo "GPUs:"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits | \
    awk -F', ' '{printf "  GPU%d: %d%% util, %.1f GB used\n", $1, $2, $3/1024}'

echo ""
echo "════════════════════════════════════════════════════════════"
echo "Run again anytime: bash /workspace/scripts/sweep_status.sh"
echo "Or watch repeatedly: watch -n 60 bash /workspace/scripts/sweep_status.sh"
echo "════════════════════════════════════════════════════════════"
