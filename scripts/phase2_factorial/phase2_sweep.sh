#!/bin/bash
# Phase 2 Factorial Sweep Launcher
# 
# Runs the remaining Phase 2 factorial cells sequentially.
# - Reads schedule from factorial_matrix.build_schedule()
# - Skips cells already complete in /workspace/registry/runs.csv
# - For each remaining cell: launches torchrun via train_orchestrator.py
# - Continues on individual cell failure (logs, moves to next)
# - Writes progress to /workspace/logs/phase2_sweep_status.log
# - Resumable: rerunning the script skips completed cells
#
# Usage:
#   bash /workspace/scripts/phase2_sweep.sh

set -u   # error on undefined variables (NOT -e, we want to continue on cell failure)

STATUS_LOG=/workspace/logs/phase2_sweep_status.log
SWEEP_START=$(date -u +%Y-%m-%dT%H:%M:%SZ)

mkdir -p /workspace/logs /workspace/checkpoints

echo "" >> "$STATUS_LOG"
echo "=== Phase 2 sweep invocation: $SWEEP_START ===" >> "$STATUS_LOG"

# Build the list of cells still TODO
CELLS_JSON=$(python3 <<'PYEOF'
import json
import csv
import sys
sys.path.insert(0, '/workspace/scripts')
from factorial_matrix import build_schedule

completed = set()
try:
    with open('/workspace/registry/runs.csv') as f:
        for r in csv.DictReader(f):
            rid = r.get('factorial_run_id', '')
            phase = r.get('phase', '')
            # Only count valid post-restart factorial cells
            # (PRE_RESTART_v1 rows excluded by exact phase match)
            if rid.startswith('RUN-P2-C') and phase == 'Phase2_Factorial':
                if r.get('metrics.validation_perplexity') and r.get('wall_clock_seconds'):
                    completed.add(rid)
except FileNotFoundError:
    pass

schedule = build_schedule()
todo = []
for cell in schedule:
    if cell.factorial_run_id in completed:
        continue
    todo.append({
        'id': cell.factorial_run_id,
        'seed': cell.seed,
        'lr_schedule': cell.lr_schedule,
        'batch_size': cell.batch_size,
        'optimiser': cell.optimiser,
        'sequence_length': cell.sequence_length,
        'data_ordering': cell.data_ordering,
    })

print(json.dumps({'todo': todo, 'completed_count': len(completed), 'total': len(schedule)}))
PYEOF
)

TODO_COUNT=$(echo "$CELLS_JSON" | python3 -c "import json, sys; d = json.load(sys.stdin); print(len(d['todo']))")
COMPLETED_COUNT=$(echo "$CELLS_JSON" | python3 -c "import json, sys; d = json.load(sys.stdin); print(d['completed_count'])")
TOTAL=$(echo "$CELLS_JSON" | python3 -c "import json, sys; d = json.load(sys.stdin); print(d['total'])")

{
echo "Total factorial cells: $TOTAL"
echo "Already completed:     $COMPLETED_COUNT"
echo "To run this session:   $TODO_COUNT"
echo "Estimated wall clock:  $((TODO_COUNT * 12)) hours ($(echo "scale=1; $TODO_COUNT * 12 / 24" | bc) days)"
echo "Estimated GPU-hours:   $((TODO_COUNT * 89))"
echo ""
} | tee -a "$STATUS_LOG"

if [ "$TODO_COUNT" -eq "0" ]; then
    echo "All cells already complete. Nothing to do." | tee -a "$STATUS_LOG"
    exit 0
fi

# Iterate through TODO list
CELL_NUM=0
echo "$CELLS_JSON" | python3 -c "
import json, sys
d = json.load(sys.stdin)
for cell in d['todo']:
    print('|'.join([cell['id'], str(cell['seed']), cell['lr_schedule'],
                    str(cell['batch_size']), cell['optimiser'],
                    str(cell['sequence_length']), cell['data_ordering']]))
" | while IFS='|' read -r CELL_ID SEED LR_SCHED BATCH_SIZE OPTIMISER SEQ_LEN DATA_ORDERING; do
    CELL_NUM=$((CELL_NUM + 1))
    
    CELL_START=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    {
    echo ""
    echo "=== [$CELL_NUM/$TODO_COUNT] $CELL_ID started: $CELL_START ==="
    echo "  seed=$SEED lr=$LR_SCHED batch=$BATCH_SIZE opt=$OPTIMISER seq=$SEQ_LEN order=$DATA_ORDERING"
    } | tee -a "$STATUS_LOG"
    
    # Clean output dir if exists from a partial prior attempt
    OUTPUT_DIR="/workspace/checkpoints/$CELL_ID"
    if [ -d "$OUTPUT_DIR" ]; then
        echo "  (cleaning partial output dir)" | tee -a "$STATUS_LOG"
        rm -rf "$OUTPUT_DIR"
    fi
    mkdir -p "$OUTPUT_DIR"
    
    LOG_FILE="/workspace/logs/phase2_${CELL_ID}_$(date -u +%Y%m%dT%H%M%SZ).log"
    
    # Launch the cell
    torchrun --nproc_per_node=8 --nnodes=1 \
        /workspace/scripts/train_orchestrator.py \
        --phase Phase2_Factorial \
        --factorial-run-id "$CELL_ID" \
        --seed "$SEED" \
        --lr-schedule "$LR_SCHED" \
        --batch-size "$BATCH_SIZE" \
        --optimiser "$OPTIMISER" \
        --sequence-length "$SEQ_LEN" \
        --data-ordering "$DATA_ORDERING" \
        --train-jsonl /workspace/data/corpus/splits_v2/pathD/train.jsonl \
        --val-jsonl /workspace/data/corpus/splits_v2/validation_shared_v3.jsonl \
        --output-dir "$OUTPUT_DIR" \
        > "$LOG_FILE" 2>&1
    
    EXIT_CODE=$?
    CELL_END=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    
    if [ "$EXIT_CODE" -eq "0" ]; then
        # Verify by checking registry has the row with required fields
        ROW_OK=$(python3 -c "
import csv
with open('/workspace/registry/runs.csv') as f:
    for r in csv.DictReader(f):
        if r.get('factorial_run_id') == '$CELL_ID' and r.get('metrics.validation_perplexity'):
            print('OK')
            break
    else:
        print('MISSING')")
        if [ "$ROW_OK" = "OK" ]; then
            echo "=== [$CELL_NUM/$TODO_COUNT] $CELL_ID COMPLETE: $CELL_END (log: $LOG_FILE) ===" | tee -a "$STATUS_LOG"
        else
            echo "=== [$CELL_NUM/$TODO_COUNT] $CELL_ID PARTIAL FAILURE: process exited 0 but registry row missing ===" | tee -a "$STATUS_LOG"
            echo "  log: $LOG_FILE" | tee -a "$STATUS_LOG"
        fi
    else
        echo "=== [$CELL_NUM/$TODO_COUNT] $CELL_ID FAILED (exit $EXIT_CODE): $CELL_END ===" | tee -a "$STATUS_LOG"
        echo "  log: $LOG_FILE" | tee -a "$STATUS_LOG"
        # Continue to next cell, don't abort the whole sweep
    fi
done

SWEEP_END=$(date -u +%Y-%m-%dT%H:%M:%SZ)
{
echo ""
echo "=== Phase 2 sweep ended: $SWEEP_END ==="
} | tee -a "$STATUS_LOG"
