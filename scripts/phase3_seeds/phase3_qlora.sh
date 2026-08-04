#!/bin/bash
# ==============================================================================
# LughaGen Phase 3 — RUN-P3-CLEAN-QLORA
# ==============================================================================
# QLoRA baseline on the filtered clean corpus.  Uses C05-S2's winning
# hyperparameters exactly (AdamW / bs=256 / seq=1024 / cosine / curriculum /
# seed=137).
#
# This is the fair-baseline reference for Phase 3 method comparison.
# FFT and CPT ablations will be compared against THIS number.
#
# Expected wall clock: ~21-22 hours
# Expected val_pph: ~23.5 to 24.0 (within noise of C05-S2's 23.704)
# ==============================================================================

set -e

CELL_ID="RUN-P3-CLEAN-QLORA"
OUTPUT_DIR="/workspace/checkpoints/${CELL_ID}"
LOG_FILE="/workspace/logs/phase3_${CELL_ID}_$(date -u +%Y%m%dT%H%M%SZ).log"
STATUS_LOG="/workspace/logs/phase3_status.log"

FILTERED_TRAIN="/workspace/data/corpus/splits_v2/pathD_v2_filtered/train.jsonl"
FILTERED_VAL="/workspace/data/corpus/splits_v2/pathD_v2_filtered/validation_shared_v3.jsonl"

# ---- Sanity checks ----
if [ ! -f "$FILTERED_TRAIN" ]; then
    echo "ERROR: Filtered training corpus missing at $FILTERED_TRAIN" >&2
    exit 1
fi

if [ -d "$OUTPUT_DIR" ]; then
    echo "WARN: $OUTPUT_DIR already exists. Removing to start clean."
    rm -rf "$OUTPUT_DIR"
fi
mkdir -p "$OUTPUT_DIR"

# ---- Log launch metadata ----
{
    echo "=========================================="
    echo "Phase 3 launch: $CELL_ID"
    echo "Started: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "Config:  QLoRA on filtered corpus (C05 hyperparameters)"
    echo "Train:   $FILTERED_TRAIN"
    echo "Val:     $FILTERED_VAL"
    echo "Log:     $LOG_FILE"
    echo "Output:  $OUTPUT_DIR"
    echo "=========================================="
} | tee -a "$STATUS_LOG"

# ---- Launch ----
CELL_START=$(date -u +%Y-%m-%dT%H:%M:%SZ)

torchrun --nproc_per_node=8 --nnodes=1 \
    /workspace/scripts/train_orchestrator.py \
    --phase Phase3_PEFT \
    --factorial-run-id "$CELL_ID" \
    --seed 137 \
    --lr-schedule cosine \
    --batch-size 256 \
    --optimiser AdamW \
    --sequence-length 1024 \
    --data-ordering curriculum \
    --peft-method qlora_r16 \
    --skip-preflight \
    --train-jsonl "$FILTERED_TRAIN" \
    --val-jsonl "$FILTERED_VAL" \
    --output-dir "$OUTPUT_DIR" \
    > "$LOG_FILE" 2>&1

EXIT_CODE=$?
CELL_END=$(date -u +%Y-%m-%dT%H:%M:%SZ)

# ---- Verify success ----
if [ "$EXIT_CODE" -eq "0" ]; then
    VAL_PPH=$(python3 -c "
import csv
for r in csv.DictReader(open('/workspace/registry/runs.csv')):
    if r.get('factorial_run_id') == '$CELL_ID':
        print(r.get('metrics.validation_perplexity', 'N/A'))
        break
" 2>/dev/null || echo "?")
    {
        echo "SUCCESS: $CELL_ID complete"
        echo "  Started:  $CELL_START"
        echo "  Ended:    $CELL_END"
        echo "  val_pph:  $VAL_PPH"
        echo "  Baseline: 23.704 (C05-S2 on unfiltered)"
        echo "  Log:      $LOG_FILE"
    } | tee -a "$STATUS_LOG"
else
    {
        echo "FAIL: $CELL_ID exited $EXIT_CODE"
        echo "  Log: $LOG_FILE"
        tail -30 "$LOG_FILE"
    } | tee -a "$STATUS_LOG"
    exit $EXIT_CODE
fi
