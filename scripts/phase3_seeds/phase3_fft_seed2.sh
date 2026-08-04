#!/bin/bash
# ==============================================================================
# LughaGen Phase 3 — RUN-P3-FFT-S2
# ==============================================================================
# Full Fine-Tuning seed 2 replicate for Gap 1 (seed variance measurement).
# IDENTICAL to phase3_fft.sh except:
#   - seed: 42 (was 137)
#   - factorial_run_id: RUN-P3-FFT-S2
#   - output dir: /workspace/checkpoints/RUN-P3-FFT-S2/
#
# Purpose: measure seed variance for the pooled val_pph of FFT to enable
# proper bootstrap CIs on the FFT-vs-CPT equivalence claim (Chapter 5).
#
# Expected wall clock: ~16-17 hours (same as original FFT run).
# Expected GPU-hours: ~133.
# ==============================================================================

set -e

# Force-activate the venv that has flash-attn, datasets, deepspeed installed.
source /workspace/.venv/bin/activate
export PATH="/workspace/.venv/bin:$PATH"

CELL_ID="RUN-P3-FFT-S2"
OUTPUT_DIR="/workspace/checkpoints/${CELL_ID}"
LOG_FILE="/workspace/logs/phase3_${CELL_ID}_$(date -u +%Y%m%dT%H%M%SZ).log"
STATUS_LOG="/workspace/logs/phase3_status.log"

FILTERED_TRAIN="/workspace/data/corpus/splits_v2/pathD_v2_filtered/train.jsonl"
FILTERED_VAL="/workspace/data/corpus/splits_v2/pathD_v2_filtered/validation_shared_v3.jsonl"
DS_CONFIG="/workspace/configs/ds_config_zero3.json"

# ---- Sanity checks ----
if [ ! -f "$FILTERED_TRAIN" ]; then
    echo "ERROR: Filtered training corpus missing at $FILTERED_TRAIN" >&2
    exit 1
fi
if [ ! -f "$DS_CONFIG" ]; then
    echo "ERROR: DeepSpeed config missing at $DS_CONFIG" >&2
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
    echo "Phase 3 launch: $CELL_ID (seed variance replicate)"
    echo "Started: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "Config:  FFT seed=42 on filtered corpus (replicate of RUN-P3-FFT for Gap 1)"
    echo "Train:   $FILTERED_TRAIN"
    echo "Val:     $FILTERED_VAL"
    echo "DS cfg:  $DS_CONFIG"
    echo "Log:     $LOG_FILE"
    echo "Output:  $OUTPUT_DIR"
    echo "=========================================="
} | tee -a "$STATUS_LOG"

CELL_START=$(date -u +%Y-%m-%dT%H:%M:%SZ)

/workspace/.venv/bin/torchrun --nproc_per_node=8 --nnodes=1 \
    /workspace/scripts/train_orchestrator.py \
    --phase Phase3_PEFT \
    --factorial-run-id "$CELL_ID" \
    --seed 42 \
    --lr-schedule cosine \
    --batch-size 256 \
    --optimiser AdamW \
    --sequence-length 1024 \
    --data-ordering curriculum \
    --peft-method full_ft \
    --peak-lr 2e-5 \
    --warmup-steps 200 \
    --weight-decay 0.01 \
    --deepspeed-config "$DS_CONFIG" \
    --skip-preflight \
    --train-jsonl "$FILTERED_TRAIN" \
    --val-jsonl "$FILTERED_VAL" \
    --output-dir "$OUTPUT_DIR" \
    > "$LOG_FILE" 2>&1

EXIT_CODE=$?
CELL_END=$(date -u +%Y-%m-%dT%H:%M:%SZ)

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
