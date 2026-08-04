#!/bin/bash
# LughaGen Phase 3 — RUN-P3-CPT-S3 (seed variance replicate, seed=1337)
set -e
source /workspace/.venv/bin/activate
export PATH="/workspace/.venv/bin:$PATH"

CELL_ID="RUN-P3-CPT-S3"
OUTPUT_DIR="/workspace/checkpoints/${CELL_ID}"
LOG_FILE="/workspace/logs/phase3_${CELL_ID}_$(date -u +%Y%m%dT%H%M%SZ).log"
STATUS_LOG="/workspace/logs/phase3_status.log"

FILTERED_TRAIN="/workspace/data/corpus/splits_v2/pathD_v2_filtered/train.jsonl"
FILTERED_VAL="/workspace/data/corpus/splits_v2/pathD_v2_filtered/validation_shared_v3.jsonl"
DS_CONFIG="/workspace/configs/ds_config_zero3.json"

[ ! -f "$FILTERED_TRAIN" ] && { echo "ERROR: train missing"; exit 1; }
[ ! -f "$DS_CONFIG" ] && { echo "ERROR: ds config missing"; exit 1; }
[ -d "$OUTPUT_DIR" ] && rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

echo "Launch $CELL_ID at $(date -u +%FT%TZ)" | tee -a "$STATUS_LOG"

/workspace/.venv/bin/torchrun --nproc_per_node=8 --nnodes=1 \
    /workspace/scripts/train_orchestrator.py \
    --phase Phase3_PEFT \
    --factorial-run-id "$CELL_ID" \
    --seed 1337 \
    --lr-schedule cosine \
    --batch-size 256 \
    --optimiser AdamW \
    --sequence-length 1024 \
    --data-ordering mixed \
    --peft-method cpt \
    --peak-lr 1e-5 \
    --warmup-steps 500 \
    --weight-decay 0.1 \
    --deepspeed-config "$DS_CONFIG" \
    --skip-preflight \
    --train-jsonl "$FILTERED_TRAIN" \
    --val-jsonl "$FILTERED_VAL" \
    --output-dir "$OUTPUT_DIR" \
    > "$LOG_FILE" 2>&1

echo "$CELL_ID exited $? at $(date -u +%FT%TZ)" | tee -a "$STATUS_LOG"
