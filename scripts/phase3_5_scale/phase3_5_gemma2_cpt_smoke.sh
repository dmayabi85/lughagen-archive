#!/bin/bash
# ==============================================================================
# LughaGen Phase 3.5 -- RUN-P3.5-GEMMA2-CPT-SMOKE (smoke test, 150 steps)
# ==============================================================================
# Validates DeepSpeed ZeRO-3 + CPU offload + Gemma remap pipeline before
# committing to the ~4-day full run. Uses orchestrator's --smoke-test flag
# which caps max_steps=150, eval_steps=50, save_steps=150, and skips registry.
#
# Expected wall clock: ~1-1.5 hrs (150 steps x 25-40 sec/step for 27B with offload,
# plus model load + eval + save)
# ==============================================================================
set -e

source /workspace/.venv/bin/activate
export PATH="/workspace/.venv/bin:$PATH"

CELL_ID="RUN-P3.5-GEMMA2-CPT-SMOKE"
OUTPUT_DIR="/workspace/checkpoints/${CELL_ID}"
LOG_FILE="/workspace/logs/phase3_5_${CELL_ID}_$(date -u +%Y%m%dT%H%M%SZ).log"
STATUS_LOG="/workspace/logs/phase3_status.log"

FILTERED_TRAIN="/workspace/data/corpus/splits_v2/pathD_v2_filtered/train.jsonl"
FILTERED_VAL="/workspace/data/corpus/splits_v2/pathD_v2_filtered/validation_shared_v3.jsonl"
DS_CONFIG="/workspace/gemma2_27b_prep/configs/ds_config_gemma27b.json"
GEMMA_MODEL="/workspace/models/gemma-2-27b"
REMAP_DIR="/workspace/gemma2_27b_prep/remapped_embeddings"

# ---- Sanity checks ----
if [ ! -f "$FILTERED_TRAIN" ]; then
    echo "ERROR: Filtered training corpus missing at $FILTERED_TRAIN" >&2
    exit 1
fi
if [ ! -f "$FILTERED_VAL" ]; then
    echo "ERROR: Validation corpus missing at $FILTERED_VAL" >&2
    exit 1
fi
if [ ! -f "$DS_CONFIG" ]; then
    echo "ERROR: DeepSpeed config missing at $DS_CONFIG" >&2
    exit 1
fi
if [ ! -d "$GEMMA_MODEL" ]; then
    echo "ERROR: Gemma 2 27B model missing at $GEMMA_MODEL" >&2
    exit 1
fi
if [ ! -f "$GEMMA_MODEL/config.json" ]; then
    echo "ERROR: Gemma model config.json missing -- download incomplete?" >&2
    exit 1
fi
if [ ! -f "$REMAP_DIR/embed_tokens.pt" ]; then
    echo "ERROR: Remapped embeddings missing at $REMAP_DIR/embed_tokens.pt" >&2
    echo "Run: python3 /workspace/gemma2_27b_prep/scripts/remap_gemma2_embeddings.py" >&2
    exit 1
fi
if [ ! -f "$REMAP_DIR/lm_head.pt" ]; then
    echo "ERROR: Remapped lm_head missing at $REMAP_DIR/lm_head.pt" >&2
    exit 1
fi

if [ -d "$OUTPUT_DIR" ]; then
    echo "WARN: $OUTPUT_DIR already exists. Removing to start clean."
    rm -rf "$OUTPUT_DIR"
fi
mkdir -p "$OUTPUT_DIR"
mkdir -p /workspace/logs

# ---- Point training_loop.py at Gemma remap tensors ----
export LUGHAGEN_REMAP_DIR="$REMAP_DIR"

# ---- Log launch metadata ----
{
    echo "=========================================="
    echo "Phase 3.5 SMOKE launch: $CELL_ID"
    echo "Started: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "Config:  Gemma 2 27B CPT SMOKE (150 steps)"
    echo "Model:   $GEMMA_MODEL"
    echo "Remap:   $REMAP_DIR"
    echo "DS cfg:  $DS_CONFIG"
    echo "Log:     $LOG_FILE"
    echo "Output:  $OUTPUT_DIR"
    echo "Env:     LUGHAGEN_REMAP_DIR=$LUGHAGEN_REMAP_DIR"
    echo "=========================================="
} | tee -a "$STATUS_LOG"

CELL_START=$(date -u +%Y-%m-%dT%H:%M:%SZ)

/workspace/.venv/bin/torchrun --nproc_per_node=8 --nnodes=1 \
    /workspace/scripts/train_orchestrator.py \
    --phase Phase3_PEFT \
    --factorial-run-id "$CELL_ID" \
    --seed 137 \
    --lr-schedule cosine \
    --batch-size 128 \
    --optimiser AdamW \
    --sequence-length 1024 \
    --data-ordering mixed \
    --peft-method cpt \
    --peak-lr 5e-6 \
    --warmup-steps 500 \
    --weight-decay 0.1 \
    --deepspeed-config "$DS_CONFIG" \
    --model-path "$GEMMA_MODEL" \
    --skip-preflight \
    --smoke-test \
    --train-jsonl "$FILTERED_TRAIN" \
    --val-jsonl "$FILTERED_VAL" \
    --output-dir "$OUTPUT_DIR" \
    > "$LOG_FILE" 2>&1

EXIT_CODE=$?
CELL_END=$(date -u +%Y-%m-%dT%H:%M:%SZ)

{
    echo "=========================================="
    echo "SMOKE $CELL_ID exited $EXIT_CODE"
    echo "  Started: $CELL_START"
    echo "  Ended:   $CELL_END"
    echo "  Log:     $LOG_FILE"
    echo "=========================================="
} | tee -a "$STATUS_LOG"

if [ "$EXIT_CODE" -ne 0 ]; then
    echo ""
    echo "=== Last 40 lines of log (for debugging) ==="
    tail -40 "$LOG_FILE"
    exit $EXIT_CODE
fi
