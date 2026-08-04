#!/bin/bash
# ==============================================================================
# LughaGen Phase 3.5 -- RUN-P3.5-GEMMA2-CPT-S137
# ==============================================================================
# Continued Pre-Training on Gemma 2 27B base with strategy_c 8K vocab resize.
# Methodology parity with LughaGen Phase 3 CPT (same corpus, same tokenizer
# strategy, same optimizer, same seed) -- testing whether stronger base model
# beats LughaGen CPT (22.20 pooled val_ppl seed 137) under matched conditions.
#
# Chapter 6 story: Gemma 2 27B CPT joins Lugha-Llama-8B-wura and AfriqueLlama-8B
# as CPT comparison baselines. All four models trained/adapted with CPT method,
# making perplexity/BPC comparisons methodologically consistent.
#
# Depends on:
#   - /workspace/models/gemma-2-27b/                 (downloaded, verified)
#   - /workspace/gemma2_27b_prep/remapped_embeddings/embed_tokens.pt (v1.1)
#   - /workspace/gemma2_27b_prep/remapped_embeddings/lm_head.pt
#   - /workspace/gemma2_27b_prep/configs/ds_config_gemma27b.json
#   - training_loop.py Amendment 10 (LUGHAGEN_REMAP_DIR env var)
#
# Hyperparameters (mirror LughaGen CPT scaled for 27B):
#   - peak_lr: 5e-6 (half of LughaGen CPT 1e-5, 27B/8B scaling)
#   - warmup: 500 steps (matches LughaGen CPT)
#   - weight_decay: 0.1 (matches LughaGen CPT -- standard pretraining value)
#   - data_ordering: mixed (matches LughaGen CPT -- natural order, no code-switch curriculum)
#   - batch_size: 128 (reduced from LughaGen 256 for memory)
#   - sequence_length: 1024 (matches LughaGen)
#   - peft_method: cpt (matches LughaGen CPT)
#
# Expected wall clock: ~4 days at bs=128 with CPU offload (~735 GPU-hours)
# ==============================================================================
set -e

source /workspace/.venv/bin/activate
export PATH="/workspace/.venv/bin:$PATH"

CELL_ID="RUN-P3.5-GEMMA2-CPT-S137"
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
    echo "Run: bash /workspace/gemma2_27b_prep/scripts/download_gemma2_27b.sh" >&2
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
    echo "Phase 3.5 launch: $CELL_ID"
    echo "Started: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "Config:  Gemma 2 27B CPT on filtered corpus (methodology parity with LughaGen CPT)"
    echo "Model:   $GEMMA_MODEL"
    echo "Remap:   $REMAP_DIR"
    echo "Train:   $FILTERED_TRAIN"
    echo "Val:     $FILTERED_VAL"
    echo "DS cfg:  $DS_CONFIG"
    echo "Log:     $LOG_FILE"
    echo "Output:  $OUTPUT_DIR"
    echo "Env:     LUGHAGEN_REMAP_DIR=$LUGHAGEN_REMAP_DIR"
    echo "Method:  cpt (peak_lr=5e-6, warmup=500, wd=0.1, mixed)"
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
    --train-jsonl "$FILTERED_TRAIN" \
    --val-jsonl "$FILTERED_VAL" \
    --output-dir "$OUTPUT_DIR" \
    > "$LOG_FILE" 2>&1

EXIT_CODE=$?
CELL_END=$(date -u +%Y-%m-%dT%H:%M:%SZ)

if [ "$EXIT_CODE" -eq 0 ]; then
    VAL_PPH=$(/workspace/.venv/bin/python -c "
import csv
val = '?'
try:
    with open('/workspace/registry/runs.csv') as f:
        for r in csv.DictReader(f):
            if r.get('factorial_run_id') == '$CELL_ID':
                val = r.get('metrics.validation_perplexity', 'N/A')
                break
except Exception:
    pass
print(val)
")
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
