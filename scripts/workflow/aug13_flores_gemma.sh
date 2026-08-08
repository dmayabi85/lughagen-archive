#!/bin/bash
# Aug 13: FLORES+ eval on Gemma models
set -e

mkdir -p /workspace/results/flores_plus

MODELS=(
    "gemma_cpt:LughaGen-Gemma2-27B-CPT-S137"
    "gemma_fft:LughaGen-Gemma2-27B-FFT-S137"
)

for MODEL_PAIR in "${MODELS[@]}"; do
    LABEL=${MODEL_PAIR%:*}
    NAME=${MODEL_PAIR#*:}
    CKPT="/workspace/releases/lughagen/$NAME"
    OUTPUT="/workspace/results/flores_plus/${LABEL}.json"
    
    echo ""
    echo "=== Running FLORES+ on $LABEL ==="
    
    if [ ! -d "$CKPT" ]; then
        echo "  ✗ Missing: $CKPT"
        continue
    fi
    
    python /workspace/scripts/eval_flores_plus.py \
        --checkpoint "$CKPT" \
        --model-family gemma \
        --output "$OUTPUT" \
        --n-samples 500 2>&1 | tee /workspace/logs/flores_${LABEL}.log
done
