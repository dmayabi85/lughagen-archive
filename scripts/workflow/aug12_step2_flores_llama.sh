#!/bin/bash
# Aug 12 Step 2: Run FLORES+ eval on Llama models (~6 hours total)
set -e

mkdir -p /workspace/results/flores_plus

MODELS=(
    "llama_cpt:LughaGen-Llama-8B-CPT-S137"
    "llama_fft:LughaGen-Llama-8B-FFT-S137"
    "llama_qlora:LughaGen-Llama-8B-QLoRA-S137"
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
        --model-family llama \
        --output "$OUTPUT" \
        --n-samples 500 2>&1 | tee /workspace/logs/flores_${LABEL}.log
    
    echo "  Result: $OUTPUT"
done
