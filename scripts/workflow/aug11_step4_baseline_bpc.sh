#!/bin/bash
# Aug 11 Step 4: Cross-tokenizer BPC eval for Chapter 6 table
set -e

CKPT="/workspace/checkpoints/RUN-P3.5-GEMMA2-FFT-S137/checkpoint-6000"
VAL_CLEAN="/workspace/data/corpus/splits_v2/pathD_v2_filtered/validation_shared_v3_clean.jsonl"

echo "=== Aug 11 Step 4: Cross-tokenizer BPC (Chapter 6) ==="

python /workspace/scripts/eval_baseline_bpc.py \
    --checkpoint "$CKPT" \
    --val-jsonl "$VAL_CLEAN" \
    --model-family gemma \
    --attn-impl sdpa \
    --output /workspace/results/baseline_bpc_lughagen_gemma_fft_clean.json 2>&1 | \
    tee /workspace/logs/aug11_gemma_fft_baseline_bpc.log

echo ""
echo "=== Chapter 6 headline ==="
python -c "
import json
with open('/workspace/results/baseline_bpc_lughagen_gemma_fft_clean.json') as f:
    d = json.load(f)
print(f'Pooled BPC: {d.get(\"pooled_bpc\", \"?\")}')
print(f'Total tokens: {d.get(\"total_tokens\", \"?\")}')
print(f'Total bytes: {d.get(\"total_bytes\", \"?\")}')
"
