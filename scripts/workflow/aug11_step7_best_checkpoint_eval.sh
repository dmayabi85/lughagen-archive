#!/bin/bash
# Aug 11 Step 7: Best-checkpoint BPC eval for available checkpoints
# Adds ~2 hours to Aug 11 workflow.
set -e

VAL_CLEAN="/workspace/data/corpus/splits_v2/pathD_v2_filtered/validation_shared_v3_clean.jsonl"

echo "=== Aug 11 Step 7: Best-Checkpoint BPC Eval ==="
echo "Available checkpoints only: Llama-FFT-S1 step 5400, Llama-QLoRA step 5400"
echo ""

# Llama-FFT step 5400 (primary case of interest)
LFFT_CKPT="/workspace/checkpoints/RUN-P3-FFT/checkpoint-5400"
LFFT_OUTPUT="/workspace/results/P3_LLAMA_FFT_S1_best_step5400_eval_CLEAN.json"

if [ -d "$LFFT_CKPT" ]; then
    echo "--- Llama-FFT-S1 step 5400 ---"
    python /workspace/scripts/eval_per_language_5tracks.py \
        --checkpoint "$LFFT_CKPT" \
        --val-jsonl "$VAL_CLEAN" \
        --output "$LFFT_OUTPUT" \
        --attn-impl sdpa \
        --model-family llama 2>&1 | \
        tee /workspace/logs/aug11_best_ckpt_llama_fft_s1.log
    
    if [ -f "$LFFT_OUTPUT" ]; then
        python -c "
import json
with open('$LFFT_OUTPUT') as f:
    d = json.load(f)
p = d.get('pooled', {})
bpc = p.get('bpc', '?')
print(f'  Best-checkpoint pooled BPC: {bpc:.4f}' if isinstance(bpc, float) else f'  BPC: {bpc}')
print(f'  Final-checkpoint reference: 1.210')
if isinstance(bpc, float):
    print(f'  ΔBPC: {bpc - 1.210:+.4f} ({(bpc - 1.210)/1.210*100:+.2f}%)')
"
    fi
else
    echo "✗ Llama-FFT-S1 step 5400 checkpoint missing"
fi

echo ""

# Llama-QLoRA step 5400 (plateau confirmation)
LQ_CKPT="/workspace/checkpoints/RUN-P3-CLEAN-QLORA/checkpoint-5400"
LQ_OUTPUT="/workspace/results/P3_LLAMA_QLORA_best_step5400_eval_CLEAN.json"

if [ -d "$LQ_CKPT" ]; then
    echo "--- Llama-QLoRA step 5400 ---"
    python /workspace/scripts/eval_per_language_5tracks.py \
        --checkpoint "$LQ_CKPT" \
        --val-jsonl "$VAL_CLEAN" \
        --output "$LQ_OUTPUT" \
        --attn-impl sdpa \
        --model-family llama 2>&1 | \
        tee /workspace/logs/aug11_best_ckpt_llama_qlora.log
    
    if [ -f "$LQ_OUTPUT" ]; then
        python -c "
import json
with open('$LQ_OUTPUT') as f:
    d = json.load(f)
p = d.get('pooled', {})
bpc = p.get('bpc', '?')
print(f'  Step 5400 pooled BPC: {bpc:.4f}' if isinstance(bpc, float) else f'  BPC: {bpc}')
print(f'  Final-checkpoint reference: 1.312')
if isinstance(bpc, float):
    print(f'  ΔBPC: {bpc - 1.312:+.4f} (expected ~0 - plateau)')
"
    fi
else
    echo "✗ Llama-QLoRA step 5400 checkpoint missing"
fi

echo ""
echo "=== Summary ==="
echo "Definitive best-checkpoint numbers now available for:"
echo "  - Llama-FFT-S1 step 5400 (soft overfitting case)"
echo "  - Llama-QLoRA step 5400 (plateau confirmation)"
echo ""
echo "Cross-seed and Llama-CPT best-checkpoint values remain estimated"
echo "from trajectory analysis (documented in Appendix 5.A)."
