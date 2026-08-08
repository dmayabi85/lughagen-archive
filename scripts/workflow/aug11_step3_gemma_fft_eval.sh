#!/bin/bash
# Aug 11 Step 3: Run per-language BPC eval on Gemma-FFT
set -e

CKPT="/workspace/checkpoints/RUN-P3.5-GEMMA2-FFT-S137/checkpoint-6000"
VAL_CLEAN="/workspace/data/corpus/splits_v2/pathD_v2_filtered/validation_shared_v3_clean.jsonl"
OUTPUT="/workspace/results/P3.5_GEMMA_FFT_S137_per_language_eval_CLEAN.json"

echo "=== Aug 11 Step 3: Gemma-FFT Per-Language Eval ==="
echo "Checkpoint: $CKPT"
echo "Validation: $VAL_CLEAN"
echo "Expected time: ~1.5-2 hours"
echo ""

# Verify inputs exist
if [ ! -d "$CKPT" ]; then
    echo "✗ Checkpoint missing"; exit 1
fi
if [ ! -f "$VAL_CLEAN" ]; then
    echo "✗ Validation missing"; exit 1
fi

# Run eval — note: uses sdpa attention per Amendment 11
python /workspace/scripts/eval_per_language_5tracks.py \
    --checkpoint "$CKPT" \
    --val-jsonl "$VAL_CLEAN" \
    --output "$OUTPUT" \
    --attn-impl sdpa 2>&1 | tee /workspace/logs/aug11_gemma_fft_eval.log

echo ""
echo "=== Result summary ==="
python << 'PYEOF'
import json
with open("/workspace/results/P3.5_GEMMA_FFT_S137_per_language_eval_CLEAN.json") as f:
    d = json.load(f)

print("Per-language BPC:")
for lang, r in d.get("per_language", {}).items():
    bpc = r.get("bpc", "?")
    print(f"  {lang}: {bpc:.4f}" if isinstance(bpc, float) else f"  {lang}: {bpc}")

pooled = d.get("pooled", {})
if pooled:
    pbpc = pooled.get("bpc", "?")
    print(f"\nPooled BPC: {pbpc:.4f}" if isinstance(pbpc, float) else f"\nPooled BPC: {pbpc}")
PYEOF
