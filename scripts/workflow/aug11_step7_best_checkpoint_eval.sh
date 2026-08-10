#!/bin/bash
# Aug 11 Step 7: Best-checkpoint / closest-available-checkpoint eval for Llama models
# Adds ~2.5 hours to Aug 11 workflow.
set -e

VAL_CLEAN="/workspace/data/corpus/splits_v2/pathD_v2_filtered/validation_shared_v3_clean.jsonl"

echo "=== Aug 11 Step 7: Llama Best-Checkpoint Eval ==="
echo ""
echo "Coverage:"
echo "  - Llama-FFT-S1 step 5400 (TRUE trajectory minimum, eval_loss 2.955)"
echo "  - Llama-QLoRA step 5400 (plateau verification, eval_loss 3.165)"
echo "  - Llama-CPT-S1 step 5600 (CLOSEST-AVAILABLE, not true minimum)"
echo ""

# -------------------------------------------------------------------
# 1. Llama-FFT-S1 step 5400 — TRUE best checkpoint
# -------------------------------------------------------------------
LFFT_CKPT="/workspace/checkpoints/RUN-P3-FFT/checkpoint-5400"
LFFT_OUTPUT="/workspace/results/P3_LLAMA_FFT_S1_best_step5400_eval_CLEAN.json"

echo "--- [1/3] Llama-FFT-S1 step 5400 (true trajectory minimum) ---"
if [ -d "$LFFT_CKPT" ]; then
    python /workspace/scripts/eval_per_language_5tracks.py \
        --checkpoint "$LFFT_CKPT" \
        --val-jsonl "$VAL_CLEAN" \
        --output "$LFFT_OUTPUT" \
        --attn-impl sdpa \
        --model-family llama 2>&1 | \
        tee /workspace/logs/aug11_best_ckpt_llama_fft_s1.log
    
    python -c "
import json
with open('$LFFT_OUTPUT') as f:
    d = json.load(f)
p = d.get('pooled', {})
bpc = p.get('bpc', '?')
print(f'  Step 5400 pooled BPC: {bpc:.4f}' if isinstance(bpc, float) else f'  BPC: {bpc}')
print(f'  Final (step 6000) reference: 1.210')
if isinstance(bpc, float):
    print(f'  ΔBPC: {bpc - 1.210:+.4f} ({(bpc - 1.210)/1.210*100:+.2f}%)')
"
else
    echo "  ✗ Checkpoint missing: $LFFT_CKPT"
fi
echo ""

# -------------------------------------------------------------------
# 2. Llama-QLoRA step 5400 — plateau verification
# -------------------------------------------------------------------
LQ_CKPT="/workspace/checkpoints/RUN-P3-CLEAN-QLORA/checkpoint-5400"
LQ_OUTPUT="/workspace/results/P3_LLAMA_QLORA_best_step5400_eval_CLEAN.json"

echo "--- [2/3] Llama-QLoRA step 5400 (plateau verification) ---"
if [ -d "$LQ_CKPT" ]; then
    python /workspace/scripts/eval_per_language_5tracks.py \
        --checkpoint "$LQ_CKPT" \
        --val-jsonl "$VAL_CLEAN" \
        --output "$LQ_OUTPUT" \
        --attn-impl sdpa \
        --model-family llama 2>&1 | \
        tee /workspace/logs/aug11_best_ckpt_llama_qlora.log
    
    python -c "
import json
with open('$LQ_OUTPUT') as f:
    d = json.load(f)
p = d.get('pooled', {})
bpc = p.get('bpc', '?')
print(f'  Step 5400 pooled BPC: {bpc:.4f}' if isinstance(bpc, float) else f'  BPC: {bpc}')
print(f'  Final (step 6000) reference: 1.312')
if isinstance(bpc, float):
    print(f'  ΔBPC: {bpc - 1.312:+.4f} (expected ~0 - plateau)')
"
else
    echo "  ✗ Checkpoint missing: $LQ_CKPT"
fi
echo ""

# -------------------------------------------------------------------
# 3. Llama-CPT-S1 step 5600 — CLOSEST-AVAILABLE (NOT true minimum)
# -------------------------------------------------------------------
LCPT_CKPT="/workspace/checkpoints/RUN-P3-CPT/checkpoint-5600"
LCPT_OUTPUT="/workspace/results/P3_LLAMA_CPT_S1_closest_avail_step5600_eval_CLEAN.json"

echo "--- [3/3] Llama-CPT-S1 step 5600 (closest-available, NOT true minimum) ---"
echo "  Note: True trajectory minimum was step 5400 (eval_loss 3.091),"
echo "        which was pruned. Step 5600 (eval_loss 3.099) is next-best available."
echo ""

if [ -d "$LCPT_CKPT" ]; then
    python /workspace/scripts/eval_per_language_5tracks.py \
        --checkpoint "$LCPT_CKPT" \
        --val-jsonl "$VAL_CLEAN" \
        --output "$LCPT_OUTPUT" \
        --attn-impl sdpa \
        --model-family llama 2>&1 | \
        tee /workspace/logs/aug11_closest_ckpt_llama_cpt_s1.log
    
    python -c "
import json
with open('$LCPT_OUTPUT') as f:
    d = json.load(f)
p = d.get('pooled', {})
bpc = p.get('bpc', '?')
print(f'  Step 5600 pooled BPC: {bpc:.4f}' if isinstance(bpc, float) else f'  BPC: {bpc}')
print(f'  Final (step 6000) reference: 1.269')
if isinstance(bpc, float):
    print(f'  ΔBPC: {bpc - 1.269:+.4f} ({(bpc - 1.269)/1.269*100:+.2f}%)')
    print(f'  Note: trajectory suggests true best (step 5400) would be ~0.005 BPC better')
"
else
    echo "  ✗ Checkpoint missing: $LCPT_CKPT"
fi

echo ""
echo "=== Summary: Aug 11 Step 7 Complete ==="
echo ""
python << 'PYEOF'
import json
from pathlib import Path

RESULTS = Path("/workspace/results/")

evals = [
    ("Llama-FFT-S1", 1.210, "P3_LLAMA_FFT_S1_best_step5400_eval_CLEAN.json", "true minimum"),
    ("Llama-QLoRA", 1.312, "P3_LLAMA_QLORA_best_step5400_eval_CLEAN.json", "plateau"),
    ("Llama-CPT-S1", 1.269, "P3_LLAMA_CPT_S1_closest_avail_step5600_eval_CLEAN.json", "closest available"),
]

print(f"{'Model':<15} {'Ckpt Method':<20} {'Final BPC':>10} {'Best-Ckpt BPC':>14} {'ΔBPC':>8} {'Δ%':>8}")
print("-" * 82)
for name, final_bpc, filename, ckpt_type in evals:
    path = RESULTS / filename
    if path.exists():
        with path.open() as f:
            data = json.load(f)
        bpc = data.get('pooled', {}).get('bpc')
        if bpc:
            delta = bpc - final_bpc
            pct = (delta / final_bpc) * 100
            print(f"{name:<15} {ckpt_type:<20} {final_bpc:>10.4f} {bpc:>14.4f} {delta:>+8.4f} {pct:>+7.2f}%")
        else:
            print(f"{name:<15} {ckpt_type:<20} {final_bpc:>10.4f} {'(eval failed)':<25}")
    else:
        print(f"{name:<15} {ckpt_type:<20} {final_bpc:>10.4f} {'(not yet run)':<25}")
PYEOF
