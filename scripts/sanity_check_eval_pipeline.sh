#!/bin/bash
# Sanity check: reproduce Gemma-CPT BPC to verify eval pipeline determinism
# Run this AFTER FFT completes on Aug 11.

set -e

CPT_CKPT="/workspace/releases/lughagen/LughaGen-Gemma2-27B-CPT-S137"
VAL_CLEAN="/workspace/data/corpus/splits_v2/pathD_v2_filtered/validation_shared_v3_clean.jsonl"
EXPECTED_BPC="1.785"

RESULTS_DIR="/workspace/results/sanity_checks"
mkdir -p $RESULTS_DIR

echo "=== Sanity Check: Reproduce Gemma-CPT BPC ==="
echo "Expected pooled BPC: $EXPECTED_BPC (from earlier eval)"
echo "Checkpoint: $CPT_CKPT"
echo ""

# Re-run per-language eval
python /workspace/scripts/eval_per_language_5tracks.py \
    --checkpoint $CPT_CKPT \
    --val-jsonl $VAL_CLEAN \
    --output $RESULTS_DIR/gemma_cpt_reeval_sanity.json 2>&1 | tee $RESULTS_DIR/gemma_cpt_reeval.log

echo ""
echo "=== Comparison ==="
python << 'PYEOF'
import json

with open("/workspace/results/P3.5_GEMMA_CPT_S137_per_language_eval_CLEAN.json") as f:
    original = json.load(f)

with open("/workspace/results/sanity_checks/gemma_cpt_reeval_sanity.json") as f:
    reeval = json.load(f)

print(f"{'Language':<12} {'Original':>10} {'Re-eval':>10} {'Δ':>8}")
print("-" * 45)
for lang in ["swahili_authentic", "kikuyu", "luo", "kamba", "sheng"]:
    if lang in original.get("per_language", {}) and lang in reeval.get("per_language", {}):
        orig = original["per_language"][lang]["bpc"]
        new = reeval["per_language"][lang]["bpc"]
        delta = new - orig
        print(f"{lang:<12} {orig:>10.4f} {new:>10.4f} {delta:>+.4f}")

orig_pooled = original.get("pooled", {}).get("bpc", None)
new_pooled = reeval.get("pooled", {}).get("bpc", None)
if orig_pooled and new_pooled:
    print("-" * 45)
    delta = new_pooled - orig_pooled
    print(f"{'POOLED':<12} {orig_pooled:>10.4f} {new_pooled:>10.4f} {delta:>+.4f}")
    
    if abs(delta) < 0.001:
        print("\n✓ PIPELINE IS DETERMINISTIC")
    elif abs(delta) < 0.01:
        print("\n○ Small variance (possibly floating-point non-determinism)")
    else:
        print("\n✗ SIGNIFICANT DEVIATION — investigate before Gemma-FFT eval")
PYEOF

echo ""
echo "Sanity check complete. If pipeline is deterministic, proceed with Gemma-FFT eval."
