#!/bin/bash
# Aug 11 Step 5: Bootstrap 95% CIs on all 5 LughaGen models
set -e

echo "=== Aug 11 Step 5: Bootstrap CIs ==="
echo ""

# Assumes eval_per_language_5tracks.py has saved per-record losses in
# {output}_per_record_losses.jsonl format
for MODEL in "P3_LLAMA_CPT_S137" "P3_LLAMA_FFT_S137" "P3_LLAMA_QLORA_S137" \
             "P3.5_GEMMA_CPT_S137" "P3.5_GEMMA_FFT_S137"; do
    LOSSES="/workspace/results/${MODEL}_per_language_eval_CLEAN_per_record_losses.jsonl"
    OUTPUT="/workspace/results/bootstrap_ci_${MODEL}.json"
    
    if [ ! -f "$LOSSES" ]; then
        echo "$MODEL: per-record file missing at $LOSSES"
        echo "  (May need to re-run eval with per-record output flag)"
        continue
    fi
    
    echo "Running bootstrap for $MODEL..."
    python /workspace/scripts/compute_bootstrap_ci.py \
        --losses-jsonl "$LOSSES" \
        --n-bootstrap 1000 \
        --output "$OUTPUT" 2>&1 | tail -20
    echo ""
done

echo ""
echo "=== Bootstrap CI summary ==="
python << 'PYEOF'
import json
from pathlib import Path

RESULTS = Path("/workspace/results/")
models = ["P3_LLAMA_CPT_S137", "P3_LLAMA_FFT_S137", "P3_LLAMA_QLORA_S137",
          "P3.5_GEMMA_CPT_S137", "P3.5_GEMMA_FFT_S137"]

print(f"{'Model':<25} {'Pooled BPC':>12} {'95% CI Low':>12} {'95% CI High':>12} {'CI Width':>12}")
print("-" * 80)
for m in models:
    ci_file = RESULTS / f"bootstrap_ci_{m}.json"
    if ci_file.exists():
        with ci_file.open() as f:
            d = json.load(f)
        p = d.get("pooled", {})
        print(f"{m:<25} {p.get('bpc', '?'):>12.4f} {p.get('ci_95_low', '?'):>12.4f} "
              f"{p.get('ci_95_high', '?'):>12.4f} {p.get('ci_width', '?'):>12.4f}")
    else:
        print(f"{m:<25} (bootstrap not run)")
PYEOF
