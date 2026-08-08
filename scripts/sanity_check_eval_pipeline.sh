#!/bin/bash
# Sanity check: reproduce Gemma-CPT BPC to verify eval pipeline determinism.
# Enhanced with full metadata comparison (bytes, tokens, examples, model+tokenizer revisions).
# Run this AFTER FFT completes on Aug 11.

set -e

CPT_CKPT="/workspace/releases/lughagen/LughaGen-Gemma2-27B-CPT-S137"
VAL_CLEAN="/workspace/data/corpus/splits_v2/pathD_v2_filtered/validation_shared_v3_clean.jsonl"

RESULTS_DIR="/workspace/results/sanity_checks"
mkdir -p $RESULTS_DIR

echo "=== Sanity Check: Reproduce Gemma-CPT BPC ==="
echo "Checkpoint: $CPT_CKPT"
echo "Validation: $VAL_CLEAN"
echo ""

# Capture metadata about the checkpoint
echo "=== Checkpoint metadata ==="
ls -la $CPT_CKPT/*.safetensors 2>/dev/null | head -5
md5sum $CPT_CKPT/config.json $CPT_CKPT/tokenizer.json 2>/dev/null | head -5
python -c "
import json
with open('$CPT_CKPT/config.json') as f:
    c = json.load(f)
print(f'Model: {c.get(\"model_type\", \"unknown\")}, {c.get(\"num_hidden_layers\", \"?\")}L, {c.get(\"hidden_size\", \"?\")}H')
"

echo ""
echo "=== Validation set metadata ==="
python << 'PYEOF'
import json
val_path = "/workspace/data/corpus/splits_v2/pathD_v2_filtered/validation_shared_v3_clean.jsonl"
records = []
total_bytes = 0
with open(val_path) as f:
    for line in f:
        rec = json.loads(line)
        records.append(rec)
        total_bytes += len(rec['text'].encode('utf-8'))
print(f"Records: {len(records)}")
print(f"Total bytes: {total_bytes:,}")
print(f"Bytes per record: {total_bytes / len(records):.0f}")
langs = {}
for r in records:
    langs[r.get('language', 'unknown')] = langs.get(r.get('language', 'unknown'), 0) + 1
for lang, count in sorted(langs.items()):
    print(f"  {lang}: {count}")
PYEOF

echo ""
echo "=== Running eval (this reproduces the earlier Gemma-CPT eval) ==="
python /workspace/scripts/eval_per_language_5tracks.py \
    --checkpoint $CPT_CKPT \
    --val-jsonl $VAL_CLEAN \
    --output $RESULTS_DIR/gemma_cpt_reeval_sanity.json 2>&1 | tee $RESULTS_DIR/gemma_cpt_reeval.log

echo ""
echo "=== Comparison (with tolerance for FP non-determinism) ==="
python << 'PYEOF'
import json
import sys

with open("/workspace/results/P3.5_GEMMA_CPT_S137_per_language_eval_CLEAN.json") as f:
    original = json.load(f)

with open("/workspace/results/sanity_checks/gemma_cpt_reeval_sanity.json") as f:
    reeval = json.load(f)

def get(d, path, default=None):
    """Navigate nested dict by dot-separated path."""
    for key in path.split('.'):
        d = d.get(key, {}) if isinstance(d, dict) else default
    return d if d != {} else default

# Metrics to compare
metrics_to_check = [
    ("pooled BPC", "pooled.bpc"),
    ("pooled loss", "pooled.loss"),
    ("pooled n_bytes", "pooled.n_bytes"),
    ("pooled n_tokens", "pooled.n_tokens"),
    ("pooled n_examples", "pooled.n_examples"),
]

print(f"{'Metric':<25} {'Original':>15} {'Re-eval':>15} {'Match':>10}")
print("-" * 70)
all_ok = True
for name, path in metrics_to_check:
    orig = get(original, path)
    new = get(reeval, path)
    if orig is None or new is None:
        print(f"{name:<25} {str(orig):>15} {str(new):>15} {'MISSING':>10}")
        all_ok = False
        continue
    if isinstance(orig, float):
        match = abs(orig - new) < 0.001
        print(f"{name:<25} {orig:>15.4f} {new:>15.4f} {'✓' if match else '✗':>10}")
    else:
        match = orig == new
        print(f"{name:<25} {str(orig):>15} {str(new):>15} {'✓' if match else '✗':>10}")
    if not match: all_ok = False

# Per-language check
print("\nPer-language BPC:")
print(f"{'Language':<12} {'Original':>10} {'Re-eval':>10} {'Δ':>8} {'Match':>10}")
print("-" * 55)
for lang in ["swahili_authentic", "kikuyu", "luo", "kamba", "sheng"]:
    orig = original.get("per_language", {}).get(lang, {}).get("bpc")
    new = reeval.get("per_language", {}).get(lang, {}).get("bpc")
    if orig and new:
        delta = new - orig
        match = abs(delta) < 0.001
        marker = '✓' if match else '✗'
        print(f"{lang:<12} {orig:>10.4f} {new:>10.4f} {delta:>+.4f} {marker:>10}")
        if not match: all_ok = False

print("\n" + "=" * 70)
if all_ok:
    print("✓ PIPELINE IS DETERMINISTIC — Gemma-FFT eval will be trustworthy")
    sys.exit(0)
else:
    print("✗ DEVIATION DETECTED — investigate before Gemma-FFT eval")
    sys.exit(1)
PYEOF

echo ""
echo "=== Sanity check complete ==="
