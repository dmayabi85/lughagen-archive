#!/bin/bash
# Aug 12 Step 1: Verify FLORES+ dataset loads
set -e

echo "=== FLORES+ Setup ==="

pip install --break-system-packages -q datasets sacrebleu

python << 'PYEOF'
from datasets import load_dataset

langs = ['eng_Latn', 'swh_Latn', 'kik_Latn', 'luo_Latn', 'kam_Latn']
for lang in langs:
    try:
        ds_dev = load_dataset('openlanguagedata/flores_plus', lang,
                              split='dev', trust_remote_code=True)
        ds_devtest = load_dataset('openlanguagedata/flores_plus', lang,
                                  split='devtest', trust_remote_code=True)
        print(f"  ✓ {lang}: {len(ds_dev)} dev, {len(ds_devtest)} devtest")
        print(f"    First devtest: {ds_devtest[0]}")
    except Exception as e:
        print(f"  ✗ {lang}: {e}")
PYEOF

echo ""
echo "If all 5 languages load, proceed to Aug 12 Step 2"
