"""
Remove Sheng records from validation_shared.jsonl and test_shared.jsonl.
Sheng records are identified by source_id in {'swahili_sheng_v1', 'sheng_external_v1'}.

Preserves originals as *_v2.jsonl (backup); writes cleaned versions as *_v3.jsonl.
"""
import json
import shutil
from pathlib import Path
from collections import Counter

VAL_SHARED = Path("/workspace/data/corpus/splits_v2/validation_shared.jsonl")
TEST_SHARED = Path("/workspace/data/corpus/splits_v2/test_shared.jsonl")

VAL_BACKUP = Path("/workspace/data/corpus/splits_v2/validation_shared_v2.jsonl")
TEST_BACKUP = Path("/workspace/data/corpus/splits_v2/test_shared_v2.jsonl")

VAL_CLEAN = Path("/workspace/data/corpus/splits_v2/validation_shared_v3.jsonl")
TEST_CLEAN = Path("/workspace/data/corpus/splits_v2/test_shared_v3.jsonl")

SHENG_SOURCE_IDS = {"swahili_sheng_v1", "sheng_external_v1"}


def filter_file(src, backup, clean, label):
    # Backup once (idempotent)
    if not backup.exists():
        shutil.copy(src, backup)
        print(f"  backed up {src.name} → {backup.name}")
    else:
        print(f"  backup already exists: {backup.name}")

    # Filter and track what got stripped
    kept = 0
    stripped_records = []
    src_counts_before = Counter()
    src_counts_after = Counter()

    with open(src) as fin, open(clean, "w") as fout:
        for line in fin:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            sid = r.get("source_id", "MISSING")
            src_counts_before[sid] += 1
            if sid in SHENG_SOURCE_IDS:
                stripped_records.append(r)
                continue
            fout.write(line)
            src_counts_after[sid] += 1
            kept += 1

    print(f"\n  {label}:")
    print(f"    kept:     {kept:,}")
    print(f"    stripped: {len(stripped_records)}")
    if stripped_records:
        # Show sample of what was stripped
        print(f"    sample stripped doc_ids: {[r.get('doc_id', '?')[:12] for r in stripped_records[:3]]}")
    return kept, len(stripped_records)


print("=== Stripping Sheng records from existing shared val/test ===\n")

print(">>> validation_shared.jsonl")
val_kept, val_stripped = filter_file(VAL_SHARED, VAL_BACKUP, VAL_CLEAN, "validation")

print("\n>>> test_shared.jsonl")
test_kept, test_stripped = filter_file(TEST_SHARED, TEST_BACKUP, TEST_CLEAN, "test")

print(f"\n=== SUMMARY ===")
print(f"Validation: {val_kept + val_stripped:,} → {val_kept:,} (stripped {val_stripped} Sheng)")
print(f"Test:       {test_kept + test_stripped:,} → {test_kept:,} (stripped {test_stripped} Sheng)")

# Verification: cleaned files should have 0 Sheng records
print(f"\n=== VERIFICATION (must be 0) ===")
import subprocess
for path in [VAL_CLEAN, TEST_CLEAN]:
    n_sheng = 0
    with open(path) as f:
        for line in f:
            try:
                r = json.loads(line)
                if r.get("source_id") in SHENG_SOURCE_IDS:
                    n_sheng += 1
            except json.JSONDecodeError:
                pass
    print(f"  {path.name}: {n_sheng} Sheng records remaining")
    assert n_sheng == 0, f"Expected 0 Sheng records in {path.name}"

print("\n✓ Cleaned shared val/test ready for use as Swahili-authentic + other-languages baseline")
