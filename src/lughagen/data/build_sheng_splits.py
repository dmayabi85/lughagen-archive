"""
Build clean Sheng val/test splits from sheng_combined_v2.jsonl.

Rule: a record is in train OR val OR test, never two.
- Hash every record in v2-D pathD/train.jsonl + existing val_shared + test_shared
- Filter combined Sheng pool to records whose hash is NOT in those used sets
- Sample 1,500 val + 1,500 test deterministically (seed=42)
- Remaining disjoint records go to sheng_pool_disjoint.jsonl (future use)
- Run sanity check: zero overlap between val/test/train
"""
import json
import hashlib
import re
import unicodedata
import random
from pathlib import Path

SEED = 42
VAL_SIZE = 1500
TEST_SIZE = 1500

SHENG_COMBINED = Path("/workspace/data/corpus/staged/synth_cs/sheng_combined_v2.jsonl")
V2D_TRAIN = Path("/workspace/data/corpus/splits_v2/pathD/train.jsonl")
VAL_SHARED = Path("/workspace/data/corpus/splits_v2/validation_shared.jsonl")
TEST_SHARED = Path("/workspace/data/corpus/splits_v2/test_shared.jsonl")

SHENG_VAL_OUT = Path("/workspace/data/corpus/splits_v2/sheng_val.jsonl")
SHENG_TEST_OUT = Path("/workspace/data/corpus/splits_v2/sheng_test.jsonl")
SHENG_POOL_REMAINING = Path("/workspace/data/corpus/splits_v2/sheng_pool_disjoint.jsonl")


def normalize_text(t):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", t)).strip()


def hash_text(t):
    return hashlib.sha1(normalize_text(t).encode("utf-8")).hexdigest()


def get_hash(r):
    doc_id = r.get("doc_id", "")
    if isinstance(doc_id, str) and len(doc_id) == 40 and all(c in "0123456789abcdef" for c in doc_id):
        return doc_id
    return hash_text(r.get("text", ""))


def collect_hashes(path, label):
    hashes = set()
    n = 0
    with open(path) as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            n += 1
            h = get_hash(r)
            if h:
                hashes.add(h)
    print(f"  {label}: {n:,} records → {len(hashes):,} unique hashes", flush=True)
    return hashes


print("=== Building used-records hash set (train ∪ val ∪ test) ===")
train_hashes = collect_hashes(V2D_TRAIN, "v2-D train")
val_hashes = collect_hashes(VAL_SHARED, "val_shared")
test_hashes = collect_hashes(TEST_SHARED, "test_shared")
used_hashes = train_hashes | val_hashes | test_hashes
print(f"Total 'used' set: {len(used_hashes):,} unique hashes")
print()

print("=== Filtering combined Sheng pool for disjoint records ===")
disjoint_records = []
total_seen = 0
with open(SHENG_COMBINED) as f:
    for line in f:
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        total_seen += 1
        h = get_hash(r)
        if h and h not in used_hashes:
            disjoint_records.append(r)

print(f"  scanned {total_seen:,} combined-pool records")
print(f"  disjoint from train/val/test: {len(disjoint_records):,}")
print()

# Deterministic shuffle
random.Random(SEED).shuffle(disjoint_records)

needed = VAL_SIZE + TEST_SIZE
if len(disjoint_records) < needed:
    print(f"  WARNING: only {len(disjoint_records)} disjoint records available; halving split sizes")
    half = len(disjoint_records) // 2
    val_size = test_size = half
else:
    val_size = VAL_SIZE
    test_size = TEST_SIZE

val_records = disjoint_records[:val_size]
test_records = disjoint_records[val_size:val_size + test_size]
remaining = disjoint_records[val_size + test_size:]

for r in val_records: r["split"] = "val"
for r in test_records: r["split"] = "test"
for r in remaining: r["split"] = "pool_disjoint"


def write_jsonl(records, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


write_jsonl(val_records, SHENG_VAL_OUT)
write_jsonl(test_records, SHENG_TEST_OUT)
write_jsonl(remaining, SHENG_POOL_REMAINING)

# Sanity check
val_hash_set = {get_hash(r) for r in val_records}
test_hash_set = {get_hash(r) for r in test_records}
overlap_val_train = len(val_hash_set & train_hashes)
overlap_test_train = len(test_hash_set & train_hashes)
overlap_val_test = len(val_hash_set & test_hash_set)
overlap_val_existing = len(val_hash_set & val_hashes)
overlap_test_existing = len(test_hash_set & test_hashes)

print("=== SANITY CHECK ===")
print(f"Val ∩ v2-D train:        {overlap_val_train}  (must be 0)")
print(f"Test ∩ v2-D train:       {overlap_test_train}  (must be 0)")
print(f"Val ∩ Test:              {overlap_val_test}  (must be 0)")
print(f"Val ∩ existing val:      {overlap_val_existing}  (must be 0)")
print(f"Test ∩ existing test:    {overlap_test_existing}  (must be 0)")
assert overlap_val_train == 0
assert overlap_test_train == 0
assert overlap_val_test == 0
assert overlap_val_existing == 0
assert overlap_test_existing == 0
print("✓ All disjointness assertions hold")
print()

# Source provenance breakdown of the splits
from collections import Counter
val_src = Counter(r["source_id"] for r in val_records)
test_src = Counter(r["source_id"] for r in test_records)
print("Val split source breakdown:")
for k, v in val_src.items(): print(f"  {k}: {v}")
print("Test split source breakdown:")
for k, v in test_src.items(): print(f"  {k}: {v}")
print()
print(f"sheng_val:               {val_size:>6} records → {SHENG_VAL_OUT}")
print(f"sheng_test:              {test_size:>6} records → {SHENG_TEST_OUT}")
print(f"sheng_pool_disjoint:     {len(remaining):>6} records → {SHENG_POOL_REMAINING}")
