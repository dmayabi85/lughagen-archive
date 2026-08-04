"""
Audit the new Sheng dataset against:
  - Existing sheng_synthetic_v1.jsonl (182,426 records)
  - v2-D training file (1,380,523 records)
  - splits_v2 validation_shared and test_shared

Uses doc_id when present (SHA-1 of text), falls back to computed hash.
"""
import json
import hashlib
import re
import unicodedata
from pathlib import Path
from collections import Counter

NEW_PATH = Path("/workspace/data/corpus/staged/synth_cs/new_drive/sheng_external_raw.jsonl")
EXISTING_SHENG = Path("/workspace/data/corpus/staged/synth_cs/sheng_synthetic_v1.jsonl")
V2D_TRAIN = Path("/workspace/data/corpus/splits_v2/pathD/train.jsonl")
VAL_SHARED = Path("/workspace/data/corpus/splits_v2/validation_shared.jsonl")
TEST_SHARED = Path("/workspace/data/corpus/splits_v2/test_shared.jsonl")
REPORT_OUT = Path("/workspace/data/corpus/staged/synth_cs/new_drive/audit_report.json")


def normalize_text(t):
    t = unicodedata.normalize("NFC", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def hash_text(t):
    return hashlib.sha1(normalize_text(t).encode("utf-8")).hexdigest()


def get_hash(r):
    doc_id = r.get("doc_id", "")
    if isinstance(doc_id, str) and len(doc_id) == 40 and all(c in "0123456789abcdef" for c in doc_id):
        return doc_id
    txt = r.get("text", "")
    if not txt:
        return None
    return hash_text(txt)


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
    print(f"  {label}: {n:,} records, {len(hashes):,} unique hashes", flush=True)
    return hashes, n


print("=== Hashing existing pools ===")
print("Existing Sheng pool...")
existing_hashes, existing_n = collect_hashes(EXISTING_SHENG, "sheng_synthetic_v1")
print("v2-D training file (largest, ~30s)...")
v2d_hashes, v2d_n = collect_hashes(V2D_TRAIN, "v2-D train")
print("val/test shared...")
val_hashes, val_n = collect_hashes(VAL_SHARED, "val_shared")
test_hashes, test_n = collect_hashes(TEST_SHARED, "test_shared")

print()
print("=== Hashing new dataset ===")
new_hashes = set()
new_n = 0
schema_fields = Counter()
text_lengths = []
dup_existing = 0
dup_v2d = 0
dup_val = 0
dup_test = 0
sample_records = []

with open(NEW_PATH) as f:
    for line in f:
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        new_n += 1
        for k in r.keys():
            schema_fields[k] += 1
        h = get_hash(r)
        if not h:
            continue
        new_hashes.add(h)
        text_lengths.append(len(r.get("text", "")))
        if h in existing_hashes:
            dup_existing += 1
        if h in v2d_hashes:
            dup_v2d += 1
        if h in val_hashes:
            dup_val += 1
        if h in test_hashes:
            dup_test += 1
        if len(sample_records) < 3:
            sample_records.append(r)

all_used = existing_hashes | v2d_hashes | val_hashes | test_hashes
disjoint = len(new_hashes - all_used)

text_lengths.sort()
n_text = len(text_lengths)

report = {
    "new_dataset_path": str(NEW_PATH),
    "new_dataset_records": new_n,
    "new_dataset_unique_hashes": len(new_hashes),
    "schema_fields_present": dict(schema_fields),
    "text_length_stats": {
        "min": text_lengths[0] if n_text else None,
        "max": text_lengths[-1] if n_text else None,
        "median": text_lengths[n_text // 2] if n_text else None,
        "p10": text_lengths[n_text // 10] if n_text else None,
        "p90": text_lengths[n_text * 9 // 10] if n_text else None,
    },
    "overlap": {
        "with_existing_sheng_pool": dup_existing,
        "with_v2d_training": dup_v2d,
        "with_validation_shared": dup_val,
        "with_test_shared": dup_test,
        "disjoint_from_everything": disjoint,
    },
    "comparison_pool_sizes": {
        "existing_sheng_pool": existing_n,
        "v2d_training": v2d_n,
        "validation_shared": val_n,
        "test_shared": test_n,
    },
    "sample_records": sample_records,
}

REPORT_OUT.parent.mkdir(parents=True, exist_ok=True)
with open(REPORT_OUT, "w") as f:
    json.dump(report, f, indent=2, ensure_ascii=False, default=str)

print(f"\n=== AUDIT RESULTS ===")
print(f"New dataset: {new_n:,} records, {len(new_hashes):,} unique hashes")
print(f"Internal duplicates within new dataset: {new_n - len(new_hashes):,}")
print()
print("Overlap with:")
print(f"  Existing Sheng pool (182k): {dup_existing:,} ({dup_existing/new_n*100:.1f}%)")
print(f"  v2-D training (1.38M):      {dup_v2d:,} ({dup_v2d/new_n*100:.1f}%)")
print(f"  val_shared:                 {dup_val:,}")
print(f"  test_shared:                {dup_test:,}")
print()
print(f"=> Disjoint records (usable for clean held-out evaluation): {disjoint:,}")
print()
print(f"Full report: {REPORT_OUT}")
