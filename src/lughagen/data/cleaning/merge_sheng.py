"""
Merge existing sheng_synthetic_v1.jsonl with new drive dataset.
Deduplicates on text-hash. Normalizes new records to canonical schema.
"""
import json, hashlib, re, unicodedata
from pathlib import Path
from datetime import datetime, timezone

EXISTING = Path("/workspace/data/corpus/staged/synth_cs/sheng_synthetic_v1.jsonl")
NEW = Path("/workspace/data/corpus/staged/synth_cs/new_drive/sheng_external_raw.jsonl")
OUT = Path("/workspace/data/corpus/staged/synth_cs/sheng_combined_v2.jsonl")


def normalize_text(t):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", t)).strip()


def hash_text(t):
    return hashlib.sha1(normalize_text(t).encode("utf-8")).hexdigest()


def get_hash(r):
    doc_id = r.get("doc_id", "")
    if isinstance(doc_id, str) and len(doc_id) == 40 and all(c in "0123456789abcdef" for c in doc_id):
        return doc_id
    return hash_text(r.get("text", ""))


def normalize_record(r, source_id_override=None):
    txt = r.get("text", "")
    return {
        "doc_id": get_hash(r),
        "text": txt,
        "language": r.get("language", "swahili"),
        "register": r.get("register", "code_switched"),
        "origin": r.get("origin", "synthetic"),
        "source_id": source_id_override or r.get("source_id", "unknown"),
        "source_url": r.get("source_url", ""),
        "license": r.get("license", "unknown_internal_use_only"),
        "domain": r.get("domain", "unknown"),
        "register_detail": r.get("register_detail", "sheng"),
        "code_switch_pair": r.get("code_switch_pair", "swahili-english"),
        "code_switch_ratio": r.get("code_switch_ratio"),
        "tokens_llama": r.get("tokens_llama"),
        "words": r.get("words", len(txt.split())),
        "quality_score": r.get("quality_score"),
        "fluency_score": r.get("fluency_score"),
        "preprocessing": r.get("preprocessing", ["unicode_nfc", "whitespace_normalise"]),
        "augmentation": r.get("augmentation", []),
        "split": "unassigned",
        "ingested_at": r.get("ingested_at", datetime.now(timezone.utc).isoformat()),
    }


seen = set()
n_existing = n_new_unique = n_new_dup = 0

OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, "w") as fout:
    print("Writing existing pool (preserving source_id='swahili_sheng_v1')...", flush=True)
    with open(EXISTING) as fin:
        for line in fin:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            h = get_hash(r)
            if h in seen:
                continue
            seen.add(h)
            fout.write(json.dumps(normalize_record(r), ensure_ascii=False) + "\n")
            n_existing += 1
    print(f"  wrote {n_existing:,} records from existing pool")
    
    print("Adding novel records from new dataset (source_id='sheng_external_v1')...", flush=True)
    with open(NEW) as fin:
        for line in fin:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            h = get_hash(r)
            if h in seen:
                n_new_dup += 1
                continue
            seen.add(h)
            fout.write(json.dumps(normalize_record(r, source_id_override="sheng_external_v1"), ensure_ascii=False) + "\n")
            n_new_unique += 1

print(f"  added {n_new_unique:,} novel records")
print(f"  skipped {n_new_dup:,} duplicates")
print()
print(f"Total combined: {n_existing + n_new_unique:,} unique records")
print(f"Output: {OUT}")
