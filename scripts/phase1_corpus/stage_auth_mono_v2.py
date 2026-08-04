"""
Stage 2: Metadata-wrap Set C clean files into auth_mono_v2/.

Inputs (Set C, AfroLID-audited by Variant B pipeline, deduped):
  /home/nvidia/clean_datasets/final_jsonl/{swahili,luo,kikuyu,kamba}_clean.jsonl

Outputs:
  /workspace/data/corpus/staged/auth_mono_v2/{swahili,luo,kikuyu,kamba}_authentic_clean_v1.jsonl

Each record receives full metadata:
  - text: (preserved)
  - language: lang
  - origin: "authentic"
  - register: "monolingual"
  - source_id: "auth_clean_v1_<lang>"
  - doc_id: "<lang>_cleanv1_000123"
  - afrolid_audited: true
  - afrolid_audit_pipeline: "variant_b_2026-05-06"
  - quality_score: null
  - license: "unknown"

Filter: drop records with text length < 10 chars (only filter applied).
"""
import sys
sys.stdout.reconfigure(line_buffering=True)

import json, time
from pathlib import Path

INPUT_DIR = Path("/home/nvidia/clean_datasets/final_jsonl")
OUTPUT_DIR = Path("/workspace/data/corpus/staged/auth_mono_v2")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MIN_TEXT_LEN = 10  # drop records with text shorter than this

LANGS = ["swahili", "luo", "kikuyu", "kamba"]

AFROLID_METADATA = {
    "afrolid_audited": True,
    "afrolid_audit_pipeline": "variant_b_2026-05-06",
    "afrolid_audit_kept_fraction": {
        "swahili": 10950047 / 11431352,  # 0.958
        "luo":      325370 /   948013,    # 0.343
        "kikuyu":    93846 /   181485,    # 0.517
        "kamba":     36891 /    85192,    # 0.433
    },
}

print(f"[{time.strftime('%H:%M:%S')}] Stage 2: metadata-wrap Set C into auth_mono_v2/")
print(f"  Input:  {INPUT_DIR}")
print(f"  Output: {OUTPUT_DIR}")
print(f"  Min text length: {MIN_TEXT_LEN}")
print()

results = {}
for lang in LANGS:
    src = INPUT_DIR / f"{lang}_clean.jsonl"
    dst = OUTPUT_DIR / f"{lang}_authentic_clean_v1.jsonl"
    
    if not src.exists():
        print(f"  {lang}: SOURCE MISSING at {src}")
        results[lang] = None
        continue
    
    print(f"  [{time.strftime('%H:%M:%S')}] {lang}: {src} -> {dst}")
    t0 = time.time()
    
    n_total = 0
    n_kept = 0
    n_filtered_short = 0
    n_filtered_empty = 0
    n_parse_errors = 0
    
    with open(src, "r") as fin, open(dst, "w") as fout:
        for line in fin:
            n_total += 1
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                n_parse_errors += 1
                continue
            
            text = rec.get("text", "").strip()
            
            if not text:
                n_filtered_empty += 1
                continue
            if len(text) < MIN_TEXT_LEN:
                n_filtered_short += 1
                continue
            
            wrapped = {
                "text": text,
                "language": lang,
                "origin": "authentic",
                "register": "monolingual",
                "source_id": f"auth_clean_v1_{lang}",
                "doc_id": f"{lang}_cleanv1_{n_kept:07d}",
                "license": "unknown",
                "quality_score": None,
                "domain": None,
                "source_url": None,
                **AFROLID_METADATA,
                "afrolid_audit_lang_kept_fraction": round(
                    AFROLID_METADATA["afrolid_audit_kept_fraction"].get(lang, 0), 4),
            }
            # Don't carry the nested dict
            wrapped.pop("afrolid_audit_kept_fraction", None)
            
            fout.write(json.dumps(wrapped, ensure_ascii=False) + "\n")
            n_kept += 1
            
            if n_kept % 500_000 == 0:
                rate = n_kept / max(0.01, time.time() - t0)
                print(f"      ... {n_kept:,} written ({rate:.0f}/s)")
    
    elapsed = time.time() - t0
    dst_size_mb = dst.stat().st_size / 1024 / 1024
    
    results[lang] = {
        "src": str(src),
        "dst": str(dst),
        "n_total_input": n_total,
        "n_kept": n_kept,
        "n_filtered_short": n_filtered_short,
        "n_filtered_empty": n_filtered_empty,
        "n_parse_errors": n_parse_errors,
        "kept_pct": round(n_kept / max(1, n_total) * 100, 2),
        "elapsed_sec": round(elapsed, 1),
        "output_size_mb": round(dst_size_mb, 1),
    }
    
    print(f"      [{lang}]  total={n_total:,}  kept={n_kept:,}  "
          f"short={n_filtered_short:,}  empty={n_filtered_empty:,}  "
          f"errors={n_parse_errors:,}  ({elapsed:.0f}s)")

# Summary
print()
print("=" * 78)
print("STAGE 2 SUMMARY")
print("=" * 78)
print()
print(f"  {'Lang':<10} {'Input':>12} {'Kept':>12} {'Short':>8} {'Empty':>8} {'Errors':>7} {'Kept%':>7} {'Size':>10}")
for lang in LANGS:
    r = results.get(lang)
    if r is None:
        print(f"  {lang:<10}  MISSING")
        continue
    print(f"  {lang:<10} {r['n_total_input']:>12,} {r['n_kept']:>12,} "
          f"{r['n_filtered_short']:>8,} {r['n_filtered_empty']:>8,} "
          f"{r['n_parse_errors']:>7,} {r['kept_pct']:>6.2f}% "
          f"{r['output_size_mb']:>8.1f}MB")

# Write metadata manifest
manifest_path = OUTPUT_DIR / "MANIFEST.json"
manifest = {
    "stage": "auth_mono_v2",
    "stage_purpose": "Authentic monolingual data, AfroLID-audited (Variant B), metadata-wrapped",
    "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "source_pipeline": "afrolid_variant_b_2026-05-06",
    "source_dir": str(INPUT_DIR),
    "filter": {"min_text_len_chars": MIN_TEXT_LEN},
    "metadata_fields_added": [
        "language", "origin", "register", "source_id", "doc_id",
        "license", "quality_score", "domain", "source_url",
        "afrolid_audited", "afrolid_audit_pipeline",
        "afrolid_audit_lang_kept_fraction",
    ],
    "files": {lang: results[lang] for lang in LANGS if results.get(lang)},
}
with open(manifest_path, "w") as f:
    json.dump(manifest, f, indent=2, ensure_ascii=False)
print()
print(f"  Manifest: {manifest_path}")
print()
print(f"[{time.strftime('%H:%M:%S')}] STAGE 2 DONE")
print()
print(f"To verify, run:")
print(f"  head -1 {OUTPUT_DIR}/swahili_authentic_clean_v1.jsonl | python3 -m json.tool")
