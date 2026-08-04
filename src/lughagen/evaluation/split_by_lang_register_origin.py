#!/usr/bin/env python3
"""
Split filtered corpus by (language, register, origin).

Output naming:
  <language>_<register>_<origin>.jsonl
  e.g.  swahili_monolingual_authentic.jsonl
        swahili_code_switched_synthetic.jsonl  <-- your Sheng synthetic
        swahili_code_switched_authentic.jsonl  <-- your Sheng authentic
        kikuyu_code_switched_authentic.jsonl   <-- Kikuyu-English if it exists
"""
import json
from collections import Counter
from pathlib import Path

CORPUS = Path("/workspace/data/corpus/splits_v2/pathD_v2_filtered/train.jsonl")
OUT_DIR = CORPUS.parent / "by_lang_register_origin"
MANIFEST = OUT_DIR / "manifest.json"
OUT_DIR.mkdir(parents=True, exist_ok=True)

writers = {}
counts = Counter()
total = 0

print(f"[read] {CORPUS}")
with open(CORPUS) as f:
    for raw_line in f:
        total += 1
        r = json.loads(raw_line)

        lang = (r.get("language") or "unknown").lower().strip()
        reg = (r.get("register") or "unknown").lower().strip()
        origin = (r.get("origin") or "unknown").lower().strip()

        key = (lang, reg, origin)
        if key not in writers:
            fname = f"{lang}_{reg}_{origin}.jsonl"
            writers[key] = open(OUT_DIR / fname, "w")
        writers[key].write(raw_line)
        counts[key] += 1

        if total % 200_000 == 0:
            print(f"  {total:,} records...")

for w in writers.values():
    w.close()

manifest = {
    "source": str(CORPUS),
    "output_dir": str(OUT_DIR),
    "total_records": total,
    "files": {},
}
for (lang, reg, origin), n in sorted(counts.items(), key=lambda x: -x[1]):
    fname = f"{lang}_{reg}_{origin}.jsonl"
    manifest["files"][fname] = {
        "language": lang,
        "register": reg,
        "origin": origin,
        "records": n,
        "size_bytes": (OUT_DIR / fname).stat().st_size,
    }
MANIFEST.write_text(json.dumps(manifest, indent=2))
print(f"[wrote] {MANIFEST}")

print()
print("=" * 78)
print(f"{'File':<48s} {'Records':>12s} {'Size (MB)':>14s}")
print("=" * 78)
for fname, meta in sorted(manifest["files"].items(), key=lambda x: -x[1]["records"]):
    print(f"{fname:<48s} {meta['records']:>12,} {meta['size_bytes']/(1024*1024):>12.1f}")
print("=" * 78)
print(f"{'TOTAL':<48s} {total:>12,}")
