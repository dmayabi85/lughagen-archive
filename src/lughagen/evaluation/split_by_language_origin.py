#!/usr/bin/env python3
"""
Split the filtered corpus into per-(language × origin) JSONL files.

Reads: /workspace/data/corpus/splits_v2/pathD_v2_filtered/train.jsonl
Writes: /workspace/data/corpus/splits_v2/pathD_v2_filtered/by_language_origin/
  swahili_authentic.jsonl
  swahili_synthetic.jsonl
  luo_authentic.jsonl
  luo_synthetic.jsonl
  kikuyu_authentic.jsonl
  kikuyu_synthetic.jsonl
  kamba_authentic.jsonl
  kamba_synthetic.jsonl
  manifest.json    (record counts, totals, sanity checks)

Each output line is the EXACT original JSONL record — no modification.
"""
from __future__ import annotations
import json
from collections import Counter, defaultdict
from pathlib import Path

CORPUS = Path("/workspace/data/corpus/splits_v2/pathD_v2_filtered/train.jsonl")
OUT_DIR = CORPUS.parent / "by_language_origin"
MANIFEST = OUT_DIR / "manifest.json"

OUT_DIR.mkdir(parents=True, exist_ok=True)

# Open one writer per (language, origin) pair — on demand
writers: dict[tuple[str, str], any] = {}
counts: Counter = Counter()
total_read = 0
total_written = 0
unknown_records: list[dict] = []  # first few odd records for inspection

print(f"[read] {CORPUS}")
try:
    with open(CORPUS) as f:
        for raw_line in f:
            total_read += 1
            try:
                r = json.loads(raw_line)
            except json.JSONDecodeError:
                counts[("__bad_json__", "__bad_json__")] += 1
                continue

            lang = (r.get("language") or "unknown").lower().strip()
            origin = (r.get("origin") or "unknown").lower().strip()

            # Sanity: track odd values (not synthetic/authentic)
            if origin not in ("authentic", "synthetic"):
                counts[(lang, origin)] += 1
                if len(unknown_records) < 5:
                    unknown_records.append({"doc_id": r.get("doc_id"),
                                            "language": lang,
                                            "origin": origin})
                # Still write to that bucket (transparency)
            counts[(lang, origin)] += 0  # no-op — count applied below unconditionally

            key = (lang, origin)
            if key not in writers:
                fname = f"{lang}_{origin}.jsonl"
                writers[key] = open(OUT_DIR / fname, "w")
            writers[key].write(raw_line)
            counts[key] += 1
            total_written += 1

            if total_read % 200_000 == 0:
                print(f"  {total_read:,} records processed...")
finally:
    for w in writers.values():
        w.close()

print(f"[done] read {total_read:,} records, wrote {total_written:,} to {len(writers)} files")

# ---------- Build manifest ----------
manifest = {
    "source": str(CORPUS),
    "output_dir": str(OUT_DIR),
    "total_records_read": total_read,
    "total_records_written": total_written,
    "files": {},
    "unknown_or_odd_origin_records_sample": unknown_records,
}

for (lang, origin), n in sorted(counts.items(), key=lambda x: (-x[1], x[0])):
    fname = f"{lang}_{origin}.jsonl"
    fpath = OUT_DIR / fname
    if fpath.exists():
        manifest["files"][fname] = {
            "language": lang,
            "origin": origin,
            "records": n,
            "size_bytes": fpath.stat().st_size,
        }

MANIFEST.write_text(json.dumps(manifest, indent=2))
print(f"[wrote] {MANIFEST}")

# ---------- Console summary ----------
print()
print("=" * 70)
print(f"{'File':<40s} {'Records':>12s} {'Size':>12s}")
print("=" * 70)
total = 0
for fname, meta in sorted(manifest["files"].items(),
                          key=lambda x: -x[1]["records"]):
    size_mb = meta["size_bytes"] / (1024 * 1024)
    print(f"{fname:<40s} {meta['records']:>12,} {size_mb:>10.1f} MB")
    total += meta["records"]
print("=" * 70)
print(f"{'TOTAL':<40s} {total:>12,}")

# Sanity: total_written should equal total_read
if total_written != total_read:
    print(f"\nWARN: {total_read - total_written} records unaccounted for. Check manifest.")
else:
    print("\n[OK] All records written. Splits sum to source total.")
