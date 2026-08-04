#!/usr/bin/env python3
"""
LughaGen Post-Processing Deduplicator
Removes exact and near-exact duplicates from clean output files.
Run after clean_synthetic_v4.py completes.
"""

import json, hashlib, collections
from pathlib import Path

BASE  = "/workspace/synthetic/tier1"
LANGS = ["kamba", "kikuyu", "luo", "swahili"]

print("=" * 60)
print("  LUGHAGEN POST-PROCESS DEDUPLICATOR")
print("=" * 60)

grand_total = grand_kept = 0

for lang in LANGS:
    path = Path(BASE) / f"{lang}_tier1.jsonl"
    if not path.exists():
        print(f"⚠️  {lang}: file not found — skipping")
        continue

    seen_exact  = set()   # MD5 of normalized text
    seen_prefix = set()   # First 80 chars normalized (catches near-dupes)
    kept = []
    stats = collections.Counter()

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            stats['total'] += 1
            try:
                d    = json.loads(line)
                text = d.get('text', '').strip()
                if not text: continue

                # Exact dedup
                norm  = ' '.join(text.lower().split())
                exact = hashlib.md5(norm.encode()).hexdigest()
                if exact in seen_exact:
                    stats['exact_dup'] += 1
                    continue
                seen_exact.add(exact)

                # Near-exact dedup (same first 80 normalized chars)
                prefix = norm[:80]
                if prefix in seen_prefix and len(norm) > 40:
                    stats['near_dup'] += 1
                    continue
                seen_prefix.add(prefix)

                kept.append(line)
                stats['kept'] += 1

            except:
                stats['error'] += 1

    # Write deduped file
    with open(path, 'w') as f:
        f.write('\n'.join(kept) + '\n')

    total   = stats['total']
    kept_n  = stats['kept']
    removed = total - kept_n
    grand_total += total
    grand_kept  += kept_n

    print(f"\n{lang.upper()}: {total:,} → {kept_n:,} "
          f"({removed:,} removed, {removed/max(total,1)*100:.1f}%)")
    print(f"  exact duplicates:     {stats['exact_dup']:>6,}")
    print(f"  near-exact duplicates:{stats['near_dup']:>6,}")
    gap = max(0, 100_000 - kept_n)
    if gap == 0:
        print(f"  ✅ Target met ({kept_n:,} ≥ 100,000)")
    else:
        print(f"  ⚠️  {gap:,} below 100K target")

print(f"\n{'=' * 60}")
print(f"  TOTAL: {grand_total:,} → {grand_kept:,} "
      f"({grand_total - grand_kept:,} removed)")
print(f"{'=' * 60}")
print("\nFinal clean counts:")
for lang in LANGS:
    path = Path(BASE) / f"{lang}_tier1.jsonl"
    if path.exists():
        count = sum(1 for _ in open(path))
        print(f"  {lang:<10} {count:,}")
