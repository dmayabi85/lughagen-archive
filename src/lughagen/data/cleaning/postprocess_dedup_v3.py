#!/usr/bin/env python3
"""
LughaGen Post-Processing Deduplicator v3
Exact dedup only — removes only truly identical texts.
Near-exact dedup abandoned: legitimate paraphrases are naturally similar.
"""
import json, hashlib, collections
from pathlib import Path

BASE  = "/workspace/synthetic/tier1"
LANGS = ["kamba", "kikuyu", "luo", "swahili"]

def normalized(text):
    return ' '.join(text.lower().split())

print("=" * 60)
print("  LUGHAGEN POST-PROCESS DEDUPLICATOR v3")
print("  (Exact dedup only — no near-exact)")
print("=" * 60)

grand_total = grand_kept = 0

for lang in LANGS:
    # Always read from backup to get full original data
    backup = Path(BASE) / f"{lang}_tier1_backup.jsonl"
    output = Path(BASE) / f"{lang}_tier1.jsonl"

    if not backup.exists():
        print(f"⚠️  {lang}: backup not found — using current file")
        backup = output

    seen_exact = set()
    kept = []
    stats = collections.Counter()

    with open(backup) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            stats['total'] += 1
            try:
                d    = json.loads(line)
                text = d.get('text', '').strip()
                if not text:
                    stats['empty'] += 1
                    continue

                # Normalize: lowercase + collapse whitespace
                norm  = normalized(text)
                exact = hashlib.md5(norm.encode()).hexdigest()

                if exact in seen_exact:
                    stats['exact_dup'] += 1
                    continue

                seen_exact.add(exact)
                kept.append(line)
                stats['kept'] += 1

            except:
                stats['error'] += 1

    # Write deduplicated output
    with open(output, 'w') as f:
        f.write('\n'.join(kept) + '\n')

    total   = stats['total']
    kept_n  = stats['kept']
    removed = total - kept_n
    grand_total += total
    grand_kept  += kept_n

    print(f"\n{lang.upper()}: {total:,} → {kept_n:,} "
          f"({removed:,} removed, {removed/max(total,1)*100:.1f}%)")
    print(f"  exact duplicates: {stats['exact_dup']:>6,}")
    print(f"  empty/errors:     {stats['empty']+stats['error']:>6,}")
    gap = max(0, 100_000 - kept_n)
    if gap == 0:
        print(f"  ✅ Target met ({kept_n:,} ≥ 100,000)")
    else:
        print(f"  ⚠️  {gap:,} below 100K target — needs more generation")

print(f"\n{'=' * 60}")
print(f"  TOTAL: {grand_total:,} → {grand_kept:,} "
      f"({grand_total - grand_kept:,} exact duplicates removed)")
print(f"{'=' * 60}")
print("\nFinal counts:")
for lang in LANGS:
    p = Path(BASE) / f"{lang}_tier1.jsonl"
    if p.exists():
        count = sum(1 for _ in open(p))
        print(f"  {lang:<10} {count:,}")
