#!/usr/bin/env python3
"""
LughaGen Post-Processing Deduplicator v2
Fixes over-aggressive near-exact dedup from v1.
- Exact dedup: MD5 of normalized text (unchanged)
- Near-exact: requires 95% character overlap (not prefix match)
  Only removes if texts are nearly identical throughout, not just at start
"""
import json, hashlib, collections
from pathlib import Path

BASE  = "/workspace/synthetic/tier1"
LANGS = ["kamba", "kikuyu", "luo", "swahili"]

def normalized(text):
    return ' '.join(text.lower().split())

def is_near_dup(text_a, text_b, threshold=0.95):
    """True only if texts share >95% of characters — catches case/punct variants."""
    if text_a == text_b:
        return True
    # Only flag if texts are very similar length
    len_a, len_b = len(text_a), len(text_b)
    if min(len_a, len_b) / max(len_a, len_b) < 0.85:
        return False  # too different in length — not a near-dup
    # Check character-level similarity using longest common subsequence ratio
    # Approximation: count matching chars in order
    shorter, longer = (text_a, text_b) if len_a <= len_b else (text_b, text_a)
    matches = 0
    j = 0
    for c in shorter:
        while j < len(longer) and longer[j] != c:
            j += 1
        if j < len(longer):
            matches += 1
            j += 1
    similarity = matches / max(len(shorter), 1)
    return similarity >= threshold

print("=" * 60)
print("  LUGHAGEN POST-PROCESS DEDUPLICATOR v2")
print("  (Fixed near-exact threshold: 95% similarity)")
print("=" * 60)

grand_total = grand_kept = 0

for lang in LANGS:
    # Read from backup (original before v1 dedup)
    backup = Path(BASE) / f"{lang}_tier1_backup.jsonl"
    output = Path(BASE) / f"{lang}_tier1.jsonl"
    
    if not backup.exists():
        print(f"⚠️  {lang}: backup not found — using current file")
        backup = output
    
    seen_exact = set()
    seen_texts = []   # for near-dup comparison (bounded)
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
                if not text: continue

                norm  = normalized(text)
                exact = hashlib.md5(norm.encode()).hexdigest()

                # Layer 1: Exact dedup
                if exact in seen_exact:
                    stats['exact_dup'] += 1
                    continue
                seen_exact.add(exact)

                # Layer 2: Near-exact dedup (95% similarity)
                # Only check last 500 texts for efficiency
                is_dup = False
                for prev in seen_texts[-500:]:
                    if is_near_dup(norm, prev):
                        is_dup = True
                        break
                if is_dup:
                    stats['near_dup'] += 1
                    continue

                seen_texts.append(norm)
                kept.append(line)
                stats['kept'] += 1

            except:
                stats['error'] += 1

    # Write output
    with open(output, 'w') as f:
        f.write('\n'.join(kept) + '\n')

    total   = stats['total']
    kept_n  = stats['kept']
    removed = total - kept_n
    grand_total += total
    grand_kept  += kept_n

    print(f"\n{lang.upper()}: {total:,} → {kept_n:,} "
          f"({removed:,} removed, {removed/max(total,1)*100:.1f}%)")
    print(f"  exact duplicates:      {stats['exact_dup']:>6,}")
    print(f"  near-exact (95% sim):  {stats['near_dup']:>6,}")
    gap = max(0, 100_000 - kept_n)
    print(f"  {'✅ Target met' if gap==0 else f'⚠️  {gap:,} below 100K target'}")

print(f"\n{'=' * 60}")
print(f"  TOTAL: {grand_total:,} → {grand_kept:,} "
      f"({grand_total - grand_kept:,} removed)")
print(f"{'=' * 60}")
print("\nFinal counts:")
for lang in LANGS:
    p = Path(BASE) / f"{lang}_tier1.jsonl"
    if p.exists():
        count = sum(1 for _ in open(p))
        print(f"  {lang:<10} {count:,}")
