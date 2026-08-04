"""
detect_paraphrastic_overlap.py
─────────────────────────────────────────────────────────────────────────
Detect paraphrastic overlap between raw seed files and synthetic outputs
using character-shingle MinHash signatures (locality-sensitive hashing).

The MinHash approach catches near-duplicates that exact-hash deduplication
misses. Two strings sharing 80%+ of their character-3-shingles are flagged
as paraphrastic variants of each other.

For every record in the raw file, this script asks: does any record in
the synthetic output share enough shingles to suggest it's a paraphrase?

Output: a JSON file marking each raw doc_id as either "unique" (safe to
ingest as authentic), "near_synthetic" (overlap detected, ingestion would
double-count), or "exact_duplicate" (identical text already in synthetic).

Usage:
    pip install datasketch --break-system-packages
    python detect_paraphrastic_overlap.py \\
        --raw      /workspace/data/cleaned_final/swahili_clean.jsonl \\
        --synthetic /workspace/synthetic/tier3/fluency_scored.jsonl \\
        --language swahili \\
        --output   /workspace/data/corpus/audit/overlap_swahili.json \\
        --threshold 0.5
"""

import argparse
import hashlib
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

from datasketch import MinHash, MinHashLSH


def normalise(text):
    t = unicodedata.normalize("NFC", text or "")
    t = re.sub(r"\s+", " ", t).strip().lower()
    return t


def shingles(text, k=3):
    """Character k-shingles of normalised text."""
    text = normalise(text)
    if len(text) < k:
        return set()
    return {text[i:i+k] for i in range(len(text) - k + 1)}


def minhash_of(text, num_perm=128, k=3):
    m = MinHash(num_perm=num_perm)
    for s in shingles(text, k=k):
        m.update(s.encode("utf-8"))
    return m


def doc_id(text):
    return hashlib.sha1(normalise(text).encode("utf-8")).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw",       required=True, help="Raw/cleaned seed JSONL to test")
    ap.add_argument("--synthetic", required=True, help="Synthetic output JSONL to test against")
    ap.add_argument("--language",  required=True, help="Filter both files to this language")
    ap.add_argument("--output",    required=True, help="Where to write overlap.json")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="Jaccard threshold; pairs above this are flagged (default 0.5)")
    ap.add_argument("--num-perm",  type=int, default=128)
    ap.add_argument("--shingle-k", type=int, default=3)
    ap.add_argument("--lang-field",default="lang")
    ap.add_argument("--text-field",default="text")
    ap.add_argument("--max-synth", type=int, default=200000,
                    help="Cap on synthetic records to index (memory/speed bound)")
    args = ap.parse_args()

    # ── Build LSH index over synthetic records ──────────────────────
    lsh = MinHashLSH(threshold=args.threshold, num_perm=args.num_perm)
    synth_index = {}    # short_id -> doc_id
    exact_set = set()

    print(f"Indexing synthetic records from {args.synthetic} (max {args.max_synth:,}) ...")
    n_indexed = 0
    with open(args.synthetic) as f:
        for line in f:
            if n_indexed >= args.max_synth:
                break
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if obj.get(args.lang_field) != args.language:
                continue
            text = obj.get(args.text_field)
            if not text or len(text) < 20:
                continue
            did = doc_id(text)
            exact_set.add(did)
            mh = minhash_of(text, num_perm=args.num_perm, k=args.shingle_k)
            short = f"s_{n_indexed}"
            lsh.insert(short, mh)
            synth_index[short] = did
            n_indexed += 1
            if n_indexed % 20000 == 0:
                print(f"  indexed {n_indexed:,}")

    print(f"  indexed {n_indexed:,} synthetic records")

    # ── Probe raw records against the LSH index ─────────────────────
    print(f"\nProbing raw records from {args.raw} ...")
    stats = Counter()
    results = []
    n_raw = 0

    with open(args.raw) as f:
        for line in f:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if obj.get(args.lang_field, args.language) != args.language:
                continue
            text = obj.get(args.text_field)
            if not text or len(text) < 20:
                continue
            n_raw += 1
            raw_did = doc_id(text)

            if raw_did in exact_set:
                verdict = "exact_duplicate"
                near = []
            else:
                mh = minhash_of(text, num_perm=args.num_perm, k=args.shingle_k)
                near = lsh.query(mh)
                verdict = "near_synthetic" if near else "unique"

            stats[verdict] += 1
            results.append({
                "raw_doc_id": raw_did,
                "verdict":    verdict,
                "near_synth_count": len(near),
            })

            if n_raw % 50000 == 0:
                print(f"  probed {n_raw:,}  {dict(stats)}")

    # ── Report ──────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print(f"  Overlap detection — {args.language}")
    print("=" * 72)
    print(f"  Raw records probed         : {n_raw:,}")
    print(f"  Exact duplicates in synth  : {stats['exact_duplicate']:,}  "
          f"({100*stats['exact_duplicate']/max(n_raw,1):.1f}%)")
    print(f"  Near-synthetic (paraphrase): {stats['near_synthetic']:,}  "
          f"({100*stats['near_synthetic']/max(n_raw,1):.1f}%)")
    print(f"  Unique (safe to ingest)    : {stats['unique']:,}  "
          f"({100*stats['unique']/max(n_raw,1):.1f}%)")
    print(f"\n  Pairwise threshold (Jaccard): {args.threshold}")
    print("=" * 72)

    out = {
        "language":       args.language,
        "raw_file":       args.raw,
        "synthetic_file": args.synthetic,
        "threshold":      args.threshold,
        "shingle_k":      args.shingle_k,
        "n_synth_indexed": n_indexed,
        "n_raw_probed":   n_raw,
        "counts":         dict(stats),
        "results":        results,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Detailed verdicts saved to {args.output}")


if __name__ == "__main__":
    main()
