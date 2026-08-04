#!/usr/bin/env python3
"""
Sample records from any JSONL corpus file with optional filters.

Usage examples:

  # 10 random samples from the main training corpus
  python3 sample_corpus.py --file train.jsonl -n 10

  # 5 Kamba synthetic samples
  python3 sample_corpus.py \
      --file /workspace/data/corpus/splits_v2/pathD_v2_filtered/train.jsonl \
      --filter language=kamba \
      --filter origin=synthetic \
      -n 5

  # 5 Sheng records with code_switch_ratio >= 0.3 (rare balanced Sheng)
  python3 sample_corpus.py \
      --file .../by_lang_register_origin/swahili_code_switched_synthetic.jsonl \
      --min code_switch_ratio=0.3 \
      -n 5

  # 3 Kikuyu records from a specific source
  python3 sample_corpus.py \
      --file train.jsonl \
      --filter language=kikuyu \
      --filter source_id=auth_clean_v1_kikuyu \
      -n 3

  # Compare 5 authentic vs 5 synthetic Swahili side-by-side
  python3 sample_corpus.py \
      --file train.jsonl --filter language=swahili --filter origin=authentic -n 5
  python3 sample_corpus.py \
      --file train.jsonl --filter language=swahili --filter origin=synthetic -n 5

  # Long records only (words >= 100)
  python3 sample_corpus.py --file train.jsonl --min words=100 -n 5

  # Full JSON output instead of formatted (for piping)
  python3 sample_corpus.py --file train.jsonl -n 5 --json

  # First N (not random) — deterministic
  python3 sample_corpus.py --file train.jsonl -n 5 --head

  # Save the sample to a file
  python3 sample_corpus.py --file train.jsonl -n 20 --output /tmp/sample.jsonl
"""
from __future__ import annotations
import argparse
import json
import random
import sys
from pathlib import Path

# Corpus shortcuts — expand common paths
SHORTCUTS = {
    "train": "/workspace/data/corpus/splits_v2/pathD_v2_filtered/train.jsonl",
    "val": "/workspace/data/corpus/splits_v2/pathD_v2_filtered/validation_shared_v3.jsonl",
    "test": "/workspace/data/corpus/splits_v2/pathD_v2_filtered/test_shared_v3.jsonl",
    "removed": "/workspace/data/corpus/splits_v2/pathD_v2_filtered/train_removed.jsonl",
    "original": "/workspace/data/corpus/splits_v2/pathD/train.jsonl",
    "sheng": "/workspace/data/corpus/splits_v2/pathD_v2_filtered/by_lang_register_origin/swahili_code_switched_synthetic.jsonl",
    "swa-auth": "/workspace/data/corpus/splits_v2/pathD_v2_filtered/by_language_origin/swahili_authentic.jsonl",
    "swa-syn": "/workspace/data/corpus/splits_v2/pathD_v2_filtered/by_language_origin/swahili_synthetic.jsonl",
    "luo-auth": "/workspace/data/corpus/splits_v2/pathD_v2_filtered/by_language_origin/luo_authentic.jsonl",
    "luo-syn": "/workspace/data/corpus/splits_v2/pathD_v2_filtered/by_language_origin/luo_synthetic.jsonl",
    "kik-auth": "/workspace/data/corpus/splits_v2/pathD_v2_filtered/by_language_origin/kikuyu_authentic.jsonl",
    "kik-syn": "/workspace/data/corpus/splits_v2/pathD_v2_filtered/by_language_origin/kikuyu_synthetic.jsonl",
    "kam-auth": "/workspace/data/corpus/splits_v2/pathD_v2_filtered/by_language_origin/kamba_authentic.jsonl",
    "kam-syn": "/workspace/data/corpus/splits_v2/pathD_v2_filtered/by_language_origin/kamba_synthetic.jsonl",
}


def parse_kv(arg: str) -> tuple[str, str]:
    if "=" not in arg:
        raise argparse.ArgumentTypeError(
            f"expected key=value, got: {arg}"
        )
    k, v = arg.split("=", 1)
    return k.strip(), v.strip()


def parse_kv_num(arg: str) -> tuple[str, float]:
    k, v = parse_kv(arg)
    try:
        return k, float(v)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"expected key=<number>, got: {arg}"
        )


def matches(record: dict, eq_filters, min_filters, max_filters,
            contains_filters) -> bool:
    for k, v in eq_filters:
        rv = record.get(k)
        if rv is None:
            return False
        if str(rv).lower() != v.lower():
            return False
    for k, v in min_filters:
        rv = record.get(k)
        if rv is None or not isinstance(rv, (int, float)) or rv < v:
            return False
    for k, v in max_filters:
        rv = record.get(k)
        if rv is None or not isinstance(rv, (int, float)) or rv > v:
            return False
    for k, sub in contains_filters:
        rv = record.get(k)
        if rv is None or sub.lower() not in str(rv).lower():
            return False
    return True


def main() -> int:
    p = argparse.ArgumentParser(
        description="Sample records from JSONL corpus files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "\nShortcuts for --file:\n  " +
            "\n  ".join(f"{k:12s} -> {v}" for k, v in SHORTCUTS.items())
        ),
    )
    p.add_argument("--file", "-f", required=True,
                   help=("Path to JSONL file OR shortcut name "
                         "(e.g. 'sheng', 'kam-auth')"))
    p.add_argument("-n", type=int, default=5,
                   help="Number of samples to show (default: 5)")
    p.add_argument("--filter", action="append", type=parse_kv, default=[],
                   metavar="KEY=VAL",
                   help="Exact-match filter (repeatable). Example: language=kamba")
    p.add_argument("--min", action="append", type=parse_kv_num, default=[],
                   metavar="KEY=NUM",
                   help="Numeric minimum (repeatable). Example: code_switch_ratio=0.3")
    p.add_argument("--max", action="append", type=parse_kv_num, default=[],
                   metavar="KEY=NUM",
                   help="Numeric maximum (repeatable). Example: words=50")
    p.add_argument("--contains", action="append", type=parse_kv, default=[],
                   metavar="KEY=SUBSTR",
                   help="Substring match on text field (repeatable). "
                        "Example: --contains text=Yesu")
    p.add_argument("--head", action="store_true",
                   help="Take first N matching records (deterministic) "
                        "instead of random")
    p.add_argument("--seed", type=int, default=None,
                   help="Random seed for reproducible sampling")
    p.add_argument("--json", action="store_true",
                   help="Output full JSON (default is formatted preview)")
    p.add_argument("--truncate", type=int, default=250,
                   help="Truncate text preview to N chars (default: 250, "
                        "ignored with --json)")
    p.add_argument("--output", "-o", type=Path, default=None,
                   help="Write sampled records to this JSONL file")
    p.add_argument("--fields", type=str, default=None,
                   help="Show only these fields (comma-separated). "
                        "Example: --fields doc_id,language,text,words")
    p.add_argument("--stats", action="store_true",
                   help="Report match count and stop (no sampling)")

    args = p.parse_args()

    # Resolve shortcut
    file_path = SHORTCUTS.get(args.file, args.file)
    file_path = Path(file_path)
    if not file_path.exists():
        print(f"ERROR: file not found: {file_path}", file=sys.stderr)
        if args.file in SHORTCUTS:
            print(f"       (from shortcut '{args.file}')", file=sys.stderr)
        return 1

    if args.seed is not None:
        random.seed(args.seed)

    fields = None
    if args.fields:
        fields = [f.strip() for f in args.fields.split(",") if f.strip()]

    # ---- Pass 1: reservoir-sample (or count for --stats) ----
    total = 0
    matched = 0
    reservoir: list[dict] = []
    head_collected: list[dict] = []

    with open(file_path) as f:
        for line in f:
            total += 1
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not matches(r, args.filter, args.min, args.max, args.contains):
                continue
            matched += 1
            if args.stats:
                continue
            if args.head:
                if len(head_collected) < args.n:
                    head_collected.append(r)
                elif matched > args.n:
                    # short-circuit for efficiency on large files
                    if not (args.filter or args.min or args.max or args.contains):
                        break
                continue
            # Reservoir sampling
            if len(reservoir) < args.n:
                reservoir.append(r)
            else:
                j = random.randint(0, matched - 1)
                if j < args.n:
                    reservoir[j] = r

    samples = head_collected if args.head else reservoir

    # ---- Report ----
    print("=" * 78)
    print(f"File:     {file_path}")
    print(f"Total records: {total:,}")
    if args.filter or args.min or args.max or args.contains:
        filts = []
        for k, v in args.filter:
            filts.append(f"{k}={v}")
        for k, v in args.min:
            filts.append(f"{k}>={v}")
        for k, v in args.max:
            filts.append(f"{k}<={v}")
        for k, v in args.contains:
            filts.append(f"{k} contains '{v}'")
        print(f"Filters:  " + " AND ".join(filts))
        print(f"Matched:  {matched:,} ({100*matched/total:.2f}% of file)")
    if args.stats:
        return 0
    print(f"Mode:     {'first' if args.head else 'random'} {len(samples)} of {matched:,}")
    print("=" * 78)

    # ---- Print samples ----
    for i, r in enumerate(samples, 1):
        if args.json:
            if fields:
                r = {k: r.get(k) for k in fields}
            print(json.dumps(r, ensure_ascii=False))
        else:
            print(f"\n[{i}]")
            preview_fields = fields or [
                "doc_id", "language", "origin", "register", "domain",
                "code_switch_ratio", "quality_score", "words", "source_id",
            ]
            for k in preview_fields:
                if k in r and r[k] is not None:
                    print(f"  {k:22s} = {r[k]}")
            if "text" in r and (not fields or "text" in fields):
                t = r["text"]
                if len(t) > args.truncate:
                    t = t[:args.truncate] + f"  [... {len(r['text'])-args.truncate} more chars]"
                print(f"  text: {t}")

    # ---- Optional file output ----
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            for r in samples:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"\n[saved] {len(samples)} records → {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
