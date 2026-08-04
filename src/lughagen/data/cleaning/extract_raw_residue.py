"""
extract_raw_residue.py
─────────────────────────────────────────────────────────────────────────
Given the overlap report from detect_paraphrastic_overlap.py, extract the
subset of raw records flagged as "unique" — i.e., records that do NOT
have a paraphrastic or exact match in the synthetic corpus.

This is the safe subset to ingest as authentic_monolingual data alongside
the synthetic corpus, without double-counting.

Output: a residue JSONL file that can then be referenced by a YAML manifest
under manifests/authentic_monolingual/.

Usage:
    python extract_raw_residue.py \\
        --raw     /workspace/data/cleaned_final/swahili_clean.jsonl \\
        --overlap /workspace/data/corpus/audit/overlap_swahili.json \\
        --output  /workspace/data/secondary/monolingual/swahili_raw_residue.jsonl \\
        --language swahili
"""

import argparse
import hashlib
import json
import re
import unicodedata
from pathlib import Path


def normalise(text):
    t = unicodedata.normalize("NFC", text or "")
    t = re.sub(r"\s+", " ", t).strip().lower()
    return t


def doc_id(text):
    return hashlib.sha1(normalise(text).encode("utf-8")).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw",      required=True)
    ap.add_argument("--overlap",  required=True)
    ap.add_argument("--output",   required=True)
    ap.add_argument("--language", required=True)
    ap.add_argument("--lang-field", default="lang")
    ap.add_argument("--text-field", default="text")
    args = ap.parse_args()

    # Load verdicts
    with open(args.overlap) as f:
        report = json.load(f)
    unique_ids = {r["raw_doc_id"] for r in report["results"] if r["verdict"] == "unique"}
    print(f"Loaded {len(unique_ids):,} unique-verdict doc_ids from overlap report")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    n_in = n_kept = n_skipped = 0
    with open(args.raw) as f_in, open(args.output, "w") as f_out:
        for line in f_in:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if obj.get(args.lang_field, args.language) != args.language:
                continue
            text = obj.get(args.text_field)
            if not text or len(text) < 20:
                continue
            n_in += 1
            if doc_id(text) in unique_ids:
                f_out.write(json.dumps(obj, ensure_ascii=False) + "\n")
                n_kept += 1
            else:
                n_skipped += 1

    print(f"\n  Residue extraction for {args.language}")
    print(f"  Input records       : {n_in:,}")
    print(f"  Kept (unique only)  : {n_kept:,}  ({100*n_kept/max(n_in,1):.1f}%)")
    print(f"  Skipped (overlap)   : {n_skipped:,}")
    print(f"  Output              : {args.output}")


if __name__ == "__main__":
    main()
