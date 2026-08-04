#!/usr/bin/env python3
"""
Build a filtered version of the training corpus by removing records flagged
as likely-drift or likely-scraped-noise in the Phase 2 contamination audit.

Writes THREE outputs:
  1. <output_dir>/train.jsonl              -- kept records
  2. <output_dir>/train_removed.jsonl      -- excluded records (quarantine)
  3. <output_dir>/filter_manifest.json     -- what was excluded and why

Keeps records categorised as legitimate (religious references, loanwords,
label ambiguity, unclear) since removing them would be over-filtering.

Matches records by doc_id.
"""
import argparse
import csv
import datetime as dt
import json
import shutil
import sys
from pathlib import Path


def load_exclusion_ids(index_path, exclude_categories):
    """Read contamination_index.csv; return dict {doc_id: category} for
    excluded records, plus counts by category."""
    excluded_map = {}      # doc_id -> category (for provenance)
    kept_counts = {}
    excluded_counts = {}

    with open(index_path) as f:
        reader = csv.DictReader(f)
        if "doc_id" not in reader.fieldnames:
            sys.exit(f"ERROR: contamination CSV missing 'doc_id'. Fields: {reader.fieldnames}")
        if "category" not in reader.fieldnames:
            sys.exit(f"ERROR: contamination CSV missing 'category'. Fields: {reader.fieldnames}")

        for row in reader:
            category = (row.get("category") or "").strip()
            doc_id = (row.get("doc_id") or "").strip()
            if not doc_id or not category:
                continue
            if category in exclude_categories:
                excluded_map[doc_id] = category
                excluded_counts[category] = excluded_counts.get(category, 0) + 1
            else:
                kept_counts[category] = kept_counts.get(category, 0) + 1

    return excluded_map, kept_counts, excluded_counts


def filter_jsonl(input_path, kept_path, removed_path, excluded_map):
    """Stream input JSONL. Records with doc_id in excluded_map go to
    removed_path (quarantine); others go to kept_path.
    """
    kept_path.parent.mkdir(parents=True, exist_ok=True)
    removed_path.parent.mkdir(parents=True, exist_ok=True)

    n_in = n_kept = n_removed = n_no_id = 0

    with open(input_path) as f_in, \
         open(kept_path, "w") as f_kept, \
         open(removed_path, "w") as f_removed:

        for line_no, line in enumerate(f_in, start=1):
            n_in += 1
            line = line.rstrip("\n")
            if not line:
                continue

            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"WARN: JSON decode error at line {line_no}: {e}", file=sys.stderr)
                continue

            doc_id = rec.get("doc_id")

            if doc_id is None:
                # No doc_id -> can't filter -> keep in main output
                n_no_id += 1
                f_kept.write(line + "\n")
                n_kept += 1
                continue

            if doc_id in excluded_map:
                # Annotate the removed record with the reason for quarantine
                rec["_removed_reason_category"] = excluded_map[doc_id]
                f_removed.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_removed += 1
            else:
                f_kept.write(line + "\n")
                n_kept += 1

    return n_in, n_kept, n_removed, n_no_id


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--contamination-index", required=True)
    ap.add_argument("--exclude-categories", required=True)
    ap.add_argument("--output", required=True,
                    help="Path to output kept train.jsonl (removed goes to same dir)")
    ap.add_argument("--copy-val-test", action="store_true")
    ap.add_argument("--val-source",
                    default="/workspace/data/corpus/splits_v2/validation_shared_v3.jsonl")
    ap.add_argument("--test-source",
                    default="/workspace/data/corpus/splits_v2/test_shared_v3.jsonl")
    args = ap.parse_args()

    input_path = Path(args.input)
    kept_path = Path(args.output)
    index_path = Path(args.contamination_index)

    if not input_path.exists():
        sys.exit(f"ERROR: input {input_path} does not exist")
    if not index_path.exists():
        sys.exit(f"ERROR: contamination index {index_path} does not exist")

    exclude_categories = set(c.strip() for c in args.exclude_categories.split(","))

    output_dir = kept_path.parent
    # Removed records go alongside the kept ones
    removed_path = output_dir / (kept_path.stem + "_removed.jsonl")
    manifest_path = output_dir / "filter_manifest.json"

    print(f"Input:        {input_path}")
    print(f"Contamination index: {index_path}")
    print(f"Kept output:  {kept_path}")
    print(f"Removed (quarantine): {removed_path}")
    print(f"Exclude categories:   {sorted(exclude_categories)}")

    print(f"\n[step 1] Loading contamination index")
    excluded_map, kept_cats, excl_cats = load_exclusion_ids(index_path, exclude_categories)

    print(f"\n  Kept categories (records NOT excluded):")
    if kept_cats:
        for cat, n in sorted(kept_cats.items(), key=lambda x: -x[1]):
            print(f"    {cat:50s}  {n:5d}")
    else:
        print("    (none)")

    print(f"\n  Excluded categories:")
    if excl_cats:
        for cat, n in sorted(excl_cats.items(), key=lambda x: -x[1]):
            print(f"    {cat:50s}  {n:5d}")
    else:
        print("    (none)")

    print(f"\n  Total doc_ids to exclude: {len(excluded_map)}")

    print(f"\n[step 2] Filtering...")
    n_in, n_kept, n_removed, n_no_id = filter_jsonl(
        input_path, kept_path, removed_path, excluded_map
    )
    print(f"  Read:          {n_in:>10,}")
    print(f"  Kept:          {n_kept:>10,}")
    print(f"  Removed (quarantined): {n_removed:>10,}")
    if n_no_id > 0:
        print(f"  No doc_id (kept anyway): {n_no_id:>5,}")
    print(f"  Removal rate:  {n_removed / max(n_in, 1) * 100:.4f}%")

    # Write manifest
    manifest = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source": {
            "input_jsonl": str(input_path),
            "contamination_index": str(index_path),
        },
        "outputs": {
            "kept_jsonl": str(kept_path),
            "removed_jsonl": str(removed_path),
        },
        "filter": {
            "exclude_categories": sorted(exclude_categories),
            "kept_categories": {k: v for k, v in sorted(kept_cats.items())},
            "excluded_categories": {k: v for k, v in sorted(excl_cats.items())},
        },
        "counts": {
            "read": n_in,
            "kept": n_kept,
            "removed": n_removed,
            "no_doc_id": n_no_id,
        },
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"\n[step 3] Wrote manifest: {manifest_path}")

    if args.copy_val_test:
        print(f"\n[step 4] Copying validation and test splits (unchanged)")
        val_out = output_dir / Path(args.val_source).name
        test_out = output_dir / Path(args.test_source).name
        shutil.copy(args.val_source, val_out)
        shutil.copy(args.test_source, test_out)
        print(f"  {args.val_source}  ->  {val_out}")
        print(f"  {args.test_source} ->  {test_out}")

    print(f"\n[done] Filtered corpus at: {output_dir}")
    print(f"       Train (kept):        {n_kept:,}")
    print(f"       Quarantine (removed): {n_removed:,}")


if __name__ == "__main__":
    main()
