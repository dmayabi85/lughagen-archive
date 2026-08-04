"""
inventory_corpus_assets.py
─────────────────────────────────────────────────────────────────────────
Walk the Brev workspace and catalogue every data file (JSONL, JSON, CSV,
TSV, TXT, parquet) with size, document count, schema preview, and a
tentative classification along the four corpus-framework axes:

    origin   = synthetic | authentic | unknown
    register = monolingual | code_switched | unknown
    language = swahili | kikuyu | kamba | luo | english | mixed | unknown
    role     = seed | training_source | derived | intermediate | reference

Classification is rule-based from the file path and a small sample of
content; you'll review and correct it before writing manifests.

Output:
    /workspace/data/corpus/audit/inventory.csv
    /workspace/data/corpus/audit/inventory.json
    Console summary table

Usage:
    python inventory_corpus_assets.py --root /workspace --output-dir /workspace/data/corpus/audit
"""

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path


# ── Rule-based classifiers from file path ───────────────────────────────
PATH_LANG = {
    "swahili": "swahili", "sw_": "swahili", "/sw/": "swahili",
    "kikuyu": "kikuyu",  "ki_": "kikuyu",  "/ki/": "kikuyu", "kik": "kikuyu",
    "kamba":  "kamba",   "kam_": "kamba",  "/kam/": "kamba",
    "luo":    "luo",     "luo_": "luo",    "/luo/": "luo",
    "english":"english", "/en/": "english",
}

PATH_ORIGIN = {
    "/synthetic/": "synthetic",
    "/tier1/":     "synthetic",
    "/tier2/":     "synthetic",
    "/tier3/":     "synthetic",
    "generated":   "synthetic",
    "/raw/":       "authentic",
    "/secondary/": "authentic",
    "/cleaned":    "authentic",
    "/clean_datasets/": "authentic",
}

PATH_ROLE_HINT = {
    "raw":          "seed_or_source",
    "cleaned":      "cleaned_seed",
    "prescrub":     "intermediate",
    "rejected":     "rejected_intermediate",
    "audited":      "intermediate",
    "final_dataset":"derived_synthetic",
    "tier3":        "derived_synthetic",
    "fluency_scored":"derived_synthetic",
    "topics_classified":"derived_synthetic",
    "secondary":    "authentic_source",
    "tokenizer":    "tooling",
    "checkpoint":   "model_artefact",
}


def classify_path(path_str: str):
    lower = path_str.lower()
    lang = next((v for k, v in PATH_LANG.items() if k in lower), "unknown")
    origin = next((v for k, v in PATH_ORIGIN.items() if k in lower), "unknown")
    role = next((v for k, v in PATH_ROLE_HINT.items() if k in lower), "unknown")
    return lang, origin, role


# ── Content sniffing ────────────────────────────────────────────────────
def sniff_jsonl(path: Path, peek_lines: int = 200):
    """Read up to peek_lines records to determine schema and sample text."""
    sample_keys = set()
    sample_text = None
    n_read = n_valid = 0
    try:
        with open(path) as f:
            for i, line in enumerate(f):
                if i >= peek_lines:
                    break
                n_read += 1
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                n_valid += 1
                if isinstance(obj, dict):
                    sample_keys.update(obj.keys())
                    if sample_text is None:
                        for k in ("text", "content", "sentence", "translation", "src", "tgt"):
                            if isinstance(obj.get(k), str) and len(obj[k]) > 20:
                                sample_text = obj[k][:160]
                                break
    except Exception as e:
        return {"error": str(e), "sample_keys": [], "sample_text": None, "valid_pct": 0}

    valid_pct = round(100 * n_valid / max(n_read, 1), 1)
    return {
        "sample_keys": sorted(sample_keys),
        "sample_text": sample_text,
        "valid_pct":   valid_pct,
        "peeked":      n_read,
    }


def line_count(path: Path) -> int:
    """Count lines fast — wc -l style."""
    try:
        n = 0
        with open(path, "rb") as f:
            for _ in f:
                n += 1
        return n
    except Exception:
        return -1


def human(n_bytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n_bytes < 1024:
            return f"{n_bytes:.1f}{unit}"
        n_bytes /= 1024
    return f"{n_bytes:.1f}PB"


# ── Walk and catalogue ──────────────────────────────────────────────────
DATA_SUFFIXES = {".jsonl", ".json", ".csv", ".tsv", ".txt", ".parquet"}
SKIP_DIRS     = {"__pycache__", ".git", "node_modules", ".cache"}


def walk_data(root: Path, min_bytes: int = 100):
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if any(part in SKIP_DIRS for part in p.parts):
            continue
        if p.suffix.lower() not in DATA_SUFFIXES:
            continue
        try:
            size = p.stat().st_size
        except OSError:
            continue
        if size < min_bytes:
            continue
        yield p, size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root",       default="/workspace")
    ap.add_argument("--output-dir", default="/workspace/data/corpus/audit")
    ap.add_argument("--min-bytes",  type=int, default=10_000,
                    help="ignore files smaller than this (default 10 KB)")
    ap.add_argument("--peek-lines", type=int, default=200,
                    help="lines to read for schema sniffing per JSONL")
    args = ap.parse_args()

    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    csv_path  = out_dir / "inventory.csv"
    json_path = out_dir / "inventory.json"

    rows = []
    by_origin = defaultdict(lambda: [0, 0])    # [files, bytes]
    by_lang   = defaultdict(lambda: [0, 0])

    print(f"Scanning {args.root} ...")
    for p, size in walk_data(Path(args.root), min_bytes=args.min_bytes):
        path_str = str(p)
        lang, origin, role = classify_path(path_str)
        ext = p.suffix.lower()

        sniff = {}
        n_lines = None
        if ext in (".jsonl", ".json"):
            sniff = sniff_jsonl(p, peek_lines=args.peek_lines)
            if ext == ".jsonl":
                n_lines = line_count(p)

        row = {
            "path":            path_str,
            "size_bytes":      size,
            "size_human":      human(size),
            "ext":             ext,
            "lines":           n_lines,
            "lang_guess":      lang,
            "origin_guess":    origin,
            "role_guess":      role,
            "schema_keys":     ", ".join(sniff.get("sample_keys", [])[:8]) if sniff.get("sample_keys") else "",
            "sample_text":     (sniff.get("sample_text") or "")[:160],
            "json_valid_pct":  sniff.get("valid_pct", ""),
        }
        rows.append(row)
        by_origin[origin][0] += 1
        by_origin[origin][1] += size
        by_lang[lang][0]   += 1
        by_lang[lang][1]   += size

    rows.sort(key=lambda r: -r["size_bytes"])

    # ── Write CSV ────────────────────────────────────────────────────────
    with open(csv_path, "w", newline="") as f:
        cols = ["path","size_human","size_bytes","ext","lines",
                "lang_guess","origin_guess","role_guess",
                "schema_keys","sample_text","json_valid_pct"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in cols})

    with open(json_path, "w") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

    # ── Console summary ─────────────────────────────────────────────────
    print("\n" + "=" * 92)
    print(f"  Inventory: {len(rows):,} files \u2265 {args.min_bytes:,} bytes under {args.root}")
    print("=" * 92)

    print(f"\n  {'Origin (guessed)':<14}  {'Files':>7}   {'Total size':>10}")
    print("  " + "-" * 40)
    for k in sorted(by_origin, key=lambda x: -by_origin[x][1]):
        n, b = by_origin[k]
        print(f"  {k:<14}  {n:>7,}   {human(b):>10}")

    print(f"\n  {'Language (guessed)':<14}  {'Files':>7}   {'Total size':>10}")
    print("  " + "-" * 40)
    for k in sorted(by_lang, key=lambda x: -by_lang[x][1]):
        n, b = by_lang[k]
        print(f"  {k:<14}  {n:>7,}   {human(b):>10}")

    print(f"\n  Top 20 files by size:")
    print(f"  {'Size':>9}  {'Lines':>9}  {'Origin':<10}  {'Lang':<10}  Path")
    print("  " + "-" * 86)
    for r in rows[:20]:
        lines_s = f"{r['lines']:,}" if r['lines'] and r['lines'] > 0 else "-"
        print(f"  {r['size_human']:>9}  {lines_s:>9}  {r['origin_guess']:<10}  "
              f"{r['lang_guess']:<10}  {r['path']}")

    print()
    print(f"  Full inventory written to:")
    print(f"    {csv_path}")
    print(f"    {json_path}")
    print()
    print("  Next step: open inventory.csv in a spreadsheet, correct the guessed")
    print("  fields where wrong, and use it to drive the manifest YAMLs.")
    print("=" * 92)


if __name__ == "__main__":
    main()
