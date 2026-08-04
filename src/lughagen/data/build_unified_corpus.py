"""
build_unified_corpus.py
─────────────────────────────────────────────────────────────────────────
Combine all files under staged/ into unified train/validation/test JSONL,
and emit detailed corpus statistics across every axis the framework tracks
(origin x register x language x domain x source).

Statistics output answers questions like:
    - "X% of training tokens are synthetic, Y% authentic"
    - "Code-switched samples are Z% of the training corpus"
    - "Domain distribution per language"
    - "Top sources contributing to the corpus"

Usage:
    python build_unified_corpus.py \\
        --staged-dir   /workspace/data/corpus/staged \\
        --splits-dir   /workspace/data/corpus/splits \\
        --stats-dir    /workspace/data/corpus/stats
"""

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


def _bump(d, key, w_docs=1, w_words=0, w_tokens=0):
    """Accumulate three parallel counters keyed by tuple."""
    s = d[key]
    s[0] += w_docs
    s[1] += w_words
    s[2] += w_tokens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--staged-dir", required=True)
    ap.add_argument("--splits-dir", required=True)
    ap.add_argument("--stats-dir",  required=True)
    args = ap.parse_args()

    staged = Path(args.staged_dir)
    splits = Path(args.splits_dir); splits.mkdir(parents=True, exist_ok=True)
    stats_dir = Path(args.stats_dir); stats_dir.mkdir(parents=True, exist_ok=True)

    # ── open split output files ──────────────────────────────────────
    out_files = {
        s: open(splits / f"{s}.jsonl", "w") for s in ("train", "validation", "test")
    }

    # ── counters: each value is [docs, words, tokens_llama] ──────────
    def newC():
        return defaultdict(lambda: [0, 0, 0])

    by_origin     = newC()
    by_register   = newC()
    by_language   = newC()
    by_domain     = newC()
    by_source     = newC()
    by_split      = newC()
    by_lang_reg   = newC()             # language x register
    by_lang_orig  = newC()             # language x origin
    by_origin_reg = newC()             # origin x register
    overall       = [0, 0, 0]

    inputs = sorted(staged.rglob("*.jsonl"))
    print(f"Combining {len(inputs)} staged files \u2192 {splits}/")

    for path in inputs:
        with open(path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue

                words  = r.get("words")  or len(r.get("text", "").split())
                tokens = r.get("tokens_llama") or 0
                origin = r.get("origin", "unknown")
                reg    = r.get("register", "unknown")
                lang   = r.get("language", "unknown")
                dom    = r.get("domain") or "unspecified"
                src    = r.get("source_id", "unknown")
                split  = r.get("split", "train")

                _bump(by_origin,     origin,            1, words, tokens)
                _bump(by_register,   reg,               1, words, tokens)
                _bump(by_language,   lang,              1, words, tokens)
                _bump(by_domain,     dom,               1, words, tokens)
                _bump(by_source,     src,               1, words, tokens)
                _bump(by_split,      split,             1, words, tokens)
                _bump(by_lang_reg,   (lang, reg),       1, words, tokens)
                _bump(by_lang_orig,  (lang, origin),    1, words, tokens)
                _bump(by_origin_reg, (origin, reg),     1, words, tokens)
                overall[0] += 1
                overall[1] += words
                overall[2] += tokens

                out_files[split].write(json.dumps(r, ensure_ascii=False) + "\n")

    for f in out_files.values():
        f.close()

    # ── helpers for reporting ────────────────────────────────────────
    def pct(part, whole):
        return (100 * part / whole) if whole else 0

    def table(name, counter, unit_idx=1):
        """Return list of {key, docs, words, %words}."""
        rows = []
        whole = sum(v[unit_idx] for v in counter.values())
        for k, v in sorted(counter.items(), key=lambda x: -x[1][unit_idx]):
            rows.append({
                "category": name,
                "key":      k if isinstance(k, str) else " x ".join(map(str, k)),
                "docs":     v[0],
                "words":    v[1],
                "tokens":   v[2],
                "pct_words": round(pct(v[1], whole), 2),
            })
        return rows

    # ── compose stats dict ───────────────────────────────────────────
    stats = {
        "overall": {
            "documents": overall[0],
            "words":     overall[1],
            "tokens_llama_estimated": overall[2],
        },
        "by_origin":         table("origin",       by_origin),
        "by_register":       table("register",     by_register),
        "by_language":       table("language",     by_language),
        "by_domain":         table("domain",       by_domain),
        "by_source":         table("source",       by_source),
        "by_split":          table("split",        by_split),
        "by_language_x_register": table("language_x_register", by_lang_reg),
        "by_language_x_origin":   table("language_x_origin",   by_lang_orig),
        "by_origin_x_register":   table("origin_x_register",   by_origin_reg),
    }

    # Headline-figure shortcuts (the kinds of statements the proposal needs)
    by_orig_words = {k: v[1] for k, v in by_origin.items()}
    by_reg_words  = {k: v[1] for k, v in by_register.items()}
    total = overall[1]
    stats["headline"] = {
        "pct_synthetic":     round(pct(by_orig_words.get("synthetic", 0), total),  2),
        "pct_authentic":     round(pct(by_orig_words.get("authentic", 0), total),  2),
        "pct_monolingual":   round(pct(by_reg_words.get("monolingual", 0), total), 2),
        "pct_code_switched": round(pct(by_reg_words.get("code_switched", 0), total), 2),
    }

    # ── write outputs ────────────────────────────────────────────────
    json_path = stats_dir / "corpus_stats.json"
    with open(json_path, "w") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    csv_path = stats_dir / "corpus_stats.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["category", "key", "docs", "words", "tokens", "pct_words"])
        for section in (
            "by_origin","by_register","by_language","by_domain","by_source",
            "by_split","by_language_x_register","by_language_x_origin","by_origin_x_register"
        ):
            for row in stats[section]:
                w.writerow([row["category"], row["key"], row["docs"],
                            row["words"], row["tokens"], row["pct_words"]])

    # ── print headline report ────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  Corpus composition (by words)")
    print("=" * 72)
    print(f"  Total documents : {overall[0]:>15,}")
    print(f"  Total words     : {overall[1]:>15,}")
    print()
    h = stats["headline"]
    print(f"  Synthetic       : {h['pct_synthetic']:>5.1f}%")
    print(f"  Authentic       : {h['pct_authentic']:>5.1f}%")
    print(f"  Monolingual     : {h['pct_monolingual']:>5.1f}%")
    print(f"  Code-switched   : {h['pct_code_switched']:>5.1f}%")
    print()
    print("  Per-language x register breakdown:")
    for row in stats["by_language_x_register"]:
        print(f"    {row['key']:<28}{row['words']:>12,}  ({row['pct_words']:>5.2f}% of corpus)")
    print()
    print(f"  Full JSON: {json_path}")
    print(f"  Full CSV : {csv_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
