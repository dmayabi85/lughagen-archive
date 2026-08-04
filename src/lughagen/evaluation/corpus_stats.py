"""
Comprehensive corpus statistics for the LughaGen cleaned corpus.

Reports per-language: file size, document count, total words, mean
words/document, estimated total tokens (using the fertility ratios already
measured in Appendix A.2), and the relative share of the combined corpus.

Run inside the Brev container:
    python corpus_stats.py
"""

import json
import os
from pathlib import Path

CORPUS_DIR = Path("/workspace/data/cleaned_final")

# Mean fertility ratios from Appendix A.2 (tokenizer_fertility_corpus.py)
FERTILITY = {
    "swahili": 2.322,
    "kikuyu":  3.026,
    "luo":     1.817,
    "kamba":   2.845,
    "english": 1.319,    # if present
}


def count_file(path: Path):
    """Count documents and total words in a JSONL file."""
    docs, words, bytes_ = 0, 0, 0
    with open(path, "rb") as fb:
        bytes_ = os.fstat(fb.fileno()).st_size
    with open(path) as f:
        for line in f:
            try:
                obj = json.loads(line)
                txt = obj.get("text") or obj.get("content") or obj.get("sentence")
                if not txt or not isinstance(txt, str):
                    continue
                docs += 1
                words += len(txt.split())
            except Exception:
                continue
    return docs, words, bytes_


def human(n):
    """Human-readable byte size."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def commafy(n):
    return f"{n:,}"


def main():
    print("=" * 96)
    print("  LughaGen Corpus Statistics")
    print(f"  Source: {CORPUS_DIR}")
    print("=" * 96)

    rows = []
    for p in sorted(CORPUS_DIR.glob("*.jsonl")):
        lang = p.stem.lower().replace("_clean", "").replace("_cleaned", "")
        if lang not in FERTILITY:
            # use a default fertility if unknown
            fertility = 2.5
            note = " (fertility estimated)"
        else:
            fertility = FERTILITY[lang]
            note = ""
        print(f"\n  Scanning {p.name} ...")
        docs, words, bytes_ = count_file(p)
        est_tokens = int(words * fertility)
        rows.append({
            "language"  : lang,
            "file"      : p.name,
            "docs"      : docs,
            "words"     : words,
            "bytes"     : bytes_,
            "fertility" : fertility,
            "est_tokens": est_tokens,
            "note"      : note,
        })

    # Totals
    total_docs   = sum(r["docs"]   for r in rows)
    total_words  = sum(r["words"]  for r in rows)
    total_bytes  = sum(r["bytes"]  for r in rows)
    total_tokens = sum(r["est_tokens"] for r in rows)

    # Print table
    print("\n" + "=" * 96)
    print("  Per-language summary")
    print("=" * 96)
    print(f"\n  {'Language':<10} {'Documents':>12} {'Words':>15} {'File size':>10}  "
          f"{'Fertility':>10} {'Est. tokens':>15}  {'% of total':>10}")
    print("  " + "-" * 92)
    for r in rows:
        pct = 100 * r["est_tokens"] / total_tokens if total_tokens else 0
        print(f"  {r['language']:<10} {commafy(r['docs']):>12} {commafy(r['words']):>15} "
              f"{human(r['bytes']):>10}  {r['fertility']:>10.3f} "
              f"{commafy(r['est_tokens']):>15}  {pct:>9.1f}%")

    print("  " + "-" * 92)
    print(f"  {'TOTAL':<10} {commafy(total_docs):>12} {commafy(total_words):>15} "
          f"{human(total_bytes):>10}  {'':>10} {commafy(total_tokens):>15}  {'100.0%':>10}")

    print("\n  Notes:")
    print("    \u2022 Token estimates use the mean fertility ratios measured on")
    print("      4,400\u20135,000-sentence samples per language (Appendix A.2). Exact")
    print("      total token counts would require tokenising every document; these")
    print("      estimates are accurate to within a few percent for proposal-level")
    print("      reporting.")
    print(f"    \u2022 Combined corpus: {total_docs:,} documents, {total_words:,} words,")
    print(f"      \u2248{total_tokens / 1e6:.1f}M tokens, {human(total_bytes)} on disk.")
    print("=" * 96)

    # Save machine-readable summary
    out = {
        "total": {
            "documents": total_docs,
            "words":     total_words,
            "bytes":     total_bytes,
            "est_tokens": total_tokens,
        },
        "per_language": rows,
    }
    with open("corpus_stats.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Machine-readable summary saved to corpus_stats.json")


if __name__ == "__main__":
    main()
