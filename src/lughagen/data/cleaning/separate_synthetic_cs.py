"""
separate_synthetic_cs.py
─────────────────────────────────────────────────────────────────────────
Separate the synthetic LughaGen corpus into monolingual and code-switched
sub-files using English-marker word ratios.

Rationale: the tier 3 synthetic output (/workspace/synthetic/tier3/fluency_scored.jsonl)
mixes naturally-pure synthetic data with synthetic samples that happened
to include English fragments. For the framework's "synth_mono vs synth_cs"
distinction, we run a post-hoc classifier on each record.

Output structure (creates these per language):
    /workspace/data/corpus/staged/synth_mono/<language>_lughagen_synth.jsonl
    /workspace/data/corpus/staged/synth_cs/<language>_lughagen_synth_cs.jsonl

Each record is augmented with code_switch_ratio (fraction of English-flag
words) and a register classification.

Usage:
    python separate_synthetic_cs.py \\
        --input /workspace/synthetic/tier3/fluency_scored.jsonl \\
        --output-dir /workspace/data/corpus/staged \\
        --threshold 0.15
"""

import argparse
import json
import re
from pathlib import Path
from collections import Counter


# English function words / very-high-frequency content words that almost never
# legitimately appear in pure Bantu/Nilotic text. Kept conservative to avoid
# false positives on borrowed words.
EN_MARKERS = {
    "the","and","of","to","a","in","is","that","for","it","with","as","was",
    "on","are","this","by","be","at","an","have","from","but","what","when",
    "which","who","will","would","can","could","has","had","were","been",
    "they","their","there","not","or","also","because","about","into","than",
    "very","more","some","any","other","such","like","just","only","one",
    "two","first","most","many","much","every","both","new","old","good",
    "you","your","we","our","my","me","i","he","she","his","her","them","us",
}


def cs_ratio(text: str) -> float:
    """Fraction of whitespace-split tokens that are English markers."""
    words = re.findall(r"[A-Za-z']+", text.lower())
    if not words:
        return 0.0
    en_count = sum(1 for w in words if w in EN_MARKERS)
    return en_count / len(words)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True,
                    help="path to fluency_scored.jsonl (tier 3 output)")
    ap.add_argument("--output-dir", required=True,
                    help="root for staged/synth_mono/ and staged/synth_cs/")
    ap.add_argument("--threshold", type=float, default=0.15,
                    help="English-marker ratio above which a record is code-switched (default 0.15)")
    ap.add_argument("--lang-field", default="lang",
                    help="JSON field naming the language of each record")
    ap.add_argument("--text-field", default="text",
                    help="JSON field carrying the text body")
    args = ap.parse_args()

    out_root = Path(args.output_dir)
    mono_dir = out_root / "synth_mono"
    cs_dir   = out_root / "synth_cs"
    mono_dir.mkdir(parents=True, exist_ok=True)
    cs_dir.mkdir(parents=True, exist_ok=True)

    # Open per-language output handles on demand
    mono_handles = {}
    cs_handles   = {}
    counts = Counter()

    def get_handle(handles, dir_, lang, suffix):
        if lang not in handles:
            path = dir_ / f"{lang}_lughagen_synth{suffix}.jsonl"
            handles[lang] = open(path, "w")
        return handles[lang]

    print(f"Reading {args.input}")
    print(f"CS threshold: English-marker ratio >= {args.threshold}")

    with open(args.input) as f:
        for i, line in enumerate(f):
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            lang = obj.get(args.lang_field)
            text = obj.get(args.text_field)
            if not lang or not text:
                continue

            ratio = cs_ratio(text)
            obj["code_switch_ratio"] = round(ratio, 4)

            if ratio >= args.threshold:
                obj["register"] = "code_switched"
                obj["code_switch_pair"] = f"{lang}-english"
                h = get_handle(cs_handles, cs_dir, lang, "_cs")
                counts[(lang, "cs")] += 1
            else:
                obj["register"] = "monolingual"
                obj["code_switch_pair"] = None
                h = get_handle(mono_handles, mono_dir, lang, "")
                counts[(lang, "mono")] += 1

            h.write(json.dumps(obj, ensure_ascii=False) + "\n")

            if (i + 1) % 50000 == 0:
                print(f"  processed {i+1:,} records")

    for h in list(mono_handles.values()) + list(cs_handles.values()):
        h.close()

    # Report
    print("\n" + "=" * 72)
    print("  Synthetic separation summary")
    print("=" * 72)
    print(f"  {'Language':<10}{'Monolingual':>14}{'Code-switched':>16}{'CS %':>10}")
    print("  " + "-" * 50)
    langs = sorted({k[0] for k in counts})
    for lang in langs:
        m = counts.get((lang, "mono"), 0)
        c = counts.get((lang, "cs"),   0)
        pct = 100 * c / (m + c) if (m + c) else 0
        print(f"  {lang:<10}{m:>14,}{c:>16,}{pct:>9.1f}%")
    print("=" * 72)


if __name__ == "__main__":
    main()
