"""
separate_synthetic_cs_v2.py
─────────────────────────────────────────────────────────────────────────
Separate synthetic LughaGen records into monolingual and code-switched
sub-files using:

  1. The pipeline's own labels (`type` and `original_style` fields), which
     mark whether the generator intended a record to be code-switched.
  2. An English-marker ratio computed per record, used as a verification
     check and to surface mislabelled records.

Preserves the rich tier 2/3 metadata (quality, fluency, GlotLID/Afrolid
audit, topic, perplexity) on every output record so downstream stages and
the framework's manifests can use them.

Output per language:
    /workspace/data/corpus/staged/synth_mono/<language>_lughagen_synth.jsonl
    /workspace/data/corpus/staged/synth_cs/<language>_lughagen_synth_cs.jsonl

Also writes a small QA report:
    /workspace/data/corpus/audit/synth_separation_qa.json

Usage:
    python separate_synthetic_cs_v2.py \\
        --input /workspace/synthetic/tier3/fluency_scored.jsonl \\
        --output-dir /workspace/data/corpus/staged \\
        --audit-dir /workspace/data/corpus/audit \\
        --en-threshold 0.15
"""

import argparse
import json
import re
from pathlib import Path
from collections import Counter


# Conservative English-marker word list (function words + very-high-frequency
# tokens that almost never legitimately appear in pure Bantu/Nilotic text)
EN_MARKERS = {
    "the","and","of","to","a","in","is","that","for","it","with","as","was",
    "on","are","this","by","be","at","an","have","from","but","what","when",
    "which","who","will","would","can","could","has","had","were","been",
    "they","their","there","not","or","also","because","about","into","than",
    "very","more","some","any","other","such","like","just","only","you",
    "your","we","our","my","me","he","she","his","her","them","us","if","so",
}

# Pipeline-label patterns that indicate the GENERATOR intended CS output
CS_TYPE_PATTERNS  = (re.compile(r"code.?switch", re.I),)
CS_STYLE_PATTERNS = (re.compile(r"code.?switch", re.I),)


def cs_ratio(text: str) -> float:
    """Fraction of whitespace-split tokens that are English markers."""
    words = re.findall(r"[A-Za-z']+", text.lower())
    if not words:
        return 0.0
    en = sum(1 for w in words if w in EN_MARKERS)
    return en / len(words)


def is_pipeline_cs(record: dict) -> bool:
    """Does the pipeline's own metadata mark this record as code-switched?"""
    type_field  = str(record.get("type", ""))
    style_field = str(record.get("original_style", ""))
    if any(p.search(type_field) for p in CS_TYPE_PATTERNS):
        return True
    if any(p.search(style_field) for p in CS_STYLE_PATTERNS):
        return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",        required=True)
    ap.add_argument("--output-dir",   required=True)
    ap.add_argument("--audit-dir",    required=True)
    ap.add_argument("--en-threshold", type=float, default=0.15,
                    help="English-marker ratio above which an unlabelled record is also flagged CS")
    ap.add_argument("--lang-field",   default="language")
    ap.add_argument("--text-field",   default="text")
    args = ap.parse_args()

    out_root = Path(args.output_dir)
    mono_dir = out_root / "synth_mono"
    cs_dir   = out_root / "synth_cs"
    mono_dir.mkdir(parents=True, exist_ok=True)
    cs_dir.mkdir(parents=True, exist_ok=True)
    Path(args.audit_dir).mkdir(parents=True, exist_ok=True)

    mono_handles = {}
    cs_handles = {}

    def get_handle(handles, dir_, lang, suffix):
        if lang not in handles:
            path = dir_ / f"{lang}_lughagen_synth{suffix}.jsonl"
            handles[lang] = open(path, "w")
        return handles[lang]

    counts = Counter()
    # Cross-classification counter for QA: (lang, pipeline_says, ratio_says)
    cross = Counter()

    print(f"Reading {args.input}")
    print(f"Source of truth: pipeline label (type / original_style fields)")
    print(f"Verification threshold: English-marker ratio >= {args.en_threshold}")

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

            pipeline_cs = is_pipeline_cs(obj)
            ratio = cs_ratio(text)
            ratio_cs = (ratio >= args.en_threshold)

            # Augment the record
            obj["code_switch_ratio"]    = round(ratio, 4)
            obj["cs_label_pipeline"]    = pipeline_cs
            obj["cs_label_en_threshold"] = ratio_cs

            # Final routing decision: pipeline label is ground truth
            if pipeline_cs:
                obj["register"] = "code_switched"
                obj["code_switch_pair"] = f"{lang}-english"
                h = get_handle(cs_handles, cs_dir, lang, "_cs")
                counts[(lang, "cs")] += 1
            else:
                obj["register"] = "monolingual"
                obj["code_switch_pair"] = None
                h = get_handle(mono_handles, mono_dir, lang, "")
                counts[(lang, "mono")] += 1

            # QA cross-classification
            cell = "agree_cs" if (pipeline_cs and ratio_cs) else \
                   "agree_mono" if (not pipeline_cs and not ratio_cs) else \
                   "pipeline_cs_low_ratio" if pipeline_cs else \
                   "pipeline_mono_high_ratio"
            cross[(lang, cell)] += 1

            h.write(json.dumps(obj, ensure_ascii=False) + "\n")

            if (i + 1) % 50000 == 0:
                print(f"  processed {i+1:,} records")

    for h in list(mono_handles.values()) + list(cs_handles.values()):
        h.close()

    # ── Report ──────────────────────────────────────────────────────
    print("\n" + "=" * 76)
    print("  Synthetic separation summary (pipeline labels = source of truth)")
    print("=" * 76)
    print(f"  {'Language':<10}{'Monolingual':>14}{'Code-switched':>16}{'Total':>14}{'CS %':>10}")
    print("  " + "-" * 64)

    langs = sorted({k[0] for k in counts})
    grand_mono = grand_cs = 0
    for lang in langs:
        m = counts.get((lang, "mono"), 0)
        c = counts.get((lang, "cs"),   0)
        grand_mono += m
        grand_cs   += c
        pct = 100 * c / (m + c) if (m + c) else 0
        print(f"  {lang:<10}{m:>14,}{c:>16,}{m+c:>14,}{pct:>9.1f}%")

    grand_tot = grand_mono + grand_cs
    grand_pct = 100 * grand_cs / grand_tot if grand_tot else 0
    print("  " + "-" * 64)
    print(f"  {'TOTAL':<10}{grand_mono:>14,}{grand_cs:>16,}{grand_tot:>14,}{grand_pct:>9.1f}%")
    print("=" * 76)

    # ── QA: agreement between pipeline label and English-marker ratio ────
    print("\n  QA: agreement between pipeline label and English-marker ratio")
    print("  " + "-" * 72)
    print(f"  {'Language':<10}{'agree_mono':>14}{'agree_cs':>10}"
          f"{'pipeline_cs_low':>18}{'pipeline_mono_hi':>20}")
    for lang in langs:
        am = cross.get((lang, "agree_mono"), 0)
        ac = cross.get((lang, "agree_cs"), 0)
        plr = cross.get((lang, "pipeline_cs_low_ratio"), 0)
        pmh = cross.get((lang, "pipeline_mono_high_ratio"), 0)
        print(f"  {lang:<10}{am:>14,}{ac:>10,}{plr:>18,}{pmh:>20,}")
    print("  " + "-" * 72)
    print("  pipeline_cs_low      = pipeline says CS but English ratio < threshold")
    print("  pipeline_mono_high   = pipeline says mono but English ratio >= threshold")
    print("    (these are worth spot-checking for mislabelled records)")

    # ── Save audit JSON ─────────────────────────────────────────────────
    audit = {
        "input":            args.input,
        "en_threshold":     args.en_threshold,
        "totals": {
            "monolingual":   grand_mono,
            "code_switched": grand_cs,
            "all":           grand_tot,
            "cs_percent":    round(grand_pct, 2),
        },
        "per_language":     {lang: {
            "monolingual":   counts.get((lang, "mono"), 0),
            "code_switched": counts.get((lang, "cs"), 0),
        } for lang in langs},
        "label_agreement":  {f"{lang}__{cell}": cross[(lang, cell)] for lang, cell in cross},
    }
    audit_path = Path(args.audit_dir) / "synth_separation_qa.json"
    with open(audit_path, "w") as f:
        json.dump(audit, f, indent=2)
    print(f"\n  QA report saved to {audit_path}")


if __name__ == "__main__":
    main()
