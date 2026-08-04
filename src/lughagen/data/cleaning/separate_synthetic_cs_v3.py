"""
separate_synthetic_cs_v3.py
─────────────────────────────────────────────────────────────────────────
Separate synthetic LughaGen records into monolingual and code-switched
sub-files, using GlotLID confidence as the PRIMARY signal.

Rationale (informed by QA from v2): the tier 2 pipeline's CS label reflects
the *prompt* rather than the *output*, with ~95% of CS-labelled records
actually producing monolingual Swahili. GlotLID's per-record language-ID
confidence is a far more reliable signal: high confidence in the target
language indicates monolingual output; low confidence indicates code-
switching or language drift.

Routing rules per record:
  1. Compute target_match = (audit_glotlid_top1 maps to record language)
  2. p = audit_glotlid_top1_p (confidence)
  3. ratio = English-marker ratio (function-word fraction)

  If target_match AND p >= high_threshold AND ratio < ratio_threshold:
      → MONOLINGUAL (clean target-language record)
  Elif (not target_match) OR (p < low_threshold) OR (ratio >= ratio_threshold):
      → CODE-SWITCHED (drifted or genuinely mixed)
  Else:
      → AMBIGUOUS_MIDDLE (default: code-switched, since subtle CS is the
        most likely cause and this preserves it as a separate bucket)

Outputs:
  /workspace/data/corpus/staged/synth_mono/<language>_lughagen_synth.jsonl
  /workspace/data/corpus/staged/synth_cs/<language>_lughagen_synth_cs.jsonl
  /workspace/data/corpus/audit/synth_separation_v3_qa.json

Usage:
    python separate_synthetic_cs_v3.py \\
        --input /workspace/synthetic/tier3/fluency_scored.jsonl \\
        --output-dir /workspace/data/corpus/staged \\
        --audit-dir /workspace/data/corpus/audit \\
        --high-threshold 0.90 \\
        --low-threshold 0.70 \\
        --ratio-threshold 0.10
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path


EN_MARKERS = {
    "the","and","of","to","a","in","is","that","for","it","with","as","was",
    "on","are","this","by","be","at","an","have","from","but","what","when",
    "which","who","will","would","can","could","has","had","were","been",
    "they","their","there","not","or","also","because","about","into","than",
    "very","more","some","any","other","such","like","just","only","you",
    "your","we","our","my","me","he","she","his","her","them","us","if","so",
    "my","am","i",
}

# Map LughaGen language names to GlotLID ISO codes
LANG_TO_GLOT = {
    "swahili": {"swh_Latn", "swc_Latn"},
    "kikuyu":  {"kik_Latn"},
    "kamba":   {"kam_Latn"},
    "luo":     {"luo_Latn"},
}


def en_ratio(text):
    words = re.findall(r"[A-Za-z']+", text.lower())
    if not words:
        return 0.0
    return sum(1 for w in words if w in EN_MARKERS) / len(words)


def classify(record, lang, args):
    """Return (register, decision_reason)."""
    glot_top    = record.get("audit_glotlid_top1") or ""
    glot_p      = record.get("audit_glotlid_top1_p")
    if glot_p is None:
        glot_p = 0.0
    text        = record.get("text", "")
    ratio       = en_ratio(text)
    record["code_switch_ratio"] = round(ratio, 4)

    expected_codes = LANG_TO_GLOT.get(lang, set())
    target_match = glot_top in expected_codes

    # Decision tree
    if target_match and glot_p >= args.high_threshold and ratio < args.ratio_threshold:
        return "monolingual", "high_glotlid_low_ratio"

    if (not target_match) or glot_p < args.low_threshold or ratio >= args.ratio_threshold:
        if not target_match:
            return "code_switched", "glotlid_not_target"
        if glot_p < args.low_threshold:
            return "code_switched", "glotlid_low_confidence"
        return "code_switched", "high_english_ratio"

    # Ambiguous middle band — route to CS for safety (subtle CS most likely)
    return "code_switched", "middle_band_default_cs"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",          required=True)
    ap.add_argument("--output-dir",     required=True)
    ap.add_argument("--audit-dir",      required=True)
    ap.add_argument("--high-threshold", type=float, default=0.90,
                    help="GlotLID p above which (with target match, low ratio) record is mono")
    ap.add_argument("--low-threshold",  type=float, default=0.70,
                    help="GlotLID p below which record is CS regardless of ratio")
    ap.add_argument("--ratio-threshold", type=float, default=0.10,
                    help="English-marker ratio above which record is CS")
    ap.add_argument("--lang-field",     default="language")
    ap.add_argument("--text-field",     default="text")
    args = ap.parse_args()

    out_root = Path(args.output_dir)
    mono_dir = out_root / "synth_mono"; mono_dir.mkdir(parents=True, exist_ok=True)
    cs_dir   = out_root / "synth_cs";   cs_dir.mkdir(parents=True, exist_ok=True)
    Path(args.audit_dir).mkdir(parents=True, exist_ok=True)

    mono_handles, cs_handles = {}, {}
    def handle(handles, dir_, lang, suffix):
        if lang not in handles:
            handles[lang] = open(dir_ / f"{lang}_lughagen_synth{suffix}.jsonl", "w")
        return handles[lang]

    counts = Counter()         # (lang, register)
    reasons = Counter()        # (lang, register, decision_reason)
    pipeline_vs_v3 = Counter() # (lang, pipeline_label, v3_label)

    print(f"Reading {args.input}")
    print(f"Rules: target+p>={args.high_threshold} -> mono;  "
          f"!target | p<{args.low_threshold} | ratio>={args.ratio_threshold} -> CS;  "
          f"middle -> CS (subtle CS default)")

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

            register, reason = classify(obj, lang, args)
            obj["register"] = register
            obj["code_switch_pair"] = f"{lang}-english" if register == "code_switched" else None
            obj["v3_decision_reason"] = reason

            pipeline_label = "cs" if (
                "code" in str(obj.get("type", "")).lower() or
                "code" in str(obj.get("original_style", "")).lower()
            ) else "mono"
            pipeline_vs_v3[(lang, pipeline_label, register)] += 1

            counts[(lang, register)] += 1
            reasons[(lang, register, reason)] += 1

            if register == "monolingual":
                h = handle(mono_handles, mono_dir, lang, "")
            else:
                h = handle(cs_handles, cs_dir, lang, "_cs")
            h.write(json.dumps(obj, ensure_ascii=False) + "\n")

            if (i + 1) % 100000 == 0:
                print(f"  processed {i+1:,}")

    for h in list(mono_handles.values()) + list(cs_handles.values()):
        h.close()

    # ── Report ──────────────────────────────────────────────────────
    langs = sorted({k[0] for k in counts})

    print("\n" + "=" * 76)
    print("  v3 separation summary (GlotLID + ratio, pipeline label ignored)")
    print("=" * 76)
    print(f"  {'Language':<10}{'Monolingual':>14}{'Code-switched':>16}{'Total':>14}{'CS %':>10}")
    print("  " + "-" * 64)
    gm = gc = 0
    for lang in langs:
        m = counts.get((lang, "monolingual"), 0)
        c = counts.get((lang, "code_switched"), 0)
        gm += m; gc += c
        pct = 100 * c / (m + c) if (m + c) else 0
        print(f"  {lang:<10}{m:>14,}{c:>16,}{m+c:>14,}{pct:>9.1f}%")
    tot = gm + gc
    print("  " + "-" * 64)
    print(f"  {'TOTAL':<10}{gm:>14,}{gc:>16,}{tot:>14,}{100*gc/tot if tot else 0:>9.1f}%")

    print("\n  Decision-reason breakdown (CS records only)")
    print("  " + "-" * 72)
    for lang in langs:
        for reason in ["glotlid_not_target", "glotlid_low_confidence",
                       "high_english_ratio", "middle_band_default_cs"]:
            n = reasons.get((lang, "code_switched", reason), 0)
            if n:
                print(f"  {lang:<10}{reason:<30}{n:>12,}")

    print("\n  Pipeline label vs v3 decision (confusion-matrix style)")
    print("  " + "-" * 72)
    print(f"  {'Language':<10}{'pipeline=cs->v3=mono':>26}{'pipeline=mono->v3=cs':>26}{'agreements':>16}")
    for lang in langs:
        flip_cs_to_mono = pipeline_vs_v3.get((lang, "cs", "monolingual"), 0)
        flip_mono_to_cs = pipeline_vs_v3.get((lang, "mono", "code_switched"), 0)
        agree = (pipeline_vs_v3.get((lang, "cs", "code_switched"), 0)
                 + pipeline_vs_v3.get((lang, "mono", "monolingual"), 0))
        print(f"  {lang:<10}{flip_cs_to_mono:>26,}{flip_mono_to_cs:>26,}{agree:>16,}")

    audit = {
        "input": args.input,
        "thresholds": {
            "high_glotlid_p":  args.high_threshold,
            "low_glotlid_p":   args.low_threshold,
            "en_ratio":        args.ratio_threshold,
        },
        "totals": {
            "monolingual":   gm,
            "code_switched": gc,
            "all":           tot,
        },
        "per_language": {
            lang: {
                "monolingual":   counts.get((lang, "monolingual"), 0),
                "code_switched": counts.get((lang, "code_switched"), 0),
            } for lang in langs
        },
        "decision_reasons": {
            f"{lang}__{reg}__{r}": n for (lang, reg, r), n in reasons.items()
        },
        "pipeline_vs_v3_flips": {
            f"{lang}__{p}__{v}": n for (lang, p, v), n in pipeline_vs_v3.items()
        },
    }
    out_path = Path(args.audit_dir) / "synth_separation_v3_qa.json"
    with open(out_path, "w") as f:
        json.dump(audit, f, indent=2)
    print(f"\n  QA report saved to {out_path}")
    print("=" * 76)


if __name__ == "__main__":
    main()
