"""
analyze_quarantine.py — LughaGen Tier 2 Quarantine Assessment

Categorizes all quarantined samples across Swahili, Kikuyu, Kamba, Luo to
understand failure modes and identify potentially salvageable data.

Categorization rules (per sample, first match wins):
  1. EMPTY            — text length < 20 chars
  2. PROMPT_LEAKAGE   — generator template artifacts ("conversation N", "turn N",
                        "user:", "assistant:", "<|...|>", etc)
  3. ENGLISH_HEAVY    — >40% of words are common English function words
  4. CODE_SWITCHED    — has English words AND target-language diacritics/markers
  5. WRONG_AFRICAN    — GlotLID predicted a related African language (top-1)
  6. LOW_CONFIDENCE   — GlotLID top-1 probability < 0.50
  7. UNCLASSIFIED     — everything else

Outputs:
  /workspace/synthetic/tier2/quarantine_assessment.json   — full breakdown
  /workspace/synthetic/tier2/quarantine_samples_<lang>.txt — examples per category
  Console summary table
"""

import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

try:
    import jsonlines
    _HAS_JSONLINES = True
except ImportError:
    _HAS_JSONLINES = False

# ── PATHS ────────────────────────────────────────────────────────
TIER2_DIR = Path("/workspace/synthetic/tier2")
QUARANTINE_FILES = {
    "swahili": TIER2_DIR / "swahili_audited_quarantined.jsonl",
    "kikuyu":  TIER2_DIR / "kikuyu_audited_quarantined.jsonl",
    "kamba":   TIER2_DIR / "kamba_audited_quarantined.jsonl",
    "luo":     TIER2_DIR / "luo_audited_quarantined.jsonl",
}
ASSESSMENT_REPORT = TIER2_DIR / "quarantine_assessment.json"

# ── CATEGORIZATION HEURISTICS ────────────────────────────────────

# Common English function words (top frequency)
ENGLISH_FUNCTION_WORDS = set("""
the and or but to of in for on at with from by is are was were be been being
this that these those i you he she it we they me him her us them my your his
hers its our their have has had do does did will would could should may might
can must shall not no yes if then else when where how why what who which
about above after again against all am an any as because before below
between both each few further here however into more most no nor now off
once only other out over own same so some such than then there through
under until up very what when while with again very just only also already
""".split())

# Language-specific diacritic / marker patterns
LANG_MARKERS = {
    "swahili": re.compile(r"\b(ni|wa|ya|na|kwa|kuwa|katika|hii|yake|yangu)\b", re.I),
    "kikuyu":  re.compile(r"[ĩũĨŨ]"),
    "kamba":   re.compile(r"[ĩũĨŨ]"),
    "luo":     re.compile(r"\b(en|mar|gi|kod|kanyo|piny|ne|gima)\b", re.I),
}

# Prompt template leakage patterns
PROMPT_PATTERNS = re.compile(
    r"(conversation\s*\d+|turn\s*\d+|user\s*:|assistant\s*:|system\s*:|"
    r"prompt\s*:|response\s*:|<\|.*?\|>|\[INST\]|\[\/INST\]|"
    r"role\s*:\s*(user|assistant)|q\d+\s*:|a\d+\s*:|"
    r"step\s*\d+\s*:|example\s*\d+\s*:)",
    re.IGNORECASE,
)

# African language codes used by GlotLID
AFRICAN_LANG_CODES = {
    "swh_Latn", "swc_Latn",  # Swahili
    "kik_Latn",              # Kikuyu
    "kam_Latn",              # Kamba
    "luo_Latn", "lwo_Latn",  # Luo
    "kin_Latn", "kon_Latn", "lin_Latn", "lug_Latn", "nya_Latn",
    "sna_Latn", "tsn_Latn", "xho_Latn", "yor_Latn", "zul_Latn",
    "hau_Latn", "ibo_Latn", "amh_Ethi", "som_Latn", "tir_Ethi",
    "ful_Latn", "wol_Latn", "ssw_Latn", "tum_Latn", "nyu_Latn",
    "mws_Latn", "ngo_Latn", "zne_Latn", "rwk_Latn", "nim_Latn",
    "dig_Latn", "asa_Latn", "sxb_Latn", "jam_Latn",
}


def english_ratio(text: str) -> float:
    """Fraction of words that are common English function words."""
    words = re.findall(r"\b[a-zA-Z]+\b", text.lower())
    if not words:
        return 0.0
    eng_hits = sum(1 for w in words if w in ENGLISH_FUNCTION_WORDS)
    return eng_hits / len(words)


def has_target_markers(text: str, lang: str) -> bool:
    pattern = LANG_MARKERS.get(lang)
    return bool(pattern.search(text)) if pattern else False


def categorize(text: str, lang: str, glotlid_top1: str, glotlid_prob: float) -> str:
    text = (text or "").strip()

    # 1. EMPTY
    if len(text) < 20:
        return "EMPTY"

    # 2. PROMPT_LEAKAGE
    if PROMPT_PATTERNS.search(text):
        return "PROMPT_LEAKAGE"

    # 3. ENGLISH_HEAVY
    eng_pct = english_ratio(text)
    if eng_pct > 0.40:
        return "ENGLISH_HEAVY"

    # 4. CODE_SWITCHED
    if eng_pct > 0.15 and has_target_markers(text, lang):
        return "CODE_SWITCHED"

    # 5. WRONG_AFRICAN — GlotLID confidently picked another African language
    if glotlid_top1 in AFRICAN_LANG_CODES and glotlid_top1 not in {
        f"{lang[:3]}_Latn", "swh_Latn", "kik_Latn", "kam_Latn", "luo_Latn"
    }:
        if glotlid_prob >= 0.50:
            return "WRONG_AFRICAN"

    # 6. LOW_CONFIDENCE
    if glotlid_prob < 0.50:
        return "LOW_CONFIDENCE"

    # 7. UNCLASSIFIED
    return "UNCLASSIFIED"


# ── MAIN ─────────────────────────────────────────────────────────

def analyze_language(lang: str, path: Path) -> dict:
    if not path.exists():
        return {"missing": True, "path": str(path)}

    print(f"\n  Analyzing {lang}...", flush=True)

    category_counts = Counter()
    category_examples = defaultdict(list)
    glotlid_pred_counts = Counter()
    afrolid_pred_counts = Counter()
    length_buckets = Counter()
    salvage_candidates = []  # high-quality samples that might be recoverable

    n_total = 0
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            n_total += 1
            text = rec.get("text") or rec.get("output") or rec.get("content") or ""
            gl_top1 = rec.get("audit_glotlid_top1") or ""
            gl_prob = rec.get("audit_glotlid_top1_p") or 0.0
            af_label = rec.get("audit_afrolid") or ""

            cat = categorize(text, lang, gl_top1, gl_prob)
            category_counts[cat] += 1

            # Keep up to 5 examples per category
            if len(category_examples[cat]) < 5:
                category_examples[cat].append({
                    "text": text[:200],
                    "glotlid_top1": gl_top1,
                    "glotlid_prob": gl_prob,
                    "afrolid": af_label,
                })

            glotlid_pred_counts[gl_top1] += 1
            afrolid_pred_counts[af_label] += 1

            # Length distribution
            L = len(text)
            if L < 20:        bucket = "<20"
            elif L < 50:      bucket = "20-50"
            elif L < 100:     bucket = "50-100"
            elif L < 200:     bucket = "100-200"
            else:             bucket = ">200"
            length_buckets[bucket] += 1

            # Salvage candidates: high GlotLID confidence in target language
            # but quarantined for other reasons (shouldn't really happen with
            # GlotLID-authoritative gating but check anyway)
            target_codes = {"swahili": "swh_Latn", "kikuyu": "kik_Latn",
                            "kamba": "kam_Latn", "luo": "luo_Latn"}
            if gl_top1 == target_codes.get(lang) and gl_prob >= 0.80:
                if len(salvage_candidates) < 20:
                    salvage_candidates.append({
                        "text": text[:200],
                        "glotlid_prob": gl_prob,
                    })

    return {
        "n_quarantined":     n_total,
        "category_counts":   dict(category_counts),
        "category_examples": dict(category_examples),
        "glotlid_top10":     dict(glotlid_pred_counts.most_common(10)),
        "afrolid_top10":     dict(afrolid_pred_counts.most_common(10)),
        "length_distribution": dict(length_buckets),
        "salvage_candidates_count": len(salvage_candidates),
        "salvage_candidates": salvage_candidates,
    }


def write_examples_file(lang: str, result: dict):
    """Human-readable examples file per language."""
    path = TIER2_DIR / f"quarantine_samples_{lang}.txt"
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# Quarantine Examples — {lang.upper()}\n")
        f.write(f"# n_quarantined: {result['n_quarantined']}\n")
        f.write(f"# Generated: {datetime.utcnow().isoformat()}Z\n\n")

        for cat, examples in result["category_examples"].items():
            count = result["category_counts"].get(cat, 0)
            pct = count / result["n_quarantined"] * 100 if result["n_quarantined"] else 0
            f.write(f"\n{'='*70}\n")
            f.write(f"  {cat}  ({count:,} samples, {pct:.1f}% of quarantined)\n")
            f.write(f"{'='*70}\n\n")
            for i, ex in enumerate(examples, 1):
                f.write(f"  [{i}] gl={ex['glotlid_top1']}/{ex['glotlid_prob']:.2f}  af={ex['afrolid']}\n")
                f.write(f"      {ex['text']}\n\n")
    return path


def main():
    if not TIER2_DIR.exists():
        print(f"✗ {TIER2_DIR} not found")
        sys.exit(1)

    print("═" * 70)
    print("  TIER 2 QUARANTINE ASSESSMENT")
    print("═" * 70)

    results = {}
    examples_files = []
    for lang, path in QUARANTINE_FILES.items():
        r = analyze_language(lang, path)
        results[lang] = r
        if not r.get("missing"):
            examples_files.append(str(write_examples_file(lang, r)))

    # Write JSON report
    report = {
        "version":       "quarantine_assessment_v1.0",
        "generated_at":  datetime.utcnow().isoformat() + "Z",
        "per_language":  results,
        "examples_files": examples_files,
    }
    ASSESSMENT_REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    # ── Console summary ──
    print("\n" + "═" * 70)
    print("  ASSESSMENT SUMMARY")
    print("═" * 70)

    # Aggregate category counts across languages
    all_cats = Counter()
    for lang, r in results.items():
        if r.get("missing"):
            continue
        for cat, count in r["category_counts"].items():
            all_cats[(lang, cat)] = count

    # Print as table
    categories = ["EMPTY", "PROMPT_LEAKAGE", "ENGLISH_HEAVY", "CODE_SWITCHED",
                  "WRONG_AFRICAN", "LOW_CONFIDENCE", "UNCLASSIFIED"]
    print(f"\n  {'Category':<18} {'Swahili':>10} {'Kikuyu':>10} {'Kamba':>10} {'Luo':>10}  {'Total':>10}")
    print("  " + "─" * 76)
    for cat in categories:
        row_total = 0
        row = f"  {cat:<18}"
        for lang in ["swahili", "kikuyu", "kamba", "luo"]:
            count = results.get(lang, {}).get("category_counts", {}).get(cat, 0)
            row += f" {count:>10,}"
            row_total += count
        row += f"  {row_total:>10,}"
        print(row)

    # Per-language summary
    print("\n  Per-language summary:")
    for lang, r in results.items():
        if r.get("missing"):
            print(f"    {lang.upper():<10} MISSING — {r['path']}")
            continue
        n = r["n_quarantined"]
        top_cat = max(r["category_counts"].items(), key=lambda x: x[1])
        print(f"    {lang.upper():<10} n={n:>6,}   top failure: {top_cat[0]} ({top_cat[1]:,}, {top_cat[1]/n*100:.1f}%)")
        if r["salvage_candidates_count"]:
            print(f"               ⚠ {r['salvage_candidates_count']} possible salvage candidates (high GlotLID prob)")

    print("\n" + "═" * 70)
    print(f"\n  Full report:    {ASSESSMENT_REPORT}")
    print(f"  Examples:       {TIER2_DIR}/quarantine_samples_<lang>.txt")
    print("═" * 70 + "\n")


if __name__ == "__main__":
    main()
