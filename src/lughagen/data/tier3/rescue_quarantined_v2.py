"""
rescue_quarantined_v2.py — LughaGen Tier 2 Quarantine Rescue (TIGHTENED)

v2 changes based on rescue audit findings:
  1. MINIMUM_LENGTH = 40 chars (was 20) — fixes "very_short" rescues
     dominating Swahili (50.6%) and Kamba (24%) buckets.
  2. afrolid_confirmed_target requires marker_density ≥ 0.05 corroboration
     — catches AfroLID Bantu-bias false positives without dropping the signal.
  3. CORRUPTION FILTER — reject samples with non-orthographic characters
     (ʉ, ɔ, ą, ć, ĉ, ŋ outside Luo) that aren't valid in any target language.
  4. marker_density signal threshold raised slightly for safety.

UNCHANGED from v1:
  - glotlid_strict_target — works, 100% cross-validation on Luo
  - glotlid_top3_target  — works, no major issues seen in audit
  - marker_density signal — works, 99.4-99.7% cross-validation rate

OUTPUTS (per language, written to /workspace/synthetic/tier2/):
  <lang>_rescued_v2.jsonl
  <lang>_final_rejected_v2.jsonl
  <lang>_final_dataset_v2.jsonl
  rescue_report_v2.json
"""

import json
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

# ── PATHS ────────────────────────────────────────────────────────
TIER2_DIR = Path("/workspace/synthetic/tier2")
RESCUE_REPORT = TIER2_DIR / "rescue_report_v2.json"

LANGUAGES = ["swahili", "kikuyu", "kamba", "luo"]

# ── TIGHTENED THRESHOLDS ─────────────────────────────────────────
MINIMUM_LENGTH = 40            # was 20 in v1
AFROLID_MIN_PROB = 0.60
AFROLID_REQUIRES_MARKER_DENSITY = 0.05   # NEW in v2

# ── PER-LANGUAGE CRITERIA ────────────────────────────────────────

GLOTLID_TARGET = {
    "swahili": {"swh_Latn", "swc_Latn"},
    "kikuyu":  {"kik_Latn"},
    "kamba":   {"kam_Latn"},
    "luo":     {"luo_Latn", "lwo_Latn", "ach_Latn", "adh_Latn", "laj_Latn", "alz_Latn"},
}

AFROLID_TARGET = {
    "swahili": {"swh", "swa", "swc", "swahili", "swahili_congo"},
    "kikuyu":  {"kik", "kikuyu", "gikuyu"},
    "kamba":   {"kam", "kamba"},
    "luo":     {"luo", "lwo", "dholuo", "ach", "acholi", "adh", "adhola", "laj", "lango", "alz", "alur"},
}

TARGET_DIACRITICS = {
    "kikuyu":  re.compile(r"[ĩũĨŨ]"),
    "kamba":   re.compile(r"[ĩũĨŨ]"),
    "swahili": None,
    "luo":     None,
}

TARGET_FUNCTION_WORDS = {
    "swahili": {"ni", "wa", "ya", "na", "kwa", "kuwa", "katika", "hii", "yake",
                "yangu", "kwamba", "lakini", "sasa", "hapa", "hivyo", "mimi", "wewe"},
    "kikuyu":  {"nĩ", "wa", "ya", "kana", "no", "atĩa", "atĩ", "ũrĩa", "kĩu",
                "thĩnĩ", "kũrĩa", "rĩu", "ngai", "mũtumia", "mũndũ"},
    "kamba":   {"nĩ", "wa", "ya", "kana", "no", "ata", "atĩa", "ũndũ", "kĩu",
                "thĩnĩ", "kũũ", "ĩsu", "mũndũ", "andũ", "mwailu", "ngai"},
    "luo":     {"en", "mar", "gi", "kod", "kanyo", "piny", "ne", "gima", "nyaka",
                "kuom", "kata", "moko", "miyo", "tiende", "ka", "ko", "machielo"},
}

# Non-orthographic characters that indicate encoding corruption or non-target text
# Note: ĩũ are valid for Kikuyu/Kamba; ʉ ɔ ą ć ĉ ę ǫ ɛ are NOT valid in our targets
CORRUPTION_CHARS = re.compile(r"[ʉɔąćĉęǫɛβγδéèáàíìóòúùåäöüâêîôûñ]")

PROMPT_TEMPLATE_PATTERNS = re.compile(
    r"(conversation\s*\d+|turn\s*\d+|user\s*:|assistant\s*:|system\s*:|"
    r"<\|.*?\|>|\[INST\]|\[\/INST\]|"
    r"role\s*:\s*(user|assistant)|"
    r'"speaker"\s*:|"turn"\s*:|"utterance"\s*:|'
    r"step\s*\d+\s*:|example\s*\d+\s*:)",
    re.IGNORECASE,
)

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


# ── METRICS ──────────────────────────────────────────────────────

def english_function_ratio(text: str) -> float:
    words = re.findall(r"\b[a-zA-Z]+\b", text.lower())
    if not words:
        return 0.0
    return sum(1 for w in words if w in ENGLISH_FUNCTION_WORDS) / len(words)


def has_prompt_template(text: str) -> bool:
    return bool(PROMPT_TEMPLATE_PATTERNS.search(text))


def has_corruption(text: str) -> bool:
    """True if text contains non-orthographic characters for the target languages."""
    return bool(CORRUPTION_CHARS.search(text))


def target_marker_density_per_token(text: str, lang: str) -> float:
    """Per-token density of target-language markers (diacritics + function words)."""
    tokens = re.findall(r"\S+", text.lower())
    if not tokens:
        return 0.0

    diacritic_pattern = TARGET_DIACRITICS.get(lang)
    diacritic_hits = sum(1 for t in tokens if diacritic_pattern and diacritic_pattern.search(t))

    fw = TARGET_FUNCTION_WORDS.get(lang, set())
    tokens_norm = [re.sub(r"[ĩ]", "i", re.sub(r"[ũ]", "u", t)) for t in tokens]
    fw_norm = {re.sub(r"[ĩ]", "i", re.sub(r"[ũ]", "u", w)) for w in fw}
    fw_hits = sum(1 for t in tokens_norm if t in fw_norm)

    return (diacritic_hits + fw_hits) / len(tokens)


def target_marker_score(text: str, lang: str) -> float:
    """Composite score (0..1) used for marker_density signal — kept same as v1."""
    score = 0.0
    diacritic_pattern = TARGET_DIACRITICS.get(lang)
    if diacritic_pattern:
        n_diacritics = len(diacritic_pattern.findall(text))
        if n_diacritics >= 1:
            score += 0.5
        if n_diacritics >= 3:
            score += 0.2

    fw = TARGET_FUNCTION_WORDS.get(lang, set())
    text_lower = text.lower()
    tokens = re.findall(r"\S+", text_lower)
    if tokens:
        tokens_norm = [re.sub(r"[ĩ]", "i", re.sub(r"[ũ]", "u", t)) for t in tokens]
        fw_norm = {re.sub(r"[ĩ]", "i", re.sub(r"[ũ]", "u", w)) for w in fw}
        hits = sum(1 for t in tokens_norm if t in fw_norm)
        density = hits / len(tokens)
        if density >= 0.05:
            score += 0.3
        if density >= 0.15:
            score += 0.2
    return min(score, 1.0)


def extract_top3_codes(rec: dict) -> set:
    codes = set()
    for entry in rec.get("audit_glotlid_top3", []) or []:
        if isinstance(entry, dict):
            codes.add(entry.get("lang"))
        elif isinstance(entry, (list, tuple)) and entry:
            codes.add(entry[0])
        elif isinstance(entry, str):
            codes.add(entry)
    return codes


# ── TIGHTENED CLASSIFICATION (v2) ────────────────────────────────

def reclassify_v2(rec: dict, lang: str) -> tuple:
    text = (rec.get("text") or rec.get("output") or rec.get("content") or "").strip()

    # ── HARD REJECTIONS ──────────────────────────────────────────
    if len(text) < MINIMUM_LENGTH:                              # v2: 40, was 20
        return "REJECT", "too_short"

    if has_prompt_template(text):
        return "REJECT", "prompt_leakage"

    if has_corruption(text):                                    # v2: NEW
        return "REJECT", "encoding_corruption"

    if english_function_ratio(text) > 0.70:
        return "REJECT", "drift_to_english"

    # ── ACCEPTANCE SIGNALS ───────────────────────────────────────
    gl_top1   = rec.get("audit_glotlid_top1") or ""
    gl_top1_p = rec.get("audit_glotlid_top1_p") or 0.0
    af_label  = rec.get("audit_afrolid") or ""
    af_prob   = rec.get("audit_afrolid_p") or 0.0
    gl_top3   = extract_top3_codes(rec)

    target_gl = GLOTLID_TARGET.get(lang, set())
    target_af = AFROLID_TARGET.get(lang, set())

    marker_density = target_marker_density_per_token(text, lang)

    # Signal 1: GlotLID top-1 = target (most reliable, no corroboration needed)
    if gl_top1 in target_gl and gl_top1_p >= 0.40:
        return "ACCEPT", "glotlid_strict_target"

    # Signal 2: Target in GlotLID top-3 (already implies some confidence)
    if target_gl & gl_top3:
        return "ACCEPT", "glotlid_top3_target"

    # Signal 3: AfroLID confirms target — v2: requires marker corroboration
    if af_label in target_af and af_prob >= AFROLID_MIN_PROB:
        if marker_density >= AFROLID_REQUIRES_MARKER_DENSITY:
            return "ACCEPT", "afrolid_confirmed_target"
        # AfroLID alone without marker corroboration is REJECTED
        # (catches Bantu→Swahili bias false positives)

    # Signal 4: Strong target-language markers
    composite_score = target_marker_score(text, lang)
    if composite_score >= 0.5:
        return "ACCEPT", "marker_density"

    # ── REJECTION CATEGORIES ─────────────────────────────────────
    if af_label in target_af and af_prob >= AFROLID_MIN_PROB and marker_density < AFROLID_REQUIRES_MARKER_DENSITY:
        return "REJECT", "afrolid_only_no_markers"   # v2: NEW — caught by tightening

    if gl_top1 in {"swh_Latn", "swc_Latn"} and lang != "swahili":
        return "REJECT", "drift_to_swahili"
    if gl_top1 == "eng_Latn":
        return "REJECT", "drift_to_english"
    if gl_top1 in {"fra_Latn", "spa_Latn", "por_Latn", "deu_Latn", "ita_Latn",
                   "dan_Latn", "nld_Latn", "swe_Latn", "nor_Latn"}:
        return "REJECT", "drift_to_european"
    if gl_top1 and gl_top1.endswith("_Latn"):
        return "REJECT", "drift_to_other_african"

    return "REJECT", "no_target_signal"


# ── PROCESSING ───────────────────────────────────────────────────

def process_language(lang: str) -> dict:
    quarantine_path = TIER2_DIR / f"{lang}_audited_quarantined.jsonl"
    passed_path     = TIER2_DIR / f"{lang}_audited_passed.jsonl"

    rescued_path        = TIER2_DIR / f"{lang}_rescued_v2.jsonl"
    final_rejected_path = TIER2_DIR / f"{lang}_final_rejected_v2.jsonl"
    final_dataset_path  = TIER2_DIR / f"{lang}_final_dataset_v2.jsonl"

    if not quarantine_path.exists():
        return {"missing": True, "language": lang}

    print(f"\n  Processing {lang}...", flush=True)

    n_rescued = 0
    n_still_rejected = 0
    accept_reason_counts = Counter()
    reject_reason_counts = Counter()

    with open(quarantine_path, "r", encoding="utf-8") as fin, \
         open(rescued_path, "w", encoding="utf-8") as fout_rescue, \
         open(final_rejected_path, "w", encoding="utf-8") as fout_reject:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            decision, reason = reclassify_v2(rec, lang)
            rec["rescue_decision"] = decision
            rec["rescue_signal"]   = reason
            rec["rescue_version"]  = "v2"

            if decision == "ACCEPT":
                fout_rescue.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_rescued += 1
                accept_reason_counts[reason] += 1
            else:
                fout_reject.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_still_rejected += 1
                reject_reason_counts[reason] += 1

    # Assemble final dataset
    n_original_passed = 0
    with open(final_dataset_path, "w", encoding="utf-8") as fout_final:
        if passed_path.exists():
            with open(passed_path, "r", encoding="utf-8") as fin:
                for line in fin:
                    if line.strip():
                        fout_final.write(line if line.endswith("\n") else line + "\n")
                        n_original_passed += 1
        with open(rescued_path, "r", encoding="utf-8") as fin:
            for line in fin:
                if line.strip():
                    fout_final.write(line if line.endswith("\n") else line + "\n")

    n_final = n_original_passed + n_rescued

    return {
        "language":              lang,
        "n_original_passed":     n_original_passed,
        "n_quarantined":         n_rescued + n_still_rejected,
        "n_rescued":             n_rescued,
        "n_still_rejected":      n_still_rejected,
        "n_final_dataset":       n_final,
        "rescue_rate_pct":       round(n_rescued / (n_rescued + n_still_rejected) * 100, 2) if (n_rescued + n_still_rejected) else 0,
        "accept_signal_counts":  dict(accept_reason_counts),
        "reject_reason_counts":  dict(reject_reason_counts),
        "files": {
            "rescued":         str(rescued_path),
            "final_rejected":  str(final_rejected_path),
            "final_dataset":   str(final_dataset_path),
        },
    }


def main():
    if not TIER2_DIR.exists():
        print(f"✗ {TIER2_DIR} not found", file=sys.stderr)
        sys.exit(1)

    print("═" * 70)
    print("  TIER 2 QUARANTINE RESCUE — v2 (TIGHTENED)")
    print("═" * 70)
    print(f"  Minimum length: {MINIMUM_LENGTH} chars (was 20 in v1)")
    print(f"  AfroLID requires marker_density ≥ {AFROLID_REQUIRES_MARKER_DENSITY}")
    print(f"  Encoding corruption filter: ENABLED")
    print("═" * 70)

    results = {}
    for lang in LANGUAGES:
        results[lang] = process_language(lang)

    report = {
        "version":      "rescue_v2.0_tightened",
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "thresholds": {
            "minimum_length":              MINIMUM_LENGTH,
            "afrolid_min_prob":            AFROLID_MIN_PROB,
            "afrolid_requires_marker":     AFROLID_REQUIRES_MARKER_DENSITY,
            "corruption_filter":           True,
        },
        "per_language": results,
    }
    RESCUE_REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    # Console summary
    print("\n" + "═" * 70)
    print("  RESCUE v2 SUMMARY")
    print("═" * 70)
    print(f"\n  {'Language':<10} {'Orig Pass':>10} {'Quarant':>9} {'Rescued':>9} {'Rejected':>9} {'Final':>10}")
    print("  " + "─" * 67)
    total_final = 0
    total_rescued = 0
    total_quarantined = 0
    for lang in LANGUAGES:
        r = results[lang]
        if r.get("missing"):
            continue
        print(f"  {lang.upper():<10} {r['n_original_passed']:>10,} {r['n_quarantined']:>9,} "
              f"{r['n_rescued']:>9,} {r['n_still_rejected']:>9,} {r['n_final_dataset']:>10,}")
        total_final += r["n_final_dataset"]
        total_rescued += r["n_rescued"]
        total_quarantined += r["n_quarantined"]
    print("  " + "─" * 67)
    print(f"  {'TOTAL':<10} {'':<10} {total_quarantined:>9,} {total_rescued:>9,} "
          f"{total_quarantined - total_rescued:>9,} {total_final:>10,}")

    print("\n  Accept signals (after tightening):")
    for lang in LANGUAGES:
        r = results[lang]
        if r.get("missing") or not r["accept_signal_counts"]:
            continue
        print(f"\n    {lang.upper()}:")
        for sig, count in sorted(r["accept_signal_counts"].items(), key=lambda x: -x[1]):
            pct = count / r["n_rescued"] * 100 if r["n_rescued"] else 0
            print(f"      {sig:<30} {count:>7,}  ({pct:5.1f}%)")

    print("\n  Top rejection reasons:")
    for lang in LANGUAGES:
        r = results[lang]
        if r.get("missing") or not r["reject_reason_counts"]:
            continue
        print(f"\n    {lang.upper()}:")
        for reason, count in sorted(r["reject_reason_counts"].items(), key=lambda x: -x[1])[:6]:
            pct = count / r["n_still_rejected"] * 100 if r["n_still_rejected"] else 0
            print(f"      {reason:<30} {count:>7,}  ({pct:5.1f}%)")

    print("\n" + "═" * 70)
    print(f"\n  Outputs in {TIER2_DIR}:")
    for lang in LANGUAGES:
        r = results[lang]
        if r.get("missing"):
            continue
        print(f"    {lang}_rescued_v2.jsonl         ({r['n_rescued']:,})")
        print(f"    {lang}_final_rejected_v2.jsonl  ({r['n_still_rejected']:,})")
        print(f"    {lang}_final_dataset_v2.jsonl   ({r['n_final_dataset']:,})  ← for Tier 3")
    print(f"\n  Report: {RESCUE_REPORT}")
    print("═" * 70 + "\n")
    print("  Next: re-audit the v2 rescues:")
    print("    Edit audit_rescued.py rescued_path to use _rescued_v2.jsonl")
    print("    Or use the v2 audit installer if provided\n")


if __name__ == "__main__":
    main()
