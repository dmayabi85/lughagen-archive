"""
audit_categories_v2.py — LughaGen Tier 2 Detailed Category Audit (v2)

Uses the ACTUAL prompt_style values from the data (Tier 2 v2.0 curriculum),
not the Tier 1 styles I previously assumed.

For each language final dataset, this reports:

  1. Distribution per prompt_style (count + %)
  2. The "code_switching" prompt style vs the DERIVED code-switching detection
     — i.e. "prompted to code-switch" vs "actually contains code-switching"
  3. Quality, length, English ratio, marker density per style
  4. Cross-language style availability matrix
  5. Examples per style per language

Inputs:
  /workspace/synthetic/tier2/<lang>_final_dataset_v2.jsonl

Outputs:
  /workspace/synthetic/tier2/category_audit_v2_report.json
  /workspace/synthetic/tier2/category_audit_v2_overall.txt
  /workspace/synthetic/tier2/category_audit_v2_<lang>.txt
"""

import json
import re
import sys
import statistics
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

TIER2_DIR = Path("/workspace/synthetic/tier2")
REPORT_PATH = TIER2_DIR / "category_audit_v2_report.json"
LANGUAGES = ["swahili", "kikuyu", "kamba", "luo"]

# Documented curriculum target (from Tier 2 v2.0 design doc)
DOCUMENTED_TARGET = {
    "context_expansion":   0.30,
    "dialogue_conversion": 0.25,
    "code_switching":      0.25,
    "noise_robust_rewrite": 0.20,
    # Styles added beyond original spec:
    # - domain_specific (all langs)
    # - natural_swahili (Swahili only)
}

# Style display order (will be augmented by discovered styles)
PREFERRED_STYLE_ORDER = [
    "context_expansion",
    "dialogue_conversion",
    "code_switching",
    "noise_robust_rewrite",
    "domain_specific",
    "natural_swahili",
]

CODESWITCH_ENG_RATIO_MIN = 0.02   # ≥2% English function words → code-switched

# Language markers
LANG_DIACRITICS = {
    "kikuyu":  re.compile(r"[ĩũĨŨ]"),
    "kamba":   re.compile(r"[ĩũĨŨ]"),
    "swahili": None,
    "luo":     None,
}
LANG_FUNCTION_WORDS = {
    "swahili": {"ni", "wa", "ya", "na", "kwa", "kuwa", "katika", "hii", "yake",
                "yangu", "kwamba", "lakini", "sasa", "hapa", "hivyo", "mimi", "wewe"},
    "kikuyu":  {"nĩ", "wa", "ya", "kana", "no", "atĩa", "atĩ", "ũrĩa", "kĩu",
                "thĩnĩ", "kũrĩa", "rĩu", "ngai", "mũtumia", "mũndũ"},
    "kamba":   {"nĩ", "wa", "ya", "kana", "no", "ata", "atĩa", "ũndũ", "kĩu",
                "thĩnĩ", "kũũ", "ĩsu", "mũndũ", "andũ", "mwailu", "ngai"},
    "luo":     {"en", "mar", "gi", "kod", "kanyo", "piny", "ne", "gima", "nyaka",
                "kuom", "kata", "moko", "miyo", "tiende", "ka", "ko", "machielo"},
}
ENGLISH_FUNCTION_WORDS = set("""
the and or but to of in for on at with from by is are was were be been being
this that these those i you he she it we they me him her us them my your his
hers its our their have has had do does did will would could should may might
can must shall not no yes if then else when where how why what who which
""".split())


# ── METRICS ──────────────────────────────────────────────────────

def english_function_ratio(text: str) -> float:
    words = re.findall(r"\b[a-zA-Z]+\b", text.lower())
    if not words:
        return 0.0
    return sum(1 for w in words if w in ENGLISH_FUNCTION_WORDS) / len(words)


def marker_density(text: str, lang: str) -> float:
    tokens = re.findall(r"\S+", text.lower())
    if not tokens:
        return 0.0
    diacritic_pat = LANG_DIACRITICS.get(lang)
    diacritic_hits = sum(1 for t in tokens if diacritic_pat and diacritic_pat.search(t))
    fw = LANG_FUNCTION_WORDS.get(lang, set())
    tokens_norm = [re.sub(r"[ĩ]", "i", re.sub(r"[ũ]", "u", t)) for t in tokens]
    fw_norm = {re.sub(r"[ĩ]", "i", re.sub(r"[ũ]", "u", w)) for w in fw}
    fw_hits = sum(1 for t in tokens_norm if t in fw_norm)
    return (diacritic_hits + fw_hits) / len(tokens)


def is_code_switched(text: str) -> bool:
    return english_function_ratio(text) >= CODESWITCH_ENG_RATIO_MIN


# ── ACCUMULATOR ──────────────────────────────────────────────────

class StyleStats:
    def __init__(self):
        self.n = 0
        self.lengths = []
        self.qualities = []
        self.eng_ratios = []
        self.markers = []
        self.codeswitch_count = 0
        self.rescued_count = 0
        self.examples = []

    def add(self, rec, text, lang):
        self.n += 1
        L = len(text)
        self.lengths.append(L)

        q = rec.get("quality_score") or rec.get("quality")
        if q is not None:
            try:
                self.qualities.append(float(q))
            except (TypeError, ValueError):
                pass

        eng = english_function_ratio(text)
        self.eng_ratios.append(eng)
        self.markers.append(marker_density(text, lang))

        if eng >= CODESWITCH_ENG_RATIO_MIN:
            self.codeswitch_count += 1
        if rec.get("rescue_signal"):
            self.rescued_count += 1

        if len(self.examples) < 5 and L >= 50:
            self.examples.append({
                "text": text[:220],
                "english_ratio": round(eng, 4),
                "marker_density": round(marker_density(text, lang), 4),
                "quality": round(float(q), 3) if q else None,
                "is_code_switched": eng >= CODESWITCH_ENG_RATIO_MIN,
            })

    def summary(self) -> dict:
        result = {"n": self.n}
        if self.lengths:
            result["length_mean"] = round(statistics.mean(self.lengths), 1)
            result["length_p50"]  = int(statistics.median(self.lengths))
            result["length_p95"]  = sorted(self.lengths)[int(len(self.lengths) * 0.95)] if len(self.lengths) > 20 else None
        if self.qualities:
            result["quality_mean"] = round(statistics.mean(self.qualities), 4)
            result["quality_p50"]  = round(statistics.median(self.qualities), 4)
        else:
            result["quality_mean"] = None
        result["english_ratio_mean"]   = round(statistics.mean(self.eng_ratios), 4) if self.eng_ratios else 0
        result["marker_density_mean"]  = round(statistics.mean(self.markers), 4) if self.markers else 0
        result["codeswitched_count"]   = self.codeswitch_count
        result["codeswitched_pct_within"] = round(self.codeswitch_count / self.n * 100, 2) if self.n else 0
        result["rescued_count"]        = self.rescued_count
        result["rescued_pct_within"]   = round(self.rescued_count / self.n * 100, 2) if self.n else 0
        result["examples"]             = self.examples
        return result


# ── PROCESSING ───────────────────────────────────────────────────

def iter_jsonl(path: Path):
    if not path.exists():
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def analyze_language(lang: str) -> dict:
    print(f"\n  Analyzing {lang}...", flush=True)
    path = TIER2_DIR / f"{lang}_final_dataset_v2.jsonl"
    if not path.exists():
        return {"language": lang, "missing": True}

    per_style = defaultdict(StyleStats)
    derived_codeswitched = StyleStats()
    n_total = 0

    for rec in iter_jsonl(path):
        style = rec.get("prompt_style") or "UNKNOWN"
        text  = rec.get("text") or rec.get("output") or rec.get("content") or ""
        per_style[style].add(rec, text, lang)
        if is_code_switched(text):
            derived_codeswitched.add(rec, text, lang)
        n_total += 1
        if n_total % 25000 == 0:
            print(f"    ... {n_total:,} records processed", flush=True)

    print(f"    Done. Total: {n_total:,}", flush=True)

    style_summaries = {style: stats.summary() for style, stats in per_style.items()}

    style_distribution = {}
    for style, stats in per_style.items():
        style_distribution[style] = {
            "n":               stats.n,
            "pct_of_dataset":  round(stats.n / n_total * 100, 2) if n_total else 0,
            "documented_target_pct": round(DOCUMENTED_TARGET.get(style, 0) * 100, 1)
                                       if style in DOCUMENTED_TARGET else None,
        }

    return {
        "language":           lang,
        "n_total":            n_total,
        "styles_discovered":  sorted(per_style.keys()),
        "style_distribution": style_distribution,
        "style_summaries":    style_summaries,
        "derived_code_switching": {
            "n":              derived_codeswitched.n,
            "pct_of_dataset": round(derived_codeswitched.n / n_total * 100, 2) if n_total else 0,
            "stats":          derived_codeswitched.summary(),
        },
    }


def style_ordering(all_results: dict) -> list:
    """Build display order: preferred first, then any extras alphabetically."""
    all_styles = set()
    for r in all_results.values():
        if not r.get("missing"):
            all_styles.update(r["style_distribution"].keys())
    ordered = [s for s in PREFERRED_STYLE_ORDER if s in all_styles]
    extras = sorted(all_styles - set(ordered))
    return ordered + extras


# ── REPORTS ──────────────────────────────────────────────────────

def write_overall_report(all_results: dict):
    path = TIER2_DIR / "category_audit_v2_overall.txt"
    styles_ordered = style_ordering(all_results)

    with open(path, "w", encoding="utf-8") as f:
        f.write("# LughaGen Tier 2 — CATEGORY AUDIT (v2 — actual styles)\n")
        f.write(f"# Generated: {datetime.utcnow().isoformat()}Z\n\n")

        # MAIN TABLE
        f.write("=" * 110 + "\n")
        f.write("  PRIMARY TABLE — prompt_style distribution per language final dataset\n")
        f.write("=" * 110 + "\n\n")
        f.write(f"  {'Style':<24}")
        for lang in LANGUAGES:
            if not all_results.get(lang, {}).get("missing"):
                f.write(f" {lang.capitalize():>17}")
        f.write(f"  {'Target':>10}\n")
        f.write("  " + "-" * 100 + "\n")
        for style in styles_ordered:
            f.write(f"  {style:<24}")
            for lang in LANGUAGES:
                r = all_results.get(lang, {})
                if r.get("missing"):
                    continue
                dist = r["style_distribution"].get(style)
                if dist:
                    f.write(f" {dist['n']:>9,} ({dist['pct_of_dataset']:>4.1f}%)")
                else:
                    f.write(f" {'—':>17}")
            tgt = DOCUMENTED_TARGET.get(style)
            f.write(f"  {tgt*100:>9.1f}%" if tgt else f"  {'(added)':>10}")
            f.write("\n")

        # Derived code_switched row
        f.write("  " + "-" * 100 + "\n")
        f.write(f"  {'(derived) code_switched':<24}")
        for lang in LANGUAGES:
            r = all_results.get(lang, {})
            if r.get("missing"):
                continue
            cs = r["derived_code_switching"]
            f.write(f" {cs['n']:>9,} ({cs['pct_of_dataset']:>4.1f}%)")
        f.write(f"  {'derived':>10}\n")
        f.write("  " + "-" * 100 + "\n")

        # Total
        f.write(f"  {'TOTAL':<24}")
        for lang in LANGUAGES:
            r = all_results.get(lang, {})
            if r.get("missing"):
                continue
            f.write(f"  {r['n_total']:>15,}")
        f.write("\n\n")

        # KEY FINDING
        f.write("=" * 110 + "\n")
        f.write("  KEY FINDING — Prompted code-switching vs ACTUAL code-switching\n")
        f.write("=" * 110 + "\n\n")
        f.write(f"  {'Language':<14} {'Prompted CS %':>15} {'Actual CS %':>14} {'Δ (delivered - prompted)':>26}  {'Verdict':<40}\n")
        f.write("  " + "-" * 100 + "\n")
        for lang in LANGUAGES:
            r = all_results.get(lang, {})
            if r.get("missing"):
                continue
            prompted = r["style_distribution"].get("code_switching", {}).get("pct_of_dataset", 0)
            actual   = r["derived_code_switching"]["pct_of_dataset"]
            delta    = actual - prompted
            if abs(delta) < 2:
                verdict = "matches expectation"
            elif delta > 5:
                verdict = "OVER-DELIVERS — generator leaks CS into other styles"
            elif delta < -5:
                verdict = "UNDER-DELIVERS — generator reverts to monolingual"
            else:
                verdict = "slight drift"
            f.write(f"  {lang.capitalize():<14} {prompted:>14.2f}% {actual:>13.2f}% {delta:>+25.2f}pp  {verdict}\n")
        f.write("\n  Interpretation:\n")
        f.write("  • Prompted CS%: share of records whose prompt_style was 'code_switching'\n")
        f.write(f"  • Actual CS%:   share of records whose text contains ≥{CODESWITCH_ENG_RATIO_MIN*100:.0f}% English\n")
        f.write("    function words (a textual indicator of code-switching)\n\n")

        # QUALITY PER STYLE
        f.write("=" * 110 + "\n")
        f.write("  MEAN QUALITY SCORE PER STYLE (where quality_score is present)\n")
        f.write("=" * 110 + "\n\n")
        f.write(f"  {'Style':<24}")
        for lang in LANGUAGES:
            if not all_results.get(lang, {}).get("missing"):
                f.write(f" {lang.capitalize():>12}")
        f.write("\n")
        f.write("  " + "-" * 80 + "\n")
        for style in styles_ordered:
            f.write(f"  {style:<24}")
            for lang in LANGUAGES:
                r = all_results.get(lang, {})
                if r.get("missing"):
                    continue
                stats = r["style_summaries"].get(style, {})
                q = stats.get("quality_mean")
                f.write(f" {q:>12.4f}" if q is not None else f" {'n/a':>12}")
            f.write("\n")

        # CODE-SWITCHING WITHIN OTHER STYLES (interesting!)
        f.write("\n" + "=" * 110 + "\n")
        f.write("  CODE-SWITCHING WITHIN EACH PROMPT STYLE (what % of style's records are actually code-switched)\n")
        f.write("=" * 110 + "\n\n")
        f.write(f"  {'Style':<24}")
        for lang in LANGUAGES:
            if not all_results.get(lang, {}).get("missing"):
                f.write(f" {lang.capitalize():>12}")
        f.write("\n")
        f.write("  " + "-" * 80 + "\n")
        for style in styles_ordered:
            f.write(f"  {style:<24}")
            for lang in LANGUAGES:
                r = all_results.get(lang, {})
                if r.get("missing"):
                    continue
                stats = r["style_summaries"].get(style, {})
                if stats:
                    pct = stats.get("codeswitched_pct_within", 0)
                    f.write(f" {pct:>11.2f}%")
                else:
                    f.write(f" {'—':>12}")
            f.write("\n")

        # MEAN LENGTH
        f.write("\n" + "=" * 110 + "\n")
        f.write("  MEAN LENGTH (chars) PER STYLE\n")
        f.write("=" * 110 + "\n\n")
        f.write(f"  {'Style':<24}")
        for lang in LANGUAGES:
            if not all_results.get(lang, {}).get("missing"):
                f.write(f" {lang.capitalize():>12}")
        f.write("\n")
        f.write("  " + "-" * 80 + "\n")
        for style in styles_ordered:
            f.write(f"  {style:<24}")
            for lang in LANGUAGES:
                r = all_results.get(lang, {})
                if r.get("missing"):
                    continue
                stats = r["style_summaries"].get(style, {})
                L = stats.get("length_mean") if stats else None
                f.write(f" {L:>12.0f}" if L is not None else f" {'—':>12}")
            f.write("\n")

        # CROSS-LANGUAGE STYLE MATRIX
        f.write("\n" + "=" * 110 + "\n")
        f.write("  STYLE AVAILABILITY MATRIX (✓ = present in that language)\n")
        f.write("=" * 110 + "\n\n")
        f.write(f"  {'Style':<24}")
        for lang in LANGUAGES:
            if not all_results.get(lang, {}).get("missing"):
                f.write(f" {lang.capitalize():>12}")
        f.write("\n")
        f.write("  " + "-" * 80 + "\n")
        for style in styles_ordered:
            f.write(f"  {style:<24}")
            for lang in LANGUAGES:
                r = all_results.get(lang, {})
                if r.get("missing"):
                    continue
                present = style in r["style_distribution"]
                f.write(f" {'✓':>12}" if present else f" {'—':>12}")
            f.write("\n")

    return path


def write_per_language_report(lang: str, result: dict):
    path = TIER2_DIR / f"category_audit_v2_{lang}.txt"
    styles_in_lang = sorted(result["style_distribution"].keys(),
                            key=lambda s: -result["style_distribution"][s]["n"])

    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# Category Audit v2 — {lang.upper()}\n")
        f.write(f"# Generated: {datetime.utcnow().isoformat()}Z\n")
        f.write(f"# Total records: {result['n_total']:,}\n\n")

        # Distribution overview
        f.write("=" * 80 + "\n")
        f.write(f"  {lang.upper()} — STYLE DISTRIBUTION\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"  {'Style':<24} {'Count':>10} {'% Dataset':>12} {'Doc Target':>12}\n")
        f.write("  " + "-" * 60 + "\n")
        for style in styles_in_lang:
            dist = result["style_distribution"][style]
            target = DOCUMENTED_TARGET.get(style)
            target_str = f"{target*100:>11.1f}%" if target else "(added)".rjust(12)
            f.write(f"  {style:<24} {dist['n']:>10,} {dist['pct_of_dataset']:>11.2f}% {target_str}\n")
        f.write("  " + "-" * 60 + "\n")
        cs = result["derived_code_switching"]
        f.write(f"  {'(derived) code_switched':<24} {cs['n']:>10,} {cs['pct_of_dataset']:>11.2f}%  (overlaps above)\n")
        f.write("  " + "-" * 60 + "\n")
        f.write(f"  {'TOTAL':<24} {result['n_total']:>10,}\n\n")

        # Per-style detail
        for style in styles_in_lang:
            stats = result["style_summaries"][style]
            if stats["n"] == 0:
                continue
            f.write("=" * 80 + "\n")
            f.write(f"  STYLE: {style}\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"  Records:                  {stats['n']:,}\n")
            f.write(f"  % of dataset:             {result['style_distribution'][style]['pct_of_dataset']:.2f}%\n")
            tgt = DOCUMENTED_TARGET.get(style)
            if tgt:
                f.write(f"  Documented target:        {tgt*100:.1f}%\n")
                drift = result['style_distribution'][style]['pct_of_dataset'] - tgt*100
                f.write(f"  Drift from documented:    {drift:+.2f}pp\n")
            f.write(f"  Mean length (chars):      {stats.get('length_mean', 'n/a')}\n")
            f.write(f"  p50 / p95 length:         {stats.get('length_p50', 'n/a')} / {stats.get('length_p95', 'n/a')}\n")
            f.write(f"  Mean quality_score:       {stats.get('quality_mean', 'n/a')}\n")
            f.write(f"  Mean English ratio:       {stats.get('english_ratio_mean', 'n/a')}\n")
            f.write(f"  Mean marker density:      {stats.get('marker_density_mean', 'n/a')}\n")
            f.write(f"  Of these, code-switched:  {stats.get('codeswitched_count', 0):,}  ({stats.get('codeswitched_pct_within', 0):.2f}% within style)\n")
            f.write(f"  Of these, rescued via v2: {stats.get('rescued_count', 0):,}  ({stats.get('rescued_pct_within', 0):.2f}% within style)\n")

            if stats.get("examples"):
                f.write(f"\n  Examples:\n")
                for i, ex in enumerate(stats["examples"], 1):
                    cs_tag = " [CS]" if ex.get("is_code_switched") else ""
                    f.write(f"    [{i}]{cs_tag}  eng={ex['english_ratio']}  markers={ex['marker_density']}  q={ex.get('quality')}\n")
                    f.write(f"        {ex['text']}\n\n")
            f.write("\n")

        # Derived code-switching detail
        cs_stats = result["derived_code_switching"]["stats"]
        if cs_stats["n"]:
            f.write("=" * 80 + "\n")
            f.write(f"  DERIVED: code_switched records (≥{CODESWITCH_ENG_RATIO_MIN*100:.0f}% English function words)\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"  Records:                  {cs_stats['n']:,}\n")
            f.write(f"  % of dataset:             {result['derived_code_switching']['pct_of_dataset']:.2f}%\n")
            f.write(f"  Mean length (chars):      {cs_stats.get('length_mean', 'n/a')}\n")
            f.write(f"  Mean quality_score:       {cs_stats.get('quality_mean', 'n/a')}\n")
            f.write(f"  Mean English ratio:       {cs_stats.get('english_ratio_mean', 'n/a')}\n")
            f.write(f"  Mean marker density:      {cs_stats.get('marker_density_mean', 'n/a')}\n")

    return path


# ── MAIN ─────────────────────────────────────────────────────────

def main():
    if not TIER2_DIR.exists():
        print(f"✗ {TIER2_DIR} not found", file=sys.stderr)
        sys.exit(1)

    print("=" * 80)
    print("  TIER 2 CATEGORY AUDIT v2 (corrected — actual Tier 2 v2.0 styles)")
    print(f"  {datetime.utcnow().isoformat()}Z")
    print("=" * 80)

    all_results = {}
    for lang in LANGUAGES:
        all_results[lang] = analyze_language(lang)

    print("\n  Writing reports...")
    report = {
        "version":        "category_audit_v2.0",
        "generated_at":   datetime.utcnow().isoformat() + "Z",
        "documented_target": DOCUMENTED_TARGET,
        "codeswitch_threshold_english_ratio": CODESWITCH_ENG_RATIO_MIN,
        "per_language":   all_results,
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"  ✓ JSON: {REPORT_PATH}")
    overall = write_overall_report(all_results)
    print(f"  ✓ Overall: {overall}")
    for lang, r in all_results.items():
        if not r.get("missing"):
            per_lang = write_per_language_report(lang, r)
            print(f"  ✓ {lang}: {per_lang}")

    # ── Console summary ─────────────────────────────────────────
    styles_ordered = style_ordering(all_results)

    print("\n" + "=" * 110)
    print("  PRIMARY TABLE — prompt_style distribution per language final dataset")
    print("=" * 110)
    print(f"\n  {'Style':<24}", end="")
    for lang in LANGUAGES:
        if not all_results.get(lang, {}).get("missing"):
            print(f" {lang.capitalize():>17}", end="")
    print(f"  {'Target':>10}")
    print("  " + "-" * 100)

    for style in styles_ordered:
        print(f"  {style:<24}", end="")
        for lang in LANGUAGES:
            r = all_results.get(lang, {})
            if r.get("missing"):
                continue
            dist = r["style_distribution"].get(style)
            if dist:
                print(f" {dist['n']:>9,} ({dist['pct_of_dataset']:>4.1f}%)", end="")
            else:
                print(f" {'—':>17}", end="")
        tgt = DOCUMENTED_TARGET.get(style)
        print(f"  {tgt*100:>9.1f}%" if tgt else f"  {'(added)':>10}")

    print("  " + "-" * 100)
    print(f"  {'(derived) code_switched':<24}", end="")
    for lang in LANGUAGES:
        r = all_results.get(lang, {})
        if r.get("missing"):
            continue
        cs = r["derived_code_switching"]
        print(f" {cs['n']:>9,} ({cs['pct_of_dataset']:>4.1f}%)", end="")
    print(f"  {'derived':>10}")
    print("  " + "-" * 100)
    print(f"  {'TOTAL':<24}", end="")
    for lang in LANGUAGES:
        r = all_results.get(lang, {})
        if r.get("missing"):
            continue
        print(f"  {r['n_total']:>15,}", end="")
    print()

    # Key finding
    print("\n" + "=" * 110)
    print("  KEY FINDING — Prompted code-switching vs ACTUAL code-switching")
    print("=" * 110 + "\n")
    print(f"  {'Language':<14} {'Prompted CS %':>15} {'Actual CS %':>14} {'Δ (Actual - Prompted)':>26}  Verdict")
    print("  " + "-" * 100)
    for lang in LANGUAGES:
        r = all_results.get(lang, {})
        if r.get("missing"):
            continue
        prompted = r["style_distribution"].get("code_switching", {}).get("pct_of_dataset", 0)
        actual   = r["derived_code_switching"]["pct_of_dataset"]
        delta    = actual - prompted
        if abs(delta) < 2:
            verdict = "matches expectation"
        elif delta > 5:
            verdict = "OVER-DELIVERS — CS leaks into other styles"
        elif delta < -5:
            verdict = "UNDER-DELIVERS — generator reverts to monolingual"
        else:
            verdict = "slight drift"
        print(f"  {lang.capitalize():<14} {prompted:>14.2f}% {actual:>13.2f}% {delta:>+25.2f}pp  {verdict}")

    print("\n" + "=" * 110)
    print(f"\n  Outputs in {TIER2_DIR}:")
    print(f"    category_audit_v2_report.json")
    print(f"    category_audit_v2_overall.txt")
    print(f"    category_audit_v2_<lang>.txt")
    print("=" * 110 + "\n")


if __name__ == "__main__":
    main()
