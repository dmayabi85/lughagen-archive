"""
audit_categories.py — LughaGen Tier 2 Detailed Category Audit

The user-requested view: for each language's final dataset, what % is each
of the six categories?

  1. paraphrase_simple
  2. paraphrase_formal
  3. spoken
  4. idiomatic
  5. dialect_variation
  6. code_switched   (DERIVED — records with ≥2% English function words
                       in target-language text, regardless of prompt_style)

For each (language, category):
  • Count and % of language's final dataset
  • Cross-tabulation: prompt_style × code-switched
  • Quality stats
  • Length stats
  • Marker density (target-language signal strength)
  • Top 5 example records

Inputs:
  /workspace/synthetic/tier2/<lang>_final_dataset_v2.jsonl

Outputs:
  /workspace/synthetic/tier2/category_audit_report.json
  /workspace/synthetic/tier2/category_audit_overall.txt
  /workspace/synthetic/tier2/category_audit_<lang>.txt
"""

import json
import re
import sys
import statistics
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

TIER2_DIR = Path("/workspace/synthetic/tier2")
REPORT_PATH = TIER2_DIR / "category_audit_report.json"
LANGUAGES = ["swahili", "kikuyu", "kamba", "luo"]

# Prompt-style categories (from the curriculum)
PROMPT_STYLES = [
    "paraphrase_simple",
    "paraphrase_formal",
    "spoken",
    "idiomatic",
    "dialect_variation",
]

# Code-switching threshold — % English function words in text
CODESWITCH_ENG_RATIO_MIN = 0.02   # ≥2% English = code-switched

# Curriculum target (for reference in report)
CURRICULUM_TARGET = {
    "paraphrase_simple": 0.35,
    "paraphrase_formal": 0.20,
    "spoken":            0.20,
    "idiomatic":         0.15,
    "dialect_variation": 0.10,
}

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
    """True if record qualifies as code-switched (≥2% English function words)."""
    return english_function_ratio(text) >= CODESWITCH_ENG_RATIO_MIN


# ── ACCUMULATOR ──────────────────────────────────────────────────

class CategoryStats:
    def __init__(self):
        self.n = 0
        self.lengths = []
        self.qualities = []
        self.eng_ratios = []
        self.markers = []
        self.examples = []
        # Cross-tabs
        self.codeswitch_count = 0
        self.rescued_count = 0

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
                "text": text[:200],
                "english_ratio": round(eng, 4),
                "marker_density": round(marker_density(text, lang), 4),
                "quality": round(float(q), 3) if q else None,
            })

    def summary(self) -> dict:
        result = {"n": self.n}
        if self.lengths:
            result["length_mean"] = round(statistics.mean(self.lengths), 1)
            result["length_p50"] = int(statistics.median(self.lengths))
            result["length_p95"] = sorted(self.lengths)[int(len(self.lengths) * 0.95)] if len(self.lengths) > 20 else None
        if self.qualities:
            result["quality_mean"] = round(statistics.mean(self.qualities), 4)
            result["quality_p50"] = round(statistics.median(self.qualities), 4)
        else:
            result["quality_mean"] = None
        result["english_ratio_mean"] = round(statistics.mean(self.eng_ratios), 4) if self.eng_ratios else 0
        result["marker_density_mean"] = round(statistics.mean(self.markers), 4) if self.markers else 0
        result["codeswitched_count"] = self.codeswitch_count
        result["codeswitched_pct_within_category"] = round(self.codeswitch_count / self.n * 100, 2) if self.n else 0
        result["rescued_count"] = self.rescued_count
        result["rescued_pct_within_category"] = round(self.rescued_count / self.n * 100, 2) if self.n else 0
        result["examples"] = self.examples
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


def detect_style_field(path: Path) -> str:
    if not path.exists():
        return None
    candidates = ["prompt_style", "style", "curriculum_style", "gen_style"]
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= 50:
                break
            try:
                rec = json.loads(line.strip())
            except json.JSONDecodeError:
                continue
            for c in candidates:
                if c in rec:
                    return c
    return None


def analyze_language(lang: str) -> dict:
    print(f"\n  Analyzing {lang}...", flush=True)
    path = TIER2_DIR / f"{lang}_final_dataset_v2.jsonl"
    if not path.exists():
        return {"language": lang, "missing": True}

    style_field = detect_style_field(path)
    if not style_field:
        # Show sample keys for debugging
        with open(path, "r") as f:
            first_line = f.readline().strip()
            try:
                sample = json.loads(first_line)
                return {"language": lang, "no_style_field": True, "sample_keys": list(sample.keys())}
            except Exception:
                return {"language": lang, "no_style_field": True}
    print(f"    Style field: '{style_field}'", flush=True)

    # Per-style accumulators (5 prompt styles + UNKNOWN)
    per_style = defaultdict(CategoryStats)
    # Code-switched accumulator (derived category, can overlap)
    code_switched_stats = CategoryStats()
    # All-records accumulator for totals
    all_stats = CategoryStats()

    # Cross-tab: style × code_switching
    crosstab = defaultdict(lambda: {"monolingual": 0, "code_switched": 0})

    n_total = 0
    for rec in iter_jsonl(path):
        style = rec.get(style_field) or "UNKNOWN"
        text = rec.get("text") or rec.get("output") or rec.get("content") or ""

        per_style[style].add(rec, text, lang)
        all_stats.add(rec, text, lang)

        if is_code_switched(text):
            code_switched_stats.add(rec, text, lang)
            crosstab[style]["code_switched"] += 1
        else:
            crosstab[style]["monolingual"] += 1

        n_total += 1
        if n_total % 25000 == 0:
            print(f"    ... {n_total:,} records processed", flush=True)

    print(f"    Done. Total: {n_total:,}", flush=True)

    # Compose result
    style_summaries = {s: per_style[s].summary() for s in per_style}

    # Per-style distribution
    style_distribution = {}
    for style in PROMPT_STYLES + ["UNKNOWN"]:
        n_in_style = per_style[style].n if style in per_style else 0
        style_distribution[style] = {
            "n": n_in_style,
            "pct_of_dataset": round(n_in_style / n_total * 100, 2) if n_total else 0,
            "curriculum_target_pct": round(CURRICULUM_TARGET.get(style, 0) * 100, 1) if style in CURRICULUM_TARGET else None,
        }

    # Code-switching summary
    cs_count = code_switched_stats.n
    code_switching_summary = {
        "n": cs_count,
        "pct_of_dataset": round(cs_count / n_total * 100, 2) if n_total else 0,
        "stats": code_switched_stats.summary(),
    }

    return {
        "language":              lang,
        "style_field":           style_field,
        "n_total":               n_total,
        "style_distribution":    style_distribution,
        "style_summaries":       style_summaries,
        "code_switching":        code_switching_summary,
        "crosstab_style_x_cs":   {s: dict(c) for s, c in crosstab.items()},
    }


# ── REPORTS ──────────────────────────────────────────────────────

def write_overall_report(all_results: dict):
    path = TIER2_DIR / "category_audit_overall.txt"
    with open(path, "w", encoding="utf-8") as f:
        f.write("# LughaGen Tier 2 — CATEGORY AUDIT (Detailed)\n")
        f.write(f"# Generated: {datetime.utcnow().isoformat()}Z\n\n")

        # ── MAIN TABLE: Category × Language ──────────────────────
        f.write("=" * 100 + "\n")
        f.write("  PRIMARY TABLE — Category share per language final dataset\n")
        f.write("=" * 100 + "\n\n")
        f.write(f"  {'Category':<22}")
        for lang in LANGUAGES:
            if not all_results.get(lang, {}).get("missing"):
                f.write(f" {lang.capitalize():>12}")
        f.write(f" {'Target':>10}\n")
        f.write("  " + "-" * 92 + "\n")

        categories = PROMPT_STYLES + ["code_switched"]
        for cat in categories:
            f.write(f"  {cat:<22}")
            for lang in LANGUAGES:
                r = all_results.get(lang, {})
                if r.get("missing") or r.get("no_style_field"):
                    continue
                if cat == "code_switched":
                    n = r["code_switching"]["n"]
                    pct = r["code_switching"]["pct_of_dataset"]
                else:
                    dist = r["style_distribution"].get(cat, {})
                    n = dist.get("n", 0)
                    pct = dist.get("pct_of_dataset", 0)
                f.write(f" {n:>6,} ({pct:>4.1f}%)")
            # Target column
            if cat in CURRICULUM_TARGET:
                f.write(f" {CURRICULUM_TARGET[cat]*100:>9.1f}%")
            elif cat == "code_switched":
                f.write(f" {'derived':>10}")
            f.write("\n")

        # Total row
        f.write("  " + "-" * 92 + "\n")
        f.write(f"  {'TOTAL':<22}")
        for lang in LANGUAGES:
            r = all_results.get(lang, {})
            if r.get("missing") or r.get("no_style_field"):
                continue
            f.write(f"  {r['n_total']:>10,}")
        f.write("\n\n")

        # Note about code_switched
        f.write("  NOTE: 'code_switched' is a DERIVED category based on text content\n")
        f.write(f"        (≥{CODESWITCH_ENG_RATIO_MIN*100:.0f}% English function words).\n")
        f.write("        It OVERLAPS with prompt styles — a record can be e.g. both 'spoken' AND 'code_switched'.\n")
        f.write("        The first 5 rows sum to ~100% (curriculum styles partition the dataset).\n\n")

        # ── CROSS-TAB: prompt_style × code-switching ─────────────
        f.write("=" * 100 + "\n")
        f.write("  CROSS-TAB — Within each prompt style, how much is code-switched? (per language)\n")
        f.write("=" * 100 + "\n\n")

        for lang in LANGUAGES:
            r = all_results.get(lang, {})
            if r.get("missing") or r.get("no_style_field"):
                continue
            f.write(f"\n  {lang.upper()}\n")
            f.write(f"  {'Style':<22} {'Monolingual':>13} {'Code-switched':>15} {'CS % of style':>15}\n")
            f.write("  " + "-" * 70 + "\n")
            for style in PROMPT_STYLES:
                ct = r["crosstab_style_x_cs"].get(style, {})
                mono = ct.get("monolingual", 0)
                cs = ct.get("code_switched", 0)
                total = mono + cs
                pct = cs / total * 100 if total else 0
                f.write(f"  {style:<22} {mono:>13,} {cs:>15,} {pct:>14.2f}%\n")

        # ── QUALITY PER CATEGORY ─────────────────────────────────
        f.write("\n" + "=" * 100 + "\n")
        f.write("  MEAN QUALITY PER CATEGORY (where quality_score is present)\n")
        f.write("=" * 100 + "\n\n")
        f.write(f"  {'Category':<22}")
        for lang in LANGUAGES:
            if not all_results.get(lang, {}).get("missing"):
                f.write(f" {lang.capitalize():>12}")
        f.write("\n")
        f.write("  " + "-" * 80 + "\n")
        for cat in categories:
            f.write(f"  {cat:<22}")
            for lang in LANGUAGES:
                r = all_results.get(lang, {})
                if r.get("missing") or r.get("no_style_field"):
                    continue
                if cat == "code_switched":
                    q = r["code_switching"]["stats"].get("quality_mean")
                else:
                    q = r["style_summaries"].get(cat, {}).get("quality_mean")
                f.write(f" {q:>12.4f}" if q is not None else f" {'n/a':>12}")
            f.write("\n")

        # ── LENGTH AND MARKER DENSITY ────────────────────────────
        f.write("\n" + "=" * 100 + "\n")
        f.write("  MEAN LENGTH (chars) PER CATEGORY\n")
        f.write("=" * 100 + "\n\n")
        f.write(f"  {'Category':<22}")
        for lang in LANGUAGES:
            if not all_results.get(lang, {}).get("missing"):
                f.write(f" {lang.capitalize():>12}")
        f.write("\n")
        f.write("  " + "-" * 80 + "\n")
        for cat in categories:
            f.write(f"  {cat:<22}")
            for lang in LANGUAGES:
                r = all_results.get(lang, {})
                if r.get("missing") or r.get("no_style_field"):
                    continue
                if cat == "code_switched":
                    L = r["code_switching"]["stats"].get("length_mean")
                else:
                    L = r["style_summaries"].get(cat, {}).get("length_mean")
                f.write(f" {L:>12.0f}" if L is not None else f" {'n/a':>12}")
            f.write("\n")

        f.write("\n" + "=" * 100 + "\n")
        f.write("  ENGLISH FUNCTION-WORD RATIO PER CATEGORY (mean)\n")
        f.write("=" * 100 + "\n\n")
        f.write(f"  {'Category':<22}")
        for lang in LANGUAGES:
            if not all_results.get(lang, {}).get("missing"):
                f.write(f" {lang.capitalize():>12}")
        f.write("\n")
        f.write("  " + "-" * 80 + "\n")
        for cat in categories:
            f.write(f"  {cat:<22}")
            for lang in LANGUAGES:
                r = all_results.get(lang, {})
                if r.get("missing") or r.get("no_style_field"):
                    continue
                if cat == "code_switched":
                    eng = r["code_switching"]["stats"].get("english_ratio_mean")
                else:
                    eng = r["style_summaries"].get(cat, {}).get("english_ratio_mean")
                f.write(f" {eng:>12.4f}" if eng is not None else f" {'n/a':>12}")
            f.write("\n")

    return path


def write_per_language_report(lang: str, result: dict):
    path = TIER2_DIR / f"category_audit_{lang}.txt"
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# Category Audit — {lang.upper()}\n")
        f.write(f"# Generated: {datetime.utcnow().isoformat()}Z\n")
        f.write(f"# Total records: {result['n_total']:,}\n")
        f.write(f"# Style field detected: {result['style_field']}\n\n")

        # Distribution table
        f.write("=" * 80 + "\n")
        f.write(f"  {lang.upper()} — DISTRIBUTION OVERVIEW\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"  {'Category':<22} {'Count':>12} {'% Dataset':>11} {'Target':>10}\n")
        f.write("  " + "-" * 60 + "\n")
        for style in PROMPT_STYLES:
            dist = result["style_distribution"].get(style, {})
            n = dist.get("n", 0)
            pct = dist.get("pct_of_dataset", 0)
            target = dist.get("curriculum_target_pct")
            target_str = f"{target:>9.1f}%" if target is not None else " " * 10
            f.write(f"  {style:<22} {n:>12,} {pct:>10.2f}% {target_str}\n")

        # Unknown styles
        unk = result["style_distribution"].get("UNKNOWN", {})
        if unk.get("n", 0) > 0:
            f.write(f"  {'UNKNOWN':<22} {unk['n']:>12,} {unk['pct_of_dataset']:>10.2f}%\n")

        f.write("  " + "-" * 60 + "\n")

        # Code-switched (derived)
        cs = result["code_switching"]
        f.write(f"  {'(derived) code_switched':<22} {cs['n']:>12,} {cs['pct_of_dataset']:>10.2f}%   (overlaps above)\n")

        f.write("  " + "-" * 60 + "\n")
        f.write(f"  {'TOTAL':<22} {result['n_total']:>12,}\n\n")

        # Per-style detail
        for style in PROMPT_STYLES:
            stats = result["style_summaries"].get(style)
            if not stats or stats["n"] == 0:
                continue

            f.write("=" * 80 + "\n")
            f.write(f"  STYLE: {style}\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"  Records:                  {stats['n']:,}\n")
            f.write(f"  % of dataset:             {result['style_distribution'][style]['pct_of_dataset']:.2f}%\n")
            f.write(f"  Curriculum target:        {result['style_distribution'][style].get('curriculum_target_pct', 'n/a')}%\n")
            f.write(f"  Mean length (chars):      {stats.get('length_mean', 'n/a')}\n")
            f.write(f"  p50 / p95 length:         {stats.get('length_p50', 'n/a')} / {stats.get('length_p95', 'n/a')}\n")
            f.write(f"  Mean quality_score:       {stats.get('quality_mean', 'n/a')}\n")
            f.write(f"  Mean English ratio:       {stats.get('english_ratio_mean', 'n/a')}\n")
            f.write(f"  Mean marker density:      {stats.get('marker_density_mean', 'n/a')}\n")
            f.write(f"  Of these, code-switched:  {stats.get('codeswitched_count', 0):,}  ({stats.get('codeswitched_pct_within_category', 0):.2f}% within style)\n")
            f.write(f"  Of these, rescued via v2: {stats.get('rescued_count', 0):,}  ({stats.get('rescued_pct_within_category', 0):.2f}% within style)\n")

            if stats.get("examples"):
                f.write(f"\n  Examples:\n")
                for i, ex in enumerate(stats["examples"], 1):
                    f.write(f"    [{i}] eng={ex['english_ratio']}  markers={ex['marker_density']}  q={ex.get('quality')}\n")
                    f.write(f"        {ex['text']}\n\n")
            f.write("\n")

        # Code-switched detail
        cs_stats = result["code_switching"]["stats"]
        if cs_stats["n"]:
            f.write("=" * 80 + "\n")
            f.write(f"  DERIVED CATEGORY: code_switched (≥{CODESWITCH_ENG_RATIO_MIN*100:.0f}% English function words)\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"  Records:                  {cs_stats['n']:,}\n")
            f.write(f"  % of dataset:             {result['code_switching']['pct_of_dataset']:.2f}%\n")
            f.write(f"  Mean length (chars):      {cs_stats.get('length_mean', 'n/a')}\n")
            f.write(f"  Mean quality_score:       {cs_stats.get('quality_mean', 'n/a')}\n")
            f.write(f"  Mean English ratio:       {cs_stats.get('english_ratio_mean', 'n/a')}\n")
            f.write(f"  Mean marker density:      {cs_stats.get('marker_density_mean', 'n/a')}\n")

            if cs_stats.get("examples"):
                f.write(f"\n  Examples:\n")
                for i, ex in enumerate(cs_stats["examples"], 1):
                    f.write(f"    [{i}] eng={ex['english_ratio']}  markers={ex['marker_density']}  q={ex.get('quality')}\n")
                    f.write(f"        {ex['text']}\n\n")

        # Cross-tab
        f.write("\n" + "=" * 80 + "\n")
        f.write("  CROSS-TAB: prompt_style × code-switching status\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"  {'Style':<22} {'Monolingual':>13} {'Code-switched':>15} {'CS %':>10}\n")
        f.write("  " + "-" * 70 + "\n")
        for style in PROMPT_STYLES:
            ct = result["crosstab_style_x_cs"].get(style, {})
            mono = ct.get("monolingual", 0)
            cs = ct.get("code_switched", 0)
            total = mono + cs
            pct = cs / total * 100 if total else 0
            f.write(f"  {style:<22} {mono:>13,} {cs:>15,} {pct:>9.2f}%\n")

    return path


# ── MAIN ─────────────────────────────────────────────────────────

def main():
    if not TIER2_DIR.exists():
        print(f"✗ {TIER2_DIR} not found", file=sys.stderr)
        sys.exit(1)

    print("=" * 80)
    print("  TIER 2 CATEGORY AUDIT (DETAILED)")
    print(f"  {datetime.utcnow().isoformat()}Z")
    print("=" * 80)

    all_results = {}
    for lang in LANGUAGES:
        all_results[lang] = analyze_language(lang)

    # Write outputs
    print("\n  Writing reports...")
    report = {
        "version":      "category_audit_v1.0",
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "curriculum_target":  CURRICULUM_TARGET,
        "code_switching_threshold_english_ratio": CODESWITCH_ENG_RATIO_MIN,
        "per_language": all_results,
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"  ✓ JSON report: {REPORT_PATH}")

    overall_path = write_overall_report(all_results)
    print(f"  ✓ Overall: {overall_path}")

    for lang, r in all_results.items():
        if r.get("missing") or r.get("no_style_field"):
            continue
        per_lang = write_per_language_report(lang, r)
        print(f"  ✓ {lang}: {per_lang}")

    # ── Console summary — THE MAIN TABLE ─────────────────────────
    print("\n" + "=" * 100)
    print("  PRIMARY DISTRIBUTION TABLE")
    print("=" * 100)
    print(f"\n  {'Category':<22}", end="")
    for lang in LANGUAGES:
        if not all_results[lang].get("missing") and not all_results[lang].get("no_style_field"):
            print(f" {lang.capitalize():>14}", end="")
    print(f" {'Target':>10}")
    print("  " + "-" * 92)

    categories = PROMPT_STYLES + ["code_switched"]
    for cat in categories:
        print(f"  {cat:<22}", end="")
        for lang in LANGUAGES:
            r = all_results[lang]
            if r.get("missing") or r.get("no_style_field"):
                continue
            if cat == "code_switched":
                n = r["code_switching"]["n"]
                pct = r["code_switching"]["pct_of_dataset"]
            else:
                dist = r["style_distribution"].get(cat, {})
                n = dist.get("n", 0)
                pct = dist.get("pct_of_dataset", 0)
            print(f" {n:>7,} ({pct:>4.1f}%)", end="")
        if cat in CURRICULUM_TARGET:
            print(f" {CURRICULUM_TARGET[cat]*100:>9.1f}%")
        elif cat == "code_switched":
            print(f" {'derived':>10}")
        else:
            print()

    print("  " + "-" * 92)
    print(f"  {'TOTAL':<22}", end="")
    for lang in LANGUAGES:
        r = all_results[lang]
        if r.get("missing") or r.get("no_style_field"):
            continue
        print(f"  {r['n_total']:>13,}", end="")
    print()

    print("\n  NOTE: code_switched is a DERIVED category. It overlaps with the 5 prompt styles above.")
    print(f"        A record is code_switched if its text has ≥{CODESWITCH_ENG_RATIO_MIN*100:.0f}% English function words.")

    # No-style-field warnings
    for lang, r in all_results.items():
        if r.get("no_style_field"):
            print(f"\n  ⚠ {lang.upper()}: no prompt-style field found in records.")
            print(f"    Sample keys present: {r.get('sample_keys', [])}")
            print(f"    The audit can only report on the code_switched category for this language.")

    print("\n" + "=" * 100)
    print(f"\n  Outputs in {TIER2_DIR}:")
    print(f"    category_audit_report.json   (full data)")
    print(f"    category_audit_overall.txt   (full tables + cross-tabs)")
    print(f"    category_audit_<lang>.txt    (per-language detail with examples)")
    print("=" * 100 + "\n")


if __name__ == "__main__":
    main()
