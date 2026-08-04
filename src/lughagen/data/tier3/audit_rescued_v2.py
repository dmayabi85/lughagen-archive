"""
audit_rescued.py — LughaGen Tier 2 Rescue Verification Audit

Inspects the output of rescue_quarantined.py to verify that the rescued
samples are legitimate target-language content (including legitimate code-
switching) and not false positives.

Verification methods:
  1. Per-signal sample showcase (10 examples per rescue signal per language)
  2. Edge-case detection (lowest-confidence rescues — most likely FPs)
  3. Comparative statistics (rescued vs original-passed: length, markers, English)
  4. Cross-validation (re-apply heuristics; confirm rescue signal still holds)
  5. Suspicious-pattern flags (high English ratio, very short, low markers)

Inputs (per language):
  /workspace/synthetic/tier2/<lang>_rescued.jsonl
  /workspace/synthetic/tier2/<lang>_audited_passed.jsonl  (baseline for comparison)

Outputs:
  /workspace/synthetic/tier2/rescue_audit_report.json
  /workspace/synthetic/tier2/rescue_audit_<lang>.txt  (one per language)
"""

import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
import random
import statistics

# ── PATHS ────────────────────────────────────────────────────────
TIER2_DIR = Path("/workspace/synthetic/tier2")
AUDIT_REPORT = TIER2_DIR / "rescue_audit_report_v2.json"

LANGUAGES = ["swahili", "kikuyu", "kamba", "luo"]

# ── LANGUAGE-SPECIFIC SIGNALS (must match rescue_quarantined.py) ──

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


def target_marker_density(text: str, lang: str) -> float:
    """Density of target-language markers (diacritics + function words) per token."""
    tokens = re.findall(r"\S+", text.lower())
    if not tokens:
        return 0.0

    # Diacritic hits (character-level, normalize to per-token rate)
    diacritic_pattern = TARGET_DIACRITICS.get(lang)
    diacritic_hits = 0
    if diacritic_pattern:
        diacritic_hits = sum(1 for t in tokens if diacritic_pattern.search(t))

    # Function word hits (normalize diacritics for matching)
    fw = TARGET_FUNCTION_WORDS.get(lang, set())
    tokens_normalized = [re.sub(r"[ĩ]", "i", re.sub(r"[ũ]", "u", t)) for t in tokens]
    fw_normalized = {re.sub(r"[ĩ]", "i", re.sub(r"[ũ]", "u", w)) for w in fw}
    fw_hits = sum(1 for t in tokens_normalized if t in fw_normalized)

    return (diacritic_hits + fw_hits) / len(tokens)


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


# ── CROSS-VALIDATION ─────────────────────────────────────────────

def verify_rescue_signal(rec: dict, lang: str) -> tuple:
    """
    Re-verify the signal that supposedly rescued this sample.
    Returns (still_valid: bool, notes: str)
    """
    signal = rec.get("rescue_signal", "")
    gl_top1 = rec.get("audit_glotlid_top1") or ""
    gl_top1_p = rec.get("audit_glotlid_top1_p") or 0.0
    af_label = rec.get("audit_afrolid") or ""
    af_prob = rec.get("audit_afrolid_p") or 0.0
    text = rec.get("text") or rec.get("output") or rec.get("content") or ""

    target_gl = GLOTLID_TARGET.get(lang, set())
    target_af = AFROLID_TARGET.get(lang, set())
    top3 = extract_top3_codes(rec)

    if signal == "glotlid_strict_target":
        valid = gl_top1 in target_gl and gl_top1_p >= 0.40
        return valid, f"gl_top1={gl_top1}/{gl_top1_p:.2f}"

    elif signal == "glotlid_top3_target":
        intersect = target_gl & top3
        return bool(intersect), f"top3∩target={intersect or 'NONE'}"

    elif signal == "afrolid_confirmed_target":
        valid = af_label in target_af and af_prob >= 0.60
        return valid, f"af_label={af_label}/{af_prob:.2f}"

    elif signal == "marker_density":
        density = target_marker_density(text, lang)
        # Original logic used a different scoring scheme; check density is reasonable
        return density >= 0.05, f"marker_density={density:.3f}"

    return False, f"unknown_signal={signal}"


# ── SUSPICIOUS PATTERN DETECTION ─────────────────────────────────

def flag_suspicious(rec: dict, lang: str) -> list:
    """Return list of suspicion flags for this rescue."""
    flags = []
    text = rec.get("text") or rec.get("output") or rec.get("content") or ""

    L = len(text)
    eng = english_function_ratio(text)
    markers = target_marker_density(text, lang)
    gl_top1_p = rec.get("audit_glotlid_top1_p") or 0.0

    if L < 30:
        flags.append("very_short")
    if eng > 0.40:
        flags.append("high_english_ratio")
    if eng > 0.20 and markers < 0.05:
        flags.append("english_heavy_low_markers")
    if markers == 0 and rec.get("rescue_signal") == "marker_density":
        flags.append("zero_markers_but_marker_rescue")
    if gl_top1_p < 0.30:
        flags.append("very_low_glotlid_confidence")

    return flags


# ── PER-LANGUAGE ANALYSIS ────────────────────────────────────────

def load_jsonl(path: Path):
    if not path.exists():
        return []
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def compute_stats(records: list, lang: str) -> dict:
    if not records:
        return {"n": 0}

    lengths = []
    eng_ratios = []
    marker_densities = []
    gl_probs = []

    for rec in records:
        text = rec.get("text") or rec.get("output") or rec.get("content") or ""
        lengths.append(len(text))
        eng_ratios.append(english_function_ratio(text))
        marker_densities.append(target_marker_density(text, lang))
        gl_probs.append(rec.get("audit_glotlid_top1_p") or 0.0)

    return {
        "n": len(records),
        "length":          {"mean": round(statistics.mean(lengths), 1),
                            "median": int(statistics.median(lengths)),
                            "min": min(lengths), "max": max(lengths)},
        "english_ratio":   {"mean": round(statistics.mean(eng_ratios), 4),
                            "median": round(statistics.median(eng_ratios), 4),
                            "p95": round(sorted(eng_ratios)[int(len(eng_ratios) * 0.95)], 4) if len(eng_ratios) > 20 else None},
        "marker_density":  {"mean": round(statistics.mean(marker_densities), 4),
                            "median": round(statistics.median(marker_densities), 4),
                            "p05": round(sorted(marker_densities)[int(len(marker_densities) * 0.05)], 4) if len(marker_densities) > 20 else None},
        "glotlid_prob":    {"mean": round(statistics.mean(gl_probs), 3),
                            "median": round(statistics.median(gl_probs), 3)},
    }


def analyze_language(lang: str, rng: random.Random) -> dict:
    rescued_path = TIER2_DIR / f"{lang}_rescued_v2.jsonl"
    passed_path  = TIER2_DIR / f"{lang}_audited_passed.jsonl"

    rescued = load_jsonl(rescued_path)
    if not rescued:
        return {"language": lang, "missing": True}

    print(f"\n  Analyzing {lang} rescued samples ({len(rescued):,})...", flush=True)

    # Load a sample of originally-passed for comparison (full file is huge)
    passed_sample = []
    if passed_path.exists():
        passed_sample = load_jsonl(passed_path)
        if len(passed_sample) > 10_000:
            passed_sample = rng.sample(passed_sample, 10_000)

    # Group by signal
    by_signal = defaultdict(list)
    for rec in rescued:
        by_signal[rec.get("rescue_signal", "unknown")].append(rec)

    # Cross-validation
    cv_results = Counter()
    cv_failures = []
    for rec in rescued:
        valid, note = verify_rescue_signal(rec, lang)
        cv_results["valid" if valid else "invalid"] += 1
        if not valid:
            cv_failures.append({
                "signal": rec.get("rescue_signal"),
                "note": note,
                "text": (rec.get("text") or "")[:150],
            })

    # Suspicious patterns
    suspicion_counts = Counter()
    suspicious_samples = []
    for rec in rescued:
        flags = flag_suspicious(rec, lang)
        for f in flags:
            suspicion_counts[f] += 1
        if flags and len(suspicious_samples) < 30:
            suspicious_samples.append({
                "flags": flags,
                "signal": rec.get("rescue_signal"),
                "text": (rec.get("text") or "")[:200],
                "gl_top1": rec.get("audit_glotlid_top1"),
                "gl_top1_p": rec.get("audit_glotlid_top1_p"),
            })

    # Sample showcase per signal (10 random per signal)
    signal_samples = {}
    for sig, recs in by_signal.items():
        sample_n = min(10, len(recs))
        chosen = rng.sample(recs, sample_n)
        signal_samples[sig] = [{
            "text": (r.get("text") or "")[:200],
            "gl_top1": r.get("audit_glotlid_top1"),
            "gl_top1_p": r.get("audit_glotlid_top1_p"),
            "af_label": r.get("audit_afrolid"),
            "af_prob": r.get("audit_afrolid_p"),
            "marker_density": round(target_marker_density(r.get("text") or "", lang), 3),
        } for r in chosen]

    # Edge cases — lowest-confidence rescues per signal
    edge_cases = {}
    for sig, recs in by_signal.items():
        scored = []
        for r in recs:
            text = r.get("text") or ""
            confidence_score = (
                (r.get("audit_glotlid_top1_p") or 0) +
                (r.get("audit_afrolid_p") or 0) +
                target_marker_density(text, lang) * 2
            )
            scored.append((confidence_score, r))
        scored.sort(key=lambda x: x[0])
        edge_cases[sig] = [{
            "score": round(s, 3),
            "text": (r.get("text") or "")[:200],
            "gl_top1": r.get("audit_glotlid_top1"),
            "gl_top1_p": r.get("audit_glotlid_top1_p"),
            "af_label": r.get("audit_afrolid"),
            "marker_density": round(target_marker_density(r.get("text") or "", lang), 3),
        } for s, r in scored[:10]]

    # Stats comparison
    rescued_stats = compute_stats(rescued, lang)
    passed_stats  = compute_stats(passed_sample, lang) if passed_sample else None

    return {
        "language":            lang,
        "n_rescued":           len(rescued),
        "n_passed_sampled":    len(passed_sample),
        "signal_breakdown":    {sig: len(recs) for sig, recs in by_signal.items()},
        "cross_validation":    dict(cv_results),
        "cross_validation_failures_sampled": cv_failures[:10],
        "suspicion_counts":    dict(suspicion_counts),
        "suspicious_samples":  suspicious_samples,
        "signal_samples":      signal_samples,
        "edge_cases":          edge_cases,
        "rescued_stats":       rescued_stats,
        "passed_stats":        passed_stats,
    }


# ── WRITE PER-LANGUAGE READABLE REPORT ───────────────────────────

def write_text_report(lang: str, result: dict):
    path = TIER2_DIR / f"rescue_audit_{lang}_v2.txt"
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# Rescue Audit — {lang.upper()}\n")
        f.write(f"# Generated: {datetime.utcnow().isoformat()}Z\n")
        f.write(f"# n_rescued: {result['n_rescued']:,}\n\n")

        # Cross-validation summary
        cv = result["cross_validation"]
        cv_total = sum(cv.values())
        valid_pct = cv.get("valid", 0) / cv_total * 100 if cv_total else 0
        f.write(f"{'='*70}\n")
        f.write(f"  CROSS-VALIDATION\n")
        f.write(f"{'='*70}\n\n")
        f.write(f"  Signal still holds:    {cv.get('valid', 0):,} ({valid_pct:.1f}%)\n")
        f.write(f"  Signal no longer holds: {cv.get('invalid', 0):,}\n\n")
        if result.get("cross_validation_failures_sampled"):
            f.write(f"  Sample failures (first 10):\n")
            for fail in result["cross_validation_failures_sampled"]:
                f.write(f"    [signal={fail['signal']}  note={fail['note']}]\n")
                f.write(f"      {fail['text']}\n\n")

        # Statistical comparison
        f.write(f"{'='*70}\n")
        f.write(f"  COMPARATIVE STATS — rescued vs original passed\n")
        f.write(f"{'='*70}\n\n")
        rs = result["rescued_stats"]
        ps = result["passed_stats"]
        if ps:
            f.write(f"  {'Metric':<25} {'Rescued':>15} {'Passed (sample)':>20}\n")
            f.write(f"  {'-'*60}\n")
            f.write(f"  {'n':<25} {rs['n']:>15,} {ps['n']:>20,}\n")
            f.write(f"  {'mean length':<25} {rs['length']['mean']:>15} {ps['length']['mean']:>20}\n")
            f.write(f"  {'median length':<25} {rs['length']['median']:>15} {ps['length']['median']:>20}\n")
            f.write(f"  {'mean english_ratio':<25} {rs['english_ratio']['mean']:>15} {ps['english_ratio']['mean']:>20}\n")
            f.write(f"  {'mean marker_density':<25} {rs['marker_density']['mean']:>15} {ps['marker_density']['mean']:>20}\n")
            f.write(f"  {'mean glotlid_prob':<25} {rs['glotlid_prob']['mean']:>15} {ps['glotlid_prob']['mean']:>20}\n\n")

        # Suspicion summary
        if result["suspicion_counts"]:
            f.write(f"{'='*70}\n")
            f.write(f"  SUSPICION FLAGS\n")
            f.write(f"{'='*70}\n\n")
            for flag, count in sorted(result["suspicion_counts"].items(), key=lambda x: -x[1]):
                pct = count / result["n_rescued"] * 100
                f.write(f"  {flag:<35} {count:>6,}  ({pct:.2f}%)\n")
            f.write("\n  Examples (first 10):\n\n")
            for s in result["suspicious_samples"][:10]:
                f.write(f"    [flags={','.join(s['flags'])}  signal={s['signal']}  gl={s['gl_top1']}/{s['gl_top1_p']}]\n")
                f.write(f"      {s['text']}\n\n")

        # Sample showcase
        f.write(f"{'='*70}\n")
        f.write(f"  SAMPLES PER SIGNAL\n")
        f.write(f"{'='*70}\n")
        for sig, samples in result["signal_samples"].items():
            count = result["signal_breakdown"].get(sig, 0)
            f.write(f"\n  ── {sig}  ({count:,} total) ──\n\n")
            for i, s in enumerate(samples, 1):
                f.write(f"    [{i}] gl={s['gl_top1']}/{s['gl_top1_p']:.2f}  af={s['af_label']}/{s['af_prob']:.2f}  markers={s['marker_density']}\n")
                f.write(f"        {s['text']}\n\n")

        # Edge cases
        f.write(f"\n{'='*70}\n")
        f.write(f"  EDGE CASES — lowest-confidence rescues per signal\n")
        f.write(f"  (These are most likely to be false positives — inspect carefully)\n")
        f.write(f"{'='*70}\n")
        for sig, samples in result["edge_cases"].items():
            f.write(f"\n  ── {sig} ──\n\n")
            for i, s in enumerate(samples, 1):
                f.write(f"    [{i}] score={s['score']}  gl={s['gl_top1']}/{s['gl_top1_p']:.2f}  af={s['af_label']}  markers={s['marker_density']}\n")
                f.write(f"        {s['text']}\n\n")

    return path


# ── MAIN ─────────────────────────────────────────────────────────

def main():
    if not TIER2_DIR.exists():
        print(f"✗ {TIER2_DIR} not found", file=sys.stderr)
        sys.exit(1)

    rng = random.Random(42)

    print("═" * 70)
    print("  RESCUE VERIFICATION AUDIT — v2 (against tightened rescues)")
    print("═" * 70)

    results = {}
    text_reports = {}
    for lang in LANGUAGES:
        r = analyze_language(lang, rng)
        results[lang] = r
        if not r.get("missing"):
            text_reports[lang] = str(write_text_report(lang, r))

    # JSON master report
    report = {
        "version":      "rescue_audit_v2.0",
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "per_language": results,
        "text_reports": text_reports,
    }
    AUDIT_REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    # ── Console summary ───────────────────────────────────────────
    print("\n" + "═" * 70)
    print("  AUDIT SUMMARY")
    print("═" * 70)

    for lang in LANGUAGES:
        r = results[lang]
        if r.get("missing"):
            print(f"\n  {lang.upper()}: no rescued file found")
            continue

        cv = r["cross_validation"]
        cv_total = sum(cv.values())
        valid_pct = cv.get("valid", 0) / cv_total * 100 if cv_total else 0
        n_suspicious = sum(1 for s in r["suspicious_samples"] if s["flags"])

        print(f"\n  {lang.upper()}")
        print(f"    Rescued:                {r['n_rescued']:>7,}")
        print(f"    Cross-validation:       {cv.get('valid', 0):>7,}/{cv_total:,} ({valid_pct:.1f}% signal still holds)")
        if r["suspicion_counts"]:
            total_susp = sum(r["suspicion_counts"].values())
            print(f"    Suspicion flags raised: {total_susp:>7,} across {len(r['suspicion_counts'])} categories")
            for flag, count in sorted(r["suspicion_counts"].items(), key=lambda x: -x[1])[:3]:
                pct = count / r["n_rescued"] * 100
                print(f"      • {flag:<30} {count:>6,}  ({pct:.2f}%)")
        else:
            print(f"    Suspicion flags raised:       0  (clean)")

        # Stats comparison teaser
        rs = r["rescued_stats"]
        ps = r["passed_stats"]
        if ps and rs["n"] > 0:
            len_ratio = rs['length']['mean'] / ps['length']['mean'] if ps['length']['mean'] else 0
            mark_ratio = rs['marker_density']['mean'] / max(ps['marker_density']['mean'], 0.001)
            print(f"    Rescued mean length:    {rs['length']['mean']:>7.0f}  (passed: {ps['length']['mean']:.0f}, ratio {len_ratio:.2f}x)")
            print(f"    Rescued mean markers:   {rs['marker_density']['mean']:>7.3f}  (passed: {ps['marker_density']['mean']:.3f}, ratio {mark_ratio:.2f}x)")
            print(f"    Rescued mean english:   {rs['english_ratio']['mean']:>7.3f}  (passed: {ps['english_ratio']['mean']:.3f})")

    print("\n" + "═" * 70)
    print(f"\n  Reports in {TIER2_DIR}:")
    print(f"    rescue_audit_report.json")
    for lang in LANGUAGES:
        if lang in text_reports:
            print(f"    rescue_audit_{lang}.txt")
    print("═" * 70 + "\n")


if __name__ == "__main__":
    main()
