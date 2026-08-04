#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════╗
║   LUGHAGEN AfroLID DISAGREEMENT ANALYSER — v2.0                 ║
║                                                                  ║
║  Architecture:                                                   ║
║    GlotLID (primary filter)                                      ║
║    Heuristics (core words)                                       ║
║    AfroLID (secondary analysis — flags, never rejects)           ║
║                                                                  ║
║  All fixes applied:                                              ║
║    FIX1: Top-K acceptance (not top-1 only)                       ║
║    FIX2: Confidence-aware disagreement severity                  ║
║    FIX3: Core word score as tie-breaker signal                   ║
║    FIX4: Reproducible sampling (seed=42)                         ║
║    FIX5: Short texts tracked separately (not skipped)            ║
║    FIX6: Strict vs relaxed purity reported separately            ║
║                                                                  ║
║  Usage:                                                          ║
║    python3 afrolid_analysis.py --tier 1                          ║
║    python3 afrolid_analysis.py --tier 2 --sample 5000            ║
║    python3 afrolid_analysis.py --tier 1 --topk 3                 ║
╚══════════════════════════════════════════════════════════════════╝
"""

import json, re, collections, argparse, os, random
from pathlib import Path
from datetime import datetime

# ── CLI ───────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="LughaGen AfroLID Analyser v2")
parser.add_argument("--tier",   type=int, choices=[1,2], required=True)
parser.add_argument("--sample", type=int, default=0,
                    help="Reproducible sample per language (0=all)")
parser.add_argument("--seed",   type=int, default=42,
                    help="Random seed for reproducibility (default: 42)")
parser.add_argument("--topk",   type=int, default=3,
                    help="Top-K predictions to consider (default: 3)")
parser.add_argument("--glotlid", default="/workspace/models/glotlid.bin")
parser.add_argument("--afrolid", default="/workspace/models/afrolid.bin")
parser.add_argument("--out",    default="/workspace/synthetic/analysis")
parser.add_argument("--min-conf", type=float, default=0.30,
                    help="Min confidence for top-K acceptance (default: 0.30)")
args = parser.parse_args()

BASE   = {1:"/workspace/synthetic/tier1", 2:"/workspace/synthetic/tier2"}[args.tier]
SUFFIX = {1:"_tier1.jsonl", 2:"_tier2.jsonl"}[args.tier]
OUT    = Path(args.out)
OUT.mkdir(parents=True, exist_ok=True)

MIN_SHORT_LEN = 25   # texts below this tracked separately


# ── Load models ───────────────────────────────────────────────────
def load_model(path, name):
    try:
        import fasttext
        if os.path.exists(path):
            m = fasttext.load_model(path)
            print(f"✅ {name} loaded from {path}")
            return m
        print(f"⚠️  {name} not found at {path}")
        return None
    except Exception as e:
        print(f"⚠️  {name} failed: {e}")
        return None

glotlid = load_model(args.glotlid, "GlotLID")

# AfroLID: load via HuggingFace transformer pipeline
try:
    from transformers import pipeline as hf_pipeline, AutoConfig
    import torch
    device    = 0 if torch.cuda.is_available() else -1
    afrolid   = hf_pipeline("text-classification",
                             model="UBC-NLP/afrolid_1.5",
                             device=device, truncation=True, max_length=128)
    af_config = AutoConfig.from_pretrained("UBC-NLP/afrolid_1.5")
    print(f"✅ AfroLID loaded (UBC-NLP/afrolid_1.5 — {len(af_config.id2label)} languages)")
    AFROLID_IS_PIPELINE = True
except Exception as e:
    afrolid = None
    AFROLID_IS_PIPELINE = False
    print(f"⚠️  AfroLID unavailable ({e}) — GlotLID + heuristics only")

if not glotlid:
    print("GlotLID required — exiting.")
    exit(1)
if not afrolid:
    print("AfroLID not available — running GlotLID + heuristics only.")


# ── Language codes ────────────────────────────────────────────────
# GlotLID: ISO 639-3 + script (e.g. kam_Latn)
GLOTLID_TARGET = {
    "kamba":   {"kam_Latn"},
    "kikuyu":  {"kik_Latn"},
    "luo":     {"luo_Latn"},
    "swahili": {"swh_Latn","swc_Latn"},
}
GLOTLID_ACCEPTABLE = {
    "kamba":   {"kam_Latn","kik_Latn","luy_Latn","guz_Latn","mer_Latn",
                "nim_Latn","dad_Latn","yaa_Latn","chk_Latn"},
    "kikuyu":  {"kik_Latn","kam_Latn","mer_Latn","luy_Latn","thk_Latn",
                "des_Latn","cbc_Latn","bsn_Latn","tav_Latn","myy_Latn"},
    "luo":     {"luo_Latn","ach_Latn","laj_Latn","adh_Latn","alz_Latn",
                "bam_Latn","tuv_Latn"},
    "swahili": {"swh_Latn","swc_Latn","vmw_Latn","hay_Latn","ssw_Latn"},
}

# AfroLID (UBC-NLP/afrolid_1.5): 518 African languages, ISO 639-3 short codes
# eng NOT in label space — English contamination detection via GlotLID only
# FINDING: AfroLID assigns Kamba→kik (97.2%) despite having 'kam' label —
# systematic bias due to Kamba/Kikuyu morphological overlap in training data
AFROLID_TARGET = {
    "kamba":   {"kam"},
    "kikuyu":  {"kik"},
    "luo":     {"luo"},
    "swahili": {"swh","swc","swa"},
}
AFROLID_ACCEPTABLE = {
    "kamba":   {"kam","kik","luy","guz","mer"},  # kik: confirmed systematic bias
    "kikuyu":  {"kik","kam","mer","luy","thk"},
    "luo":     {"luo","ach","laj","adh","alz"},
    "swahili": {"swh","swc","swa","vmw"},
}

# ── Core words (FIX3: tie-breaker signal) ─────────────────────────
LANG_CORE = {
    "swahili": {
        "na","ni","wa","ya","ka","kwa","la","za","pa","mwa","katika",
        "kuwa","pia","au","lakini","kama","hii","hizo","hivyo","kwamba",
        "yote","wote","hata","bali","ili","sana","zaidi","sawa","bado",
        "alikuwa","wanaweza","tunaweza","ninaweza","anaweza","naweza",
        "alipata","alisema","alifanya","mtu","watu","mwaka","nchi",
        "chakula","maji","kazi","mimi","wewe","yeye","sisi","wao",
        "ambao","ambaye","ambazo","mchezaji","kusema","kufanya","kwenda",
    },
    "kamba": {
        "na","ni","wa","ya","ka","ma","la","ta","ku","mu","si","vi",
        "nĩ","wĩ","yĩ","kĩ","mĩ","ũ","ĩ","ũndũ","mũndu","ngai","ndĩ",
        "kana","nthini","kyau","mawu","ngũ","ũkũ","mũsyi","mwanake",
    },
    "kikuyu": {
        "na","ni","wa","ya","ka","ma","nĩ","wĩ","yĩ","kĩ","mĩ","ũ",
        "ĩ","mũndũ","ngai","ndĩ","ũndũ","rĩĩtwa","gĩkũyũ","wĩra",
        "mũciĩ","mũgũnda","nyũmba","mwĩrĩtu","mwanake","ciana",
    },
    "luo": {
        "gi","ni","bi","di","ma","ka","la","wa","ya","na","to","go",
        "ne","ji","jii","wach","tich","dala","luo","nyaka","kendo",
        "bende","mana","kata","kuom","ndiko","joluo","wuod","nyar",
    },
}


def tokenize(text):
    return re.findall(r"[a-zA-ZÀ-ÿũĩ']+", text.lower())


def core_word_score(text, lang):
    """FIX3: Linguistic tie-breaker — fraction of core language words."""
    words = tokenize(text)
    if not words:
        return 0.0
    core  = LANG_CORE.get(lang, set())
    hits  = sum(1 for w in words if w in core)
    return round(hits / len(words), 4)


def predict(model, text, k=5):
    if model is None:
        return []
    try:
        # Detect model type by class name — more reliable than hasattr
        cls = type(model).__name__
        if cls in ("FastText", "_FastText", "_FastText"):
            # GlotLID (fasttext)
            text_clean = text.replace("\n"," ").strip()[:500]
            labels, probs = model.predict(text_clean, k=k)
            return [(l.replace("__label__",""), float(p))
                    for l,p in zip(labels,probs)]
        else:
            # AfroLID (HuggingFace transformer pipeline)
            text_clean = text.replace("\n"," ").strip()[:512]
            results = model(text_clean, top_k=k)
            if results and isinstance(results[0], list):
                results = results[0]
            return [(r["label"], float(r["score"])) for r in results[:k]]
    except Exception as e:
        return []


def is_in_topk(preds, target_set, acceptable_set, k, min_conf):
    """
    FIX1: Top-K acceptance.
    Returns (verdict, top_label, top_conf, matched_label, matched_conf)
    verdict: "target" | "acceptable" | "wrong"
    """
    if not preds:
        return "unknown", "?", 0.0, "?", 0.0

    top_label, top_conf = preds[0]

    # Check top-K for any target or acceptable match
    for label, conf in preds[:k]:
        if conf < min_conf:
            break  # sorted descending — no point continuing
        if label in target_set:
            return "target", top_label, top_conf, label, conf
        if label in acceptable_set:
            return "acceptable", top_label, top_conf, label, conf

    return "wrong", top_label, top_conf, top_label, top_conf


def disagreement_severity(gl_conf, af_conf):
    """
    FIX2: Confidence-aware severity classification.
    Returns severity label and numeric score.
    """
    diff = abs(gl_conf - af_conf)

    if gl_conf >= 0.80 and af_conf <= 0.40:
        return "strong_glotlid_win", diff
    elif af_conf >= 0.80 and gl_conf <= 0.40:
        return "strong_afrolid_win", diff
    elif gl_conf >= 0.70 and af_conf >= 0.70:
        return "high_confidence_both", diff
    elif diff <= 0.20:
        return "ambiguous", diff
    else:
        return "moderate_glotlid_win" if gl_conf > af_conf else "moderate_afrolid_win", diff


# ── Main analysis loop ────────────────────────────────────────────
print(f"\n{'='*65}")
print(f"  LUGHAGEN AfroLID ANALYSIS — TIER {args.tier}")
print(f"  {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
print(f"  Seed={args.seed}  TopK={args.topk}  MinConf={args.min_conf}")
print(f"{'='*65}\n")

all_results = []

for lang in ["kamba","kikuyu","luo","swahili"]:
    path = Path(BASE) / f"{lang}{SUFFIX}"
    if not path.exists():
        print(f"⚠️  {path} not found")
        continue

    print(f"\nAnalysing {lang}...")

    # FIX4: Reproducible sampling
    all_samples = []
    with open(path) as f:
        for line in f:
            try: all_samples.append(json.loads(line))
            except: pass

    if args.sample and args.sample < len(all_samples):
        random.seed(args.seed)
        samples = random.sample(all_samples, args.sample)
        print(f"  Sampled {args.sample:,} / {len(all_samples):,} (seed={args.seed})")
    else:
        samples = all_samples
        print(f"  Full dataset: {len(samples):,} samples")

    stats = collections.defaultdict(int)
    stats["total"] = len(samples)

    glotlid_confusion = collections.Counter()
    afrolid_confusion = collections.Counter()
    severity_counts   = collections.Counter()
    disagreements     = []

    for d in samples:
        text = d.get("text","").strip()

        # FIX5: Track short texts separately, don't skip
        is_short = len(text) < MIN_SHORT_LEN
        if is_short:
            stats["short_text"] += 1
            # Still analyse but mark
            text_for_pred = text
        else:
            text_for_pred = text

        cw_score = core_word_score(text, lang)

        # ── GlotLID ───────────────────────────────────────────────
        gl_preds  = predict(glotlid, text_for_pred, k=args.topk+2)
        gl_verdict, gl_top, gl_top_conf, gl_match, gl_match_conf = is_in_topk(
            gl_preds,
            GLOTLID_TARGET.get(lang,set()),
            GLOTLID_ACCEPTABLE.get(lang,set()),
            args.topk, args.min_conf
        )
        gl_passes = gl_verdict in ("target","acceptable")

        # FIX6: Strict accuracy (target only)
        if gl_verdict == "target":   stats["gl_strict"] += 1
        if gl_passes:                stats["gl_relaxed"] += 1
        else:
            stats["gl_wrong"] += 1
            glotlid_confusion[gl_top] += 1

        # ── AfroLID ───────────────────────────────────────────────
        af_preds = predict(afrolid, text_for_pred, k=args.topk+2) if afrolid else []
        if af_preds:
            af_verdict, af_top, af_top_conf, af_match, af_match_conf = is_in_topk(
                af_preds,
                AFROLID_TARGET.get(lang,set()),
                AFROLID_ACCEPTABLE.get(lang,set()),
                args.topk, args.min_conf
            )
            af_passes = af_verdict in ("target","acceptable")

            if af_verdict == "target": stats["af_strict"] += 1
            if af_passes:              stats["af_relaxed"] += 1
            else:
                stats["af_wrong"] += 1
                afrolid_confusion[af_top] += 1

            # ── Disagreement analysis ─────────────────────────────
            if gl_passes and af_passes:
                stats["agreement"] += 1

            elif gl_passes and not af_passes:
                # GlotLID accepts, AfroLID rejects → flag for review
                stats["gl_pass_af_fail"] += 1
                sev, diff = disagreement_severity(gl_match_conf, af_top_conf)
                severity_counts[sev] += 1

                if len(disagreements) < 200:
                    disagreements.append({
                        "lang":         lang,
                        "text":         text[:200],
                        "is_short":     is_short,
                        "core_score":   cw_score,  # FIX3
                        "glotlid_top":  gl_top,
                        "glotlid_match":gl_match,
                        "glotlid_conf": round(gl_match_conf,3),
                        "afrolid_top":  af_top,
                        "afrolid_conf": round(af_top_conf,3),
                        "type":         "gl_pass_af_fail",
                        "severity":     sev,        # FIX2
                        "conf_diff":    round(diff,3),
                    })

            elif not gl_passes and af_passes:
                stats["gl_fail_af_pass"] += 1
                sev, diff = disagreement_severity(gl_top_conf, af_match_conf)
                severity_counts[sev] += 1

                if len(disagreements) < 200:
                    disagreements.append({
                        "lang":         lang,
                        "text":         text[:200],
                        "is_short":     is_short,
                        "core_score":   cw_score,
                        "glotlid_top":  gl_top,
                        "glotlid_conf": round(gl_top_conf,3),
                        "afrolid_top":  af_top,
                        "afrolid_match":af_match,
                        "afrolid_conf": round(af_match_conf,3),
                        "type":         "gl_fail_af_pass",
                        "severity":     sev,
                        "conf_diff":    round(diff,3),
                    })
            else:
                stats["both_wrong"] += 1

    # ── Export disagreements ──────────────────────────────────────
    if disagreements:
        dpath = OUT / f"{lang}_tier{args.tier}_disagreements_v2.jsonl"
        with open(dpath,"w") as f:
            for d in disagreements:
                f.write(json.dumps(d, ensure_ascii=False)+"\n")
        print(f"  → {len(disagreements)} disagreements → {dpath.name}")

    # ── Language report ───────────────────────────────────────────
    t = max(stats["total"],1)
    gl_strict_pct  = stats["gl_strict"]  / t * 100
    gl_relaxed_pct = stats["gl_relaxed"] / t * 100
    af_strict_pct  = stats["af_strict"]  / t * 100 if afrolid else None
    af_relaxed_pct = stats["af_relaxed"] / t * 100 if afrolid else None

    print(f"\n  {'─'*60}")
    print(f"  {lang.upper()} — {stats['total']:,} samples "
          f"(short={stats['short_text']:,})")

    # FIX6: Strict and relaxed purity
    print(f"  GlotLID strict  : {gl_strict_pct:.2f}%  "
          f"(exact target match, top-{args.topk})")
    print(f"  GlotLID relaxed : {gl_relaxed_pct:.2f}%  "
          f"(target + acceptable Bantu family)")
    if afrolid:
        print(f"  AfroLID strict  : {af_strict_pct:.2f}%")
        print(f"  AfroLID relaxed : {af_relaxed_pct:.2f}%")
        agree_pct = stats["agreement"]/t*100
        print(f"  Agreement       : {agree_pct:.2f}%")
        print(f"  GL✓ AF✗         : {stats['gl_pass_af_fail']:,} "
              f"({stats['gl_pass_af_fail']/t*100:.2f}%) ← review")
        print(f"  GL✗ AF✓         : {stats['gl_fail_af_pass']:,} "
              f"({stats['gl_fail_af_pass']/t*100:.2f}%)")
        print(f"  Both wrong      : {stats['both_wrong']:,} "
              f"({stats['both_wrong']/t*100:.2f}%)")
        if severity_counts:
            print(f"  Severity breakdown:")
            for sev, cnt in severity_counts.most_common():
                print(f"    {sev:<30} {cnt:>5,} ({cnt/max(stats['gl_pass_af_fail']+stats['gl_fail_af_pass'],1)*100:.1f}%)")

    if glotlid_confusion:
        print(f"  GlotLID top-5 wrong: {dict(glotlid_confusion.most_common(5))}")
    if afrolid and afrolid_confusion:
        print(f"  AfroLID top-5 wrong: {dict(afrolid_confusion.most_common(5))}")

    all_results.append({
        "lang":              lang,
        "total":             stats["total"],
        "short_text":        stats["short_text"],
        "gl_strict_pct":     round(gl_strict_pct,2),
        "gl_relaxed_pct":    round(gl_relaxed_pct,2),
        "af_strict_pct":     round(af_strict_pct,2) if afrolid else None,
        "af_relaxed_pct":    round(af_relaxed_pct,2) if afrolid else None,
        "agreement_pct":     round(stats["agreement"]/t*100,2) if afrolid else None,
        "gl_pass_af_fail":   stats["gl_pass_af_fail"],
        "gl_fail_af_pass":   stats["gl_fail_af_pass"],
        "both_wrong":        stats["both_wrong"],
        "severity":          dict(severity_counts),
        "gl_confusion":      dict(glotlid_confusion.most_common(10)),
        "af_confusion":      dict(afrolid_confusion.most_common(10)) if afrolid else {},
    })


# ── Grand summary ─────────────────────────────────────────────────
print(f"\n{'='*65}")
print(f"  GRAND SUMMARY — TIER {args.tier} | TopK={args.topk} | Seed={args.seed}")
print(f"{'─'*65}")

if afrolid:
    print(f"  {'Lang':<10} {'GL-Strict':>10} {'GL-Relax':>10} "
          f"{'AF-Strict':>10} {'GL✓AF✗%':>9} {'Both✗%':>8}")
    print(f"  {'─'*60}")
    for r in all_results:
        t = max(r["total"],1)
        print(f"  {r['lang']:<10} {r['gl_strict_pct']:>9.2f}% "
              f"{r['gl_relaxed_pct']:>9.2f}% "
              f"{r['af_strict_pct']:>9.2f}% "
              f"{r['gl_pass_af_fail']/t*100:>8.2f}% "
              f"{r['both_wrong']/t*100:>7.2f}%")
else:
    print(f"  {'Lang':<10} {'GL-Strict':>10} {'GL-Relax':>10} {'Short':>8}")
    print(f"  {'─'*45}")
    for r in all_results:
        print(f"  {r['lang']:<10} {r['gl_strict_pct']:>9.2f}% "
              f"{r['gl_relaxed_pct']:>9.2f}% "
              f"{r['short_text']:>7,}")

# Save report
report = {
    "tier":       args.tier,
    "timestamp":  datetime.utcnow().isoformat(),
    "config":     {"topk":args.topk,"seed":args.seed,"min_conf":args.min_conf,
                   "sample":args.sample,"glotlid":args.glotlid,"afrolid":args.afrolid},
    "results":    all_results,
}
rpath = OUT / f"tier{args.tier}_analysis_v2_seed{args.seed}.json"
with open(rpath,"w") as f:
    json.dump(report, f, indent=2)

print(f"\n  Report → {rpath}")
print(f"  Disagreements → {OUT}/*_disagreements_v2.jsonl")
print(f"{'='*65}")
