#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════╗
║      LUGHAGEN TIER 1 GLOTLID VERIFICATION CLEANER               ║
║                                                                  ║
║  Applies positive GlotLID language verification to Tier 1.       ║
║  Tier 1 was already cleaned by clean_synthetic_v4.py and         ║
║  targeted GlotLID contamination removal — this adds positive     ║
║  verification to catch any remaining wrong-language samples.     ║
║                                                                  ║
║  Usage:                                                          ║
║    python3 clean_tier1_glotlid.py kamba --dry-run                ║
║    python3 clean_tier1_glotlid.py --all                          ║
║    python3 clean_tier1_glotlid.py --all --dry-run                ║
║    python3 clean_tier1_glotlid.py luo --export-rejected rej.jsonl║
╚══════════════════════════════════════════════════════════════════╝
"""

import json, re, collections, shutil, argparse, os
from pathlib import Path
from datetime import datetime

# ── CLI ───────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="LughaGen Tier 1 GlotLID Verification Cleaner"
)
parser.add_argument("lang", nargs="?",
                    choices=["kamba","kikuyu","luo","swahili"])
parser.add_argument("--base", default="/workspace/synthetic/tier1")
parser.add_argument("--glotlid", default="/workspace/models/glotlid.bin")
parser.add_argument("--glotlid-threshold", type=float, default=0.50)
parser.add_argument("--dry-run", action="store_true")
parser.add_argument("--all", action="store_true")
parser.add_argument("--export-rejected", type=str, default=None)
args = parser.parse_args()

if not args.all and not args.lang:
    parser.error("Specify a language or use --all")

langs_to_run = (["kamba","kikuyu","luo","swahili"]
                if args.all else [args.lang])


# ── GlotLID loader ────────────────────────────────────────────────
def load_glotlid(model_path):
    try:
        import fasttext
        candidates = [
            (model_path,                       "GlotLID"),
            ("/workspace/models/lid.176.bin",  "fastText-lid176"),
        ]
        for path, name in candidates:
            if os.path.exists(path):
                m = fasttext.load_model(path)
                print(f"✅ {name} loaded from {path}")
                return m, name
        print("⚠️  No language ID model found")
        return None, "none"
    except Exception as e:
        print(f"⚠️  Load failed: {e}")
        return None, "none"


# ── GlotLID codes ─────────────────────────────────────────────────
GLOTLID_TARGET = {
    "kamba":   {"kam_Latn"},
    "kikuyu":  {"kik_Latn"},
    "luo":     {"luo_Latn"},
    "swahili": {"swh_Latn","swc_Latn"},
}

# Confirmed acceptable + confirmed GlotLID false positives
# (verified by manual inspection during Tier 1 audit)
GLOTLID_ACCEPTABLE = {
    "kamba": {
        "kam_Latn","kik_Latn","luy_Latn","guz_Latn","mer_Latn",
        "nim_Latn",  # Nimadi (India)     — FP: ĩ/ũ diacritic confusion
        "dad_Latn",  # Marik (PNG)        — FP
        "yaa_Latn",  # Yagua (Peru)       — FP
        "chk_Latn",  # Chuukese (Pacific) — FP
        "mbl_Latn",  # Matigsalug (Phil.) — FP confirmed in Tier 1 audit
        "amu_Latn",  # Amuzgo (Mexico)    — FP confirmed in Tier 1 audit
        "xrb_Latn",  # rare               — FP confirmed in Tier 1 audit
    },
    "kikuyu": {
        "kik_Latn","kam_Latn","mer_Latn","luy_Latn","thk_Latn",
        "des_Latn","cbc_Latn","bsn_Latn","tav_Latn","myy_Latn",
        "bam_Latn",  # Bambara — FP confirmed in Tier 1 audit
        "sri_Latn",  # FP confirmed in Tier 1 audit
        "kyz_Latn",  # Kyzyl (Siberia) — FP confirmed in Tier 1 audit
    },
    "luo": {
        "luo_Latn","ach_Latn","laj_Latn","adh_Latn","alz_Latn",
        "bam_Latn","tuv_Latn",
        "vmw_Latn",  # Makonde — confirmed in Tier 1 audit
        "tsc_Latn",  # FP confirmed in Tier 1 audit
    },
    "swahili": {
        "swh_Latn","swc_Latn","vmw_Latn","hay_Latn","ssw_Latn",
        "kin_Latn",  # Kinyarwanda — acceptable Bantu
    },
}

# Confirmed contamination — reject at high confidence
REJECT_LANGUAGES = {
    "kamba":   {"eng_Latn","fra_Latn","por_Latn","spa_Latn","deu_Latn",
                "swh_Latn","swc_Latn"},
    "kikuyu":  {"eng_Latn","fra_Latn","por_Latn","spa_Latn","deu_Latn",
                "swh_Latn","swc_Latn"},
    "luo":     {"eng_Latn","fra_Latn","por_Latn","spa_Latn","deu_Latn",
                "swh_Latn","swc_Latn"},
    "swahili": {"eng_Latn","fra_Latn","por_Latn","spa_Latn","deu_Latn"},
}

MIN_GLOTLID_LEN = 25

# ── Core words (heuristic fallback) ──────────────────────────────
LANG_CORE = {
    "swahili": {
        "na","ni","wa","ya","ka","kwa","la","za","pa","mwa","katika",
        "kuwa","pia","au","lakini","kama","hii","hizo","hivyo","kwamba",
        "yote","wote","hata","bali","ili","sana","zaidi","sawa","bado",
        "alikuwa","wanaweza","tunaweza","ninaweza","anaweza","naweza",
        "alipata","alisema","alifanya","mtu","watu","mwaka","nchi",
        "wakati","siku","leo","nyumba","chakula","maji","kazi","pesa",
        "mimi","wewe","yeye","sisi","wao","ambao","ambaye","ambazo",
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

CORE_THRESHOLD = {"swahili":0.08,"kamba":0.05,"kikuyu":0.05,"luo":0.05}

# ── Invalid diacritics (Baltic/Nordic) ───────────────────────────
INVALID_DIACRITICS = set("āēīōūķļņģțșłńśźżřěůąęðþæøåßœÿ")


def tokenize(text):
    return re.findall(r"[a-zA-ZÀ-ÿũĩ']+", text.lower())


def has_enough_core_words(words, lang):
    """Requires hits>=2 AND ratio>=threshold."""
    core   = LANG_CORE.get(lang, set())
    thresh = CORE_THRESHOLD.get(lang, 0.05)
    if len(words) < 8:
        return True
    hits  = sum(1 for w in words if w in core)
    ratio = hits / len(words)
    return not (hits < 2 and ratio < thresh)


def glotlid_verify(model, text, lang, threshold, words):
    """
    Positive language verification.
    Returns (passes, detected_lang, confidence).
    """
    if model is None:
        return True, "no_model", 0.0
    if len(text.strip()) < MIN_GLOTLID_LEN:
        return True, "short_skip", 0.0
    try:
        text_clean = text.replace("\n"," ").strip()[:500]
        labels, probs = model.predict(text_clean, k=5)
        top_label = labels[0].replace("__label__","")
        top_prob  = float(probs[0])

        target     = GLOTLID_TARGET.get(lang, set())
        acceptable = GLOTLID_ACCEPTABLE.get(lang, set())
        reject     = REJECT_LANGUAGES.get(lang, set())

        # Hard reject — high-confidence confirmed contamination
        if top_label in reject and top_prob >= threshold:
            return False, top_label, top_prob

        # Accept — exact target
        if top_label in target:
            return True, top_label, top_prob

        # Accept — acceptable family or confirmed false positive
        if top_label in acceptable:
            return True, top_label, top_prob

        # Low confidence — gate on core words
        if top_prob < threshold:
            return has_enough_core_words(words, lang), top_label, top_prob

        # High confidence wrong language
        return False, top_label, top_prob

    except Exception:
        return True, "error", 0.0


def clean_tier1_language(lang, base, glotlid_model, glotlid_threshold,
                          glotlid_name, dry_run, export_rejected_path):

    path   = base / f"{lang}_tier1.jsonl"
    backup = base / f"{lang}_tier1_preglotlid2.jsonl"
    output = base / f"{lang}_tier1_glotlid_clean.jsonl"

    if not path.exists():
        print(f"⚠️  {path} not found")
        return {}

    if not dry_run:
        shutil.copy2(path, backup)

    kept            = []
    removed         = collections.Counter()
    examples        = {}
    rejected_export = []
    errors          = 0

    with open(backup if not dry_run else path) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                d    = json.loads(line)
                text = d.get("text","").strip()

                # Sanitize surrogates
                text = text.encode("utf-8","ignore").decode("utf-8")
                d["text"] = text

                words = tokenize(text)

                # Invalid diacritics (Baltic/Nordic — confirmed bad)
                inv = sum(1 for c in text if c in INVALID_DIACRITICS)
                if inv > 5:
                    removed[f"invalid_diacritics:{inv}"] += 1
                    examples.setdefault("invalid_diacritics", text[:100])
                    if export_rejected_path:
                        rejected_export.append(json.dumps(
                            {"reason":"invalid_diacritics","lang":lang,
                             "text":text[:200]}, ensure_ascii=False))
                    continue

                # GlotLID positive verification
                passes, detected, conf = glotlid_verify(
                    glotlid_model, text, lang, glotlid_threshold, words
                )
                if not passes:
                    reason = f"glotlid:{detected}({conf:.2f})"
                    removed[reason] += 1
                    examples.setdefault(reason, text[:100])
                    if export_rejected_path:
                        rejected_export.append(json.dumps(
                            {"reason":reason,"lang":lang,
                             "detected":detected,"conf":conf,
                             "text":text[:200]}, ensure_ascii=False))
                    continue

                # Core word check (fallback)
                if not has_enough_core_words(words, lang):
                    removed[f"low_{lang}_markers"] += 1
                    examples.setdefault(f"low_{lang}_markers", text[:100])
                    continue

                kept.append(json.dumps(d, ensure_ascii=False))

            except Exception:
                errors += 1
                examples.setdefault("parse_error", line[:80])

    removed["parse_error"] += errors

    if not dry_run:
        with open(output,"w") as f:
            f.write("\n".join(kept)+"\n")
        shutil.move(str(output), str(path))
        if export_rejected_path and rejected_export:
            with open(export_rejected_path,"a") as f:
                f.write("\n".join(rejected_export)+"\n")
            print(f"  Rejected → {export_rejected_path}")

    return {
        "lang":    lang,
        "total":   len(kept)+sum(removed.values()),
        "kept":    len(kept),
        "removed": dict(removed),
        "examples":examples,
        "model":   glotlid_name,
        "dry_run": dry_run,
    }


def print_report(r):
    if not r: return
    lang    = r["lang"]
    total   = r["total"]
    kept    = r["kept"]
    n_rem   = sum(r["removed"].values())
    print(f"\n{'═'*68}")
    print(f"  {lang.upper()} | {total:,} → {kept:,} kept "
          f"({n_rem:,} removed, {n_rem/max(total,1)*100:.2f}%)"
          + (" [DRY RUN]" if r["dry_run"] else ""))
    print(f"  Model: {r['model']}")
    print(f"{'─'*68}")
    for reason, count in sorted(r["removed"].items(), key=lambda x:-x[1]):
        if count > 0:
            pct  = count/max(total,1)*100
            flag = "  ⚠️ " if pct > 0.5 else "    "
            print(f"  {flag}{reason:<45} {count:>6,} ({pct:.3f}%)")
    if r["examples"]:
        print("\n  Examples removed:")
        for reason, text in list(r["examples"].items())[:4]:
            print(f"    [{reason}]")
            print(f"    {text[:120]}")
    gap = max(0, 99_000 - kept)  # Tier 1 target ~99K per language
    removed_pct = n_rem/max(total,1)*100
    if removed_pct < 1.0:
        print(f"\n  ✅ Clean ({removed_pct:.2f}% removed — within expected range)")
    else:
        print(f"\n  ⚠️  {removed_pct:.2f}% removed — review before accepting")


def main():
    print("╔" + "═"*66 + "╗")
    print("║  LUGHAGEN TIER 1 GLOTLID VERIFICATION CLEANER".ljust(67) + "║")
    print(f"║  {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}".ljust(67) + "║")
    if args.dry_run:
        print("║  *** DRY RUN — no files modified ***".ljust(67) + "║")
    print("╚" + "═"*66 + "╝\n")
    print(f"  Languages  : {', '.join(langs_to_run)}")
    print(f"  Base dir   : {args.base}")
    print(f"  GlotLID    : {args.glotlid}")
    print(f"  Threshold  : {args.glotlid_threshold}")
    if args.export_rejected:
        print(f"  Export rej : {args.export_rejected}")
    print()

    glotlid_model, glotlid_name = load_glotlid(args.glotlid)
    base = Path(args.base)

    all_results = []
    grand_total = grand_kept = 0

    for lang in langs_to_run:
        print(f"Processing {lang}...")
        r = clean_tier1_language(
            lang                = lang,
            base                = base,
            glotlid_model       = glotlid_model,
            glotlid_threshold   = args.glotlid_threshold,
            glotlid_name        = glotlid_name,
            dry_run             = args.dry_run,
            export_rejected_path= args.export_rejected,
        )
        print_report(r)
        if r:
            all_results.append(r)
            grand_total += r["total"]
            grand_kept  += r["kept"]

    if len(langs_to_run) > 1:
        grand_removed = grand_total - grand_kept
        print(f"\n{'═'*68}")
        print(f"  TOTAL: {grand_total:,} → {grand_kept:,} kept "
              f"({grand_removed:,} removed, "
              f"{grand_removed/max(grand_total,1)*100:.2f}%)")
        print(f"{'─'*68}")
        for r in all_results:
            n   = r["kept"]
            rem = sum(r["removed"].values())
            print(f"  {'✅' if rem/max(r['total'],1)<0.01 else '⚠️ '} "
                  f"{r['lang']:<10} {n:>7,} ({rem} removed)")
        print(f"{'═'*68}")
        print(f"\n  Backups: *_tier1_preglotlid2.jsonl")
        print(f"  Run audit: python3 /workspace/scripts/audit_dataset.py --full")


if __name__ == "__main__":
    main()
