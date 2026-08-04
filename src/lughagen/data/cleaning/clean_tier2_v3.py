#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════╗
║      LUGHAGEN TIER 2 DATA CLEANER — v3.1 (GlotLID Edition)      ║
║  FIX1: Dedup AFTER filtering                                     ║
║  FIX2: hits<2 AND ratio<thresh                                   ║
║  FIX3: Swahili check gated on GlotLID unavailable                ║
║  FIX4: Low-confidence GlotLID gated on core-word check           ║
║  NEW:  --export-rejected flag                                     ║
║  Usage:                                                          ║
║    python3 clean_tier2_v3.py swahili                             ║
║    python3 clean_tier2_v3.py kamba --dry-run                     ║
║    python3 clean_tier2_v3.py --all                               ║
║    python3 clean_tier2_v3.py luo --export-rejected rejected.jsonl║
╚══════════════════════════════════════════════════════════════════╝
"""

import json, re, collections, shutil, sys, argparse, os
from pathlib import Path
from datetime import datetime

parser = argparse.ArgumentParser(description="LughaGen Tier 2 Cleaner v3.1")
parser.add_argument("lang", nargs="?", choices=["kamba","kikuyu","luo","swahili"])
parser.add_argument("--base", default="/workspace/synthetic/tier2")
parser.add_argument("--min-length", type=int, default=10)
parser.add_argument("--glotlid", default="/workspace/models/glotlid.bin")
parser.add_argument("--glotlid-threshold", type=float, default=0.50)
parser.add_argument("--dry-run", action="store_true")
parser.add_argument("--all", action="store_true")
parser.add_argument("--export-rejected", type=str, default=None)
args = parser.parse_args()

if not args.all and not args.lang:
    parser.error("Specify a language or use --all")

langs_to_run = (["kamba","kikuyu","luo","swahili"] if args.all else [args.lang])


def load_glotlid(model_path):
    try:
        import fasttext
        candidates = [
            (model_path, "GlotLID"),
            ("/workspace/models/lid.176.bin", "fastText-lid176"),
            ("/tmp/lid.176.bin", "fastText-lid176"),
        ]
        for path, name in candidates:
            if os.path.exists(path):
                m = fasttext.load_model(path)
                print(f"✅ Language ID: {name} ({path})")
                return m, name
        print("⚠️  No language ID model — heuristics only")
        return None, "none"
    except Exception as e:
        print(f"⚠️  LID load failed ({e}) — heuristics only")
        return None, "none"


GLOTLID_TARGET = {
    "kamba":   {"kam_Latn"},
    "kikuyu":  {"kik_Latn"},
    "luo":     {"luo_Latn"},
    "swahili": {"swh_Latn","swc_Latn"},
}

GLOTLID_ACCEPTABLE = {
    "kamba": {
        "kam_Latn","kik_Latn","luy_Latn","guz_Latn","mer_Latn",
        "nim_Latn","dad_Latn","yaa_Latn","chk_Latn",
    },
    "kikuyu": {
        "kik_Latn","kam_Latn","mer_Latn","luy_Latn","thk_Latn",
        "des_Latn","cbc_Latn","bsn_Latn","tav_Latn","myy_Latn",
    },
    "luo": {
        "luo_Latn","ach_Latn","laj_Latn","adh_Latn","alz_Latn",
        "bam_Latn","tuv_Latn",
    },
    "swahili": {
        "swh_Latn","swc_Latn","vmw_Latn","hay_Latn","ssw_Latn",
    },
}

REJECT_LANGUAGES = {
    "kamba":   {"eng_Latn","fra_Latn","por_Latn","spa_Latn","deu_Latn","swh_Latn","swc_Latn"},
    "kikuyu":  {"eng_Latn","fra_Latn","por_Latn","spa_Latn","deu_Latn","swh_Latn","swc_Latn"},
    "luo":     {"eng_Latn","fra_Latn","por_Latn","spa_Latn","deu_Latn","swh_Latn","swc_Latn"},
    "swahili": {"eng_Latn","fra_Latn","por_Latn","spa_Latn","deu_Latn"},
}

MIN_GLOTLID_LEN = 25

ENGLISH_WORDS = {
    "the","and","is","are","was","were","to","of","in","that","it",
    "for","on","with","as","by","at","from","this","be","or","an",
    "but","not","have","has","had","do","does","will","would","could",
    "i","you","he","she","we","they","my","your","his","her","our",
    "its","their","what","which","who","how","all","been","if","so",
    "just","also","very","can","may","more","there","when","about",
    "one","two","some","any","each","both","few","these","those",
    "being","because","while","where","though","although","should",
    "people","time","year","way","day","man","woman","world","life",
    "hand","part","place","case","week","company","system","program",
    "question","government","number","night","point","home","water",
    "room","mother","area","money","story","fact","month","right",
    "study","book","job","word","business","issue","side","kind",
    "head","house","service","friend","father","power","hour","game",
    "line","end","city","play","small","large","national","data",
    "network","information","technology","management","community",
    "development","education","health","food","school","including",
    "following","however","therefore","through","between","during",
    "before","after","against","without","within","upon","around",
}

INVALID_DIACRITICS = set("āēīōūķļņģțșłńśźżřěůąęðþæøåßœÿ")

NON_LATIN_RE = re.compile(
    r"[\u0400-\u052F\u0600-\u06FF\u4E00-\u9FFF\u0590-\u05FF"
    r"\u0E00-\u0E7F\u0900-\u097F\uAC00-\uD7AF\u10A0-\u10FF"
    r"\u0530-\u058F\u3040-\u30FF\u0370-\u03FF\u0A80-\u0AFF\u0B80-\u0BFF]"
)

LANG_CORE = {
    "swahili": {
        "na","ni","wa","ya","ka","kwa","la","za","pa","mwa","katika",
        "kuwa","pia","au","lakini","kama","hii","hizo","hivyo","kwamba",
        "yote","wote","hata","bali","ili","sana","zaidi","sawa","bado",
        "alikuwa","wanaweza","tunaweza","ninaweza","anaweza","naweza",
        "alipata","alisema","alifanya","ninataka","anataka","mtu","watu",
        "mwaka","nchi","wakati","siku","leo","kesho","jana","mji",
        "shule","hospitali","nyumba","chakula","maji","kazi","pesa",
        "kubwa","ndogo","nzuri","mbaya","nyingi","kidogo","mengi",
        "mimi","wewe","yeye","sisi","wao","huyu","hawa","pamoja",
        "ambao","ambaye","ambazo","mchezaji","mchezo","kusema",
        "kufanya","kwenda","kupata","kujua","kuwa","kupenda",
    },
    "kamba": {
        "na","ni","wa","ya","ka","ma","la","ta","ku","mu","si","vi",
        "nĩ","wĩ","yĩ","kĩ","mĩ","ũ","ĩ","ũndũ","mũndu","ngai","ndĩ",
        "kana","nthini","kyau","mawu","ngũ","ũkũ","mũsyi","mwanake",
        "ĩtũmi","kũthima","kũsoma","kwĩ","ĩno","ĩyo","ĩyi",
    },
    "kikuyu": {
        "na","ni","wa","ya","ka","ma","nĩ","wĩ","yĩ","kĩ","mĩ","ũ",
        "ĩ","mũndũ","ngai","ndĩ","ũndũ","rĩĩtwa","gĩkũyũ","wĩra",
        "mũciĩ","mũgũnda","nyũmba","mwĩrĩtu","mwanake","ciana",
        "mũrũ","mũirĩtu","kũrĩa","kũnywa","kũigua","kũona",
    },
    "luo": {
        "gi","ni","bi","di","ma","ka","la","wa","ya","na","to","go",
        "ne","ji","jii","wach","tich","dala","luo","nyaka","kendo",
        "bende","mana","kata","kuom","ndiko","okelo","owuor","auma",
        "nyaluo","joluo","wuod","nyar","miyo","yudo","neno",
    },
}

CORE_THRESHOLD = {"swahili":0.08,"kamba":0.05,"kikuyu":0.05,"luo":0.05}


def tokenize(text):
    return re.findall(r"[a-zA-ZÀ-ÿũĩ']+", text.lower())


def has_enough_core_words(words, lang):
    core   = LANG_CORE.get(lang, set())
    thresh = CORE_THRESHOLD.get(lang, 0.05)
    if len(words) < 8:
        return True
    hits  = sum(1 for w in words if w in core)
    ratio = hits / len(words)
    return not (hits < 2 and ratio < thresh)


def glotlid_check(model, text, lang, threshold, words):
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
        if top_label in reject and top_prob >= threshold:
            return False, top_label, top_prob
        if top_label in target:
            return True, top_label, top_prob
        if top_label in acceptable:
            return True, top_label, top_prob
        if top_prob < threshold:
            return has_enough_core_words(words, lang), top_label, top_prob
        return False, top_label, top_prob
    except Exception:
        return True, "error", 0.0


def is_bad_structural(text, style, min_length):
    if len(text.strip()) < min_length:
        return True, "too_short"
    if NON_LATIN_RE.search(text):
        return True, "bad_script"
    inv = sum(1 for c in text if c in INVALID_DIACRITICS)
    if inv > 5:
        return True, f"invalid_diacritics:{inv}"
    stripped = text.strip()
    if stripped.startswith("{") and any(
        k in text for k in ['"Person A"','"A":','"v1"','"v2"']
    ):
        return True, "raw_json_dialogue"
    if re.search(r"https?://|www\.", text):
        return True, "url"
    if re.search(r"(.)\1{6,}", text):
        return True, "char_repeat"
    if re.search(r"&#\d+;|&[a-zA-Z]+;", text):
        return True, "html_entities"
    words = tokenize(text)
    if len(words) >= 6:
        en_hits  = sum(1 for w in words if w in ENGLISH_WORDS)
        en_ratio = en_hits / len(words)
        threshold = 0.35 if style == "code_switching" else 0.38
        if en_ratio > threshold:
            return True, f"english:{en_ratio:.2f}"
    return False, ""


def fails_positive_check(text, lang, words, glotlid_model,
                          glotlid_threshold, glotlid_name):
    passes, detected, conf = glotlid_check(
        glotlid_model, text, lang, glotlid_threshold, words
    )
    if not passes:
        return True, f"glotlid:{detected}({conf:.2f})"
    if not has_enough_core_words(words, lang):
        return True, f"low_{lang}_markers"
    if lang == "swahili" and glotlid_name == "none" and len(words) >= 8:
        sw_core = LANG_CORE["swahili"]
        if sum(1 for w in words if w in sw_core) == 0:
            return True, "no_swahili_markers"
    return False, ""


def clean_language(lang, base, min_length, glotlid_model,
                   glotlid_threshold, glotlid_name,
                   dry_run, export_rejected_path=None):
    path   = base / f"{lang}_tier2.jsonl"
    backup = base / f"{lang}_tier2_raw.jsonl"
    output = base / f"{lang}_tier2_clean.jsonl"
    if not path.exists():
        print(f"⚠️  {path} not found")
        return {}
    if not dry_run:
        shutil.copy2(path, backup)
    kept    = []
    removed = collections.Counter()
    examples = {}
    rejected_export = []
    errors  = 0
    seen    = set()
    with open(backup if not dry_run else path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d     = json.loads(line)
                text  = d.get("text","").strip()
                style = d.get("prompt_style","")
                text = text.encode("utf-8","ignore").decode("utf-8")
                d["text"] = text
                words = tokenize(text)
                bad, reason = is_bad_structural(text, style, min_length)
                if bad:
                    removed[reason] += 1
                    examples.setdefault(reason, text[:100])
                    if export_rejected_path:
                        rejected_export.append(
                            json.dumps({"reason":reason,"lang":lang,
                                        "text":text[:200]}, ensure_ascii=False))
                    continue
                fails, reason = fails_positive_check(
                    text, lang, words, glotlid_model,
                    glotlid_threshold, glotlid_name
                )
                if fails:
                    removed[reason] += 1
                    examples.setdefault(reason, text[:100])
                    if export_rejected_path:
                        rejected_export.append(
                            json.dumps({"reason":reason,"lang":lang,
                                        "text":text[:200]}, ensure_ascii=False))
                    continue
                norm = " ".join(text.lower().split())
                if norm in seen:
                    removed["duplicate"] += 1
                    continue
                seen.add(norm)
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
            print(f"  Rejected exported → {export_rejected_path}")
    return {"lang":lang,"total":len(kept)+sum(removed.values()),
            "kept":len(kept),"removed":dict(removed),
            "examples":examples,"glotlid_model":glotlid_name,
            "dry_run":dry_run}


def print_report(r):
    if not r: return
    lang    = r["lang"]
    total   = r["total"]
    kept    = r["kept"]
    removed = sum(r["removed"].values())
    print(f"\n{'═'*68}")
    print(f"  {lang.upper()} | {total:,} → {kept:,} "
          f"({removed:,} removed {removed/max(total,1)*100:.1f}%)"
          + (" [DRY RUN]" if r["dry_run"] else ""))
    print(f"  LID model: {r['glotlid_model']}")
    print(f"{'─'*68}")
    for reason, count in sorted(r["removed"].items(), key=lambda x:-x[1]):
        if count > 0:
            pct  = count/max(total,1)*100
            flag = "  ⚠️ " if pct > 1.0 else "    "
            print(f"  {flag}{reason:<45} {count:>6,} ({pct:.2f}%)")
    if r["examples"]:
        print("\n  Examples:")
        for reason, text in list(r["examples"].items())[:5]:
            print(f"    [{reason}]: {text[:100]}")
    gap = max(0, 150_000 - kept)
    print(f"\n  {'⚠️  '+str(gap)+' below 150K' if gap else '✅ Target met'}")


def main():
    print("╔" + "═"*66 + "╗")
    print("║  LUGHAGEN TIER 2 CLEANER v3.1 — GlotLID Edition".ljust(67) + "║")
    print("║  " + datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC").ljust(64) + "║")
    if args.dry_run:
        print("║  *** DRY RUN ***".ljust(67) + "║")
    print("╚" + "═"*66 + "╝\n")
    print(f"  Languages    : {', '.join(langs_to_run)}")
    print(f"  Base dir     : {args.base}")
    print(f"  GlotLID      : {args.glotlid}")
    print(f"  Threshold    : {args.glotlid_threshold}")
    if args.export_rejected:
        print(f"  Export rej.  : {args.export_rejected}")
    print()
    glotlid_model, glotlid_name = load_glotlid(args.glotlid)
    base = Path(args.base)
    all_results = []
    grand_total = grand_kept = 0
    for lang in langs_to_run:
        print(f"\nProcessing {lang}...")
        result = clean_language(
            lang                = lang,
            base                = base,
            min_length          = args.min_length,
            glotlid_model       = glotlid_model,
            glotlid_threshold   = args.glotlid_threshold,
            glotlid_name        = glotlid_name,
            dry_run             = args.dry_run,
            export_rejected_path= args.export_rejected,
        )
        print_report(result)
        if result:
            all_results.append(result)
            grand_total += result["total"]
            grand_kept  += result["kept"]
    if len(langs_to_run) > 1:
        grand_removed = grand_total - grand_kept
        print(f"\n{'═'*68}")
        print(f"  TOTAL: {grand_total:,} → {grand_kept:,} "
              f"({grand_removed:,} removed, "
              f"{grand_removed/max(grand_total,1)*100:.1f}%)")
        for r in all_results:
            k   = r["kept"]
            gap = max(0, 150_000 - k)
            print(f"  {'✅' if gap==0 else '⚠️ '} {r['lang']:<10} "
                  f"{k:>7,} {'✓' if gap==0 else f'({gap:,} short)'}")
        print(f"{'═'*68}")
    if not args.dry_run:
        print("\nNext: python3 /workspace/scripts/postprocess_dedup_v3.py")


if __name__ == "__main__":
    main()
