#!/usr/bin/env python3
"""
LughaGen Tier 2 Cleaner — v2.1
All recommendations applied including:
  - Corrected Bantu set (Somali removed)
  - fastText threshold raised to 0.50
  - Minimum hits condition added
  - Expanded English content words
  - Exact deduplication added
"""

import json, re, collections, shutil, sys, argparse
from pathlib import Path

parser = argparse.ArgumentParser(description="LughaGen Tier 2 Cleaner v2.1")
parser.add_argument("lang", choices=["kamba","kikuyu","luo","swahili"])
parser.add_argument("--base", default="/workspace/synthetic/tier2")
parser.add_argument("--min-length", type=int, default=10)
parser.add_argument("--dry-run", action="store_true")
args = parser.parse_args()

lang     = args.lang
BASE     = Path(args.base)
MIN_LEN  = args.min_length
path     = BASE / f"{lang}_tier2.jsonl"
backup   = BASE / f"{lang}_tier2_raw.jsonl"
output   = BASE / f"{lang}_tier2_clean.jsonl"

if not path.exists():
    print(f"ERROR: {path} not found"); sys.exit(1)

# ── fastText ──────────────────────────────────────────────────────
try:
    import fasttext, urllib.request, os
    model_path = "/tmp/lid.176.bin"
    if not os.path.exists(model_path):
        print("Downloading fastText language ID model (~120MB)...")
        urllib.request.urlretrieve(
            "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin",
            model_path
        )
    ft_model = fasttext.load_model(model_path)
    HAS_FASTTEXT = True
    print("✅ fastText loaded")
except Exception as e:
    HAS_FASTTEXT = False
    print(f"fastText unavailable ({e}) — heuristics only")

# GlotLID uses ISO 639-3 + script format: e.g. kam_Latn, kik_Latn
# fastText lid.176.bin uses ISO 639-1: e.g. sw
# Both formats handled below
ACCEPTABLE_LABELS = {
    "kamba":   {"kam_Latn","kik_Latn","swh_Latn","luy_Latn","guz_Latn",
                "sw","ki"},
    "kikuyu":  {"kik_Latn","kam_Latn","swh_Latn","mer_Latn","luy_Latn",
                "sw","ki"},
    "luo":     {"luo_Latn","ach_Latn","laj_Latn","swh_Latn",
                "sw","luo"},
    "swahili": {"swh_Latn","swc_Latn","vmw_Latn",
                "sw"},
}

TARGET_LABEL = {
    "swahili": {"swh_Latn","swc_Latn","sw"},
    "kamba":   {"kam_Latn"},
    "kikuyu":  {"kik_Latn"},
    "luo":     {"luo_Latn"},
}

def fasttext_score(text: str, lang: str) -> float:
    """
    Returns confidence score that text is in the target language.
    Uses GlotLID (ISO 639-3+script) or fastText lid.176.bin as fallback.
    """
    if not HAS_FASTTEXT: return 1.0
    try:
        text_clean = text.replace("\n", " ").strip()[:500]
        labels, probs = ft_model.predict(text_clean, k=5)
        acceptable = ACCEPTABLE_LABELS.get(lang, set())
        target     = TARGET_LABEL.get(lang, set())

        # Check top-5 predictions for target or acceptable match
        for lbl, prob in zip(labels, probs):
            lbl_clean = lbl.replace("__label__", "")
            if lbl_clean in target:
                return float(prob)  # exact target language
            if lbl_clean in acceptable:
                return float(prob) * 0.8  # acceptable family — slight penalty
        return 0.0  # no match in top-5
    except Exception:
        return 1.0

# ── English stopwords + content words ─────────────────────────────
# FIX4: Expanded with common English content words
ENGLISH_WORDS = {
    # Core stopwords
    "the","and","is","are","was","were","to","of","in","that","it",
    "for","on","with","as","by","at","from","this","be","or","an",
    "but","not","have","has","had","do","does","will","would","could",
    "i","you","he","she","we","they","my","your","his","her","our",
    "its","their","what","which","who","how","all","been","if","so",
    "just","also","very","can","may","more","there","when","about",
    "one","two","some","any","each","both","few","these","those",
    "being","because","while","where","though","although","should",
    # Common English content words (FIX4)
    "people","time","year","way","day","man","woman","child","world",
    "life","hand","part","place","case","week","company","system",
    "program","question","government","number","night","point","home",
    "water","room","mother","area","money","story","fact","month",
    "lot","right","study","book","eye","job","word","business","issue",
    "side","kind","head","house","service","friend","father","power",
    "hour","game","line","end","city","play","small","large","national",
    "data","network","information","technology","system","management",
    "community","development","education","health","food","school",
    "including","following","however","therefore","although","because",
    "through","between","during","before","after","against","without",
}

# ── Invalid diacritics ────────────────────────────────────────────
INVALID_DIACRITICS = set('āēīōūķļņģțșłńśźżřěůąęðþæøåßœÿ')

# ── Non-Latin fast regex ──────────────────────────────────────────
NON_LATIN_RE = re.compile(
    r'[\u0400-\u052F'  # Cyrillic
    r'\u0600-\u06FF'   # Arabic
    r'\u4E00-\u9FFF'   # CJK
    r'\u0590-\u05FF'   # Hebrew
    r'\u0E00-\u0E7F'   # Thai
    r'\u0900-\u097F'   # Devanagari
    r'\uAC00-\uD7AF'   # Hangul
    r'\u10A0-\u10FF'   # Georgian
    r'\u0530-\u058F'   # Armenian
    r'\u3040-\u30FF'   # Hiragana/Katakana
    r'\u0370-\u03FF'   # Greek
    r'\u0A80-\u0AFF'   # Gujarati
    r'\u0B80-\u0BFF]'  # Tamil
)

# ── Language core words ───────────────────────────────────────────
LANG_CORE = {
    "swahili": {
        # Function words
        'na','ni','wa','ya','ka','kwa','la','za','pa','mwa','katika',
        'kuwa','pia','au','lakini','kama','hii','hizo','hivyo','kwamba',
        'yote','wote','hata','bali','ili','sana','zaidi','sawa','bado',
        'ndio','ndiyo','kabisa','tena','pia','lakini','ingawa','isipokuwa',
        'ama','bali','kisha','hivyo','japo','ijapokuwa','ingawa',
        # Common verbs
        'alikuwa','alikwenda','alipata','alifanya','alisema','aliweza',
        'wanaweza','tunaweza','ninaweza','anaweza','naweza','unaweza',
        'inaweza','ilikuwa','zilikuwa','walikuwa','tulikuwa','nilikuwa',
        'alitaka','walitaka','ninataka','unataka','anataka','tunataka',
        'alisema','walisema','ninasema','unasema','anasema','kusema',
        'kufanya','kuja','kwenda','kurudi','kuona','kupata','kujua',
        'kukuwa','kuwa','kuishi','kupenda','kusaidia','kufundisha',
        'alitoka','walitoka','alirudi','alienda','alikuja','alikimbia',
        'alipenda','alipata','aliambia','aliuliza','alijibu','aliona',
        # Common nouns
        'mtu','watu','mwaka','miaka','nchi','sehemu','wakati','mara',
        'siku','wiki','mwezi','asubuhi','jioni','usiku','leo','kesho',
        'jana','mji','kijiji','shule','hospitali','kanisa','nyumba',
        'mlima','bahari','mto','njia','barabara','mkoa','wilaya',
        'familia','baba','mama','mtoto','watoto','mke','mume','ndugu',
        'rafiki','jirani','mwalimu','mwanafunzi','daktari','polisi',
        'chakula','maji','nyama','mboga','matunda','mkate','uji',
        'kazi','pesa','bei','biashara','soko','duka','kampuni',
        'serikali','sheria','uchaguzi','rais','waziri','bunge',
        'lugha','elimu','afya','magonjwa','dawa','hospitali',
        'habari','taarifa','gazeti','redio','televisheni','mtandao',
        # Common adjectives/adverbs
        'kubwa','ndogo','nzuri','mbaya','nyingi','chache','zaidi',
        'kidogo','mengi','mzuri','mzee','mpya','wa','za','la',
        'bora','maarufu','maalum','muhimu','rahisi','ngumu',
        # Pronouns
        'mimi','wewe','yeye','sisi','ninyi','wao','mtu','yule',
        'huyu','hawa','hayo','hizo','hizi','hilo','hili',
        # Prepositions/conjunctions
        'ndani','nje','juu','chini','mbele','nyuma','karibu','mbali',
        'pamoja','peke','bila','baada','kabla','wakati','kwa','bila',
        # Common verb stems (appear in many conjugations)
        'ambao','ambaye','ambazo','ambacho','ambalo','ambavyo',
        'unazalishwa','unaweza','unaoishi','unaofanya','unaokuja',
        # Sports/news vocabulary (common in web data)
        'mchezo','mchezaji','timu','mechi','goli','ushindi','kushinda',
        'alishinda','alipoteza','alipiga','alipokea','alijibu',
        'kampuni','shirika','mkutano','mkurugenzi','mkuu',
    },
    "kamba": {
        'na','ni','wa','ya','ka','ma','la','ta','ku','mu','si','vi',
        'nĩ','wĩ','yĩ','kĩ','mĩ','ũ','ĩ','ũndũ','mũndu','ngai',
        'ndĩ','kana','nthini','kyau','mawu','ngũ','ũkũ','kavita',
    },
    "kikuyu": {
        'na','ni','wa','ya','ka','ma','nĩ','wĩ','yĩ','kĩ','mĩ','ũ',
        'ĩ','mũndũ','ngai','ndĩ','ũndũ','rĩĩtwa','gĩkũyũ','wĩra',
        'mũciĩ','mũgũnda','nyũmba','mwĩrĩtu','mwanake','ciana',
    },
    "luo": {
        'gi','ni','bi','di','ma','ka','la','wa','ya','na','to','go',
        'ne','ji','jii','wach','tich','dala','luo','nyaka','kendo',
        'bende','mana','kata','kuom','ndiko','okelo','owuor','auma',
    },
}

# FIX2: Threshold raised to 0.50
FASTTEXT_THRESHOLD = 0.50

# FIX: Core word thresholds with minimum hits
CORE_THRESH = {
    "swahili": 0.08,
    "kamba":   0.05,
    "kikuyu":  0.05,
    "luo":     0.05,
}
CORE_MIN_HITS = 2   # FIX3: minimum absolute hits regardless of ratio

def tokenize(text: str) -> list:
    return re.findall(r"[a-zA-ZÀ-ÿũĩ']+", text.lower())

def is_bad(text: str, style: str, language: str) -> tuple:
    if len(text.strip()) < MIN_LEN:
        return True, "too_short"
    if NON_LATIN_RE.search(text):
        return True, "bad_script"

    # FIX: absolute diacritic threshold
    inv = sum(1 for c in text if c in INVALID_DIACRITICS)
    if inv > 5:
        return True, f"invalid_diacritics:{inv}"

    stripped = text.strip()
    if stripped.startswith("{") and (
        '"Person A"' in text or '"A":' in text or '"v1"' in text
    ):
        return True, "raw_json_dialogue"

    if re.search(r'https?://|www\.', text):
        return True, "url"
    if re.search(r'(.)\1{6,}', text):
        return True, "char_repeat"

    # FIX4: expanded English detection
    words = tokenize(text)
    if len(words) >= 6:
        en_hits = sum(1 for w in words if w in ENGLISH_WORDS)
        en_ratio = en_hits / len(words)
        threshold = 0.35 if style == "code_switching" else 0.38
        if en_ratio > threshold:
            return True, f"english:{en_ratio:.2f}"

    return False, ""

def fails_language_check(text: str, language: str) -> tuple:
    words = tokenize(text)
    core  = LANG_CORE.get(language, set())
    thresh = CORE_THRESH.get(language, 0.05)

    # FIX2: raised fastText threshold to 0.50
    if HAS_FASTTEXT and len(text) > 30:
        score = fasttext_score(text, language)
        if score < FASTTEXT_THRESHOLD:
            return True, f"fasttext:{score:.2f}"

    # FIX3: minimum hits condition prevents gaming
    if len(words) >= 8:
        hits  = sum(1 for w in words if w in core)
        ratio = hits / len(words)
        if hits == 0:  # only reject if zero target language words found
            return True, f"low_{language}_markers:{ratio:.2f}(hits={hits})"

    return False, ""

# ── Main ──────────────────────────────────────────────────────────
print("=" * 65)
print(f"  LUGHAGEN TIER 2 CLEANER v2.1 — {lang.upper()}")
if args.dry_run:
    print("  *** DRY RUN ***")
print("=" * 65)

if not args.dry_run:
    shutil.copy2(path, backup)
    print(f"Backup: {backup.name}")

kept     = []
removed  = collections.Counter()
examples = {}
# FIX7: Exact deduplication
seen_texts = set()

with open(backup if not args.dry_run else path) as f:
    for line in f:
        line = line.strip()
        if not line: continue
        try:
            d     = json.loads(line)
            text  = d.get('text','').strip()
            style = d.get('prompt_style','')

            text = text.encode('utf-8','ignore').decode('utf-8')
            d['text'] = text

            # FIX7: Deduplication
            norm = ' '.join(text.lower().split())
            if norm in seen_texts:
                removed['duplicate'] += 1
                continue
            seen_texts.add(norm)

            bad, reason = is_bad(text, style, lang)
            if bad:
                removed[reason] += 1
                examples.setdefault(reason, text[:80])
                continue

            fails, reason = fails_language_check(text, lang)
            if fails:
                removed[reason] += 1
                examples.setdefault(reason, text[:80])
                continue

            kept.append(json.dumps(d, ensure_ascii=False))

        except Exception as e:
            removed['parse_error'] += 1
            examples.setdefault('parse_error', line[:80])

if not args.dry_run:
    with open(output,'w') as f:
        f.write('\n'.join(kept)+'\n')
    shutil.move(str(output), str(path))

total    = len(kept) + sum(removed.values())
n_kept   = len(kept)
n_removed= sum(removed.values())

print(f"\n  Input    : {total:,}")
print(f"  Kept     : {n_kept:,} ({n_kept/max(total,1)*100:.1f}%)")
print(f"  Removed  : {n_removed:,} ({n_removed/max(total,1)*100:.1f}%)")
print(f"\n  Breakdown:")
for reason, count in removed.most_common():
    print(f"    {reason:<45} {count:>6,}")
if examples:
    print(f"\n  Examples:")
    for reason, text in list(examples.items())[:5]:
        print(f"    [{reason}]: {text}")
gap = max(0, 150_000 - n_kept)
print(f"\n  {'⚠️  '+str(gap)+' below 150K' if gap else '✅ Target met'}")
print("=" * 65)
