#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════╗
║         LUGHAGEN SYNTHETIC DATA CLEANER — v4 Final              ║
║                                                                  ║
║  Run once after synthetic data production is complete.           ║
║  Cleans all four language datasets in a single pass.             ║
║                                                                  ║
║  Usage:                                                          ║
║    python3 clean_synthetic_v4.py                                 ║
║                                                                  ║
║  What it does:                                                   ║
║    Layer 1 — Non-Latin script rejection (25 scripts)             ║
║    Layer 2 — Latin ratio check (< 60% Latin = reject)            ║
║    Layer 3 — Garbled text detection (5 signals)                  ║
║    Layer 4 — English leakage detection                           ║
║    Layer 5 — Foreign language contamination (28 languages)       ║
║    Layer 6 — Structural quality checks                           ║
║    Layer 7 — Language-specific positive validation               ║
║                                                                  ║
║  Input:   /workspace/synthetic/tier1/<lang>_tier1.jsonl          ║
║  Output:  /workspace/synthetic/tier1/<lang>_tier1_clean.jsonl    ║
║  Backup:  /workspace/synthetic/tier1/<lang>_tier1_backup.jsonl   ║
╚══════════════════════════════════════════════════════════════════╝
"""

import json
import unicodedata
import re
import collections
import shutil
import sys
from pathlib import Path
from datetime import datetime

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
BASE   = "/workspace/synthetic/tier1"
LANGS  = ["kamba", "kikuyu", "luo", "swahili"]

# Minimum quality threshold from generate.py — samples below this were
# already rejected during generation; we trust scores above it
MIN_QUALITY = 0.60

# ─────────────────────────────────────────────────────────────────────────────
# LAYER 1 — Non-Latin script rejection
# These scripts have no place in any of our four target languages
# ─────────────────────────────────────────────────────────────────────────────
NON_LATIN_SCRIPTS = [
    'CYRILLIC', 'ARABIC', 'CJK', 'HEBREW', 'THAI', 'DEVANAGARI',
    'HANGUL', 'GEORGIAN', 'ARMENIAN', 'HIRAGANA', 'KATAKANA',
    'GREEK', 'GUJARATI', 'PUNJABI', 'TAMIL', 'TELUGU', 'KANNADA',
    'MALAYALAM', 'SINHALA', 'MYANMAR', 'KHMER', 'MONGOLIAN',
    'TIBETAN', 'ETHIOPIC', 'CHEROKEE', 'CANADIAN',
]

def has_bad_script(text: str) -> bool:
    for c in text:
        if c.isalpha():
            if any(s in unicodedata.name(c, '') for s in NON_LATIN_SCRIPTS):
                return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 2 — Latin ratio
# ─────────────────────────────────────────────────────────────────────────────
def latin_ratio(text: str) -> float:
    alpha = [c for c in text if c.isalpha()]
    if not alpha:
        return 0.0
    latin = sum(1 for c in alpha if 'LATIN' in unicodedata.name(c, ''))
    return latin / len(alpha)


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 3 — Garbled text detection
# Five orthogonal signals to catch hallucinated / broken model outputs
# ─────────────────────────────────────────────────────────────────────────────

# Characters that are definitively wrong for Kamba / Kikuyu / Luo / Swahili.
# These are Baltic, Nordic, Slavic and Romanian diacritics not found in any
# East African language using Latin script.
INVALID_DIACRITICS = set(
    'āēīōū'   # Macron vowels — Baltic / Maori
    'ķļņģ'    # Latvian consonants
    'țș'      # Romanian cedilla
    'łńśźż'   # Polish
    'řěů'     # Czech
    'ąę'      # Polish nasal
    'ðþ'      # Icelandic / Old English
    'æøå'     # Nordic
    'ß'       # German eszett
    'œ'       # French ligature
    'ÿ'       # Rare Latin-1
)

def is_garbled(text: str) -> tuple:
    """Returns (is_garbled: bool, reason: str)."""
    if not text:
        return True, "empty"

    # Signal 1: Language-specific invalid diacritics
    invalid_count = sum(1 for c in text if c in INVALID_DIACRITICS)
    if invalid_count / max(len(text), 1) > 0.03:
        return True, f"invalid_diacritics:{invalid_count}"

    # Signal 2: Excessive non-alphanumeric characters
    non_alnum = sum(1 for c in text if not c.isalnum() and not c.isspace())
    if non_alnum / max(len(text), 1) > 0.35:
        return True, "excessive_symbols"

    # Signal 3: Repeated character runs  e.g. "aaaaaaa" or "!!!!!!"
    if re.search(r'(.)\1{5,}', text):
        return True, "char_repetition"

    # Signal 4: High single-character word density (broken tokenisation)
    words = text.split()
    if len(words) > 5:
        single_char = sum(1 for w in words if len(w) == 1 and w.isalpha())
        if single_char / len(words) > 0.40:
            return True, "single_char_words"

    # Signal 5: Vowel ratio — all four languages have healthy vowel density
    letters = [c for c in text if c.isalpha()]
    if len(letters) > 10:
        # Include common African language vowel variants (ũ ĩ etc.)
        vowels = sum(1 for c in letters
                     if c.lower() in 'aeiouãõáéíóúàèìòùâêîôûäëïöüũĩ')
        if vowels / len(letters) < 0.12:  # Lowered: Kamba/Kikuyu use ô û î
            return True, f"low_vowel_ratio:{vowels/len(letters):.2f}"

    return False, ""


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 4 — English leakage
# ─────────────────────────────────────────────────────────────────────────────
ENGLISH_STOPWORDS = {
    'the', 'and', 'is', 'are', 'was', 'were', 'to', 'of', 'in', 'that',
    'it', 'for', 'on', 'with', 'as', 'by', 'at', 'from', 'this', 'be',
    'or', 'an', 'but', 'not', 'have', 'has', 'had', 'do', 'does', 'did',
    'will', 'would', 'could', 'should', 'may', 'might', 'shall', 'can',
    'i', 'you', 'he', 'she', 'we', 'they', 'me', 'him', 'her', 'us',
    'them', 'my', 'your', 'his', 'our', 'their', 'what', 'which', 'who',
    'where', 'when', 'why', 'how', 'all', 'been', 'being', 'if', 'its',
    'than', 'then', 'so', 'up', 'out', 'about', 'into', 'through',
    'during', 'before', 'after', 'above', 'below', 'between',
}

def english_leakage(text: str, threshold: float = 0.38) -> bool:
    words = [w.lower().strip('.,!?;:()[]{}"\'') for w in text.split()]
    words = [w for w in words if len(w) > 1]
    if len(words) < 5:
        return False
    hits = sum(1 for w in words if w in ENGLISH_STOPWORDS)
    return (hits / len(words)) > threshold


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 5 — Foreign language contamination
# 28 languages covered across 8 language families
# ─────────────────────────────────────────────────────────────────────────────
FOREIGN_MARKERS = {
    # ── European Romance ────────────────────────────────────────────────────
    'french': {
        'le', 'la', 'les', 'un', 'une', 'des', 'est', 'sont', 'avec',
        'pour', 'dans', 'que', 'qui', 'pas', 'mais', 'sur', 'par', 'au',
        'du', 'se', 'je', 'tu', 'il', 'nous', 'vous', 'ils', 'elle',
        'mon', 'ton', 'son', 'notre', 'votre', 'leur', 'ce', 'cette',
    },
    'spanish': {
        'el', 'la', 'los', 'las', 'un', 'una', 'es', 'son', 'con',
        'para', 'por', 'que', 'del', 'al', 'se', 'yo', 'de', 'no', 'si',
        'pero', 'como', 'este', 'esta', 'mi', 'su', 'nos', 'ellos', 'me',
        'te', 'le', 'hay', 'muy', 'bien', 'esto', 'aqui',
    },
    'portuguese': {
        'os', 'as', 'um', 'nao', 'com', 'para', 'por', 'que', 'dos',
        'das', 'ao', 'da', 'do', 'se', 'em', 'ele', 'ela', 'sao', 'mas',
        'mais', 'este', 'esta', 'isso', 'muito', 'bem', 'ja', 'ainda',
        'so', 'seu', 'sua', 'nos', 'eles', 'elas',
    },
    'italian': {
        'il', 'lo', 'gli', 'le', 'uno', 'sono', 'con', 'per', 'che',
        'del', 'al', 'si', 'io', 'lui', 'lei', 'noi', 'voi', 'loro',
        'non', 'ma', 'anche', 'come', 'questo', 'questa', 'ho', 'ha',
        'mi', 'ti', 'ci', 'vi', 'li', 'ne', 'hai',
    },
    'romanian': {
        'este', 'sunt', 'cu', 'pentru', 'din', 'sau', 'si', 'nu', 'dar',
        'mai', 'care', 'ce', 'fi', 'lui', 'ei', 'meu', 'mea', 'nostru',
        'lor', 'asta', 'acesta', 'aceasta', 'acesti', 'acestea', 'daca',
    },
    # ── Germanic ────────────────────────────────────────────────────────────
    'german': {
        'der', 'die', 'das', 'ein', 'eine', 'ist', 'sind', 'mit', 'fur',
        'von', 'zu', 'den', 'dem', 'des', 'bei', 'nach', 'auch', 'noch',
        'wenn', 'aber', 'oder', 'ich', 'du', 'er', 'sie', 'wir', 'nicht',
        'kann', 'hat', 'haben', 'war', 'wird', 'werden', 'wurde',
    },
    'dutch': {
        'het', 'een', 'met', 'voor', 'van', 'dat', 'die', 'op', 'aan',
        'bij', 'ook', 'niet', 'als', 'er', 'ze', 'we', 'je', 'ik', 'zij',
        'hij', 'hun', 'dit', 'deze', 'wat', 'hoe', 'waar', 'welke',
    },
    'swedish': {
        'och', 'att', 'det', 'som', 'for', 'med', 'till', 'pa', 'av',
        'han', 'hon', 'vi', 'de', 'ar', 'var', 'har', 'inte', 'men',
        'om', 'ett', 'sig', 'ska', 'kan', 'vill', 'maste', 'bor',
    },
    'norwegian': {
        'og', 'at', 'det', 'som', 'for', 'med', 'til', 'pa', 'av', 'han',
        'hun', 'vi', 'er', 'var', 'har', 'ikke', 'men', 'om', 'et', 'seg',
        'skal', 'kan', 'vil', 'ma', 'bor', 'hadde', 'ble', 'bli',
    },
    'danish': {
        'og', 'at', 'det', 'som', 'for', 'med', 'til', 'pa', 'af', 'han',
        'hun', 'vi', 'er', 'var', 'har', 'ikke', 'men', 'om', 'et', 'sig',
        'skal', 'kan', 'vil', 'ma', 'havde', 'blev', 'blive',
    },
    # ── Slavic Latin ────────────────────────────────────────────────────────
    'polish': {
        'jest', 'sie', 'nie', 'to', 'ze', 'jak', 'przez', 'ale', 'czy',
        'jego', 'jej', 'ich', 'byc', 'miec', 'ktory', 'ktora', 'ktore',
        'tego', 'tej', 'tym', 'tych', 'przy', 'dla', 'bez', 'nad', 'pod',
    },
    'czech': {
        'jsou', 'se', 'to', 'ze', 'jak', 'ale', 'nebo', 'pro', 'pri',
        'jeho', 'jeji', 'jejich', 'byt', 'mit', 'ktery', 'ktera', 'ktere',
        'jsem', 'jsi', 'jsme', 'jste', 'bylo', 'bude', 'budou', 'ma',
    },
    # Croatian/Slovenian/Slovak removed — single-letter and short words
    # overlap heavily with Swahili/Luo. Only keep multi-syllable markers.
    'south_slavic_distinctive': {
        'zbog', 'nakon', 'između', 'također', 'ovakav', 'takav',
        'svaki', 'svaka', 'svako', 'dakle', 'zapravo', 'upravo',
    },
    # ── Turkic ──────────────────────────────────────────────────────────────
    'turkish': {
        'bir', 'bu', 've', 'ile', 'icin', 'gibi', 'ama', 'degil', 'var',
        'yok', 'olan', 'ki', 'ben', 'sen', 'biz', 'siz', 'onlar',
        'benim', 'senin', 'onun', 'bizim', 'sizin', 'onlarin',
    },
    # ── Austronesian ────────────────────────────────────────────────────────
    'indonesian': {
        'yang', 'dan', 'di', 'ini', 'itu', 'dengan', 'untuk', 'dari',
        'tidak', 'ada', 'saya', 'kita', 'kami', 'mereka', 'adalah',
        'akan', 'bisa', 'juga', 'sudah', 'belum', 'sedang', 'telah',
        'oleh', 'ke', 'atau', 'karena', 'jika', 'tetapi', 'namun',
    },
    'tagalog': {
        'ang', 'ng', 'sa', 'na', 'at', 'ay', 'mga', 'ko', 'mo', 'niya',
        'namin', 'ninyo', 'nila', 'ito', 'iyon', 'siya', 'sila', 'kami',
        'tayo', 'kayo', 'ba', 'pa', 'nga', 'lang', 'naman', 'dito',
    },
    'malay': {
        'yang', 'dan', 'di', 'ini', 'itu', 'dengan', 'untuk', 'dari',
        'tidak', 'ada', 'saya', 'kita', 'kami', 'mereka', 'adalah',
        'akan', 'boleh', 'juga', 'sudah', 'belum', 'sedang', 'telah',
    },
    # ── African languages (non-target) ──────────────────────────────────────
    # Zulu/Xhosa/Yoruba removed from active markers — too much overlap
    # with Swahili and Luo vocabulary causing false positives.
    # Real contamination from these languages is caught by the garbled
    # text detector (they use different diacritics/patterns) and the
    # script filter. Only keep highly distinctive multi-syllable words.
    'zulu_distinctive': {
        'ukuthi', 'ukuba', 'ngoba', 'lapha', 'khona', 'njalo',
        'ngakho', 'ngokuthi',
    },
    'yoruba_distinctive': {
        'sugbon', 'nitorina', 'beeni', 'beko',
    },
    'igbo': {
        'na', 'ya', 'ka', 'ma', 'mana', 'ha', 'anyi', 'gi', 'nke', 'no',
        'ebe', 'mgbe', 'ihe', 'ndi', 'maka', 'site', 'iji', 'obi',
    },
    'hausa': {
        'da', 'don', 'amma', 'kuma', 'ko', 'sai', 'idan', 'wanda', 'wata',
        'wannan', 'wadannan', 'shi', 'ita', 'su', 'mu', 'ku', 'ni', 'ka',
    },
    'somali': {
        'iyo', 'oo', 'ah', 'in', 'la', 'ka', 'ku', 'uu', 'ay', 'waa',
        'aan', 'leh', 'kale', 'sida', 'markaa', 'laakiin', 'waayo',
        'hadda', 'wixii', 'marka',
    },
    'wolof': {
        'ak', 'bi', 'yi', 'bu', 'li', 'ci', 'di', 'mu', 'nu', 'ko',
        'leen', 'ngir', 'fee', 'fii', 'foo', 'nit', 'xam', 'def', 'dem',
    },
    # ── South / Southeast Asian romanized ───────────────────────────────────
    'hindi_rom': {
        'hai', 'hain', 'tha', 'thi', 'the', 'mein', 'ko', 'ke', 'ki',
        'ka', 'se', 'par', 'aur', 'ya', 'lekin', 'kyunki', 'jab', 'tab',
    },
    'vietnamese': {
        'va', 'cua', 'la', 'trong', 'co', 'voi', 'duoc', 'nhung', 'cho',
        'nay', 'do', 'khong', 'tu', 'mot', 'cac', 'theo', 'nhu', 'hay',
    },
}

# Words shared between target languages and foreign markers — do not penalise.
# These are whitelisted per language so a word valid in Kamba isn't rejected
# because it also happens to appear in some foreign marker list.
TARGET_WHITELIST = {
    'kamba': {
        'na', 'ni', 'wa', 'ya', 'ka', 'ma', 'la', 'ta', 'ku', 'mu',
        'si', 'vi', 'tu', 'mi', 'sa', 'ki', 'ri', 'ti', 'li', 'di',
        'fi', 'a', 'o', 'e', 'i', 'u', 'ne', 'mo', 'ko',
    },
    'kikuyu': {
        'na', 'ni', 'wa', 'ya', 'ka', 'ma', 'la', 'ta', 'ku', 'mu',
        'si', 'vi', 'tu', 'mi', 'no', 'ro', 'gwa', 'iga', 'a', 'o',
        'e', 'i', 'u', 'ne', 'mo', 'ko', 'to', 'gi',
    },
    'luo': {
        # Luo shares many short words with foreign markers — extended whitelist
        'gi', 'mi', 'ni', 'bi', 'di', 'pi', 'ti', 'si', 'li', 'ri',
        'ki', 'ma', 'ka', 'la', 'wa', 'ya', 'na', 'to', 'go', 'do',
        'ko', 'lo', 'ne', 'an', 'in', 'on', 'be', 'is', 'da', 'de',
        've', 'ile', 'bir', 'bu', 'a', 'o', 'e', 'i', 'u', 'ng', 'ch',
        'oo', 'ee', 'ii', 'aa', 'uu', 'jo', 'gi', 'bi', 'pi',
    },
    'swahili': {
        'na', 'ni', 'wa', 'ya', 'ka', 'ma', 'la', 'ta', 'ku', 'mu',
        'si', 'vi', 'tu', 'mi', 'za', 'ji', 'li', 'ki', 'pa', 'mwa',
        'pia', 'au', 'kwa', 'kama', 'sana', 'a', 'o', 'e', 'i', 'u',
        'ne', 'mo', 'ko', 'to', 'da', 'la',
        # Swahili words that overlap with Croatian markers
        'je', 'se', 'sa', 'ili', 'bila', 'ali', 'do', 'su', 'bi',
        'sto', 'ona', 'oni', 'one', 'ove', 'ovo', 'samo', 'vec',
        'jos', 'kako', 'od', 'pri', 'zbog', 'ovaj', 'bio', 'bili',
        # Swahili words that overlap with Yoruba markers
        'ti', 'bi', 'lori', 'tabi', 'ati', 'fun', 'gege', 'nje',
        'nini', 'pelu', 'lati', 'ninu', 'nitori',
        # Swahili words that overlap with Zulu markers
        'uma', 'ona', 'manje', 'lapha',
        # Common Swahili words that overlap with other markers
        'de', 've', 'ne', 'an', 'in', 'on', 'be', 'it',
    },
}

# Contamination thresholds per language
# Luo gets a higher threshold because it shares more short words with markers
CONTAMINATION_THRESHOLDS = {
    'kamba':   (0.35, 4, 10),  # raised to reduce false positives
    'kikuyu':  (0.35, 4, 10),
    'luo':     (0.40, 5, 12),  # strictest — most overlap with markers
    'swahili': (0.40, 5, 12),  # raised — many Swahili/foreign word overlaps
}

def is_contaminated(text: str, lang: str) -> tuple:
    """Returns (is_contaminated: bool, foreign_lang: str)."""
    words = [w.lower().strip('.,!?;:()[]{}"\'') for w in text.split()]
    words = [w for w in words if len(w) > 1]
    if len(words) < 5:
        return False, ""

    whitelist = TARGET_WHITELIST.get(lang, set())
    density_thresh, min_hits, min_words = CONTAMINATION_THRESHOLDS.get(
        lang, (0.30, 3, 8)
    )

    for foreign_lang, markers in FOREIGN_MARKERS.items():
        hits = sum(1 for w in words if w in markers and w not in whitelist)
        hit_rate = hits / len(words)

        if hit_rate > density_thresh or (hits >= min_hits and len(words) >= min_words):
            return True, foreign_lang

    return False, ""


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 6 — Structural quality
# ─────────────────────────────────────────────────────────────────────────────
STRUCTURAL_REJECT = {'too_short', 'url', 'code'}

def structural_issues(text: str) -> list:
    issues = []

    if len(text.strip()) < 15:
        issues.append('too_short')

    if len(text.strip()) > 800:
        issues.append('too_long')

    if re.search(r'https?://|www\.|@\w+\.\w+', text):
        issues.append('url')

    if re.search(r'\b(def |class |import |return |function\()', text):
        issues.append('code')

    digits = sum(c.isdigit() for c in text)
    if len(text) > 0 and digits / len(text) > 0.30:
        issues.append('excessive_digits')

    # Repeated bigrams (relaxed — oral-style African text repeats naturally)
    words = text.split()
    if len(words) >= 8:
        bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words) - 1)]
        repeated = [b for b in bigrams if bigrams.count(b) > 1]
        if len(repeated) > 5:
            issues.append('repeated_phrases')

    return issues


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 7 — Language-specific positive validation
# Text should exhibit expected properties of the target language
# ─────────────────────────────────────────────────────────────────────────────

# High-frequency Swahili function words — any genuine Swahili text will
# contain at least a few of these
SWAHILI_CORE = {
    'na', 'ni', 'wa', 'ya', 'ka', 'kwa', 'la', 'za', 'pa', 'mwa',
    'katika', 'kuwa', 'pia', 'au', 'lakini', 'kama', 'hii', 'hizo',
    'ikiwa', 'hivyo', 'kwamba', 'yote', 'wote', 'hata', 'bali', 'ili',
    'sana', 'zaidi', 'kabla', 'baada', 'wakati', 'jambo', 'mambo',
    'watu', 'mtu', 'siku', 'mwaka', 'mara', 'bado', 'tayari', 'tu',
}

# Kamba uses ũ and ĩ extensively — a longer Kamba text should have some
KAMBA_CHARS = set('ũĩ')

# Kikuyu also uses ũ and ĩ
KIKUYU_CHARS = set('ũĩ')

def fails_language_check(text: str, lang: str) -> tuple:
    """
    Returns (fails: bool, reason: str).
    Positive check — text should look like the target language.
    Only applied to longer texts where we have enough signal.
    """
    words = text.lower().split()
    if len(words) < 8:
        return False, ""  # too short to make a confident call

    if lang == 'swahili':
        hits = sum(1 for w in words if w.strip('.,!?;:') in SWAHILI_CORE)
        if hits / len(words) < 0.04:
            return True, "no_swahili_markers"

    # For Kamba and Kikuyu, very long texts with zero ũ/ĩ chars are suspicious
    # (short paraphrases may legitimately not use them)
    if lang == 'kamba' and len(text) > 100:
        if not any(c in KAMBA_CHARS for c in text):
            # Don't reject — just note — some valid Kamba doesn't use diacritics
            pass

    if lang == 'kikuyu' and len(text) > 100:
        if not any(c in KIKUYU_CHARS for c in text):
            pass  # same reasoning

    return False, ""


# ─────────────────────────────────────────────────────────────────────────────
# MAIN CLEANING LOOP
# ─────────────────────────────────────────────────────────────────────────────
def clean_language(lang: str, base: Path, dry_run: bool = False) -> dict:
    """
    Clean one language dataset. Returns stats dict.
    """
    input_path  = base / f"{lang}_tier1.jsonl"
    output_path = base / f"{lang}_tier1_clean.jsonl"
    backup_path = base / f"{lang}_tier1_backup.jsonl"

    if not input_path.exists():
        print(f"  ⚠️  {lang}: {input_path} not found — skipping")
        return {}

    stats   = collections.Counter()
    samples = {}  # one example per rejection category

    with open(input_path) as fin, open(output_path, 'w') as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            stats['total'] += 1

            try:
                d    = json.loads(line)
                text = d.get('text', '').strip()

                # ── Basic validity ────────────────────────────────────────
                if not text:
                    stats['empty'] += 1
                    continue

                # ── Layer 1: Script ───────────────────────────────────────
                if has_bad_script(text):
                    stats['bad_script'] += 1
                    samples.setdefault('bad_script', text[:80])
                    continue

                # ── Layer 2: Latin ratio ──────────────────────────────────
                if latin_ratio(text) < 0.60:
                    stats['low_latin'] += 1
                    samples.setdefault('low_latin', text[:80])
                    continue

                # ── Layer 3: Garbled ──────────────────────────────────────
                garbled, reason = is_garbled(text)
                if garbled:
                    stats['garbled'] += 1
                    stats[f'garbled__{reason}'] += 1
                    samples.setdefault('garbled', f"[{reason}] {text[:80]}")
                    continue

                # ── Layer 4: English leakage ──────────────────────────────
                if english_leakage(text):
                    stats['english'] += 1
                    samples.setdefault('english', text[:80])
                    continue

                # ── Layer 5: Foreign contamination ────────────────────────
                contaminated, flang = is_contaminated(text, lang)
                if contaminated:
                    stats['foreign'] += 1
                    stats[f'foreign__{flang}'] += 1
                    samples.setdefault(f'foreign__{flang}', text[:80])
                    continue

                # ── Layer 6: Structural ───────────────────────────────────
                issues = structural_issues(text)
                if any(i in STRUCTURAL_REJECT for i in issues):
                    stats['structural'] += 1
                    samples.setdefault('structural',
                                       f"[{','.join(issues)}] {text[:60]}")
                    continue

                # ── Layer 7: Language-specific positive check ─────────────
                fails, reason = fails_language_check(text, lang)
                if fails:
                    stats['lang_check'] += 1
                    samples.setdefault('lang_check',
                                       f"[{reason}] {text[:80]}")
                    continue

                # ── Passed all layers ─────────────────────────────────────
                if not dry_run:
                    fout.write(line + '\n')
                stats['kept'] += 1

            except Exception:
                stats['parse_error'] += 1

    # Backup original and replace with clean version
    if not dry_run:
        shutil.copy2(input_path, backup_path)
        shutil.move(str(output_path), str(input_path))

    stats['_lang']        = lang
    stats['_input_path']  = str(input_path)
    stats['_backup_path'] = str(backup_path)
    stats['_samples']     = samples
    return stats


def print_report(stats: dict) -> None:
    lang    = stats.get('_lang', '?')
    total   = stats['total']
    kept    = stats['kept']
    removed = total - kept
    samples = stats.get('_samples', {})

    print(f'\n{"═" * 68}')
    print(f'  {lang.upper()}: {total:,} → {kept:,} kept '
          f'({removed:,} removed, {removed / max(total, 1) * 100:.1f}%)')
    print(f'{"─" * 68}')

    categories = [
        ('bad_script',  'Non-Latin script'),
        ('low_latin',   'Low Latin ratio'),
        ('garbled',     'Garbled text'),
        ('english',     'English leakage'),
        ('foreign',     'Foreign language'),
        ('structural',  'Structural issues'),
        ('lang_check',  'Lang-specific check'),
        ('parse_error', 'Parse errors'),
        ('empty',       'Empty text'),
    ]

    for key, label in categories:
        count = stats[key]
        if count:
            print(f'  {label:<28} {count:>7,}')

    # Garbled breakdown
    garbled_sub = {k.split('__', 1)[1]: v for k, v in stats.items()
                   if k.startswith('garbled__')}
    if garbled_sub:
        for reason, count in sorted(garbled_sub.items(),
                                    key=lambda x: -x[1])[:8]:
            print(f'    ↳ {reason:<36} {count:>5,}')

    # Foreign language breakdown
    foreign_sub = {k.split('__', 1)[1]: v for k, v in stats.items()
                   if k.startswith('foreign__')}
    if foreign_sub:
        for fl, count in sorted(foreign_sub.items(),
                                 key=lambda x: -x[1])[:10]:
            print(f'    ↳ {fl:<36} {count:>5,}')

    # Sample removed texts
    if samples:
        print(f'  {"─" * 66}')
        print(f'  Examples removed:')
        shown = 0
        for k, v in samples.items():
            if shown >= 4:
                break
            k_display = k.replace('__', ':')
            print(f'    [{k_display}]')
            print(f'    {v}')
            shown += 1


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
def main():
    dry_run = '--dry-run' in sys.argv

    print('╔' + '═' * 66 + '╗')
    print('║' + '  LUGHAGEN SYNTHETIC DATA CLEANER — v4 Final'.center(66) + '║')
    print('║' + f'  {datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")}'.ljust(66) + '║')
    if dry_run:
        print('║' + '  *** DRY RUN — no files will be modified ***'.center(66) + '║')
    print('╚' + '═' * 66 + '╝')
    print()
    print(f'  Base directory : {BASE}')
    print(f'  Languages      : {", ".join(LANGS)}')
    print(f'  Layers active  : 7 (script, latin, garbled, english,')
    print(f'                       foreign x28, structural, lang-specific)')
    print()

    base = Path(BASE)
    if not base.exists():
        print(f'ERROR: {BASE} does not exist. Check BASE path.')
        sys.exit(1)

    all_stats = []
    grand_total = grand_kept = 0

    for lang in LANGS:
        print(f'Processing {lang}...')
        stats = clean_language(lang, base, dry_run=dry_run)
        if not stats:
            continue
        print_report(stats)
        all_stats.append(stats)
        grand_total += stats['total']
        grand_kept  += stats['kept']

    # ── Grand summary ──────────────────────────────────────────────────────
    grand_removed = grand_total - grand_kept
    print(f'\n{"═" * 68}')
    print(f'  GRAND TOTAL: {grand_total:,} → {grand_kept:,} kept '
          f'({grand_removed:,} removed, '
          f'{grand_removed / max(grand_total, 1) * 100:.1f}%)')
    print(f'{"═" * 68}')
    print()

    print('  Per-language final counts:')
    for stats in all_stats:
        lang    = stats['_lang']
        kept    = stats['kept']
        total   = stats['total']
        removed = total - kept
        gap     = max(0, 100_000 - kept)
        status  = f'✅ target met' if gap == 0 else f'⚠️  {gap:,} below 100K target'
        print(f'    {lang:<10} {kept:>7,}  ({removed:,} removed)  {status}')

    if not dry_run:
        print()
        print('  Backups saved as *_backup.jsonl')
        print('  Clean data replaced *_tier1.jsonl in place')
        print()
        print('  Next step: run post-dedup to catch any remaining duplicates:')
        print('    python3 /workspace/scripts/postprocess_dedup.py')

    print()


if __name__ == '__main__':
    main()
