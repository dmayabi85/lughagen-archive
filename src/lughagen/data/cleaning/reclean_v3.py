"""
Re-clean v3 — Remove low_lang_score filter for low-resource languages.
Philosophy: trust the script rejection filter instead.
If it passes Arabic/Chinese/Hindi detection → it's African language.
"""

import json
import re
import hashlib
import logging
from pathlib import Path
from datetime import datetime, UTC

Path("/workspace/logs/pipeline").mkdir(parents=True, exist_ok=True)
Path("/workspace/data/cleaned_v3").mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    handlers=[
        logging.FileHandler("/workspace/logs/pipeline/reclean_v3.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger("reclean_v3")

MIN_CHARS         = 15
MAX_CHARS         = 2000
MIN_WORDS         = 2
FOREIGN_RATIO_MAX = 0.25

REJECT_SCRIPT_PATTERNS = [
    r'[\u0900-\u097F]', r'[\u0980-\u09FF]', r'[\u0A00-\u0A7F]',
    r'[\u0A80-\u0AFF]', r'[\u0B00-\u0B7F]', r'[\u0B80-\u0BFF]',
    r'[\u0C00-\u0C7F]', r'[\u0C80-\u0CFF]', r'[\u0D00-\u0D7F]',
    r'[\u0600-\u06FF]', r'[\u0750-\u077F]', r'[\u08A0-\u08FF]',
    r'[\u4E00-\u9FFF]', r'[\u3400-\u4DBF]',
    r'[\u3040-\u309F]', r'[\u30A0-\u30FF]',
    r'[\uAC00-\uD7AF]', r'[\u0400-\u04FF]',
    r'[\u0370-\u03FF]', r'[\u1200-\u137F]',
]
REJECT_CHAR_RE = re.compile('|'.join(REJECT_SCRIPT_PATTERNS))

# Minimal noise — only clear garbage
NOISE_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [
    r'http[s]?://',
    r'(@\w+\s*){3,}',
    r'(#\w+\s*){3,}',
    r'^\s*[\d\s\W]+$',
]]

# Positive language signals (used for BONUS scoring, not filtering)
LANG_SIGNALS = {
    "kikuyu": re.compile(r'[ĩũ]|nĩ|gũ|kũ|mũ|wĩ|tũ|atĩ', re.I),
    "kamba":  re.compile(r'[ĩũ]|kũ|mũ|kĩ|nthĩ|ũkĩ', re.I),
    "luo":    re.compile(
        r'\b(ni|ma|ka|gi|od|ok|dhi|en|ot|ng|bed|tim|'
        r'osiepe|wuod|nyar|ja|jo|ber|rach|kendo)\b', re.I),
    "swahili": re.compile(
        r'\b(na|ya|wa|ku|ni|kwa|hii|lakini|pia|cha)\b', re.I),
}

# Hard-reject: clearly not African language
# Only rejects if text is overwhelmingly ASCII-Latin with no African markers
def is_clearly_wrong_language(text: str, lang: str) -> bool:
    """
    Only hard-reject if:
    1. Text is >90% pure ASCII alphabetic
    2. AND has zero language-specific markers
    This catches Indonesian, Tagalog, Vietnamese etc.
    while preserving code-switching and loanwords.
    """
    words      = text.split()
    if not words:
        return True
    ascii_only = sum(1 for w in words
                     if w.isascii() and w.isalpha() and len(w) > 2)
    ascii_ratio = ascii_only / len(words)

    if ascii_ratio < 0.80:
        return False   # has enough non-ASCII → keep

    # High ASCII ratio — check for language signals
    signal = LANG_SIGNALS.get(lang)
    if signal and signal.search(text):
        return False   # has markers → keep

    return True        # no markers + high ASCII → reject

def foreign_ratio(text: str) -> float:
    if not text:
        return 0.0
    return sum(1 for ch in text if REJECT_CHAR_RE.match(ch)) / len(text)

def is_noise(text: str) -> bool:
    return any(p.search(text) for p in NOISE_PATTERNS)

def quality_score(text: str, lang: str) -> float:
    score = 1.0
    if len(text) < 30:
        score -= 0.15
    if re.search(r'(.)\1{4,}', text):
        score -= 0.3
    if re.search(r'\b(\w+)\s+\1\s+\1', text):
        score -= 0.2
    if text.isupper() and len(text) > 10:
        score -= 0.2
    # Bonus for language signals
    sig = LANG_SIGNALS.get(lang)
    if sig and sig.search(text):
        score += 0.05
    return max(0.0, min(score, 1.0))

def is_valid(text: str, lang: str):
    if not text or not isinstance(text, str):
        return False, "empty"
    text = text.strip()
    if len(text) < MIN_CHARS:
        return False, "too_short"
    if len(text.split()) < MIN_WORDS:
        return False, "few_words"
    if len(text) > MAX_CHARS:
        return False, "too_long"
    if foreign_ratio(text) > FOREIGN_RATIO_MAX:
        return False, "foreign_script"
    if is_noise(text):
        return False, "noise"
    # Replace low_lang_score with smarter ASCII detection
    if is_clearly_wrong_language(text, lang):
        return False, "wrong_language"
    return True, "ok"

def normalize_for_dedup(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^\w\sĩũ]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

TOKENS_PER_SAMPLE = {"swahili": 25, "kikuyu": 20, "luo": 22, "kamba": 20}

def clean_file(lang: str, input_path: str, output_path: str) -> dict:
    if not Path(input_path).exists():
        log.warning(f"Missing: {input_path}")
        return {}

    log.info(f"\n{'='*55}")
    log.info(f"Re-cleaning v3: {lang.upper()} (no lang_score filter)")

    total = kept = total_len = 0
    quality_sum = 0.0
    removed = {}
    bucket  = {"high": 0, "medium": 0, "low": 0}
    seen    = set()

    with open(input_path, "r", encoding="utf-8", errors="ignore") as inf, \
         open(output_path, "w", encoding="utf-8") as outf:

        for line in inf:
            total += 1
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except Exception:
                removed["bad_json"] = removed.get("bad_json", 0) + 1
                continue

            text  = item.get("text", "").strip()
            valid, reason = is_valid(text, lang)
            if not valid:
                removed[reason] = removed.get(reason, 0) + 1
                continue

            norm = normalize_for_dedup(text)
            h    = hashlib.md5(norm.encode()).hexdigest()
            if h in seen:
                removed["duplicate"] = removed.get("duplicate", 0) + 1
                continue
            seen.add(h)

            q = quality_score(text, lang)
            b = "high" if q > 0.75 else "medium" if q > 0.5 else "low"
            bucket[b] += 1

            outf.write(json.dumps({
                "text": text, "quality": q, "bucket": b
            }, ensure_ascii=False) + "\n")

            kept        += 1
            total_len   += len(text)
            quality_sum += q

            if kept % 200_000 == 0:
                log.info(f"  {lang}: {kept:,} kept...")

    est_tokens = kept * TOKENS_PER_SAMPLE.get(lang, 22)
    pct_kept   = kept / max(total, 1) * 100

    log.info(f"  Total    : {total:,}")
    log.info(f"  Kept     : {kept:,} ({pct_kept:.1f}%)")
    log.info(f"  Tokens   : ~{est_tokens/1e6:.2f}M")
    log.info(f"  Buckets  : {bucket}")
    log.info(f"  Removed  : {removed}")

    return {
        "lang":       lang,
        "total":      total,
        "kept":       kept,
        "pct_kept":   round(pct_kept, 2),
        "est_tokens": est_tokens,
        "buckets":    bucket,
        "removed":    removed,
    }

if __name__ == "__main__":
    LANGS = {
        "kikuyu": "/workspace/data/raw/kikuyu_all.jsonl",
        "luo":    "/workspace/data/raw/luo_all.jsonl",
        "kamba":  "/workspace/data/raw/kamba_all.jsonl",
    }

    V_PREV = {"kikuyu": 151023, "luo": 544562, "kamba": 54241}

    summary = {}
    for lang, path in LANGS.items():
        out = f"/workspace/data/cleaned_v3/{lang}_clean.jsonl"
        summary[lang] = clean_file(lang, path, out)

    log.info("\n" + "="*55)
    log.info("V2 → V3 COMPARISON")
    log.info("="*55)
    for lang, s in summary.items():
        prev = V_PREV.get(lang, 0)
        gain = s.get("kept", 0) - prev
        log.info(
            f"  {lang:<12}: "
            f"v2={prev:>8,} → v3={s.get('kept',0):>8,} "
            f"(+{gain:,}) "
            f"~{s.get('est_tokens',0)/1e6:.2f}M tokens"
        )

    Path("/workspace/data/meta").mkdir(exist_ok=True)
    with open("/workspace/data/meta/reclean_v3_summary.json", "w") as f:
        json.dump({
            "timestamp": str(datetime.now(UTC)),
            "summary":   summary
        }, f, indent=2)
    log.info("\nDone → /workspace/data/cleaned_v3/")
