"""
Re-clean Kikuyu, Luo, Kamba with relaxed thresholds.
Key changes:
  1. LANG_SCORE_MIN_SHORT: 0.10 → 0.03
  2. LANG_SCORE_MIN_LONG:  0.05 → 0.02
  3. MIN_CHARS: 20 → 15 (capture shorter utterances)
  4. MIN_WORDS: 3  → 2  (capture 2-word phrases)
  5. Dedup: normalize aggressively to recover near-duplicates that differ only by punctuation
"""

import json
import re
import hashlib
import logging
from pathlib import Path
from datetime import datetime

Path("/workspace/logs/pipeline").mkdir(parents=True, exist_ok=True)
Path("/workspace/data/cleaned_v2").mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    handlers=[
        logging.FileHandler("/workspace/logs/pipeline/reclean.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger("reclean")

# ── RELAXED CONFIG ────────────────────────────────────────────────
MIN_CHARS            = 15    # was 20
MAX_CHARS            = 1500  # was 1000 — allow longer sentences
MIN_WORDS            = 2     # was 3
FOREIGN_RATIO_MAX    = 0.25  # was 0.20 — allow some loanwords
LANG_SCORE_MIN_SHORT = 0.03  # was 0.10 — much more permissive
LANG_SCORE_MIN_LONG  = 0.02  # was 0.05 — much more permissive

REJECT_SCRIPT_PATTERNS = [
    r'[\u0900-\u097F]', r'[\u0980-\u09FF]', r'[\u0A00-\u0A7F]',
    r'[\u0A80-\u0AFF]', r'[\u0B00-\u0B7F]', r'[\u0B80-\u0BFF]',
    r'[\u0C00-\u0C7F]', r'[\u0C80-\u0CFF]', r'[\u0D00-\u0D7F]',
    r'[\u0600-\u06FF]', r'[\u0750-\u077F]', r'[\u08A0-\u08FF]',
    r'[\u4E00-\u9FFF]', r'[\u3400-\u4DBF]',
    r'[\u3040-\u309F]', r'[\u30A0-\u30FF]',
    r'[\uAC00-\uD7AF]',
    r'[\u0400-\u04FF]', r'[\u0370-\u03FF]',
    r'[\u1200-\u137F]',
]
REJECT_CHAR_RE = re.compile('|'.join(REJECT_SCRIPT_PATTERNS))

NOISE_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [
    r'http[s]?://',
    r'(@\w+\s*){3,}',      # was 2, now 3 — allow single mentions
    r'(#\w+\s*){3,}',      # was 2, now 3
    r'^\s*[\d\s\W]+$',
    r'bringback|mahinda|naina|salaaah',
]]

# Expanded markers for better recall
LANG_MARKERS = {
    "kikuyu": re.compile(
        # Added more common Kikuyu words beyond just diacritics
        r'[ĩũ]|nĩ|gũ|kũ|mũ|wĩ|tũ|atĩ|'
        r'\b(na|wa|ni|ku|cia|ria|kia|mia|tia|gia|bia|pia|'
        r'ũguo|ũrĩa|ũndũ|mũndũ|kĩrĩ|gĩkũyũ|ngũ)\b',
        re.I),
    "kamba": re.compile(
        r'[ĩũ]|kũ|mũ|kĩ|ũkĩ|nthĩ|ĩndĩ|nzeo|ũtĩ|'
        r'\b(na|wa|ni|ku|ma|ka|nĩ|ũ|ĩ|syakwa|wakwa|'
        r'kĩkamba|ukambani|machakos|kitui)\b',
        re.I),
    "luo": re.compile(
        # Much broader — many valid Luo sentences lack the narrow markers
        r'\b(ni|ma|ka|gi|mi|od|ok|dhi|en|ot|ng|bed|tim|go|'
        r'osiepe|nyathini|wuod|nyar|ja|jo|'
        r'kisumu|nyanza|siaya|homa|migori|'
        r'ber|rach|tek|yot|bor|machiegni|e|kendo|to|'
        r'nobet|nober|nothum|noyud|nosomo)\b',
        re.I),
}

TOKENS_PER_SAMPLE = {
    "kikuyu": 20, "kamba": 20, "luo": 22,
}

def foreign_ratio(text: str) -> float:
    if not text:
        return 0.0
    return sum(1 for ch in text if REJECT_CHAR_RE.match(ch)) / len(text)

def lang_score(text: str, lang: str) -> float:
    pattern = LANG_MARKERS.get(lang)
    if not pattern:
        return 1.0
    return len(pattern.findall(text)) / max(len(text.split()), 1)

def is_noise(text: str) -> bool:
    return any(p.search(text) for p in NOISE_PATTERNS)

def normalize_for_dedup(text: str) -> str:
    """More aggressive normalization — catches punctuation variants."""
    text = text.lower()
    text = re.sub(r'[^\w\sĩũ]', '', text)  # strip punctuation
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def quality_score(text: str) -> float:
    score = 1.0
    if len(text) < 30:
        score -= 0.1     # softer penalty than before
    if re.search(r'(.)\1{4,}', text):
        score -= 0.3
    if re.search(r'\b(\w+)\s+\1\s+\1', text):
        score -= 0.2
    if text.isupper() and len(text) > 10:
        score -= 0.2
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
    # Only apply lang_score to languages with good markers
    if lang in LANG_MARKERS:
        words = len(text.split())
        threshold = LANG_SCORE_MIN_SHORT if words < 15 else LANG_SCORE_MIN_LONG
        if lang_score(text, lang) < threshold:
            return False, "low_lang_score"
    return True, "ok"

def clean_file(lang: str, input_path: str, output_path: str) -> dict:
    if not Path(input_path).exists():
        log.warning(f"{lang}: missing {input_path}")
        return {}

    log.info(f"\n{'='*55}")
    log.info(f"Re-cleaning: {lang.upper()} (relaxed thresholds)")

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

            # Aggressive dedup normalization
            norm = normalize_for_dedup(text)
            h    = hashlib.md5(norm.encode()).hexdigest()
            if h in seen:
                removed["duplicate"] = removed.get("duplicate", 0) + 1
                continue
            seen.add(h)

            q = quality_score(text)
            b = "high" if q > 0.75 else "medium" if q > 0.5 else "low"
            bucket[b] += 1

            outf.write(json.dumps({
                "text": text, "quality": q, "bucket": b
            }, ensure_ascii=False) + "\n")

            kept        += 1
            total_len   += len(text)
            quality_sum += q

            if kept % 100_000 == 0:
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
        # Only re-clean the under-retained languages
        "kikuyu": "/workspace/data/raw/kikuyu_all.jsonl",
        "luo":    "/workspace/data/raw/luo_all.jsonl",
        "kamba":  "/workspace/data/raw/kamba_all.jsonl",
    }

    summary = {}
    for lang, path in LANGS.items():
        out = f"/workspace/data/cleaned_v2/{lang}_clean.jsonl"
        summary[lang] = clean_file(lang, path, out)

    # Compare v1 vs v2
    V1 = {"kikuyu": 124076, "luo": 284763, "kamba": 42938}

    log.info("\n" + "="*55)
    log.info("RE-CLEAN COMPARISON: v1 vs v2")
    log.info("="*55)
    for lang, s in summary.items():
        v1 = V1.get(lang, 0)
        gain = s["kept"] - v1
        log.info(
            f"  {lang:<12}: "
            f"v1={v1:>8,} → v2={s['kept']:>8,} "
            f"(+{gain:,} samples recovered) "
            f"~{s['est_tokens']/1e6:.2f}M tokens"
        )

    # Save metadata
    import json as _json
    Path("/workspace/data/meta").mkdir(exist_ok=True)
    with open("/workspace/data/meta/reclean_v2_summary.json", "w") as f:
        _json.dump({
            "timestamp": str(datetime.utcnow()),
            "summary":   summary
        }, f, indent=2)
