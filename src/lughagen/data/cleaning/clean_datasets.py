import json
import re
import hashlib
import logging
from pathlib import Path
from datetime import datetime

Path("/workspace/logs/pipeline").mkdir(parents=True, exist_ok=True)
Path("/workspace/data/cleaned").mkdir(parents=True, exist_ok=True)
Path("/workspace/data/meta").mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    handlers=[
        logging.FileHandler("/workspace/logs/pipeline/cleaning.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger("cleaner")

MIN_CHARS            = 20
MAX_CHARS            = 1000
MIN_WORDS            = 3
FOREIGN_RATIO_MAX    = 0.20
LANG_SCORE_MIN_SHORT = 0.10
LANG_SCORE_MIN_LONG  = 0.05

TOKENS_PER_SAMPLE = {
    "swahili": 25, "kikuyu": 20, "luo": 22, "kamba": 20,
}

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
    r'(@\w+\s*){2,}',
    r'(#\w+\s*){2,}',
    r'^\s*[\d\s\W]+$',
    r'bringback|mahinda|naina|salaaah',
    r'^(no more|RT @)',
]]

LANG_MARKERS = {
    "swahili": re.compile(r'\b(na|ya|wa|ku|ni|kwa|hii|hiyo|lakini|pia)\b', re.I),
    "kikuyu":  re.compile(r'[ĩũ]|nĩ|gũ|kũ|mũ|wĩ|tũ|atĩ', re.I),
    "kamba":   re.compile(r'[ĩũ]|kũ|mũ|kĩ|nthĩ|ũkĩ', re.I),
    "luo":     re.compile(r'\b(ni|ma|gi|od|ok|dhi|en|ot)\b', re.I),
}

def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^\w\sĩũ]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

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

def quality_score(text: str, lang: str) -> float:
    score = 1.0
    if len(text) < 50:
        score -= 0.2
    if re.search(r'(.)\1{4,}', text):
        score -= 0.3
    if re.search(r'\b(\w+)\s+\1\s+\1', text):
        score -= 0.2
    if text.isupper() and len(text) > 10:
        score -= 0.2
    digit_ratio = sum(c.isdigit() for c in text) / max(len(text), 1)
    if digit_ratio > 0.3:
        score -= 0.2
    if lang_score(text, lang) > 0.2:
        score += 0.1
    return max(0.0, min(score, 1.0))

def is_valid(text: str, lang: str):
    if not text:
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
    threshold = LANG_SCORE_MIN_SHORT if len(text.split()) < 20 else LANG_SCORE_MIN_LONG
    if lang_score(text, lang) < threshold:
        return False, "low_lang_score"
    return True, "ok"

def clean_file(lang, input_path, output_path):
    if not Path(input_path).exists():
        log.warning(f"{lang}: missing file {input_path}")
        return {"lang": lang, "kept": 0, "total": 0}

    log.info(f"\n{'='*55}")
    log.info(f"Cleaning: {lang.upper()} | {input_path}")

    total = kept = total_len = 0
    quality_sum = 0.0
    removed = {}
    bucket  = {"high": 0, "medium": 0, "low": 0}
    seen    = set()

    with open(input_path, "r", encoding="utf-8", errors="ignore") as inf, \
         open(output_path, "w", encoding="utf-8") as outf:

        for line in inf:
            total += 1
            try:
                item = json.loads(line)
            except Exception:
                removed["bad_json"] = removed.get("bad_json", 0) + 1
                continue

            text  = item.get("text", "")
            valid, reason = is_valid(text, lang)
            if not valid:
                removed[reason] = removed.get(reason, 0) + 1
                continue

            norm = normalize(text)
            h    = hashlib.md5(norm.encode()).hexdigest()
            if h in seen:
                removed["duplicate"] = removed.get("duplicate", 0) + 1
                continue
            seen.add(h)

            q  = quality_score(text, lang)
            b  = "high" if q > 0.75 else "medium" if q > 0.5 else "low"
            bucket[b] += 1

            outf.write(json.dumps({
                "text": text, "quality": q, "bucket": b
            }, ensure_ascii=False) + "\n")

            kept        += 1
            total_len   += len(text)
            quality_sum += q

            if kept % 500_000 == 0:
                log.info(f"  {lang}: {kept:,} kept...")

    est_tokens = kept * TOKENS_PER_SAMPLE.get(lang, 22)
    pct_kept   = kept / max(total, 1) * 100

    log.info(f"  Kept     : {kept:,} / {total:,} ({pct_kept:.1f}%)")
    log.info(f"  Tokens   : ~{est_tokens/1e6:.2f}M")
    log.info(f"  Buckets  : {bucket}")
    log.info(f"  Removed  : {removed}")

    return {
        "lang":        lang,
        "total":       total,
        "kept":        kept,
        "pct_kept":    round(pct_kept, 2),
        "avg_length":  round(total_len / max(kept, 1), 1),
        "avg_quality": round(quality_sum / max(kept, 1), 3),
        "est_tokens":  est_tokens,
        "buckets":     bucket,
        "removed":     removed,
    }

if __name__ == "__main__":
    LANGS = {
        "swahili": "/workspace/data/raw/swahili_all.jsonl",
        "kikuyu":  "/workspace/data/raw/kikuyu_all.jsonl",
        "luo":     "/workspace/data/raw/luo_all.jsonl",
        "kamba":   "/workspace/data/raw/kamba_all.jsonl",
    }

    summary      = {}
    total_tokens = 0

    for lang, path in LANGS.items():
        out          = f"/workspace/data/cleaned/{lang}_clean.jsonl"
        summary[lang] = clean_file(lang, path, out)
        total_tokens += summary[lang]["est_tokens"]

    meta = {
        "timestamp":    str(datetime.utcnow()),
        "summary":      summary,
        "total_tokens": total_tokens
    }
    with open("/workspace/data/meta/cleaning_summary.json", "w") as f:
        json.dump(meta, f, indent=2)

    log.info("\n" + "="*55)
    log.info("PIPELINE COMPLETE")
    log.info("="*55)
    for lang, s in summary.items():
        log.info(
            f"  {lang:<12}: {s['kept']:>10,} samples | "
            f"{s['pct_kept']:>5.1f}% kept | "
            f"~{s['est_tokens']/1e6:.2f}M tokens"
        )
    log.info(f"\n  TOTAL: ~{total_tokens/1e6:.2f}M tokens")
    log.info(f"  Meta : /workspace/data/meta/cleaning_summary.json")
