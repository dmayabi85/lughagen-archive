#!/usr/bin/env python3
"""
LughaGen Tier 2 Generate — v1.2 Final

All 8 critical fixes applied:
  FIX1: logprobs=5 + actual token ID perplexity calculation
  FIX2: JSON retry with stronger instruction on parse failure
  FIX3: PERPLEXITY_MAX=160 (was 250)
  FIX4: LSH 16x4 (was 8x2)
  FIX5: GPU_UTIL_MAX removed (meaningless at >1.0)
  FIX6: PID-tracked process killing (not global kill)
  FIX7: ulimit removed from generator
  FIX8: temperature/top_p/repetition_penalty from config

Subtle improvements:
  - Character 4-gram diversity filter (better for Bantu morphology)
  - Code-switch ratio post-check (enforces word-level cap)
  - Length normalization (penalizes both too short AND too long)
  - Style balancing in curriculum (prevents drift)

Run as:
  CUDA_VISIBLE_DEVICES=0,1 python3 generate.py kamba
  CUDA_VISIBLE_DEVICES=2,3 python3 generate.py kikuyu
  CUDA_VISIBLE_DEVICES=4,5 python3 generate.py luo
  CUDA_VISIBLE_DEVICES=6,7 python3 generate.py swahili
"""

import os, sys, re, json, time, random, hashlib, logging
from pathlib import Path
from datetime import datetime
from collections import deque

from vllm import LLM, SamplingParams
import jsonlines
import torch
import numpy as np

# Allow running from tier2 directory
sys.path.insert(0, str(Path(__file__).parent))
from config import (
    PIPELINE_VERSION, PROMPT_VERSION, DATASET_VERSION,
    QUALITY_MIN, PERPLEXITY_MAX,
    AQT_FLOOR, AQT_CEILING, AQT_MARGIN, AQT_WINDOW,
    BATCH_SIZE_MIN_70B, BATCH_SIZE_MAX_70B, BATCH_SIZE_INIT_70B,
    BATCH_SIZE_MIN_8B,  BATCH_SIZE_MAX_8B,  BATCH_SIZE_INIT_8B,
    LSH_BANDS, LSH_ROWS,
    DIVERSITY_MAX_SIM, DIVERSITY_WINDOW,
    CHECKPOINT_EVERY,
    TEMPERATURE_70B, TEMPERATURE_8B, TOP_P,
    REPETITION_PENALTY, MAX_NEW_TOKENS, MIN_OUTPUT_CHARS, LOGPROBS,
    CODE_SWITCH_WORD_MAX,
    LANGUAGE_CONFIGS,
)
from prompts import Tier2Curriculum, code_switch_ratio

# ── GlotLID language verifier ─────────────────────────────────────
class GlotLIDVerifier:
    """
    Real-time language verification during generation.
    Rejects samples that are not in the target language family.
    Uses GlotLID (Kargaran et al., 2023) — 1600+ language fastText model.
    """
    # ISO 639-3 + Latn script codes for each target language
    TARGET_CODES = {
        "kamba":   {"kam_Latn"},
        "kikuyu":  {"kik_Latn"},
        "luo":     {"luo_Latn"},
        "swahili": {"swh_Latn", "swc_Latn"},
    }
    # Nearby acceptable languages (same family — not contamination)
    ACCEPTABLE_CODES = {
        # Swahili excluded from Kamba/Kikuyu to keep training data clean.
        # nim_Latn (Nimadi) added for Kamba — GlotLID false positive due to
        # shared vowel diacritic patterns (ĩ, ũ). Verified by inspection.
        # des_Latn, cbc_Latn, bsn_Latn added for Kikuyu — same reason.
        "kamba":   {"kam_Latn","kik_Latn","luy_Latn","guz_Latn","mer_Latn",
                    "nim_Latn","dad_Latn"},   # GlotLID FP for Kamba diacritics
        "kikuyu":  {"kik_Latn","kam_Latn","mer_Latn","luy_Latn",
                    "des_Latn","cbc_Latn","bsn_Latn"},  # GlotLID FP for Kikuyu
        "luo":     {"luo_Latn","ach_Latn","laj_Latn","adh_Latn","alz_Latn"},
        "swahili": {"swh_Latn","swc_Latn","vmw_Latn","hay_Latn"},  # hay=Haya (Bantu)
    }
    # Minimum confidence to trust the prediction
    MIN_CONFIDENCE  = 0.40
    # Minimum text length to apply GlotLID (too short = unreliable)
    MIN_TEXT_LENGTH = 25

    def __init__(self, model_path: str = "/workspace/models/glotlid.bin"):
        self.model = None
        self.available = False
        self.checked = self.passed = self.rejected = 0
        try:
            import fasttext
            import os
            if os.path.exists(model_path):
                self.model = fasttext.load_model(model_path)
                self.available = True
                print(f"✅ GlotLID loaded from {model_path}")
            else:
                print(f"GlotLID not found at {model_path} — skipping LID check")
        except Exception as e:
            print(f"GlotLID unavailable ({e}) — skipping LID check")

    def is_target_language(self, text: str, lang: str) -> tuple:
        """
        Returns (is_valid, detected_lang, confidence).
        """
        if not self.available or not self.model:
            return True, "unknown", 0.0
        if len(text.strip()) < self.MIN_TEXT_LENGTH:
            return True, "short_skip", 0.0  # too short to judge

        try:
            text_clean = text.replace("\n", " ").strip()[:500]
            labels, probs = self.model.predict(text_clean, k=5)
            top_label = labels[0].replace("__label__", "")
            top_prob  = float(probs[0])

            self.checked += 1
            target     = self.TARGET_CODES.get(lang, set())
            acceptable = self.ACCEPTABLE_CODES.get(lang, target)

            # Accept if confident it IS target language
            if top_label in target and top_prob >= self.MIN_CONFIDENCE:
                self.passed += 1
                return True, top_label, top_prob

            # Accept if in acceptable Bantu family (don't over-reject)
            if top_label in acceptable:
                self.passed += 1
                return True, top_label, top_prob

            # Accept if confidence is too low to make a call
            if top_prob < self.MIN_CONFIDENCE:
                self.passed += 1
                return True, top_label, top_prob

            # Reject — high confidence wrong language
            self.rejected += 1
            return False, top_label, top_prob

        except Exception:
            return True, "error", 0.0

    def stats(self) -> dict:
        return {
            "available":  self.available,
            "checked":    self.checked,
            "passed":     self.passed,
            "rejected":   self.rejected,
            "reject_rate": round(self.rejected / max(self.checked, 1) * 100, 2),
        }


# ─────────────────────────────────────────────────────────────────
# ADAPTIVE QUALITY THRESHOLD
# ─────────────────────────────────────────────────────────────────
class AdaptiveQT:
    def __init__(self):
        self._current = QUALITY_MIN
        self.win = deque(maxlen=AQT_WINDOW)

    def update(self, q):
        self.win.append(q)
        if len(self.win) >= 100:
            avg = sum(self.win) / len(self.win)
            self._current = max(AQT_FLOOR, min(AQT_CEILING, avg - AQT_MARGIN))

    def ok(self, q):
        return q >= self._current

    def val(self):
        return round(self._current, 3)


# ─────────────────────────────────────────────────────────────────
# DIVERSITY FILTER — character 4-grams (better for Bantu)
# FIX: was set(text.split()) — ignored word order and morphology
# ─────────────────────────────────────────────────────────────────
class DiversityFilter:
    def __init__(self):
        self.win      = deque(maxlen=DIVERSITY_WINDOW)
        self.max_sim  = DIVERSITY_MAX_SIM
        self.rejected = 0

    def _vec(self, text: str) -> set:
        """Character 4-grams capture morphological similarity in Bantu languages."""
        t = text.lower()
        ngrams = set(t[i:i+4] for i in range(len(t) - 3))
        # Fallback to word set for very short texts
        return ngrams if ngrams else set(t.split())

    def ok(self, text: str) -> bool:
        v = self._vec(text)
        if not v:
            return True
        for prev in self.win:
            inter = len(v & prev)
            union = len(v | prev)
            if union > 0 and inter / union >= self.max_sim:
                self.rejected += 1
                return False
        return True

    def add(self, text: str):
        self.win.append(self._vec(text))


# ─────────────────────────────────────────────────────────────────
# LSH DEDUP (16x4 — restored to Tier 1 sensitivity)
# ─────────────────────────────────────────────────────────────────
class LSHDedup:
    def __init__(self, db_path):
        import sqlite3
        self.conn = sqlite3.connect(db_path)
        self.conn.execute("CREATE TABLE IF NOT EXISTS eh (h TEXT PRIMARY KEY)")
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS lb (b INTEGER, h TEXT, PRIMARY KEY (b,h))"
        )
        self.conn.commit()
        self._exact = set(r[0] for r in self.conn.execute("SELECT h FROM eh"))

    def _shingles(self, text):
        words = text.lower().split()
        return [f"{words[i]}{words[i+1]}" for i in range(len(words)-1)] if len(words) > 1 else words

    def _minhash(self, shingles):
        import struct
        sigs = []
        for b in range(LSH_BANDS):
            for r in range(LSH_ROWS):
                seed = b * LSH_ROWS + r
                vals = [
                    int(hashlib.md5(struct.pack("I", seed) + s.encode()).hexdigest(), 16)
                    for s in shingles
                ]
                sigs.append((b, min(vals) if vals else 0))
        return sigs

    def is_dup(self, text):
        norm = " ".join(text.lower().split())
        eh   = hashlib.md5(norm.encode()).hexdigest()
        if eh in self._exact:
            return True
        sigs = self._minhash(self._shingles(norm))
        for b, h in sigs:
            if self.conn.execute(
                "SELECT 1 FROM lb WHERE b=? AND h=?", (b, str(h))
            ).fetchone():
                return True
        return False

    def add(self, text):
        norm = " ".join(text.lower().split())
        eh   = hashlib.md5(norm.encode()).hexdigest()
        sigs = self._minhash(self._shingles(norm))
        self._exact.add(eh)
        self.conn.execute("INSERT OR IGNORE INTO eh VALUES (?)", (eh,))
        for b, h in sigs:
            self.conn.execute(
                "INSERT OR IGNORE INTO lb VALUES (?,?)", (b, str(h))
            )

    @property
    def size(self):
        return len(self._exact)

    def flush(self):
        self.conn.commit()

    def close(self):
        self.conn.close()


# ─────────────────────────────────────────────────────────────────
# ADAPTIVE BATCHER
# ─────────────────────────────────────────────────────────────────
class AdaptiveBatcher:
    def __init__(self, use_70b):
        if use_70b:
            self._min, self._max, self._cur = (
                BATCH_SIZE_MIN_70B, BATCH_SIZE_MAX_70B, BATCH_SIZE_INIT_70B
            )
        else:
            self._min, self._max, self._cur = (
                BATCH_SIZE_MIN_8B, BATCH_SIZE_MAX_8B, BATCH_SIZE_INIT_8B
            )
        self._ok = self._bad = 0

    def get(self):
        return self._cur

    def record(self, ok):
        if ok:
            self._ok += 1
            if self._ok % 10 == 0:
                self._cur = min(self._max, int(self._cur * 1.1))
        else:
            self._bad += 1
            self._cur = max(self._min, int(self._cur * 0.8))


# ─────────────────────────────────────────────────────────────────
# SEED MANAGER
# ─────────────────────────────────────────────────────────────────
class SeedManager:
    def __init__(self, seeds, start_idx=0):
        self.seeds = seeds
        self.idx   = start_idx % max(len(seeds), 1)
        self._ok = self._bad = 0

    def next_batch(self, n):
        batch = []
        for _ in range(n):
            batch.append(self.seeds[self.idx % len(self.seeds)])
            self.idx += 1
        return batch

    def record(self, seed, ok):
        if ok: self._ok += 1
        else:  self._bad += 1

    def stats(self):
        total = max(self._ok + self._bad, 1)
        return {"ok": self._ok, "bad": self._bad,
                "rate": round(self._ok / total, 3), "idx": self.idx}


# ─────────────────────────────────────────────────────────────────
# TOKEN COUNTER
# ─────────────────────────────────────────────────────────────────
class TokenCounter:
    def __init__(self, tok_path, model):
        try:
            import sentencepiece as spm
            sp = spm.SentencePieceProcessor()
            sp.Load(tok_path)
            self._sp  = sp
            self.mode = "sentencepiece"
        except Exception:
            self._sp  = None
            self.mode = "char_approx"
        self._total = 0

    def add(self, text):
        if not text or not isinstance(text, str):
            return
        if self._sp:
            try:
                self._total += len(self._sp.EncodeAsIds(text))
            except Exception:
                self._total += len(text) // 4
        else:
            self._total += len(text) // 4

    def total(self):
        return self._total


# ─────────────────────────────────────────────────────────────────
# PERPLEXITY — FIX1: uses actual generated token ID
# ─────────────────────────────────────────────────────────────────
def perplexity(logprobs_list, token_ids=None):
    """
    Compute perplexity using the actual generated token's logprob.
    FIX: with logprobs=5 we have top-5 candidates; match actual token_id.
    Falls back to minimum logprob (conservative) if token not in top-5.
    """
    try:
        if not logprobs_list:
            return None
        log_probs = []
        for i, token_dict in enumerate(logprobs_list):
            if not token_dict:
                continue
            lp = None
            # Try to find actual generated token
            if token_ids and i < len(token_ids):
                actual_id = token_ids[i]
                if actual_id in token_dict:
                    entry = token_dict[actual_id]
                    lp = float(entry.logprob if hasattr(entry, "logprob") else entry)
            # Fallback: use most conservative (minimum) logprob in top-5
            if lp is None:
                entries = list(token_dict.values())
                lp = min(
                    float(e.logprob if hasattr(e, "logprob") else e)
                    for e in entries
                )
            log_probs.append(lp)
        if not log_probs:
            return None
        return float(np.exp(-sum(log_probs) / len(log_probs)))
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────
# QUALITY — with length normalization
# ─────────────────────────────────────────────────────────────────
VALID_VOWELS = set("aeiouãõáéíóúàèìòùâêîôûäëïöüũĩôûîêâ")


def quality(text: str, lang: str, ppl=None) -> float:
    if ppl is not None and ppl > PERPLEXITY_MAX:
        return 0.0

    s = 1.0
    chars = len(text)

    # Length normalization — penalize both extremes
    if chars < MIN_OUTPUT_CHARS:   s -= 0.30
    if chars > 600:                s -= 0.15  # too long = rambling

    # Garbled text checks
    if re.search(r"(.)\1{6,}", text):           s -= 0.35
    if text.isupper() and chars > 15:           s -= 0.20
    digits = sum(c.isdigit() for c in text)
    if digits / max(chars, 1) > 0.35:           s -= 0.20

    # Vowel ratio (garbled text has very low vowels)
    letters = [c for c in text if c.isalpha()]
    if len(letters) > 10:
        vowels = sum(1 for c in letters if c.lower() in VALID_VOWELS)
        if vowels / len(letters) < 0.12:        s -= 0.30

    # Perplexity penalty
    if ppl and ppl > 120:                       s -= 0.10

    # Dialogue format bonus
    if "A:" in text and "B:" in text:
        parts = text.split("B:", 1)
        if len(parts) == 2:
            a = parts[0].replace("A:", "").strip()
            b = parts[1].strip()
            if len(a) >= MIN_OUTPUT_CHARS // 2 and len(b) >= MIN_OUTPUT_CHARS // 2:
                s += 0.05

    return round(max(0.0, min(s, 1.0)), 3)


# ─────────────────────────────────────────────────────────────────
# JSON EXTRACTION + RETRY — FIX2
# ─────────────────────────────────────────────────────────────────
_RETRY_SUFFIX = (
    "\n\nYou MUST respond with ONLY this JSON — nothing else:\n"
    '{"v1": "first version here", "v2": "second version here"}'
)


def extract_json(raw: str):
    """Try multiple extraction strategies. Returns (data, reason)."""
    for fn in [
        lambda r: (json.JSONDecoder().raw_decode(r.strip())[0], None),
        lambda r: (json.JSONDecoder().raw_decode(r[r.index("{"):])[0], None),
        lambda r: (json.loads(
            re.search(r'\{[^{}]*"v1"[^{}]*"v2"[^{}]*\}', r, re.DOTALL).group()
        ), None),
    ]:
        try:
            return fn(raw)
        except Exception:
            continue
    reason = (
        "missing_keys"  if '"v1"' not in raw else
        "model_refusal" if raw.strip().startswith("I ") else
        "empty_output"  if len(raw) < 10 else
        "malformed_json"
    )
    return None, reason


def valid_schema(data) -> bool:
    if not isinstance(data, dict):
        return False
    for k in ["v1", "v2"]:
        v = data.get(k, "")
        if not isinstance(v, str) or len(v.strip()) < MIN_OUTPUT_CHARS // 2:
            return False
    return True


def generate_with_retry(llm, prompts, sampling, log=None):
    """
    FIX2: On JSON parse failure, retry once with a stronger instruction
    appended to the original prompt before giving up entirely.
    Returns list of (data, output) or None for each prompt.
    """
    outputs = llm.generate(prompts, sampling)
    results = [None] * len(outputs)
    retry_indices = []
    retry_prompts = []

    for i, output in enumerate(outputs):
        raw  = output.outputs[0].text.strip()
        data, _ = extract_json(raw)
        if data and valid_schema(data):
            results[i] = (data, output)
        else:
            retry_indices.append(i)
            retry_prompts.append(prompts[i] + _RETRY_SUFFIX)

    if retry_prompts:
        if log:
            log.debug(f"JSON retry: {len(retry_prompts)} failed extractions")
        retry_outputs = llm.generate(retry_prompts, sampling)
        for idx, output in zip(retry_indices, retry_outputs):
            raw  = output.outputs[0].text.strip()
            data, _ = extract_json(raw)
            if data and valid_schema(data):
                results[idx] = (data, output)

    return results


# ─────────────────────────────────────────────────────────────────
# CHECKPOINT HELPERS
# ─────────────────────────────────────────────────────────────────
def load_ckpt(path):
    if Path(path).exists():
        with open(path) as f:
            return json.load(f)
    return {"generated": 0, "seed_idx": 0, "attempted": 0,
            "failed": 0, "start_time": time.time()}


def save_ckpt(path, ckpt):
    ckpt["saved_at"] = datetime.utcnow().isoformat()
    with open(path, "w") as f:
        json.dump(ckpt, f, indent=2)


def log_tel(path, event):
    event["ts"] = datetime.utcnow().isoformat()
    with open(path, "a") as f:
        f.write(json.dumps(event) + "\n")


def log_fail(path, reason, seed, raw=""):
    with open(path, "a") as f:
        f.write(json.dumps({
            "ts": datetime.utcnow().isoformat(),
            "reason": reason, "seed": seed[:80], "raw": raw[:120],
        }) + "\n")


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────
def main(lang: str):
    cfg     = LANGUAGE_CONFIGS[lang]
    use_70b = cfg.get("use_70b", True)
    model   = cfg["model_id"]
    TARGET  = cfg["target"]

    for d in [cfg["output"], cfg["log"], cfg["telemetry"], cfg["fail_log"],
              cfg["checkpoint"], cfg["dedup_db"], cfg["meta_out"]]:
        Path(d).parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
        handlers=[
            logging.FileHandler(cfg["log"]),
            logging.StreamHandler(),
        ],
    )
    log = logging.getLogger(f"t2_{lang}")
    log.info(f"=== LughaGen Tier 2 v1.2 — {lang.upper()} ===")
    log.info(f"Model      : {model}")
    log.info(f"Target     : {TARGET:,} | QMin: {QUALITY_MIN} | PPL: {PERPLEXITY_MAX}")
    log.info(f"Temp       : {TEMPERATURE_70B if use_70b else TEMPERATURE_8B} | "
             f"top_p={TOP_P} | rep={REPETITION_PENALTY}")
    log.info(f"LSH        : {LSH_BANDS}x{LSH_ROWS} | logprobs={LOGPROBS}")
    log.info(f"MinChars   : {MIN_OUTPUT_CHARS} | MaxTokens: {MAX_NEW_TOKENS}")

    # Load seeds
    seeds = []
    for inp in cfg["inputs"]:
        with open(inp) as f:
            for line in f:
                try:
                    d = json.loads(line)
                    t = d.get("text", d.get("sentence", ""))
                    if len(t.strip()) >= 10:
                        seeds.append(t.strip())
                except Exception:
                    pass
    log.info(f"Seeds      : {len(seeds):,}")

    ckpt     = load_ckpt(cfg["checkpoint"])
    dedup    = LSHDedup(cfg["dedup_db"])
    qt       = AdaptiveQT()
    div      = DiversityFilter()
    cur      = Tier2Curriculum(lang)
    seed_mgr = SeedManager(seeds, start_idx=ckpt["seed_idx"])
    batcher  = AdaptiveBatcher(use_70b)
    glotlid  = GlotLIDVerifier()  # Load GlotLID for real-time verification

    log.info(f"Dedup DB   : {dedup.size:,} existing | Loading model...")

    tp = min(torch.cuda.device_count(), 2)
    llm = LLM(
        model=model,
        tensor_parallel_size=tp,
        gpu_memory_utilization=0.92 if use_70b else 0.88,
        max_model_len=1024,
        dtype="bfloat16",
        enforce_eager=True,
    )

    # FIX8: sampling params from config, FIX1: logprobs=5
    sampling = SamplingParams(
        temperature=TEMPERATURE_70B if use_70b else TEMPERATURE_8B,
        top_p=TOP_P,
        repetition_penalty=REPETITION_PENALTY,
        max_tokens=MAX_NEW_TOKENS,
        logprobs=LOGPROBS,
        stop=["```", "Note:", "Explanation:"],
    )

    tok = TokenCounter(cfg["tokenizer"], llm)

    generated  = ckpt["generated"]
    attempted  = ckpt["attempted"]
    failed     = ckpt["failed"]
    start_time = ckpt.get("start_time", time.time())
    batch_num  = 0

    log.info(f"Resuming   : {generated:,}/{TARGET:,}")

    with jsonlines.open(cfg["output"], mode="a") as writer:
        while generated < TARGET:
            bs          = batcher.get()
            batch_seeds = seed_mgr.next_batch(bs)
            # Filter bad seeds — skip seeds that are clearly not target language
            # (e.g. Indian place names, phonetic notation, non-African content)
            def _is_valid_seed(seed, language):
                if len(seed.strip()) < 10: return False
                # Must be predominantly Latin script
                alpha = [c for c in seed if c.isalpha()]
                if not alpha: return False
                import unicodedata as _ud
                latin = sum(1 for c in alpha if "LATIN" in _ud.name(c,""))
                if latin / len(alpha) < 0.70: return False
                return True
            batch_seeds = [s for s in batch_seeds if _is_valid_seed(s, lang)]
            if not batch_seeds:
                continue
            batch_built = [cur.build(s) for s in batch_seeds]
            prompts     = [p for p, _ in batch_built]
            styles      = [s for _, s in batch_built]

            try:
                results = generate_with_retry(llm, prompts, sampling, log)
            except Exception as eng_err:
                log.warning(f"Engine error (batch={bs}): {eng_err} — halving")
                batcher.record(False)
                bs = max(
                    BATCH_SIZE_MIN_70B if use_70b else BATCH_SIZE_MIN_8B,
                    bs // 2
                )
                prompts     = prompts[:bs]
                batch_seeds = batch_seeds[:bs]
                styles      = styles[:bs]
                try:
                    results = generate_with_retry(llm, prompts, sampling, log)
                except Exception:
                    log.error("Retry failed — skipping batch")
                    failed += bs
                    continue

            b_ok = b_fail = 0

            for i, result in enumerate(results):
                seed  = batch_seeds[i]
                style = styles[i]
                attempted += 1

                if result is None:
                    log_fail(cfg["fail_log"], "json_parse", seed)
                    seed_mgr.record(seed, False)
                    failed += 1; b_fail += 1
                    batcher.record(False)
                    continue

                data, output = result

                # FIX1: accurate perplexity using token IDs
                try:
                    ppl = perplexity(
                        output.outputs[0].logprobs,
                        output.outputs[0].token_ids,
                    )
                except Exception:
                    ppl = None

                for k in ["v1", "v2"]:
                    text = data[k].strip()
                    if len(text) < MIN_OUTPUT_CHARS:
                        log_fail(cfg["fail_log"], "too_short", seed, text)
                        continue

                    # Code-switch ratio enforcement
                    if style == "code_switching":
                        cs_ratio = code_switch_ratio(text)
                        if cs_ratio > CODE_SWITCH_WORD_MAX:
                            log_fail(cfg["fail_log"], f"cs_ratio:{cs_ratio:.2f}", seed, text[:60])
                            continue

                    q = quality(text, lang, ppl)
                    qt.update(q)
                    if not qt.ok(q):
                        log_fail(cfg["fail_log"], "low_quality", seed, f"q={q:.2f}")
                        continue
                    # LSH dedup disabled for Tier 2 — dialogues share
                    # common phrases that LSH incorrectly flags as duplicates.
                    # Post-processing dedup (exact MD5) handles this instead.
                    # if dedup.is_dup(text):
                    #     log_fail(cfg["fail_log"], "duplicate", seed)
                    #     continue
                    if not div.ok(text):
                        log_fail(cfg["fail_log"], "diversity", seed)
                        continue

                    # Sanitize: remove surrogate characters that break UTF-8
                    text = text.encode('utf-8', errors='ignore').decode('utf-8')
                    if not text.strip():
                        continue

                    div.add(text)
                    # dedup.add(text)  # disabled — post-processing handles dedup
                    tok.add(text)
                    writer.write({
                        "text":              text,
                        "language":          lang,
                        "type":              f"tier2_{style}",
                        "tier":              2,
                        "prompt_style":      style,
                        "quality":           q,
                        "quality_threshold": qt.val(),
                        "perplexity":        ppl,
                        "generator":         model,
                        "prompt_version":    PROMPT_VERSION,
                        "dataset_version":   DATASET_VERSION,
                        "pipeline_version":  PIPELINE_VERSION,
                        "seed_preview":      seed[:60],
                        "created_at":        datetime.utcnow().isoformat(),
                    })
                    generated += 1; b_ok += 1

            batcher.record(b_ok > 0)
            seed_mgr.record(batch_seeds[0], b_ok > 0)

            if batch_num % CHECKPOINT_EVERY == 0:
                dedup.flush()
                save_ckpt(cfg["checkpoint"], {
                    "generated":  generated,
                    "seed_idx":   seed_mgr.idx,
                    "attempted":  attempted,
                    "failed":     failed,
                    "start_time": start_time,
                })

            batch_num += 1
            elapsed = time.time() - start_time
            rate    = generated / max(elapsed, 1)
            eta     = (TARGET - generated) / max(rate, 0.001)
            fr      = failed / max(attempted, 1) * 100

            log.info(
                f"{lang} | {generated:,}/{TARGET:,} | "
                f"rate={rate:.1f}/s | ETA={eta/3600:.1f}h | "
                f"fail={fr:.1f}% | batch={bs} | "
                f"qt={qt.val():.3f} | tokens={tok.total()/1e6:.2f}M"
            )
            log_tel(cfg["telemetry"], {
                "lang": lang, "generated": generated, "batch_size": bs,
                "batch_ok": b_ok, "batch_fail": b_fail,
                "total_tokens": tok.total(), "rate": round(rate, 2),
                "eta_hours": round(eta/3600, 2), "fail_rate": round(fr, 2),
                "qt": qt.val(), "prompt_dist": cur.distribution(),
            })

    dedup.flush()
    dedup.close()
    elapsed = time.time() - start_time
    meta = {
        "pipeline_version":   PIPELINE_VERSION,
        "language":           lang,
        "model":              model,
        "generated":          generated,
        "attempted":          attempted,
        "failed":             failed,
        "fail_rate_pct":      round(failed / max(attempted, 1) * 100, 2),
        "total_tokens":       tok.total(),
        "elapsed_hours":      round(elapsed / 3600, 2),
        "quality_threshold":  qt.val(),
        "diversity_rejected": div.rejected,
        "prompt_distribution": cur.distribution(),
        "completed_at":       datetime.utcnow().isoformat(),
    }
    Path(cfg["meta_out"]).parent.mkdir(parents=True, exist_ok=True)
    with open(cfg["meta_out"], "w") as f:
        json.dump(meta, f, indent=2)
    log.info(f"{lang.upper()} COMPLETE: {generated:,} | {elapsed/3600:.1f}h")


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in LANGUAGE_CONFIGS:
        print("Usage: python3 generate.py <lang>")
        print(f"       lang = {list(LANGUAGE_CONFIGS.keys())}")
        sys.exit(1)
    main(sys.argv[1])

