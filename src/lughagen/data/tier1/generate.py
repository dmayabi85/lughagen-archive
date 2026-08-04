"""
LughaGen Tier 1 Generation — Main Script
High-quality conservative parallel corpus generation.

Run as:
  CUDA_VISIBLE_DEVICES=0,1 python3 generate.py kamba
  CUDA_VISIBLE_DEVICES=2,3 python3 generate.py kikuyu
  CUDA_VISIBLE_DEVICES=4,5 python3 generate.py luo
  CUDA_VISIBLE_DEVICES=6,7 python3 generate.py swahili
"""

import os
import sys
import json
import re
import time
import math
import sqlite3
import hashlib
import logging
import random
import jsonlines
from pathlib import Path
from datetime import datetime
from collections import deque, defaultdict
from vllm import LLM, SamplingParams

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    PIPELINE_VERSION, PROMPT_VERSION, DATASET_VERSION,
    MODEL_70B, MODEL_8B, QUALITY_MIN, PERPLEXITY_MAX,
    AQT_FLOOR, AQT_CEILING, AQT_MARGIN, AQT_WINDOW,
    BATCH_SIZE_MIN_70B, BATCH_SIZE_MAX_70B, BATCH_SIZE_INIT_70B,
    BATCH_SIZE_MIN_8B,  BATCH_SIZE_MAX_8B,  BATCH_SIZE_INIT_8B,
    CHECKPOINT_EVERY, GPU_UTIL_MAX, GPU_UTIL_SLEEP,
    LSH_BANDS, LSH_ROWS, DIVERSITY_MAX_SIM, DIVERSITY_WINDOW,
    SEED_MAX_FAILS, SEED_FAIL_WINDOW, LANGUAGE_CONFIGS
)
from prompts import Tier1Curriculum


# ── LSH DEDUP ─────────────────────────────────────────────────────
class LSHDedup:
    def __init__(self, db_path: str):
        self.bands    = LSH_BANDS
        self.rows     = LSH_ROWS
        self.n        = LSH_BANDS * LSH_ROWS
        self._exact   = set()
        self._buckets = [dict() for _ in range(self.bands)]
        self._conn    = self._init_db(db_path)

    def _init_db(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(path, check_same_thread=False)
        conn.execute("CREATE TABLE IF NOT EXISTS eh (hash TEXT PRIMARY KEY, ts TEXT)")
        conn.execute("CREATE TABLE IF NOT EXISTS lb (band INT, bucket TEXT, PRIMARY KEY(band,bucket))")
        conn.commit()
        for (h,) in conn.execute("SELECT hash FROM eh"):
            self._exact.add(h)
        for (b, bk) in conn.execute("SELECT band, bucket FROM lb"):
            self._buckets[b][bk] = True
        return conn

    def _sh(self, text):
        t = text.lower().strip()
        return {t[i:i+3] for i in range(len(t)-2)}

    def _mh(self, sh):
        if not sh: return [0]*self.n
        return [min(int(hashlib.md5(f"{i}:{s}".encode()).hexdigest(),16)
                    for s in sh) for i in range(self.n)]

    def _bb(self, sig):
        return [hashlib.md5(
            f"{b}:{tuple(sig[b*self.rows:(b+1)*self.rows])}".encode()
        ).hexdigest() for b in range(self.bands)]

    def is_dup(self, text):
        eh = hashlib.md5(text.lower().strip().encode()).hexdigest()
        if eh in self._exact: return True
        sig = self._mh(self._sh(text))
        bb  = self._bb(sig)
        for i, bk in enumerate(bb):
            if bk in self._buckets[i]: return True
        self._exact.add(eh)
        for i, bk in enumerate(bb):
            self._buckets[i][bk] = True
        try:
            self._conn.execute("INSERT OR IGNORE INTO eh VALUES(?,?)",
                               (eh, datetime.utcnow().isoformat()))
            for i, bk in enumerate(bb):
                self._conn.execute("INSERT OR IGNORE INTO lb VALUES(?,?)", (i, bk))
        except Exception: pass
        return False

    def flush(self):
        try: self._conn.commit()
        except Exception: pass

    def close(self):
        self.flush(); self._conn.close()

    @property
    def size(self): return len(self._exact)


# ── ADAPTIVE QUALITY THRESHOLD ────────────────────────────────────
class AdaptiveQT:
    def __init__(self):
        self.floor    = AQT_FLOOR
        self.ceiling  = AQT_CEILING
        self.margin   = AQT_MARGIN
        self.win      = deque(maxlen=AQT_WINDOW)
        self._current = QUALITY_MIN

    def update(self, score):
        self.win.append(score)
        if len(self.win) >= 50:
            mean = sum(self.win) / len(self.win)
            self._current = max(self.floor,
                                min(self.ceiling, mean - self.margin))

    def ok(self, score): return score >= self._current
    def val(self): return round(self._current, 3)

    def stats(self):
        return {
            "threshold": self.val(),
            "window_mean": round(sum(self.win)/len(self.win), 3) if self.win else 0,
            "window_size": len(self.win),
        }


# ── DIVERSITY FILTER ──────────────────────────────────────────────
class DiversityFilter:
    def __init__(self):
        self.win      = deque(maxlen=DIVERSITY_WINDOW)
        self.max_sim  = DIVERSITY_MAX_SIM
        self.rejected = 0

    def _ws(self, text): return set(re.findall(r'\b\w+\b', text.lower()))
    def _j(self, a, b):
        if not a or not b: return 0.0
        return len(a & b) / len(a | b)

    def ok(self, text):
        ws = self._ws(text)
        for recent in list(self.win)[-100:]:
            if self._j(ws, recent) > self.max_sim:
                self.rejected += 1
                return False
        return True

    def add(self, text):
        self.win.append(self._ws(text))


# ── GPU MONITOR ───────────────────────────────────────────────────
class GPUMonitor:
    def __init__(self):
        self.mode = "none"
        self._handles = []
        self._nvml = None
        try:
            import pynvml
            pynvml.nvmlInit()
            env = os.environ.get("CUDA_VISIBLE_DEVICES", "")
            ids = ([int(x) for x in env.split(",")]
                   if env and env != "NoDeviceFiles"
                   else list(range(pynvml.nvmlDeviceGetCount())))
            self._handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in ids]
            self._nvml    = pynvml
            self.mode     = "pynvml"
        except Exception:
            self.mode = "none"

    def util(self):
        if self.mode == "pynvml":
            try:
                u = [self._nvml.nvmlDeviceGetUtilizationRates(h).gpu/100.0
                     for h in self._handles]
                return sum(u)/len(u) if u else 0.0
            except Exception: return 0.0
        return 0.0

    def free_gb(self):
        if self.mode == "pynvml":
            try:
                return round(sum(
                    self._nvml.nvmlDeviceGetMemoryInfo(h).free
                    for h in self._handles) / (1024**3), 2)
            except Exception: return 0.0
        return 0.0

    def wait(self, log=None):
        u = self.util()
        if u >= GPU_UTIL_MAX:
            msg = f"GPU {u*100:.0f}% saturated — waiting {GPU_UTIL_SLEEP}s"
            if log: log.info(msg)
            else: print(msg)
            time.sleep(GPU_UTIL_SLEEP)
            if self.util() >= GPU_UTIL_MAX:
                time.sleep(GPU_UTIL_SLEEP)


# ── ADAPTIVE BATCHER ─────────────────────────────────────────────
class AdaptiveBatcher:
    def __init__(self, use_70b: bool):
        if use_70b:
            self.size = BATCH_SIZE_INIT_70B
            self.mn   = BATCH_SIZE_MIN_70B
            self.mx   = BATCH_SIZE_MAX_70B
            self.step = 10
        else:
            self.size = BATCH_SIZE_INIT_8B
            self.mn   = BATCH_SIZE_MIN_8B
            self.mx   = BATCH_SIZE_MAX_8B
            self.step = 20
        self.win = deque(maxlen=20)

    def record(self, ok): self.win.append(1 if ok else 0)

    def update(self, free_gb=None):
        if len(self.win) < 10: return
        r = sum(self.win) / len(self.win)
        if r > 0.85: self.size = min(self.size + self.step, self.mx)
        elif r < 0.60: self.size = max(self.size - self.step, self.mn)
        if free_gb is not None and free_gb < 5.0:
            self.size = max(self.size - self.step, self.mn)

    def get(self): return self.size


# ── SEED QUALITY TRACKER ─────────────────────────────────────────
class SeedTracker:
    def __init__(self):
        self.history  = defaultdict(lambda: deque(maxlen=SEED_FAIL_WINDOW))
        self.scores   = defaultdict(lambda: 1.0)
        self.black    = set()
        self.dropped  = 0

    def _key(self, seed): return hashlib.md5(seed[:80].encode()).hexdigest()[:12]

    def record(self, seed, ok):
        k = self._key(seed)
        self.history[k].append(1 if ok else 0)
        h = self.history[k]
        if len(h) >= SEED_MAX_FAILS and sum(h) == 0:
            self.black.add(k); self.dropped += 1; return
        if len(h) >= 3:
            self.scores[k] = sum(h) / len(h)

    def is_black(self, seed): return self._key(seed) in self.black

    def stats(self):
        return {"tracked": len(self.history),
                "blacklisted": len(self.black),
                "dropped": self.dropped}


# ── SEED MANAGER ─────────────────────────────────────────────────
class SeedManager:
    def __init__(self, seeds, start_idx=0):
        self.seeds   = seeds
        self.n       = len(seeds)
        self.idx     = start_idx
        self.tracker = SeedTracker()

    def next_batch(self, size):
        batch = []
        tried = 0
        while len(batch) < size and tried < size * 4:
            seed = self.seeds[self.idx % self.n]
            self.idx += 1; tried += 1
            if self.tracker.is_black(seed): continue
            w = self.tracker.scores[self.tracker._key(seed)]
            if w < 0.5 and random.random() > w * 2: continue
            batch.append(seed)
        if len(batch) < size:
            good = [s for s in self.seeds if not self.tracker.is_black(s)]
            if good:
                batch += random.choices(good, k=size - len(batch))
        return batch

    def record(self, seed, ok): self.tracker.record(seed, ok)
    def curr(self): return self.idx
    def stats(self): return self.tracker.stats()


# ── TOKEN COUNTER ─────────────────────────────────────────────────
class TokenCounter:
    def __init__(self, tok_path, model_id):
        self.mode = "estimate"; self._total = 0; self._sp = None
        try:
            import sentencepiece as spm
            if Path(tok_path).exists():
                self._sp = spm.SentencePieceProcessor()
                self._sp.Load(tok_path); self.mode = "sentencepiece"
        except Exception: pass

    def count(self, texts):
        n = (sum(len(self._sp.EncodeAsIds(t)) for t in texts)
             if self.mode == "sentencepiece"
             else sum(len(t)//3 for t in texts))
        self._total += n; return n

    def total(self): return self._total


# ── PERPLEXITY ────────────────────────────────────────────────────
def perplexity(logprobs):
    if logprobs is None: return None
    try:
        vals = []
        for lp in logprobs:
            if lp is None: continue
            if isinstance(lp, dict):
                for v in lp.values():
                    x = (v.logprob if hasattr(v,'logprob') else
                         float(v) if isinstance(v,(int,float)) else None)
                    if x and not math.isnan(x) and not math.isinf(x):
                        vals.append(x); break
            elif isinstance(lp,(int,float)): vals.append(float(lp))
        return round(math.exp(-sum(vals)/len(vals)),2) if vals else None
    except Exception: return None


# ── QUALITY SCORE ─────────────────────────────────────────────────
LANG_MARKERS = {
    "kikuyu":  re.compile(r'[ĩũ]|nĩ|gũ|kũ|mũ|wĩ|tũ|atĩ', re.I),
    "kamba":   re.compile(r'[ĩũ]|kũ|mũ|kĩ|nthĩ|ũkĩ', re.I),
    "luo":     re.compile(r'\b(ni|ma|ka|gi|od|ok|dhi|en|ot)\b', re.I),
    "swahili": re.compile(r'\b(na|ya|wa|ku|ni|kwa|hii|lakini|pia)\b', re.I),
}

def quality(text, lang, ppl=None):
    if ppl is not None and ppl > PERPLEXITY_MAX: return 0.0
    s = 1.0
    if len(text) < 50: s -= 0.20
    if re.search(r'(.)\1{4,}', text): s -= 0.30
    if re.search(r'\b(\w+)\s+\1\s+\1', text): s -= 0.20
    if text.isupper() and len(text) > 10: s -= 0.20
    if sum(c.isdigit() for c in text)/max(len(text),1) > 0.3: s -= 0.20
    # Penalize English leakage — exempt Latin-script languages (Swahili is fully ASCII)
    LATIN_SCRIPT_LANGS = {"swahili"}
    words  = text.split()
    if lang not in LATIN_SCRIPT_LANGS:
        ascii_w = sum(1 for w in words if w.isascii() and w.isalpha() and len(w)>2)
        if words and ascii_w/len(words) > 0.50: s -= 0.25
    if ppl and ppl > 80: s -= 0.15
    m = LANG_MARKERS.get(lang)
    if m and len(m.findall(text))/max(len(words),1) > 0.15: s += 0.10
    return round(max(0.0, min(s, 1.0)), 3)


# ── JSON EXTRACTION ───────────────────────────────────────────────
def extract_json(raw):
    # Strategy 1: raw_decode — tolerates trailing text after valid JSON
    try:
        dec = json.JSONDecoder()
        obj, _ = dec.raw_decode(raw.strip())
        return obj, None
    except Exception: pass
    # Strategy 2: find outermost {...} and raw_decode from there
    try:
        start = raw.index("{")
        dec = json.JSONDecoder()
        obj, _ = dec.raw_decode(raw[start:])
        return obj, None
    except Exception: pass
    # Strategy 3: regex extract then raw_decode
    try:
        m = re.search(r'\{[^{}]*"v1"[^{}]*"v2"[^{}]*\}', raw, re.DOTALL)
        if m:
            obj, _ = json.JSONDecoder().raw_decode(m.group())
            return obj, None
    except Exception: pass
    # Strategy 4: clean and parse the substring
    try:
        chunk = raw[raw.find("{"):raw.rfind("}")+1]
        chunk = re.sub(r'[\x00-\x1f]', ' ', chunk)
        chunk = re.sub(r',\s*}', '}', chunk)
        return json.loads(chunk), None
    except Exception: pass
    reason = ("missing_keys"  if '"v1"' not in raw else
              "model_refusal" if raw.strip().startswith("I ") else
              "empty_output"  if len(raw) < 10 else "malformed_json")
    return None, reason

def valid_schema(data):
    if not isinstance(data, dict): return False
    for k in ["v1","v2"]:
        v = data.get(k,"")
        if not isinstance(v,str) or len(v.strip())<15: return False
    return True


# ── CHECKPOINT ────────────────────────────────────────────────────
def load_ckpt(path):
    if Path(path).exists():
        with open(path) as f: return json.load(f)
    return {"generated":0,"seed_idx":0,"attempted":0,
            "failed":0,"start_time":time.time()}

def save_ckpt(path, ckpt):
    ckpt["saved_at"] = datetime.utcnow().isoformat()
    with open(path,"w") as f: json.dump(ckpt, f, indent=2)

def log_tel(path, event):
    event["ts"] = datetime.utcnow().isoformat()
    with open(path,"a") as f: f.write(json.dumps(event)+"\n")

def log_fail(path, reason, seed, raw=""):
    with open(path,"a") as f:
        f.write(json.dumps({"ts":datetime.utcnow().isoformat(),
            "reason":reason,"seed":seed[:80],"raw":raw[:120]})+"\n")


# ── MAIN ─────────────────────────────────────────────────────────
def main(lang):
    cfg = LANGUAGE_CONFIGS[lang]
    for d in [cfg["output"],cfg["log"],cfg["telemetry"],
              cfg["fail_log"],cfg["checkpoint"],cfg["dedup_db"],cfg["meta_out"]]:
        Path(d).parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
        handlers=[logging.FileHandler(cfg["log"]), logging.StreamHandler()]
    )
    log = logging.getLogger(f"t1_{lang}")

    use_70b = cfg.get("use_70b", True)
    model   = cfg["model_id"]

    log.info("="*60)
    log.info(f"LughaGen Tier 1 {PIPELINE_VERSION} — {lang.upper()}")
    log.info(f"Model   : {model} ({'70B' if use_70b else '8B'})")
    log.info(f"Target  : {cfg['target']:,} | QMin: {QUALITY_MIN} | PPL: {PERPLEXITY_MAX}")
    log.info(f"CUDA    : {os.environ.get('CUDA_VISIBLE_DEVICES','all')}")
    log.info("="*60)

    ckpt = load_ckpt(cfg["checkpoint"])
    if ckpt["generated"] >= cfg["target"]:
        log.info("Already complete. Exiting."); return

    seeds = []
    for p in cfg["inputs"]:
        if Path(p).exists():
            with jsonlines.open(p) as r:
                for item in r:
                    t = item.get("text","").strip()
                    if len(t) > 30: seeds.append(t)
    log.info(f"Seeds loaded: {len(seeds):,}")
    if not seeds: log.error("No seeds."); return

    dedup    = LSHDedup(cfg["dedup_db"])
    qt       = AdaptiveQT()
    div      = DiversityFilter()
    cur      = Tier1Curriculum(lang)
    seed_mgr = SeedManager(seeds, start_idx=ckpt["seed_idx"])
    gpu      = GPUMonitor()
    batcher  = AdaptiveBatcher(use_70b)
    tok      = TokenCounter(cfg["tokenizer"], model)

    log.info(f"GPU monitor  : {gpu.mode}")
    log.info(f"Token counter: {tok.mode}")
    log.info(f"Dedup DB     : {dedup.size:,} existing entries")

    import torch
    tp = min(torch.cuda.device_count(), 2)
    log.info(f"Loading {model} on {tp} GPU(s)...")

    llm = LLM(
        model=model,
        tensor_parallel_size=tp,
        gpu_memory_utilization=0.92 if use_70b else 0.88,
        max_model_len=1024,
        dtype="bfloat16",
        enforce_eager=True,
    )
    # Neither 70B nor 8B-Instruct should use \n\n as stop token
    # 70B with few-shot priming also outputs multi-line JSON
    stop_tokens = ["```","Note:","Explanation:","Translation:"]
    sampling = SamplingParams(
        temperature=0.75,   # stricter than Tier 2 (0.85)
        top_p=0.90,
        max_tokens=400,
        logprobs=1,
        stop=stop_tokens,
    )

    generated  = ckpt["generated"]
    attempted  = ckpt.get("attempted",0)
    failed     = ckpt.get("failed",0)
    start_time = ckpt.get("start_time",time.time())
    batch_num  = 0
    TARGET     = cfg["target"]

    with jsonlines.open(cfg["output"], mode="a") as writer:
        while generated < TARGET:
            gpu.wait(log)
            bs           = batcher.get()
            batch_seeds  = seed_mgr.next_batch(bs)
            batch_built  = [cur.build(s) for s in batch_seeds]
            prompts      = [p for p,_ in batch_built]
            styles       = [s for _,s in batch_built]

            t0 = time.time()
            try:
                outputs = llm.generate(prompts, sampling)
            except Exception as eng_err:
                log.warning(f"Engine error (batch={bs}): {eng_err} — halving batch and retrying")
                batcher.record(False)
                bs          = max(BATCH_SIZE_MIN_70B if use_70b else BATCH_SIZE_MIN_8B, bs // 2)
                prompts     = prompts[:bs]
                batch_seeds = batch_seeds[:bs]
                batch_built = batch_built[:bs]
                styles      = styles[:bs]
                try:
                    outputs = llm.generate(prompts, sampling)
                except Exception:
                    log.error("Retry failed — skipping batch")
                    failed += bs
                    continue
            gt = time.time() - t0

            b_ok = b_fail = 0
            b_texts = []

            for seed, output, style in zip(batch_seeds, outputs, styles):
                attempted += 1
                raw = output.outputs[0].text.strip()
                data, reason = extract_json(raw)

                if data is None or not valid_schema(data):
                    log_fail(cfg["fail_log"], reason or "schema", seed, raw)
                    seed_mgr.record(seed, False)
                    failed += 1; b_fail += 1
                    batcher.record(False); continue

                try: ppl = perplexity(output.outputs[0].logprobs)
                except Exception: ppl = None

                acc = 0
                for k in ["v1","v2"]:
                    text = data[k].strip()
                    if len(text) < 20: continue

                    q = quality(text, lang, ppl)
                    qt.update(q)
                    if not qt.ok(q):
                        log_fail(cfg["fail_log"],"low_quality",seed,f"q={q:.2f}"); continue
                    # dedup disabled — post-processing pass will handle
                    # if dedup.is_dup(text):
                    #     log_fail(cfg["fail_log"],"duplicate",seed); continue
                    if not div.ok(text):
                        log_fail(cfg["fail_log"],"diversity",seed); continue

                    div.add(text)
                    writer.write({
                        "text":             text,
                        "language":         lang,
                        "type":             f"tier1_{style}",
                        "tier":             1,
                        "prompt_style":     style,
                        "quality":          q,
                        "quality_threshold":qt.val(),
                        "perplexity":       ppl,
                        "generator":        model,
                        "prompt_version":   PROMPT_VERSION,
                        "dataset_version":  DATASET_VERSION,
                        "pipeline_version": PIPELINE_VERSION,
                        "seed_preview":     seed[:60],
                        "created_at":       datetime.utcnow().isoformat(),
                    })
                    b_texts.append(text)
                    generated += 1; acc += 1; b_ok += 1
                    if generated >= TARGET: break

                seed_mgr.record(seed, acc > 0)
                batcher.record(acc > 0)
                if generated >= TARGET: break

            free_gb = gpu.free_gb()
            batcher.update(free_gb)
            tok.count(b_texts)

            if batch_num % 10 == 0: dedup.flush()

            if generated % CHECKPOINT_EVERY < bs:
                save_ckpt(cfg["checkpoint"], {
                    "generated":  generated,
                    "seed_idx":   seed_mgr.curr(),
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
                f"gpu={gpu.util()*100:.0f}% | "
                f"qt={qt.val():.3f} | tokens={tok.total()/1e6:.2f}M"
            )
            log_tel(cfg["telemetry"], {
                "lang":lang,"generated":generated,
                "batch_size":bs,"batch_ok":b_ok,"batch_fail":b_fail,
                "total_tokens":tok.total(),"rate":round(rate,2),
                "eta_hours":round(eta/3600,2),"fail_rate":round(fr,2),
                "gpu_util":round(gpu.util()*100,1),
                "qt":qt.val(),"qt_stats":qt.stats(),
                "div_rejected":div.rejected,
                "seed_stats":seed_mgr.stats(),
                "prompt_dist":cur.distribution(),
            })

    dedup.flush(); dedup.close()
    elapsed = time.time() - start_time

    meta = {
        "pipeline_version":   PIPELINE_VERSION,
        "dataset_version":    DATASET_VERSION,
        "model":              model,
        "language":           lang,
        "generated":          generated,
        "attempted":          attempted,
        "failed":             failed,
        "fail_rate_pct":      round(failed/max(attempted,1)*100,2),
        "total_tokens":       tok.total(),
        "tokenizer_mode":     tok.mode,
        "elapsed_hours":      round(elapsed/3600,2),
        "quality_threshold":  qt.val(),
        "quality_stats":      qt.stats(),
        "diversity_rejected": div.rejected,
        "seed_stats":         seed_mgr.stats(),
        "prompt_distribution":cur.distribution(),
        "completed_at":       datetime.utcnow().isoformat(),
    }
    Path(cfg["meta_out"]).parent.mkdir(parents=True, exist_ok=True)
    with open(cfg["meta_out"],"w") as f: json.dump(meta, f, indent=2)

    log.info("="*60)
    log.info(f"{lang.upper()} COMPLETE: {generated:,} samples | "
             f"{tok.total()/1e6:.2f}M tokens | {elapsed/3600:.1f}h")
    log.info(f"Prompts : {cur.distribution()}")
    log.info(f"Seeds   : {seed_mgr.stats()}")
    log.info("="*60)


if __name__ == "__main__":
    lang = sys.argv[1] if len(sys.argv) > 1 else None
    if lang not in LANGUAGE_CONFIGS:
        print(f"Usage: python3 generate.py <language>")
        print(f"Available: {list(LANGUAGE_CONFIGS.keys())}")
        sys.exit(1)
    main(lang)
