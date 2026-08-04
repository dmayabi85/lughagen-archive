"""
fluency_scorer.py — LughaGen Tier 3 External Fluency Scoring

Computes an independent fluency score using a SECOND model (afro-xlmr-base
by default), distinct from the generator (LLaMA). Reduces self-confirmation
bias from using the same model to both generate and score.

Scoring strategy: pseudo-log-likelihood (PLL) via masked language modeling.
For each text, we mask tokens one at a time and average the log-prob of
recovering the true token. Higher PLL = more fluent.

Output: fluency_scored.jsonl
  Adds:
    fluency_pll      — raw average log-prob (negative; closer to 0 = better)
    fluency_score    — normalized to [0, 1] using a corpus-level z-score
    combined_quality — HEURISTIC_WEIGHT * orig_quality + FLUENCY_WEIGHT * fluency

Resume-safe via SQLite progress tracker.

NOTES:
  • Pseudo-PLL is expensive (one forward pass per token). We sample
    a maximum of 32 tokens per text and use sliding-window masking.
  • Runs on GPU 0 by default; expects the generator's vLLM server to
    be SHUT DOWN before running this (memory clash otherwise).
"""

import json
import logging
import math
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Iterator, List

import numpy as np
import torch
import jsonlines
from transformers import AutoTokenizer, AutoModelForMaskedLM

import config as cfg

logger = logging.getLogger("tier3.fluency")


# ── PROGRESS ─────────────────────────────────────────────────────

def init_db():
    conn = sqlite3.connect(cfg.PROGRESS_DB)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS fluency_progress (
            n_done     INTEGER NOT NULL,
            updated_at TEXT NOT NULL,
            id         INTEGER PRIMARY KEY DEFAULT 1
        )
    """)
    conn.commit()
    return conn


def get_done(conn) -> int:
    row = conn.execute("SELECT n_done FROM fluency_progress WHERE id=1").fetchone()
    return row[0] if row else 0


def set_done(conn, n: int):
    conn.execute("""
        INSERT INTO fluency_progress(id, n_done, updated_at) VALUES(1, ?, ?)
        ON CONFLICT(id) DO UPDATE SET n_done=excluded.n_done, updated_at=excluded.updated_at
    """, (n, datetime.utcnow().isoformat() + "Z"))
    conn.commit()


# ── PSEUDO LOG-LIKELIHOOD ────────────────────────────────────────

class FluencyScorer:
    def __init__(self, model_id: str, device: str = "cuda:0",
                 max_mask_samples: int = 32, max_seq_len: int = 256):
        self.device = device
        self.max_mask_samples = max_mask_samples
        self.max_seq_len = max_seq_len

        logger.info("Loading fluency model: %s", model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForMaskedLM.from_pretrained(
            model_id, torch_dtype=torch.float16,
        ).to(device).eval()
        self.mask_id = self.tokenizer.mask_token_id

    @torch.inference_mode()
    def score_batch(self, texts: List[str]) -> List[float]:
        """Returns list of pseudo-log-likelihoods (mean over masked tokens)."""
        results = []
        for text in texts:
            results.append(self._score_one(text))
        return results

    @torch.inference_mode()
    def _score_one(self, text: str) -> float:
        if not text.strip():
            return float("-inf")

        enc = self.tokenizer(
            text, return_tensors="pt", truncation=True,
            max_length=self.max_seq_len, add_special_tokens=True,
        )
        input_ids = enc["input_ids"].to(self.device)
        attn      = enc["attention_mask"].to(self.device)
        seq_len   = input_ids.size(1)

        # Pick token positions to mask (skip special tokens at edges)
        candidate_positions = list(range(1, seq_len - 1))
        if len(candidate_positions) == 0:
            return float("-inf")

        if len(candidate_positions) > self.max_mask_samples:
            rng = np.random.default_rng(seed=hash(text) & 0xFFFFFFFF)
            candidate_positions = rng.choice(
                candidate_positions, size=self.max_mask_samples, replace=False,
            ).tolist()

        # Build batch: one row per masked position
        batch_size = len(candidate_positions)
        batch_ids = input_ids.repeat(batch_size, 1).clone()
        batch_attn = attn.repeat(batch_size, 1)
        true_ids = []
        for i, pos in enumerate(candidate_positions):
            true_ids.append(batch_ids[i, pos].item())
            batch_ids[i, pos] = self.mask_id

        # Forward
        out = self.model(input_ids=batch_ids, attention_mask=batch_attn)
        logits = out.logits  # (batch, seq_len, vocab)

        log_probs = []
        for i, pos in enumerate(candidate_positions):
            lp = torch.log_softmax(logits[i, pos], dim=-1)
            log_probs.append(lp[true_ids[i]].item())

        return float(np.mean(log_probs))


# ── NORMALIZATION ────────────────────────────────────────────────

def normalize_pll_to_score(pll_values: np.ndarray) -> np.ndarray:
    """
    Convert raw PLL (negative numbers; closer to 0 = better) to [0, 1].
    Uses robust scaling via percentiles to avoid outliers blowing the range.
    """
    finite = pll_values[np.isfinite(pll_values)]
    if len(finite) == 0:
        return np.zeros_like(pll_values)

    p5  = np.percentile(finite, 5)
    p95 = np.percentile(finite, 95)
    if p95 - p5 < 1e-6:
        return np.full_like(pll_values, 0.5)

    scaled = (pll_values - p5) / (p95 - p5)
    return np.clip(scaled, 0.0, 1.0)


# ── MAIN ─────────────────────────────────────────────────────────

def run_fluency_scoring():
    cfg.ensure_dirs()

    if not cfg.TOPICS_OUTPUT.exists():
        raise FileNotFoundError(
            f"Run topic_balancer.py first — missing {cfg.TOPICS_OUTPUT}"
        )

    conn = init_db()
    done = get_done(conn)
    logger.info("Resuming from offset %d", done)

    scorer = FluencyScorer(cfg.FLUENCY_MODEL_ID, device="cuda:0")

    # Two passes: pass 1 collect raw PLLs (resumable), pass 2 normalize.
    # We write to a temp file in pass 1, then rewrite final with normalized scores.

    tmp_path = cfg.FLUENCY_OUTPUT.with_suffix(".tmp.jsonl")
    pll_log_path = cfg.TIER3_DIR / "fluency_pll_raw.npy"

    # --- Pass 1: compute raw PLLs ---
    if not pll_log_path.exists() or done < _count_lines(cfg.TOPICS_OUTPUT):
        pll_values = []
        records_buffer = []
        n = 0

        with jsonlines.open(cfg.TOPICS_OUTPUT) as reader, \
             jsonlines.open(tmp_path, mode="a") as writer:

            for _ in range(done):
                try:
                    next(reader.iter())
                except StopIteration:
                    break

            for batch in _batched(reader, cfg.FLUENCY_BATCH_SIZE):
                texts = [r.get("text") or r.get("output") or r.get("content") or ""
                         for r in batch]
                plls = scorer.score_batch(texts)
                for r, pll in zip(batch, plls):
                    r["fluency_pll"] = pll
                    writer.write(r)
                    pll_values.append(pll)
                n += len(batch)
                done += len(batch)

                if done % cfg.CHECKPOINT_EVERY < cfg.FLUENCY_BATCH_SIZE:
                    set_done(conn, done)
                    np.save(pll_log_path, np.array(pll_values))
                    logger.info("Pass 1: %d scored", done)

        set_done(conn, done)
        np.save(pll_log_path, np.array(pll_values))
        logger.info("Pass 1 complete: %d records scored", done)

    # Free GPU memory before pass 2
    del scorer
    torch.cuda.empty_cache()

    # --- Pass 2: normalize and combine ---
    logger.info("Pass 2: normalizing fluency scores...")
    pll_array = np.load(pll_log_path)
    fluency_norm = normalize_pll_to_score(pll_array)

    with jsonlines.open(tmp_path) as reader, \
         jsonlines.open(cfg.FLUENCY_OUTPUT, mode="w") as writer:
        for i, r in enumerate(reader):
            r["fluency_score"] = float(fluency_norm[i]) if i < len(fluency_norm) else 0.5
            orig_q = r.get("quality_score", 0.5)
            r["combined_quality"] = (
                cfg.HEURISTIC_WEIGHT * orig_q +
                cfg.FLUENCY_WEIGHT * r["fluency_score"]
            )
            writer.write(r)

    tmp_path.unlink(missing_ok=True)
    logger.info("Fluency scoring complete: %s", cfg.FLUENCY_OUTPUT)


# ── HELPERS ──────────────────────────────────────────────────────

def _batched(it: Iterator, size: int):
    batch = []
    for x in it:
        batch.append(x)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def _count_lines(path: Path) -> int:
    n = 0
    with open(path, "rb") as f:
        for _ in f:
            n += 1
    return n


if __name__ == "__main__":
    logging.basicConfig(
        level=cfg.LOG_LEVEL,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    run_fluency_scoring()
