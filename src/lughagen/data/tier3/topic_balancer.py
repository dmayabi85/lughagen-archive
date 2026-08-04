"""
topic_balancer.py — LughaGen Tier 3 Topic Tagging & Distribution Control

Tags every sample with one of TOPIC_ANCHORS (config.py) using cosine
similarity to anchor embeddings. Output: topics_classified.jsonl with
a `topic` and `topic_score` field added per sample.

The actual ratio enforcement happens in reshaper.py — this module
just produces tags + distribution stats.

Approach:
  1. Embed all anchor sentences once (avg per topic).
  2. Stream JSONL inputs, embed in batches of EMBED_BATCH_SIZE.
  3. Argmax cosine vs anchor matrix → assigned topic.
  4. Write augmented record to TOPICS_OUTPUT.
  5. Resume-safe via SQLite progress tracker.
"""

import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Iterator, List, Tuple

import numpy as np
import jsonlines
from embed_local import SentenceTransformer

import config as cfg

logger = logging.getLogger("tier3.topic_balancer")


# ── PROGRESS TRACKING ────────────────────────────────────────────

def init_progress_db():
    cfg.PROGRESS_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(cfg.PROGRESS_DB)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS topic_progress (
            source     TEXT PRIMARY KEY,
            n_done     INTEGER NOT NULL DEFAULT 0,
            updated_at TEXT NOT NULL
        )
    """)
    conn.commit()
    return conn


def get_progress(conn, source: str) -> int:
    row = conn.execute("SELECT n_done FROM topic_progress WHERE source=?",
                       (source,)).fetchone()
    return row[0] if row else 0


def set_progress(conn, source: str, n: int):
    conn.execute("""
        INSERT INTO topic_progress(source, n_done, updated_at)
        VALUES(?,?,?)
        ON CONFLICT(source) DO UPDATE SET n_done=excluded.n_done, updated_at=excluded.updated_at
    """, (source, n, datetime.utcnow().isoformat() + "Z"))
    conn.commit()


# ── ANCHOR EMBEDDINGS ────────────────────────────────────────────

def build_anchor_matrix(embed_model) -> Tuple[List[str], np.ndarray]:
    """Returns (topic_names, anchor_matrix [n_topics x dim])."""
    topic_names = list(cfg.TOPIC_ANCHORS.keys())
    rows = []
    for topic in topic_names:
        emb = embed_model.encode(
            cfg.TOPIC_ANCHORS[topic],
            convert_to_numpy=True, normalize_embeddings=True,
        )
        avg = emb.mean(axis=0)
        avg = avg / np.linalg.norm(avg)
        rows.append(avg)
    return topic_names, np.vstack(rows)


# ── BATCH ITERATOR ───────────────────────────────────────────────

def batch_records(records: Iterator[dict], size: int) -> Iterator[List[dict]]:
    batch = []
    for r in records:
        batch.append(r)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


# ── MAIN TAGGING ─────────────────────────────────────────────────

def tag_source(input_path: Path, language: str, embed_model,
               topic_names: List[str], anchor_matrix: np.ndarray,
               output_writer: jsonlines.Writer, conn) -> dict:
    """Tag all records in one input JSONL. Returns topic counts."""

    src_key = f"{language}::{input_path.name}"
    n_done = get_progress(conn, src_key)
    logger.info("Resuming %s from offset %d", src_key, n_done)

    counts = {t: 0 for t in topic_names}
    n_processed = 0

    with jsonlines.open(input_path) as reader:
        # Skip already-processed
        for _ in range(n_done):
            try:
                next(reader.iter())
            except StopIteration:
                logger.warning("Source %s shorter than checkpoint; resetting", src_key)
                n_done = 0
                break

        for batch in batch_records(reader, cfg.EMBED_BATCH_SIZE):
            texts = [r.get("text") or r.get("output") or r.get("content") or ""
                     for r in batch]
            non_empty = [(i, t) for i, t in enumerate(texts) if t.strip()]
            if not non_empty:
                continue

            indices, valid_texts = zip(*non_empty)
            emb = embed_model.encode(
                list(valid_texts), batch_size=cfg.EMBED_BATCH_SIZE,
                show_progress_bar=False, convert_to_numpy=True,
                normalize_embeddings=True,
            )
            sim = emb @ anchor_matrix.T  # (n, n_topics)
            assigned = sim.argmax(axis=1)
            scores   = sim.max(axis=1)

            # Re-attach to original batch order; empty texts get "unknown"
            for batch_idx, rec in enumerate(batch):
                if batch_idx in indices:
                    pos = indices.index(batch_idx)
                    topic = topic_names[assigned[pos]]
                    score = float(scores[pos])
                else:
                    topic, score = "unknown", 0.0

                rec["topic"]       = topic
                rec["topic_score"] = score
                rec["language"]    = language
                output_writer.write(rec)
                counts[topic] = counts.get(topic, 0) + 1

            n_processed += len(batch)
            n_done      += len(batch)

            if n_done % cfg.CHECKPOINT_EVERY < cfg.EMBED_BATCH_SIZE:
                set_progress(conn, src_key, n_done)
                logger.info("  %s: %d tagged", language, n_done)

    set_progress(conn, src_key, n_done)
    logger.info("Done %s: %d records tagged", src_key, n_processed)
    return counts


def run_topic_tagging():
    cfg.ensure_dirs()
    conn = init_progress_db()

    logger.info("Loading embedding model: %s", cfg.EMBED_MODEL_ID)
    embed = SentenceTransformer(cfg.EMBED_MODEL_ID)

    topic_names, anchor_matrix = build_anchor_matrix(embed)
    logger.info("Built %d topic anchors", len(topic_names))

    cfg.TOPICS_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    # Append-mode writer for resume-safety
    out = jsonlines.open(cfg.TOPICS_OUTPUT, mode="a")

    all_counts = {}

    # Tier 2 — Swahili
    if cfg.TIER2_INPUT.exists():
        all_counts["swahili"] = tag_source(
            cfg.TIER2_INPUT, "swahili", embed,
            topic_names, anchor_matrix, out, conn,
        )

    # Tier 1 — Kikuyu, Kamba, Luo
    for lang, path in cfg.TIER1_INPUTS.items():
        if path.exists():
            all_counts[lang] = tag_source(
                path, lang, embed, topic_names, anchor_matrix, out, conn,
            )

    out.close()

    # Summary
    summary = {
        "tier3_version": cfg.TIER3_VERSION,
        "generated_at":  datetime.utcnow().isoformat() + "Z",
        "topic_counts":  all_counts,
        "topic_ratios":  {
            lang: {t: c / sum(counts.values()) if sum(counts.values()) else 0
                   for t, c in counts.items()}
            for lang, counts in all_counts.items()
        },
    }
    summary_path = cfg.TIER3_DIR / "topic_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    logger.info("Topic summary: %s", summary_path)


if __name__ == "__main__":
    logging.basicConfig(
        level=cfg.LOG_LEVEL,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    run_topic_tagging()
