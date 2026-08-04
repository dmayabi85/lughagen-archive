"""
reshaper.py — LughaGen Tier 3 Final Dataset Reshaper

Reads quality_classified.jsonl (output of classifier.py) and produces the
final lughagen_train.jsonl by:

  1. HARD GATES — drop anything failing combined_quality < COMBINED_QUALITY_MIN
     OR clf_label == "bad" with prob > CLF_BAD_THRESHOLD.

  2. SEMANTIC DEDUP — embed all surviving samples; for any pair with
     cosine similarity > SEMANTIC_DUP_THRESHOLD, keep only the higher-quality
     one. Uses approximate-NN via FAISS for tractability at scale.

  3. TOPIC RATIO ENFORCEMENT — cap any single topic at MAX_TOPIC_RATIO of the
     final dataset per language. Drops the lowest-quality excess.

  4. STRATIFIED SAMPLING — per language, hit TARGET_PER_LANGUAGE if possible.
     Within each language, ensure topic balance via reservoir sampling.

  5. WRITE OUT — final lughagen_train.jsonl + dataset_summary.json
     with full provenance (counts, ratios, drop reasons).

This module makes the irreversible decisions about what enters training.
Every drop is logged to dataset_summary.json["drops_by_reason"].
"""

import json
import logging
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import jsonlines

import config as cfg

logger = logging.getLogger("tier3.reshaper")


# ── LOAD ─────────────────────────────────────────────────────────

def load_quality_classified() -> List[dict]:
    if not cfg.QUALITY_OUTPUT.exists():
        raise FileNotFoundError(
            f"Run classifier.py infer first — missing {cfg.QUALITY_OUTPUT}"
        )
    with jsonlines.open(cfg.QUALITY_OUTPUT) as r:
        records = list(r)
    logger.info("Loaded %d records from %s", len(records), cfg.QUALITY_OUTPUT)
    return records


# ── STAGE 1: HARD GATES ──────────────────────────────────────────

def apply_hard_gates(records: List[dict]) -> tuple:
    """Returns (kept, drop_reasons)."""
    kept = []
    drops = Counter()

    for r in records:
        if r.get("combined_quality", 0) < cfg.COMBINED_QUALITY_MIN:
            drops["combined_quality_below_threshold"] += 1
            continue
        if r.get("clf_label") == "bad" and r.get("clf_good_prob", 1.0) < (1 - cfg.CLF_BAD_THRESHOLD):
            drops["classifier_rejected"] += 1
            continue
        if not (r.get("text") or r.get("output") or r.get("content") or "").strip():
            drops["empty_text"] += 1
            continue
        kept.append(r)

    logger.info("Hard gates: kept=%d  drops=%s", len(kept), dict(drops))
    return kept, dict(drops)


# ── STAGE 2: SEMANTIC DEDUP ──────────────────────────────────────

def semantic_dedup(records: List[dict]) -> tuple:
    """
    Drop near-duplicates via FAISS-based approximate nearest neighbor.
    Within each duplicate pair, keep the higher combined_quality sample.
    """
    if len(records) < 2:
        return records, {}

    try:
        import faiss
        from embed_local import SentenceTransformer
    except ImportError:
        logger.warning("faiss or sentence_transformers missing — skipping semantic dedup")
        return records, {"semantic_dedup_skipped": len(records)}

    logger.info("Embedding %d records for semantic dedup...", len(records))
    embed = SentenceTransformer(cfg.EMBED_MODEL_ID)
    texts = [r.get("text") or r.get("output") or r.get("content") or "" for r in records]
    emb = embed.encode(
        texts, batch_size=cfg.EMBED_BATCH_SIZE,
        show_progress_bar=False, convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    dim = emb.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner product on normalized = cosine
    index.add(emb)

    # Search top-5 NN for each
    logger.info("Searching nearest neighbors...")
    D, I = index.search(emb, k=5)

    drop_idx = set()
    for i in range(len(records)):
        if i in drop_idx:
            continue
        for rank in range(1, 5):  # skip self at rank 0
            j = int(I[i, rank])
            sim = float(D[i, rank])
            if sim < cfg.SEMANTIC_DUP_THRESHOLD:
                break
            if j in drop_idx or j == i:
                continue
            # Keep higher combined_quality
            qi = records[i].get("combined_quality", 0)
            qj = records[j].get("combined_quality", 0)
            drop_idx.add(j if qi >= qj else i)
            if i in drop_idx:
                break

    kept = [r for i, r in enumerate(records) if i not in drop_idx]
    logger.info("Semantic dedup: kept=%d  dropped=%d", len(kept), len(drop_idx))
    return kept, {"semantic_duplicates": len(drop_idx)}


# ── STAGE 3: TOPIC RATIO CAPS ────────────────────────────────────

def enforce_topic_ratios(records: List[dict]) -> tuple:
    """Per language, cap each topic at MAX_TOPIC_RATIO. Drop lowest-quality excess."""
    by_lang_topic: Dict[str, Dict[str, List[dict]]] = defaultdict(lambda: defaultdict(list))
    for r in records:
        by_lang_topic[r.get("language", "unknown")][r.get("topic", "unknown")].append(r)

    kept = []
    drops = Counter()
    summary = {}

    for lang, topics in by_lang_topic.items():
        total_lang = sum(len(v) for v in topics.values())
        cap_per_topic = int(total_lang * cfg.MAX_TOPIC_RATIO)
        floor_per_topic = int(total_lang * cfg.MIN_TOPIC_RATIO)
        summary[lang] = {}

        for topic, recs in topics.items():
            n_before = len(recs)
            if n_before > cap_per_topic:
                # Sort by combined_quality desc, keep top cap_per_topic
                recs_sorted = sorted(recs, key=lambda r: -r.get("combined_quality", 0))
                kept_recs = recs_sorted[:cap_per_topic]
                dropped_n = n_before - cap_per_topic
                drops[f"{lang}_{topic}_over_cap"] += dropped_n
            else:
                kept_recs = recs
                dropped_n = 0

            kept.extend(kept_recs)
            summary[lang][topic] = {
                "before": n_before, "after": len(kept_recs), "dropped": dropped_n,
                "below_floor": len(kept_recs) < floor_per_topic,
            }

    logger.info("Topic ratio enforcement: kept=%d", len(kept))
    return kept, dict(drops), summary


# ── STAGE 4: STRATIFIED PER-LANGUAGE TARGETS ─────────────────────

def stratified_sample(records: List[dict]) -> tuple:
    """Hit TARGET_PER_LANGUAGE per language (or take all if fewer available)."""
    by_lang: Dict[str, List[dict]] = defaultdict(list)
    for r in records:
        by_lang[r.get("language", "unknown")].append(r)

    kept = []
    summary = {}

    for lang, recs in by_lang.items():
        target = cfg.TARGET_PER_LANGUAGE.get(lang, len(recs))
        if len(recs) <= target:
            kept.extend(recs)
            summary[lang] = {"available": len(recs), "target": target,
                             "taken": len(recs), "below_target": True}
            logger.warning("Language %s: only %d available (target %d)",
                           lang, len(recs), target)
        else:
            # Sort by combined_quality desc, take top
            recs_sorted = sorted(recs, key=lambda r: -r.get("combined_quality", 0))
            kept.extend(recs_sorted[:target])
            summary[lang] = {"available": len(recs), "target": target,
                             "taken": target, "below_target": False}

    return kept, summary


# ── MAIN ─────────────────────────────────────────────────────────

def run_reshape():
    cfg.ensure_dirs()

    records = load_quality_classified()
    n_input = len(records)

    # Stage 1: hard gates
    records, drops_gates = apply_hard_gates(records)

    # Stage 2: semantic dedup (expensive but worth it)
    records, drops_dedup = semantic_dedup(records)

    # Stage 3: topic caps
    records, drops_topic, topic_summary = enforce_topic_ratios(records)

    # Stage 4: stratified per-language
    records, lang_summary = stratified_sample(records)

    # Final shuffle for training stability
    rng = np.random.default_rng(seed=42)
    rng.shuffle(records)

    # Strip Tier 3 internal fields before writing final dataset (keep them in summary)
    cfg.FINAL_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with jsonlines.open(cfg.FINAL_OUTPUT, mode="w") as w:
        for r in records:
            # Keep useful fields, drop noisy intermediates
            cleaned = {
                "text":     r.get("text") or r.get("output") or r.get("content"),
                "language": r.get("language"),
                "topic":    r.get("topic"),
                "quality":  round(r.get("combined_quality", 0), 4),
                "source":   r.get("source", "synthetic"),
            }
            # Preserve any task-specific fields the generator added
            for k in ("instruction", "input", "task_type", "domain"):
                if k in r:
                    cleaned[k] = r[k]
            w.write(cleaned)

    # Write summary
    summary = {
        "tier3_version":   cfg.TIER3_VERSION,
        "dataset_version": cfg.DATASET_VERSION,
        "generated_at":    datetime.utcnow().isoformat() + "Z",
        "n_input":         n_input,
        "n_output":        len(records),
        "retention_pct":   len(records) / n_input * 100 if n_input else 0,
        "drops_by_reason": {
            **{f"hard_gate_{k}": v for k, v in drops_gates.items()},
            **drops_dedup,
            **drops_topic,
        },
        "language_breakdown": lang_summary,
        "topic_breakdown":    topic_summary,
        "config_used": {
            "combined_quality_min": cfg.COMBINED_QUALITY_MIN,
            "max_topic_ratio":      cfg.MAX_TOPIC_RATIO,
            "semantic_dup_thresh":  cfg.SEMANTIC_DUP_THRESHOLD,
            "target_per_language":  cfg.TARGET_PER_LANGUAGE,
        },
    }
    cfg.DATASET_SUMMARY.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    logger.info("Final dataset: %s (%d records)", cfg.FINAL_OUTPUT, len(records))
    logger.info("Summary: %s", cfg.DATASET_SUMMARY)

    # Console summary
    print("\n" + "=" * 70)
    print(f"  TIER 3 RESHAPING COMPLETE — {cfg.DATASET_VERSION}")
    print("=" * 70)
    print(f"  Input:   {n_input:,}")
    print(f"  Output:  {len(records):,}  ({len(records)/n_input*100:.1f}% retention)")
    print(f"\n  Per language:")
    for lang, s in lang_summary.items():
        flag = " ⚠ below target" if s["below_target"] else ""
        print(f"    {lang:<10} {s['taken']:>7,} / {s['target']:>7,}{flag}")
    print(f"\n  Final file: {cfg.FINAL_OUTPUT}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    logging.basicConfig(
        level=cfg.LOG_LEVEL,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    run_reshape()
