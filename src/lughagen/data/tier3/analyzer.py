"""
analyzer.py — LughaGen Tier 3 Distribution Analyzer
Reads all Tier 1 (Kikuyu/Kamba/Luo) and Tier 2 (Swahili) JSONL files
and produces a comprehensive diagnostic report:

  • Per-language sample counts
  • Length distribution (chars, tokens)
  • Lexical diversity — Type-Token Ratio (TTR), MTLD
  • Quality score histogram (heuristic from generation pipeline)
  • Semantic clustering (sentence-embedding KMeans)
  • Approximate topic distribution (anchor-based cosine)
  • Suspected near-duplicates (semantic)

Outputs:
  • analysis_report.json   — machine-readable diagnostics
  • plots/*.png            — matplotlib visualizations
  • Console summary table

This module reads only — it never modifies the source JSONL.
"""

import json
import logging
import math
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import jsonlines

import config as cfg

# Lazy imports for heavy deps — only needed when run as main
def _lazy_imports():
    global SentenceTransformer, MiniBatchKMeans, plt
    from embed_local import SentenceTransformer
    from sklearn.cluster import MiniBatchKMeans
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt


logger = logging.getLogger("tier3.analyzer")


# ── CORE METRICS ─────────────────────────────────────────────────

def lexical_ttr(tokens: List[str]) -> float:
    """Type-Token Ratio. Sensitive to length but cheap."""
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)


def lexical_mtld(tokens: List[str], threshold: float = 0.72) -> float:
    """
    Measure of Textual Lexical Diversity (McCarthy & Jarvis 2010).
    Length-independent. Higher = more diverse.
    """
    if len(tokens) < 50:
        return 0.0

    def _factor_count(seq):
        types, tokens_seen, factors, partial = set(), 0, 0, 0.0
        for tok in seq:
            tokens_seen += 1
            types.add(tok)
            ttr = len(types) / tokens_seen
            if ttr <= threshold:
                factors += 1
                types, tokens_seen = set(), 0
        if tokens_seen > 0:
            partial = (1.0 - ttr) / (1.0 - threshold)
        return factors + partial

    forward  = _factor_count(tokens)
    backward = _factor_count(list(reversed(tokens)))
    if forward == 0 or backward == 0:
        return 0.0
    return len(tokens) / ((forward + backward) / 2)


# ── DATA LOADING ─────────────────────────────────────────────────

def load_all_samples() -> Dict[str, List[dict]]:
    """Load every Tier 1 + Tier 2 sample, grouped by language."""
    samples = {}

    # Tier 2 — Swahili
    if cfg.TIER2_INPUT.exists():
        with jsonlines.open(cfg.TIER2_INPUT) as r:
            samples["swahili"] = list(r)
        logger.info("Loaded %d Swahili samples (Tier 2)", len(samples["swahili"]))
    else:
        logger.warning("Tier 2 input missing: %s", cfg.TIER2_INPUT)

    # Tier 1 — Kikuyu, Kamba, Luo
    for lang, path in cfg.TIER1_INPUTS.items():
        if path.exists():
            with jsonlines.open(path) as r:
                samples[lang] = list(r)
            logger.info("Loaded %d %s samples (Tier 1)", len(samples[lang]), lang)
        else:
            logger.warning("Tier 1 %s input missing: %s", lang, path)

    return samples


# ── PER-LANGUAGE STATS ───────────────────────────────────────────

def language_stats(records: List[dict]) -> dict:
    if not records:
        return {"count": 0}

    texts   = [r.get("text") or r.get("output") or r.get("content") or "" for r in records]
    quality = [r.get("quality_score", r.get("quality", 0.0)) for r in records]

    char_lens  = [len(t) for t in texts]
    token_lens = [len(t.split()) for t in texts]

    # Lexical diversity on a sample (full corpus is too slow for MTLD)
    sample_tokens = []
    for t in texts[:5000]:
        sample_tokens.extend(t.split())

    return {
        "count":           len(records),
        "char_len_mean":   float(np.mean(char_lens)),
        "char_len_p50":    float(np.percentile(char_lens, 50)),
        "char_len_p95":    float(np.percentile(char_lens, 95)),
        "token_len_mean":  float(np.mean(token_lens)),
        "token_len_p50":   float(np.percentile(token_lens, 50)),
        "token_len_p95":   float(np.percentile(token_lens, 95)),
        "quality_mean":    float(np.mean(quality)) if quality else 0.0,
        "quality_p50":     float(np.percentile(quality, 50)) if quality else 0.0,
        "quality_below_0_70_pct": float(sum(1 for q in quality if q < 0.70) / len(quality) * 100) if quality else 0.0,
        "ttr_sampled":     lexical_ttr(sample_tokens[:50_000]),
        "mtld_sampled":    lexical_mtld(sample_tokens[:50_000]),
    }


# ── SEMANTIC CLUSTERING ──────────────────────────────────────────

def semantic_clusters(texts: List[str], embed_model, k: int = 50) -> dict:
    """
    Cluster texts in embedding space and return distribution stats.
    Cluster imbalance flags potential redundancy.
    """
    if len(texts) < k * 2:
        k = max(2, len(texts) // 2)

    # Process ALL texts (full audit mode)
    sample_size = len(texts)
    sample = texts
    logger.info("Embedding all %d texts for clustering (full audit mode)...", sample_size)
    embeddings = embed_model.encode(
        sample, batch_size=cfg.EMBED_BATCH_SIZE,
        show_progress_bar=False, convert_to_numpy=True,
        normalize_embeddings=True,
    )

    logger.info("Running MiniBatchKMeans (k=%d) on %d vectors...", k, embeddings.shape[0])
    km = MiniBatchKMeans(n_clusters=k, n_init=10, random_state=42, batch_size=4096)
    labels = km.fit_predict(embeddings)

    cluster_sizes = Counter(labels.tolist())
    sizes = sorted(cluster_sizes.values(), reverse=True)

    return {
        "n_clusters":          k,
        "sample_size":         sample_size,
        "largest_cluster_pct": sizes[0] / sample_size * 100,
        "top5_clusters_pct":   sum(sizes[:5]) / sample_size * 100,
        "min_cluster_size":    min(sizes),
        "max_cluster_size":    max(sizes),
        "size_std":            float(np.std(sizes)),
    }


# ── APPROXIMATE TOPIC DISTRIBUTION ───────────────────────────────

def topic_distribution(texts: List[str], embed_model, anchor_embeddings: dict) -> dict:
    """
    Tag each text with the nearest topic anchor; return distribution.
    Lightweight — uses the same embedding model as clustering.
    """
    # Process ALL texts (full audit mode)
    sample_size = len(texts)
    sample = texts
    logger.info("Embedding all %d texts for topic distribution...", sample_size)

    embeddings = embed_model.encode(
        sample, batch_size=cfg.EMBED_BATCH_SIZE,
        show_progress_bar=False, convert_to_numpy=True,
        normalize_embeddings=True,
    )

    topic_names = list(anchor_embeddings.keys())
    anchor_matrix = np.vstack([anchor_embeddings[t] for t in topic_names])  # (n_topics, dim)

    # Cosine similarity (embeddings are already normalized)
    sim = embeddings @ anchor_matrix.T  # (n_samples, n_topics)
    assigned = sim.argmax(axis=1)

    counts = Counter([topic_names[a] for a in assigned])
    return {t: counts.get(t, 0) / sample_size for t in topic_names}


def build_anchor_embeddings(embed_model) -> dict:
    """Embed each topic's anchor sentences; average to get a single vector."""
    anchor_emb = {}
    for topic, sentences in cfg.TOPIC_ANCHORS.items():
        emb = embed_model.encode(
            sentences, convert_to_numpy=True, normalize_embeddings=True,
        )
        avg = emb.mean(axis=0)
        avg = avg / np.linalg.norm(avg)  # re-normalize
        anchor_emb[topic] = avg
    return anchor_emb


# ── PLOTTING ─────────────────────────────────────────────────────

def make_plots(report: dict, plots_dir: Path):
    plots_dir.mkdir(parents=True, exist_ok=True)

    # 1. Sample count per language
    fig, ax = plt.subplots(figsize=(8, 4))
    langs = list(report["per_language"].keys())
    counts = [report["per_language"][l].get("count", 0) for l in langs]
    ax.bar(langs, counts, color="#534AB7")
    ax.set_title("Sample count per language")
    ax.set_ylabel("Samples")
    fig.tight_layout()
    fig.savefig(plots_dir / "sample_counts.png", dpi=120)
    plt.close(fig)

    # 2. Quality score distribution per language (mean)
    fig, ax = plt.subplots(figsize=(8, 4))
    qualities = [report["per_language"][l].get("quality_mean", 0) for l in langs]
    ax.bar(langs, qualities, color="#1D9E75")
    ax.axhline(0.65, color="red", linestyle="--", label="Quality gate (0.65)")
    ax.set_title("Mean quality score per language")
    ax.set_ylabel("Quality (0–1)")
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "quality_per_language.png", dpi=120)
    plt.close(fig)

    # 3. Topic distribution
    if "topic_distribution" in report:
        fig, ax = plt.subplots(figsize=(10, 5))
        for lang, dist in report["topic_distribution"].items():
            topics = list(dist.keys())
            vals   = [dist[t] * 100 for t in topics]
            ax.plot(topics, vals, marker="o", label=lang)
        ax.axhline(cfg.MAX_TOPIC_RATIO * 100, color="red", linestyle="--",
                   label=f"Max ratio cap ({cfg.MAX_TOPIC_RATIO*100:.0f}%)")
        ax.set_title("Approximate topic distribution per language")
        ax.set_ylabel("% of samples")
        ax.legend()
        plt.xticks(rotation=30, ha="right")
        fig.tight_layout()
        fig.savefig(plots_dir / "topic_distribution.png", dpi=120)
        plt.close(fig)

    logger.info("Plots saved to %s", plots_dir)


# ── MAIN ─────────────────────────────────────────────────────────

def run_analysis():
    cfg.ensure_dirs()
    _lazy_imports()

    samples_by_lang = load_all_samples()
    if not samples_by_lang:
        logger.error("No input data found. Aborting.")
        return

    logger.info("Loading embedding model: %s", cfg.EMBED_MODEL_ID)
    embed = SentenceTransformer(cfg.EMBED_MODEL_ID)

    # Anchor embeddings (Swahili-anchored, but multilingual model embeds
    # related concepts in nearby space — usable as a rough proxy for all langs)
    logger.info("Building topic anchor embeddings...")
    anchor_emb = build_anchor_embeddings(embed)

    report = {
        "tier3_version": cfg.TIER3_VERSION,
        "generated_at":  datetime.utcnow().isoformat() + "Z",
        "per_language":  {},
        "semantic_clusters": {},
        "topic_distribution": {},
    }

    for lang, records in samples_by_lang.items():
        logger.info("─── %s ───", lang.upper())
        report["per_language"][lang] = language_stats(records)

        texts = [r.get("text") or r.get("output") or r.get("content") or "" for r in records]
        texts = [t for t in texts if t.strip()]
        if not texts:
            continue

        report["semantic_clusters"][lang]    = semantic_clusters(texts, embed, k=cfg.N_SEMANTIC_CLUSTERS)
        report["topic_distribution"][lang]   = topic_distribution(texts, embed, anchor_emb)

    # Write report
    cfg.ANALYSIS_REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    logger.info("Report written: %s", cfg.ANALYSIS_REPORT)

    make_plots(report, cfg.ANALYSIS_PLOTS)

    # Console summary
    print("\n" + "=" * 70)
    print(f"  TIER 3 ANALYSIS — {cfg.TIER3_VERSION}")
    print("=" * 70)
    for lang, s in report["per_language"].items():
        print(f"\n  {lang.upper():<10}  n={s.get('count', 0):>7,}  "
              f"q̄={s.get('quality_mean', 0):.3f}  "
              f"MTLD={s.get('mtld_sampled', 0):.1f}")
        if lang in report["topic_distribution"]:
            top3 = sorted(report["topic_distribution"][lang].items(),
                          key=lambda x: -x[1])[:3]
            print(f"             top topics: " +
                  ", ".join(f"{k}={v*100:.0f}%" for k, v in top3))
    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    logging.basicConfig(
        level=cfg.LOG_LEVEL,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    run_analysis()
