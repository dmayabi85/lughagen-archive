# LughaGen Tier 3 — Dataset Intelligence Layer

> **Purpose:** Take the completed Tier 1 (Kikuyu/Kamba/Luo) and Tier 2 (Swahili) synthetic outputs and produce a balanced, diverse, deduplicated, distribution-controlled `lughagen_train.jsonl` ready for continued pretraining.

## Pipeline

```
Tier 1 (filtered) ──┐
                    ├──► analyzer ──► topic_balancer ──► fluency_scorer ──► classifier ──► reshaper ──► lughagen_train.jsonl
Tier 2 (filtered) ──┘
```

| Stage | Script | What it does | Approx time (1×A100) |
|---|---|---|---|
| 1 | `analyzer.py` | Distribution diagnostics + plots | ~10 min |
| 2 | `topic_balancer.py` | Tag every sample with topic | ~30 min |
| 3 | `fluency_scorer.py` | External pseudo-PLL (afro-xlmr) | ~3–4 hours |
| 4a | `classifier.py train` | Fine-tune good/bad discriminator | ~30 min |
| 4b | `classifier.py infer` | Apply classifier to all samples | ~20 min |
| 5 | `reshaper.py` | Hard gates · dedup · caps · stratify | ~20 min |

**Total: ~5–6 hours end-to-end.**

## Prerequisites

Tier 2 (Swahili) generation must be complete and quality-filtered. Tier 1 (Kikuyu/Kamba/Luo) must be at least partially done — the pipeline gracefully handles missing languages.

Before running Tier 3, **shut down any running vLLM generation server** to free GPU memory:

```bash
pkill -f "vllm.entrypoints"
nvidia-smi  # confirm GPUs free
```

## Setup

```bash
cd /workspace/scripts/tier3
chmod +x setup.sh
./setup.sh
```

This installs deps, creates dirs, pre-downloads models, and verifies inputs.

## Run

```bash
# Full pipeline
python run_tier3.py

# Or stage by stage (recommended first time, easier to debug)
python run_tier3.py --only analyze
python run_tier3.py --only tag-topics
python run_tier3.py --only score-fluency
python run_tier3.py --only train-clf
python run_tier3.py --only infer-clf
python run_tier3.py --only reshape

# Resume from a specific stage
python run_tier3.py --from score-fluency

# Skip classifier training (model already trained)
python run_tier3.py --skip-train
```

## Outputs

```
/workspace/synthetic/tier3/
├── analysis_report.json          # diagnostics
├── plots/                        # matplotlib figures
│   ├── sample_counts.png
│   ├── quality_per_language.png
│   └── topic_distribution.png
├── topics_classified.jsonl       # + topic, topic_score
├── fluency_scored.jsonl          # + fluency_pll, fluency_score, combined_quality
├── quality_classified.jsonl      # + clf_good_prob, clf_label
└── tier3_progress.db             # resume state

/workspace/data/final_training/
├── lughagen_train.jsonl          # ★ FINAL DATASET ★
└── dataset_summary.json          # provenance & drop reasons
```

## Tuning

All thresholds live in `config.py`:

| Parameter | Default | Purpose |
|---|---|---|
| `COMBINED_QUALITY_MIN` | 0.60 | Hard floor on heuristic+fluency blend |
| `CLF_BAD_THRESHOLD` | 0.70 | Drop if classifier ≥70% sure it's bad |
| `MAX_TOPIC_RATIO` | 0.20 | No single topic >20% per language |
| `SEMANTIC_DUP_THRESHOLD` | 0.92 | Cosine ≥0.92 = near-duplicate |
| `TARGET_PER_LANGUAGE` | 80k/60k/60k/60k | Cap per language in final dataset |

## Recovery

Every stage is resumable. If interrupted, just re-run the same command — progress is tracked in `tier3_progress.db`.

If you want to re-run a stage from scratch:

```bash
sqlite3 /workspace/synthetic/tier3/tier3_progress.db "DELETE FROM topic_progress;"
rm /workspace/synthetic/tier3/topics_classified.jsonl
python run_tier3.py --only tag-topics
```

## Sanity checks after running

```bash
# Final dataset size
wc -l /workspace/data/final_training/lughagen_train.jsonl

# Per-language breakdown
jq -r '.language' /workspace/data/final_training/lughagen_train.jsonl | sort | uniq -c

# Topic balance
jq -r '.topic' /workspace/data/final_training/lughagen_train.jsonl | sort | uniq -c

# Quality distribution
jq -r '.quality' /workspace/data/final_training/lughagen_train.jsonl | \
  awk '{n++; s+=$1; if($1<min||!min)min=$1; if($1>max)max=$1} END {print "n="n" mean="s/n" min="min" max="max}'
```

## Notes & gotchas

- **GPU memory:** fluency_scorer + classifier each load a separate model. Run them sequentially, not in parallel.
- **Topic anchors are Swahili-only** (config.py). The multilingual MiniLM embeds related concepts in nearby space, so this works as a rough proxy for Kikuyu/Kamba/Luo, but topic labels for those languages are noisier than Swahili. Consider adding language-specific anchors before final training if topic balance per low-resource language matters.
- **The classifier's training data depends on Tier 2 v2.0's `quality_history.db`** containing rejected samples. If that DB doesn't exist, the classifier falls back to using only low-combined-quality samples as negatives, which is weaker. Check `REJECTED_DB` path in `classifier.py`.
- **Semantic dedup uses FAISS.** On 200k+ records it needs ~3–4 GB RAM. Should fit comfortably on the Brev container.

## Provenance

Every record in `lughagen_train.jsonl` is traceable: the `dataset_summary.json` records every drop reason, ratio cap, and language target. Re-run the pipeline with the same `config.py` and you get bit-identical output.
