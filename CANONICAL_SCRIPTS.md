# Canonical Scripts — What Actually Did What

This file identifies the SINGLE canonical script for each major task in
the LughaGen thesis. The repository accumulated multiple versions during
6 months of research; this document tells you which version to use.

**For a first-time reader**: This is the file to read. Everything else is
either supporting infrastructure or an earlier version we learned from.

---

## Quick answer: What produced our results?

| Task | Canonical Script |
|------|-----------------|
| Build training corpus | `scripts/phase1_corpus/stage_auth_mono_v2.py` |
| Audit corpus quality | `scripts/audit_corpus_comprehensive.py` |
| Data leakage check | `scripts/leakage_audit.py` |
| Train Llama-8B (FFT) | `scripts/phase3_seeds/phase3_fft_seed1.sh` |
| Train Llama-8B (CPT) | `scripts/phase3_seeds/phase3_cpt_seed1.sh` |
| Train Llama-8B (QLoRA) | `scripts/phase3_seeds/phase3_qlora.sh` |
| Train Gemma-27B (CPT) | `scripts/phase3_5_scale/phase3_5_gemma2_cpt.sh` |
| Train Gemma-27B (FFT) | `scripts/phase3_5_scale/phase3_5_gemma2_fft.sh` |
| Evaluate model on 5 languages | `src/lughagen/evaluation/eval_per_language_5tracks.py` |
| Cross-tokenizer BPC comparison | `src/lughagen/evaluation/eval_baseline_bpc.py` |
| Reproduce final results | Follow `scripts/workflow/README.md` for Aug 11-15 workflow |

---

## Detailed script guide

### 1. Corpus construction

**Canonical**: `scripts/phase1_corpus/stage_auth_mono_v2.py`

**What it does**: Aggregates the authentic monolingual corpus from
15 source datasets across 5 languages (Swahili, Kikuyu, Luo, Kamba,
Sheng). Produces the "pathD_v2_filtered" corpus at
`/workspace/data/corpus/splits_v2/pathD_v2_filtered/`.

**Result**: 1,380,264 records after filtering.

**Do NOT use**:
- Anything with `_v1` suffix (earlier corpus version, superseded)
- `pathA/`, `pathB/`, `pathC/` variants (experimental alternatives)

### 2. Corpus quality audit

**Canonical**: `scripts/audit_corpus_comprehensive.py`

**What it does**: 100% coverage audit of all 1.38M records for
contamination, generator drift, scraping artifacts, encoding issues.

**Result**: 6 genuine issues identified across 1.38M records (0.0004%).
Output in `results/corpus/audit_comprehensive/`.

**Do NOT use**:
- `scripts/audit_corpus_quality.py` — earlier partial-coverage version
- `scripts/phase1_corpus/audit_dataset.py` — per-source spot check only
- `scripts/phase1_corpus/audit_dataset_tier2.py` — per-tier check only

Why we kept the earlier ones: each addressed a specific audit dimension
during corpus construction. The comprehensive version supersedes them
for the final quality claim.

### 3. Data cleaning

**Canonical stack** (executed in order):
1. `src/lughagen/data/cleaning/clean_tier1_glotlid.py` — Language ID filtering
2. `src/lughagen/data/cleaning/clean_tier2_v3.py` — Content quality filtering
3. `src/lughagen/data/cleaning/clean_synthetic_v4.py` — Synthetic data filtering
4. `src/lughagen/data/cleaning/reclean_v3.py` — Final consolidated re-cleaning

**Do NOT use**:
- `clean_datasets.py` — older monolithic version replaced by the tier stack
- `corpus_full_audit.py`, `corpus_full_glotlid_audit.py` — audit versions,
  not cleaning versions

The tier-based approach was adopted because different data sources
required different cleaning intensity.

### 4. Data leakage check

**Canonical**: `scripts/leakage_audit.py`

**What it does**: Verifies training and validation sets have no
overlapping records. Handles exact match, normalized match, and near-
duplicate detection.

**Result**: 24 duplicates found and removed from validation set.
Clean validation now has 5,952 records (from 5,976 original).

### 5. Model training

**Canonical scripts by adaptation method**:

Llama 3.1 8B family:
- FFT (Full Fine-Tuning): `scripts/phase3_seeds/phase3_fft_seed1.sh`
- CPT (Continued Pre-Training): `scripts/phase3_seeds/phase3_cpt_seed1.sh`
- QLoRA (Parameter-Efficient): `scripts/phase3_seeds/phase3_qlora.sh`

Gemma 2 27B family:
- CPT: `scripts/phase3_5_scale/phase3_5_gemma2_cpt.sh`
- FFT: `scripts/phase3_5_scale/phase3_5_gemma2_fft.sh`

**Timing**:
- Llama 8B methods: 17-22 hours on 8xA100 80GB
- Gemma 27B methods: 3-4 days on 8xA100 80GB

**Do NOT use**:
- Anything in `scripts/phase0_bootstrap/` for final training — those
  were smoke tests and infrastructure validation
- Anything in `scripts/phase2_factorial/` for final training — those
  ran the 2^4 factorial hyperparameter sweep to CHOOSE the config
  now used in phase3

**Multi-seed replication**: `scripts/phase3_seeds/phase3_fft_seed2.sh`,
`phase3_fft_seed3.sh` (same config, different random seeds for
variance quantification).

### 6. Evaluation

**Canonical for per-language BPC**:
`src/lughagen/evaluation/eval_per_language_5tracks.py`

**What it does**: Loads a trained model, evaluates on the 5,952-record
clean validation set, reports per-language and pooled bits-per-character
(BPC).

**Usage**:
    python src/lughagen/evaluation/eval_per_language_5tracks.py \
        --checkpoint /path/to/model \
        --val-jsonl /path/to/validation.jsonl \
        --output results.json

**Canonical for cross-tokenizer comparisons**:
`src/lughagen/evaluation/eval_baseline_bpc.py`

**When to use which**:
- `eval_per_language_5tracks.py`: LughaGen models on our tokenizer
- `eval_baseline_bpc.py`: When comparing models with DIFFERENT tokenizers
  (e.g., LughaGen vs AfriqueGemma with its native tokenizer). Bits-per-
  character is tokenizer-independent, allowing fair comparison.

**Do NOT use**:
- `scripts/evaluate.py` — earlier version, only handles single language
- `scripts/evaluation/smoke_cpt.py` — smoke test, uses 150-step models

### 7. Aug 11-15 workflow (thesis submission)

**Canonical location**: `scripts/workflow/`

See `scripts/workflow/README.md` for the complete 8-step sequence
that runs after Gemma-FFT training completes:

1. `aug11_step1_verify_completion.sh` — Verify training finished cleanly
2. `sanity_check_eval_pipeline.sh` — Verify eval pipeline reproducibility
3. `aug11_step3_gemma_fft_eval.sh` — Per-language BPC on Gemma-FFT
4. `aug11_step4_baseline_bpc.sh` — Cross-tokenizer BPC on Gemma-FFT
5. `aug11_step5_bootstrap_cis.sh` — Bootstrap 95 percent CIs on all 5 models
6. `aug11_step6_per_language_table.py` — Comparison table across models
7. `aug11_step7_best_checkpoint_eval.sh` — Best-checkpoint eval (3 Llama models)
8. `aug11_step8_gemma_fft_step4000_eval.sh` — Efficiency measurement

---

## What's in the repo but NOT canonical

The following exist for research process transparency but are NOT the
scripts that produced final thesis results:

- **Phase 0 bootstrap** (`scripts/phase0_bootstrap/`): Smoke tests,
  micro-benchmarks, memory accounting. Used to size the training run.
- **Phase 1 audit variants** (`scripts/phase1_corpus/audit_*`): Different
  audit passes at different stages of corpus construction.
- **Phase 2 factorial** (`scripts/phase2_factorial/`): 2^4 hyperparameter
  sweep to select the config used in Phase 3.
- **Automation scripts** (`scripts/automation/`): Watchdogs, notifiers,
  and dashboards for long training runs.
- **Older evaluation** (`scripts/evaluate.py`): Superseded by
  `eval_per_language_5tracks.py`.
- **Cleaning intermediates** (`clean_datasets.py`, etc.): Superseded by
  the tier-based cleaning stack.

These are preserved for:
1. Traceability of research decisions
2. Reproducing intermediate findings
3. Educational value (showing what was tried)

But if you want to REPRODUCE THE FINAL RESULTS, use only the scripts
identified as "Canonical" above.

---

## Summary

**For a layperson visitor**: Read this file, then check `README.md`.

**For a reviewer**: Read `REVIEW_GUIDE.md`, then use canonical scripts
in this file as the "ground truth" of what was run.

**For a researcher reproducing results**: Follow the canonical scripts
above in the order:
1. Corpus construction (Section 1)
2. Cleaning stack (Section 3)
3. Audit + leakage (Sections 2 and 4)
4. Training (Section 5)
5. Evaluation (Section 6)
6. Aug 11 workflow if reproducing thesis-specific analyses (Section 7)
