# Corpus Cleaning Framework — File Inventory (discovered 2026-07-27)

This is the raw inventory. Detailed documentation and thesis methodology
chapter to be written subsequently.

## Tier 1 — Initial synthetic generation (/workspace/scripts/tier1/)

- config.py (3.5 KB)
- prompts.py (8 KB)
- generate.py (25.7 KB) — main generation
- monitor.sh — process monitor
- run_sequential.sh — sequential runner
- safe_generate.sh — wrapper with safety checks
- stop_lang.sh — kill switch per language
- watchdog.sh — supervisor process

## Tier 2 — Refined synthetic generation (/workspace/scripts/tier2/)

- config.py (3.8 KB)
- prompts.py (14.7 KB) — expanded prompt library
- generate.py (32.8 KB)
- run_tier2.sh (7.8 KB) — orchestrator

## Tier 3 — Audit, quarantine, rescue (/workspace/scripts/tier3/)

- README.md (5.3 KB) — HAS EXISTING DOCUMENTATION
- run_tier3.py (3.8 KB) — top-level entry point
- config.py (6.9 KB)
- setup.sh (3.4 KB)

Audit modules:
- audit_categories.py (26.8 KB) + v2 (25.6 KB)
- audit_domains.py (30.1 KB) + v2 (31.7 KB)
- audit_rescued.py (21.0 KB) + v2 (21.1 KB)

Analysis:
- analyzer.py (12.9 KB)
- analyze_quarantine.py (12.1 KB)

Classification / scoring:
- classifier.py (11.6 KB)
- fluency_scorer.py (9.5 KB) — AfroXLMR-based
- embed_local.py (3.3 KB)

Quarantine / rescue:
- tier2_quarantine.py (16 KB)
- rescue_quarantined.py (17.2 KB) + v2 (17.2 KB)
- reshaper.py (11.3 KB)
- topic_balancer.py (7.6 KB)

## Top-level cleaning scripts (/workspace/scripts/)

Language ID:
- clean_tier1_glotlid.py — GlotLID gate application
- corpus_full_glotlid_audit.py — full-corpus GlotLID pass
- afrolid_analysis.py — AfroLID secondary analysis

Later iterations:
- clean_tier2_v2.py, clean_tier2_v3.py
- clean_synthetic_v4.py

Corpus construction:
- normalise_ingest.py — ingestion normalization
- stage_auth_mono_v2.py — staging pipeline
- audit_auth_mono_v2.py — auth mono audit
- corpus_full_audit.py — full corpus audit
- merge_sheng.py — Sheng merge
- separate_synthetic_cs.py, _v2, _v3 — code-switch separation
- detect_paraphrastic_overlap.py — semantic dedup
- build_splits_v2.py — final split creation
- build_filtered_corpus.py — Phase 3 last-mile filter

## Manifests

- /workspace/data/corpus/staged/auth_mono_v2/MANIFEST.json
- /workspace/data/corpus/staged/auth_mono_v1_archive/ARCHIVE.md

## Pipeline order (inferred, needs verification)

1. Tier 1 generation (initial synthetic corpus)
2. Tier 2 generation (refined synthetic)
3. normalise_ingest.py (schema normalization)
4. clean_tier1_glotlid.py (LangID gate)
5. clean_tier2_v2/v3.py (register/quality checks)
6. clean_synthetic_v4.py (synthetic-specific)
7. Tier 3 audit pipeline (via run_tier3.py):
   - fluency_scorer.py (AfroXLMR scoring)
   - classifier.py (categorization)
   - tier2_quarantine.py (quarantine bad records)
   - audit_categories, audit_domains (deep audit)
   - rescue_quarantined.py (re-examine and possibly rescue)
   - topic_balancer.py (enforce topic balance)
8. detect_paraphrastic_overlap.py (semantic dedup)
9. build_splits_v2.py (train/val/test splits)
10. corpus_full_audit.py (final verification)
11. build_filtered_corpus.py (Phase 3 last-mile filter, we wrote this)

## Next: draft thesis methodology chapter based on this inventory
