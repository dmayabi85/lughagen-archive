# Review Guide — LughaGen Research Programme

For reviewers of the `a100-training-setup` branch. This guide identifies
the highest-signal files for evaluating the LughaGen research contribution.

## Start here (5 min orientation)

1. **[README.md](README.md)** — Project overview and research-phase mapping
2. **[docs/LughaGen_Training_Framework_Spec_v1.5.1.md](docs/LughaGen_Training_Framework_Spec_v1.5.1.md)** — Framework design specification
3. **[docs/amendments.md](docs/amendments.md)** — Amendments applied during training (embedding remap, remap dir config, Gemma SDPA routing, checkpoint pre-save prune)
4. **[docs/crash_postmortems.md](docs/crash_postmortems.md)** — Production incidents and fixes

## Headline results

- **[results/phase3_definitive_pph_matrix.md](results/phase3_definitive_pph_matrix.md)** — Phase 3 method comparison (FFT / CPT / QLoRA)
- **[results/baseline_bpc/](results/baseline_bpc/)** — Cross-tokenizer BPC vs 6 external African-language baselines
- **[results/corpus/audit_comprehensive/](results/corpus/audit_comprehensive/)** — 100% corpus quality audit (comprehensive scan of 1.38M records)

## Research-phase code map

| Phase | Purpose | Key files |
|---|---|---|
| Phase 0 | Tokenizer trials | `src/lughagen/tokenizer/`, `scripts/phase0_bootstrap/phase0_runner.py`, `configs/registry/phase0_winner.yaml` |
| Phase 1 | Three-tier corpus construction | `src/lughagen/data/`, `scripts/phase1_corpus/`, `results/corpus/` |
| Phase 2 | Factorial hyperparameter sweep | `scripts/phase2_factorial/factorial_matrix.py`, `configs/registry/factorial_design.csv` |
| Phase 3 | Method comparison (FFT/CPT/QLoRA) | `scripts/phase3_seeds/phase3_fft_seed1.sh`, `results/phase3_per_language/` |
| Gap 3 | Cross-tokenizer BPC vs external baselines | `src/lughagen/evaluation/eval_baseline_bpc.py`, `results/baseline_bpc/` |
| Phase 3.5 | Scale-x-specialization on Gemma 2 27B | `scripts/phase3_5_scale/`, `configs/deepspeed/ds_config_gemma27b.json` |

## Framework specification

Training used a hyperparameter-first framework specified in
`docs/LughaGen_Training_Framework_Spec_v1.5.1.md`. Amendments applied during
execution are documented separately in `docs/amendments.md`:

- **Amendment 8**: Token-aware embedding remap
- **Amendment 10**: Configurable remap directory for model-family swap
- **Amendment 11**: Softcap disable + SDPA routing for Gemma-family
- **Amendment 12**: Pre-save checkpoint prune for disk-race prevention

## Contact

Repository maintained by Derek Mayabi (@dmayabi85). Research supervised by
Dr. Lawrence Nderu (JKUAT).
