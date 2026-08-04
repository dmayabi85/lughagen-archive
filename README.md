# LughaGen

LughaGen is a collaborative project to develop and evaluate an LLM for low-resourced African languages, with an initial focus on **Swahili, Kikuyu, Luo, and Kamba**.

## Overview

The project is organized into four working areas:

- **Data** — collection, cleaning, normalization, documentation, and dataset governance
- **Model Training** — preprocessing, training, fine-tuning, and checkpoint management
- **Evaluation / Trust** — benchmarking, error analysis, safety, bias, and robustness checks
- **Deployment** — packaging, serving, inference APIs, and release management

Academic faculty and supervisors provide research oversight, review, and guidance.

## Project goals

- Build a strong multilingual foundation for under-resourced African languages
- Create a transparent and reproducible research workflow
- Document datasets, experiments, and model releases clearly
- Support responsible evaluation and deployment

## Repository structure

- `data/` — data assets, metadata, and versioned dataset pointers
- `src/lughagen/` — project code
- `docs/` — project documentation, decisions, meeting notes, data/model cards
- `configs/` — training and inference configuration files
- `notebooks/` — exploratory analysis and prototyping
- `scripts/` — utility scripts for data preparation, training, and evaluation
- `experiments/` — experiment logs and run summaries
- `models/` — model metadata and release references
- `reports/` — generated evaluation outputs and figures
- `tests/` — automated checks

## Working principles

- Use pull requests for all substantive changes
- Keep `main` protected
- Assign code ownership by folder
- Do not commit secrets
- Do not store large raw datasets or large model binaries directly in Git
- Document every dataset and model release

## Documentation

This repository contains:

- a project README
- contribution guidelines
- a code of conduct
- a citation file
- a dataset card
- a model card
- evaluation summaries
- meeting notes and decision records

## Getting started

1. Clone the repository
2. Create a feature branch
3. Install dependencies
4. Prepare data using the scripts in `scripts/`
5. Run training or evaluation from `src/lughagen/`
6. Open a pull request for review

## License

Project license to be added here.

## Citation

Project citation to be added here.

## Contact

Project maintainers: team contacts and GitHub handles.

---

## MSc thesis extension (LughaGen research programme)

This section extends the team README with the specific work of the LughaGen MSc research programme at JKUAT, run on the Brev 8x A100 80GB instance during May-August 2026. It documents the thesis-phase mapping to the code in this repository.

### Research questions

1. **Chapter 3 (data)**: How do you construct a native-language corpus for East African languages (Swahili, Kikuyu, Luo, Kamba, Sheng) that is small, high-quality, and adaptation-ready?
2. **Chapter 4 (design)**: What factorial hyperparameter design isolates the drivers of adaptation success on a Llama 3.1 8B base?
3. **Chapter 5 (method comparison)**: Does full fine-tuning (FFT), continued pre-training (CPT), or QLoRA give the strongest East African LM adaptation at recommended operating points, with n=3 seed reproducibility?
4. **Chapter 6 (scale x specialization)**: When we swap the base model from Llama 3.1 8B to Gemma 2 27B (3.4x parameters), does the specialization advantage compound with scale?

### Thesis-phase mapping

| Phase | Purpose | Thesis chapter | Key artifacts |
|---|---|---|---|
| Phase 0 | Bootstrap: tokenizer fertility trials + pilot training | Chapter 3 | `configs/registry/phase0_winner.yaml` (strategy_c 8K BPE selected), `scripts/phase0_bootstrap/` |
| Phase 1 | Three-tier corpus construction (native + synthetic + rescue) | Chapter 3 | `src/lughagen/data/`, `scripts/phase1_corpus/`, `results/corpus/` |
| Phase 2 | 16-cell factorial hyperparameter sweep on Llama 3.1 8B | Chapter 4 | `scripts/phase2_factorial/`, `configs/registry/factorial_*.{json,csv}` |
| Phase 3 | Method comparison: FFT/CPT/QLoRA x 3 seeds on Llama 3.1 8B | Chapter 5 | `scripts/phase3_seeds/`, `results/phase3_n3_summary.json`, `results/phase3_per_language/` |
| Gap 3 | Cross-tokenizer BPC eval of 6 external baselines | Chapter 6 | `src/lughagen/evaluation/eval_baseline_bpc.py`, `results/baseline_bpc/` |
| Phase 3.5 | Scale x specialization: CPT + FFT on Gemma 2 27B | Chapter 6 | `scripts/phase3_5_scale/`, `configs/deepspeed/ds_config_gemma27b.json` |

### Framework specification

The full training framework is specified in [`docs/LughaGen_Training_Framework_Spec_v1.5.1.md`](docs/LughaGen_Training_Framework_Spec_v1.5.1.md). Amendments applied during execution are documented separately in [`docs/amendments.md`](docs/amendments.md):

- **Amendment 8** (2026-05-27) — Token-aware embedding remap (exact-match + subword-decompose)
- **Amendment 10** (2026-08-01) — Configurable remap directory for model-family swap
- **Amendment 11** (2026-08-02) — Softcap disable + SDPA routing for Gemma-family
- **Amendment 12** (2026-08-03) — Pre-save checkpoint prune for 27B disk safety

Two disk-race crashes and their post-mortems are in [`docs/crash_postmortems.md`](docs/crash_postmortems.md).

### Results summary (as of 2026-08-04)

**Chapter 5 pooled BPC (from n=3 seeds each on Llama 3.1 8B):**

| Method | Pool BPC | Std (n=3) | GPU-hours (actual) |
|---|---:|---:|---:|
| FFT | 1.210 | +/- 0.020 | ~134 |
| CPT | 1.269 | +/- 0.052 | ~133 |
| QLoRA merged | 1.312 | (n=1) | ~174 |

**Chapter 6 cross-tokenizer BPC on 5-track validation** (`results/baseline_bpc/`, all values pooled):

| Model | Pool BPC |
|---|---:|
| AfriqueGemma-12B (best external) | 2.066 |
| Gemma 2 27B untuned | 2.127 |
| Lugha-Llama-wura | 2.138 |
| AfriqueLlama-8B | 2.139 |
| **LughaGen-Llama-QLoRA** | **1.312** |
| **LughaGen-Llama-CPT** | **1.269** |
| **LughaGen-Llama-FFT** | **1.210** |

LughaGen-Llama-FFT beats the best external baseline (AfriqueGemma-12B) by 41% on pooled BPC using ~134 A100-hours vs an estimated ~3000 A100-hours for AfriqueGemma-12B (published token count, estimated compute). Efficiency: **340x higher delta-BPC per 100 GPU-hours**, using **~5% of the training tokens** (1B vs ~20B for WURA).

### Reproduction — quick paths

Prepare corpus (Phase 1):

```bash
python -m src.lughagen.data.build_filtered_corpus
```

Train (Phase 3 FFT on Llama 3.1 8B, seed 137):

```bash
bash scripts/phase3_seeds/phase3_fft_seed1.sh
```

Evaluate per-language on the 5-track validation set:

```bash
python -m src.lughagen.evaluation.eval_per_language_5tracks \
    --model-path checkpoints/RUN-P3-FFT/checkpoint-5400
```

Cross-tokenizer BPC against an external baseline:

```bash
python -m src.lughagen.evaluation.eval_baseline_bpc \
    --model-path meta-llama/Llama-3.1-8B \
    --output results/baseline_bpc/base_llama_8b.json
```

### Model releases

Trained models will be released on HuggingFace Hub post-thesis. See [`scripts/release/`](scripts/release/) for the upload pipeline (`stage_release_dir.sh`, `generate_model_cards.py`, `upload_lughagen_hf.sh`).

Local staging is at `/workspace/releases/lughagen/` on the training instance, with self-contained directories per model (hard-linked safetensors, LOAD.py demo, per-language eval JSON, auto-generated model card).

### Base model licenses (for derived checkpoints)

- Llama 3.1: https://www.llama.com/llama3_1/license/
- Gemma 2: https://ai.google.dev/gemma/terms

### Citation

```bibtex
@mastersthesis{lughagen2026,
    author  = {Mayabi, Derek},
    title   = {LughaGen: Adapting Large Language Models for East African Languages},
    school  = {Jomo Kenyatta University of Agriculture and Technology (JKUAT)},
    year    = {2026},
    type    = {{MSc} thesis},
    address = {Nairobi, Kenya}
}
```

Supervised by Dr. Lawrence Nderu at JKUAT. Compute infrastructure provided by Brev.dev.
