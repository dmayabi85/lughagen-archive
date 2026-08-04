# Gemma 2 27B FFT — Prep Directory

Prep infrastructure for running Gemma 2 27B FFT after LughaGen Tier 1 completes,
enabling direct raw-quality comparison against LughaGen FFT (measured 19.88 pooled val_ppl).

## Scope

- Base model: google/gemma-2-27b (base variant, not the -it instruction-tuned)
- Method: FFT only
- Tokenizer strategy: Resize to strategy_c 8K vocab (LughaGen methodology parity)
- Corpus: pathD_v2_filtered (1,380,264 records — identical to LughaGen; no augmentation)
- Comparison target: LughaGen FFT at 19.88 pooled val_ppl
- Research question: Under identical corpus and methodology, does Gemma 2 27B FFT beat LughaGen FFT on raw quality?

## Why no corpus augmentation

Investigation of publicly available multilingual datasets (aya_dataset, xP3x) revealed
that meaningful native monolingual content for our four target languages does not
exist in public sources beyond our own corpus. See docs/augmentation_investigation.md
for empirical findings. Adding available public content would introduce data type
inconsistency (translation templates vs native text) without meaningful volume gains.
Running Run A only isolates the base model effect cleanly.

## Budget

- Compute: ~735 GPU-hours (~4 days wall-clock on 8×A100-80GB)
- Storage: ~140 GB peak (base 54 GB + checkpoint 54 GB + training state ~30 GB)
- Engineering: 3-5 days for tokenizer resize + smoke test
- Time to results: ~2 weeks from start

## What is here

    gemma2_27b_prep/
    ├── README.md                              # this file
    ├── scripts/
    │   └── download_gemma2_27b.sh             # ready to run
    ├── docs/
    │   ├── tokenizer_strategy_decision.md
    │   ├── framework_spec_gemma2_extension.md
    │   ├── storage_and_cleanup_plan.md
    │   └── augmentation_investigation.md
    ├── configs/                               # empty until DeepSpeed config written
    └── logs/                                  # download + prep logs

## What is ready

- Prep directory structure
- Planning docs including augmentation investigation
- Download script (safe to run after LughaGen chain completes)

## What is not ready yet

- Embedding remap code — needs to inspect Gemma 2 state_dict structure
- Baseline eval script — needs remapped tokenizer to measure fairly
- FFT launcher script — depends on remap working
- DeepSpeed config for 27B — adapted from ds_config_zero3.json

Those get built after download completes and model can be inspected.

## Isolation from LughaGen

Nothing here touches:

- /workspace/checkpoints/
- /workspace/scripts/
- /workspace/configs/ds_config_zero3.json
- /workspace/registry/

All new files under /workspace/gemma2_27b_prep/ and /workspace/models/gemma-2-27b/.

## Workflow when ready to launch

1. LughaGen chain completes (~Friday morning)
2. Run download script (~1-3 hours, run after chain completes)
3. Write embedding remap code (inspect Gemma structure)
4. Write baseline eval + FFT launcher
5. Untuned baseline measurement (~30 min compute)
6. Smoke test 100 steps (~1 hour)
7. Full FFT run (~4 days)
8. Per-language eval + PPH + analysis

## References

- LughaGen FFT measured: 19.88 pooled val_ppl, 133 GPU-hours, Llama-3.1-8B base
- AfroBench (Ojo et al. 2024): Gemma 2 27B strongest open base for African NLP
- Corpus: /workspace/data/corpus/splits_v2/pathD_v2_filtered/
- Untuned Llama-3.1-8B baseline: 8,364.21 pooled val_ppl (V3)
