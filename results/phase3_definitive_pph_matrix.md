# Phase 3 Definitive PPH Matrix

Generated: 2026-07-28T08:28:04Z

## Methodology

All three methods evaluated via **direct HuggingFace causal LM forward pass** with the strategy_c tokenizer + token-aware embedding remap applied. This is commensurable across methods:

- **QLoRA-clean**: LoRA adapter merged into base weights via `peft.merge_and_unload()` before eval, ensuring the same eval pathway as FFT/CPT. The merge required a manual `lm_head` patch due to a PEFT save quirk with `tie_word_embeddings`.
- **CPT / FFT**: full-model checkpoints loaded directly.
- **Baselines**: untuned Llama-3.1-8B with tokenizer resize (128k → 8k East African vocab) and remap applied. Per-language baselines computed on each language slice of the filtered validation set.

## Per-language baselines (untuned Llama-3.1-8B, direct eval)

| Track | Baseline PPL | Interpretation |
|---|---:|---|
| Swahili (authentic) | 7,151.37 | Best-represented in Llama pretraining |
| Kamba | 7,480.48 | Bantu, low but not lowest |
| Kikuyu | 8,573.20 | Bantu, moderate |
| Sheng | 13,542.60 | Code-switched, high entropy |
| Luo | 16,089.55 | Nilotic (typologically distant), highest |
| Pooled | 8,364.21 | Weighted average |

## Perplexity results (direct eval, all commensurable)

| Track | QLoRA (merged) | CPT | FFT |
|---|---:|---:|---:|
| Swahili | 16.34 | 14.48 | **13.42** |
| Sheng | 24.18 | 17.53 | **14.47** |
| Luo | 26.46 | 24.35 | **19.57** |
| Kikuyu | 35.10 | 31.61 | **26.52** |
| Kamba | 48.94 | 43.02 | **35.80** |
| Pooled | 25.58 | 22.81 | **19.88** |

**FFT wins raw quality on all 5 tracks.**

## Linear PPH matrix

| Track | QLoRA (merged) | CPT | FFT | Winner |
|---|---:|---:|---:|---|
| Swahili | 41.02 | **54.03** | 53.64 | CPT (by 0.39) |
| Sheng | 77.71 | **102.39** | 101.67 | CPT (by 0.72) |
| Luo | 92.34 | **121.61** | 120.77 | CPT (by 0.84) |
| Kikuyu | 49.08 | **64.66** | 64.23 | CPT (by 0.43) |
| Kamba | 42.72 | **56.30** | 55.95 | CPT (by 0.35) |
| Pooled | 47.94 | **63.14** | 62.71 | CPT (by 0.43) |

**CPT wins linear PPH on every track**, but by small margins (0.35–0.84 PPH), driven by CPT's marginally lower GPU-hours (132.10 vs 133.06 for FFT — 0.7% difference).

## Log-ratio PPH matrix (values ×1000)

| Track | QLoRA (merged) | CPT | FFT | Winner |
|---|---:|---:|---:|---|
| Swahili | 34.96 | 46.95 | **47.18** | FFT (by 0.23) |
| Sheng | 36.38 | 50.34 | **51.42** | FFT (by 1.08) |
| Luo | 36.85 | 49.16 | **50.44** | FFT (by 1.29) |
| Kikuyu | 31.61 | 42.41 | **43.43** | FFT (by 1.01) |
| Kamba | 28.91 | 39.05 | **40.15** | FFT (by 1.10) |
| Pooled | 33.28 | 44.70 | **45.41** | FFT (by 0.71) |

**FFT wins log-ratio PPH on every track**, driven by larger relative perplexity reduction.

## Ranking disagreement — expected per framework spec §4.5

Framework spec §4.5 explicitly states:

> "if PPH and PPH log-ratio rankings disagree, the disagreement is flagged in the report for manual review"

The CPT vs FFT disagreement is real: CPT wins linear PPH (small absolute cost advantage), FFT wins log-ratio PPH (larger relative improvement per hour). Both agree that QLoRA is 3rd on every track.

## Recommendation

For the LughaGen production candidate: **FFT**. Reasoning:
1. FFT achieves best raw perplexity on all 5 tracks
2. FFT wins log-ratio PPH on all 5 tracks
3. CPT-vs-FFT linear-PPH advantage is negligible (<1% on all tracks)
4. FFT quality advantage over CPT is substantial (3-8 PPL on 4 of 5 tracks)

For budget-constrained applications where marginal cost matters more than absolute quality, CPT is an alternative — but the quality gap is significant enough that FFT is preferred whenever quality is the deliverable.

**QLoRA is Pareto-dominated on both quality and both PPH forms across all tracks.**

## Discussion — the Luo finding

Luo has the highest baseline (16,089) among monolingual languages, ~2.3x higher than Swahili's (7,151). This reflects Luo's typological distance from Llama's pretraining data (Nilotic vs Indo-European/Sino-Tibetan/Bantu). As a consequence:

- Luo has the largest absolute PPH numbers (120+ for FFT/CPT)
- Adaptation methods produce their most dramatic improvement on Luo (16,089 → 19.57, a ~800x reduction)
- Per-compute-hour, adaptation is most cost-effective for the least-represented languages

This inverts the naive assumption that under-resourced languages require MORE compute to reach parity: they actually receive MORE improvement per compute-hour precisely because they start from a lower baseline. The absolute end-state remains worse than well-represented languages, but the cost-effectiveness of adaptation is higher.

## Files referenced

- Baselines: `/workspace/results/base_llama_per_language_baseline.json`
- QLoRA merged eval: `/workspace/results/P3_QLORA_MERGED_per_language_eval.json`
- CPT eval: `/workspace/results/P3_CPT_per_language_eval.json`
- FFT eval: `/workspace/results/P3_FFT_per_language_eval.json`
- Framework spec: `/workspace/scripts/LughaGen_Training_Framework_Spec_v1.5.1.md`
- PPH calculator: `/workspace/scripts/pph_calculator.py`
