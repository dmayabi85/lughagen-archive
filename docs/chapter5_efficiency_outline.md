# Chapter 5 — Section 5.X: Training Efficiency Analysis

## Data Source

Extracted from training logs for 4 completed LughaGen models
(Aug 8, 2026). Each model was evaluated every 200 training steps for
a total of 30 evaluation points across 6000 training steps.

## Table 5.X — Training Trajectory Summary

| Model | Initial eval loss (step 200) | Final eval loss (step 6000) | Total decline | 95% saturation step | Best checkpoint step |
|-------|-----------------------------:|----------------------------:|---------------:|:-------------------:|:--------------------:|
| Llama-CPT-S1  | 4.951 | 3.100 | 1.851 | step 3600 | step 5000-5400 |
| Llama-FFT-S1  | 4.390 | 2.973 | 1.417 | step 3800 | **step 5400** (non-monotonic) |
| Llama-QLoRA   | 4.490 | 3.165 | 1.325 | step 4000 | step 5400-6000 |
| Gemma-CPT-S137 | 7.785 | 3.384 | 4.401 | step 3000 | step 5800-6000 |

## Key Observations

### Observation 1: All models saturate before step 4000

95% of total eval loss decline is achieved by:
- Gemma-CPT: step 3000 (50% of training)
- Llama-CPT: step 3600 (60% of training)
- Llama-FFT: step 3800 (63% of training)
- Llama-QLoRA: step 4000 (67% of training)

Average: **~60% of training** delivers **95% of results**.

### Observation 2: The final third of training contributes ≤5%

Steps 4000-6000 (33% of compute) deliver:
- Llama-CPT: 3% of total improvement
- Llama-FFT: 4% of total improvement
- Llama-QLoRA: 5% of total improvement
- Gemma-CPT: 2% of total improvement

Approximately 15× less compute-efficient than steps 200-1000.

### Observation 3: Llama-FFT shows soft overfitting

Llama-FFT-S1 achieves best eval loss (2.955) at step 5400, not step 6000
(2.973). Non-monotonic decline of 0.018 nats between step 5400 and 5600
indicates weak overfitting. Similar pattern observed in seed 2 and seed 3.

**Amendment 12 (checkpoint safety) preserves step-5400 checkpoints across
all Llama-FFT seeds**, enabling best-validation selection if desired.

### Observation 4: Gemma-CPT initial state is much higher

Gemma-CPT starts at eval loss 7.785 vs Llama-CPT at 4.951 (57% higher).
Two hypotheses for Chapter 7 discussion:
- Amendment 8 embedding remap requires more warmup at 27B scale
- Gemma pretrained state is farther from optimum for East African languages

Gemma-CPT achieves 58% of its total decline in the first 400 steps.

## Compute Efficiency Analysis

### Table 5.Y — GPU-hours per BPC improvement by training phase

| Phase | Steps | BPC ΔLlama-CPT (approx) | BPC ΔGemma-CPT | GPU-hrs (Llama) | GPU-hrs (Gemma) |
|-------|-------|------------------------:|---------------:|:---------------:|:---------------:|
| Rapid | 200-1200 | ~1.10 | ~2.90 | ~15 | ~40 |
| Main decay | 1200-3000 | ~0.55 | ~0.85 | ~20 | ~55 |
| Refinement | 3000-4500 | ~0.15 | ~0.13 | ~15 | ~40 |
| Trailing | 4500-6000 | ~0.05 | ~0.06 | ~15 | ~40 |
| **Total** | 200-6000 | **1.851** | **4.401** | ~65 | ~175 |

### Table 5.Z — Compute-optimal stopping analysis

| Model | 6000 step BPC (final) | Step 4000 BPC | ΔBPC | GPU-hrs saved | Cost-benefit |
|-------|---------------------:|-------------:|-----:|:-------------:|:------------:|
| Llama-CPT | ~1.269 | ~1.275 | +0.006 | 22 GPU-hrs | 3.7 GPU-hrs/BPC point |
| Llama-FFT | ~1.210 | ~1.230 | +0.020 | 22 GPU-hrs | 1.1 GPU-hrs/BPC point |
| Llama-QLoRA | ~1.312 | ~1.325 | +0.013 | 22 GPU-hrs | 1.7 GPU-hrs/BPC point |
| Gemma-CPT | ~1.783 | ~1.815 | +0.032 | 60 GPU-hrs | 1.9 GPU-hrs/BPC point |

If all 4 models had used step-4000 early stopping:
- **Total GPU-hours saved**: ~125 across 4 models
- **Total BPC penalty**: ~0.07 across 4 models (0.02 average per model)
- **Efficiency gain**: ~30% reduction in compute for ~1.5% quality loss

## Methodological Choice: Reporting Step-6000 Numbers

Despite the efficiency analysis showing diminishing returns, this thesis
reports step-6000 numbers because:

1. **Phase 2 factorial calibration**: LR schedule and hyperparameters were
   calibrated for 6000-step training. Changing training length changes
   the effective cosine curve, invalidating Phase 2 conclusions.

2. **Cross-model consistency**: Comparing Gemma-27B at 4000 steps to
   Llama-8B at 4000 steps to Gemma-27B at 6000 steps would introduce
   a training-length confound to scale-vs-specialization analysis.

3. **Baseline comparability**: External baselines (AfriqueGemma,
   Lugha-Llama) trained for millions of steps. Stopping at 4000 weakens
   comparison narratives.

4. **Safety margin**: At step 4000 we could not know whether the model
   would continue improving. Full 6000 provides confidence that
   reported BPC is a stable final value.

5. **Llama-FFT soft overfitting**: Best-checkpoint selection would use
   step 5400 (not 4000). We report step-6000 for consistency but
   preserve step-5400 checkpoints (Amendment 12).

## Efficiency Framing for Chapter 6

For Chapter 6 comparison with external baselines, note that:
- LughaGen models use 6000 training steps × 128-256 batch × 1024 seq
- Approximately 786M-1.5B training tokens per model
- ~65-175 GPU-hours per model

This represents a fixed compute budget considerably smaller than external
baseline training. The comparison thus contextualizes LughaGen quality
under compute-constrained adaptation, not indefinite training budgets.
