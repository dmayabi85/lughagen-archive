# Chapter 8 — Section 8.X: Adaptive Training Length (Future Work)

## Draft

### 8.X.1 Empirical Observation

Analysis of LughaGen training trajectories (Section 5.X) reveals a
consistent saturation pattern across model families and adaptation methods:

- 95% of final eval loss reduction is achieved between step 3000-4000
- The final 33% of compute contributes ≤5% of total improvement
- Best-validation checkpoint occurs at step 5000-5400 (not step 6000)
  for one of four completed models (Llama-FFT-S1)

### 8.X.2 Future Research Directions

**1. Automatic eval-plateau early stopping**

A criterion such as "stop when Δ eval loss < 0.005 over 500 consecutive
steps" applied to LughaGen trajectories would have stopped training at:

- Gemma-CPT: step 5200 (would save 800 steps × 30 GPU-hrs = 24 GPU-hrs)
- Llama-CPT: step 5400 (would save 600 steps × 6 GPU-hrs = 4 GPU-hrs)
- Llama-FFT: step 5400 (peak, would save 600 steps × 6 GPU-hrs = 4 GPU-hrs)
- Llama-QLoRA: step 5400 (would save 600 steps × 6 GPU-hrs = 4 GPU-hrs)

Total savings: ~36 GPU-hours across 4 models. Applied to Phase 3 (3 seeds
each), savings extrapolate to ~50-70 GPU-hours.

**2. Language-specific saturation study**

Preliminary evidence suggests different East African languages may saturate
at different training points. Kikuyu (initial tokenizer fertility 3.03) may
benefit from longer training than Swahili (fertility 2.32). A per-language
saturation study could reveal whether:

- Kikuyu benefits from 500-1000 additional steps beyond common saturation
- Swahili plateaus at earlier step counts than pooled metric suggests

**3. Method-aware saturation dynamics**

Observed saturation patterns:
- CPT (both families): ~step 3600
- FFT: ~step 3800 (with post-4000 non-monotonic behavior in Llama)
- QLoRA: ~step 4000 (perfectly monotonic)

Systematic characterization of these dynamics could inform method-aware
training budgets.

**4. Cross-language regularization from longer training**

Llama-FFT-S1 shows soft overfitting after step 5400. Whether this indicates:
- Overfitting to training data distribution
- Optimizer instability in the final decay phase
- Language-specific overfitting (e.g., Kikuyu-specific)

...is worth studying with per-language eval loss trajectories.

### 8.X.3 Practical Recommendation

For future LughaGen framework applications, we suggest:

- **Initial commitment**: 4000 training steps (achieves 95% of ultimate BPC)
- **Continuation criterion**: If Δ eval loss > 0.01 over previous 500 steps,
  continue to step 5000
- **Maximum**: 6000 steps (methodological consistency, cross-baseline
  comparability)
- **Best-checkpoint selection**: Preserve every-200-step checkpoints via
  Amendment 12; use minimum-eval-loss checkpoint for final release

This maintains fixed-compute budget framing while allowing per-run
adaptation to observed dynamics.

## Contribution

If retrospectively applied:
- ~36 GPU-hours saved across the 4 completed LughaGen models
- ~50-70 GPU-hours saved across all Phase 3 seed variants
- <1.5% BPC quality penalty at each model
- Best-checkpoint selection would provide small (0.02 BPC) improvement
  for Llama-FFT specifically

## Placement

Between Section 8.2 (Discussion of results) and Section 8.3 (Broader impacts).
Approximately 1-2 pages.

## Cross-references

- Chapter 4: Fixed compute budget framing
- Chapter 5.X: Efficiency analysis
- Chapter 5.3: Per-language BPC (Kikuyu vs Swahili patterns)
- Chapter 6: BPC/GPU-hour metric
