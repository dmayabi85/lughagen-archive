# Thesis Defense Notes — Scale vs Specialization Framing

## Purpose

Reference document for defending the LughaGen-Gemma vs LughaGen-Llama
comparison in Chapters 5, 6, and 7. Captures the framing developed during
Phase 3.5 training and the diagnostic questions to have prepared answers for.

## Central Framing

**The primary research question** is NOT:
> "Does Gemma 27B beat Llama 8B?"

It IS:
> "Under a fixed compute budget for low-resource East African language
> adaptation, does increased parameter scale translate to improved downstream
> language modeling performance?"

The answer is scientifically valuable in either direction.

## Two Orthogonal Comparisons

**Scale effect** (isolating parameter count):
- Llama-8B FFT vs Gemma-27B FFT
- Same data, same evaluation, same method
- Different: model architecture, tokenizer, parameter count, pretraining mix

**Adaptation method effect** (isolating training approach):
- Gemma-27B CPT vs Gemma-27B FFT
- Same model, same data, same evaluation
- Different: training objective (LM only vs generation-focused), hyperparams

Both are valid, non-overlapping empirical questions. Don't conflate.

## Report These Metrics

For each model:
- **Pooled BPC** (primary Chapter 6 number)
- **Per-language BPC** (5 languages)
- **Bootstrap 95% CI** on pooled and per-language
- **Number of bytes evaluated** (for reproducibility)
- **Number of tokens evaluated** (for reproducibility)
- **Number of examples evaluated** (for reproducibility)

Comparative deltas:
- **Δ_scale = BPC(Gemma-FFT) - BPC(Llama-FFT)**
- **Δ_adaptation = BPC(Gemma-CPT) - BPC(Gemma-FFT)** ← the FFT improvement
- **Relative improvement over CPT baseline** = 100% × (BPC_CPT - BPC_FFT) / BPC_CPT
- **Structural factor**: tokens-per-parameter for Llama vs Gemma

## Statistical Claims We Can Make

1. **Cross-tokenizer BPC comparison**: Same held-out validation set
   (validation_shared_v3_clean.jsonl, 5,952 records), same UTF-8 byte counting,
   loss computed in each model's own tokenizer then normalized to bits per
   character. This IS a valid cross-tokenizer comparison because BPC on bytes
   is model-quality-per-byte, invariant to tokenizer choice.

2. **N=1 seed for Phase 3.5**: Bootstrap CIs on the eval set itself give
   confidence intervals on the point estimate. This is weaker than n=3 seeds
   (which would test training reproducibility) but is stronger than a bare
   point estimate.

3. **Train-eval gap consistency**: Gemma-CPT gap was 0.80 nats
   (train 2.59 → eval 3.39). If Gemma-FFT has similar gap magnitude
   (~0.6-0.8 nats), no overfitting concern. If gap doubles, flag it.

## Interpreting Whatever Result Comes Out

Rather than pre-committing to interpretation bands, evaluate what actually
emerges against these questions:

1. **Did FFT meaningfully improve on Gemma-CPT?**
   - If ΔI_FFT > 5%: significant adaptation improvement
   - If ΔI_FFT 1-5%: marginal
   - If ΔI_FFT < 0%: FFT failed / needs investigation

2. **Did Gemma-FFT close the gap to Llama-FFT?**
   - Report the gap as-is with CIs
   - Don't frame in binary "won/lost" terms
   - Report per-language pattern (may be non-uniform)

3. **What does the tokens-per-parameter ratio suggest?**
   - Llama-8B: ~1.4B / 8B = 0.175 tokens/param
   - Gemma-27B: same 1.4B / 27B = 0.052 tokens/param
   - Gemma has 3.4× less data-per-parameter for adaptation
   - This structural factor partly explains any Gemma disadvantage

## Key Diagnostic Questions to Prepare Answers For

### Q1: "Isn't 27B just overfitting on 1.38M records?"

**Answer**: Loss curve shape (cosine decay) matches expected trajectory,
no divergence. Train-eval gap of ~0.6-0.8 nats is consistent across CPT
and FFT (Gemma-CPT: 0.80 nats). Corpus effective tokens: ~1.4B (128 batch
× 1024 seq × 6000 steps × epochs); sufficient signal but not enormous.
Amendment 12 preserved checkpoints allow best-validation selection if
needed. Bootstrap CIs quantify measurement uncertainty.

### Q2: "Is your cross-tokenizer BPC methodology correct?"

**Answer**: Implementation in `src/lughagen/evaluation/eval_baseline_bpc.py`.
Formula: BPC = NLL_nats / (n_bytes × ln(2)). Byte count:
`len(text.encode('utf-8'))` on RAW validation text. Loss: computed in
model's own tokenizer, aggregated to nats. Cross-tokenizer validity:
BPC on bytes is language-model-quality-per-byte, invariant to how the
tokenizer splits the text. Sanity check: re-running eval on identical
checkpoint produces identical numbers (deterministic pipeline verified).

### Q3: "Why not train longer to close the gap?"

**Answer**: Fixed compute budget was thesis scope. 6000 steps × 87-92
hours saturates a single 8×A100 instance for the training window. Longer
training would require additional infrastructure OR abandoning n=3 seeds
for Phase 3 comparisons. Scaling to 12000-24000 steps is future work.

### Q4: "Why isn't LughaGen instruction-tuned or evaluated on downstream tasks?"

**Answer**: Thesis scope is language modeling adaptation (base model
quality). Downstream evaluation: FLORES-200 planned for Chapter 6
(Swahili, Kikuyu, Luo, Kamba). Sheng not in FLORES-200 (code-switched
register, no ISO 639-3). Instruction tuning deferred to future work.

### Q5: "Why Gemma 2 27B specifically and not another 20-30B model?"

**Answer**: Fair comparison to Llama 3.1 8B in the same open-model space.
Available under permissive research license. Prior work (AfriqueGemma)
demonstrates baseline African-language quality. Comparable pretraining
scale to Llama 3.1 family (both ~15T tokens).

### Q6: "Are the hyperparameters comparable between Llama and Gemma FFT?"

**Answer**: Recipe scaled inversely with model size (Llama peak_lr=2e-5,
Gemma peak_lr=1e-5, matching the sqrt(1/N) scaling rule). Both use
warmup=200, weight_decay=0.01, 6000 steps. Batch size scaled to memory
(Llama 256, Gemma 128) but total tokens seen per step is comparable.
Gemma FFT is if anything MORE conservative than Llama FFT (halved LR).

### Q7: "What if the eval reveals a surprise (BPC 1.9+)?"

**Answer**: If Gemma-FFT lands substantially above Gemma-CPT baseline
(1.783), first investigate:
- Sanity check verified pipeline is deterministic (Gemma-CPT reproduced)
- Check train-eval gap: is it >1.5 nats (overfitting) or normal?
- Consider selecting best-validation checkpoint from Amendment 12
- Report whatever the actual number is with CIs

## Sanity Checks to Run When FFT Completes (Aug 11)

1. **Reproducibility check**: Re-run eval on Gemma-CPT-6000, verify
   identical BPC + n_bytes + n_tokens + n_examples to earlier eval (1.783).
   Confirms pipeline is deterministic.

2. **Per-language pattern check**: Look at Gemma-FFT per-language BPC vs
   Gemma-CPT and Llama-FFT. Expected pattern:
   - Swahili: closest to Llama-FFT (best base coverage)
   - Kikuyu/Kamba/Luo: largest gap (weakest base coverage)
   - Sheng: substantial improvement from CPT (rare register)

3. **Bootstrap CI computation**: 1000 resamples per language, 95% CI.

4. **Cross-family baseline comparison**: Compare Gemma-FFT vs external
   Gemma-based baselines (AfriqueGemma-12B, AfriqueGemma-4B). Even if
   Gemma-FFT doesn't beat Llama-FFT, it should CRUSH Gemma baselines.

5. **Train-eval gap check**: Compare actual gap to Gemma-CPT gap
   (0.80 nats). If gap >1.5 nats, use best-validation checkpoint.

## References for Defense Prep

- Amendment 11 (Gemma SDPA routing): `docs/amendments.md`
- Cross-tokenizer BPC implementation: `src/lughagen/evaluation/eval_baseline_bpc.py`
- Phase 3.5 training recipe: `scripts/phase3_5_scale/`
- Corpus quality audit: `results/corpus/audit_comprehensive/`
- Data leakage audit: `results/data_leakage_check.json`
- Sanity check protocol: `scripts/sanity_check_eval_pipeline.sh`
- Bootstrap CI computation: `scripts/compute_bootstrap_ci.py`

## The Bottom Line

The scientifically interesting question is per-language behavior at scale,
not the pooled BPC point estimate. Prepare defense around per-language
patterns, tokens-per-parameter structural factor, and the fixed-compute
framing — not around whether pooled numbers hit specific bands.

**Freeze the evaluation protocol NOW. Do not adjust methodology after
seeing results.**
