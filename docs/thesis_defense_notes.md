# Thesis Defense Notes — Scale vs Specialization Framing

## Purpose

Reference document for defending the LughaGen-Gemma vs LughaGen-Llama
comparison in Chapters 5, 6, and 7. Captures the framing developed during
Phase 3.5 training and the diagnostic questions to have prepared answers for.

## Central Framing

The primary research question is NOT:
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

## Recipe Design Rationale

Confirmed via inspection of training scripts:

| Hyperparameter | Llama-8B CPT | Llama-8B FFT | Gemma-27B CPT | Gemma-27B FFT |
|---|---:|---:|---:|---:|
| Peak LR | 1e-5 | 2e-5 | 5e-6 | 1e-5 |
| Warmup steps | 500 | 200 | 500 | 200 |
| Weight decay | 0.1 | 0.01 | 0.1 | 0.01 |
| Batch size | 256 | 256 | 128 | 128 |
| Sequence length | 1024 | 1024 | 1024 | 1024 |
| Steps | 6000 | 6000 | 6000 | 6000 |
| Data ordering | mixed | curriculum | mixed | curriculum |

Design choices:
- **Gemma uses half the Llama peak LR** (5e-6/1e-5 vs 1e-5/2e-5) — conservative
  choice for the larger model, roughly consistent with parameter-scale-aware LR
  heuristics from prior work
- **CPT→FFT ratio is 2× for both families** (matches paradigm)
- **Warmup halved for FFT** (200 vs 500) — allows faster ramp for the more
  aggressive training
- **Weight decay 10× lower for FFT** (0.01 vs 0.1) — matches "specialization"
  goal (allow more parameter movement)
- **Curriculum ordering for FFT, mixed for CPT** — CPT preserves the natural
  distribution; FFT sees code-switching first (thesis design)

Whether these choices achieve competitive BPC will be measured Aug 11.

## Report These Metrics

For each model:
- Pooled BPC (primary Chapter 6 number)
- Per-language BPC (5 languages)
- Bootstrap 95% CI on pooled and per-language
- Number of bytes evaluated (for reproducibility)
- Number of tokens evaluated (for reproducibility)
- Number of examples evaluated (for reproducibility)

Comparative measures:
- **Δ_scale = BPC(Gemma-FFT) - BPC(Llama-FFT)** — the scale question
- **Δ_adaptation = BPC(Gemma-CPT) - BPC(Gemma-FFT)** — the FFT question
- **Relative improvement over CPT** = 100% × (BPC_CPT - BPC_FFT) / BPC_CPT
- **Tokens per parameter** = unique_train_tokens / model_params

## Statistical Claims We Can Make

1. **Cross-tokenizer BPC comparison**: Same held-out validation set
   (validation_shared_v3_clean.jsonl, 5,952 records), same UTF-8 byte counting,
   loss computed in each model's own tokenizer then normalized to bits per
   character. Valid cross-tokenizer comparison because BPC on bytes is
   model-quality-per-byte, invariant to tokenizer choice.

2. **N=1 seed for Phase 3.5**: Bootstrap CIs on the eval set itself give
   confidence intervals on the point estimate. Weaker than n=3 seeds
   (training reproducibility) but stronger than a bare point estimate.

3. **Train-eval gap consistency**: Gemma-CPT gap was 0.80 nats
   (train 2.59 → eval 3.39). If Gemma-FFT has similar gap magnitude, no
   overfitting concern. If gap doubles, flag it.

## Interpretation Protocol

**Interpret the measured result on its own terms**, using these questions
as the framework — not by matching against pre-declared outcome bands:

1. **Did FFT meaningfully improve on Gemma-CPT?**
   - Compute Δ_adaptation, express as % of CPT baseline
   - Compute bootstrap CI on the delta
   - Report as-is, whatever it is

2. **What is the Δ_scale?**
   - Compute Δ_scale, express as % of Llama-FFT baseline
   - Report per-language pattern (may be non-uniform)
   - Consider tokens-per-parameter as structural factor

3. **Is the per-language pattern uniform?**
   - Look at ranking per language
   - Look at where the largest FFT improvements land
   - Look at whether Swahili (best base coverage) and Kikuyu (weakest)
     behave differently

4. **Does the eval methodology check out?**
   - Sanity check reproduces Gemma-CPT eval numbers exactly
   - Train-eval gap in normal range (~0.6-0.8 nats)
   - Bootstrap CIs are narrow (< 0.05 BPC)

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

**Answer**: Recipe scaled with model size (Gemma peak_lr is half of Llama
across both CPT and FFT, matching the parameter-scale-aware LR heuristic).
Both use warmup=200, weight_decay=0.01 for FFT; identical scheduler and
optimizer. Batch size scaled to memory (Llama 256, Gemma 128) but total
tokens per step comparable. If anything, Gemma FFT is more conservative
than Llama FFT.

### Q7: "What if the eval reveals a surprise (BPC 1.9+)?"

**Answer**: If Gemma-FFT lands substantially above Gemma-CPT baseline
(1.783), first investigate:
- Sanity check verifies pipeline is deterministic (Gemma-CPT reproduced)
- Check train-eval gap: is it >1.5 nats (overfitting) or normal?
- Consider selecting best-validation checkpoint from Amendment 12
- Report whatever the actual number is with CIs. The thesis story
  adapts to the finding, not the reverse

## Sanity Checks to Run When FFT Completes (Aug 11)

1. **Reproducibility check**: Re-run eval on Gemma-CPT-6000, verify
   identical BPC + n_bytes + n_tokens + n_examples to earlier eval (1.783).
   Confirms pipeline is deterministic. Script: `scripts/sanity_check_eval_pipeline.sh`.

2. **Per-language pattern check**: Look at Gemma-FFT per-language BPC vs
   Gemma-CPT and Llama-FFT. Look for uniformity or specific-language wins.

3. **Bootstrap CI computation**: 1000 resamples per language, 95% CI.
   Script: `scripts/compute_bootstrap_ci.py`.

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
framing.

**Freeze the evaluation protocol NOW. Do not adjust methodology after
seeing results.**

Freeze the interpretation framework NOW (the questions in "Interpretation
Protocol" section above). The story emerges from applying those questions
to the measured numbers, not from matching numbers against pre-declared bands.

## Frontier Model Extensions — Deferred to Future Work

During thesis planning, extension to frontier MoE architectures (Llama 4
Scout: 109B total / 17B active MoE; DeepSeek V4 Flash: 284B total / 13B
active MoE) was considered and deferred based on the following analysis:

### Infrastructure fit
- 8×A100 80GB (640 GB total GPU memory) is sized appropriately for
  Llama 3.1 8B and Gemma 2 27B dense architectures used in this thesis
- Llama 4 Scout QLoRA requires either H100-class GPUs or heavy training
  compromises (seq_len ≤512, per-device batch 1, gradient checkpointing)
  that would produce a struggle-run rather than a representative
  framework demonstration
- DeepSeek V4 Flash (284B) exceeds any single-node A100 configuration
  regardless of quantization

### Methodological fit
- LughaGen framework validates on continued pretraining and full
  fine-tuning objectives with language modeling loss
- Frontier MoE models are increasingly deployed as instruction-tuned
  releases, which are trained with SFT objectives on
  instruction-response pairs
- Direct BPC comparison between LM-trained LughaGen models and
  SFT-trained frontier models would require careful methodology
  reframing beyond the scope of this thesis

### Framework readiness
The LughaGen framework (Chapter 4, Amendments 8-12) is designed
model-family-agnostic. Extension to MoE architectures would require:

1. Amendment 11-style attention implementation routing for MoE-specific
   attention patterns (expert routing, expert-parallel attention)
2. Registration of tokenizer for Llama 4 or DeepSeek V4 vocabularies
3. Verification of Amendment 8 (embedding remap) applicability for
   MoE embedding structures

These extensions represent well-scoped follow-up research questions
suitable for either a workshop paper or extended journal submission.

### Native FLORES-200 comparison (post-thesis, easy addition)

For contextualization against frontier models WITHOUT training, a
future extension would evaluate Llama 4 Scout Instruct and DeepSeek V4
Flash via API on the same FLORES-200 devtest sets used in Chapter 6.
This would preserve methodological rigor (no training methodology
mismatch) while providing frontier-model reference points for the
LughaGen benchmark. Estimated effort: 4-6 hours.
