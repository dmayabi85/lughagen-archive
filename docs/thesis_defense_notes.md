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

## Possible Outcomes and Their Defense Framing

### Outcome A: Gemma-FFT BPC ≤ 1.30 (matches/beats Llama-FFT 1.210)

**Defense narrative**: "Scale + intensive fine-tuning compound. Full fine-tuning
unlocked Gemma 2 27B's capacity for East African languages, achieving parity
with a smaller purpose-adapted model."

**Novel contribution**: Reproducible recipe (framework + Amendment 11 SDPA
routing + Amendment 12 checkpoint safety) enables 27B-scale specialization for
low-resource languages on modest compute (8×A100 for 90h).

### Outcome B: Gemma-FFT BPC 1.30-1.65 (strong improvement over Gemma-CPT)

**Defense narrative**: "Scale + specialization each contribute independently.
FFT delivers meaningful improvement over CPT (1.78→1.55, ~15% BPC reduction)
but does not close the gap to Llama-8B FFT (1.21). This suggests
tokenizer/pretraining-alignment dominates parameter-count in the specific
regime of low-resource East African languages under fixed compute."

**Novel contribution** (strongest thesis story): Empirical evidence that at
the intersection of small corpus (1.38M records), small compute (~90 GPU-hours
per model), and low-resource target languages, model-family alignment with
target languages beats raw scale. **This is precisely the message for
compute-efficient adaptation research.**

### Outcome C: Gemma-FFT BPC 1.65-1.85 (marginal improvement over CPT)

**Defense narrative**: "Full fine-tuning provides incremental improvement over
continued pretraining at this scale, suggesting the CPT stage captures most of
the recoverable adaptation benefit for the Gemma 2 27B base model on
low-resource East African languages under our data budget."

**Novel contribution**: Documents that CPT can be nearly optimal at 27B scale
for low-resource domains, informing future practitioners' compute allocation.

## Statistical Claims We Can Make

1. **Cross-tokenizer BPC comparison** (Chapter 6):
   - Same held-out validation set (validation_shared_v3_clean.jsonl, 5,952 records)
   - Same UTF-8 byte counting methodology
   - Loss computed in each model's own tokenizer, then normalized to bits per character
   - This IS a valid cross-tokenizer comparison — the metric is BPC on
     bytes, not perplexity per token

2. **N=1 seed for Phase 3.5** (both CPT and FFT):
   - Not directly comparable to Phase 3 which used n=3 seeds
   - We cannot claim statistical significance vs baselines for Phase 3.5
   - Framing: "Phase 3.5 provides scale evidence at n=1; per-language
     patterns are the primary insight, not the pooled point estimate"

3. **Train-eval gap** (Chapter 5/7 discussion):
   - Gemma-CPT: train loss 2.59 → eval loss 3.39 (gap: 0.80 nats)
   - Gemma-FFT projected: train 2.30 → eval 2.90 (gap: 0.60 nats)
   - Llama-FFT: [check actual gap]
   - Consistent train-eval gap magnitude across methods indicates the
     evaluation is measuring generalization, not memorization

## Key Diagnostic Questions to Prepare Answers For

### Q1: "Isn't 27B just overfitting on 1.38M records?"

**Answer**:
- Loss curve shape (cosine decay) matches expected trajectory, no divergence
- Train-eval gap of ~0.6-0.8 nats is standard for our corpus size across all
  three method comparisons (CPT/FFT/QLoRA)
- Amendment 12 preserved checkpoints allow best-validation selection if needed
- Corpus effective tokens: ~1.4B (128 batch × 1024 seq × 6000 steps × 2
  epochs); not enormous but sufficient signal for adaptation

### Q2: "Is your cross-tokenizer BPC methodology correct?"

**Answer**:
- Implementation in `src/lughagen/evaluation/eval_baseline_bpc.py`
- Formula: BPC = NLL_nats / (n_bytes × ln(2))
- Byte count: `len(text.encode('utf-8'))` on RAW validation text
- Loss: computed in model's own tokenizer, aggregated to nats
- Cross-tokenizer validity: BPC on bytes is language-model-quality-per-byte,
  invariant to how the tokenizer splits the text
- Sanity check: re-running eval on identical checkpoint produces identical
  numbers (deterministic)

### Q3: "Why not train longer to close the gap?"

**Answer**:
- Fixed compute budget was thesis scope
- 6000 steps × 87-92 hours saturates a single 8×A100 instance for the
  training window
- Longer training would require additional infrastructure OR abandoning n=3
  for other Phase 3 comparisons
- Scaling to 12000-24000 steps is future work

### Q4: "Why isn't LughaGen instruction-tuned or evaluated on downstream tasks?"

**Answer**:
- Thesis scope: language modeling adaptation (base model quality)
- Instruction tuning requires additional dataset construction (SFT format)
  which is separate research
- Downstream evaluation: FLORES-200 planned for Chapter 6 (Swahili, Kikuyu,
  Luo, Kamba). Sheng not in FLORES-200 (code-switched register, no ISO 639-3)
- Deferred to future work: reasoning, in-context task adaptation, multi-turn

### Q5: "Why Gemma 2 27B specifically and not another 20-30B model?"

**Answer**:
- Fair comparison to Llama 3.1 8B in the same open-model space
- Available under permissive research license
- Prior work (AfriqueGemma) demonstrates baseline African-language quality
- Comparable pretraining scale to Llama 3.1 family (both ~15T tokens)
- Mixture-of-experts models (e.g., Mixtral) would require different
  distributed training infrastructure

### Q6: "The 1.977 BPC projection worried me — is that overfitting?"

**Answer**:
- 1.977 was a rough projection from mid-training loss, not measured
- Actual eval will be computed on final checkpoint (Aug 11)
- If actual eval BPC lands in the 1.5-1.8 range (as projected trajectory
  suggests), no overfitting concern — matches Gemma-CPT gap magnitude
- If actual eval BPC exceeds 1.9, we investigate: check-pointing may have
  captured overfitting, would use best-validation checkpoint instead

## Sanity Checks to Run When FFT Completes (Aug 11)

1. **Reproducibility check**: Re-run eval on Gemma-CPT-6000, verify identical
   BPC to earlier eval (1.783). Confirms pipeline is deterministic.

2. **Per-language pattern check**: Look at Gemma-FFT per-language BPC vs
   Gemma-CPT and Llama-FFT. Expected pattern:
   - Swahili: closest to Llama-FFT (best base coverage)
   - Kikuyu/Kamba/Luo: largest gap (weakest base coverage)
   - Sheng: substantial improvement from CPT (rare register)

3. **Cross-family baseline comparison**: Compare Gemma-FFT vs external
   Gemma-based baselines (AfriqueGemma-12B, AfriqueGemma-4B). Even if Gemma-FFT
   doesn't beat Llama-FFT, it should CRUSH Gemma baselines.

4. **Train-eval gap check**: Compare actual gap to Gemma-CPT gap (0.80 nats).
   If gap doubled (1.6+ nats), that's a real overfitting signal → use
   best-validation checkpoint.

## References for Defense Prep

- Amendment 11 (Gemma SDPA routing): `docs/amendments.md`
- Cross-tokenizer BPC implementation: `src/lughagen/evaluation/eval_baseline_bpc.py`
- Phase 3.5 training recipe: `scripts/phase3_5_scale/`
- Corpus quality audit: `results/corpus/audit_comprehensive/`
- Data leakage audit: `results/data_leakage_check.json`

## The Bottom Line

The scientifically interesting question is per-language behavior at scale,
not the pooled BPC point estimate. Prepare defense around per-language
patterns and the fixed-compute framing, not around whether pooled numbers
beat a specific target.
