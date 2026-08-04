# Framework Spec Extension — Phase 3.5: Base Model Comparison

Addendum to LughaGen_Training_Framework_Spec_v1.5.1.md.
Adds a single experiment testing whether Gemma 2 27B FFT outperforms LughaGen FFT
on raw quality under matched methodology.

## Phase 3.5 experiment definition

RUN-P3.5-GEMMA2-FFT-S137

- Base model: google/gemma-2-27b (base variant)
- Tokenizer: strategy_c 8K vocab (resized from Gemma native 256K)
- Method: FFT (all parameters trainable)
- Corpus: pathD_v2_filtered (1,380,264 records — identical to LughaGen)
- Seed: 137 (matches LughaGen FFT S1 seed for direct comparability)
- Steps: 6000 (matches LughaGen)
- Effective batch size: 128 (reduced from LughaGen 256 due to memory pressure)
- Sequence length: 1024 (matches LughaGen)
- Learning rate: 5e-6 (extrapolated from LughaGen 1e-5, scaled for larger model)
- Warmup: 3 percent (matches LughaGen)
- Optimizer: AdamW (matches LughaGen)
- Precision: bf16 mixed (matches LughaGen)
- DeepSpeed: ZeRO-3 (matches LughaGen but new config for 27B memory)
- Gradient checkpointing: enabled
- Top-K checkpoint retention: 1 (reduced from LughaGen 3 to save storage)

## Evaluation

Same 5-track per-language evaluation as LughaGen:

- Swahili val_ppl
- Sheng val_ppl
- Luo val_ppl
- Kikuyu val_ppl
- Kamba val_ppl
- Pooled val_ppl

Validation set: /workspace/data/corpus/splits_v2/pathD_v2_filtered/validation_shared_v3.jsonl
(5,976 records, 300,386 tokens — identical to LughaGen evaluation)

## Baselines

Two baselines for PPH computation:

- Baseline-shared: 8,364.21 (Llama-3.1-8B untuned V3 — same as LughaGen)
  Purpose: PPH values directly comparable to LughaGen PPH values
  
- Baseline-native: Gemma 2 27B untuned per-language perplexity on validation set
  Purpose: PPH values in Gemma native terms
  Measurement: separate baseline eval before FFT run (~30 min compute)

Both baselines get logged. Chapter 6 discussion presents both.

## Comparison quantities

Primary comparison against RUN-P3-FFT (LughaGen FFT seed 137):

- Per-language val_ppl deltas
- Pooled val_ppl delta
- PPH-shared delta (using 8,364.21 baseline for both)
- Absolute compute cost (GPU-hours)
- Storage cost (checkpoint size)

Secondary: comparison against RUN-P3-CPT, RUN-P3-CLEAN-QLORA where informative.

## Predictions to test

From tonight's analysis (Chapter 6 draft):

- Predicted pooled val_ppl: 14 to 18 (LughaGen measured 19.88)
- Predicted per-language: Swahili 8-12, Sheng 10-14, Luo 15-18, Kikuyu 20-25, Kamba 28-33
- Predicted compute: 550-750 GPU-hours (LughaGen measured 133)
- Predicted PPH-shared: ~13 (LughaGen measured 62.74)

Empirical result will confirm or refute these predictions.

## Statistical treatment

Point estimate only for Phase 3.5 (single seed, n=1).
Chapter 6 discussion notes this as future work: seed replicates for statistical CIs
would require additional ~2,200 GPU-hours (3 more seeds), which is out of current scope.

## Compliance with existing framework spec

The following spec requirements carry over unchanged:

- Direct-eval measurement methodology (no wrapping for FFT — same as LughaGen)
- Preflight checks (adapted for Gemma paths)
- Registry logging (new run_id: RUN-P3.5-GEMMA2-FFT-S137)
- Global constants: new baseline entry for Gemma 2 27B untuned

The following require Gemma-specific handling:

- Embedding remap (see tokenizer_strategy_decision.md)
- DeepSpeed config for 27B memory footprint
- tie_word_embeddings save handling (may need PEFT-style manual patch)
- Logit softcapping in loss calculation (Gemma-specific)

## What this experiment answers

- Under matched methodology, does 27B > 8B raw quality justify 5x compute cost?
- Does Gemma 2 base pretraining exposure to African languages (per AfroBench) transfer
  to native language modeling, or is it primarily useful for translation tasks?
- Where does LughaGen sit on the Pareto (quality, compute) frontier?

## What this experiment does not answer

- Whether CPT or QLoRA on Gemma 2 27B would produce different tradeoffs
  (out of scope — FFT only)
- Whether Gemma 2 27B with different tokenizer strategy would perform better
  (out of scope — methodology parity is the point)
- Whether smaller Gemma variants (2B, 9B) would offer better cost-quality tradeoffs
  (out of scope — future work)

## Deliverables

After completion:

- /workspace/results/P3.5_GEMMA2_FFT_per_language_eval.json
- /workspace/results/gemma2_27b_untuned_baseline.json
- /workspace/results/phase3_5_gemma2_analysis.md (comparison writeup)
- Updated Chapter 6 with empirical Gemma-vs-LughaGen data (replaces prediction paragraphs)
- Registry entry: RUN-P3.5-GEMMA2-FFT-S137 in runs.csv

## Success criteria

Not defined as "Gemma must beat LughaGen." Either outcome is publishable:

- If Gemma 2 27B FFT beats LughaGen by predicted 10-20 percent: validates scaling hypothesis;
  suggests LughaGen v2 should use larger base
  
- If Gemma 2 27B FFT ties or loses to LughaGen: suggests methodology-corpus fit matters
  more than base model scale for East African adaptation; strengthens LughaGen's
  contribution

Either finding informs future work directions.
