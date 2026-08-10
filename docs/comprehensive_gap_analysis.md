# Comprehensive Thesis Gap Analysis (as of Aug 8, 2026)

## Purpose

Complete inventory of thesis defensibility gaps and their planned resolutions.
Categorized by priority. Each has a plan, timeline, and success criterion.

## Priority Legend

- 🔴 Critical: Must be addressed before defense
- 🟡 Important: Strengthens defense; should be addressed
- 🟢 Nice-to-have: If time allows
- 🔵 Already resolved

## Master Gap Table

### Critical (🔴)

| # | Gap | Action | Timeline | Success Criterion |
|---|-----|--------|:--------:|-------------------|
| 1 | Gemma-FFT final BPC | Run eval on final checkpoint | Aug 11 | JSON saved, per-language extracted |
| 2 | Bootstrap CIs | Run compute_bootstrap_ci.py | Aug 11 | CIs for all comparisons |
| 3 | Per-language table | Aggregate all model eval JSONs | Aug 11 | Table 6.X in Chapter 6 |
| 4 | FLORES-200 downstream | 5 models × 4 langs × 2 directions | Aug 12-13 | Table 6.Y in Chapter 6 |
| 5 | GPU-hour / PPH / Pareto | Compute + plot | Aug 12-13 | Figure 6.Z with frontier |
| 6 | Causal claim wording | Chapter sweep | Aug 15 | Consistent vocabulary |
| 7 | Explicit limitations | Chapter 7 section | Aug 15 | 2-3 pages |

### Important (🟡)

| # | Gap | Action | Timeline | Success Criterion |
|---|-----|--------|:--------:|-------------------|
| 8 | Tokenizer contribution | Limitation statement | Aug 14 | Paragraph in Chapter 7 |
| 9 | Factorial main effects | Analyze existing results | Aug 14 | Update in Chapter 4 |
| 10 | Random baseline | Compute log2(vocab) | Aug 13 | Add row to tables |
| 11 | Untuned ceiling | Already have Gemma/Llama base | Aug 13 | Add to Chapter 6 tables |
| 12 | Baseline versions | Verify + document | Aug 12 | Chapter 6 footnote |
| 13 | Baseline methodology | Align/acknowledge | Aug 12 | Chapter 6 footnote |
| 14 | Validation characterization | Compute stats | Aug 13 | Chapter 3 addition |
| 15 | Framework generalization | Chapter 7 statement | Aug 15 | Paragraph |

### Nice-to-Have (🟢)

| # | Gap | Action | Timeline | Success Criterion |
|---|-----|--------|:--------:|-------------------|
| 16 | Learning curve extrapolation | Trajectory analysis | Aug 15 | Chapter 6/8 mention |
| 17 | Data ordering | Note in future work | Aug 14 | Chapter 8 |
| 18 | Hyperparameter sensitivity | Note in future work | Aug 15 | Chapter 7 |
| 19 | Native-speaker check | If feasible | Optional | Chapter 6 side note |
| 20 | Cross-family baseline | Post-thesis | Deferred | Future work list |

### Resolved (🔵)

| # | Gap | Documentation |
|---|-----|---------------|
| 21 | Corpus quality audit | `results/corpus/audit_comprehensive/` |
| 22 | Data leakage audit | `results/data_leakage_check.json` |
| 23 | Overfitting analysis | `docs/overfitting_verification.md` |
| 24 | Reproducibility protocol | `scripts/sanity_check_eval_pipeline.sh` |

## Success Criteria per Chapter

**Chapter 3 (Corpus)**: Comprehensive audit results + validation set characterization + leakage correction documented.

**Chapter 4 (Experimental design)**: (2^4) factorial + main effects analysis + design justification.

**Chapter 5 (Method comparison)**: 4 LughaGen models on same corpus + bootstrap CIs + per-language BPC + overfitting verification.

**Chapter 6 (Baselines)**: 6 external baselines + cross-tokenizer BPC + FLORES-200 downstream + Pareto frontier + random floor.

**Chapter 7 (Limitations)**: 8-item explicit limitations section + tokenizer caveat + causal-claim discipline verified.

**Chapter 8 (Discussion)**: Scale vs specialization synthesis + compute efficiency conclusions + future work.

## Threats to Validity

1. **Model family confound**: Only Llama and Gemma tested. Findings may not
   generalize to Mistral/Qwen/DeepSeek.

2. **Compute budget confound**: Results specific to 8×A100 configuration
   and 90-hour training window. Longer training may reverse rankings.

3. **Language scope**: 4 primary + Sheng. Not representative of all African
   languages or all East African language families.

4. **Corpus scope**: 1.38M records. Larger corpus may produce different
   scale-vs-specialization dynamics.

5. **Downstream scope**: FLORES-200 measures translation. Other tasks
   (QA, reasoning, generation) may show different patterns.

6. **Seed variance**: Gemma models use n=1 seed. Confidence in Gemma-Llama
   comparison is weaker than Llama internal comparison (n=3).

7. **Tokenizer confound**: Cross-tokenizer BPC is methodologically sound,
   but tokenizer efficiency itself is a variable we cannot isolate.

## Update — Aug 8 Afternoon

### Frontier Model Coverage Decision

Item added and resolved: Llama 4 Scout QLoRA + DeepSeek V4 Flash QLoRA
training was proposed to add frontier model coverage. After analysis of:

- Infrastructure (8×A100 80GB vs H100 recommendations)
- Methodology (LM-based BPC vs SFT-based instruction training mismatch)
- Timeline (5-8 days consumption of critical writing window)
- Better alternatives (already-strong Llama 3.1 8B specialization story;
  Chapter 8 framework generalization discussion; post-thesis Instruct
  API evaluation)

**Decision**: Skip Llama 4 Scout and DeepSeek V4 training. Defer to
post-thesis future work. Add framework extension discussion to Chapter 8.

**Rationale documented**: `docs/thesis_defense_notes.md` (Frontier Model
Extensions section)

## Reviewer Recommendations Added Aug 9 (Session Late Additions)

External reviewer feedback identified 11 additional methodological gaps.
After triage, 7 are being integrated into the Aug 11-15 workflow. All 
are documented here for execution during the writing phase.

### Gap 25: Paired Bootstrap CIs for Differences (CRITICAL)

Priority: Critical
When: Aug 11 (extend Step 5 bootstrap CI script)
Effort: ~1 hour

Problem: Per-model bootstrap CIs do not test whether the difference
between two models is statistically meaningful. Two models can have
overlapping single-model CIs while their paired difference is
significantly non-zero.

Solution: For each primary comparison, compute bootstrap CI on
BPC(A) - BPC(B) using PAIRED sampling (same records resampled for
both models). Report Delta BPC with 95 percent CI. Comparison is
meaningful if CI does not include zero.

Primary comparisons to include:
1. LughaGen-Llama-FFT vs LughaGen-Llama-CPT (adaptation method effect)
2. LughaGen-Llama-FFT vs LughaGen-Llama-QLoRA (full FT vs parameter-efficient)
3. LughaGen-Gemma-FFT vs LughaGen-Gemma-CPT (adaptation within Gemma)
4. LughaGen-Gemma-FFT vs LughaGen-Llama-FFT (scale effect at fixed method)
5. LughaGen best vs best external baseline (framework contribution)

Implementation: Extend compute_bootstrap_ci.py to add paired difference
computation. Draft script name: aug11_step5b_paired_bootstrap.py.

### Gap 27: Seed Variability Quantified (HIGH)

Priority: High
When: Aug 12-13 during Chapter 5 writing
Effort: 30 minutes

Problem: Reporting single-seed BPC obscures configuration robustness.
Reader cannot distinguish "method A truly beats method B" from
"method A happens to have a good seed."

Data available (from previous session records):
- Llama-FFT n=3 seeds: 19.879, 19.918, 19.888 -> mean 19.895, std 0.020
- Llama-CPT n=2 usable seeds: 22.812, 22.916 -> mean 22.864, std 0.052
- Llama-QLoRA n=1
- Gemma-CPT n=1
- Gemma-FFT n=1

Solution: Report "Method: mean +/- std (n seeds)" throughout Chapter 5.
Acknowledge n=1 for Gemma models limits variance estimation.

Example Chapter 5 sentence: "LughaGen-Llama-FFT achieves pooled BPC
1.210 +/- 0.002 (n=3 seeds); LughaGen-Llama-CPT achieves 1.269 +/- 0.003
(n=2 usable seeds). The ~5 percent relative improvement of FFT over CPT
is substantially larger than the cross-seed variance of either method
individually."

### Gap 28: FLORES Contamination Check (CRITICAL)

Priority: Critical (specifically for FLORES eval methodology)
When: Aug 12 (integrate into aug12_step1_flores_setup.sh)
Effort: 1-2 hours

Problem: FLORES-200 sentences might appear in the training corpus
(via crawled web content). Would inflate downstream performance
artificially.

Solution: Before running FLORES eval, check overlap:
1. Extract FLORES devtest sentences for target languages
2. Normalize (lowercase, strip punctuation)
3. Check exact matches against training corpus
4. Check 8-gram overlap against training corpus
5. Report N sentences with any overlap
6. If greater than 5 percent overlap, exclude those sentences

Chapter 6 addition: Report contamination check results transparently.
If no overlap (expected), state so. If some overlap, exclude and report.

### Gap 29: Macro-BPC alongside Micro-Pooled (HIGH)

Priority: High
When: Aug 11 (Step 6 addition)
Effort: 30 minutes

Problem: Reported pooled BPC is micro-averaged (weighted by validation
tokens/bytes), which is dominated by Swahili (~57 percent of corpus).
This can mask poor performance on low-resource languages.

Concrete example (LughaGen-Llama-FFT):
- Micro-pooled BPC: 1.210 (heavily weighted by Swahili 0.950)
- Macro BPC = mean(0.950, 1.401, 1.318, 1.655, 1.369) = 1.339
- Difference: 0.129 BPC (macro is 11 percent worse than micro)

Solution: Report BOTH throughout Chapters 5 and 6:
- Micro-pooled BPC (existing metric, corpus-weighted, comparable to baselines)
- Macro BPC (equal weighting across languages, reflects language equity)

Chapter 5/6 addition: State methodology explicitly. Note that macro
still shows LughaGen wins but by narrower margins on Bantu languages.

### Gap 31: Formal GPU-Hour Accounting (CRITICAL)

Priority: Critical
When: Aug 12 (Chapter 6 methodology section)
Effort: 30 minutes

Problem: Compute-efficiency framing is central to the thesis, but
GPU-hours has not been formally defined. Reviewer will ask what counts.

Solution: Formal definition in Chapter 6 methodology:

    LughaGen Training GPU-hours (reported metric) =
        wall_clock_training_duration_seconds * n_gpus_allocated / 3600

    INCLUDES:
    - All training steps from step 0 to final step
    - Amendment 12 checkpoint save time (part of training loop)
    - DDP/DeepSpeed communication overhead

    EXCLUDES:
    - Evaluation compute (reported separately)
    - Preprocessing / corpus audit (Phase 0)
    - Hyperparameter search (Phase 2 factorial)
    - Failed runs / smoke tests
    - Warmup / model download

    REPORTED SEPARATELY:
    - Training compute (above)
    - Evaluation compute (~15-30 GPU-min per per-language eval)
    - Data-processing compute

Actual GPU-hours per run (verified from logs):
- Llama-CPT: 133 GPU-hours
- Llama-FFT: 134 GPU-hours per seed x 3 = 402 GPU-hours
- Llama-QLoRA: 174 GPU-hours
- Gemma-CPT: ~360 GPU-hours (estimated)
- Gemma-FFT: ~180-200 GPU-hours (estimated)

### Gap 32: Equal-Compute Comparison (CRITICAL)

Priority: Critical (directly supports thesis question)
When: Aug 12-13 (using existing trajectory data)
Effort: 1 hour

Problem: Comparing methods at FINAL BPC obscures whether one method
achieves better BPC per unit compute. The thesis question is about
compute efficiency, so we should show performance at EQUAL compute
across methods.

Data available: Full eval loss trajectories at 200-step intervals
for all 4 completed models. Enables:
- Extract eval loss at any GPU-hour point across all models
- Compare methods at 25, 50, 75, 100 percent of training budget
- Show which method dominates the efficiency frontier

Chapter 5.X / Chapter 6 addition:

    Table X - Method Comparison at Equal Compute Budget

    At 33 percent training budget (step 2000, ~44 GPU-hours):
      Llama-CPT:   eval loss 3.420 -> BPC ~1.44
      Llama-FFT:   eval loss 3.295 -> BPC ~1.39
      Llama-QLoRA: eval loss 3.445 -> BPC ~1.45

    At 67 percent training budget (step 4000, ~88 GPU-hours):
      Llama-CPT:   eval loss 3.155 -> BPC ~1.33
      Llama-FFT:   eval loss 3.026 -> BPC ~1.28
      Llama-QLoRA: eval loss 3.224 -> BPC ~1.36

    At 100 percent training budget (step 6000, ~133 GPU-hours):
      Llama-CPT:   eval loss 3.100 -> BPC 1.269 (final)
      Llama-FFT:   eval loss 2.973 -> BPC 1.210 (final)
      Llama-QLoRA: eval loss 3.165 -> BPC 1.312 (final)

Story: FFT dominates at every compute budget checkpoint. Stronger
claim than comparing final BPC alone.

### Gap 35: Empirical vs Causal Claim Discipline (CRITICAL)

Priority: Critical
When: Aug 15 (already scripted as aug15_causal_claim_review.sh)
Effort: 1-2 hours during Chapter 15 writing

Problem: Cross-model comparisons vary MULTIPLE factors simultaneously
(model family, parameter count, tokenizer, adaptation method,
hyperparameters, batch size). Cannot causally attribute observed
differences to any single factor.

Solution: Systematic wording sweep across all draft chapters. Replace
overclaim patterns with empirical statements.

Replacements:
- "CPT is better than FFT" -> "Under the tested per-paradigm
  configurations, LughaGen-Llama-FFT achieved lower BPC than
  LughaGen-Llama-CPT"
- "Model scale improves performance" -> "The larger-parameter model
  tested (Gemma 27B) did not outperform the smaller-parameter model
  (Llama 8B) under our framework"
- "The 8k tokenizer causes fertility reduction" -> "The 8k custom
  tokenizer reduced pooled fertility to 1.85 tokens/word (vs 3.03
  for Llama native tokenizer)"
- "FFT causes better specialization" -> "FFT with more aggressive
  hyperparameters achieved lower validation BPC. Contribution of
  paradigm vs hyperparameter regime was not experimentally isolated"
- "X is significant" -> "X difference is meaningful given the observed
  95 percent CI of [Y, Z]"

## Not Adding (Reviewer Recommendations Declined)

- Gap 26 (multiple-comparison correction): Simpler approach via defining
  3-4 primary comparisons upfront, treating others as descriptive.
- Gap 30 (macro vs corpus-weighted): Bundled with Gap 29 (macro-BPC).
- Gap 33 (best-checkpoint selection bias): Already handled in Appendix 5.A.
- Gap 34 (effect sizes): Natural byproduct of Gap 25 (paired bootstrap).

## Summary

Total additional effort: ~5-6 hours across Aug 11-15.
Aug 11 workflow: extends Steps 5 and 6 (paired bootstrap + macro-BPC).
Aug 12 workflow: adds FLORES contamination check, GPU-hour definition,
equal-compute comparison.
Aug 13-15: seed variability report and causal claim discipline sweep.
