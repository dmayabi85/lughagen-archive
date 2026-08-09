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
