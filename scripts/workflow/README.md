# Aug 11-15 Workflow

Sequenced scripts for post-training evaluation and analysis.

## Execution order

Aug 11 (execution day, ~11 hours):
1. aug11_step1_verify_completion.sh
2. sanity_check_eval_pipeline.sh
3. aug11_step3_gemma_fft_eval.sh
4. aug11_step4_baseline_bpc.sh
5. aug11_step5_bootstrap_cis.sh (extend with paired bootstrap - Gap 25)
6. aug11_step6_per_language_table.py (extend with macro-BPC - Gap 29)
7. aug11_step7_best_checkpoint_eval.sh (3 Llama variants)
8. aug11_step8_gemma_fft_step4000_eval.sh

Aug 12:
- aug12_step1_flores_setup.sh (add contamination check - Gap 28)
- aug12_step2_flores_llama.sh
- aug12_step3_compute_efficiency.py (add equal-compute table - Gap 32)
- Write formal GPU-hour accounting (Gap 31)

Aug 13:
- aug13_flores_gemma.sh
- Add seed variability report to Chapter 5 (Gap 27)

Aug 14:
- aug14_factorial_effects.py

Aug 15:
- aug15_causal_claim_review.sh (Gap 35 already scripted)

## References
- docs/comprehensive_gap_analysis.md - all 31 gaps
- docs/best_checkpoint_analysis_outline.md - Appendix 5.A structure
- docs/chapter5_efficiency_outline.md - trajectory analysis
- docs/thesis_defense_notes.md - defense framing


## Late Reviewer Additions (Aug 9 second round)

Framing and methodology improvements integrated:

- Item 1: Co-primary outcomes framing (Chapter 3, Aug 15-16)
- Item 4-fix: Preregistered contamination exclusion rule (aug12_step1)
- Item 9: Controlled vs empirical evidence labels (Chapters 4/5/6)
- Item 10: Primary/secondary outcome hierarchy (Chapter 3)

Language corrections during writing:
- "95 percent CI excludes zero" not "statistically significant"
- Report absolute delta + relative percentage + CI + GPU-hours together
- "Under our conditions" not causal attribution for cross-model comparisons
