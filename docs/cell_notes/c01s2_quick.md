# RUN-P2-C01-S2 Quick Record

Completed: 2026-06-14 ~11:40 UTC
val_pph: 27.7 (vs C01-S1: 27.79; delta -0.09 = 0.3%)
Wall clock: 15h 26m (vs C01-S1: 12h 13m; +26% slower)
Step time: 9.26 s/step (vs C01-S1: 7.33; +26%)
Final train loss: 2.736 (vs C01-S1: 2.668)
Final grad_norm: 1.078 (vs C01-S1: 1.074)
Config: cosine/128/AdamW/1024/mixed, seed=137 (vs S1 seed=42)
Phase tag: Phase2_Factorial (clean)

Seed variance: 0.09 PPL = effectively zero
Methodological conclusion: pipeline reproducible; sweep is justified

Concerns:
- 26% slower wall clock (cause TBD; likely disk pressure or cache miss)
- Root disk hit 96% during run (must be addressed before next launch)
