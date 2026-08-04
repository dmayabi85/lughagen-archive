# RUN-P2-C03-S1 Quick Record

Completed: 2026-06-17 ~00:00 UTC
val_pph: 31.30 (vs AdamW cells ~27.7-27.8)
Wall clock: 12h 15m
Step time: 7.35 s/step (matches AdamW at seq=1024)
Final train loss: 2.763
Final grad_norm: 1.552
Config: cosine/128/Lion/1024/curriculum, seed=42
Phase tag: Phase2_Factorial (clean)
LR: peaked at 2e-5 (Lion-paper recommendation, 10x lower than AdamW)

** FIRST CELL WITH SIGNIFICANT FACTOR EFFECT **
Lion at lr=2e-5 converges to ~3.5 PPL worse than AdamW (12.6% relative)
No Mode B divergence (May 27 amendment fix worked correctly)
Steady descent throughout 6000 steps

Four-cell summary:
- C01-S1: 27.79 (AdamW/1024/mixed/s42)
- C01-S2: 27.71 (AdamW/1024/mixed/s137)
- C02-S1: 27.76 (AdamW/2048/curriculum/s42)
- C03-S1: 31.30 (Lion/1024/curriculum/s42)  ← OPTIMISER EFFECT IDENTIFIED

Effect magnitudes:
- Seed variance: 0.08 PPL
- Seq/order combined: 0.05 PPL (negligible)
- Optimiser (AdamW → Lion): 3.5 PPL  ← REAL FACTOR EFFECT

Next: launch full sweep (γ) — factor effects exist; sweep will produce analyzable data.
