# RUN-P2-C01-S1 Quick Record

Completed: 2026-06-12 20:44 UTC
val_pph: 27.79 (matches v2-D 28.16, delta -1.3%)
Wall clock: 12h 13m, ~98 GPU-hours
MFU mean: 17.0% (median 17.9%, p10-p90 14.0-18.1%)
Config: cosine/128/AdamW/1024/mixed, seed=42, 7-module LoRA
Phase tag: Phase2_Factorial (clean)
Anomaly: per-lang grad norms only capture swahili+luo (kikuyu/kamba missing)

Pre-launch fixes applied:
- phase2_sweep.sh phase filter (Task A)
- phase2_sweep.sh path overrides (Task B)
- MFU instrumentation wired into training_loop.py

Backups: bak-pre-phasefilter-20260612T062850Z, bak-pre-paths-20260612T064201Z, bak-pre-mfu-20260611T080525Z

Next: launch RUN-P2-C01-S2 (seed=137) to estimate seed variance.
