# Phase 2 Restart Decision Record
**Date**: 2026-05-27
**Author**: Derek Mayabi (Model Lead)
**Status**: Approved — proceed to implementation

## Decision

Full restart of Phase 2 sweep with two amendments to Spec v1.5.1.

## Reasons

### Amendment 1: LoRA target modules (4 → 7)

**Spec §8 (current)**: implicit attention-only LoRA via `lora_target_modules` default
`("q_proj", "k_proj", "v_proj", "o_proj")` in `training_loop.py`

**Audit finding**: Spec is silent on target-module choice. Original April 2026
planning documents specified all 7 modules. Implementation drift to 4 modules
between April and May 2026 launch is undocumented.

**Amendment**: target_modules = ("q_proj", "k_proj", "v_proj", "o_proj",
"gate_proj", "up_proj", "down_proj"). Total trainable parameters grows
79.17M → 107.48M (1.34% of base). Matches literature norm (Lugha-Llama,
AfriqueLLM).

### Amendment 2: Lion learning rate (2e-4 → 2e-5)

**Spec §3.1 (current)**: "peak LR is held constant at 2 × 10⁻⁴ across all 32
base factorial runs regardless of optimiser choice" — intentional conservatism.

**Audit finding**: Cell 13 (RUN-P2-C07-S1) ran 1,500+ steps with:
- grad_norm = inf / nan (32 occurrences)
- Loss frozen at ~9.63 (initialization value)
- Per-language gradient norms: swahili=inf, kikuyu=inf, luo=1e15-1e16
- Zero learning signal extracted

This is Mode B (numerical divergence), not the Mode A (mild underperformance)
that the spec anticipated. Continuing at Lion lr = 2e-4 produces no
informative data — all 16 Lion cells would yield PPH ≈ 0.

**Amendment**: lion_peak_lr = 2e-5 (10× lower than AdamW per Lion paper
recommendation and lion-pytorch library default). Optimiser main effect now
compares "Lion-at-tuned-lr vs AdamW-at-tuned-lr" rather than the previous
"both at AdamW's lr" framing.

## Data discarded

8 production cells from initial sweep:
- RUN-P2-C01-S1 (val_pph=11757, PPH=24.76)
- RUN-P2-C01-S2 (val_pph=11758, PPH=24.23)
- RUN-P2-C02-S1 (val_pph=12872, PPH=7.09)
- RUN-P2-C02-S2 (val_pph=12872, PPH=7.10)
- RUN-P2-C05-S1 (val_pph=11710, PPH=15.54)
- RUN-P2-C05-S2 (val_pph=11708, PPH=15.57)
- RUN-P2-C06-S1 (val_pph=12847, PPH=4.35)
- RUN-P2-C06-S2 (val_pph=12845, PPH=4.37)
- Cell 12 (RUN-P2-C09-S1) — completion status TBD
- Cell 13 (RUN-P2-C07-S1) — killed at step 1563 (gradient explosion)

Plus 4 failed Lion cells (C03-S1, C03-S2, C04-S1, C04-S2) from earlier
wiring bug.

Total discarded: ~1,340 GPU-hours of compute (4.3% of 32,000 budget).
Disk archive: /workspace/restart_backup_20260527T052144Z/ (9.7 GB)

## Preflight protocol (new in this restart)

After applying Amendments 1-2, validate via:
1. Syntax check (training_loop.py parses)
2. AdamW 50-step smoke test (regression check: loss decreasing, grad_norm < 10)
3. Lion 50-step smoke test (catch Mode B early: loss decreasing, grad_norm finite)
4. 7-module memory test at seq=2048 × batch=256 (worst-case memory)
5. Full pilot cell (RUN-P2-PILOT-7MOD, C01-S1 recipe, ~12 hours)
6. Compare pilot vs old C01-S1 val_pph (expect lower with 7-mod = better quality)

Only after all 6 pass: full sweep restart.

## Risk acknowledgment

The Lion-lr change makes the optimiser comparison "tuned-vs-tuned" rather
than "same-lr conservative." This is methodologically stronger but represents
a spec amendment. Documented in this record and in updated thesis methodology
section.

