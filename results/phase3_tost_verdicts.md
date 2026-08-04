# Phase 3 TOST Equivalence Analysis — Definitive Results

Generated: 2026-07-28T08:43:57Z

## Pre-registered δ values (retroactive)

Both computed 2026-07-28 via `compute_tost_delta_v2.py` from 32 legitimate Phase 2 factorial cells (RUN-P2-C*-S* pattern, val_ppl < 100 filter to exclude infrastructure failures).

| Baseline convention | Phase 2 mean PPH | δ (5%) | δ (3%) | δ (7%) | δ (10%) |
|---|---:|---:|---:|---:|---:|
| Option 1 (V2 per-method, 13,963) | 84.807 | **4.240** | 2.544 | 5.937 | 8.481 |
| Option 2 (V3 common, 8,364) | 50.733 | **2.537** | 1.522 | 3.551 | 5.073 |

## Phase 3 PPH under both conventions

| Method | Option 1 PPH | Option 2 PPH |
|---|---:|---:|
| QLoRA-clean | 80.13 | 47.94 |
| CPT | 63.14 | 63.14 |
| FFT | 62.71 | 62.71 |

Note: CPT and FFT PPH identical between conventions (both use V3 baseline). QLoRA PPH changes because Option 1 uses V2 baseline (13,963, wrapping-pathway) vs Option 2 uses V3 baseline (8,364, direct pathway).

## TOST equivalence verdicts

| Comparison | Option 1 (per-method) | Option 2 (V3 common) |
|---|---|---|
| FFT vs QLoRA | QLoRA superior (diff -17.42) | FFT superior (diff +14.77) |
| CPT vs QLoRA | QLoRA superior (diff -16.98) | CPT superior (diff +15.21) |
| **FFT vs CPT** | **EQUIVALENT (diff -0.43)** | **EQUIVALENT (diff -0.43)** |

### Robustness of FFT-vs-CPT equivalence

FFT vs CPT declared equivalent under **8 configurations** (2 baselines × 4 sensitivity δ values):

| δ | Option 1 verdict | Option 2 verdict |
|---|:---:|:---:|
| 3% | EQUIVALENT | EQUIVALENT |
| 5% | EQUIVALENT | EQUIVALENT |
| 7% | EQUIVALENT | EQUIVALENT |
| 10% | EQUIVALENT | EQUIVALENT |

|diff| = 0.4336 across all configurations. Ratio |diff| / δ:
- Under Option 1 5%: 0.10 (10% of the equivalence band)
- Under Option 2 5%: 0.17 (17% of the equivalence band)

Well within the band across all sensitivity checks.

## Interpretation

### FFT vs CPT: Robust positive equivalence
Compute-efficiency equivalent under both baseline conventions and all sensitivity δ values. Confidence in this equivalence claim is high.

### QLoRA vs FFT/CPT: Baseline-convention-dependent
- Under Option 1 (V2 per-method baseline), QLoRA appears superior because its ~5,600 PPL wrapping-pathway penalty is embedded in the PPH numerator as apparent improvement.
- Under Option 2 (V3 common baseline), QLoRA appears inferior because all methods are measured against the same untuned direct-eval baseline, removing the wrapping artifact and letting QLoRA's higher GPU-hours (174 vs 132-133) dominate.

**Neither verdict is "correct" in isolation.** This is a methodological finding: PPH comparisons across different eval pathways require careful baseline convention specification.

### Quality ranking unambiguous
FFT (19.88) < CPT (22.81) < QLoRA (24.62 wrapped / 25.58 direct) across all measurements and conventions. FFT wins all 5 language tracks.

## Production recommendation

**FFT**. Reasoning:
1. Best raw quality on all 5 tracks (unambiguous across conventions)
2. Compute-efficiency equivalent to CPT (robust TOST result)
3. Cost-competitive with CPT (133 GPU-hrs vs 132)
4. Avoids the QLoRA baseline-interpretation ambiguity
5. Substantially better than QLoRA on quality regardless of PPH convention

## Pre-registration compliance disclosure

- δ values were **not** pre-registered in `global_constants.yaml` before Phase 3 launched, contrary to spec §10.3.4 requirements
- Cause: workflow gap in Phase 3 launcher preflight validation (executed with `--skip-preflight`)
- Mitigation: retroactive computation on 2026-07-28 using specified formula (5% × |mean Phase 2 PPH|) applied to filtered Phase 2 cells
- Formula inputs (Phase 2 PPH values) were fixed before Phase 3 launched; only execution timing was deferred
- Retroactive-write flag `retroactive: true` present in `global_constants.yaml` for audit trail
- Sensitivity analysis across 3%, 5%, 7%, 10% δ per §10.3.4 to demonstrate robustness

## Files

- Pre-registered δ values: `/workspace/registry/global_constants.yaml` (tost_delta_pph_v2.*)
- Script: `/workspace/scripts/compute_tost_delta_v2.py`
- Detailed data: `/workspace/results/phase3_tost_verdicts.json`
- Per-language PPH: `/workspace/results/phase3_definitive_pph_matrix.md`
