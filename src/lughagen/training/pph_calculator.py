#!/usr/bin/env python3
"""
/workspace/scripts/pph_calculator.py

LughaGen Training Framework v1.5.1 §4.3, §4.5
Consolidated calculator for the primary outcome metric (Perplexity Per Hour, PPH)
and its log-ratio form.

Definitions (per spec):
  §4.2  P_baseline = pooled validation perplexity of the unmodified base
        (LLaMA-3.1-8B + Phase-0 tokenizer) on the 4-language test set
  §4.3  PPH = (P_baseline - P_final) / gpu_hours_consumed
              ───────────────────────────
              numerator: perplexity reduction
              denominator: GPU-hours actually spent on this run
  §4.5  PPH_log_ratio = log(P_baseline / P_final) / gpu_hours_consumed
                       ─────────────────────────────────────────────────
                       symmetric, scale-invariant, used for primary statistical tests

Why two forms:
  - Linear PPH is human-readable: "reduces perplexity by X per GPU-hour"
  - Log-ratio is what we use for confidence intervals (§10.3) because the log
    transform makes the distribution closer to normal across the factorial space

Negative PPH (P_final >= P_baseline) is logged with flag_negative=true and
appears in the registry record's metrics block. A negative PPH means the run
made the model worse on validation — usually a sign of a broken recipe, but
real (recorded, not silently dropped) so the factorial ANOVA sees it.

Usage:
    from pph_calculator import compute_pph
    result = compute_pph(
        p_baseline=4.213,
        p_final=3.456,
        gpu_hours=72.5,
    )
    # result.pph                — linear form
    # result.pph_log_ratio      — log-ratio form
    # result.flag_negative      — boolean: did P_final exceed P_baseline?
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Optional, Iterable


@dataclass
class PPHResult:
    """Output of compute_pph(). Fields map directly to registry §7.1 schema."""
    p_baseline: float
    p_final: float
    gpu_hours_consumed: float
    pph: float
    pph_log_ratio: float
    flag_negative: bool
    flag_minimal_compute: bool          # gpu_hours < 1.0 → result not reliable
    notes: str = ""

    def to_metrics_dict(self) -> dict:
        """Map to the metrics block of §7.1 registry schema."""
        return {
            "metrics.baseline_perplexity": self.p_baseline,
            "metrics.validation_perplexity": self.p_final,
            "metrics.pph_score": self.pph,
            "metrics.pph_log_ratio": self.pph_log_ratio,
            "metrics.pph_flag_negative": self.flag_negative,
        }


def compute_pph(
    p_baseline: float,
    p_final: float,
    gpu_hours: float,
    min_compute_hours: float = 1.0,
) -> PPHResult:
    """
    Compute PPH (§4.3) and PPH log-ratio (§4.5) for a single training run.

    Args:
        p_baseline: pooled validation perplexity of the base model (§4.2)
        p_final: pooled validation perplexity at end of training
        gpu_hours: total GPU-hours consumed by this run
        min_compute_hours: below this, the result is flagged as unreliable
                          (default 1.0 — smaller runs are smoke tests or failed)

    Returns:
        PPHResult dataclass with both metrics and quality flags.

    Raises:
        ValueError: if perplexities are non-positive or gpu_hours is zero.
    """
    if p_baseline <= 0:
        raise ValueError(f"p_baseline must be > 0, got {p_baseline}")
    if p_final <= 0:
        raise ValueError(f"p_final must be > 0, got {p_final}")
    if gpu_hours <= 0:
        raise ValueError(f"gpu_hours must be > 0, got {gpu_hours}")

    # §4.3: linear form
    pph = (p_baseline - p_final) / gpu_hours

    # §4.5: log-ratio form. log(P_b / P_f) > 0 means improvement
    pph_log_ratio = math.log(p_baseline / p_final) / gpu_hours

    flag_negative = p_final >= p_baseline
    flag_minimal_compute = gpu_hours < min_compute_hours

    notes_parts = []
    if flag_negative:
        notes_parts.append(
            f"p_final ({p_final:.4f}) >= p_baseline ({p_baseline:.4f}): "
            f"run did not improve over baseline"
        )
    if flag_minimal_compute:
        notes_parts.append(
            f"gpu_hours ({gpu_hours:.2f}) < {min_compute_hours}: "
            f"PPH unreliable for short runs"
        )

    return PPHResult(
        p_baseline=p_baseline,
        p_final=p_final,
        gpu_hours_consumed=gpu_hours,
        pph=pph,
        pph_log_ratio=pph_log_ratio,
        flag_negative=flag_negative,
        flag_minimal_compute=flag_minimal_compute,
        notes="; ".join(notes_parts),
    )


def compute_pph_with_ci(
    p_baseline: float,
    p_final_per_seed: Iterable[float],
    gpu_hours_per_seed: Iterable[float],
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> dict:
    """
    Compute mean PPH and bootstrap confidence interval across multiple seeds.
    Used at Phase 2 cell-level (each factorial cell has 2 seeds; §3.3).

    Args:
        p_baseline: shared baseline (a single value, not per-seed)
        p_final_per_seed: validation perplexities for each seed in the cell
        gpu_hours_per_seed: matching gpu_hours per seed
        n_bootstrap: number of bootstrap resamples
        ci_level: confidence level (default 0.95)
        seed: RNG seed for reproducible bootstrap

    Returns:
        Dict with keys: mean_pph, ci_low, ci_high, mean_log_ratio,
        log_ratio_ci_low, log_ratio_ci_high, n_seeds, n_negative.
    """
    import random
    p_final_per_seed = list(p_final_per_seed)
    gpu_hours_per_seed = list(gpu_hours_per_seed)
    n = len(p_final_per_seed)
    if n != len(gpu_hours_per_seed):
        raise ValueError("p_final and gpu_hours arrays must have equal length")
    if n == 0:
        raise ValueError("at least one seed required")

    per_seed_results = [
        compute_pph(p_baseline, pf, gh)
        for pf, gh in zip(p_final_per_seed, gpu_hours_per_seed)
    ]
    pphs = [r.pph for r in per_seed_results]
    log_ratios = [r.pph_log_ratio for r in per_seed_results]

    rng = random.Random(seed)
    boot_means_pph = []
    boot_means_log = []
    for _ in range(n_bootstrap):
        sample_pphs = [rng.choice(pphs) for _ in range(n)]
        sample_logs = [rng.choice(log_ratios) for _ in range(n)]
        boot_means_pph.append(sum(sample_pphs) / n)
        boot_means_log.append(sum(sample_logs) / n)

    boot_means_pph.sort()
    boot_means_log.sort()
    alpha = (1.0 - ci_level) / 2.0
    low_idx = int(alpha * n_bootstrap)
    high_idx = int((1.0 - alpha) * n_bootstrap) - 1

    return {
        "mean_pph": sum(pphs) / n,
        "pph_ci_low": boot_means_pph[low_idx],
        "pph_ci_high": boot_means_pph[high_idx],
        "mean_log_ratio": sum(log_ratios) / n,
        "log_ratio_ci_low": boot_means_log[low_idx],
        "log_ratio_ci_high": boot_means_log[high_idx],
        "n_seeds": n,
        "n_negative": sum(1 for r in per_seed_results if r.flag_negative),
        "n_minimal_compute": sum(1 for r in per_seed_results if r.flag_minimal_compute),
    }


def _smoke_test() -> int:
    """In-line smoke test for the calculator."""
    print("Running pph_calculator.py smoke test...")

    # Case 1: typical improvement
    r = compute_pph(p_baseline=4.213, p_final=3.456, gpu_hours=72.5)
    assert r.pph > 0, f"expected positive PPH for improvement, got {r.pph}"
    assert r.pph_log_ratio > 0
    assert not r.flag_negative
    assert not r.flag_minimal_compute
    print(f"  Case 1 (improvement)        : pph={r.pph:.6f}, log_ratio={r.pph_log_ratio:.6f}")

    # Case 2: degradation (p_final > p_baseline)
    r2 = compute_pph(p_baseline=4.213, p_final=4.500, gpu_hours=72.5)
    assert r2.pph < 0, f"expected negative PPH for degradation, got {r2.pph}"
    assert r2.flag_negative
    print(f"  Case 2 (degradation)        : pph={r2.pph:.6f}, flag_negative={r2.flag_negative}")

    # Case 3: short run
    r3 = compute_pph(p_baseline=4.213, p_final=4.000, gpu_hours=0.3)
    assert r3.flag_minimal_compute
    print(f"  Case 3 (short run)          : pph={r3.pph:.6f}, flag_minimal={r3.flag_minimal_compute}")

    # Case 4: bootstrap CI on a 2-seed cell
    ci = compute_pph_with_ci(
        p_baseline=4.213,
        p_final_per_seed=[3.456, 3.512],
        gpu_hours_per_seed=[72.5, 71.8],
        n_bootstrap=500,
    )
    assert ci["pph_ci_low"] <= ci["mean_pph"] <= ci["pph_ci_high"]
    print(f"  Case 4 (2-seed CI bootstrap): "
          f"mean={ci['mean_pph']:.6f}, CI=[{ci['pph_ci_low']:.6f}, {ci['pph_ci_high']:.6f}]")

    # Case 5: error handling
    try:
        compute_pph(p_baseline=-1.0, p_final=3.0, gpu_hours=10.0)
    except ValueError:
        print(f"  Case 5 (negative baseline)  : correctly raised ValueError")

    print("  ALL PASS")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(_smoke_test())
