#!/usr/bin/env python3
"""
/workspace/scripts/select_confirmation_runs.py

LughaGen Training Framework v1.5 §3.4
Selects the 4 confirmation runs after Phase 2's 32 factorial runs complete:
  - Slots 1-3: highest predicted PPH among untested half-fraction
  - Slot 4: same factor settings as slot 1 but with balanced_25_25_25_25 recipe
            (paired sensitivity)

Bootstrap CIs (1000 resamples) are computed on each prediction. If any of the
top-3 predictions has a CI overlapping rank-4's by more than 50% of the band
width, the selection is flagged for manual supervisor review.

Outputs:
  - /workspace/registry/confirmation_schedule.csv  (4 rows, in run order)
  - /workspace/registry/confirmation_predictions.json  (full prediction table)
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, "/workspace/scripts")
from registry_csv import read_runs_by_phase
from factorial_matrix import LEVEL_MAP, FACTOR_TO_FIELD


PHASE2_NAME = "Phase2_Factorial"


def encode_factors(row: dict) -> dict[str, int]:
    """Convert a registry row's factor values to {-1, +1} coding."""
    inv = {
        "lr_schedule":    {v: k for k, v in LEVEL_MAP["A"].items()},
        "batch_size":     {v: k for k, v in LEVEL_MAP["B"].items()},
        "optimiser":      {v: k for k, v in LEVEL_MAP["C"].items()},
        "sequence_length": {v: k for k, v in LEVEL_MAP["D"].items()},
        "data_ordering":  {v: k for k, v in LEVEL_MAP["E"].items()},
    }
    # Coerce types: batch_size and seq_length are ints in LEVEL_MAP
    bs = int(row["factors.batch_size"])
    sl = int(row["factors.sequence_length"])
    return {
        "A": inv["lr_schedule"][row["factors.lr_schedule"]],
        "B": inv["batch_size"][bs],
        "C": inv["optimiser"][row["factors.optimiser"]],
        "D": inv["sequence_length"][sl],
        "E": inv["data_ordering"][row["factors.data_ordering"]],
    }


def build_design_row(levels: dict[str, int]) -> np.ndarray:
    """
    Construct the design-matrix row for the linear model:
      [intercept, A, B, C, D, E, AB, AC, AD, AE, BC, BD, BE, CD, CE, DE]
    Total 16 features (matches Resolution V clean estimates).
    """
    A, B, C, D, E = (levels["A"], levels["B"], levels["C"], levels["D"], levels["E"])
    return np.array([
        1,            # intercept
        A, B, C, D, E,
        A*B, A*C, A*D, A*E,
        B*C, B*D, B*E,
        C*D, C*E,
        D*E,
    ], dtype=float)


def collect_factorial_data() -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """
    Read the 32 factorial runs from runs.csv and build (X, y, rows).

    Returns:
      X: (32, 16) design matrix
      y: (32,) PPH scores
      rows: list of registry dicts in the same order
    """
    runs = read_runs_by_phase(PHASE2_NAME)
    factorial = [r for r in runs if r.get("is_confirmation_run", "false") == "false"]

    if len(factorial) != 32:
        raise RuntimeError(
            f"Expected 32 factorial runs; found {len(factorial)} in registry."
        )

    X_rows = []
    y_rows = []
    for r in factorial:
        levels = encode_factors(r)
        X_rows.append(build_design_row(levels))
        pph_str = r.get("metrics.pph_score") or ""
        if pph_str == "":
            raise RuntimeError(f"Run {r.get('factorial_run_id')} has empty pph_score")
        y_rows.append(float(pph_str))

    return np.array(X_rows), np.array(y_rows), factorial


def enumerate_full_factorial() -> list[dict[str, int]]:
    """All 32 configurations of the full 2^5 design."""
    return [
        {"A": A, "B": B, "C": C, "D": D, "E": E}
        for A, B, C, D, E in itertools.product([-1, +1], repeat=5)
    ]


def tested_configurations(factorial_rows: list[dict]) -> set[tuple]:
    """Set of (A,B,C,D,E) tuples present in the 32 factorial runs."""
    seen = set()
    for r in factorial_rows:
        lv = encode_factors(r)
        seen.add((lv["A"], lv["B"], lv["C"], lv["D"], lv["E"]))
    return seen


def fit_linear_model(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """OLS fit; returns coefficient vector (length 16)."""
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta


def bootstrap_predictions(
    X: np.ndarray,
    y: np.ndarray,
    X_candidates: np.ndarray,
    n_boot: int = 1000,
    seed: int = 20260518,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Bootstrap predictions over n_boot resamples of (X, y).

    Returns:
      mean: (n_candidates,) mean predicted PPH
      lo:   (n_candidates,) 2.5th percentile
      hi:   (n_candidates,) 97.5th percentile
    """
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    preds = np.zeros((n_boot, X_candidates.shape[0]))
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        Xb = X[idx]
        yb = y[idx]
        try:
            beta, *_ = np.linalg.lstsq(Xb, yb, rcond=None)
        except np.linalg.LinAlgError:
            # Resample may be rank-deficient; fall back to full-sample fit
            beta = fit_linear_model(X, y)
        preds[b] = X_candidates @ beta

    mean = preds.mean(axis=0)
    lo = np.percentile(preds, 2.5, axis=0)
    hi = np.percentile(preds, 97.5, axis=0)
    return mean, lo, hi


def overlap_fraction(a_lo: float, a_hi: float, b_lo: float, b_hi: float) -> float:
    """
    Compute the fraction of band A that overlaps with band B.
    Returns a value in [0, 1].
    """
    a_width = a_hi - a_lo
    if a_width <= 0:
        return 1.0 if (b_lo <= a_lo <= b_hi) else 0.0
    overlap = max(0.0, min(a_hi, b_hi) - max(a_lo, b_lo))
    return overlap / a_width


@dataclass
class Candidate:
    levels: dict[str, int]
    predicted_pph_mean: float
    predicted_pph_lo: float
    predicted_pph_hi: float
    rank: int          # 1 = highest mean prediction
    flagged: bool      # CI overlaps rank-4 by more than 50%


def select_top_three(
    candidates: list[Candidate], overlap_threshold: float = 0.5
) -> tuple[list[Candidate], bool]:
    """
    Select the top-3 candidates by mean predicted PPH; flag if their CIs
    overlap rank-4's by more than `overlap_threshold` of band width.

    Returns (top_three, any_flagged).
    """
    sorted_c = sorted(candidates, key=lambda c: c.predicted_pph_mean, reverse=True)
    for i, c in enumerate(sorted_c, start=1):
        c.rank = i

    top_three = sorted_c[:3]
    rank_four = sorted_c[3] if len(sorted_c) >= 4 else None

    any_flagged = False
    if rank_four is not None:
        for c in top_three:
            ov = overlap_fraction(
                c.predicted_pph_lo, c.predicted_pph_hi,
                rank_four.predicted_pph_lo, rank_four.predicted_pph_hi,
            )
            if ov > overlap_threshold:
                c.flagged = True
                any_flagged = True

    return top_three, any_flagged


def to_human_factors(levels: dict[str, int]) -> dict:
    """Map {-1,+1} levels to human-readable factor values."""
    return {
        "lr_schedule":    LEVEL_MAP["A"][levels["A"]],
        "batch_size":     LEVEL_MAP["B"][levels["B"]],
        "optimiser":      LEVEL_MAP["C"][levels["C"]],
        "sequence_length": LEVEL_MAP["D"][levels["D"]],
        "data_ordering":  LEVEL_MAP["E"][levels["E"]],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Select Phase 2 confirmation runs (§3.4)")
    parser.add_argument("--output-schedule", type=Path,
                        default=Path("/workspace/registry/confirmation_schedule.csv"))
    parser.add_argument("--output-predictions", type=Path,
                        default=Path("/workspace/registry/confirmation_predictions.json"))
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--overlap-threshold", type=float, default=0.5,
                        help="If any top-3 CI overlaps rank-4's by more than this "
                             "fraction, flag for manual review (default: 0.5).")
    parser.add_argument("--seed", type=int, default=20260518)
    parser.add_argument("--from-fixture", type=Path, default=None,
                        help="Load 32 factorial runs from a JSON fixture instead "
                             "of the registry (for unit testing).")
    args = parser.parse_args()

    # Read factorial data
    if args.from_fixture is not None:
        with args.from_fixture.open() as f:
            fixture = json.load(f)
        X = np.array(fixture["X"])
        y = np.array(fixture["y"])
        factorial = fixture.get("rows", [])
    else:
        X, y, factorial = collect_factorial_data()

    print(f"Loaded {X.shape[0]} factorial runs (X: {X.shape}, y: {y.shape})")

    # Identify untested configurations
    all_configs = enumerate_full_factorial()
    tested = tested_configurations(factorial) if factorial else set()
    if factorial:
        untested = [c for c in all_configs
                    if (c["A"], c["B"], c["C"], c["D"], c["E"]) not in tested]
    else:
        # Fixture mode without rows — predict for all 32
        untested = all_configs

    print(f"Untested configurations: {len(untested)}")

    # Build design matrix for candidates
    X_candidates = np.array([build_design_row(c) for c in untested])

    # Bootstrap predictions
    print(f"Running {args.n_bootstrap} bootstrap resamples...")
    pred_mean, pred_lo, pred_hi = bootstrap_predictions(
        X, y, X_candidates,
        n_boot=args.n_bootstrap, seed=args.seed,
    )

    candidates = [
        Candidate(
            levels=untested[i],
            predicted_pph_mean=float(pred_mean[i]),
            predicted_pph_lo=float(pred_lo[i]),
            predicted_pph_hi=float(pred_hi[i]),
            rank=0,
            flagged=False,
        )
        for i in range(len(untested))
    ]

    # Select top 3 with overlap check
    top_three, any_flagged = select_top_three(candidates, args.overlap_threshold)

    # Build the 4-row confirmation schedule
    # Slot 4: same factor settings as slot 1, but recipe = balanced_25_25_25_25
    slot1 = top_three[0]
    schedule_rows = []
    for slot_idx, c in enumerate(top_three, start=1):
        f = to_human_factors(c.levels)
        schedule_rows.append({
            "factorial_run_id": f"RUN-P2-CONF{slot_idx:02d}-S1",
            "confirmation_slot": slot_idx,
            "confirmation_slot_type": "pph_predicted",
            "recipe": "full",
            "lr_schedule": f["lr_schedule"],
            "batch_size": f["batch_size"],
            "optimiser": f["optimiser"],
            "sequence_length": f["sequence_length"],
            "data_ordering": f["data_ordering"],
            "predicted_pph_mean": c.predicted_pph_mean,
            "predicted_pph_bootstrap_low": c.predicted_pph_lo,
            "predicted_pph_bootstrap_high": c.predicted_pph_hi,
            "selection_rank": c.rank,
            "flagged_for_manual_review": c.flagged,
            "seed": 19,  # fresh seed not in default {42, 137}
        })

    # Slot 4: paired sensitivity
    f1 = to_human_factors(slot1.levels)
    schedule_rows.append({
        "factorial_run_id": "RUN-P2-CONF04-S1",
        "confirmation_slot": 4,
        "confirmation_slot_type": "paired_sensitivity",
        "recipe": "balanced_25_25_25_25",
        "lr_schedule": f1["lr_schedule"],
        "batch_size": f1["batch_size"],
        "optimiser": f1["optimiser"],
        "sequence_length": f1["sequence_length"],
        "data_ordering": f1["data_ordering"],
        "predicted_pph_mean": slot1.predicted_pph_mean,
        "predicted_pph_bootstrap_low": slot1.predicted_pph_lo,
        "predicted_pph_bootstrap_high": slot1.predicted_pph_hi,
        "selection_rank": 1,
        "flagged_for_manual_review": slot1.flagged,
        "seed": 23,  # different fresh seed
    })

    # Write schedule CSV
    args.output_schedule.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(schedule_rows[0].keys())
    with args.output_schedule.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in schedule_rows:
            w.writerow(row)
    print(f"Wrote confirmation schedule to {args.output_schedule}")

    # Write full prediction table for audit
    predictions_payload = {
        "n_factorial_runs": int(X.shape[0]),
        "n_bootstrap": args.n_bootstrap,
        "overlap_threshold": args.overlap_threshold,
        "any_flagged_for_manual_review": any_flagged,
        "candidates": [
            {
                "levels": c.levels,
                "factors": to_human_factors(c.levels),
                "predicted_pph_mean": c.predicted_pph_mean,
                "predicted_pph_bootstrap_low": c.predicted_pph_lo,
                "predicted_pph_bootstrap_high": c.predicted_pph_hi,
                "rank": c.rank,
                "flagged": c.flagged,
            }
            for c in sorted(candidates, key=lambda x: x.rank)
        ],
    }
    with args.output_predictions.open("w") as f:
        json.dump(predictions_payload, f, indent=2)
    print(f"Wrote prediction details to {args.output_predictions}")

    if any_flagged:
        print("\n*** WARNING: one or more top-3 confirmation predictions have CIs "
              "overlapping rank-4 by more than 50% of band width. Manual supervisor "
              "review required before launching confirmation runs (§3.4). ***")
        return 2  # non-fatal but distinguishable exit code

    print(f"\nSelected confirmation runs (no manual review required):")
    for row in schedule_rows:
        print(f"  {row['factorial_run_id']}: "
              f"slot={row['confirmation_slot']} type={row['confirmation_slot_type']} "
              f"recipe={row['recipe']} "
              f"pred_PPH={row['predicted_pph_mean']:.4f} "
              f"[{row['predicted_pph_bootstrap_low']:.4f}, "
              f"{row['predicted_pph_bootstrap_high']:.4f}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
