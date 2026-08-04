#!/usr/bin/env python3
"""
/workspace/scripts/compute_tost_delta.py

LughaGen Training Framework v1.5 §10.3.4 + §7.4
Pre-registers the TOST equivalence band δ as 5% of Phase 2 mean PPH. MUST run
AFTER all 36 Phase 2 runs complete and BEFORE any Phase 3 run launches.

The registry's immutable-write enforcement guarantees δ cannot be modified
once written — this eliminates the post-hoc HARKing objection (§10.3.4
rationale).

Procedure:
  1. Read all Phase 2 runs from /workspace/registry/runs.csv
  2. Verify exactly 36 runs (32 factorial + 4 confirmation)
  3. Compute mean PPH across all 36
  4. Set delta = 0.05 * abs(mean_pph)  [absolute value handles negative case]
  5. Write to global_constants.yaml under tost_delta_pph.*
  6. Refuse to overwrite (§7.4)
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, "/workspace/scripts")
from registry_csv import (
    read_runs_by_phase,
    write_global_constant,
    get_global_constant,
    utc_now_iso,
)


EXPECTED_PHASE2_RUNS = 36
PHASE2_NAME = "Phase2_Factorial"


def compute_phase2_mean_pph(allow_partial: bool = False) -> tuple[float, int]:
    """
    Read Phase 2 runs from runs.csv and compute mean PPH.

    Returns (mean_pph, n_runs).
    Raises if n_runs != 36 unless allow_partial is True.
    """
    runs = read_runs_by_phase(PHASE2_NAME)
    n = len(runs)

    if n != EXPECTED_PHASE2_RUNS and not allow_partial:
        raise RuntimeError(
            f"Expected {EXPECTED_PHASE2_RUNS} Phase 2 runs in registry; "
            f"found {n}. Refusing to compute δ until Phase 2 is complete."
        )

    pph_values: list[float] = []
    for r in runs:
        v = r.get("metrics.pph_score") or ""
        if v == "":
            continue
        try:
            pph_values.append(float(v))
        except ValueError:
            continue

    if not pph_values:
        raise RuntimeError("No valid pph_score values found in Phase 2 runs.")

    return statistics.fmean(pph_values), n


def main() -> int:
    parser = argparse.ArgumentParser(description="Pre-register TOST δ (§10.3.4)")
    parser.add_argument(
        "--fraction", type=float, default=0.05,
        help="Fraction of Phase 2 mean PPH to use as δ (default: 0.05 = 5%%).",
    )
    parser.add_argument(
        "--allow-partial", action="store_true",
        help="Allow computation with fewer than 36 Phase 2 runs (TESTING ONLY).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Compute and print δ without writing to global_constants.yaml.",
    )
    args = parser.parse_args()

    # Check for prior write
    existing = get_global_constant("tost_delta_pph.value")
    if existing is not None and not args.dry_run:
        print(f"tost_delta_pph.value is already set to {existing}. "
              f"§7.4 forbids overwriting; nothing to do.")
        return 0

    print(f"Reading Phase 2 runs from registry...")
    mean_pph, n_runs = compute_phase2_mean_pph(allow_partial=args.allow_partial)
    print(f"  n_runs = {n_runs}")
    print(f"  mean_pph = {mean_pph:.6f}")

    # Use absolute value so a negative mean (worst case: all configs degrade base)
    # produces a positive δ. §10.3.4 documents this behaviour.
    abs_mean = abs(mean_pph)
    delta = args.fraction * abs_mean

    if mean_pph < 0:
        derivation = "5_percent_of_absolute_phase2_mean_pph"
        print(f"  WARNING: mean_pph is negative; δ derived from absolute value.")
        print(f"           Phase 3 inferential framing shifts to 'which method "
              f"degrades least per GPU-hour' (§10.3.4).")
    else:
        derivation = "5_percent_of_phase2_mean_pph"

    print(f"  delta = {args.fraction * 100:.1f}% × |mean_pph| = {delta:.6f}")

    if args.dry_run:
        print("\nDry run; not writing to global_constants.yaml.")
        return 0

    # Write to global_constants.yaml (immutable-write)
    write_global_constant("tost_delta_pph.value", float(delta))
    write_global_constant("tost_delta_pph.computed_at", utc_now_iso())
    write_global_constant("tost_delta_pph.derivation", derivation)
    write_global_constant("tost_delta_pph.phase2_runs_included", n_runs)
    write_global_constant("tost_delta_pph.phase2_mean_pph", float(mean_pph))

    print(f"\nWritten to /workspace/registry/global_constants.yaml")
    print(f"  tost_delta_pph.value = {delta}")
    print(f"\nPhase 3 may now launch (preflight will verify δ is recorded).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
