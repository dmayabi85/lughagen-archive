#!/usr/bin/env python3
"""
/workspace/scripts/compute_tost_delta_v2.py
LughaGen Training Framework v1.5 §10.3.4 + §7.4 — RETROACTIVE COMPUTATION VARIANT

Modifies compute_tost_delta.py (original) to:
  1. Filter Phase 2 runs by regex pattern RUN-P2-C{cell}-S{seed} (exclude
     microbenchmarks, baselines, PRE_RESTART_v1 runs).
  2. Additionally filter by val_ppl < 100 (exclude infrastructure failures
     that completed with garbage output).
  3. Accept --baseline flag for choosing between:
     - v2-per-method (13963.03, spec §4.2 wrapped-eval convention)
     - v3-common     (8364.21, direct-eval commensurable convention)
  4. Write to distinct global_constants.yaml keys per baseline choice.
  5. Compute PPH from val_ppl and gpu_hours_consumed directly (not from
     the recorded pph_score column, which may reflect a different baseline
     than the one selected here).

The original compute_tost_delta.py is preserved unchanged as the reference
implementation from framework spec v1.5.1.

Provenance disclosure:
This script was authored 2026-07-28, after Phase 3 completion, as part of a
retroactive TOST equivalence-band pre-registration. The pre-registration
compliance gap and the sensitivity-analysis rationale are documented in the
thesis Chapter 4 (Methods) and Chapter 6 (Limitations) per §10.3.3.
"""
from __future__ import annotations
import argparse
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, "/workspace/scripts")
from registry_csv import (
    read_runs_by_phase,
    write_global_constant,
    get_global_constant,
    utc_now_iso,
)

CELL_PATTERN = re.compile(r'^RUN-P2-C\d+-S\d+$')
PHASE2_NAME = "Phase2_Factorial"

BASELINES = {
    "v2-per-method": {
        "value": 13963.03,
        "description": "V2 baseline (QLoRA-wrapped model via trainer.evaluate()); framework spec §4.2 original convention",
        "key_suffix": "option1_v2_per_method",
    },
    "v3-common": {
        "value": 8364.21,
        "description": "V3 baseline (direct forward pass on untuned model with tokenizer resize + remap); commensurable across eval pathways",
        "key_suffix": "option2_v3_common",
    },
}


def compute_filtered_phase2_stats(baseline: float) -> tuple[float, int, list[dict]]:
    """
    Read Phase 2 runs filtered to legitimate factorial cells; compute mean PPH.
    Returns (mean_pph, n_runs, per_cell_records).
    """
    runs = read_runs_by_phase(PHASE2_NAME)
    legit = []
    for r in runs:
        rid = r.get("factorial_run_id", "")
        if not CELL_PATTERN.match(rid):
            continue
        vp_raw = r.get("metrics.validation_perplexity", "")
        gh_raw = r.get("gpu_hours_consumed", "")
        if not vp_raw or not gh_raw:
            continue
        try:
            vp = float(vp_raw)
            gh = float(gh_raw)
            if vp >= 100:  # exclude broken runs (baseline is ~8k, well-trained ~25)
                continue
            pph = (baseline - vp) / gh
            legit.append({
                "run_id": rid,
                "val_ppl": vp,
                "gpu_hours": gh,
                "pph": pph,
            })
        except ValueError:
            continue

    if not legit:
        raise RuntimeError(f"No legitimate Phase 2 cells found for baseline {baseline}")

    mean_pph = statistics.fmean(c["pph"] for c in legit)
    return mean_pph, len(legit), legit


def main() -> int:
    p = argparse.ArgumentParser(description="Retroactive TOST δ (§10.3.4 + §7.4)")
    p.add_argument("--baseline", required=True, choices=list(BASELINES.keys()),
                   help="Baseline convention for PPH computation.")
    p.add_argument("--fraction", type=float, default=0.05,
                   help="Fraction of mean PPH to use as δ (default 0.05).")
    p.add_argument("--dry-run", action="store_true",
                   help="Compute and print δ without writing.")
    args = p.parse_args()

    baseline_cfg = BASELINES[args.baseline]
    baseline_val = baseline_cfg["value"]
    key_root = f"tost_delta_pph_v2.{baseline_cfg['key_suffix']}"

    # Check for prior write of this specific variant
    existing = get_global_constant(f"{key_root}.value")
    if existing is not None and not args.dry_run:
        print(f"{key_root}.value already set to {existing}. §7.4 forbids overwriting.")
        return 0

    print(f"Reading Phase 2 runs with baseline={args.baseline} ({baseline_val})...")
    mean_pph, n, records = compute_filtered_phase2_stats(baseline_val)

    print(f"  n_cells: {n}")
    print(f"  mean_pph: {mean_pph:.6f}")

    abs_mean = abs(mean_pph)
    delta = args.fraction * abs_mean
    derivation = f"{args.fraction * 100:.0f}_percent_of_|phase2_mean_pph|_filtered"
    if mean_pph < 0:
        print(f"  WARNING: mean_pph negative; using |mean_pph| per §10.3.4.")

    print(f"  delta ({args.fraction * 100:.1f}% of |mean|): {delta:.6f}")
    print(f"  sensitivity δ:")
    for frac in (0.03, 0.07, 0.10):
        print(f"    {frac*100:.0f}%: {frac * abs_mean:.4f}")

    if args.dry_run:
        print("\n[dry-run] Not writing to global_constants.yaml.")
        return 0

    # Write primary δ
    write_global_constant(f"{key_root}.value", float(delta))
    write_global_constant(f"{key_root}.computed_at", utc_now_iso())
    write_global_constant(f"{key_root}.derivation", derivation)
    write_global_constant(f"{key_root}.baseline_used", float(baseline_val))
    write_global_constant(f"{key_root}.baseline_description", baseline_cfg["description"])
    write_global_constant(f"{key_root}.phase2_cells_included", n)
    write_global_constant(f"{key_root}.phase2_mean_pph", float(mean_pph))
    write_global_constant(f"{key_root}.script", "compute_tost_delta_v2.py")
    write_global_constant(f"{key_root}.retroactive", True)
    write_global_constant(f"{key_root}.compliance_note",
        "Framework spec §10.3.4 required pre-Phase-3 computation. "
        "This value was computed retroactively after Phase 3 completion "
        "due to preflight validation workflow gap. Documented in thesis "
        "Chapter 4 (methods) and Chapter 6 (limitations)."
    )

    # Also write sensitivity δ
    for frac in (0.03, 0.07, 0.10):
        write_global_constant(f"{key_root}.sensitivity.pct_{int(frac*100)}", frac * abs_mean)

    print(f"\n[done] Written to /workspace/registry/global_constants.yaml")
    print(f"  {key_root}.value = {delta}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
