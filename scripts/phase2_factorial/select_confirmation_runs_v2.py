#!/usr/bin/env python3
"""
/workspace/scripts/select_confirmation_runs_v2.py

LughaGen Training Framework v1.5 §3.4 — REINTERPRETED for 2^4 full factorial

Background
----------
Spec §3.4 specified 4 confirmation runs after Phase 2's factorial screen,
in a context where Phase 2 was a 2^(5-1) Resolution V half-fraction. In that
context, "confirmation" meant: predict PPH for the *untested* half of the
design space using the fitted model, then run the top-3 predictions to
validate them, plus a 4th slot under the balanced_25_25_25_25 recipe.

After the pre-Phase-2 implementation audit, the design was reduced to a 2^4
full factorial (Section 6.3 of the proposal; Spec §3.1 reinterpretation).
In a *full* factorial, every configuration is tested — the prediction step
has no untested points to predict. The original confirmation-run rationale
does not apply.

This v2 script reinterprets §3.4 as:
  - Slots 1-3: rerun the top-3 highest-PPH *observed* configurations with
               new seeds (operationally analogous to confirmation runs;
               provides additional seed replication on the most influential
               configurations).
  - Slot 4:    rerun the slot-1 config with the balanced_25_25_25_25 recipe
               (preserves the §3.4 paired-sensitivity intent).

Confidence:
  - PPH per config is averaged over the 2 Phase 2 seeds (42, 137).
  - A bootstrap CI is computed via resampling the 32 row dataset; if the
    top-3 means' CIs overlap with rank-4 by more than --overlap-threshold,
    the selection is flagged for supervisor review.

Outputs (same schema as original):
  - /workspace/registry/confirmation_schedule.csv  (4 rows, in run order)
  - /workspace/registry/confirmation_predictions.json  (full ranking + CIs)

Usage
-----
  python3 /workspace/scripts/select_confirmation_runs_v2.py [--dry-run]

Run this AFTER all 32 Phase 2 cells complete. Will error if fewer than 32
cells with valid PPH are present.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Constants — match factorial_matrix.py LEVEL_MAP / FACTOR_TO_FIELD
# ---------------------------------------------------------------------------

REGISTRY_PATH = Path("/workspace/registry/runs.csv")
PHASE_NAME = "Phase2_Factorial"

# Factor mapping: registry column name -> (level_neg1_value, level_pos1_value)
FACTOR_LEVELS = {
    "factors.lr_schedule":     ("cosine", "linear_warmup_constant"),
    "factors.batch_size":      ("128", "256"),
    "factors.optimiser":       ("AdamW", "Lion"),
    "factors.sequence_length": ("1024", "2048"),
}
# Short codes for printout
FACTOR_CODES = ["A", "B", "C", "D"]
FACTOR_COLS = list(FACTOR_LEVELS.keys())

# Confirmation seeds — DIFFERENT from Phase 2's (42, 137)
# Picked deterministically; see spec §3.4 note about reproducibility
CONFIRMATION_SEEDS = [314]  # one new seed per slot

# Slot-4 recipe (Spec §5.4)
SLOT4_RECIPE = "balanced_25_25_25_25"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_phase2_rows(registry_path: Path = REGISTRY_PATH) -> list[dict]:
    """Read Phase 2 factorial rows from runs.csv.

    Returns rows that:
      - have factorial_run_id starting with 'RUN-P2-C'
      - have a non-empty metrics.validation_perplexity (complete cells only)
      - have a non-empty metrics.pph_score
    """
    if not registry_path.exists():
        raise FileNotFoundError(f"Registry not found: {registry_path}")

    with open(registry_path) as f:
        reader = csv.DictReader(f)
        all_rows = list(reader)

    phase2 = []
    for r in all_rows:
        rid = r.get("factorial_run_id", "") or ""
        if not rid.startswith("RUN-P2-C"):
            continue
        if not r.get("metrics.validation_perplexity"):
            continue
        if not r.get("metrics.pph_score"):
            continue
        phase2.append(r)

    return phase2


def encode_factors(row: dict) -> dict[str, int]:
    """Convert a registry row to coded ±1 levels for the 4 factors.

    Returns dict like {"A": -1, "B": +1, "C": -1, "D": +1}.
    Raises ValueError if any factor value is unrecognised.
    """
    out = {}
    for code, col in zip(FACTOR_CODES, FACTOR_COLS):
        val = str(row.get(col, "")).strip()
        levels = FACTOR_LEVELS[col]
        if val == levels[0]:
            out[code] = -1
        elif val == levels[1]:
            out[code] = +1
        else:
            raise ValueError(
                f"Row {row.get('factorial_run_id', '?')}: "
                f"unrecognised level {val!r} for {col}; "
                f"expected one of {levels}"
            )
    return out


def extract_config_id(row: dict) -> Optional[int]:
    """Get config_id; fall back to parsing factorial_run_id 'RUN-P2-CNN-Sn'."""
    cid = row.get("config_id", "")
    if cid and str(cid).strip():
        try:
            return int(cid)
        except ValueError:
            pass
    rid = row.get("factorial_run_id", "")
    # Parse 'RUN-P2-C04-S2' -> 4
    try:
        return int(rid.split("-C")[1].split("-S")[0])
    except (IndexError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

@dataclass
class ConfigSummary:
    """Summary of one factorial configuration across its seeds."""
    config_id: int
    levels: dict[str, int]            # {"A": -1, ...}
    factor_strings: dict[str, str]    # human-readable {"lr_schedule": "cosine", ...}
    n_seeds: int
    pph_values: list[float]
    pph_mean: float
    pph_std: float                    # 0.0 if n_seeds < 2
    val_pph_values: list[float]
    gpu_hours_values: list[float]
    cell_ids: list[str]

    def as_dict(self) -> dict:
        d = asdict(self)
        # Convert levels to ints for clean JSON
        d["levels"] = {k: int(v) for k, v in d["levels"].items()}
        return d


def aggregate_by_config(rows: list[dict]) -> list[ConfigSummary]:
    """Group rows by config_id; compute per-config PPH summary statistics."""
    by_config: dict[int, list[dict]] = {}
    for r in rows:
        cid = extract_config_id(r)
        if cid is None:
            print(f"WARN: could not extract config_id from {r.get('factorial_run_id')!r}; skipping",
                  file=sys.stderr)
            continue
        by_config.setdefault(cid, []).append(r)

    summaries = []
    for cid in sorted(by_config.keys()):
        cell_rows = by_config[cid]
        levels = encode_factors(cell_rows[0])
        # Human-readable factor strings (from first row; should be identical across seeds)
        factor_strings = {}
        for code, col in zip(FACTOR_CODES, FACTOR_COLS):
            short_name = col.split(".", 1)[1]  # "factors.lr_schedule" -> "lr_schedule"
            factor_strings[short_name] = str(cell_rows[0].get(col, "")).strip()

        pph_values = [float(r["metrics.pph_score"]) for r in cell_rows]
        val_pph_values = [float(r["metrics.validation_perplexity"]) for r in cell_rows]
        gpu_hours_values = [float(r.get("gpu_hours_consumed", 0) or 0) for r in cell_rows]
        cell_ids = [r["factorial_run_id"] for r in cell_rows]

        summaries.append(ConfigSummary(
            config_id=cid,
            levels=levels,
            factor_strings=factor_strings,
            n_seeds=len(cell_rows),
            pph_values=pph_values,
            pph_mean=float(np.mean(pph_values)),
            pph_std=float(np.std(pph_values, ddof=1)) if len(pph_values) >= 2 else 0.0,
            val_pph_values=val_pph_values,
            gpu_hours_values=gpu_hours_values,
            cell_ids=cell_ids,
        ))

    return summaries


# ---------------------------------------------------------------------------
# Bootstrap CIs on observed means
# ---------------------------------------------------------------------------

def bootstrap_config_pph(
    summaries: list[ConfigSummary],
    n_boot: int = 1000,
    seed: int = 20260518,
) -> dict[int, tuple[float, float, float]]:
    """For each config, bootstrap-resample its PPH values and compute (mean, lo95, hi95).

    Resampling is per-config (within-config seed resampling). For 2-seed configs,
    bootstrap is necessarily limited — there are only 2^n_boot unique resamples
    possible — but the CI still captures the seed variance qualitatively.

    Returns: {config_id: (mean, ci_lo, ci_hi)}
    """
    rng = np.random.default_rng(seed)
    result = {}
    for s in summaries:
        if s.n_seeds < 2:
            # No bootstrap possible; CI = point value
            result[s.config_id] = (s.pph_mean, s.pph_mean, s.pph_mean)
            continue
        values = np.array(s.pph_values, dtype=float)
        # Resample with replacement n_boot times
        idx = rng.integers(0, len(values), size=(n_boot, len(values)))
        boot_means = values[idx].mean(axis=1)
        lo = float(np.percentile(boot_means, 2.5))
        hi = float(np.percentile(boot_means, 97.5))
        result[s.config_id] = (s.pph_mean, lo, hi)
    return result


def overlap_fraction(a_lo: float, a_hi: float, b_lo: float, b_hi: float) -> float:
    """Fraction of the smaller CI's width that is covered by the overlap with the larger CI."""
    overlap_lo = max(a_lo, b_lo)
    overlap_hi = min(a_hi, b_hi)
    overlap = max(0.0, overlap_hi - overlap_lo)
    smaller_width = min(a_hi - a_lo, b_hi - b_lo)
    if smaller_width <= 0:
        return 1.0 if overlap > 0 else 0.0
    return overlap / smaller_width


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------

def select_top_three(
    summaries: list[ConfigSummary],
    bootstrap_cis: dict[int, tuple[float, float, float]],
    overlap_threshold: float = 0.5,
) -> tuple[list[ConfigSummary], list[str]]:
    """Pick top-3 by mean PPH. Return (selected, flags).

    Flags is a list of warning strings; non-empty means manual review is recommended.
    """
    # Rank by mean PPH descending
    ranked = sorted(summaries, key=lambda s: -s.pph_mean)
    if len(ranked) < 4:
        raise ValueError(f"Need at least 4 configs to rank top-3 + check rank-4; got {len(ranked)}")

    top_three = ranked[:3]
    rank_four = ranked[3]
    flags = []

    # Check CI overlap between each of top-3 and rank-4
    _, r4_lo, r4_hi = bootstrap_cis[rank_four.config_id]
    for rank_i, cfg in enumerate(top_three, start=1):
        _, lo, hi = bootstrap_cis[cfg.config_id]
        ov = overlap_fraction(lo, hi, r4_lo, r4_hi)
        if ov > overlap_threshold:
            flags.append(
                f"Rank-{rank_i} (config_id={cfg.config_id}) CI overlaps rank-4 "
                f"(config_id={rank_four.config_id}) by {ov:.1%} > {overlap_threshold:.0%} threshold. "
                f"Selection may be ambiguous; consider manual review."
            )

    return top_three, flags


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def build_schedule(top_three: list[ConfigSummary]) -> list[dict]:
    """Build the 4-row confirmation schedule.

    Slots 1-3: replicate top-3 configs with a new seed under 'full' recipe.
    Slot 4:    slot-1 config but with balanced_25_25_25_25 recipe (§3.4 paired sensitivity).
    """
    schedule = []
    base_seed = CONFIRMATION_SEEDS[0]
    for slot_i, cfg in enumerate(top_three, start=1):
        schedule.append({
            "slot": slot_i,
            "factorial_run_id": f"RUN-P2-CONF-{slot_i:02d}",
            "source_config_id": cfg.config_id,
            "seed": base_seed,
            "recipe": "full",
            "lr_schedule": cfg.factor_strings["lr_schedule"],
            "batch_size": cfg.factor_strings["batch_size"],
            "optimiser": cfg.factor_strings["optimiser"],
            "sequence_length": cfg.factor_strings["sequence_length"],
            "data_ordering": "mixed",  # placebo factor; held constant per §6.3
            "observed_pph_mean": f"{cfg.pph_mean:.4f}",
            "observed_pph_n_seeds": cfg.n_seeds,
        })

    # Slot 4: paired sensitivity for slot-1 config under different recipe
    slot1_cfg = top_three[0]
    schedule.append({
        "slot": 4,
        "factorial_run_id": f"RUN-P2-CONF-04",
        "source_config_id": slot1_cfg.config_id,
        "seed": base_seed,
        "recipe": SLOT4_RECIPE,
        "lr_schedule": slot1_cfg.factor_strings["lr_schedule"],
        "batch_size": slot1_cfg.factor_strings["batch_size"],
        "optimiser": slot1_cfg.factor_strings["optimiser"],
        "sequence_length": slot1_cfg.factor_strings["sequence_length"],
        "data_ordering": "mixed",
        "observed_pph_mean": f"{slot1_cfg.pph_mean:.4f}",
        "observed_pph_n_seeds": slot1_cfg.n_seeds,
    })

    return schedule


def write_schedule(schedule: list[dict], path: Path) -> None:
    fields = list(schedule[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in schedule:
            w.writerow(row)


def write_predictions(
    summaries: list[ConfigSummary],
    bootstrap_cis: dict[int, tuple[float, float, float]],
    flags: list[str],
    path: Path,
) -> None:
    ranked = sorted(summaries, key=lambda s: -s.pph_mean)
    payload = {
        "method": "Option B — replicate top observed configs (full factorial)",
        "n_configs": len(summaries),
        "n_total_rows": sum(s.n_seeds for s in summaries),
        "flags": flags,
        "rankings": [
            {
                **s.as_dict(),
                "bootstrap_ci_lo": bootstrap_cis[s.config_id][1],
                "bootstrap_ci_hi": bootstrap_cis[s.config_id][2],
            }
            for s in ranked
        ],
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Select Phase 2 confirmation runs (§3.4 v2 — replication interpretation)"
    )
    ap.add_argument("--registry", type=Path, default=REGISTRY_PATH,
                    help="Path to runs.csv (default: %(default)s)")
    ap.add_argument("--output-schedule", type=Path,
                    default=Path("/workspace/registry/confirmation_schedule.csv"))
    ap.add_argument("--output-predictions", type=Path,
                    default=Path("/workspace/registry/confirmation_predictions.json"))
    ap.add_argument("--n-bootstrap", type=int, default=1000)
    ap.add_argument("--overlap-threshold", type=float, default=0.5,
                    help="CI overlap between rank-i (i<=3) and rank-4 that triggers manual review")
    ap.add_argument("--seed", type=int, default=20260518)
    ap.add_argument("--min-cells", type=int, default=32,
                    help="Refuse to run if fewer than this many Phase 2 rows are complete")
    ap.add_argument("--dry-run", action="store_true",
                    help="Compute everything, print summary, but don't write output files")
    args = ap.parse_args()

    # Load
    rows = load_phase2_rows(args.registry)
    print(f"Loaded {len(rows)} Phase 2 rows from {args.registry}")
    if len(rows) < args.min_cells:
        print(f"ERROR: only {len(rows)}/{args.min_cells} cells complete; aborting.",
              file=sys.stderr)
        print("       (Use --min-cells=N to override for testing on partial data.)",
              file=sys.stderr)
        return 1

    # Aggregate
    summaries = aggregate_by_config(rows)
    print(f"Aggregated to {len(summaries)} unique configurations")
    if len(summaries) < 4:
        print(f"ERROR: only {len(summaries)} unique configs; need at least 4.",
              file=sys.stderr)
        return 1

    # Bootstrap CIs
    print(f"Running {args.n_bootstrap} bootstrap resamples per config...")
    bootstrap_cis = bootstrap_config_pph(
        summaries, n_boot=args.n_bootstrap, seed=args.seed,
    )

    # Select top-3 + slot-4
    top_three, flags = select_top_three(
        summaries, bootstrap_cis, overlap_threshold=args.overlap_threshold,
    )

    # Print ranking
    print("\n" + "=" * 76)
    print("RANKING (all configs by mean PPH, descending)")
    print("=" * 76)
    print(f"{'Rank':<5} {'cfg':<5} {'A':>3} {'B':>3} {'C':>3} {'D':>3} "
          f"{'mean PPH':>12} {'CI_lo':>10} {'CI_hi':>10} {'n':>3}")
    ranked = sorted(summaries, key=lambda s: -s.pph_mean)
    for rank_i, s in enumerate(ranked, start=1):
        mean, lo, hi = bootstrap_cis[s.config_id]
        marker = "  ★" if rank_i <= 3 else ("  ◇" if rank_i == 4 else "")
        print(f"{rank_i:<5} {s.config_id:<5} "
              f"{s.levels['A']:>+3d} {s.levels['B']:>+3d} {s.levels['C']:>+3d} {s.levels['D']:>+3d} "
              f"{mean:>12.4f} {lo:>10.4f} {hi:>10.4f} {s.n_seeds:>3}{marker}")

    if flags:
        print("\n*** SELECTION FLAGS — supervisor review recommended ***")
        for f in flags:
            print(f"  ! {f}")
    else:
        print("\nNo selection ambiguity flags raised.")

    # Build schedule
    schedule = build_schedule(top_three)
    print("\n" + "=" * 76)
    print("CONFIRMATION RUN SCHEDULE")
    print("=" * 76)
    for row in schedule:
        print(f"  Slot {row['slot']}: {row['factorial_run_id']} "
              f"(from config {row['source_config_id']}, observed PPH={row['observed_pph_mean']})")
        print(f"           {row['lr_schedule']}, batch={row['batch_size']}, "
              f"{row['optimiser']}, seq={row['sequence_length']}, recipe={row['recipe']}, "
              f"seed={row['seed']}")

    if args.dry_run:
        print("\nDRY RUN — no files written.")
        return 0

    # Write outputs
    args.output_schedule.parent.mkdir(parents=True, exist_ok=True)
    write_schedule(schedule, args.output_schedule)
    print(f"\nWrote confirmation schedule to {args.output_schedule}")
    write_predictions(summaries, bootstrap_cis, flags, args.output_predictions)
    print(f"Wrote prediction details to {args.output_predictions}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
