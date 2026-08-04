#!/usr/bin/env python3
"""
/workspace/scripts/factorial_matrix.py

LughaGen Training Framework v1.5 §3.1-§3.3
Generates the 2^(5-1) Resolution V fractional factorial matrix with defining
relation E = ABCD, plus seed doubling to produce the 32-run base schedule.

Factor mapping (must match §3.1 exactly):
    A — Learning-Rate Schedule    (-1: cosine,           +1: linear_warmup_constant)
    B — Effective Batch Size      (-1: 128,              +1: 256)
    C — Optimiser                 (-1: AdamW,            +1: Lion)
    D — Sequence Length           (-1: 1024,             +1: 2048)
    E — Data Ordering             (-1: curriculum,       +1: mixed)

Defining relation: E = ABCD (i.e., I = ABCDE)

Usage:
    python factorial_matrix.py --output /workspace/registry/factorial_schedule.csv
    python factorial_matrix.py --output factorial_schedule.csv --seeds 13 17
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


# Factor level mappings (§3.1)
LEVEL_MAP = {
    "A": {-1: "cosine",            +1: "linear_warmup_constant"},
    "B": {-1: 128,                  +1: 256},
    "C": {-1: "AdamW",              +1: "Lion"},
    "D": {-1: 1024,                 +1: 2048},
    "E": {-1: "curriculum",         +1: "mixed"},
}

FACTOR_TO_FIELD = {
    "A": "lr_schedule",
    "B": "batch_size",
    "C": "optimiser",
    "D": "sequence_length",
    "E": "data_ordering",
}

# Default seeds (§3.3: 2 random seeds per configuration)
DEFAULT_SEEDS = (42, 137)


@dataclass(frozen=True)
class FactorialRun:
    """A single row of the experiment schedule."""
    factorial_run_id: str       # e.g. RUN-P2-C04-S1
    config_id: int              # 1..16, the row of the half-fraction
    seed: int
    A: int                      # -1 or +1
    B: int
    C: int
    D: int
    E: int                      # always derived as A*B*C*D
    lr_schedule: str
    batch_size: int
    optimiser: str
    sequence_length: int
    data_ordering: str

    def as_dict(self) -> dict:
        return {
            "factorial_run_id": self.factorial_run_id,
            "config_id": self.config_id,
            "seed": self.seed,
            "A_level": self.A,
            "B_level": self.B,
            "C_level": self.C,
            "D_level": self.D,
            "E_level": self.E,
            "lr_schedule": self.lr_schedule,
            "batch_size": self.batch_size,
            "optimiser": self.optimiser,
            "sequence_length": self.sequence_length,
            "data_ordering": self.data_ordering,
        }


def generate_half_fraction() -> list[tuple[int, int, int, int, int]]:
    """
    Enumerate the 16 rows of the 2^(5-1) Resolution V design under E = ABCD.

    Returns: list of (A, B, C, D, E) tuples with values in {-1, +1}.
    """
    rows = []
    for A, B, C, D in itertools.product([-1, +1], repeat=4):
        E = A * B * C * D
        rows.append((A, B, C, D, E))
    assert len(rows) == 16, "Half-fraction must have exactly 16 rows"
    return rows


def alias_structure() -> dict[str, list[str]]:
    """
    Return the alias structure for E = ABCD (identity I = ABCDE).

    At Resolution V:
      - Main effects are aliased with 4-factor interactions (negligible)
      - Two-factor interactions are aliased with 3-factor interactions (negligible)
      - Main effects and two-factor interactions are clean
    """
    return {
        "I":   ["ABCDE"],
        "A":   ["BCDE"],
        "B":   ["ACDE"],
        "C":   ["ABDE"],
        "D":   ["ABCE"],
        "E":   ["ABCD"],
        "AB":  ["CDE"],
        "AC":  ["BDE"],
        "AD":  ["BCE"],
        "AE":  ["BCD"],
        "BC":  ["ADE"],
        "BD":  ["ACE"],
        "BE":  ["ACD"],
        "CD":  ["ABE"],
        "CE":  ["ABD"],
        "DE":  ["ABC"],
    }


def build_schedule(seeds: Iterable[int] = DEFAULT_SEEDS) -> list[FactorialRun]:
    """
    Build the full 32-run base schedule (16 configurations × 2 seeds).

    Confirmation runs (§3.4) are NOT included — they are appended after Phase 2
    completes by select_confirmation_runs.py.
    """
    seeds = list(seeds)
    rows = generate_half_fraction()
    schedule: list[FactorialRun] = []

    for config_id, (A, B, C, D, E) in enumerate(rows, start=1):
        for seed_idx, seed in enumerate(seeds, start=1):
            run_id = f"RUN-P2-C{config_id:02d}-S{seed_idx}"
            schedule.append(FactorialRun(
                factorial_run_id=run_id,
                config_id=config_id,
                seed=seed,
                A=A, B=B, C=C, D=D, E=E,
                lr_schedule=LEVEL_MAP["A"][A],
                batch_size=LEVEL_MAP["B"][B],
                optimiser=LEVEL_MAP["C"][C],
                sequence_length=LEVEL_MAP["D"][D],
                data_ordering=LEVEL_MAP["E"][E],
            ))

    assert len(schedule) == 32, f"Schedule must have exactly 32 rows; got {len(schedule)}"

    # Verify uniqueness on (config_id, seed)
    keys = [(r.config_id, r.seed) for r in schedule]
    assert len(set(keys)) == len(keys), "Duplicate (config_id, seed) detected"

    return schedule


def write_schedule_csv(schedule: list[FactorialRun], output_path: Path) -> None:
    """Write the schedule to CSV with stable column ordering."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(schedule[0].as_dict().keys())
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for run in schedule:
            writer.writerow(run.as_dict())


def write_alias_log(output_path: Path) -> None:
    """Write the alias structure to a JSON log file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "defining_relation": "E = ABCD",
        "identity": "I = ABCDE",
        "resolution": "V",
        "factor_mapping": {
            k: {"field": FACTOR_TO_FIELD[k], "levels": LEVEL_MAP[k]}
            for k in "ABCDE"
        },
        "alias_structure": alias_structure(),
        "note": (
            "Main effects aliased with 4-factor interactions (negligible). "
            "Two-factor interactions aliased with 3-factor interactions (negligible). "
            "Main effects and two-factor interactions are clean."
        ),
    }
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate the LughaGen 2^(5-1) factorial schedule")
    parser.add_argument(
        "--output", type=Path,
        default=Path("/workspace/registry/factorial_schedule.csv"),
        help="Path to write the 32-run schedule (CSV).",
    )
    parser.add_argument(
        "--alias-log", type=Path,
        default=Path("/workspace/registry/factorial_alias_structure.json"),
        help="Path to write the alias-structure log (JSON).",
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS),
        help="Two integer seeds for replication (default: 42 137).",
    )
    args = parser.parse_args()

    if len(args.seeds) != 2:
        print(f"Error: exactly 2 seeds required; got {len(args.seeds)}", file=sys.stderr)
        return 1

    schedule = build_schedule(seeds=args.seeds)
    write_schedule_csv(schedule, args.output)
    write_alias_log(args.alias_log)

    print(f"Wrote {len(schedule)} factorial runs to {args.output}")
    print(f"Wrote alias structure to {args.alias_log}")

    # Print a concise verification table
    print("\nFirst 4 rows of schedule:")
    for run in schedule[:4]:
        print(f"  {run.factorial_run_id}: "
              f"LR={run.lr_schedule}, batch={run.batch_size}, "
              f"opt={run.optimiser}, seq={run.sequence_length}, order={run.data_ordering}")

    # Sanity: every E value equals A*B*C*D
    for run in schedule:
        assert run.E == run.A * run.B * run.C * run.D, \
            f"Defining relation violated at {run.factorial_run_id}"

    print("\nDefining relation E = ABCD verified for all 32 rows.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
