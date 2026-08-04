#!/usr/bin/env python3
"""
/workspace/scripts/phase0_runner.py

LughaGen Training Framework v1.5.1 §6 (Phase 0)
Trains all three tokenizer candidates (Strategies A, B, C) on the Phase 0
training subset and selects the winner by validation fertility.

Procedure (§6.1, §6.2, §6.4):
  1. Sample Phase 0 training corpus from the unified corpus (recipe: full)
  2. For each strategy in [A, B, C]:
     a. Run the strategy's training script
     b. Log fertility on the validation set every 100 steps
     c. Apply §6.4 plateau check at step 1000:
        - If any strategy's slope > 0.5%/100 steps in last 200 steps,
          extend training to step 1500 for ALL strategies (uniform fairness)
        - If still not converged at step 1500, log warning and continue
  3. At step 1500 (or earlier), select winner by validation fertility on
     the 4-language test set
  4. Record winning tokenizer to global_constants.yaml: tokenizer_version.value
  5. Re-run compute_baseline_perplexity.py with the winning tokenizer (§4.2)

Outputs:
  - /workspace/data/processed/tokenizers/strategy_{a,b,c}/  — tokenizer files
  - /workspace/registry/phase0_fertility_log.csv             — per-step fertility
  - /workspace/registry/phase0_winner.yaml                   — winning strategy
  - /workspace/registry/global_constants.yaml.tokenizer_version  — pinned (§7.4)
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

sys.path.insert(0, "/workspace/scripts")
from registry_csv import (
    write_global_constant, get_global_constant, utc_now_iso, REGISTRY_DIR,
)


# Where tokenizers live after training
TOKENIZER_DIR = Path("/workspace/data/processed/tokenizers")
FERTILITY_LOG = REGISTRY_DIR / "phase0_fertility_log.csv"
WINNER_FILE = REGISTRY_DIR / "phase0_winner.yaml"

# §6.4 plateau check parameters
PLATEAU_CHECK_STEP = 1000
PLATEAU_EXTENSION_STEP = 1500
SLOPE_THRESHOLD_PCT_PER_100 = 0.5
SLOPE_WINDOW_STEPS = 200


@dataclass
class StrategyResult:
    """Result of training one tokenizer strategy."""
    strategy: str                       # 'A', 'B', 'C'
    final_step: int
    fertility_history: list[tuple[int, dict[str, float]]] = field(default_factory=list)
    final_fertility_per_lang: dict[str, float] = field(default_factory=dict)
    tokenizer_path: Optional[Path] = None
    converged: bool = False             # met plateau criterion at step 1000

    def slope_per_100_steps(self) -> Optional[float]:
        """
        Compute percent change per 100 steps over the last SLOPE_WINDOW_STEPS
        of training. Negative slope = fertility improving.
        Returns None if insufficient history.
        """
        # Use pooled mean fertility for slope calculation
        if len(self.fertility_history) < 2:
            return None
        recent = [
            (step, sum(f.values()) / max(len(f), 1))
            for step, f in self.fertility_history
            if step >= max(self.fertility_history)[0] - SLOPE_WINDOW_STEPS
        ]
        if len(recent) < 2:
            return None
        first_step, first_fert = recent[0]
        last_step, last_fert = recent[-1]
        if last_step == first_step:
            return None
        pct_change = (last_fert - first_fert) / max(first_fert, 1e-9) * 100
        slope_per_100 = pct_change * 100 / (last_step - first_step)
        return abs(slope_per_100)

    def is_plateau_at_step(self, step: int) -> bool:
        """True if fertility curve is flat (within tolerance) at given step."""
        slope = self.slope_per_100_steps()
        if slope is None:
            return False
        return slope < SLOPE_THRESHOLD_PCT_PER_100


def import_strategy_module(strategy: str):
    """Dynamic import: 'A' -> tokenizer_strategy_a, etc."""
    import importlib
    module_name = f"tokenizer_strategy_{strategy.lower()}"
    return importlib.import_module(module_name)


def run_strategy(
    strategy: str,
    corpus_path: Path,
    val_path: Path,
    output_dir: Path,
    target_step: int,
    vocab_size_extension: int = 8000,
    languages: tuple = ("swahili", "kikuyu", "kamba", "luo"),
) -> StrategyResult:
    """
    Run one tokenizer strategy. Each strategy module exposes:
        train(corpus_path, output_dir, target_step, vocab_size_extension=...) -> Path
        evaluate_fertility(tokenizer_path, val_path, languages) -> dict[lang, float]
    """
    print(f"\n=== Strategy {strategy} ===")
    print(f"  Training to step {target_step}, output to {output_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    mod = import_strategy_module(strategy)
    tokenizer_path = mod.train(
        corpus_path=corpus_path,
        output_dir=output_dir,
        target_step=target_step,
        vocab_size_extension=vocab_size_extension,
    )

    print(f"  Evaluating fertility on {val_path.name}...")
    fertility = mod.evaluate_fertility(
        tokenizer_path=tokenizer_path,
        val_path=val_path,
        languages=languages,
    )
    print(f"  Fertility (final): {fertility}")

    # Read the per-step log written by the strategy
    step_log_path = output_dir / "step_fertility.jsonl"
    history = []
    if step_log_path.exists():
        with step_log_path.open() as f:
            for line in f:
                rec = json.loads(line)
                history.append((rec["step"], rec["fertility"]))

    result = StrategyResult(
        strategy=strategy,
        final_step=target_step,
        fertility_history=history,
        final_fertility_per_lang=fertility,
        tokenizer_path=tokenizer_path,
    )
    result.converged = result.is_plateau_at_step(PLATEAU_CHECK_STEP)
    return result


def select_winner(results: dict[str, StrategyResult]) -> str:
    """
    Select winner by mean per-language fertility (lower = better).
    Rejects fertility < 1.0 as mathematically impossible (broken evaluation).
    """
    means_raw = {
        s: sum(r.final_fertility_per_lang.values()) / len(r.final_fertility_per_lang)
        for s, r in results.items()
        if r.final_fertility_per_lang
    }
    if not means_raw:
        raise RuntimeError("No strategies completed with valid fertility")

    SANITY_MIN = 1.0
    invalid = {s: m for s, m in means_raw.items() if m < SANITY_MIN}
    valid = {s: m for s, m in means_raw.items() if m >= SANITY_MIN}
    print(f"\nMean fertility (raw): {means_raw}")
    if invalid:
        print(f"REJECTED (fertility < {SANITY_MIN}, broken): {invalid}")
    if not valid:
        raise RuntimeError(
            f"All strategies rejected (fertility < {SANITY_MIN}). "
            f"Inspect each tokenizer output manually."
        )
    winner = min(valid, key=valid.get)
    print(f"Winner: Strategy {winner} (fertility={valid[winner]:.4f})")
    return winner


def log_phase0_results(results: dict[str, StrategyResult], winner: str) -> None:
    """Write fertility log CSV + winner YAML."""
    import yaml

    REGISTRY_DIR.mkdir(parents=True, exist_ok=True)

    # CSV: one row per (strategy, step, language)
    with FERTILITY_LOG.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["strategy", "step", "language", "fertility"])
        for s, r in results.items():
            for step, fert_dict in r.fertility_history:
                for lang, fert in fert_dict.items():
                    w.writerow([s, step, lang, fert])

    # YAML: winner summary
    summary = {
        "phase": "Phase0_Tokenizer",
        "selected_at": utc_now_iso(),
        "winner_strategy": winner,
        "winner_tokenizer_path": str(results[winner].tokenizer_path) if results[winner].tokenizer_path else None,
        "winner_final_fertility": results[winner].final_fertility_per_lang,
        "plateau_check": {
            "step": PLATEAU_CHECK_STEP,
            "slope_threshold_pct_per_100": SLOPE_THRESHOLD_PCT_PER_100,
            "extension_step": PLATEAU_EXTENSION_STEP,
        },
        "all_strategies": {
            s: {
                "final_step": r.final_step,
                "converged_at_step_1000": r.converged,
                "final_fertility_per_lang": r.final_fertility_per_lang,
                "tokenizer_path": str(r.tokenizer_path) if r.tokenizer_path else None,
            }
            for s, r in results.items()
        },
    }
    with WINNER_FILE.open("w") as f:
        yaml.safe_dump(summary, f, sort_keys=False, default_flow_style=False)

    # Pin tokenizer version (immutable §7.4)
    write_global_constant(
        "tokenizer_version.value",
        str(results[winner].tokenizer_path) if results[winner].tokenizer_path else f"strategy_{winner}"
    )
    write_global_constant("tokenizer_version.selected_at", utc_now_iso())
    write_global_constant("tokenizer_version.winner_strategy", winner)


def main() -> int:
    parser = argparse.ArgumentParser(description="LughaGen Phase 0 tokenizer runner (§6)")
    parser.add_argument("--corpus-path", type=Path,
                        default=Path("/workspace/data/corpus/splits/train.jsonl"))
    parser.add_argument("--val-path", type=Path,
                        default=Path("/workspace/data/corpus/splits/validation.jsonl"))
    parser.add_argument("--output-dir", type=Path,
                        default=TOKENIZER_DIR)
    parser.add_argument("--vocab-extension", type=int, default=8000,
                        help="New tokens to add (Strategy A extension target)")
    parser.add_argument("--strategies", nargs="+", default=["A", "B", "C"],
                        choices=["A", "B", "C"])
    parser.add_argument("--target-step", type=int, default=PLATEAU_CHECK_STEP,
                        help=f"Initial target step (default: {PLATEAU_CHECK_STEP})")
    parser.add_argument("--max-step", type=int, default=PLATEAU_EXTENSION_STEP,
                        help=f"Maximum step after plateau extension (default: {PLATEAU_EXTENSION_STEP})")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate setup without running tokenizer training")
    args = parser.parse_args()

    if not args.corpus_path.exists():
        print(f"ERROR: corpus not found at {args.corpus_path}", file=sys.stderr)
        return 1

    print(f"=== Phase 0: tokenizer comparison ===")
    print(f"  Corpus     : {args.corpus_path}")
    print(f"  Validation : {args.val_path}")
    print(f"  Output     : {args.output_dir}")
    print(f"  Strategies : {args.strategies}")
    print(f"  Target step: {args.target_step}")

    if args.dry_run:
        print("\nDry-run: validating strategy modules...")
        for s in args.strategies:
            mod = import_strategy_module(s)
            assert hasattr(mod, "train"), f"Strategy {s} missing train()"
            assert hasattr(mod, "evaluate_fertility"), f"Strategy {s} missing evaluate_fertility()"
            print(f"  Strategy {s}: OK")
        return 0

    # ----- Round 1: train each strategy to target_step -----
    results: dict[str, StrategyResult] = {}
    for s in args.strategies:
        out = args.output_dir / f"strategy_{s.lower()}"
        results[s] = run_strategy(
            strategy=s,
            corpus_path=args.corpus_path,
            val_path=args.val_path,
            output_dir=out,
            target_step=args.target_step,
            vocab_size_extension=args.vocab_extension,
        )

    # ----- §6.4 plateau check -----
    print(f"\n=== §6.4 Plateau check at step {args.target_step} ===")
    any_non_converged = False
    for s, r in results.items():
        slope = r.slope_per_100_steps()
        slope_str = f"{slope:.3f}%/100" if slope is not None else "N/A"
        verdict = "converged" if r.converged else "NOT converged"
        print(f"  Strategy {s}: slope={slope_str}  -> {verdict}")
        if not r.converged:
            any_non_converged = True

    # ----- Round 2: extend ALL to max_step if any did not converge -----
    if any_non_converged and args.target_step < args.max_step:
        print(f"\n=== Extending ALL strategies to step {args.max_step} (uniform fairness) ===")
        for s in args.strategies:
            out = args.output_dir / f"strategy_{s.lower()}"
            results[s] = run_strategy(
                strategy=s,
                corpus_path=args.corpus_path,
                val_path=args.val_path,
                output_dir=out,
                target_step=args.max_step,
                vocab_size_extension=args.vocab_extension,
            )
        # Log if still not converged
        for s, r in results.items():
            if not r.is_plateau_at_step(args.max_step):
                print(f"  WARNING: Strategy {s} still not converged at step {args.max_step}")

    # ----- Select winner -----
    winner = select_winner(results)
    log_phase0_results(results, winner)

    print(f"\nPhase 0 complete. Winner: Strategy {winner}")
    print(f"Logged to: {FERTILITY_LOG}, {WINNER_FILE}")
    print(f"\nNext step: re-run compute_baseline_perplexity.py with the new tokenizer (§4.2)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
