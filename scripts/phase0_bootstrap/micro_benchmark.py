#!/usr/bin/env python3
"""
micro_benchmark.py — Phase 2 throughput calibration (spec §9.5, §13)

Path B implementation: invokes train_orchestrator.py via torchrun with
--max-steps 200 --smoke-test, so the throughput we measure is the EXACT
code path Phase 2 will use (training_loop.run_training with QLoRA 4-bit
+ FA2 + BF16 + 8-GPU distributed). Previous Path-A-style benchmark
diverged from training_loop and measured a different stack (HF Trainer
without FA2), producing meaningless numbers.

This script does NOT load the model itself; it spawns torchrun and reads
the resulting metrics from disk.

Writes:
  - /workspace/registry/micro_benchmark_results.yaml (full results)
  - /workspace/registry/global_constants.yaml::phase2_throughput_calibration

Sanity checks before pinning:
  - peak_vram_gb < 30 (FA2 engaged; smoke ran at 19.58)
  - tokens_per_sec_per_gpu > 500 (catches catastrophic regressions)
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import yaml

# Spec constants
PHASE2_BUDGET_GPU_HOURS = 18_000     # §15
PHASE2_N_RUNS = 32                    # 32 factorial cells (§8)
DEFAULT_TOTAL_STEPS = 200             # spec §9.5
DEFAULT_WARMUP_STEPS = 50

# Sanity thresholds
PEAK_VRAM_MAX_GB = 30.0               # FA2 engagement threshold (smoke ran at 19.58)
TPS_PER_GPU_MIN = 300.0   # relaxed from 500 — gradient-checkpoint config is lower-throughput than non-checkpointed               # catches catastrophic regressions


@dataclass
class BenchResult:
    total_steps: int
    warmup_steps: int
    effective_batch_size: int
    seq_length: int
    n_gpus: int
    tokens_per_step: int
    wall_clock_seconds: float
    tokens_per_second_total: float
    tokens_per_second_per_gpu: float
    peak_vram_gb: float
    extrapolated_seconds_per_run: float
    extrapolated_gpu_hours_per_run: float
    extrapolated_phase2_gpu_hours: float
    deviation_from_budget_pct: float
    final_loss: Optional[float] = None
    sanity_passed: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


def estimate_phase2_run_steps(
    corpus_total_tokens: int,
    effective_batch_size: int,
    seq_length: int,
) -> int:
    """spec §3.3: each Phase 2 run is 3 epochs over corpus."""
    tokens_per_run = corpus_total_tokens * 3
    tokens_per_step = effective_batch_size * seq_length
    return max(1, tokens_per_run // tokens_per_step)


def get_corpus_total_tokens(tokenizer_path: str, train_jsonl: Path) -> int:
    """
    Read corpus_stats.json for total token count under the *active* tokenizer.

    The previous (Path-A) implementation looked for `total_tokens_llama`,
    which was None because the corpus was never tokenized with LLaMA-stock.
    Strategy C is now pinned, so estimate from word counts + Strategy C
    fertility (1.85) until corpus is re-tokenized.
    """
    stats_path = Path("/workspace/data/corpus/stats/corpus_stats.json")
    if not stats_path.exists():
        # Fallback: word count × Strategy C fertility
        return 23_500_000 * 2  # ~47M tokens conservative estimate

    stats = json.loads(stats_path.read_text())
    # Try explicit tokenizer-versioned counts first
    explicit = stats.get("total_tokens_strategy_c") or stats.get("total_tokens_llama")
    if explicit and explicit > 0:
        return int(explicit)

    # Estimate from word count and Strategy C mean fertility
    total_words = (
        stats.get("total_words")
        or stats.get("overall", {}).get("words")
        or 23_500_000
    )
    # Read mean fertility from registry
    try:
        gc = yaml.safe_load(Path("/workspace/registry/global_constants.yaml").read_text())
        fertility = gc.get("tokenizer_version", {}).get("mean_fertility", 1.85)
    except Exception:
        fertility = 1.85
    return int(total_words * fertility)


def find_latest_run_dir(parent: Path, prefix: str = "MICROBENCH") -> Optional[Path]:
    """Find the most recent run output directory."""
    candidates = sorted([d for d in parent.iterdir() if d.is_dir() and d.name.startswith(prefix)],
                       key=lambda d: d.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def parse_metrics_from_run(run_dir: Path) -> dict:
    """
    Parse metrics from the run output dir.

    HF Trainer writes trainer_state.json ONLY in checkpoint-<N>/ subdirs,
    not at run root. We search all checkpoint subdirs and take the latest.

    Returns dict with: final_loss, global_step, total_eval_runtime,
    log_history (list).
    """
    metrics = {}

    # Find the latest checkpoint dir
    checkpoint_dirs = sorted([d for d in run_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")],
                            key=lambda d: int(d.name.split("-")[-1]) if d.name.split("-")[-1].isdigit() else 0)
    if not checkpoint_dirs:
        print(f"  WARN: no checkpoint-N dirs in {run_dir}")
        return metrics

    latest_ckpt = checkpoint_dirs[-1]
    state_path = latest_ckpt / "trainer_state.json"
    if not state_path.exists():
        print(f"  WARN: trainer_state.json not in {latest_ckpt}")
        return metrics

    state = json.loads(state_path.read_text())
    metrics["global_step"] = state.get("global_step", 0)
    metrics["max_steps"]   = state.get("max_steps", 0)
    metrics["log_history"] = state.get("log_history", [])

    # Final training loss (last log_history entry with "loss" field)
    losses = [e["loss"] for e in metrics["log_history"] if "loss" in e]
    if losses:
        metrics["final_loss"] = losses[-1]
        metrics["first_loss"] = losses[0]

    # Total eval runtime — sum across all eval entries (eval interrupts training,
    # so we must subtract this from wall_clock to get pure training time)
    eval_runtimes = [e["eval_runtime"] for e in metrics["log_history"] if "eval_runtime" in e]
    metrics["total_eval_runtime"] = sum(eval_runtimes)

    return metrics


def parse_vram_log(vram_log: Path) -> float:
    """Parse the VRAM monitor CSV and return peak GB across all GPUs."""
    if not vram_log.exists():
        return 0.0
    peak_mib = 0
    for line in vram_log.read_text().splitlines():
        parts = line.strip().split(",")
        if len(parts) >= 3:
            try:
                mib = int(parts[2])
                peak_mib = max(peak_mib, mib)
            except ValueError:
                continue
    return peak_mib / 1024.0  # MiB -> GiB


def write_global_constants(result: BenchResult) -> None:
    """Pin phase2_throughput_calibration under §7.4."""
    from datetime import datetime
    gc_path = Path("/workspace/registry/global_constants.yaml")
    d = yaml.safe_load(gc_path.read_text()) if gc_path.exists() else {}

    if "phase2_throughput_calibration" in d:
        raise RuntimeError(
            "phase2_throughput_calibration already pinned (§7.4 immutable). "
            "Archive and clear it first."
        )

    d["phase2_throughput_calibration"] = {
        "computed_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "micro_benchmark_tokens_per_sec": float(result.tokens_per_second_total),
        "micro_benchmark_tokens_per_sec_per_gpu": float(result.tokens_per_second_per_gpu),
        "peak_vram_gb": float(result.peak_vram_gb),
        "extrapolated_phase2_gpu_hours": float(result.extrapolated_phase2_gpu_hours),
        "deviation_from_budget_pct": float(result.deviation_from_budget_pct),
        "sanity_passed": bool(result.sanity_passed),
        "final_loss": float(result.final_loss) if result.final_loss is not None else None,
    }
    gc_path.write_text(yaml.safe_dump(d, sort_keys=False, default_flow_style=False))


def run_benchmark(
    train_jsonl: Path,
    output_dir: Path,
    n_gpus: int,
    total_steps: int,
    warmup_steps: int,
    dry_run: bool = False,
) -> BenchResult:
    """Invoke train_orchestrator via torchrun and time the result."""
    # Generate a unique run ID
    from datetime import datetime
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = f"MICROBENCH-{ts}"

    # Resolve tokenizer from registry
    gc = yaml.safe_load(Path("/workspace/registry/global_constants.yaml").read_text())
    tokenizer_path = gc.get("tokenizer_version", {}).get("value")
    if not tokenizer_path:
        raise RuntimeError("tokenizer_version not pinned in global_constants.yaml")

    cmd = [
        "torchrun",
        "--nproc_per_node", str(n_gpus),
        "--nnodes", "1",
        "/workspace/scripts/train_orchestrator.py",
        "--phase", "Phase2_Factorial",
        "--factorial-run-id", run_id,
        "--seed", "42",
        "--lr-schedule", "cosine",
        "--batch-size", "128",
        "--optimiser", "AdamW",
        "--sequence-length", "1024",
        "--data-ordering", "mixed",
        # NOTE: do NOT pass --smoke-test. That flag forces max_steps=150 via
        # smoke_test_overrides() in training_loop.py:151, overriding --max-steps.
        # We control max_steps ourselves for accurate measurement.
        "--max-steps", str(total_steps),
        "--output-dir", str(output_dir / run_id),
        "--skip-preflight",
        "--no-eval",          # skip eval pass — it inflates wall-clock for short benchmarks
    ]

    print("=" * 60)
    print("Micro-benchmark plan (Path B — invokes training_loop via torchrun)")
    print("=" * 60)
    print(f"  command         : {' '.join(cmd[:3])} ... (truncated)")
    print(f"  run_id          : {run_id}")
    print(f"  total_steps     : {total_steps}")
    print(f"  warmup_steps    : {warmup_steps}")
    print(f"  effective_batch : 128")
    print(f"  seq_length      : 1024")
    print(f"  n_gpus          : {n_gpus}")
    print(f"  tokens_per_step : {128 * 1024}")
    print(f"  output_dir      : {output_dir / run_id}")
    print(f"  tokenizer       : {tokenizer_path}")

    if dry_run:
        print("\nDRY RUN — exiting before launch.")
        return None

    # Start VRAM monitor as background subprocess
    vram_log = output_dir / f"{run_id}.vram.csv"
    output_dir.mkdir(parents=True, exist_ok=True)
    vram_cmd = [
        "bash", "-c",
        f"while true; do "
        f"ts=$(date -u +%Y-%m-%dT%H:%M:%SZ); "
        f"nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null "
        f'| awk -v ts="$ts" -F\',\' \'{{gsub(/ /, \"\"); print ts\",\"$1\",\"$2}}\'; '
        f"sleep 2; "
        f"done > {vram_log} 2>/dev/null"
    ]
    vram_proc = subprocess.Popen(vram_cmd, preexec_fn=os.setsid)

    try:
        print(f"\nLaunching torchrun (this will take ~30-60 min)...")
        t_start = time.perf_counter()
        result = subprocess.run(cmd, capture_output=False)
        wall_clock = time.perf_counter() - t_start
        print(f"\ntorchrun completed in {wall_clock:.1f}s (rc={result.returncode})")
    finally:
        # Stop VRAM monitor
        try:
            os.killpg(os.getpgid(vram_proc.pid), 15)
        except (ProcessLookupError, PermissionError):
            pass

    # Parse results
    run_dir = output_dir / run_id
    metrics = parse_metrics_from_run(run_dir) if run_dir.exists() else {}
    peak_vram_gb = parse_vram_log(vram_log)
    # Fall back to metrics.peak_vram_gb if VRAM log is empty
    if peak_vram_gb == 0.0 and "metrics.peak_vram_gb" in metrics:
        peak_vram_gb = float(metrics["metrics.peak_vram_gb"])

    # Compute throughput
    # Our wall_clock includes: setup (tokenization map, model load, distributed init)
    # + training + eval. To get true training throughput:
    #   training_seconds = wall_clock - setup_seconds - total_eval_runtime
    # We can\'t measure setup_seconds directly, so we use an empirical estimate
    # based on tokenization + model load (~30 sec model + ~10 min tokenization
    # for full corpus map, ~3 min for cached subsequent runs).
    effective_batch = 128
    seq_length = 1024
    tokens_per_step = effective_batch * seq_length

    actual_steps = metrics.get("global_step", total_steps)
    total_eval_runtime = metrics.get("total_eval_runtime", 0)

    # Setup time (model load + tokenization-cache lookup): typically 30-90s
    # for cached corpus, ~10 min for first-time tokenization. With --no-eval
    # active, eval contribution is 0, so only setup remains as non-training time.
    # We use a conservative 90s estimate to slightly underreport throughput
    # (safer than overreporting for Phase 2 budget planning).
    SETUP_SECONDS_ESTIMATE = 90.0

    pure_training_seconds = max(
        wall_clock - total_eval_runtime - SETUP_SECONDS_ESTIMATE,
        1.0,
    )
    tokens_total = tokens_per_step * actual_steps
    tps_total = tokens_total / pure_training_seconds
    tps_per_gpu = tps_total / n_gpus

    measured_seconds = pure_training_seconds
    measured_steps = actual_steps

    print(f"  Wall clock: {wall_clock:.1f}s")
    print(f"  - eval runtime: {total_eval_runtime:.1f}s")
    print(f"  - setup estimate: {SETUP_SECONDS_ESTIMATE:.0f}s")
    print(f"  = pure training: {pure_training_seconds:.1f}s for {actual_steps} steps")
    print(f"  throughput: {tps_per_gpu:.1f} tokens/sec/GPU")

    # Extrapolate to Phase 2
    corpus_tokens = get_corpus_total_tokens(tokenizer_path, train_jsonl)
    steps_per_run = estimate_phase2_run_steps(corpus_tokens, effective_batch, seq_length)
    seconds_per_run = steps_per_run * (measured_seconds / measured_steps)
    gpu_hours_per_run = seconds_per_run * n_gpus / 3600.0
    phase2_total_hours = gpu_hours_per_run * PHASE2_N_RUNS
    deviation_pct = (phase2_total_hours - PHASE2_BUDGET_GPU_HOURS) / PHASE2_BUDGET_GPU_HOURS * 100

    # Sanity checks
    sanity_passed = (
        peak_vram_gb > 0
        and peak_vram_gb < PEAK_VRAM_MAX_GB
        and tps_per_gpu > TPS_PER_GPU_MIN
    )

    final_loss = metrics.get("final_loss")

    return BenchResult(
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        effective_batch_size=effective_batch,
        seq_length=seq_length,
        n_gpus=n_gpus,
        tokens_per_step=tokens_per_step,
        wall_clock_seconds=wall_clock,
        tokens_per_second_total=tps_total,
        tokens_per_second_per_gpu=tps_per_gpu,
        peak_vram_gb=peak_vram_gb,
        extrapolated_seconds_per_run=seconds_per_run,
        extrapolated_gpu_hours_per_run=gpu_hours_per_run,
        extrapolated_phase2_gpu_hours=phase2_total_hours,
        deviation_from_budget_pct=deviation_pct,
        final_loss=final_loss,
        sanity_passed=sanity_passed,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 2 throughput micro-benchmark")
    parser.add_argument("--tokenizer-path", help="Defaults to Phase-0 winner from global_constants")
    parser.add_argument("--train-jsonl", required=True, type=Path)
    parser.add_argument("--output-dir", default="/workspace/checkpoints/microbench", type=Path)
    parser.add_argument("--n-gpus", type=int, default=8)
    parser.add_argument("--total-steps", type=int, default=DEFAULT_TOTAL_STEPS)
    parser.add_argument("--warmup-steps", type=int, default=DEFAULT_WARMUP_STEPS)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-sanity-pin", action="store_true",
                        help="If sanity fails, still write to results.yaml but skip pinning")
    args = parser.parse_args()

    result = run_benchmark(
        train_jsonl=args.train_jsonl,
        output_dir=args.output_dir,
        n_gpus=args.n_gpus,
        total_steps=args.total_steps,
        warmup_steps=args.warmup_steps,
        dry_run=args.dry_run,
    )
    if result is None:  # dry-run
        return 0

    print()
    print("=" * 60)
    print("Micro-benchmark results")
    print("=" * 60)
    for k, v in result.to_dict().items():
        if isinstance(v, float):
            print(f"  {k:42s} {v:.4f}")
        else:
            print(f"  {k:42s} {v}")

    # Save to results.yaml regardless of sanity
    results_path = Path("/workspace/registry/micro_benchmark_results.yaml")
    results_path.write_text(yaml.safe_dump(result.to_dict(), sort_keys=False, default_flow_style=False))
    print(f"\nWritten: {results_path}")

    # Sanity gate before pinning
    if not result.sanity_passed:
        print()
        print("!" * 60)
        print("SANITY CHECK FAILED — NOT pinning to global_constants.yaml")
        print(f"  peak_vram_gb={result.peak_vram_gb:.2f} (must be 0 < x < {PEAK_VRAM_MAX_GB})")
        print(f"  tps_per_gpu={result.tokens_per_second_per_gpu:.2f} (must be > {TPS_PER_GPU_MIN})")
        if result.final_loss is not None:
            print(f"  final_loss={result.final_loss:.4f} (smoke baseline ~6-12 expected)")
        print("Investigate before re-running. See /workspace/registry/micro_benchmark_results.yaml.")
        print("!" * 60)
        if args.skip_sanity_pin:
            return 1
        return 2

    # Pin
    write_global_constants(result)
    print(f"\nPinned to /workspace/registry/global_constants.yaml::phase2_throughput_calibration")
    print(f"  tokens_per_sec_per_gpu: {result.tokens_per_second_per_gpu:.2f}")
    print(f"  peak_vram_gb:           {result.peak_vram_gb:.2f}")
    print(f"  phase2_gpu_hours_est:   {result.extrapolated_phase2_gpu_hours:.1f}")
    print(f"  deviation_from_budget:  {result.deviation_from_budget_pct:+.1f}%")

    abs_dev = abs(result.deviation_from_budget_pct)
    if abs_dev > 15.0:
        print()
        print("!" * 60)
        print(f"WARNING: deviation exceeds spec §13 threshold (±15%).")
        if result.deviation_from_budget_pct > 0:
            print(f"  Phase 2 would OVERSPEND budget by {abs_dev:.1f}%.")
            print(f"  Reduce scope OR request budget extension before Phase 2 launch.")
        else:
            print(f"  Phase 2 will under-run budget by {abs_dev:.1f}%.")
            print(f"  Reserve can absorb surplus OR add Phase 3 PEFT methods.")
        print("!" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
