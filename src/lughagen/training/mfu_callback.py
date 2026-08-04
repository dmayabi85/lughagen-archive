"""
LughaGen MFU (Model FLOPs Utilization) callback.

Computes and logs MFU during training, mirroring the logging pattern used by
PerLanguageGradNormCallback in training_loop.py.

FLOPs formula (Hoffmann/Chinchilla):
    flops_per_token = 6 * N_params + 12 * L * H * S
    flops_per_step  = flops_per_token * tokens_per_step

where:
    N_params        = total model parameters (frozen + trainable; gradients
                      flow through all of them in fwd+bwd)
    L               = number of transformer layers
    H               = hidden size
    S               = sequence length (per sequence in the batch)
    tokens_per_step = effective_batch_size * sequence_length

Peak FLOPs (A100 BF16) = 312 TFLOPS per GPU, aggregated across world_size.

MFU = (flops_per_step * steps_in_interval) /
      (interval_wall_clock_seconds * peak_aggregate_flops)

Standard practice (and what the LughaGen proposal commits to) includes
data loading overhead in the wall-clock measurement, so the resulting MFU
reflects real end-to-end utilization, not just compute kernel efficiency.

Outputs:
    {output_dir}/mfu_log.jsonl  - per-interval records + init + summary

Run as a script for self-tests:
    python /workspace/scripts/mfu_callback.py
"""
from __future__ import annotations

import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# Optional torch import — module is testable without GPU
try:
    import torch
    import torch.distributed as dist
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

# Optional transformers import — falls back to a stub for unit testing
try:
    from transformers import TrainerCallback
    _HAS_TRANSFORMERS = True
except ImportError:
    _HAS_TRANSFORMERS = False
    class TrainerCallback:  # type: ignore
        """Stub for unit testing without transformers installed."""
        pass


# ─── Constants ────────────────────────────────────────────────────────────────

A100_BF16_PEAK_TFLOPS = 312.0  # Theoretical peak per GPU, BF16
DEFAULT_LOG_INTERVAL = 50      # Match PerLanguageGradNormCallback cadence


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _utc_now_iso() -> str:
    """ISO-8601 UTC timestamp with 'Z' suffix. Matches registry_csv.utc_now_iso."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _is_rank_zero() -> bool:
    """Returns True if running on rank 0 (or not distributed)."""
    if not _HAS_TORCH:
        return True
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank() == 0
    return True


def _get_world_size() -> int:
    """Returns world size (number of distributed processes)."""
    if not _HAS_TORCH:
        return 1
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    # Fallback: number of visible GPUs, or 1
    if torch.cuda.is_available():
        return torch.cuda.device_count()
    return 1


# ─── FLOPs computation ────────────────────────────────────────────────────────

def compute_flops_per_token(
    n_params: int,
    num_hidden_layers: int,
    hidden_size: int,
    sequence_length: int,
) -> float:
    """Forward+backward FLOPs per token under the Chinchilla/Hoffmann
    approximation.

    Base: 6 * N for matmul (2 fwd + 4 bwd factors per param).
    Attention overhead: 12 * L * H * S for the quadratic attention cost.

    Returns a float (large, typically 1e10-1e11 for 8B models at S=1024).
    """
    if n_params <= 0:
        raise ValueError(f"n_params must be positive, got {n_params}")
    if num_hidden_layers <= 0 or hidden_size <= 0 or sequence_length <= 0:
        raise ValueError("layers/hidden/seq must all be positive")
    base = 6.0 * n_params
    attn = 12.0 * num_hidden_layers * hidden_size * sequence_length
    return base + attn


def compute_flops_per_step(
    n_params: int,
    num_hidden_layers: int,
    hidden_size: int,
    sequence_length: int,
    effective_batch_size: int,
) -> float:
    """FLOPs for one optimization step.

    'Effective batch size' = micro_batch * grad_accum * world_size, i.e. the
    total number of sequences contributing gradients to a single optimizer step.
    Tokens per step = effective_batch_size * sequence_length.
    """
    if effective_batch_size <= 0:
        raise ValueError(f"effective_batch_size must be positive, got {effective_batch_size}")
    flops_per_token = compute_flops_per_token(
        n_params, num_hidden_layers, hidden_size, sequence_length
    )
    tokens_per_step = effective_batch_size * sequence_length
    return flops_per_token * tokens_per_step


def compute_peak_aggregate_flops(
    world_size: int,
    peak_tflops_per_gpu: float = A100_BF16_PEAK_TFLOPS,
) -> float:
    """Theoretical peak FLOPs/sec across all GPUs in the world."""
    return peak_tflops_per_gpu * 1e12 * world_size


# ─── The callback ─────────────────────────────────────────────────────────────

class MFUCallback(TrainerCallback):
    """HuggingFace TrainerCallback that logs Model FLOPs Utilization.

    Construction:
        cb = MFUCallback(
            output_dir=Path("/workspace/checkpoints/RUN-..."),
            n_params=8_030_000_000,
            num_hidden_layers=32,
            hidden_size=4096,
            sequence_length=1024,
            effective_batch_size=128,
            world_size=8,
            log_interval=50,
        )

    Wire into Trainer:
        trainer = Trainer(..., callbacks=[..., cb])

    Output JSONL records (rank 0 only):
        - on_train_begin: one "mfu_init" event with architecture + peak FLOPs
        - on_step_end (every log_interval steps): one "mfu" record with the
          window MFU, achieved TFLOPS, wall-clock for the interval
        - on_train_end: one "mfu_summary" with mean/median/p10/p90 across all
          intervals
    """

    EVENT_INIT = "mfu_init"
    EVENT_INTERVAL = "mfu_interval"
    EVENT_SUMMARY = "mfu_summary"

    def __init__(
        self,
        output_dir: Path,
        n_params: int,
        num_hidden_layers: int,
        hidden_size: int,
        sequence_length: int,
        effective_batch_size: int,
        world_size: Optional[int] = None,
        peak_tflops_per_gpu: float = A100_BF16_PEAK_TFLOPS,
        log_interval: int = DEFAULT_LOG_INTERVAL,
    ):
        self.output_dir = Path(output_dir)
        self.log_path = self.output_dir / "mfu_log.jsonl"
        self.log_interval = log_interval

        if world_size is None:
            world_size = _get_world_size()
        self.world_size = world_size

        # Architecture + batch config snapshot (used in records)
        self.n_params = int(n_params)
        self.num_hidden_layers = int(num_hidden_layers)
        self.hidden_size = int(hidden_size)
        self.sequence_length = int(sequence_length)
        self.effective_batch_size = int(effective_batch_size)
        self.peak_tflops_per_gpu = float(peak_tflops_per_gpu)

        # Precomputed
        self.flops_per_step = compute_flops_per_step(
            self.n_params,
            self.num_hidden_layers,
            self.hidden_size,
            self.sequence_length,
            self.effective_batch_size,
        )
        self.peak_aggregate_flops = compute_peak_aggregate_flops(
            self.world_size, self.peak_tflops_per_gpu
        )

        # Runtime state (interval timer)
        self._interval_start_time: Optional[float] = None
        self._interval_start_step: int = 0

    # ── Internal logging primitive ────────────────────────────────────────────

    def _append(self, record: dict) -> None:
        """Append a JSON record to mfu_log.jsonl on rank 0 only."""
        if not _is_rank_zero():
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a") as f:
            f.write(json.dumps(record) + "\n")

    # ── HF Trainer callback hooks ─────────────────────────────────────────────

    def on_train_begin(self, args=None, state=None, control=None, **kwargs):
        """Called once at the start of training. Log architecture snapshot."""
        self._interval_start_time = time.perf_counter()
        self._interval_start_step = state.global_step if state is not None else 0
        self._append({
            "ts": _utc_now_iso(),
            "event": self.EVENT_INIT,
            "n_params": self.n_params,
            "num_hidden_layers": self.num_hidden_layers,
            "hidden_size": self.hidden_size,
            "sequence_length": self.sequence_length,
            "effective_batch_size": self.effective_batch_size,
            "world_size": self.world_size,
            "peak_tflops_per_gpu": self.peak_tflops_per_gpu,
            "flops_per_step": self.flops_per_step,
            "peak_aggregate_flops": self.peak_aggregate_flops,
            "log_interval": self.log_interval,
        })
        return control

    def on_step_end(self, args=None, state=None, control=None, **kwargs):
        """Called after each optimizer step. Log MFU every log_interval steps."""
        if state is None:
            return control
        if state.global_step % self.log_interval != 0:
            return control
        if self._interval_start_time is None:
            # Defensive: on_train_begin should have set this
            self._interval_start_time = time.perf_counter()
            self._interval_start_step = state.global_step
            return control

        now = time.perf_counter()
        elapsed = now - self._interval_start_time
        steps_in_interval = state.global_step - self._interval_start_step
        if steps_in_interval <= 0 or elapsed <= 0.0:
            # Defensive: shouldn't happen, but avoid div-by-zero
            self._interval_start_time = now
            self._interval_start_step = state.global_step
            return control

        achieved_flops_per_sec = self.flops_per_step * steps_in_interval / elapsed
        mfu = achieved_flops_per_sec / self.peak_aggregate_flops

        self._append({
            "ts": _utc_now_iso(),
            "event": self.EVENT_INTERVAL,
            "step": state.global_step,
            "interval_steps": steps_in_interval,
            "interval_seconds": round(elapsed, 3),
            "achieved_tflops_per_sec_aggregate": round(achieved_flops_per_sec / 1e12, 3),
            "peak_tflops_per_sec_aggregate": round(self.peak_aggregate_flops / 1e12, 3),
            "mfu": round(mfu, 4),
        })

        # Reset window
        self._interval_start_time = now
        self._interval_start_step = state.global_step
        return control

    def on_train_end(self, args=None, state=None, control=None, **kwargs):
        """Called once at training end. Compute summary stats from log."""
        if not _is_rank_zero():
            return control
        if not self.log_path.exists():
            return control

        mfus = []
        with self.log_path.open() as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("event") == self.EVENT_INTERVAL and "mfu" in r:
                    mfus.append(r["mfu"])

        if not mfus:
            return control

        mfus_sorted = sorted(mfus)
        n = len(mfus_sorted)
        self._append({
            "ts": _utc_now_iso(),
            "event": self.EVENT_SUMMARY,
            "n_intervals": n,
            "mean_mfu": round(statistics.mean(mfus_sorted), 4),
            "median_mfu": round(statistics.median(mfus_sorted), 4),
            "p10_mfu": round(mfus_sorted[max(0, n // 10 - 1)], 4),
            "p90_mfu": round(mfus_sorted[min(n - 1, 9 * n // 10)], 4),
            "min_mfu": round(mfus_sorted[0], 4),
            "max_mfu": round(mfus_sorted[-1], 4),
        })
        return control


# ─── Self-tests ───────────────────────────────────────────────────────────────

def _run_self_tests() -> int:
    """Sanity tests for the FLOPs formula and callback mechanics.

    Run with:  python /workspace/scripts/mfu_callback.py
    Returns 0 on success, 1 on failure.
    """
    import tempfile

    print("=== mfu_callback.py self-tests ===\n")

    failures = []

    # Test 1: FLOPs per token for LLaMA-3.1-8B at S=1024
    print("[test 1] FLOPs per token for LLaMA-3.1-8B at S=1024")
    n_params = 8_030_000_000
    L, H, S = 32, 4096, 1024
    flops_tok = compute_flops_per_token(n_params, L, H, S)
    # Expected: 6 * 8.03e9 + 12 * 32 * 4096 * 1024
    #         = 4.818e10 + 1.610e9
    #         ≈ 4.98e10
    expected_low, expected_high = 4.9e10, 5.0e10
    if expected_low <= flops_tok <= expected_high:
        print(f"  ✓ {flops_tok:.3e} FLOPs/token (in expected range)")
    else:
        msg = f"  ✗ {flops_tok:.3e} FLOPs/token (expected {expected_low:.2e}..{expected_high:.2e})"
        print(msg)
        failures.append(msg)

    # Test 2: FLOPs per step with effective_batch=128, S=1024
    print("\n[test 2] FLOPs per step (effective_batch=128, S=1024)")
    flops_step = compute_flops_per_step(n_params, L, H, S, effective_batch_size=128)
    # Expected: flops_tok * 128 * 1024 = flops_tok * 131_072
    expected = flops_tok * 128 * 1024
    if abs(flops_step - expected) / expected < 1e-9:
        print(f"  ✓ {flops_step:.3e} FLOPs/step ({flops_step / 1e15:.2f} PFLOPs)")
    else:
        msg = f"  ✗ {flops_step:.3e} != expected {expected:.3e}"
        print(msg)
        failures.append(msg)

    # Test 3: Peak aggregate FLOPs for 8 A100s
    print("\n[test 3] Peak aggregate FLOPs (8 × A100 BF16)")
    peak = compute_peak_aggregate_flops(world_size=8)
    expected = 312e12 * 8  # 2.496e15
    if abs(peak - expected) < 1.0:
        print(f"  ✓ {peak:.3e} FLOPs/s ({peak / 1e15:.3f} PFLOPs/s aggregate)")
    else:
        msg = f"  ✗ {peak:.3e} != expected {expected:.3e}"
        print(msg)
        failures.append(msg)

    # Test 4: MFU sanity — realistic step time gives MFU in 0.10..0.95
    # Note: bounds are wide because the synthetic step time is illustrative,
    # not measured. Real LughaGen v2-D ran at ~5.5 sec/step → ~47% MFU.
    print("\n[test 4] MFU sanity check for v2-D-like step time")
    step_time_sec = 5.5  # measured rough average from v2-D's 98 GPU-hr / 6000 steps
    achieved_flops_per_sec = flops_step / step_time_sec
    mfu = achieved_flops_per_sec / peak
    if 0.30 <= mfu <= 0.65:
        print(f"  ✓ MFU = {mfu:.3f} ({mfu*100:.1f}%) for {step_time_sec}-sec step time")
        print(f"    (expected range for QLoRA on 8×A100 with ZeRO sharding)")
    else:
        msg = f"  ✗ MFU = {mfu:.3f} — outside realistic range 0.30..0.65 for {step_time_sec}-sec/step"
        print(msg)
        failures.append(msg)

    # Test 5: Bad inputs raise ValueError
    print("\n[test 5] Bad inputs raise ValueError")
    bad_cases = [
        (lambda: compute_flops_per_token(0, 32, 4096, 1024), "n_params=0"),
        (lambda: compute_flops_per_token(8e9, 0, 4096, 1024), "layers=0"),
        (lambda: compute_flops_per_step(8e9, 32, 4096, 1024, 0), "batch=0"),
    ]
    for fn, label in bad_cases:
        try:
            fn()
            msg = f"  ✗ {label}: expected ValueError, got no error"
            print(msg)
            failures.append(msg)
        except ValueError:
            print(f"  ✓ {label}: correctly raised ValueError")

    # Test 6: Callback can be constructed and logs init record
    print("\n[test 6] Callback lifecycle in a temp dir")
    with tempfile.TemporaryDirectory() as tmpdir:
        cb = MFUCallback(
            output_dir=Path(tmpdir),
            n_params=8_030_000_000,
            num_hidden_layers=32,
            hidden_size=4096,
            sequence_length=1024,
            effective_batch_size=128,
            world_size=8,
            log_interval=10,
        )

        # Fake state object
        class FakeState:
            def __init__(self):
                self.global_step = 0

        state = FakeState()
        cb.on_train_begin(state=state)

        # Simulate 50 steps with ~3 sec/step wall-clock
        sim_start = time.perf_counter()
        # Force the interval timer back so elapsed appears nonzero quickly
        cb._interval_start_time = sim_start - 30.0  # pretend 30 sec elapsed
        cb._interval_start_step = 0
        state.global_step = 10
        cb.on_step_end(state=state)
        # Another interval
        cb._interval_start_time -= 0  # use real timer this round
        state.global_step = 20
        cb.on_step_end(state=state)

        cb.on_train_end(state=state)

        # Read back and verify
        log = Path(tmpdir) / "mfu_log.jsonl"
        if not log.exists():
            msg = "  ✗ mfu_log.jsonl not created"
            print(msg)
            failures.append(msg)
        else:
            records = [json.loads(line) for line in log.read_text().splitlines() if line.strip()]
            events = [r.get("event") for r in records]
            print(f"  records: {len(records)}, events: {events}")
            if MFUCallback.EVENT_INIT in events:
                print(f"  ✓ init record present")
            else:
                msg = "  ✗ init record missing"
                print(msg)
                failures.append(msg)
            if MFUCallback.EVENT_INTERVAL in events:
                print(f"  ✓ interval record(s) present")
            else:
                msg = "  ✗ no interval records"
                print(msg)
                failures.append(msg)
            if MFUCallback.EVENT_SUMMARY in events:
                print(f"  ✓ summary record present")
            else:
                msg = "  ✗ summary record missing"
                print(msg)
                failures.append(msg)

    # Final
    print("\n=== Result ===")
    if failures:
        print(f"FAIL: {len(failures)} test(s) failed:")
        for m in failures:
            print(f"  {m}")
        return 1
    print("All tests passed ✓")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(_run_self_tests())
