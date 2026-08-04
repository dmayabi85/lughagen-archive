#!/usr/bin/env python3
"""
/workspace/scripts/kill_switch.py

LughaGen Training Framework v1.5.1 §13
File-presence-based kill switch for graceful training termination.

Mechanism:
  - During training, every N steps the loop checks for /workspace/registry/KILL_SWITCH
  - If the file exists, the loop:
      1. Logs the kill request with timestamp + reason (file contents)
      2. Saves a final checkpoint via Trainer.save_state()
      3. Marks the run record with training_events.early_stopped = true
      4. Exits gracefully (return code 0)
  - Distributed coordination: only rank-0 reads the file; rank-0 broadcasts the
    decision to all ranks via torch.distributed so all GPUs exit together.

Usage from training loop:
    from kill_switch import KillSwitchMonitor
    monitor = KillSwitchMonitor(check_interval_steps=10)
    # ... inside training loop ...
    if monitor.should_stop(step, trainer):
        trainer.save_state()
        sys.exit(0)

Triggering kill from outside:
    echo "out of time before supervisor meeting" > /workspace/registry/KILL_SWITCH
    # Training will detect this within ~30 seconds and shut down cleanly

Clearing:
    rm /workspace/registry/KILL_SWITCH    # done automatically after detection
"""

from __future__ import annotations

import datetime as dt
import os
import sys
from pathlib import Path
from typing import Optional


KILL_SWITCH_PATH = Path("/workspace/registry/KILL_SWITCH")
KILL_LOG_PATH = Path("/workspace/registry/kill_switch_log.jsonl")


def utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class KillSwitchMonitor:
    """
    Polls the kill-switch file at most every `check_interval_steps` steps.
    Cheap when no file is present (single os.path.exists call).

    Distributed-aware: only rank-0 inspects the filesystem. The decision
    is broadcast to all ranks via torch.distributed if available.
    """

    def __init__(
        self,
        check_interval_steps: int = 10,
        kill_path: Path = KILL_SWITCH_PATH,
        log_path: Path = KILL_LOG_PATH,
        rank: Optional[int] = None,
    ):
        self.check_interval_steps = max(1, check_interval_steps)
        self.kill_path = kill_path
        self.log_path = log_path
        self.rank = rank if rank is not None else int(os.environ.get("RANK", "0"))
        self._last_check_step = -1
        self._triggered = False

    def _rank0_check(self) -> tuple[bool, str]:
        """Inspect filesystem from rank 0. Returns (should_stop, reason)."""
        if not self.kill_path.exists():
            return False, ""
        try:
            reason = self.kill_path.read_text().strip()
        except OSError:
            reason = "(could not read reason file)"
        return True, reason or "(no reason supplied)"

    def _broadcast_decision(self, should_stop: bool, reason: str) -> tuple[bool, str]:
        """
        Broadcast rank-0's decision to all ranks. Returns the broadcast values.

        If torch.distributed is not initialised (single-GPU or pre-DDP), returns
        the local decision unchanged.
        """
        try:
            import torch
            import torch.distributed as dist
        except ImportError:
            return should_stop, reason
        if not dist.is_available() or not dist.is_initialized():
            return should_stop, reason
        # Use a single-element tensor for the boolean
        flag = torch.tensor([1 if should_stop else 0], dtype=torch.int32, device="cuda")
        dist.broadcast(flag, src=0)
        decision = bool(flag.item())
        # Reason text broadcast is best-effort; lower ranks just need the flag
        return decision, reason

    def should_stop(self, step: int, log_to_jsonl: bool = True) -> tuple[bool, str]:
        """
        Check whether the loop should stop. Returns (should_stop, reason).

        Call this from inside the training loop, e.g. as a callback hook.
        Cheap: only does a filesystem check every `check_interval_steps`.
        """
        if self._triggered:
            return True, "(already triggered)"

        if step - self._last_check_step < self.check_interval_steps:
            return False, ""
        self._last_check_step = step

        if self.rank == 0:
            should_stop, reason = self._rank0_check()
        else:
            should_stop, reason = False, ""

        # Broadcast so all ranks agree
        should_stop, reason = self._broadcast_decision(should_stop, reason)

        if should_stop:
            self._triggered = True
            if log_to_jsonl and self.rank == 0:
                self._log_trigger(step, reason)
                # Move (don't delete) the trigger file so we have an audit trail
                try:
                    archive = self.kill_path.parent / "kill_switch_history" / (
                        f"KILL_SWITCH_{dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.txt"
                    )
                    archive.parent.mkdir(parents=True, exist_ok=True)
                    self.kill_path.rename(archive)
                except OSError:
                    pass    # if move fails, file stays; not fatal

        return should_stop, reason

    def _log_trigger(self, step: int, reason: str) -> None:
        """Append kill event to the audit log."""
        import json
        record = {
            "ts": utc_now_iso(),
            "step": step,
            "reason": reason,
            "rank": self.rank,
            "pid": os.getpid(),
        }
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a") as f:
                f.write(json.dumps(record) + "\n")
        except OSError as e:
            print(f"WARNING: could not write kill_switch_log: {e}", file=sys.stderr)


def is_kill_requested() -> bool:
    """Non-distributed convenience check. Useful for short utility scripts."""
    return KILL_SWITCH_PATH.exists()


def request_kill(reason: str = "") -> None:
    """Programmatically request kill (CLI helper, mainly for testing)."""
    KILL_SWITCH_PATH.parent.mkdir(parents=True, exist_ok=True)
    KILL_SWITCH_PATH.write_text(reason or f"requested at {utc_now_iso()}")
    print(f"Kill request written to {KILL_SWITCH_PATH}")
    if reason:
        print(f"  Reason: {reason}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Kill switch utility (§13)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_kill = sub.add_parser("request", help="Write a kill request")
    p_kill.add_argument("--reason", default="", help="Reason for the kill request")

    sub.add_parser("status", help="Check whether a kill is pending")
    sub.add_parser("clear", help="Remove a pending kill request")

    args = parser.parse_args()

    if args.cmd == "request":
        request_kill(args.reason)
    elif args.cmd == "status":
        if is_kill_requested():
            print(f"KILL PENDING at {KILL_SWITCH_PATH}")
            print(f"  Reason: {KILL_SWITCH_PATH.read_text().strip()}")
            sys.exit(1)
        else:
            print(f"No kill pending ({KILL_SWITCH_PATH} not present)")
    elif args.cmd == "clear":
        if KILL_SWITCH_PATH.exists():
            KILL_SWITCH_PATH.unlink()
            print(f"Cleared {KILL_SWITCH_PATH}")
        else:
            print(f"Nothing to clear")
