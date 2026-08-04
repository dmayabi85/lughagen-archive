"""
Extract pilot training metrics from saved trainer_state.json files.

This script reads the HuggingFace Trainer state files produced during the
LughaGen pilot runs and prints a compact summary table plus a JSON dump
suitable for inclusion in the proposal as an appendix or figure caption.

Run from the Brev container (where /workspace/checkpoints/.../trainer_state.json
exists), or copy the JSON files locally and adjust the paths below.

Usage:  python parse_training_state.py [path_to_trainer_state.json] [run_label]
        python parse_training_state.py /workspace/checkpoints/llama-3.1-8b-swahili/final/trainer_state.json "LLaMA-3.1-8B QLoRA"
"""

import json
import sys
from pathlib import Path


def summarise(state_path: Path, label: str):
    if not state_path.exists():
        print(f"  [missing] {state_path}")
        return None

    with open(state_path) as f:
        state = json.load(f)

    log = state.get("log_history", [])
    train_logs = [r for r in log if "loss" in r and "eval_loss" not in r]
    eval_logs  = [r for r in log if "eval_loss" in r]

    if not train_logs:
        print(f"  [empty log] {state_path}")
        return None

    first_loss = train_logs[0].get("loss")
    last_loss  = train_logs[-1].get("loss")
    last_step  = train_logs[-1].get("step", state.get("global_step"))
    last_epoch = train_logs[-1].get("epoch")
    last_lr    = train_logs[-1].get("learning_rate")
    train_rt   = next((r.get("train_runtime") for r in log if "train_runtime" in r), None)
    samples_s  = next((r.get("train_samples_per_second") for r in log if "train_samples_per_second" in r), None)
    final_eval = eval_logs[-1].get("eval_loss") if eval_logs else None

    summary = {
        "label"              : label,
        "first_loss"         : round(first_loss, 4) if first_loss else None,
        "final_train_loss"   : round(last_loss, 4) if last_loss else None,
        "final_eval_loss"    : round(final_eval, 4) if final_eval else None,
        "total_steps"        : last_step,
        "final_epoch"        : round(last_epoch, 2) if last_epoch else None,
        "final_lr"           : last_lr,
        "train_runtime_sec"  : train_rt,
        "train_runtime_hr"   : round(train_rt / 3600, 2) if train_rt else None,
        "samples_per_second" : round(samples_s, 2) if samples_s else None,
    }

    print(f"\n  {label}")
    print(f"  {'-' * 60}")
    for k, v in summary.items():
        if v is None or k == "label":
            continue
        print(f"  {k:<22s}: {v}")

    return summary


# ─────────────────────────────────────────────────────────────────
# Run on the three pilot checkpoints
# (adjust paths to wherever your trainer_state.json files live)
# ─────────────────────────────────────────────────────────────────
DEFAULT_RUNS = [
    ("/workspace/checkpoints/afro-xlmr-swahili/final/trainer_state.json",
     "AfroXLMR-Large v1 (full FT)"),
    ("/workspace/checkpoints/afro-xlmr-swahili-v2/final/trainer_state.json",
     "AfroXLMR-Large v2 (full FT, retrained)"),
    ("/workspace/checkpoints/llama-3.1-8b-swahili/final/trainer_state.json",
     "LLaMA-3.1-8B QLoRA Swahili"),
]


if __name__ == "__main__":
    # Custom single-run mode: python parse_training_state.py PATH LABEL
    if len(sys.argv) == 3:
        summaries = [summarise(Path(sys.argv[1]), sys.argv[2])]
    else:
        print("=" * 70)
        print("  LughaGen Pilot Training Metrics (from trainer_state.json)")
        print("=" * 70)
        summaries = [summarise(Path(p), lbl) for p, lbl in DEFAULT_RUNS]

    # Filtered summaries → JSON for proposal appendix
    summaries = [s for s in summaries if s]
    with open("pilot_training_summary.json", "w") as f:
        json.dump(summaries, f, indent=2)

    print(f"\n  Summary saved to pilot_training_summary.json")
    print("=" * 70)
