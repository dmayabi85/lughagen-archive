"""
Extract pilot training metrics — auto-discovers trainer_state.json files.

The HuggingFace Trainer saves trainer_state.json inside each checkpoint-N/
directory, not necessarily in final/. This script scans each pilot run's
checkpoint root, finds the highest-numbered checkpoint, and extracts the
metrics from its trainer_state.json.

Run inside the Brev container:
    python parse_training_state.py

Output:
    pilot_training_summary.json  (machine-readable summary)
    Console table              (paste into proposal appendix)
"""

import json
import re
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────
# Pilot run roots — script will search inside each for checkpoint-N/
# ─────────────────────────────────────────────────────────────────────
PILOT_ROOTS = [
    ("/workspace/checkpoints/afro-xlmr-swahili",      "AfroXLMR-Large v1 (full FT)"),
    ("/workspace/checkpoints/afro-xlmr-swahili-v2",   "AfroXLMR-Large v2 (full FT, retrained)"),
    ("/workspace/checkpoints/llama-3.1-8b-swahili",   "LLaMA-3.1-8B QLoRA Swahili"),
]


def find_latest_trainer_state(root: Path):
    """
    Return path to the highest-numbered checkpoint's trainer_state.json,
    falling back to root/final/trainer_state.json if no checkpoint-N found.
    """
    if not root.exists():
        return None

    candidates = []
    # checkpoint-N directories
    for d in root.iterdir():
        if d.is_dir():
            m = re.match(r"checkpoint-(\d+)$", d.name)
            if m and (d / "trainer_state.json").exists():
                candidates.append((int(m.group(1)), d / "trainer_state.json"))
    if candidates:
        candidates.sort()
        return candidates[-1][1]   # highest step number

    # fallback: try final/ directly
    if (root / "final" / "trainer_state.json").exists():
        return root / "final" / "trainer_state.json"

    # last resort: any trainer_state.json under root
    matches = list(root.rglob("trainer_state.json"))
    if matches:
        return matches[-1]
    return None


def summarise(state_path: Path, label: str):
    """Read trainer_state.json and extract the most useful metrics."""
    with open(state_path) as f:
        state = json.load(f)

    log = state.get("log_history", [])
    train_logs = [r for r in log if "loss" in r and "eval_loss" not in r]
    eval_logs  = [r for r in log if "eval_loss" in r]

    if not train_logs:
        return {"label": label, "source": str(state_path), "note": "empty log"}

    first      = train_logs[0]
    last       = train_logs[-1]
    first_loss = first.get("loss")
    last_loss  = last.get("loss")
    last_step  = last.get("step", state.get("global_step"))
    last_epoch = last.get("epoch")
    last_lr    = last.get("learning_rate")
    grad_norm  = last.get("grad_norm")
    train_rt   = next((r.get("train_runtime") for r in log if "train_runtime" in r), None)
    samples_s  = next((r.get("train_samples_per_second") for r in log if "train_samples_per_second" in r), None)
    steps_s    = next((r.get("train_steps_per_second") for r in log if "train_steps_per_second" in r), None)
    final_eval = eval_logs[-1].get("eval_loss") if eval_logs else None
    best_eval  = min((e.get("eval_loss", float("inf")) for e in eval_logs),
                     default=None)

    return {
        "label"              : label,
        "source"             : str(state_path),
        "first_train_loss"   : round(first_loss, 4) if first_loss else None,
        "final_train_loss"   : round(last_loss, 4) if last_loss else None,
        "final_eval_loss"    : round(final_eval, 4) if final_eval else None,
        "best_eval_loss"     : round(best_eval, 4) if best_eval and best_eval != float("inf") else None,
        "total_steps"        : last_step,
        "final_epoch"        : round(last_epoch, 2) if last_epoch else None,
        "final_lr"           : last_lr,
        "final_grad_norm"    : round(grad_norm, 4) if grad_norm else None,
        "train_runtime_sec"  : train_rt,
        "train_runtime_hr"   : round(train_rt / 3600, 2) if train_rt else None,
        "samples_per_second" : round(samples_s, 2) if samples_s else None,
        "steps_per_second"   : round(steps_s, 3) if steps_s else None,
    }


def print_summary(s):
    print(f"\n  ── {s['label']} ──")
    print(f"  Source: {s['source']}")
    if s.get("note"):
        print(f"  NOTE: {s['note']}")
        return
    print(f"  {'-' * 60}")
    skip = {"label", "source"}
    for k, v in s.items():
        if k in skip or v is None:
            continue
        print(f"  {k:<22s}: {v}")


def print_comparative_table(summaries):
    """Compact comparison across runs — paste into appendix."""
    if not summaries:
        return
    print("\n" + "=" * 92)
    print("  Pilot Training Comparison Summary")
    print("=" * 92)
    cols = [
        ("Steps",            "total_steps",        "{}"),
        ("Final loss",       "final_train_loss",   "{:.4f}"),
        ("Best eval loss",   "best_eval_loss",     "{:.4f}"),
        ("Wall-clock (h)",   "train_runtime_hr",   "{:.2f}"),
        ("samples/s",        "samples_per_second", "{:.2f}"),
    ]
    print(f"  {'Run':<42} " + " ".join(f"{c[0]:>15}" for c in cols))
    print("  " + "-" * 90)
    for s in summaries:
        if s.get("note"):
            continue
        row = f"  {s['label']:<42} "
        for _, key, fmt in cols:
            v = s.get(key)
            row += " " + (fmt.format(v) if v is not None else "n/a").rjust(15)
        print(row)
    print("=" * 92)


if __name__ == "__main__":
    print("=" * 92)
    print("  LughaGen Pilot Training Metrics (extracted from trainer_state.json)")
    print("=" * 92)

    summaries = []
    for root, label in PILOT_ROOTS:
        ts = find_latest_trainer_state(Path(root))
        if ts is None:
            print(f"\n  [missing] no trainer_state.json found under {root}")
            continue
        s = summarise(ts, label)
        summaries.append(s)
        print_summary(s)

    print_comparative_table(summaries)

    with open("pilot_training_summary.json", "w") as f:
        json.dump(summaries, f, indent=2)
    print(f"\n  Saved pilot_training_summary.json")
    print("=" * 92)
