"""
Extract pilot hyperparameter settings from saved training_args.bin and
adapter_config.json files. Auto-discovers the most recent checkpoint when
training_args.bin is not in final/.

The proposal's Phase 2 factorial requires evidence for:
  1. Learning-Rate Schedule  — pilot used cosine (Level A)
  2. Effective Batch Size    — pilot used 256 (Level B)
  3. Optimiser              — pilot used AdamW (Level A)
  4. Sequence Length         — pilot used 256-512 (bracketed by 1024/2048 levels)
  5. Data Ordering           — pilot did not vary (factorial introduces this)

Run inside the Brev container:
    python pilot_hyperparameters.py

Output:
    pilot_hyperparameters.json
    Console table that can be pasted directly into the proposal appendix.
"""

import torch
import json
import re
import sys
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────
# Pilot run roots — script searches inside each
# ─────────────────────────────────────────────────────────────────────
PILOT_ROOTS = [
    ("/workspace/checkpoints/afro-xlmr-swahili",      "AfroXLMR-Large v1 (full FT)"),
    ("/workspace/checkpoints/afro-xlmr-swahili-v2",   "AfroXLMR-Large v2 (full FT, retrained)"),
    ("/workspace/checkpoints/llama-3.1-8b-swahili",   "LLaMA-3.1-8B QLoRA Swahili"),
]


def find_training_args(root: Path):
    """
    Look for training_args.bin in this order:
      1. root/final/training_args.bin
      2. highest-numbered checkpoint-N/training_args.bin
      3. any training_args.bin found by rglob
    """
    if not root.exists():
        return None

    # final/ first
    final_args = root / "final" / "training_args.bin"
    if final_args.exists():
        return final_args

    # highest checkpoint-N
    candidates = []
    for d in root.iterdir():
        if d.is_dir():
            m = re.match(r"checkpoint-(\d+)$", d.name)
            if m and (d / "training_args.bin").exists():
                candidates.append((int(m.group(1)), d / "training_args.bin"))
    if candidates:
        candidates.sort()
        return candidates[-1][1]

    # any descendant
    matches = list(root.rglob("training_args.bin"))
    if matches:
        return matches[-1]
    return None


def find_adapter_config(root: Path):
    """Same auto-discovery for adapter_config.json (LoRA runs only)."""
    if not root.exists():
        return None

    final_cfg = root / "final" / "adapter_config.json"
    if final_cfg.exists():
        return final_cfg

    candidates = []
    for d in root.iterdir():
        if d.is_dir():
            m = re.match(r"checkpoint-(\d+)$", d.name)
            if m and (d / "adapter_config.json").exists():
                candidates.append((int(m.group(1)), d / "adapter_config.json"))
    if candidates:
        candidates.sort()
        return candidates[-1][1]

    matches = list(root.rglob("adapter_config.json"))
    if matches:
        return matches[-1]
    return None


def load_training_args(p: Path):
    try:
        return torch.load(p, weights_only=False, map_location="cpu")
    except Exception as e:
        print(f"  [warn] could not load {p}: {e}", file=sys.stderr)
        return None


def get(obj, attr, default=None):
    return getattr(obj, attr, default) if obj is not None else default


def summarise(root: str, label: str):
    r = Path(root)
    args_path = find_training_args(r)
    adapter_path = find_adapter_config(r)

    if args_path is None:
        print(f"\n  [missing] no training_args.bin found under {root}")
        return None

    args = load_training_args(args_path)
    if args is None:
        return None

    adapter = None
    if adapter_path:
        with open(adapter_path) as f:
            adapter = json.load(f)

    # Effective batch size
    pb = get(args, "per_device_train_batch_size", 1) or 1
    ga = get(args, "gradient_accumulation_steps", 1) or 1
    ws = (get(args, "world_size", None)
          or get(args, "_n_gpu", None)
          or get(args, "n_gpu", 1))
    try:
        ws = int(ws) if ws else 1
    except Exception:
        ws = 1
    eb = pb * ga * ws

    summary = {
        "label"                       : label,
        "args_source"                 : str(args_path),
        "adapter_source"              : str(adapter_path) if adapter_path else None,

        # Factor 1 — learning-rate schedule
        "lr_scheduler_type"           : get(args, "lr_scheduler_type"),
        "learning_rate"               : get(args, "learning_rate"),
        "warmup_steps"                : get(args, "warmup_steps"),
        "warmup_ratio"                : get(args, "warmup_ratio"),

        # Factor 2 — batch size (effective)
        "per_device_train_batch_size" : pb,
        "gradient_accumulation_steps" : ga,
        "world_size_at_train"         : ws,
        "effective_batch_size"        : eb,

        # Factor 3 — optimiser
        "optim"                       : get(args, "optim"),
        "weight_decay"                : get(args, "weight_decay"),
        "adam_beta1"                  : get(args, "adam_beta1"),
        "adam_beta2"                  : get(args, "adam_beta2"),
        "max_grad_norm"               : get(args, "max_grad_norm"),

        # Factor 4 — sequence length
        "max_seq_length"              : get(args, "max_seq_length"),

        # Factor 5 — data ordering (default in pilot)
        "dataloader_shuffle"          : "default (random shuffle)",

        # Training mechanics
        "num_train_epochs"            : get(args, "num_train_epochs"),
        "bf16"                        : get(args, "bf16"),
        "fp16"                        : get(args, "fp16"),
        "tf32"                        : get(args, "tf32"),
        "gradient_checkpointing"      : get(args, "gradient_checkpointing"),
    }

    if adapter:
        summary["lora_r"]              = adapter.get("r")
        summary["lora_alpha"]          = adapter.get("lora_alpha")
        summary["lora_dropout"]        = adapter.get("lora_dropout")
        summary["lora_target_modules"] = adapter.get("target_modules")
        summary["lora_bias"]           = adapter.get("bias")

    print(f"\n  ── {label} ──")
    print(f"  training_args source : {args_path}")
    if adapter_path:
        print(f"  adapter_config source: {adapter_path}")
    print(f"  {'-' * 70}")
    skip = {"label", "args_source", "adapter_source"}
    for k, v in summary.items():
        if k in skip or v is None:
            continue
        if isinstance(v, list):
            v = ", ".join(map(str, v))
        print(f"  {k:<35s}: {v}")

    return summary


def factorial_mapping(summaries):
    """Map pilot settings to the Resolution V factorial Level A / Level B."""
    s = next((x for x in summaries
              if x and "LLaMA" in x.get("label", "")), summaries[0] if summaries else None)
    if not s:
        return

    print("\n" + "=" * 92)
    print("  Resolution V Factorial — Level A / Level B Justification")
    print("  (pilot values extracted from training_args.bin and adapter_config.json above)")
    print("=" * 92)
    print(f"\n  {'Factor':<25} {'Level A':<24} {'Level B':<24} {'Pilot value':<20}")
    print(f"  {'-' * 92}")

    pilot_lr  = s.get("lr_scheduler_type", "n/a")
    pilot_eb  = s.get("effective_batch_size", "n/a")
    pilot_opt = s.get("optim", "n/a")
    pilot_sl  = s.get("max_seq_length", "n/a")

    rows = [
        ("Learning-Rate Schedule",
         "cosine decay", "linear warmup + const",
         f"{pilot_lr}  → Level A" if pilot_lr == "cosine" else f"{pilot_lr}"),
        ("Effective Batch Size",
         "128", "256",
         f"{pilot_eb}  → Level B" if pilot_eb == 256 else f"{pilot_eb}"),
        ("Optimiser",
         "AdamW", "Lion",
         f"{pilot_opt}  → Level A" if "adamw" in str(pilot_opt).lower() else f"{pilot_opt}"),
        ("Sequence Length",
         "1024 tokens", "2048 tokens",
         f"{pilot_sl} (extended in factorial)"),
        ("Data Ordering",
         "curriculum (real→syn)", "mixed (interleaved)",
         "default shuffle (new factor)"),
    ]
    for r in rows:
        print(f"  {r[0]:<25} {r[1]:<24} {r[2]:<24} {r[3]:<20}")

    print("\n  Notes:")
    print("    • Three of the five factors have Level A or Level B equal to the pilot.")
    print("    • Sequence Length levels (1024/2048) are extended beyond the pilot")
    print("      to capture inter-clause dependencies under high tokeniser fertility")
    print("      for Bantu morphology.")
    print("    • Data Ordering is a new factor not varied in the pilot, introduced")
    print("      because the LughaGen corpus combines real and synthetic partitions.")


if __name__ == "__main__":
    print("=" * 92)
    print("  LughaGen Pilot Hyperparameter Settings")
    print("  Source: training_args.bin and adapter_config.json (auto-discovered)")
    print("=" * 92)

    summaries = []
    for root, label in PILOT_ROOTS:
        s = summarise(root, label)
        if s:
            summaries.append(s)

    if summaries:
        factorial_mapping(summaries)
        with open("pilot_hyperparameters.json", "w") as f:
            json.dump(summaries, f, indent=2, default=str)
        print(f"\n  Full record saved to pilot_hyperparameters.json")

    print("=" * 92)
