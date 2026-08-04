#!/usr/bin/env python3
"""
CPT smoke test for Phase 3 infrastructure.

Same code path as FFT (use_qlora_4bit=False, DeepSpeed ZeRO-3), but with
CPT-specific hyperparameters:
  - peak_lr=1e-5 (half of FFT's 2e-5)
  - data_ordering="mixed" (not curriculum)
  - longer warmup

If FFT smoke passed, this should also pass. The purpose is to verify the
CPT config choices don't destabilize training.

Success criteria (see /workspace/logs/smoke_cpt_*.log):
  1. Model loads (same path as FFT)
  2. Loss decreases from ~4 to below 3.8 by step 150
  3. Checkpoint saves
  4. No crashes

Launch:
  torchrun --nproc_per_node=8 --nnodes=1 /workspace/scripts/smoke_cpt.py
"""
import datetime as dt
import sys
import uuid
from pathlib import Path

sys.path.insert(0, "/workspace/scripts")
from training_loop import RunConfig, run_training  # noqa: E402


def main():
    output_dir = Path("/workspace/checkpoints/SMOKE-P3-CPT-01")
    output_dir.mkdir(parents=True, exist_ok=True)

    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    config = RunConfig(
        # Identity
        factorial_run_id="SMOKE-P3-CPT-01",
        experiment_id=f"EXP-SMOKE-{ts}-{uuid.uuid4().hex[:8].upper()}",
        phase="Phase3_PEFT",
        spec_version="1.5.1",
        seed=137,

        # Factor levels (C05 config, CPT-specific data ordering)
        lr_schedule="cosine",
        batch_size=128,
        optimiser="AdamW",
        sequence_length=1024,
        data_ordering="mixed",          # CPT: natural order, no curriculum

        # Hyperparameters — CPT-specific
        peak_lr=1e-5,                   # CPT: 20x lower than QLoRA, half of FFT
        warmup_steps=30,                # slightly longer than FFT for stability
        weight_decay=0.1,               # standard pre-training value
        max_grad_norm=1.0,

        # THE KEY SWITCHES (same as FFT)
        use_qlora_4bit=False,
        deepspeed_config="/workspace/configs/ds_config_zero3.json",

        # Smoke mode: max_steps=150, eval_steps=50, save_steps=150, logging_steps=5
        smoke_test=True,

        # Corpus paths — filtered clean corpus
        train_file="/workspace/data/corpus/splits_v2/pathD_v2_filtered/train.jsonl",
        val_file="/workspace/data/corpus/splits_v2/pathD_v2_filtered/validation_shared_v3.jsonl",

        # Output
        output_dir=output_dir,

        # Bookkeeping
        started_at=dt.datetime.now(dt.timezone.utc).isoformat(),
    )

    # Header
    print("=" * 70)
    print("SMOKE TEST: CPT infrastructure (Phase 3)")
    print("=" * 70)
    print(f"  Run ID:         {config.factorial_run_id}")
    print(f"  Phase:          {config.phase}")
    print(f"  Method:         Continued Pre-Training (use_qlora_4bit={config.use_qlora_4bit})")
    print(f"  DeepSpeed cfg:  {config.deepspeed_config}")
    print(f"  Batch size:     {config.batch_size}")
    print(f"  Peak LR:        {config.peak_lr}")
    print(f"  Warmup steps:   {config.warmup_steps}")
    print(f"  Weight decay:   {config.weight_decay}")
    print(f"  Data ordering:  {config.data_ordering}")
    print(f"  Smoke mode:     True (max_steps=150)")
    print(f"  Corpus (train): {config.train_file}")
    print(f"  Corpus (val):   {config.val_file}")
    print(f"  Output:         {output_dir}")
    print("=" * 70, flush=True)
    print()

    # Fire
    result = run_training(config)

    # Report (rank-0 only would be cleaner but this is fine for smoke)
    import os
    if int(os.environ.get("RANK", "0")) == 0:
        print("\n" + "=" * 70)
        print("SMOKE TEST COMPLETE (CPT)")
        print("=" * 70)
        if isinstance(result, dict):
            for k, v in sorted(result.items()):
                print(f"  {k:40s} = {v}")
        print()
        print("Inspect checkpoint output:")
        print(f"  ls -la {output_dir}/")


if __name__ == "__main__":
    main()
