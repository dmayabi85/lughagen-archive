#!/usr/bin/env python3
"""
FFT smoke test for Phase 3 infrastructure.

Isolates the training_loop.py FFT patches (bypasses train_orchestrator.py).
Uses smoke_test=True to get 150 steps, eval at 50, save at 150, log every 5.

Success criteria:
  1. Model loads (bf16, no bnb, DeepSpeed ZeRO-3 sharding across 8 GPUs)
  2. '[FFT] trainable params: ~8B' logged (edit C branch fired)
  3. Loss decreases from ~4 to below 3.8 by step 150
  4. Checkpoint saves at step 150
  5. No crashes

Launch:
  torchrun --nproc_per_node=8 --nnodes=1 /workspace/scripts/smoke_fft.py
"""
import datetime as dt
import sys
import uuid
from pathlib import Path

sys.path.insert(0, "/workspace/scripts")
from training_loop import RunConfig, run_training  # noqa: E402


def main():
    output_dir = Path("/workspace/checkpoints/SMOKE-P3-FFT-01")
    output_dir.mkdir(parents=True, exist_ok=True)

    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    config = RunConfig(
        # Identity
        factorial_run_id="SMOKE-P3-FFT-01",
        experiment_id=f"EXP-SMOKE-{ts}-{uuid.uuid4().hex[:8].upper()}",
        phase="Phase3_PEFT",
        spec_version="1.5.1",
        seed=137,

        # Factor levels (C05 config, batch reduced for FFT memory safety)
        lr_schedule="cosine",
        batch_size=128,                 # FFT: half C05's 256 to reduce OOM risk on first attempt
        optimiser="AdamW",
        sequence_length=1024,
        data_ordering="curriculum",

        # Hyperparameters — FFT-specific
        peak_lr=2e-5,                   # 10x lower than QLoRA (standard FFT rate)
        warmup_steps=20,                # shorter for 150-step smoke
        weight_decay=0.01,
        max_grad_norm=1.0,

        # THE KEY SWITCHES for FFT
        use_qlora_4bit=False,
        deepspeed_config="/workspace/configs/ds_config_zero3.json",

        # Smoke mode — RunConfig.smoke_test_overrides() will set
        # max_steps=150, eval_steps=50, save_steps=150, logging_steps=5
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
    print("SMOKE TEST: FFT infrastructure (Phase 3)")
    print("=" * 70)
    print(f"  Run ID:         {config.factorial_run_id}")
    print(f"  Phase:          {config.phase}")
    print(f"  Method:         Full Fine-Tuning (use_qlora_4bit={config.use_qlora_4bit})")
    print(f"  DeepSpeed cfg:  {config.deepspeed_config}")
    print(f"  Batch size:     {config.batch_size}")
    print(f"  Peak LR:        {config.peak_lr}")
    print(f"  Smoke mode:     True (max_steps will be reset to 150 by run_training)")
    print(f"  Corpus (train): {config.train_file}")
    print(f"  Corpus (val):   {config.val_file}")
    print(f"  Output:         {output_dir}")
    print("=" * 70, flush=True)
    print()

    # Fire
    result = run_training(config)

    # Report
    print("\n" + "=" * 70)
    print("SMOKE TEST COMPLETE")
    print("=" * 70)
    if isinstance(result, dict):
        for k, v in sorted(result.items()):
            print(f"  {k:40s} = {v}")
    print()
    print("Inspect checkpoint output:")
    print(f"  ls -la {output_dir}/")


if __name__ == "__main__":
    main()
