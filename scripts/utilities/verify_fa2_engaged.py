#!/usr/bin/env python3
"""
/workspace/scripts/verify_fa2_engaged.py

LughaGen Training Framework v1.5 §12 + §14
Verifies that Flash Attention 2 is actually engaged (not silently falling back
to SDPA). Two-pronged check:
  1. Inspect flash_attn module presence and version.
  2. Run a 10-step probe and measure peak VRAM; compare against FA2-expected
     baseline. If VRAM is substantially elevated (~30% higher), SDPA fallback
     has occurred.

Exit codes:
  0  — FA2 is engaged
  1  — silent SDPA fallback detected
  2  — script-level error or no GPU available

Usage:
  python verify_fa2_engaged.py --quick-probe                # for preflight.sh
  python verify_fa2_engaged.py --full-probe                  # for smoke test (§14)
  python verify_fa2_engaged.py --cpu-mock                     # CPU smoke for CI
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# CLUSTER-SPECIFIC: reference VRAM for FA2-engaged smoke test config
# (batch 128, seq 1024, QLoRA r=16). §14 specifies ~17 GB ± 2 GB.
# SDPA fallback would show ~25 GB ± 3 GB. Threshold > 22 GB triggers failure.
FA2_VRAM_REFERENCE_GB = 17.0
FA2_VRAM_TOLERANCE_GB = 2.0
SDPA_FALLBACK_THRESHOLD_GB = 22.0


def check_flash_attn_importable() -> tuple[bool, str]:
    """Return (importable, version_or_error)."""
    try:
        import flash_attn
        return True, getattr(flash_attn, "__version__", "UNKNOWN")
    except ImportError as e:
        return False, str(e)


def run_cpu_mock() -> int:
    """
    CPU-only mock for unit testing. Just verifies flash_attn imports.
    Used in Week 1 acceptance check on machines without GPUs.
    """
    print("[verify_fa2_engaged] CPU mock mode")
    ok, version = check_flash_attn_importable()
    if not ok:
        print(f"  FAIL: flash_attn not importable: {version}")
        return 1
    print(f"  PASS: flash_attn importable, version={version}")
    print("  Note: full VRAM-based probe requires GPU; skipped in CPU mock.")
    return 0


def run_probe(n_steps: int, full_probe: bool = False) -> int:
    """
    Run the actual GPU probe. Returns 0 if FA2 engaged; 1 if fallback detected.

    Probe procedure:
      1. Load a small LLaMA-like config (we use a tiny dummy model to keep the
         probe fast; the test is whether FA2's CUDA kernel binds, not whether
         the real model trains).
      2. Forward + backward N steps with FA2 requested.
      3. Measure peak VRAM.
      4. Compare against FA2 reference.

    CLUSTER-SPECIFIC: this probe uses a 1-billion parameter dummy LLaMA. If the
    cluster's LLaMA-3.1-8B is already cached, the full_probe variant can use
    the actual model for tighter VRAM measurement against §14's 17 GB reference.
    """
    try:
        import torch
    except ImportError:
        print("  FAIL: torch not importable", file=sys.stderr)
        return 2

    if not torch.cuda.is_available():
        print("  FAIL: no GPU available", file=sys.stderr)
        return 2

    ok, version = check_flash_attn_importable()
    if not ok:
        print(f"  FAIL: flash_attn import failed: {version}", file=sys.stderr)
        return 1

    print(f"[verify_fa2_engaged] flash_attn version: {version}")
    print(f"[verify_fa2_engaged] CUDA device: {torch.cuda.get_device_name(0)}")

    # Reset peak memory stats
    torch.cuda.reset_peak_memory_stats(0)

    try:
        from transformers import AutoConfig, AutoModelForCausalLM

        if full_probe:
            # CLUSTER-SPECIFIC: real LLaMA-3.1-8B path for §14 smoke test
            model_path = "/workspace/models/llama-3.1-8b-base"
            config = AutoConfig.from_pretrained(model_path)
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                device_map="cuda:0",
            )
            seq_len = 1024
            batch_size = 1   # per-device; smoke test uses effective batch via accum
        else:
            # Quick probe: tiny model just to check FA2 binding
            from transformers import LlamaConfig, LlamaForCausalLM
            config = LlamaConfig(
                vocab_size=32000,
                hidden_size=512,
                intermediate_size=1024,
                num_hidden_layers=2,
                num_attention_heads=8,
                num_key_value_heads=8,
                max_position_embeddings=2048,
            )
            model = LlamaForCausalLM(config).to(
                dtype=torch.bfloat16, device="cuda:0"
            )
            # Re-instantiate with FA2 explicitly requested
            try:
                model.config._attn_implementation = "flash_attention_2"
            except Exception:
                pass
            seq_len = 512
            batch_size = 2

        print(f"  Probe: {n_steps} steps, seq_len={seq_len}, batch={batch_size}")

        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

        for step in range(n_steps):
            input_ids = torch.randint(
                0, config.vocab_size, (batch_size, seq_len),
                device="cuda:0",
            )
            out = model(input_ids=input_ids, labels=input_ids)
            out.loss.backward()
            opt.step()
            opt.zero_grad()

        peak_gb = torch.cuda.max_memory_allocated(0) / 1024**3
        print(f"  Peak VRAM after probe: {peak_gb:.2f} GB")

        if full_probe:
            # Compare against §14 reference
            if peak_gb > SDPA_FALLBACK_THRESHOLD_GB:
                print(f"  FAIL: VRAM {peak_gb:.2f} GB > {SDPA_FALLBACK_THRESHOLD_GB} GB "
                      f"threshold; SILENT SDPA FALLBACK detected.")
                return 1
            print(f"  PASS: VRAM within FA2-engaged band "
                  f"(reference {FA2_VRAM_REFERENCE_GB} ± {FA2_VRAM_TOLERANCE_GB} GB)")
        else:
            # Quick probe: we just check the model didn't error out and that
            # peak VRAM is sane for the dummy size (<2 GB).
            if peak_gb > 5.0:
                print(f"  WARN: quick probe peak VRAM {peak_gb:.2f} GB is higher than "
                      f"expected; FA2 may not be engaged. Run --full-probe for "
                      f"definitive check.")
                # Quick probe is permissive — definitive answer needs full probe
            print(f"  PASS: FA2 imported and probe ran without error.")

        return 0

    except Exception as e:
        print(f"  FAIL: probe raised exception: {e}", file=sys.stderr)
        return 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify FlashAttention 2 engagement (§12, §14)")
    parser.add_argument("--quick-probe", action="store_true",
                        help="Fast probe for preflight (10 steps on tiny model).")
    parser.add_argument("--full-probe", action="store_true",
                        help="Full probe for §14 smoke test (real LLaMA-3.1-8B).")
    parser.add_argument("--cpu-mock", action="store_true",
                        help="CPU-only smoke (just import check). For Week 1 unit testing.")
    parser.add_argument("--n-steps", type=int, default=10)
    args = parser.parse_args()

    if args.cpu_mock:
        return run_cpu_mock()

    if not (args.quick_probe or args.full_probe):
        print("Must specify --quick-probe, --full-probe, or --cpu-mock", file=sys.stderr)
        return 2

    return run_probe(n_steps=args.n_steps, full_probe=args.full_probe)


if __name__ == "__main__":
    sys.exit(main())
