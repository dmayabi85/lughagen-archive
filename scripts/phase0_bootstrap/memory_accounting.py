"""
LLaMA-3.1-8B memory accounting.

Demonstrates why full fine-tuning of LLaMA-3.1-8B exceeds a single A100 80GB
and why QLoRA fits within ~16 GB per GPU.

Run:  python memory_accounting.py
No GPU required. Pure arithmetic from documented parameter counts and
bytes-per-parameter for each numerical format.
"""

# Empirically measured from `model.num_parameters()` on
# meta-llama/Meta-Llama-3.1-8B (see pilot training script):
TOTAL_PARAMS       = 8_030_261_248         # full model
LORA_R16_PARAMS    = 41_943_040            # LoRA r=16 trainable params
LORA_PCT           = LORA_R16_PARAMS / TOTAL_PARAMS * 100

GB = 1e9   # bytes per GB

def fmt(label, val_gb):
    print(f"  {label:<40s} {val_gb:>8.2f} GB")

print("=" * 70)
print("  LLaMA-3.1-8B Memory Accounting (per-GPU, AdamW optimiser)")
print("=" * 70)
print(f"  Total parameters       : {TOTAL_PARAMS:>15,}")
print(f"  LoRA r=16 trainable    : {LORA_R16_PARAMS:>15,}  ({LORA_PCT:.2f}%)")
print()

# ─────────────────────────────────────────────────────────────────────
# (A) Full fine-tuning: BF16 weights + BF16 gradients + FP32 AdamW state
# ─────────────────────────────────────────────────────────────────────
print("[A] FULL FINE-TUNING (BF16 weights, BF16 grads, FP32 AdamW states)")
w   = TOTAL_PARAMS * 2 / GB     # BF16 weights:    2 bytes/param
g   = TOTAL_PARAMS * 2 / GB     # BF16 gradients:  2 bytes/param
m   = TOTAL_PARAMS * 4 / GB     # AdamW momentum:  4 bytes/param (FP32)
v   = TOTAL_PARAMS * 4 / GB     # AdamW variance:  4 bytes/param (FP32)

fmt("Weights (BF16)",            w)
fmt("Gradients (BF16)",          g)
fmt("AdamW momentum (FP32)",     m)
fmt("AdamW variance (FP32)",     v)
ft_total = w + g + m + v
fmt("Parameter-state subtotal",  ft_total)
fmt("+ activations & KV cache",  10.0)
ft_with_act = ft_total + 10
fmt("Single-GPU footprint",      ft_with_act)
print(f"  vs A100 80GB capacity                    {'EXCEEDS' if ft_with_act > 80 else 'fits'}"
      f" {'by ' + f'{ft_with_act - 80:.1f} GB' if ft_with_act > 80 else ''}")
print(f"  → must shard across all 8 GPUs (ZeRO-3 / FSDP)")
print()

# ─────────────────────────────────────────────────────────────────────
# (B) QLoRA: 4-bit NF4 base + BF16 LoRA adapters
# ─────────────────────────────────────────────────────────────────────
print("[B] QLoRA (4-bit NF4 base + BF16 LoRA adapters)")
base    = TOTAL_PARAMS * 0.5 / GB                 # NF4: 0.5 bytes/param
lw      = LORA_R16_PARAMS * 2 / GB                # BF16 LoRA weights
lg      = LORA_R16_PARAMS * 2 / GB                # BF16 LoRA gradients
lm      = LORA_R16_PARAMS * 4 / GB                # AdamW momentum (FP32)
lv      = LORA_R16_PARAMS * 4 / GB                # AdamW variance (FP32)

fmt("Base weights (NF4, 4-bit)",   base)
fmt("LoRA weights (BF16)",         lw)
fmt("LoRA gradients (BF16)",       lg)
fmt("AdamW momentum on LoRA",      lm)
fmt("AdamW variance on LoRA",      lv)
qlora_total = base + lw + lg + lm + lv
fmt("Parameter-state subtotal",    qlora_total)
fmt("+ activations & KV cache",    11.0)
qlora_with_act = qlora_total + 11
fmt("Single-GPU footprint",        qlora_with_act)
print(f"  vs A100 80GB capacity                    "
      f"fits with ~{80 - qlora_with_act:.0f} GB headroom")
print()

# ─────────────────────────────────────────────────────────────────────
# Verdict
# ─────────────────────────────────────────────────────────────────────
print("─" * 70)
print(f"  Full FT footprint  : ~{ft_with_act:.1f} GB  → exceeds single A100 80GB")
print(f"  QLoRA footprint    : ~{qlora_with_act:.1f} GB  → fits comfortably")
print(f"  Compression factor : {ft_with_act/qlora_with_act:.1f}× less memory with QLoRA")
print("─" * 70)
