"""
Generate Chapter 5/6 per-language BPC comparison table.
Reads all LughaGen model per-language eval JSONs and creates a formatted table.
"""

import json
from pathlib import Path

RESULTS = Path("/workspace/results/")

models = {
    "LughaGen-Llama-CPT-S137": "P3_LLAMA_CPT_S137_per_language_eval_CLEAN.json",
    "LughaGen-Llama-FFT-S137": "P3_LLAMA_FFT_S137_per_language_eval_CLEAN.json",
    "LughaGen-Llama-QLoRA-S137": "P3_LLAMA_QLORA_S137_per_language_eval_CLEAN.json",
    "LughaGen-Gemma-CPT-S137": "P3.5_GEMMA_CPT_S137_per_language_eval_CLEAN.json",
    "LughaGen-Gemma-FFT-S137": "P3.5_GEMMA_FFT_S137_per_language_eval_CLEAN.json",
}

languages = ["swahili_authentic", "kikuyu", "luo", "kamba", "sheng"]

# Load all
data = {}
for name, filename in models.items():
    path = RESULTS / filename
    if path.exists():
        with path.open() as f:
            data[name] = json.load(f)

# Build table
print("=" * 100)
print("PER-LANGUAGE BPC — All 5 LughaGen Models")
print("=" * 100)
print()

# Header row
langs_short = {"swahili_authentic": "Swahili", "kikuyu": "Kikuyu",
                "luo": "Luo", "kamba": "Kamba", "sheng": "Sheng"}
header = f"{'Model':<28}"
for lang in languages:
    header += f"{langs_short.get(lang, lang):>10}"
header += f"{'Pooled':>10}"
print(header)
print("-" * 100)

# Data rows
for name in models.keys():
    if name not in data:
        row = f"{name:<28}" + "".join(f"{'—':>10}" for _ in languages) + f"{'—':>10}"
        print(row)
        continue
    row = f"{name:<28}"
    for lang in languages:
        pl = data[name].get("per_language", {}).get(lang, {})
        bpc = pl.get("bpc", "?")
        if isinstance(bpc, float):
            row += f"{bpc:>10.4f}"
        else:
            row += f"{'?':>10}"
    pooled = data[name].get("pooled", {}).get("bpc", "?")
    if isinstance(pooled, float):
        row += f"{pooled:>10.4f}"
    else:
        row += f"{'?':>10}"
    print(row)

# Deltas
print()
print("=" * 100)
print("Δ ANALYSIS")
print("=" * 100)
print()

# Δ Gemma-FFT vs Gemma-CPT (adaptation effect)
if "LughaGen-Gemma-CPT-S137" in data and "LughaGen-Gemma-FFT-S137" in data:
    print("Δ Gemma FFT vs CPT (Adaptation effect):")
    for lang in languages + ["pooled"]:
        cpt = (data["LughaGen-Gemma-CPT-S137"].get("per_language", {}).get(lang, {}) if lang != "pooled"
               else data["LughaGen-Gemma-CPT-S137"].get("pooled", {})).get("bpc")
        fft = (data["LughaGen-Gemma-FFT-S137"].get("per_language", {}).get(lang, {}) if lang != "pooled"
               else data["LughaGen-Gemma-FFT-S137"].get("pooled", {})).get("bpc")
        if isinstance(cpt, float) and isinstance(fft, float):
            delta = fft - cpt
            pct = (cpt - fft) / cpt * 100
            marker = "✓" if fft < cpt else "✗"
            print(f"  {langs_short.get(lang, lang):<12} CPT: {cpt:.4f} → FFT: {fft:.4f} "
                  f"(Δ: {delta:+.4f}, {pct:+.1f}%) {marker}")

# Δ Gemma-FFT vs Llama-FFT (scale effect)
if "LughaGen-Llama-FFT-S137" in data and "LughaGen-Gemma-FFT-S137" in data:
    print("\nΔ Gemma-FFT vs Llama-FFT (Scale effect):")
    for lang in languages + ["pooled"]:
        llama = (data["LughaGen-Llama-FFT-S137"].get("per_language", {}).get(lang, {}) if lang != "pooled"
                 else data["LughaGen-Llama-FFT-S137"].get("pooled", {})).get("bpc")
        gemma = (data["LughaGen-Gemma-FFT-S137"].get("per_language", {}).get(lang, {}) if lang != "pooled"
                 else data["LughaGen-Gemma-FFT-S137"].get("pooled", {})).get("bpc")
        if isinstance(llama, float) and isinstance(gemma, float):
            delta = gemma - llama
            marker = "✓ Gemma wins" if gemma < llama else "○ Llama wins"
            print(f"  {langs_short.get(lang, lang):<12} Llama: {llama:.4f} vs Gemma: {gemma:.4f} "
                  f"(Δ: {delta:+.4f}) {marker}")

# Save to file for Chapter 6
output_file = RESULTS / "per_language_summary_table.txt"
# (Would repeat the print block writing to file — omitted for brevity)
print(f"\n\n(Also save to: {output_file})")
