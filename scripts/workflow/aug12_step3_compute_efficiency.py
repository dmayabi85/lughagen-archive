"""
Compute GPU-hours, PPH, and Pareto frontier for Chapter 6.
Combines training time + evaluation numbers.
"""

import json
from pathlib import Path

# Manual data — populate from actual training runs
# Fill these in from your actual training log times (already in your context)
model_data = {
    "LughaGen-Llama-CPT-S137": {
        "wall_clock_hours": 24,  # Update with actual — approx from log timestamps
        "n_gpus": 8,
        "pooled_bpc": 1.269,
        "improvement_over_base": 1.5,  # From Llama-8B untuned baseline
    },
    "LughaGen-Llama-FFT-S137": {
        "wall_clock_hours": 24,  # Same as CPT approximately
        "n_gpus": 8,
        "pooled_bpc": 1.210,
        "improvement_over_base": 1.5,
    },
    "LughaGen-Llama-QLoRA-S137": {
        "wall_clock_hours": 24,
        "n_gpus": 8,
        "pooled_bpc": 1.312,
        "improvement_over_base": 1.5,
    },
    "LughaGen-Gemma-CPT-S137": {
        "wall_clock_hours": 96,  # ~4 days
        "n_gpus": 8,
        "pooled_bpc": 1.783,
        "improvement_over_base": 2.127,  # Gemma-27B untuned
    },
    "LughaGen-Gemma-FFT-S137": {
        "wall_clock_hours": 90,  # ETA 90 hours
        "n_gpus": 8,
        "pooled_bpc": None,  # TBD — will populate Aug 11
        "improvement_over_base": 2.127,
    },
}

print("=== Compute Efficiency Analysis ===")
print()
print(f"{'Model':<28} {'Hours':>8} {'GPU-hrs':>10} {'BPC':>8} {'ΔBPC':>8} {'PPH':>10}")
print("-" * 80)

results = []
for name, d in model_data.items():
    if d["pooled_bpc"] is None:
        continue
    gpu_hours = d["wall_clock_hours"] * d["n_gpus"]
    delta_bpc = d["improvement_over_base"] - d["pooled_bpc"]
    pph = delta_bpc / gpu_hours * 1000  # per 1000 GPU-hours
    results.append({
        "name": name,
        "wall_clock_hours": d["wall_clock_hours"],
        "gpu_hours": gpu_hours,
        "pooled_bpc": d["pooled_bpc"],
        "delta_bpc": delta_bpc,
        "pph": pph,
    })
    print(f"{name:<28} {d['wall_clock_hours']:>8} {gpu_hours:>10.0f} "
          f"{d['pooled_bpc']:>8.3f} {delta_bpc:>8.3f} {pph:>10.4f}")

# Pareto frontier: given BPC and GPU-hours, which are Pareto-optimal?
print()
print("=== Pareto Frontier ===")
# Model X dominates model Y if BPC_X ≤ BPC_Y AND GPU_hours_X ≤ GPU_hours_Y
# with at least one strict inequality
sorted_by_bpc = sorted(results, key=lambda r: r["pooled_bpc"])

pareto = []
best_hours_so_far = float('inf')
for r in sorted_by_bpc:
    if r["gpu_hours"] < best_hours_so_far:
        pareto.append(r["name"])
        best_hours_so_far = r["gpu_hours"]

print("Pareto-optimal (best BPC per GPU-hour spent):")
for name in pareto:
    r = next(x for x in results if x["name"] == name)
    print(f"  {name} — BPC {r['pooled_bpc']:.3f} in {r['gpu_hours']:.0f} GPU-hrs")

# Save
output = {
    "model_data": model_data,
    "efficiency_results": results,
    "pareto_frontier": pareto,
}
with open("/workspace/results/compute_efficiency_pareto.json", "w") as f:
    json.dump(output, f, indent=2, default=str)
print(f"\n[write] /workspace/results/compute_efficiency_pareto.json")
