#!/usr/bin/env python3
"""
LughaGen Phase 2 factorial analysis.
Reads runs.csv, filters to real Phase 2 cell rows, computes main effects,
2-factor interactions, and noise floor.  Uses validation_perplexity
(lower is better) as the response metric.

Usage:
  python3 analyze_factorial.py \
    --input /workspace/registry/runs.csv \
    --output /workspace/results/phase2_final_analysis.md
"""
import csv
import argparse
import statistics
import sys
from pathlib import Path
from collections import defaultdict
from itertools import combinations

FACTORS = [
    "optimiser",
    "sequence_length",
    "batch_size",
    "lr_schedule",
    "data_ordering",
]

METRIC_KEY = "metrics.validation_perplexity"  # lower is better


def load_cells(csv_path):
    """Filter to real Phase 2 cell rows only."""
    rows = []
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            if r.get("phase") != "Phase2_Factorial":
                continue
            if not r.get("factorial_run_id", "").startswith("RUN-P2-C"):
                continue
            try:
                r["_pph"] = float(r[METRIC_KEY])
            except (KeyError, ValueError, TypeError):
                continue
            rows.append(r)
    return rows


def main_effect(rows, factor):
    key = f"factors.{factor}"
    by_level = defaultdict(list)
    for r in rows:
        by_level[r[key]].append(r["_pph"])
    levels = sorted(by_level.keys())
    if len(levels) != 2:
        return None
    lo, hi = levels
    return {
        "factor": factor,
        "level_low": lo,
        "level_high": hi,
        "mean_low": statistics.mean(by_level[lo]),
        "mean_high": statistics.mean(by_level[hi]),
        "effect": statistics.mean(by_level[hi]) - statistics.mean(by_level[lo]),
        "n_low": len(by_level[lo]),
        "n_high": len(by_level[hi]),
    }


def two_way_interaction(rows, f1, f2):
    k1, k2 = f"factors.{f1}", f"factors.{f2}"
    cell = defaultdict(list)
    for r in rows:
        cell[(r[k1], r[k2])].append(r["_pph"])
    if len(cell) != 4:
        return None
    means = {k: statistics.mean(v) for k, v in cell.items()}
    ks = sorted(cell.keys())
    (l1, l2), (_, h2), (h1, _), (_, _) = ks
    m_ll = means[(l1, l2)]
    m_lh = means[(l1, h2)]
    m_hl = means[(h1, l2)]
    m_hh = means[(h1, h2)]
    interaction = 0.5 * ((m_hh - m_hl) - (m_lh - m_ll))
    return {
        "factors": (f1, f2),
        "interaction": interaction,
        "cell_means": {f"{k[0]}|{k[1]}": v for k, v in means.items()},
    }


def noise_estimate(rows):
    """Range within same-cell different-seed pairs."""
    by_cell = defaultdict(list)
    for r in rows:
        cell = tuple(r[f"factors.{f}"] for f in FACTORS)
        by_cell[cell].append(r["_pph"])
    ranges = [max(v) - min(v) for v in by_cell.values() if len(v) >= 2]
    return {
        "n_pairs": len(ranges),
        "mean_range": statistics.mean(ranges) if ranges else 0.0,
        "median_range": statistics.median(ranges) if ranges else 0.0,
        "stdev_of_ranges": statistics.stdev(ranges) if len(ranges) > 1 else 0.0,
    }


def format_report(rows, noise, main_effects, interactions):
    best = min(rows, key=lambda r: r["_pph"])
    worst = max(rows, key=lambda r: r["_pph"])

    lines = [
        "# LughaGen Phase 2 Factorial Analysis",
        "",
        f"Response metric: `{METRIC_KEY}` (lower is better)",
        f"Cells analysed: **{len(rows)}**",
        "",
        "## Best and worst runs",
        "",
        f"- **Best:** {best['factorial_run_id']} — pph = **{best['_pph']:.3f}**",
    ]
    for f in FACTORS:
        lines.append(f"  - {f}: {best[f'factors.{f}']}")
    lines.append(f"  - seed: {best['seed']}")
    lines.append(f"  - wall clock: {float(best.get('wall_clock_seconds', 0)) / 3600:.1f} h")
    lines.append("")
    lines.append(f"- **Worst:** {worst['factorial_run_id']} — pph = **{worst['_pph']:.3f}**")
    for f in FACTORS:
        lines.append(f"  - {f}: {worst[f'factors.{f}']}")
    lines.append("")
    lines.append(f"- **Spread (worst - best):** {worst['_pph'] - best['_pph']:.3f} pph")
    lines.append("")

    nf = noise["mean_range"]
    lines += [
        "## Noise floor (same-cell seed variance)",
        "",
        f"- Pairs: {noise['n_pairs']}",
        f"- Mean within-cell range: **{nf:.4f}** pph",
        f"- Median within-cell range: {noise['median_range']:.4f} pph",
        f"- Stdev of ranges: {noise['stdev_of_ranges']:.4f} pph",
        "",
        "The mean within-cell range is the effective noise floor for interpreting all effects below.",
        "",
        "## Main effects",
        "",
        "Ranked by |effect| / noise_floor. Effect = mean(high) - mean(low).",
        "Negative effect means the 'high' level improves (lowers) perplexity.",
        "",
        "| Factor | Low | High | Mean(low) | Mean(high) | Effect | |Effect|/noise |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for e in sorted(main_effects, key=lambda x: abs(x["effect"]), reverse=True):
        ratio = abs(e["effect"]) / nf if nf else float("inf")
        lines.append(
            f"| **{e['factor']}** | {e['level_low']} | {e['level_high']} "
            f"| {e['mean_low']:.3f} | {e['mean_high']:.3f} "
            f"| **{e['effect']:+.3f}** | **{ratio:.1f}x** |"
        )
    lines.append("")

    lines += [
        "## Two-factor interactions",
        "",
        "Ranked by |interaction| / noise_floor.",
        "",
        "| Factor 1 | Factor 2 | Interaction | |Interaction|/noise |",
        "|---|---|---:|---:|",
    ]
    for i in sorted(interactions, key=lambda x: abs(x["interaction"]), reverse=True):
        ratio = abs(i["interaction"]) / nf if nf else float("inf")
        lines.append(
            f"| {i['factors'][0]} | {i['factors'][1]} "
            f"| {i['interaction']:+.3f} | {ratio:.1f}x |"
        )
    lines.append("")

    lines += [
        "## All cells (sorted by pph)",
        "",
        "| Run ID | pph | opt | seq | batch | lr | order | seed | wall (h) |",
        "|---|---:|---|---|---|---|---|---:|---:|",
    ]
    for r in sorted(rows, key=lambda x: x["_pph"]):
        wall_h = float(r.get("wall_clock_seconds", 0)) / 3600
        lines.append(
            f"| {r['factorial_run_id']} | {r['_pph']:.3f} "
            f"| {r['factors.optimiser']} | {r['factors.sequence_length']} "
            f"| {r['factors.batch_size']} | {r['factors.lr_schedule']} "
            f"| {r['factors.data_ordering']} | {r['seed']} | {wall_h:.1f} |"
        )
    lines.append("")

    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    rows = load_cells(args.input)
    if not rows:
        sys.exit("No Phase 2 cell rows found.")
    print(f"Loaded {len(rows)} Phase 2 cell rows.")

    noise = noise_estimate(rows)
    main_effects = [me for f in FACTORS if (me := main_effect(rows, f))]
    interactions = [
        ix for f1, f2 in combinations(FACTORS, 2)
        if (ix := two_way_interaction(rows, f1, f2))
    ]

    report = format_report(rows, noise, main_effects, interactions)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report)
    print(f"Wrote {out}")

    # Also print the summary tables to stdout for quick eyeball
    print("\n" + "=" * 70)
    print("SUMMARY (see full report for details)")
    print("=" * 70)
    nf = noise["mean_range"]
    print(f"\nBest cell: {min(rows, key=lambda r: r['_pph'])['factorial_run_id']} "
          f"@ pph = {min(r['_pph'] for r in rows):.3f}")
    print(f"Noise floor: {nf:.4f} pph")
    print(f"\nMain effects (sorted by |effect|):")
    for e in sorted(main_effects, key=lambda x: abs(x["effect"]), reverse=True):
        ratio = abs(e["effect"]) / nf if nf else float("inf")
        arrow = "-> lowers pph" if e["effect"] < 0 else "-> raises pph"
        print(f"  {e['factor']:20} {e['level_low']:>10} vs {e['level_high']:<10} "
              f"effect = {e['effect']:+7.3f}   ({ratio:5.1f}x noise, {arrow})")


if __name__ == "__main__":
    main()
