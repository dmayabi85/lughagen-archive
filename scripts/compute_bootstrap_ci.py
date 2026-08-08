"""
compute_bootstrap_ci.py

Compute bootstrap confidence intervals on BPC from raw evaluation losses.
Resamples the evaluation set with replacement N times, recomputes BPC each time,
returns percentile-based CI.

Usage:
    python compute_bootstrap_ci.py \
        --losses-jsonl /workspace/results/gemma_fft_per_record_losses.jsonl \
        --n-bootstrap 1000 \
        --output /workspace/results/gemma_fft_bootstrap_ci.json
"""

import argparse
import json
import numpy as np
from pathlib import Path


def bpc_from_records(records):
    """Compute BPC from a list of per-record eval results."""
    total_nll_nats = sum(r["loss"] * r["n_tokens"] for r in records)
    total_bytes = sum(r["n_bytes"] for r in records)
    if total_bytes == 0:
        return float('nan')
    ln2 = 0.6931471805599453
    return total_nll_nats / (total_bytes * ln2)


def bootstrap_ci(records, n_bootstrap=1000, alpha=0.05, seed=42):
    """Return (point_estimate, ci_low, ci_high)."""
    rng = np.random.default_rng(seed)
    n = len(records)
    
    point = bpc_from_records(records)
    bootstraps = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        sample = [records[i] for i in idx]
        bootstraps.append(bpc_from_records(sample))
    
    bootstraps.sort()
    ci_low = np.percentile(bootstraps, 100 * (alpha / 2))
    ci_high = np.percentile(bootstraps, 100 * (1 - alpha / 2))
    return point, ci_low, ci_high, bootstraps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--losses-jsonl", required=True,
                        help="JSONL with per-record: loss, n_tokens, n_bytes, language")
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--alpha", type=float, default=0.05,
                        help="1-confidence, e.g. 0.05 for 95% CI")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    records = []
    with open(args.losses_jsonl) as f:
        for line in f:
            records.append(json.loads(line))
    
    print(f"[bootstrap] {len(records)} records, {args.n_bootstrap} resamples")
    
    # Overall
    point, ci_low, ci_high, _ = bootstrap_ci(records, args.n_bootstrap, args.alpha, args.seed)
    print(f"  Pooled BPC: {point:.4f} (95% CI: {ci_low:.4f} - {ci_high:.4f})")
    
    # Per-language
    per_language = {}
    langs = sorted(set(r["language"] for r in records))
    for lang in langs:
        lang_records = [r for r in records if r["language"] == lang]
        if len(lang_records) < 10:
            continue
        point, ci_low, ci_high, _ = bootstrap_ci(lang_records, args.n_bootstrap, args.alpha, args.seed)
        per_language[lang] = {
            "n_records": len(lang_records),
            "bpc": round(point, 4),
            "ci_95_low": round(ci_low, 4),
            "ci_95_high": round(ci_high, 4),
            "ci_width": round(ci_high - ci_low, 4),
        }
        print(f"  {lang}: {point:.4f} (95% CI: {ci_low:.4f} - {ci_high:.4f})")
    
    # Pooled
    point, ci_low, ci_high, _ = bootstrap_ci(records, args.n_bootstrap, args.alpha, args.seed)
    output = {
        "n_records": len(records),
        "n_bootstrap": args.n_bootstrap,
        "alpha": args.alpha,
        "pooled": {
            "bpc": round(point, 4),
            "ci_95_low": round(ci_low, 4),
            "ci_95_high": round(ci_high, 4),
            "ci_width": round(ci_high - ci_low, 4),
        },
        "per_language": per_language,
    }
    
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"[write] {args.output}")


if __name__ == "__main__":
    main()
