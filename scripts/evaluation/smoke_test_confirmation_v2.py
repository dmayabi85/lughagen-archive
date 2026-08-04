#!/usr/bin/env python3
"""
Smoke test for select_confirmation_runs_v2.py.

Runs the script with --min-cells=2 against the current partial Phase 2 data.
With only 2 cells (1 config), we can't actually do ranking, but we CAN:
  - confirm imports work
  - confirm CSV reading works
  - confirm encode_factors() handles real registry rows
  - confirm config_id extraction works (including the fallback path)
  - print what the script SEES so we can spot data issues

This is purely a sanity check; the real validation will happen at sweep end.

Usage:
  python3 /workspace/scripts/smoke_test_confirmation_v2.py
"""

import subprocess
import sys

SCRIPT = "/workspace/scripts/select_confirmation_runs_v2.py"

print("=" * 76)
print(f"Smoke testing {SCRIPT}")
print("=" * 76)

# Test 1: --help works (no actual data needed)
print("\n[1/3] Testing --help ...")
r = subprocess.run(
    ["python3", SCRIPT, "--help"],
    capture_output=True, text=True,
)
if r.returncode == 0:
    print("  ✓ --help succeeded")
else:
    print(f"  ✗ --help FAILED: {r.stderr}")
    sys.exit(1)

# Test 2: --dry-run with low --min-cells, will fail at "need 4 configs" but should
#         get far enough to exercise CSV reading + encoding
print("\n[2/3] Testing data loading with --min-cells=1 (will likely fail at ranking step)...")
r = subprocess.run(
    ["python3", SCRIPT, "--dry-run", "--min-cells", "1", "--n-bootstrap", "100"],
    capture_output=True, text=True,
)
print("  stdout:")
for line in r.stdout.splitlines()[:30]:
    print(f"    {line}")
print(f"  Return code: {r.returncode}")
if r.stderr:
    print("  stderr:")
    for line in r.stderr.splitlines():
        print(f"    {line}")

# Test 3: Verify the load + aggregate path produces sensible output
print("\n[3/3] Direct function test — does aggregation work against real data?")
sys.path.insert(0, "/workspace/scripts")
try:
    import importlib.util
    spec = importlib.util.spec_from_file_location("scrv2", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["scrv2"] = mod  # required for @dataclass to find the module
    spec.loader.exec_module(mod)

    rows = mod.load_phase2_rows()
    print(f"  Loaded {len(rows)} Phase 2 rows with valid PPH")
    for r in rows[:5]:
        print(f"    {r['factorial_run_id']}: "
              f"PPH={r['metrics.pph_score']}, "
              f"baseline={r['metrics.baseline_perplexity']}")

    summaries = mod.aggregate_by_config(rows)
    print(f"\n  Aggregated to {len(summaries)} unique configs")
    for s in summaries:
        print(f"    config {s.config_id}: levels={s.levels}, "
              f"n_seeds={s.n_seeds}, mean_PPH={s.pph_mean:.4f}, std={s.pph_std:.4f}")
        for cid, pph in zip(s.cell_ids, s.pph_values):
            print(f"      {cid}: PPH={pph:.4f}")

    if len(summaries) >= 4:
        print(f"\n  ✓ Have {len(summaries)} configs; ranking step would succeed")
    else:
        print(f"\n  ✓ Aggregation works (only {len(summaries)} configs — ranking deferred)")

except Exception as e:
    print(f"  ✗ Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 76)
print("Smoke test complete — the v2 script can read your registry without errors.")
print("=" * 76)
