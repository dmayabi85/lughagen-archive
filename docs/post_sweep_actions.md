# Phase 2 End-of-Sweep Action Plan

**File location**: `/workspace/docs/POST_SWEEP_ACTIONS.md`
**Created**: 2026-05-20
**Trigger**: All 32 Phase 2 factorial cells complete (no failures, registry-validated)

This is a checklist of actions to perform AFTER the Phase 2 sweep (32 cells) completes and BEFORE writing the thesis Results chapter. Each action is independently safe to run. Order matters in some places (noted).

---

## Pre-action verification

Before doing anything else, confirm the sweep actually completed cleanly:

```bash
# 1. Did the sweep launcher finish?
tail -30 /workspace/logs/phase2_sweep_status.log

# 2. Are all 32 cells in the registry with required fields?
python3 -c "
import csv
with open('/workspace/registry/runs.csv') as f:
    cells = [r for r in csv.DictReader(f)
             if r.get('factorial_run_id','').startswith('RUN-P2-C')
             and r.get('metrics.validation_perplexity')
             and r.get('wall_clock_seconds')]
print(f'Completed Phase 2 cells: {len(cells)}/32')
if len(cells) < 32:
    completed_ids = {r['factorial_run_id'] for r in cells}
    expected = {f'RUN-P2-C{c:02d}-S{s}' for c in range(1,17) for s in (1,2)}
    print(f'Missing: {sorted(expected - completed_ids)}')
"

# 3. Final GPU-hour total
python3 -c "
import csv
total = 0.0
with open('/workspace/registry/runs.csv') as f:
    for r in csv.DictReader(f):
        if r.get('factorial_run_id','').startswith('RUN-P2-C'):
            try: total += float(r.get('gpu_hours_consumed', 0) or 0)
            except: pass
print(f'Total Phase 2 compute: {total:.2f} GPU-hr (budget headroom: {31200-total:.0f} GPU-hr)')
"
```

**Expected output**: 32/32 cells complete, total ~2,800 GPU-hr, ~28,400 GPU-hr remaining.

If any cells are missing, investigate the corresponding log file in `/workspace/logs/` before proceeding.

---

## ACTION 1 — Run add_log_pph.py to populate log-PPH metric

**Purpose**: Add `metrics.pph_log` and `metrics.pph_log_ratio` columns alongside the existing `metrics.pph_score` for methodological robustness (addresses the reviewer's perplexity-nonlinearity concern).

**Compute cost**: zero (arithmetic only).

**Safe to re-run**: yes — script is idempotent.

```bash
# Dry-run first to preview
python3 /workspace/scripts/add_log_pph.py --dry-run

# If output looks correct (32 cells updated, baseline + val + gh all valid), run for real
python3 /workspace/scripts/add_log_pph.py
```

**Verification after**:
```bash
python3 -c "
import csv
with open('/workspace/registry/runs.csv') as f:
    for r in csv.DictReader(f):
        rid = r.get('factorial_run_id','')
        if rid.startswith('RUN-P2-C'):
            print(f'{rid}: PPH={r.get(\"metrics.pph_score\", \"\")[:8]}, log_PPH={r.get(\"metrics.pph_log\", \"\")[:12]}')" | head -10
```

---

## ACTION 2 — Backfill config_id column

**Purpose**: The orchestrator never populates `config_id` in the registry (cosmetic bug). Recover it from `factorial_run_id` so the registry is self-documenting.

**Compute cost**: zero.

```bash
python3 <<'PYEOF'
import csv, re, shutil
from pathlib import Path
from datetime import datetime, timezone

REG = Path('/workspace/registry/runs.csv')
ts = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
shutil.copy(REG, REG.with_suffix(f'.csv.bak-config-id-{ts}'))

with open(REG) as f:
    reader = csv.DictReader(f)
    fields = list(reader.fieldnames)
    rows = list(reader)

pattern = re.compile(r'RUN-P2-C(\d+)-S\d+')
n_backfilled = 0
for r in rows:
    if not r.get('config_id') or str(r.get('config_id')).strip() == '':
        rid = r.get('factorial_run_id', '')
        m = pattern.match(rid)
        if m:
            r['config_id'] = int(m.group(1))
            n_backfilled += 1

with open(REG, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(rows)

print(f"Backfilled config_id for {n_backfilled} rows")
PYEOF
```

---

## ACTION 3 — Dedup the multi-rank legacy rows

**Purpose**: MICROBENCH and PERGRAD-PREFLIGHT rows (pre-rank-guard era) appear 7-8 times each in the registry. Harmless for factorial analysis (we filter on `RUN-P2-C` prefix) but pollutes summary counts.

**Compute cost**: zero.

**OPTIONAL** — only if you want a clean registry for the thesis. Otherwise skip.

```bash
python3 <<'PYEOF'
import csv, shutil
from pathlib import Path
from datetime import datetime, timezone

REG = Path('/workspace/registry/runs.csv')
ts = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
shutil.copy(REG, REG.with_suffix(f'.csv.bak-dedup-{ts}'))

with open(REG) as f:
    reader = csv.DictReader(f)
    fields = list(reader.fieldnames)
    rows = list(reader)

# Keep first occurrence of each (factorial_run_id, environment.hostname) for legacy rows
# Production RUN-P2-C* rows are single-row already (rank guard active); leave them alone
seen = set()
out = []
duplicates = 0
for r in rows:
    rid = r.get('factorial_run_id', '')
    # Production cells: keep unchanged
    if rid.startswith('RUN-P2-C') or rid.startswith('BASELINE') or rid.startswith('RUN-P2-INTERMEDIATE'):
        out.append(r)
        continue
    # Legacy multi-rank rows: keep only the first
    key = (rid, r.get('environment.hostname', ''), r.get('environment.datetime_utc', ''))
    if key in seen:
        duplicates += 1
        continue
    seen.add(key)
    out.append(r)

with open(REG, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(out)

print(f"Dedup complete: removed {duplicates} duplicate rows. Remaining: {len(out)}")
PYEOF
```

**Sanity check after**:
```bash
python3 -c "
import csv
from collections import Counter
c = Counter()
with open('/workspace/registry/runs.csv') as f:
    for r in csv.DictReader(f):
        rid = r.get('factorial_run_id','')
        c[rid] += 1
# Print any IDs appearing more than once
dups = {k:v for k,v in c.items() if v > 1}
print(f'IDs with duplicates remaining: {len(dups)}')
for k,v in dups.items():
    print(f'  {k}: {v}')"
```

Expected: 0 duplicates after dedup.

---

## ACTION 4 — Compute per-language post-hoc validation perplexity

**Purpose**: The training callback's per-language gradient norms are sampled (known artifact, ~12 events per cell with usually 1-2 languages observed). The rigorous per-language story comes from running validation perplexity on each cell's final checkpoint, language-by-language.

**Compute cost**: ~18 GPU-hr total (eval-only, no training). One eval pass per language per cell. 32 cells × 4 languages × ~5 min each.

**Important consideration**: This is also when to **split Swahili into monolingual vs code-switched (Sheng)**. The corpus has the `register` field; the eval script should report 5 buckets, not 4:
- swahili_monolingual
- swahili_code_switched (Sheng — 12.33% of training)
- kikuyu
- kamba
- luo

**Script outline** (to be written; not currently in the codebase):

```python
# /workspace/scripts/post_hoc_per_language_eval.py
# Pseudocode — needs implementation
from datasets import load_dataset
import torch

for cell_checkpoint in glob('/workspace/checkpoints/RUN-P2-C*/checkpoint-6000'):
    model = load_peft_model(cell_checkpoint)
    val = load_dataset('json', data_files='/workspace/data/corpus/splits/validation.jsonl')
    
    for lang in ['swahili_monolingual', 'swahili_code_switched', 'kikuyu', 'kamba', 'luo']:
        if lang.startswith('swahili'):
            register = 'monolingual' if 'mono' in lang else 'code_switched'
            subset = val.filter(lambda r: r['language']=='swahili' and r['register']==register)
        else:
            subset = val.filter(lambda r: r['language']==lang)
        ppl = compute_perplexity(model, subset)
        # Write back to registry as metrics.val_ppl_{lang}
```

Run order: AFTER actions 1-3 (so registry is clean for receiving new per-language columns).

---

## ACTION 5 — Select confirmation runs

**Purpose**: Per Spec §3.4, after the 32-cell factorial screen we select 4 configs for confirmation runs (3 PPH-predicted optima + 1 paired balanced-recipe).

**Compute cost**: zero (analysis only).

```bash
# Script should exist at /workspace/scripts/select_confirmation_runs.py
# If not, the analysis is:
#   1. Run factorial regression on 32-row data (lm.fit() on factor levels → PPH)
#   2. Predict PPH for all 2^4 = 16 configs from the fitted model
#   3. Identify top-3 predicted PPH configs
#   4. Identify the "balanced-recipe" config defined by spec (4th confirmation slot)
#   5. Output 4 cell IDs for the confirmation runs
python3 /workspace/scripts/select_confirmation_runs.py 2>/dev/null || echo "Script not found — needs to be written from spec §3.4"
```

If the script doesn't exist, write it referencing Spec §3.4 before proceeding.

---

## ACTION 6 — Launch 4 confirmation runs

**Purpose**: Re-run the 4 selected configurations to confirm the PPH-predicted optima are stable, per Spec §3.4. These give the final 36-row dataset.

**Compute cost**: ~360 GPU-hr (4 cells × ~90 GPU-hr each, same as Phase 2 cells).

**Wall clock**: ~2 days sequential.

```bash
# Launcher should select cell configs from action 5, then run each via:
torchrun --nproc_per_node=8 /workspace/scripts/train_orchestrator.py \
    --phase Phase2_Confirmation \
    --factorial-run-id RUN-P2-CONF-01 \
    --seed 42 \
    --lr-schedule <from-config> \
    --batch-size <from-config> \
    --optimiser <from-config> \
    --sequence-length <from-config> \
    --data-ordering mixed \
    --output-dir /workspace/checkpoints/RUN-P2-CONF-01 \
    2>&1 | tee /workspace/logs/conf-01_$(date -u +%Y%m%dT%H%M%SZ).log
```

After confirmation runs complete, re-run Action 1 to populate log-PPH for those rows too.

---

## ACTION 7 — Filter analysis dataset and run factorial regression

**Purpose**: Aggregate the 36 production rows (32 factorial + 4 confirmation) into the dataset for factorial regression analysis. This becomes the empirical core of the thesis Results chapter.

**Compute cost**: zero.

```python
# /workspace/scripts/factorial_analysis.py
import pandas as pd
import statsmodels.formula.api as smf

df = pd.read_csv('/workspace/registry/runs.csv')
# Filter to production rows only
phase2 = df[df['factorial_run_id'].str.startswith('RUN-P2-C')].copy()
# Convert factor strings to coded levels (-1/+1)
phase2['A'] = (phase2['factors.lr_schedule'] == 'linear_warmup_constant').astype(int) * 2 - 1
phase2['B'] = (phase2['factors.batch_size'] == 256).astype(int) * 2 - 1
phase2['C'] = (phase2['factors.optimiser'] == 'Lion').astype(int) * 2 - 1
phase2['D'] = (phase2['factors.sequence_length'] == 2048).astype(int) * 2 - 1

# Main effects + all 2-way interactions
formula = 'pph_score ~ A * B * C * D'  # full factorial including all interactions
model = smf.ols(formula, data=phase2).fit()
print(model.summary())

# Repeat with log-PPH and compare
formula_log = 'pph_log ~ A * B * C * D'
model_log = smf.ols(formula_log, data=phase2).fit()
print(model_log.summary())

# Spearman rank concordance between PPH and log-PPH
from scipy.stats import spearmanr
rho, p = spearmanr(phase2['pph_score'], phase2['pph_log'])
print(f'\nSpearman ρ (PPH vs log-PPH): {rho:.4f} (p={p:.2e})')
```

The thesis Results chapter draws directly from this output.

---

## ACTION 8 — File Spec §13 budget review note

**Purpose**: The original Spec projected Phase 2 at ~18,000 GPU-hr. Actual will be ~2,800 GPU-hr (15× under-utilisation). Spec §13 requires a budget review note documenting how the surplus is reallocated.

**Compute cost**: zero (writing only).

Suggested allocation of the ~12,000 GPU-hr surplus from Phase 2:
- **+2,000 GPU-hr to Phase 3** — increase from 5 seeds to 7-10 seeds per method (tighter PEFT comparison)
- **+1,000 GPU-hr to per-language post-hoc evaluation** — finer-grained Sheng analysis
- **+500 GPU-hr reserve** — for any failed/re-run cells
- **+8,500 GPU-hr returned to the grant** — surplus capacity for future LughaGen work

Write the note to `/workspace/docs/budget_review_2026-XX.md` and reference it in the thesis methodology chapter.

---

## ACTION 9 — Generate thesis figures

**Purpose**: Build the publication-quality plots for the thesis Results chapter.

**Compute cost**: zero (plotting only).

Plots to generate:
1. **Pareto frontier** — PPH vs final validation perplexity for all 36 cells
2. **Factor effect plot** — bar chart of estimated main effects (A, B, C, D) with 95% CIs
3. **Interaction plot** — 6 panels showing all 2-way interactions
4. **Convergence curves** — training loss over steps for the 4 confirmation runs
5. **Per-language PPH** — 5 bars per cell (sw_mono, sw_cs, ki, ka, lu) for the 4 confirmation runs
6. **Metric concordance** — scatter plot of PPH vs log-PPH with Spearman ρ in caption
7. **Budget utilization** — cumulative GPU-hours over project timeline

---

## ACTION 10 — Phase 3 preparation

**Purpose**: After Phase 2 + confirmation is fully complete, prepare for the Phase 3 PEFT comparison.

Phase 3 needs (deferred from Week 6 of the original plan):
- DeepSpeed ZeRO-3 config for the full fine-tuning method
- Equal-budget allocation script (5 seeds × 4 methods × 4 languages = 80 runs)
- Total Phase 3 budget: ~600-2,500 GPU-hr depending on per-run wall-clock

Action items:
1. Write `/workspace/scripts/zero3_config.json` per HuggingFace DeepSpeed docs
2. Write `/workspace/scripts/phase3_sweep.sh` mirroring the Phase 2 launcher pattern
3. Smoke-test each of 4 methods on a single language for 500 steps before launching the full sweep

---

## Summary timeline (estimated)

| Stage | Wall clock | GPU-hours | When |
|---|---|---|---|
| Sweep completion | — | — | T = day 16 (2026-06-05 approx) |
| Actions 1-3 (registry cleanup) | 30 min | 0 | T |
| Action 4 (per-lang eval) | 1 day | ~18 | T+1 |
| Action 5-6 (confirmation runs) | 2-3 days | ~360 | T+1 to T+4 |
| Actions 7-8 (analysis + budget) | 2-3 days | 0 | T+4 to T+7 |
| Action 9 (figures) | 2 days | 0 | T+7 to T+9 |
| Action 10 (Phase 3 setup) | 1 day | 0 | T+9 |
| **Phase 3 sweep** | **5-10 days** | **~600** | **T+10 to T+20** |

Thesis Results chapter writing can begin during action 4 in parallel.

---

## Open issues to track during sweep

These don't block sweep completion but worth tracking:

- [ ] `spec_hash` field is empty in registry rows. Patch reads `spec_versions.yaml::current` but actual file uses different key. Fix during a between-cells window or post-sweep.
- [ ] `config_id` empty (covered by Action 2).
- [ ] Per-language callback sampling artifact — known, documented in thesis methodology, addressed by Action 4 post-hoc eval.
- [ ] Multi-rank legacy duplicate rows in registry — covered by Action 3 (optional).
- [ ] `metrics.tokens_per_sec` shows the same value across cells (19,607). This is suspicious — looks like a static value from calibration rather than a per-cell measurement. Verify after sweep: if all cells show identical tps, the metric is not actually being computed per-cell. Could explain by either checking source code or accepting it as cosmetic.

---

## Risk register

| Risk | Mitigation |
|---|---|
| A cell crashes mid-sweep | Launcher continues to next cell. Re-run failed cell standalone after sweep. |
| Instance restarts | tmux survives; sweep continues. If instance is rebuilt, restart launcher (it skips completed cells). |
| Registry corruption | Daily backups via cron, plus all action scripts back up before modifying. |
| Spec compliance challenge from supervisor | Action 8 budget review note + thesis limitations §9 transparency cover this. |
| Per-cell results show high seed variance | Cell 1 vs Cell 2 reproducibility (0.01% delta) suggests this won't happen, but if it does: increase seeds during Phase 3 (Action 8 surplus covers this). |

---

*End of post-sweep action plan. Save this file to /workspace/docs/POST_SWEEP_ACTIONS.md on the instance.*
