# LughaGen Training Framework — Implementation Specification v1.5.1

**Author:** Derek Mayabi (Model Lead, LughaGen Research Programme)
**Institution:** Jomo Kenyatta University of Agriculture and Technology
**Programme:** MSc in Artificial Intelligence
**Date:** May 2026
**Status:** Implementation-ready
**Supersedes:** v1.5 (2026-05-18)

---

## 0. Document Purpose

This document is the implementation contract for the LughaGen training
framework. It supersedes v1.0 through v1.5 and is a point release of v1.5
incorporating two corpus-level decisions that emerged from the corpus
inventory review on 2026-05-17:

  1. Exclusion of LughaGen-pipeline-generated code-switched data from
     active training across all four target languages.
  2. Reclassification of the Sheng corpus from `authentic` to `synthetic`
     based on its actual provenance (third-party generator output).

These changes alter the corpus composition reported in §2.3 and §5.3 and
introduce additional limitations documentation in §10.3.3. The factorial
design, PPH metric, registry schema, training pipeline, and timeline are
unchanged from v1.5.

Any deviation during implementation requires updating this document first
and writing a new spec version to `/workspace/registry/spec_versions.yaml`
(procedure in §18).

The framework operationalises the methodology approved in the LughaGen MSc
research proposal (May 2026, Compute-Efficient Training and Adaptation of
Low-Resource African Language Models under Fixed GPU Budgets).

---

## 1. Primary Research Objective

Investigate how training-engineering decisions affect model quality and
compute efficiency during continued pre-training of LLaMA-3.1-8B for:

- Swahili
- Kikuyu
- Luo
- Kamba

The framework must generate empirical evidence identifying which training
factors contribute most to performance improvements under constrained
compute budgets.

---

## 2. Strict Scope Constraints

### 2.1 In scope

- Model architecture: LLaMA-3.1-8B only
- Primary task: Continued pre-training
- Phase 3 PEFT comparison: LoRA, QLoRA, adapter tuning, full fine-tuning
  (ZeRO-3 sharded, capacity upper-bound reference only)

### 2.2 Explicitly out of scope (must not consume the 32,000 GPU-hour budget)

- Models larger than 8B parameters
- Instruction tuning
- RLHF / RLAIF / DPO / preference tuning
- Deployment stack implementation
- Inference optimization
- Architectures unrelated to LLaMA-3.1-8B
- Languages beyond Swahili, Kikuyu, Luo, Kamba

### 2.3 Corpus Composition — Two Methodological Decisions

The training corpus reflects two corpus-level decisions taken during the
inventory review on 2026-05-17, both documented for the thesis Limitations
chapter (§10.3.3).

**Decision 1 — LughaGen-pipeline code-switched data is excluded from
training.** The tier-2 LughaGen generation pipeline produced 45,883
code-switched documents (708K words) across the four target languages.
QA inspection during corpus integration found ~95% disagreement between
the pipeline's CS labels and the actual code-switching content of the
generated text (the generator frequently produced monolingual content
when prompted to produce code-switched output). These four manifests
are therefore archived outside the active manifests directory at
`/workspace/data/corpus/_excluded/synthetic_cs_2026-05-17/` and are not
ingested into the unified corpus.

**Decision 2 — The Sheng corpus is classified as synthetic, not
authentic.** Sheng (Swahili-English code-switching) was originally
ingested under `manifests/authentic_code_switched/` based on assumption.
Subsequent verification established that the Sheng dataset
(`/workspace/data/secondary/code_switched/syntheticsentences.jsonl`,
183,049 records) is third-party generator output, not human-authored
text. The manifest was relocated to
`manifests/synthetic_code_switched/swahili_sheng_v1.yaml` and its
`origin` field changed to `synthetic`. The corpus contents are unchanged
by this relabel — only the classification metadata changes. Sheng is
retained in training because its CS content is assessed as
higher-quality than the LughaGen-pipeline output (the basis of
Decision 1).

**Result: all code-switched training data is synthetic; no authentic
code-switched data exists in the corpus.** The corpus is 47.4% authentic
monolingual residue and 52.6% synthetic (LughaGen monolingual + Sheng
code-switched).

Measured corpus composition (by words, from `corpus_stats.json` after
both decisions are applied):

**By origin:**

| Origin | Words | Share |
|---|---|---|
| Synthetic | 12,367,076 | 52.58% |
| Authentic | 11,152,689 | 47.42% |

**By register:**

| Register | Words | Share |
|---|---|---|
| Monolingual | 17,383,419 | 73.91% |
| Code-switched (Sheng only) | 6,136,346 | 26.09% |

**By language:**

| Language | Words | Share | Origin breakdown |
|---|---|---|---|
| Swahili | 17,903,558 | 76.12% | 9.21M authentic mono + 2.56M synth mono + 6.14M synth CS |
| Luo | 2,391,513 | 10.17% | 1.64M authentic mono + 0.76M synth mono |
| Kikuyu | 1,667,424 | 7.09% | 0.21M authentic mono + 1.46M synth mono |
| Kamba | 1,557,270 | 6.62% | 0.10M authentic mono + 1.45M synth mono |
| **Total** | **23,519,765** | **100%** | |

**Per-language × register matrix (every cell is monolingual except
Swahili × code_switched):**

| Cell | Words | Share |
|---|---|---|
| Swahili × monolingual | 11,767,212 | 50.03% |
| Swahili × code_switched (Sheng) | 6,136,346 | 26.09% |
| Luo × monolingual | 2,391,513 | 10.17% |
| Kikuyu × monolingual | 1,667,424 | 7.09% |
| Kamba × monolingual | 1,557,270 | 6.62% |
| Kikuyu/Kamba/Luo × code_switched | **0** | **0%** |

The asymmetry across languages remains a deliberate methodological choice
(see §10.3.3 limitations). Three justifications:

1. Forcing 25/25/25/25 balance would require either subsampling Swahili
   to Kamba's ~1.6M words (losing 73% of available data including the
   entire Sheng corpus advantage) or upsampling the smaller three via
   3-7× duplication (introducing overfit on duplicated content).
2. The natural distribution reflects real-world data availability that
   any African-language model would face in deployment.
3. The framework supports a `balanced_25_25_25_25` recipe (§5.4) for
   sensitivity analysis, so the balance effect can be measured rather
   than assumed away.

Phase 2 factorial runs use the natural-distribution `full` recipe as the
primary corpus. The balanced recipe is reserved for confirmation-run
sensitivity analysis paired with the predicted-optimum natural-corpus run
(see §3.4, §17 Week 5).

Per-language perplexity is tracked separately in all reporting so the
asymmetry's effect on each language is measurable, not hidden.

**Per-language gradient monitoring.** Given the 76/10/7/7 language split,
smaller languages may receive disproportionately small gradient updates
per training step. The framework therefore tracks
`per_language_grad_norm_ratio` during training (§9.4) as a diagnostic
signal for whether the smaller languages are actually being learned vs
under-represented in the gradient signal.

**Code-switching coverage limitation.** Kikuyu, Kamba, and Luo training
data contains zero code-switched content. The model cannot learn CS
phenomena for these three languages from this corpus. Any CS evaluation
on these three languages will be measuring zero-shot CS generalisation
from monolingual training, not learned CS behaviour. The thesis
Limitations chapter (§10.3.3) documents this explicitly.

Any experimentation outside §2.1 must be marked as future work and must
not consume the approved GPU budget allocation.

---

## 3. Core Scientific Requirement — Fractional Factorial Design

### 3.1 Design Specification

The framework implements a **2^(5-1) Resolution V fractional factorial
design** (Box, Hunter & Hunter, 2005).

**Defining relation:** `E = ABCD`

Where factors A-E correspond, in order, to:

| Symbol | Factor | Level A (-1) | Level B (+1) |
|---|---|---|---|
| **A** | Learning-Rate Schedule | Cosine decay | Linear warmup + constant |
| **B** | Effective Batch Size | 128 | 256 |
| **C** | Optimiser | AdamW | Lion |
| **D** | Sequence Length | 1,024 tokens | 2,048 tokens |
| **E** | Data Ordering | Curriculum (real→synthetic) | Mixed (interleaved) |

**Note on Lion learning rate (NEW in v1.5).** Lion's optimal learning
rate is typically 3-10× smaller than AdamW's at the same batch size, due
to differences in update magnitude. To preserve the factorial design's
orthogonality, **peak LR is held constant at 2 × 10⁻⁴ across all 32 base
factorial runs regardless of optimiser choice**. This means the "Lion
vs AdamW" comparison is intentionally conservative for Lion: Lion is
evaluated at AdamW's preferred LR rather than at a Lion-tuned LR. This
preserves the factorial's interpretability (the optimiser main effect
measures *the effect of swapping AdamW for Lion at otherwise-identical
hyperparameters*) at the cost of potentially underestimating Lion's
peak performance. The thesis's Limitations chapter (§10.3) documents
this conservatism explicitly.

### 3.2 Alias Structure (logged to registry at matrix-creation time)

Under defining relation `E = ABCD` (identity `I = ABCDE`):

- Main effects aliased with 4-factor interactions:
  - `A = BCDE`, `B = ACDE`, `C = ABDE`, `D = ABCE`, `E = ABCD`
- Two-factor interactions aliased with three-factor interactions:
  - `AB = CDE`, `AC = BDE`, `AD = BCE`, `AE = BCD`
  - `BC = ADE`, `BD = ACE`, `BE = ACD`
  - `CD = ABE`, `CE = ABD`, `DE = ABC`

At Resolution V, main effects and two-factor interactions are not aliased
with each other or with other main effects — they are clean.

### 3.3 Run Schedule

- 16 base factorial configurations (one per row of the 2^(5-1) matrix)
- 2 random seeds per configuration → 32 factorial runs
- 4 confirmation runs (3 PPH-predicted + 1 paired balanced-recipe
  sensitivity — see §3.4)
- **Total: 36 runs**

### 3.4 Confirmation Run Selection (REVISED in v1.5)

After 32 factorial runs complete:

1. Fit a linear regression model to PPH as a function of the five factor
   main effects and all ten clean two-factor interactions.
2. Use the fitted model to predict PPH for all 32 full-factorial
   configurations (including the 16 not represented in the
   half-fraction).
3. **Compute bootstrap uncertainty bands on each prediction (NEW in
   v1.5).** Use 1,000 bootstrap resamples of the 32 factorial runs;
   refit the linear model on each resample; record the 95% percentile
   range of predicted PPH for each untested configuration. Store as
   `predicted_pph_bootstrap_low`, `predicted_pph_bootstrap_high` per
   candidate configuration in the registry.
4. **Slots 1-3 (PPH-predicted confirmation):** Select the 3
   configurations with the highest mean predicted PPH that were *not*
   present in the 32 factorial runs. Each uses a fresh random seed not
   used in the factorial; recipe = `full`.
5. **Robustness check on the top-3 selection.** Compare each top-3
   prediction's bootstrap CI against the rank-4 configuration's
   bootstrap CI. If any top-3 CI overlaps with rank-4's by more than 50%
   of the band width, flag the selection for manual supervisor review
   before launch. This catches the case where Resolution V's
   assumption of negligible 3-way interactions is violated and the
   linear model's "best 3" predictions are statistically
   indistinguishable from a wider candidate pool.
6. **Slot 4 (paired balanced-recipe sensitivity):** Use the same factor
   settings as the **highest-predicted untested configuration from
   step 4** (i.e., the same configuration as slot 1) but trained on
   the `balanced_25_25_25_25` recipe instead of `full`. Uses a fresh
   random seed not used in the factorial.

**Rationale for bootstrap-CI addition.** Resolution V cleanly estimates
main effects and two-factor interactions but assumes higher-order
interactions are negligible. If 3-way interactions are strong, the
linear-model predictions for untested configurations will mislead.
Bootstrap CIs surface this uncertainty quantitatively rather than
hiding it behind point estimates. The 50%-overlap threshold in step 5 is
a heuristic safeguard against over-confident selection.

**Rationale for the slot-4 pairing.** Slot 4 is methodologically
stronger than a standalone sensitivity run: pairing it with slot 1 (the
predicted-optimum natural-corpus configuration) means the balance
effect can be tested directly via a single paired comparison rather
than against a distribution of natural-corpus runs. The two
slot-1/slot-4 runs form the dataset for the balance-effect TOST
equivalence test in Phase 2's sensitivity analysis.

### 3.5 Framework Requirements

The factorial engine must automatically:

- Generate the full experiment matrix from the defining relation
- Assign unique run identifiers: `RUN-{phase}-{config_id}-{seed}`
- Preserve factor-level metadata per run
- Prevent duplicate configurations
- Track seed assignments explicitly
- Track confirmation runs separately (`is_confirmation_run: bool`) and
  distinguish PPH-predicted vs paired-sensitivity slots
  (`confirmation_slot_type: pph_predicted | paired_sensitivity`)
- Log the defining relation and full alias structure to the registry on
  matrix creation
- Log bootstrap CIs for confirmation-run predictions
- Support downstream analysis: main effects, two-factor interactions,
  factor importance ranking, compute-efficiency trade-offs, seed
  variance

---

## 4. Performance-Per-GPU-Hour (PPH) — Primary Optimization Metric

### 4.1 Definition

```
PPH = (P_baseline - P_config) / GPU-hours_consumed
```

Where:
- **P_baseline**: validation perplexity of zero-shot LLaMA-3.1-8B on the
  held-out LughaGen corpus (see §4.2 for exact procedure)
- **P_config**: validation perplexity of the trained configuration on
  the same held-out corpus, under the same tokenizer
- **GPU-hours_consumed**: wall-clock time × number of GPUs utilised

Lower perplexity is better; positive PPH indicates improvement over
baseline; higher PPH indicates greater improvement per GPU-hour
consumed.

### 4.2 Baseline Perplexity Computation (one-time, fixed for the study)

P_baseline is computed once and stored as a global constant in the
registry. The procedure:

1. Load LLaMA-3.1-8B base weights with no adaptation (`from_pretrained`,
   no LoRA, no QLoRA).
2. Tokenize `/workspace/data/corpus/splits/validation.jsonl` using the
   Phase-0-winning tokenizer.
3. Compute perplexity as `exp(mean(cross_entropy_per_token))` over the
   full validation split with `max_seq_length=2048`, no padding loss
   contribution, BF16 precision.
4. Report a single pooled scalar (P_baseline) plus four per-language
   scalars (P_baseline_swahili, P_baseline_kikuyu, P_baseline_kamba,
   P_baseline_luo).
5. If the Phase-0 tokenizer choice changes later, P_baseline must be
   recomputed and all downstream PPH values invalidated.

### 4.3 Negative-PPH Handling

**For Phase 2 (factorial) runs:**

- Negative PPH does NOT abort training. Run completes its full
  schedule.
- `pph_flag_negative: true` is written to the registry.
- A warning is emitted to the monitoring dashboard.
- The run is excluded from confirmation-run candidate selection (§3.4).

**For Phase 3 (PEFT comparison) runs:**

- Same behaviour as Phase 2: complete training, flag, exclude.

**PPH-based early stopping is DISABLED in both phases** to preserve
experimental integrity. The kill criteria in §13 are independent of
PPH.

### 4.4 Reporting Requirements

- PPH leaderboards auto-update after every run completes
- All statistical comparisons use PPH as the principal dependent
  variable
- Raw perplexity alone must not be used as the primary comparison
  metric
- PPH is computed automatically post-run and stored in the registry

### 4.5 Secondary Metrics (REVISED in v1.5)

Logged in every registry record alongside PPH:

- **PPH log-ratio (NEW in v1.5):** `log(P_baseline / P_config) /
  GPU-hours_consumed`. Captures multiplicative perplexity improvement
  rather than additive; more robust when perplexity deltas are small
  (<0.1) where seed noise can dominate the raw PPH metric. Reported
  alongside PPH in all Phase 2 and Phase 3 leaderboards as a
  robustness check; if PPH and PPH log-ratio rankings disagree, the
  disagreement is flagged in the report for manual review.
- Validation perplexity (pooled + per-language)
- Tokens/sec throughput
- GPU utilisation (%)
- VRAM peak usage (GB)
- Training stability (gradient norm trajectory)
- Token fertility ratio
- Perplexity convergence rate (steps to within 5% of minimum)
- **Per-language gradient norm ratio (NEW in v1.5):** Mean ratio of
  Swahili gradient norm to non-Swahili gradient norm across the training
  run, computed by partitioning each batch's loss by source language
  before backward pass and computing per-partition gradient norms. See
  §9.4 for the per-step logging procedure.

---

## 5. Dataset Integration

### 5.1 Source of Truth

The framework consumes the unified corpus produced by the LughaGen
Corpus Framework (`/workspace/scripts/build_unified_corpus.py`):

| Artefact | Path |
|---|---|
| Training data | `/workspace/data/corpus/splits/train.jsonl` |
| Validation data | `/workspace/data/corpus/splits/validation.jsonl` |
| Test data | `/workspace/data/corpus/splits/test.jsonl` |
| Composition stats | `/workspace/data/corpus/stats/corpus_stats.json` |
| Manifests | `/workspace/data/corpus/manifests/*.yaml` |
| Audit log | `/workspace/data/corpus/audit/ingest_log.jsonl` |
| Recipes | `/workspace/data/corpus/splits/recipes.json` |

### 5.2 Dataset Version Identifier

```python
dataset_version = sha256(
    sorted(manifest_source_ids)
    + corpus_stats_json_canonical_form
).hexdigest()[:12]
```

Computed at registry-write time, stored alongside the human-readable
label (e.g., `"lughaGen-v1.2-{hash}"`).

### 5.3 Tracked Metadata per Run

```yaml
dataset:
  version: "lughaGen-v1.3"               # post-relabel + exclusion
  checksum_sha256: "<full hash>"
  manifests_used: [<list of 9 source_ids>]   # 4 auth_mono + 4 synth_mono + 1 synth_cs (Sheng)
  language_split:
    swahili: 0.7612
    luo:     0.1017
    kikuyu:  0.0709
    kamba:   0.0662
  origin_split:
    synthetic: 0.5258                    # Sheng (relabeled) + 4 synth_mono
    authentic: 0.4742                    # 4 raw residue only
  register_split:
    monolingual:   0.7391
    code_switched: 0.2609                # Sheng only; no Kikuyu/Kamba/Luo CS
  total_words: 23519765
  total_tokens_llama: <filled after tokenization>
  total_documents: 1434117
  recipe: "full"
```

### 5.4 Recipes

| Recipe | Effect | Primary use |
|---|---|---|
| `full` | All records (post-exclusion, post-relabel) | Phase 2 main runs (32 factorial + 3 confirmation) |
| `balanced_25_25_25_25` | Subsample Swahili to match Kamba volume | Phase 2 slot-4 paired sensitivity (§3.4) |
| `synth_only` | Origin = synthetic (includes Sheng since relabel) | Phase 2 ablation (informational) |
| `auth_only` | Origin = authentic (monolingual residue only) | Phase 2 ablation (informational) |
| `mono_only` | Register = monolingual | Phase 2 ablation (informational) |
| `cs_only` | Register = code-switched (Sheng only — Swahili) | Phase 2 ablation (informational) |
| `no_synth_cs` | Equivalent to `full` after 2026-05-17 exclusion | Phase 2 ablation; retained for backward-compatibility audit |

Each Phase 2 run's `dataset.recipe` field records which recipe was
used.

**Recipe semantics after the 2026-05-17 corpus changes:**

- `auth_cs_only` (if invoked) returns **zero records** — no authentic CS data
  exists in the corpus. The recipe is retained in `recipes.json` for audit
  transparency but produces an empty set.
- `synth_cs_only` (if invoked) returns the Sheng corpus (Swahili only).
- `cs_only` and `synth_cs_only` are functionally equivalent after the
  Sheng relabel.

The four LughaGen-pipeline-generated synth-CS manifests live at
`/workspace/data/corpus/_excluded/synthetic_cs_2026-05-17/` and are NOT
loaded by `recipes.json`. They are preserved for potential future studies
on synthetic-CS quality (out of scope for the current 32,000-GPU-hour
budget).

---

## 6. Tokenizer Evaluation (Phase 0)

Phase 0 must complete and write a winning-tokenizer record to the
registry before any Phase 2 or Phase 3 runs begin.

### 6.1 Strategies Compared

- **Strategy A**: Default LLaMA-3.1 tokenizer (128,256-token BPE
  vocabulary)
- **Strategy B**: LLaMA-3.1 vocabulary extended by 10,000–20,000 units
  learned jointly across the four African corpora; preserves existing
  English/multilingual capability
- **Strategy C**: Hybrid joint tokenizer combining a frozen English
  vocabulary with a learned African vocabulary

### 6.2 Per-Tokenizer Tracking

- Token fertility ratio per language (target: lower than baseline)
- Unknown token rate
- Compression efficiency vs LLaMA-3.1 default (token count ratio)
- Morphological segmentation quality (noun-class markers, verb
  extensions)
- Token distribution across languages
- Vocabulary coverage on held-out authentic text

### 6.3 Reference Fertility Baselines (from pilot Appendix A.2)

| Language | Mean | Q3 | vs. English |
|---|---|---|---|
| Swahili | 2.322 | 2.476 | 1.76× |
| Kikuyu | 3.026 | 3.444 | 2.29× |
| Kamba | 2.845 | 3.308 | 2.16× |
| Luo | 1.817 | 2.000 | 1.38× |
| English (Wikipedia) | 1.319 | 1.407 | baseline |

### 6.4 Selection Criterion + Convergence Safeguard (REVISED in v1.5)

The winning Strategy minimises:

```
selection_score = sum_over_languages(fertility_relative_to_baseline)
                  + lambda * avg_validation_ppl_after_short_run
```

Both terms are added because both should be minimised. `lambda = 0.5`
(tunable), and "short run" defaults to 1,000 training steps at default
Phase 2 hyperparameters.

**Convergence safeguard (NEW in v1.5):** A short run's validation PPL
may not have plateaued by step 1,000, which would bias the selection
score toward the strategy that happens to be converging fastest at the
arbitrary cutoff. The framework therefore:

1. Logs validation PPL every 50 steps during each strategy's short run.
2. After step 1,000, computes the PPL slope over the last 300 steps
   (steps 700-1000) as `(PPL_700 - PPL_1000) / PPL_1000`.
3. **If any strategy's slope exceeds 0.5% per 100 steps** (i.e., still
   measurably descending), all three strategies extend uniformly to
   step 1,500. The extension is uniform to preserve cross-strategy
   comparability — extending only the non-plateaued strategy would bias
   in the opposite direction.
4. If after step 1,500 any strategy still has slope > 0.5%/100 steps,
   log a warning, proceed with selection, and document the
   non-convergence in the Phase 0 report.

**Identical-conditions requirement.** All three strategies must use
identical initialization seed, learning rate, batch size, sequence
length, and data sampling order during the short run. This is enforced
by `phase0_runner.py` reading a single configuration block and varying
only the tokenizer.

The winning tokenizer is frozen and logged to the registry as
`registry.tokenizer_version`. All subsequent phases use this tokenizer;
no in-phase tokenizer changes are permitted.

**Budget implication.** Phase 0 may consume up to 1,500 GPU-hours
(1,000 nominal + 500 extension contingency). The §8.2 budget reflects
the maximum; the contingency reverts to reserve if not used.

---

## 7. Automated Experiment Registry (Mandatory)

No experiment may execute without automatic registry logging.

### 7.1 Per-Run Schema

```yaml
experiment_id: "EXP-20260517-001"
factorial_run_id: "RUN-P2-C04-S1"
phase: "Phase2_Factorial"
is_confirmation_run: false
confirmation_slot_type: null
spec_version: "1.5"
spec_hash: "<sha256 of this document>"

factors:
  lr_schedule: "cosine"
  batch_size: 128
  optimiser: "AdamW"
  sequence_length: 1024
  data_ordering: "curriculum"

seed: 42

dataset:
  version: "lughaGen-v1.2-a3f8b2c"
  checksum_sha256: "<hash>"
  manifests_used: [<source_ids>]
  language_split: {swahili: 0.7521, luo: 0.0997, kikuyu: 0.0776, kamba: 0.0706}
  origin_split:   {synthetic: 0.286, authentic: 0.714}
  register_split: {monolingual: 0.717, code_switched: 0.283}
  total_words: 24985667
  total_tokens_llama: <int>
  recipe: "full"

tokenizer_version: "strategy_B_extended_v1"

hyperparameters:
  peak_lr: 2e-4                       # held constant across all factorial runs (§3.1)
  warmup_steps: 0
  weight_decay: 0.0
  max_grad_norm: 1.0
  lora_r: 16
  lora_alpha: 32
  lora_dropout: 0.05
  num_train_epochs: 3
  lion_implementation: "lion-pytorch==<verified version>"

gpu_config:
  gpu_count: 8
  gpu_type: "A100_80GB"
  precision: "BF16+TF32"
  gradient_checkpointing: true
  flash_attention_version: "<verified version>"
  flash_attention_engaged: <bool>     # NEW in v1.5; verified at run start (§14)

gpu_hours_consumed: 0.0
wall_clock_seconds: 0
checkpoint_paths:
  intermediate: []
  final: ""

confirmation_run_metadata:            # null unless is_confirmation_run=true
  predicted_pph_mean: null
  predicted_pph_bootstrap_low: null   # NEW in v1.5
  predicted_pph_bootstrap_high: null  # NEW in v1.5
  selection_rank: null
  flagged_for_manual_review: false    # NEW in v1.5

metrics:
  baseline_perplexity: <P_baseline scalar from §4.2>
  baseline_perplexity_per_lang:
    swahili: null
    luo:     null
    kikuyu:  null
    kamba:   null
  validation_perplexity: null
  validation_perplexity_per_lang:
    swahili: null
    luo:     null
    kikuyu:  null
    kamba:   null
  pph_score: null
  pph_log_ratio: null                 # NEW in v1.5
  pph_flag_negative: false
  tokens_per_sec: null
  peak_vram_gb: null
  gpu_utilization_pct: null
  convergence_steps_to_5pct: null
  token_fertility_ratio: null
  per_language_grad_norm_ratio_mean: null   # NEW in v1.5 — mean across training
  per_language_grad_norm_ratio_per_lang:    # NEW in v1.5 — per-language means
    swahili: null
    kikuyu: null
    kamba:  null
    luo:    null

training_events:
  gradient_anomalies: []
  instability_events: []
  runtime_failures: []
  early_stopped: false
  early_stop_reason: ""

environment:
  torch_version: ""
  transformers_version: ""
  peft_version: ""
  deepspeed_version: ""
  cuda_version: ""
  flash_attn_version: ""
  lion_pytorch_version: ""
  hostname: ""
  datetime_utc: ""
```

### 7.2 Backend Support

1. **CSV (mandatory, Week 1)** — `/workspace/registry/runs.csv`
2. **SQLite (mandatory, Week 2)** — `/workspace/registry/runs.db`
3. **MLflow (recommended, Week 3)** — primary experiment tracking
4. **Weights & Biases (optional, Weeks 3-4)** — real-time dashboards
5. **PostgreSQL (future)** — not required for Phase 2/3

CSV and SQLite are sufficient for thesis defence.

### 7.3 Registry Capabilities

- PPH leaderboards auto-updated after every run (sorted by both PPH and
  PPH log-ratio with disagreement flags where rankings differ)
- Negative-PPH warning flags with full config metadata
- Cross-phase experiment comparison
- Reproducibility snapshots (full config + environment)
- Dataset provenance chain per run
- Budget consumption view
- Confirmation vs. factorial filtering with `confirmation_slot_type`
- Per-language perplexity reported separately
- Per-language gradient norm ratio reporting

### 7.4 Global Constants

The registry maintains a `global_constants.yaml` file storing values
that must be fixed for the duration of the study and cannot be modified
once written:

```yaml
# /workspace/registry/global_constants.yaml
baseline_perplexity:
  pooled: <float, set after §4.2 procedure>
  per_language:
    swahili: <float>
    kikuyu: <float>
    kamba: <float>
    luo: <float>
  computed_at: <ISO timestamp>
  tokenizer_version: <string, must match Phase-0 winner>

tost_delta_pph:                       # see §10.3.4
  value: <float, set before Phase 3 begins>
  computed_at: <ISO timestamp>
  derivation: "5_percent_of_phase2_mean_pph"
  phase2_runs_included: 36
  phase2_mean_pph: <float>

phase2_throughput_calibration:        # NEW in v1.5 — see §9.5
  micro_benchmark_tokens_per_sec: <float>
  expected_phase2_gpu_hours_per_run: <float>
  expected_phase2_total_gpu_hours: <float>
  computed_at: <ISO timestamp>

tokenizer_version:
  value: <string, set after Phase 0>
  computed_at: <ISO timestamp>

spec_version_current:
  value: "1.5"
  spec_hash: "<sha256 of this document>"
```

Once a constant is written with a non-null value, the registry refuses
overwrites. The launcher refuses to start a phase if any required
constant for that phase is missing. Phase 3 cannot launch until
`tost_delta_pph.value` is non-null. Phase 2 launch is preceded by the
micro-benchmark recording `phase2_throughput_calibration` (§9.5).

---

## 8. Compute Budget Enforcement

### 8.1 Global Constraints

- **Global hard ceiling: 32,000 GPU-hours**
- Cumulative GPU-hour accounting updated after every run
- Per-experiment GPU-hour allocation and actuals tracking
- Remaining-budget forecasting based on rolling average of last 5 runs

### 8.2 Sub-Budget Allocation (REVISED in v1.5)

| Phase | Allocation | Purpose |
|---|---|---|
| Phase 0 — Tokenizer | 1,500 max | Fertility calibration; nominal 1,000 + 500 plateau-extension contingency (§6.4) |
| Phase 1 — Corpus Integration | 200 | Sanity-check adaptation on integrated corpus |
| Phase 2 — Factorial (36 runs) | 18,000 | Primary scientific contribution |
| Phase 3 — PEFT (20 runs) | 10,000 | Secondary contribution (see §8.3) |
| Micro-benchmark (Week 2) | 5 | Throughput calibration (§9.5) — NEW in v1.5 |
| Smoke tests (5 × 5 GPU-hours) | 25 | Explicit smoke-test line item |
| Reserve — Validation + Recovery | 2,270 | Confirmation runs, failure recovery, sensitivity analyses |
| **Total** | **32,000** | |

If Phase 0 consumes only the 1,000 nominal allocation (no plateau-check
extension triggered), the unused 500 GPU-hours revert to the reserve,
effectively growing it to 2,770. The 32,000 ceiling is the only hard
boundary; sub-budgets are planning targets. Per-run costs must be
measured empirically and budgets re-forecast after each phase
completes.

### 8.3 Phase 3 Design — 20-Run Joint Multilingual Training

**Design:** 4 PEFT methods × 5 random seeds = **20 runs total**, each
trained on the full multilingual corpus jointly.

**Rationale:**

1. **Matches Phase 2's training regime.** Joint training across all four
   languages isolates the PEFT-method effect cleanly.
2. **Statistically appropriate for the constraints.** 4 methods × 5
   seeds = 20 runs supports paired analyses across seeds (n=5 per
   method-pair). See §10.3 for the inferential framing.
3. **Closer to deployment scenario.** LughaGen targets a single 8B model
   serving all four languages.

**Per-language analysis as output, not input.** Each Phase 3 run
reports per-language perplexity in its registry record. Per-language
method ranking is derived from these output metrics during analysis.

**Compute allocation per cell:** 10,000 GPU-hours ÷ 20 runs = **500
GPU-hours per run** ≈ 62 hours wall-clock on 8 A100s.

**Per-language isolation as future work.** A follow-up study could
isolate per-language training (4 methods × 4 languages × 5 seeds = 80
runs). Out of scope for the current 32,000-GPU-hour budget.

### 8.4 Warning and Enforcement Thresholds

- **70% (22,400 GPU-hours)**: Warning — review remaining phase allocations
- **85% (27,200 GPU-hours)**: Warning — restrict to high-priority runs
- **95% (30,400 GPU-hours)**: Critical — freeze discretionary runs
- **100% (32,000 GPU-hours)**: Hard block — no new launches permitted

### 8.5 Reservation Enforcement

Confirmation runs, replication runs, and the Phase 3 PEFT comparison
have reserved budget. The framework refuses to launch discretionary
runs that would consume reserved allocation.

---

## 9. Training Pipeline Requirements

### 9.1 Core Training Features

- Continued pre-training (Phase 2) and PEFT adaptation (Phase 3)
- Mixed precision: BF16 weights + TF32 matrix math
- Distributed training via ZeRO-3 (DeepSpeed or FSDP)
- Gradient checkpointing
- Flash Attention 2.x (version pinned per §9.2; engagement verified per
  §14)
- Checkpoint resumption with automatic versioning
- Deterministic reproducibility (fixed seeds, deterministic CUDA ops)

### 9.2 Environment Lock Procedure

#### 9.2.1 Verification Script (Week 1 deliverable)

`verify_environment.sh` runs the following on the Brev cluster and
writes the result to `/workspace/registry/environment_lock.yaml`:

```bash
#!/usr/bin/env bash
set -euo pipefail

OUT=/workspace/registry/environment_lock.yaml
mkdir -p "$(dirname "$OUT")"

cat > "$OUT" <<EOF
# Captured by verify_environment.sh on $(date -u +%Y-%m-%dT%H:%M:%SZ)
# Host: $(hostname)
verified_versions:
  python: "$(python -c 'import sys; print(sys.version.split()[0])')"
  torch: "$(python -c 'import torch; print(torch.__version__)')"
  cuda_torch: "$(python -c 'import torch; print(torch.version.cuda)')"
  cuda_driver: "$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
  transformers: "$(python -c 'import transformers; print(transformers.__version__)' 2>/dev/null || echo MISSING)"
  peft: "$(python -c 'import peft; print(peft.__version__)' 2>/dev/null || echo MISSING)"
  datasets: "$(python -c 'import datasets; print(datasets.__version__)' 2>/dev/null || echo MISSING)"
  deepspeed: "$(python -c 'import deepspeed; print(deepspeed.__version__)' 2>/dev/null || echo MISSING)"
  bitsandbytes: "$(python -c 'import bitsandbytes; print(bitsandbytes.__version__)' 2>/dev/null || echo MISSING)"
  flash_attn: "$(python -c 'import flash_attn; print(flash_attn.__version__)' 2>/dev/null || echo MISSING)"
  lion_pytorch: "$(python -c 'import lion_pytorch; print(lion_pytorch.__version__)' 2>/dev/null || echo MISSING)"
  accelerate: "$(python -c 'import accelerate; print(accelerate.__version__)' 2>/dev/null || echo MISSING)"
  sentencepiece: "$(python -c 'import sentencepiece; print(sentencepiece.__version__)' 2>/dev/null || echo MISSING)"
  tokenizers: "$(python -c 'import tokenizers; print(tokenizers.__version__)' 2>/dev/null || echo MISSING)"
gpu_inventory:
  gpu_count: $(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
  gpu_type: "$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
  total_vram_per_gpu_gb: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | awk '{print int($1/1024)}')
disk:
  workspace_free_gb: $(df -BG /workspace | awk 'NR==2 {gsub("G",""); print $4}')
EOF

echo "Wrote environment lock to $OUT"
cat "$OUT"
```

#### 9.2.2 Lock-File Schema

`/workspace/registry/environment_lock.yaml` is the canonical record of
the installed environment. Every run's `environment` block in the
registry is populated by re-reading this lock file at run-start time.

#### 9.2.3 Acceptance Criteria

- No package in §9.2.1's verification list reports `MISSING`.
- `flash_attn` version is ≥ 2.0.
- `lion_pytorch` version is captured and pinned to the exact value seen
  on the cluster.
- `torch` CUDA version matches `cuda_driver` major version.
- GPU count = 8; total VRAM per GPU ≥ 70 GB.
- `/workspace` has ≥ 500 GB free.

If any acceptance criterion fails, raise a blocker and resolve before
Phase 0 begins.

#### 9.2.4 Drift Detection

Every preflight check (§12) re-reads `environment_lock.yaml` and
compares against the currently-installed versions. If any version has
drifted, the preflight refuses launch and prints the diff.

#### 9.2.5 Backward Compatibility

If a package upgrade is required mid-study (e.g., a security patch):

1. Update the package on the cluster.
2. Re-run `verify_environment.sh` to refresh `environment_lock.yaml`.
3. Append the previous environment lock to a versioned archive at
   `/workspace/registry/environment_lock_history/<timestamp>.yaml`.
4. Annotate any in-progress runs as `environment_drift: true`.

### 9.3 Mandatory Operational Safeguards (from pilot Appendix A.4)

- `local_rank==0` guards on all save operations
- `torch.distributed.barrier()` before and after every `save_model()`
  call
- `destroy_process_group()` cleanup on run completion
- `tmux + nohup` for all distributed training sessions
- `nvidia-smi` GPU count verification immediately after every
  `torchrun`
- Memory pre-check before any new configuration
- Per-device batch size capped at 8; gradient accumulation to achieve
  target effective batch size

### 9.4 Per-Run Tracked Metrics (REVISED in v1.5)

- Tokens processed (cumulative)
- Throughput (tokens/sec, samples/sec)
- GPU utilisation (%)
- VRAM peak usage (GB)
- Training loss (every step)
- Validation loss (every 50 steps)
- Gradient norm (per step)
- **Per-language gradient norm (NEW in v1.5).** Every 50 steps, the
  forward pass partitions the current batch by source language using
  the record's `language` field, computes per-partition loss, performs
  a separate backward pass per partition, and records the gradient norm
  before zeroing for the next partition. The per-language ratios are
  reported as Swahili-norm-to-other-language-norm ratios:
  `swahili_to_kikuyu`, `swahili_to_kamba`, `swahili_to_luo`. A
  ratio close to 1.0 indicates balanced gradient signal; a ratio
  substantially > 1.0 indicates Swahili dominance.
- Gradient anomaly events
- Training instability events
- Runtime failure events and recovery actions

### 9.5 Phase 2 Throughput Calibration (NEW in v1.5)

A throughput micro-benchmark runs in Week 2 (immediately after the
smoke test) to empirically measure tokens/sec on the actual cluster
configuration. The pilot's per-run wall-clock estimates are based on
assumed throughput; the micro-benchmark verifies or revises them.

**Procedure:**

1. Run a single 1,000-step training run at the pilot's default
   configuration (cosine LR schedule, batch_size 256, AdamW,
   seq_length 512, mixed data ordering).
2. Log tokens/sec, samples/sec, peak VRAM, and wall-clock seconds.
3. Compute expected Phase 2 per-run wall-clock from observed tokens/sec
   and the planned 3-epoch × ~33M token training budget.
4. Write to `/workspace/registry/global_constants.yaml`:
   ```yaml
   phase2_throughput_calibration:
     micro_benchmark_tokens_per_sec: <observed>
     expected_phase2_gpu_hours_per_run: <derived>
     expected_phase2_total_gpu_hours: <derived × 32 factorial runs>
     computed_at: <timestamp>
   ```
5. If `expected_phase2_total_gpu_hours` differs from the 18,000-hour
   allocation by more than ±15%, raise a blocker; supervisor review
   re-allocates budget before Phase 2 launch.

**Why this matters.** 562 GPU-hours per run assumes ~1.5k tokens/sec
sustained. If the cluster sustains 2.5k tokens/sec, the actual budget
is ~10,800 hours, freeing ~7,200 hours for additional confirmation
runs, replication seeds, or expanded sensitivity analyses. Conversely,
if throughput falls below 1.0k tokens/sec, Phase 2 would overspend by
~25%, threatening the global ceiling. The micro-benchmark surfaces
this in 3 GPU-hours rather than 14 hours into the first real run.

**Budget impact.** 5 GPU-hours allocated in §8.2 (covers the 1,000-step
run plus overhead).

---

## 10. Evaluation Framework

### 10.1 Intrinsic Evaluation (Phase 2)

- Validation perplexity on held-out LughaGen corpus (every 50 steps)
- Cross-language perplexity (per-language breakdown — mandatory given
  §2.3 language asymmetry)
- Language retention analysis
- Code-switch robustness (perplexity on code-switched test subset)
- PPH analysis (primary ranking metric) + PPH log-ratio (secondary
  robustness metric)
- Per-language gradient norm ratio (Swahili-vs-others) — reports whether
  the smaller languages received proportionate gradient signal during
  training

### 10.2 Extrinsic Evaluation (representative subset, coordinated with Trust workstream)

- MasakhaNER (named entity recognition)
- AfroMT (translation benchmarks)
- Rank correlation between PPH and downstream task performance reported

### 10.3 Statistical Validation

#### 10.3.1 The n=5 Wilcoxon Ceiling

The Phase 3 design (20 runs = 4 PEFT methods × 5 seeds) yields n=5
paired observations per method-pair comparison. **The Wilcoxon
signed-rank test at n=5 has a minimum achievable two-tailed p-value of
0.0625.** No method-pair comparison can achieve conventional p < 0.05
significance via Wilcoxon, regardless of how strongly the data favour
one method. This is a hard ceiling imposed by the discrete sample space
at n=5, not a power deficit that better data could resolve.

#### 10.3.2 Inferential Framework

**Phase 3 conclusions rely on:**

1. **Effect-size estimation as primary inference.** Bootstrap 95%
   confidence intervals (1,000 resamples) for the PPH difference between
   each method-pair. These intervals are informative regardless of any
   p-value and form the basis for ranking claims. PPH log-ratio CIs
   (§4.5) are reported alongside for robustness.
2. **TOST equivalence testing as a positive-result mechanism.** When the
   bootstrap CI for a method-pair difference lies entirely within the
   pre-registered equivalence band [−δ, +δ] (see §10.3.4 for δ
   pre-registration), the two methods are declared compute-efficiency
   equivalent.
3. **Wilcoxon p-values as diagnostic only.** Reported in supplementary
   tables for transparency but not used as a primary inference vehicle.

**Phase 2 factorial analysis** uses linear model fits with bootstrap
CIs on factor-effect estimates. n=2 seeds per factorial cell × 16 cells
= 32 runs provides sufficient power for main-effect detection under the
Resolution V design's clean alias structure.

#### 10.3.3 Limitations Documentation Requirement

The thesis's Limitations section (Chapter 5) must include paragraphs
documenting:

1. The n=5 Wilcoxon ceiling, the choice to lean on bootstrap and TOST,
   and the implications for superiority claims.
2. The **Lion learning-rate conservatism (§3.1)**: Lion is evaluated at
   AdamW's preferred LR (2 × 10⁻⁴) rather than at a Lion-tuned LR. The
   "Optimiser=Lion" main effect therefore represents *swapping
   optimisers at otherwise-identical hyperparameters* rather than
   *Lion-at-its-best vs AdamW-at-its-best*. If Lion underperforms
   AdamW in the factorial, this should not be interpreted as Lion being
   inferior in general; only as Lion being inferior at the held LR.
3. The **language-distribution asymmetry (§2.3)**: per-language
   gradient norm ratios from §9.4 are reported; if Swahili gradients
   substantially dominate the smaller languages' gradients, this
   limitation is acknowledged.
4. The **code-switching coverage asymmetry (§2.3, NEW in v1.5.1)**: all
   code-switched training data originates from the Sheng corpus
   (Swahili-English only). Kikuyu, Kamba, and Luo training data
   contains zero code-switched content. Any CS evaluation on these
   three languages measures zero-shot CS generalisation from
   monolingual training, not learned CS capability. The proposal's
   code-switching contribution claim is restricted to Swahili.
5. The **synthetic-CS admission asymmetry (§2.3, NEW in v1.5.1)**: the
   Sheng corpus (synthetic, third-party generator) is admitted to
   training while the LughaGen-pipeline-generated synthetic CS
   (synthetic, in-project generator) is excluded. This asymmetry is
   quality-based, not origin-based: pipeline QA found ~95% disagreement
   between LughaGen-pipeline CS labels and actual code-switching
   content. The thesis frames this as *"all admitted synthetic CS data
   was assessed for CS quality; sources failing the quality bar were
   excluded regardless of which generator produced them."* An examiner
   may legitimately press on the quality criterion; the response is
   that the criterion (English-marker ratio agreement with declared CS
   label) is publishable and reproducible.

#### 10.3.4 Pre-Registration of TOST Equivalence Band δ

The TOST equivalence band δ is a single global scalar used across all
Phase 3 method-pair equivalence tests. To prevent post-hoc selection
that could be argued to favour equivalence claims, δ is computed and
written to the registry **before any Phase 3 run launches**.

**Derivation procedure:**

1. After all 36 Phase 2 runs complete (32 factorial + 4 confirmation),
   compute the mean PPH across all 36 runs:
   ```
   mean_pph_phase2 = mean([run.metrics.pph_score for run in Phase2 runs])
   ```
2. Set δ as 5% of this mean:
   ```
   delta_tost_pph = 0.05 * mean_pph_phase2
   ```
3. Write to `/workspace/registry/global_constants.yaml` (§7.4).
4. The registry refuses overwrites of `tost_delta_pph.value` once
   written. The Phase 3 launcher refuses to start any Phase 3 run if
   `tost_delta_pph.value` is null.

**Sensitivity reporting.** The thesis also reports Phase 3 equivalence
results at three additional δ values (3%, 7%, and 10% of mean PPH) in a
supplementary table, so readers can see how the equivalence conclusions
depend on the choice of band.

**If Phase 2 produces negative mean PPH**, δ is computed as 5% of the
*absolute value* of mean PPH, and the Phase 3 inferential framing
shifts to *"which method degrades least per GPU-hour"*.

### 10.4 Automated Report Generation

After each phase, the framework auto-generates:

- Factor-effect tables (main effects and two-factor interactions)
- PPH leaderboard rankings (pooled + per-language), with both PPH and
  PPH log-ratio shown side-by-side; disagreements flagged
- Interaction plots (10 plots for the 10 clean two-factor interactions)
- Compute-efficiency plots (PPH vs GPU-hours scatter)
- Training-loss curves and validation-perplexity curves
- Ablation summaries
- Seed-variance tables
- Per-language perplexity tables
- Per-language gradient norm ratio summary (Swahili-vs-others)
- Phase 3: TOST equivalence band plots with bootstrap CIs overlaid at
  pre-registered δ and three sensitivity δ values

---

## 11. Checkpoint Management

- Checkpoint every N steps (configurable; default: 200)
- Retain final checkpoint + top-3 intermediate checkpoints by validation
  perplexity
- All checkpoint paths logged to the registry immediately on save
- Checkpoints transferable to Trust and Evaluation workstream
- Internal-use research checkpoints tagged
  `LughaGen-F-8B-{factor_config}-{seed}`

### 11.1 LLaMA 3.1 License Compliance for Public Release (NEW in v1.5)

LLaMA-3.1-8B base model weights are governed by Meta's LLaMA 3.1
Community License Agreement. Public release of LughaGen-F-8B
checkpoints (which are derivative works of LLaMA-3.1-8B) requires:

**Compliance requirements:**

1. **Include the upstream LLaMA 3.1 license file** in any checkpoint
   distribution bundle (`LICENSE.llama3.1.txt`).
2. **Documentation that the derivative inherits the license.** The
   distribution README must state: *"LughaGen-F-8B is a derivative
   work of LLaMA-3.1-8B and inherits the LLaMA 3.1 Community License
   Agreement. Use of LughaGen-F-8B is subject to the same terms."*
3. **Attribution language.** *"Built with LLaMA. Llama 3.1 is licensed
   under the LLaMA 3.1 Community License Agreement, Copyright © Meta
   Platforms, Inc."*
4. **MAU threshold check.** LLaMA 3.1's license requires that any
   entity with >700M monthly active users obtain a separate commercial
   license from Meta before using the base model or derivatives. For
   JKUAT and academic re-distributors, this threshold is not binding;
   the public release must include a note that downstream commercial
   users with >700M MAU must obtain Meta's commercial license
   independently.
5. **Acceptable use policy compliance.** Distribution metadata must
   acknowledge the LLaMA 3.1 Acceptable Use Policy and prohibit use in
   the categories listed therein.

**Internal-use research checkpoints (the 32 factorial runs, the 4
confirmation runs, and the 20 Phase 3 runs) are not subject to public
release compliance** — they live on the JKUAT compute environment and
are used only by the LughaGen team during the study.

**Pre-release verification.** Before any LughaGen-F-8B checkpoint is
made publicly available, the supervisor must verify:

- Compliance items 1-5 above are met.
- The checkpoint's training data manifests (especially the Sheng
  corpus) have clear redistribution rights, or are stripped from the
  distribution if not. The Sheng corpus has
  `license: "unknown_internal_use_only"` and is now classified as
  synthetic (third-party generator output, provenance not fully
  established). Pre-release clearance must establish (a) the
  generator's identity, (b) whether redistribution of generator output
  is permitted, and (c) whether attribution to the upstream generator
  is required. See §20 risk register row.

This verification is a mandatory hard gate before any
`LughaGen-F-8B-*` checkpoint leaves the JKUAT environment.

---

## 12. Pre-Flight Checklist (REVISED in v1.5)

Before every `torchrun` invocation, a programmatic checklist runs. Any
FAIL aborts launch:

```yaml
preflight_checklist:
  - confirm_gpu_count: 8
  - confirm_vram_available_per_gpu_gb: ">= 70"
  - confirm_tmux_session_active: true
  - confirm_disk_space_free_gb: ">= 200"
  - confirm_corpus_file_exists: /workspace/data/corpus/splits/train.jsonl
  - confirm_validation_file_exists: /workspace/data/corpus/splits/validation.jsonl
  - confirm_tokenizer_version_matches_registry: true
  - confirm_registry_writable: /workspace/registry/
  - confirm_baseline_perplexity_recorded: true
  - confirm_no_zombie_gpu_processes: true
  - confirm_budget_remaining_gpu_hours_at_least: 50
  - confirm_dataset_checksum_matches_manifest: true
  - confirm_environment_lock_matches_installed: true
  - confirm_lion_pytorch_version_matches_lock: true
  - confirm_flash_attn_version_matches_lock: true
  - confirm_spec_version_current: "1.5"
  - confirm_tost_delta_pph_recorded_if_phase3: true
  - confirm_phase2_throughput_calibration_recorded_if_phase2: true  # NEW in v1.5
  - confirm_flash_attn_engaged_probe: true                           # NEW in v1.5 — see §14
```

`confirm_flash_attn_engaged_probe` runs a 10-step probe before the full
training launch and inspects FA2's internal `is_initialised` flag plus
peak VRAM during the probe. If FA2 has silently fallen back to SDPA,
peak VRAM will be ~30% higher than expected for the configuration and
the probe fails. Implementation: 30-line Python helper invoked from
`preflight.sh`.

Implementation: 90-line bash script `preflight.sh` invoked by the
launcher. Exits 0 on PASS; non-zero on any FAIL with descriptive
diagnostic. Failed checks log to registry as `runtime_failures`.

---

## 13. Kill-Switch Criteria

A monitoring sidecar process running alongside training terminates the
run if any of the following triggers:

- **Gradient norm explosion**: gradient norm > 1,000 for 50+
  consecutive steps → kill, log
  `instability_event: gradient_explosion`
- **NaN loss**: training loss is NaN for 5+ consecutive steps → kill,
  log `instability_event: nan_loss`
- **VRAM OOM imminent**: peak VRAM > 78 GB on any GPU (≥95% of 80 GB)
  for 3+ consecutive logging intervals → kill, log
  `runtime_failure: vram_pre_oom`
- **Stuck worker**: step time > 4× pilot's per-step time for 20+
  consecutive steps → kill, log `runtime_failure: stuck_worker`
- **Disk full**: < 20 GB free on `/workspace/` partition → kill, log
  `runtime_failure: disk_full`

All kills are graceful: SIGTERM → 30-second wait → SIGKILL. The kill
event is appended to the registry record before exit. The kill-switch
is independent of PPH (negative-PPH runs complete; only operational
failures trigger termination).

---

## 14. Smoke Test (REVISED in v1.5)

Before the first factorial run launches, a smoke test exercises every
code path in ≤30 minutes:

```yaml
smoke_test:
  experiment_id: "EXP-SMOKE-001"
  factors:
    lr_schedule: cosine
    batch_size: 128
    optimiser: AdamW
    sequence_length: 1024
    data_ordering: mixed
  num_train_steps: 200
  max_train_records: 10000
  validation_every_steps: 50
  expected_wall_clock_minutes: "<= 30"
  must_produce:
    - checkpoint at /workspace/checkpoints/smoke/checkpoint-200/
    - validation_perplexity (scalar, finite)
    - registry entry in CSV and SQLite
    - pph_score (scalar, finite, possibly negative)
    - pph_log_ratio (scalar, finite)                       # NEW in v1.5
    - environment block populated from environment_lock.yaml
    - per_language_grad_norm_ratio recorded                 # NEW in v1.5
    - flash_attention_engaged: true                          # NEW in v1.5
    - measured_vram_within_5pct_of_expected_for_fa2_engaged  # NEW in v1.5
    - preflight checklist passed
  must_not_produce:
    - NaN losses
    - vram_pre_oom
    - stuck_worker events
    - silent FA2 fallback (detected via VRAM excess)         # NEW in v1.5
```

**FA2 engagement verification.** The smoke test explicitly verifies
Flash Attention 2 is engaged (not silently falling back to SDPA) via
two checks:

1. Inspect `flash_attn`'s internal state to confirm
   `is_initialised: True`.
2. Compare measured peak VRAM against the FA2-expected reference. For
   the smoke test's configuration (batch 128, seq 1024, QLoRA r=16),
   FA2-engaged peak VRAM should be ~17 GB ± 2 GB per GPU. SDPA
   fallback would show ~25 GB ± 3 GB. A peak VRAM > 22 GB on any GPU
   triggers a `silent_fa2_fallback` runtime failure.

The smoke test is run once after Phase 1 completes and again any time
the training code, corpus, environment lock, or CUDA/driver version is
changed materially. Smoke tests consume budget from the dedicated
25-GPU-hour allocation (§8.2).

---

## 15. Phase Sequencing (Hard Gating)

```
Verify Environment (Week 1)
  ──→ Phase 0 (Tokenizer, with plateau check)
  ──→ Phase 1 (Corpus Integration)
  ──→ Smoke Test (FA2 engagement verified)
  ──→ Micro-Benchmark (throughput calibration)
  ──→ Phase 2 (Factorial, 36 runs)
  ──→ Compute & lock δ_TOST (§10.3.4)
  ──→ Phase 3 (PEFT, 20 runs)
```

A phase cannot begin until the prior phase has written its outputs to
the registry and been marked `phase_complete: true`. Environment
verification (§9.2) is a Week 1 hard gate; Phase 0 cannot begin until
`environment_lock.yaml` exists. **Phase 2 cannot begin until
`global_constants.yaml.phase2_throughput_calibration` is recorded.**
**Phase 3 cannot begin until
`global_constants.yaml.tost_delta_pph.value` is non-null.**

---

## 16. Out-of-Scope — Future Work Only

The following must not consume any GPU budget:

- Models larger than 8B parameters
- Instruction tuning or RLHF
- Deployment stack or inference optimization
- Architectures unrelated to LLaMA-3.1-8B
- Languages beyond Swahili, Kikuyu, Luo, Kamba
- Per-language isolated PEFT comparison (80-run design)
- Balanced-corpus training as primary regime
- Increasing Phase 3 seeds beyond 5
- Post-hoc adjustment of TOST equivalence band δ after Phase 3 begins
- Lion learning-rate tuning (Lion held at AdamW's LR by design — see
  §3.1)
- Public release of LughaGen-F-8B checkpoints (subject to §11.1
  compliance gate; gate decision is out of scope of the current 32K
  budget)

---

## 17. Implementation Timeline (REVISED in v1.5)

The framework is built and run in seven weeks, with each week producing
a usable deliverable.

### Week 1 — Critical-path scaffolding + environment verification (no GPU)

**Deliverables:**

- `verify_environment.sh` — writes `environment_lock.yaml`. **Must run first.**
- `factorial_matrix.py` — generates the 16 base configurations
- `registry_csv.py` — CSV backend
- `migrate_spec_version.py` — spec-version lock procedure per §18
- `preflight.sh` — §12 checklist (16 checks)
- `train_orchestrator.py` — wraps the pilot QLoRA training script
- `compute_baseline_perplexity.py` — §4.2 procedure
- `compute_tost_delta.py` — §10.3.4 procedure
- `select_confirmation_runs.py` — §3.4 procedure with bootstrap CIs
  (NEW in v1.5)
- `verify_fa2_engaged.py` — §12, §14 procedure (NEW in v1.5)
- `partition_loss_by_language.py` — §9.4 per-language gradient
  procedure (NEW in v1.5)

**Acceptance:**

- `environment_lock.yaml` exists with all §9.2.3 acceptance criteria met
- Factorial matrix generates exactly 32 rows
- Registry writes pass round-trip serialisation
- Preflight passes on the existing Brev environment
- Spec-version migration leaves prior runs untouched
- `select_confirmation_runs.py` produces sensible bootstrap CIs on a
  fixture of 32 dummy Phase 2 runs
- `verify_fa2_engaged.py` correctly detects FA2 vs SDPA on a CPU
  fixture (mocked) and on a one-step GPU probe

**GPU consumption:** 0 (CPU work only)

### Week 2 — Smoke test + Phase 0 (tokenizer) + Throughput micro-benchmark

**Deliverables:**

- Run §14 smoke test (includes FA2 engagement verification)
- Run §9.5 throughput micro-benchmark; write
  `phase2_throughput_calibration` to global constants
- Implement the three tokenizer strategies
- Run Phase 0: three short adaptation runs with plateau check (§6.4);
  extend to 1,500 steps if any strategy hasn't converged
- Pick the winning tokenizer per §6.4; freeze and log
- Add `registry_sqlite.py` — SQLite backend
- Generate Phase 0 report including PPL convergence curves

**Acceptance:** smoke test passes in ≤30 min with FA2 engagement
confirmed; throughput micro-benchmark produces calibrated Phase 2
budget forecast within ±15% of the 18,000-hour allocation;
tokenizer-comparison report in `/workspace/registry/phase0_report.md`
with PPL curves; registry has `tokenizer_version` set.

**GPU consumption:** ~1,010 GPU-hours (5 micro-benchmark + 1,005
Phase 0, assuming no plateau extension; up to ~1,510 if extension
triggered)

### Week 3 — Compute baseline + Phase 1 (corpus integration)

**Deliverables:**

- Run `compute_baseline_perplexity.py`
- Phase 1 sanity-check adaptation
- `kill_switch.py` — §13 monitoring sidecar
- (Optional) MLflow integration

**Acceptance:** P_baseline recorded; one Phase-1 reference run
completes end-to-end; kill-switch tested via manual NaN-injection.

**GPU consumption:** ~205 GPU-hours

### Weeks 4–5 — Phase 2 factorial (32 base + 4 confirmation)

**Deliverables:**

- Launch the 32 factorial runs
- After every batch, refresh PPH and PPH log-ratio leaderboards
- After all 32 complete, run `select_confirmation_runs.py` with
  bootstrap CIs; review any flagged overlapping-CI candidates with
  supervisor
- Launch the 4 confirmation runs (3 PPH-predicted + 1 paired
  balanced-recipe)
- **After all 36 Phase 2 runs complete, run `compute_tost_delta.py`**
  to pre-register δ_TOST in `global_constants.yaml` (§10.3.4)
- Generate Phase 2 report including per-language gradient norm ratio
  summaries and PPH-vs-log-ratio agreement check

**Acceptance:** 36 registry entries with `phase_complete: true`;
factor-effect ANOVA p-values reported; 3 PPH-predicted confirmation
runs in top-3 by mean predicted PPH with bootstrap CIs documented;
slot-4 paired analysis produces TOST equivalence result;
`tost_delta_pph.value` is non-null; per-language gradient norm ratio
table populated for all 36 runs.

**GPU consumption:** ~18,000 GPU-hours

### Weeks 6–7 — Phase 3 PEFT comparison (20 runs)

**Deliverables:**

- 4 PEFT methods × 5 seeds = 20 runs, each ~500 GPU-hours wall-clock
- Per-language perplexity reported per run
- Statistical analysis per §10.3 (bootstrap CIs primary, TOST at
  pre-registered δ as positive-result mechanism, Wilcoxon diagnostic
  only)
- Sensitivity reporting at δ = 3%, 7%, 10% of Phase 2 mean PPH
- Final PEFT comparison report including per-language rankings and
  PPH-vs-log-ratio agreement check

**Acceptance:** 20 registry entries; bootstrap 95% CIs for all 6
method-pair PPH differences; TOST equivalence declarations at
pre-registered δ; sensitivity table at three additional δ values;
per-language method-ranking table populated.

**GPU consumption:** ~10,000 GPU-hours

### Buffer week — Reserve + thesis writing

**Deliverables:**

- Use the ~2,270 GPU-hour reserve for: confirmation-run replications,
  additional sensitivity analyses, recovery from any failures.
- Write Chapter 4 (results) and Chapter 5 (limitations: n=5 ceiling,
  Lion LR conservatism, language asymmetry, TOST δ pre-registration).

### Total

- Wall-clock: **7 weeks** with concurrent factorial; **10–12 weeks**
  serial.
- GPU-hours: **32,000 ± 5%** (within budget)

---

## 18. Spec Version Lock Procedure

The framework maintains `/workspace/registry/spec_versions.yaml` as the
canonical record of every spec version used during the study.

### 18.1 Spec versions file schema

```yaml
spec_versions:
  - version: "1.0"
    sha256: "<hash>"
    file: "LughaGen_Training_Framework_Spec_v1.0.md"
    valid_from: "2026-05-17T08:00:00Z"
    valid_until: "2026-05-17T15:00:00Z"
    superseded_by: "1.1"
  - version: "1.1"
    sha256: "<hash>"
    file: "LughaGen_Training_Framework_Spec_v1.1.md"
    valid_from: "2026-05-17T15:00:00Z"
    valid_until: "2026-05-17T22:00:00Z"
    superseded_by: "1.2"
  - version: "1.2"
    sha256: "<hash>"
    file: "LughaGen_Training_Framework_Spec_v1.2.md"
    valid_from: "2026-05-17T22:00:00Z"
    valid_until: "2026-05-17T23:30:00Z"
    superseded_by: "1.3"
  - version: "1.3"
    sha256: "<hash>"
    file: "LughaGen_Training_Framework_Spec_v1.3.md"
    valid_from: "2026-05-17T23:30:00Z"
    valid_until: "2026-05-18T00:15:00Z"
    superseded_by: "1.4"
  - version: "1.4"
    sha256: "<hash>"
    file: "LughaGen_Training_Framework_Spec_v1.4.md"
    valid_from: "2026-05-18T00:15:00Z"
    valid_until: "2026-05-18T01:00:00Z"
    superseded_by: "1.5"
  - version: "1.5"
    sha256: "<hash>"
    file: "LughaGen_Training_Framework_Spec_v1.5.md"
    valid_from: "2026-05-18T01:00:00Z"
    valid_until: null
    superseded_by: null
```

### 18.2 Lock procedure (`migrate_spec_version.py`)

When the spec is updated:

1. Compute SHA-256 of the new spec file.
2. Set `valid_until` of the previous version to the current UTC time
   and `superseded_by` to the new version number.
3. Append the new version to `spec_versions.yaml`.
4. Do not modify any existing registry runs' `spec_hash` field.
5. The launcher checks `spec_versions.yaml` at preflight (§12) and
   refuses to launch if the currently-loaded spec file's SHA-256 does
   not match the most recent entry with `valid_until: null`.

### 18.3 Auditability guarantee

The `spec_hash` field on every run in the registry, joined with
`spec_versions.yaml`, reconstructs which version of the spec governed
each run.

---

## 19. Hand-off Strategy — Implementing Against This Spec

1. **Week 1 — hand sections §3, §4, §5, §7, §9.2, §18 + Week 1
   deliverables list.** Reject any output adding MLflow or W&B before
   Week 3.
2. **Week 2 — add §6, §9.5, §10.3, §12, §14.**
3. **Week 3 — add §13.**
4. **Weeks 4–5 — hand the whole spec.**

The spec itself is registered as a registry artefact: SHA-256 hash
written to every run's `spec_hash` field.

---

## 20. Risk Register (REVISED in v1.5.1)

| Risk | Mitigation |
|---|---|
| Phase 2 runs exceed planned wall-clock | Kill-switch §13; sub-budgets re-forecast each phase; throughput micro-benchmark §9.5 calibrates upfront |
| Tokenizer change required mid-study | §4.2 mandates P_baseline recomputation |
| Factorial runs produce all-negative PPH | §4.3 logs but doesn't abort; §10.3.4 handles negative mean PPH for δ derivation |
| Lion implementation divergence | §9.2 environment lock; preflight verifies |
| Disk fills mid-Phase-2 | Kill-switch; reserve allocation |
| **Sheng corpus license + provenance question (UPDATED in v1.5.1)** | Marked `unknown_internal_use_only`; now classified synthetic (§2.3 Decision 2); §11.1 pre-release verification establishes generator identity, redistribution rights, attribution requirements; §10.3.3 limitations documents the asymmetric admission rule |
| ZeRO-3 + Flash Attention 2 interaction bug | §14 smoke test verifies FA2 engagement via VRAM comparison; §12 preflight runs FA2 probe before each launch |
| Per-language asymmetry overwhelms smaller-language signal | Slot-4 paired sensitivity (§3.4); per-language gradient norm ratio tracked per run (§9.4); thesis Limitations chapter documents (§10.3.3) |
| Spec version drift mid-study | §18 lock procedure |
| Phase 3 n=5 Wilcoxon ceiling | §10.3 reframes around bootstrap CIs and TOST |
| Environment lock includes stale version | §9.2.3 acceptance criteria; §9.2.4 drift detection |
| Cluster CUDA version incompatibility | §9.2.1 verification captures actual versions |
| Post-hoc TOST δ selection objection | §10.3.4 pre-registration; registry refuses overwrite; sensitivity table at three additional δ values |
| Throughput substantially over/under-estimates causing budget mis-allocation | §9.5 throughput micro-benchmark runs in Week 2 (5 GPU-hours); flags >±15% deviation from 18,000-hour Phase 2 allocation for supervisor review before Phase 2 launches |
| Phase 0 tokenizer selection biased by non-convergence | §6.4 plateau check at step 1,000; uniform extension to step 1,500 if any strategy slope >0.5%/100 steps; non-convergence after step 1,500 logged as warning |
| PPH metric noise dominates when perplexity deltas are small | §4.5 PPH log-ratio logged as secondary; PPH-vs-log-ratio rank disagreements flagged in reports |
| Confirmation-run linear-model predictions mislead under strong 3-way interactions | §3.4 bootstrap CIs on predictions; 50% CI-overlap with rank-4 triggers manual supervisor review before launch |
| Lion underperforms due to held LR not LR-tuned setting | §3.1 documents the conservatism; §10.3.3 limitations chapter notes that "Optimiser=Lion" measures swap-at-held-LR not Lion-at-best-LR |
| Flash Attention 2 silently falls back to SDPA causing throughput loss | §14 smoke test verifies FA2 engagement via internal flag + VRAM comparison; §12 preflight runs 10-step FA2 probe before each training launch; SDPA fallback raises `runtime_failure: silent_fa2_fallback` |
| LLaMA 3.1 license violation on public checkpoint release | §11.1 compliance gate: license file inclusion, attribution language, MAU note, AUP acknowledgment, supervisor sign-off mandatory before any LughaGen-F-8B public release |
| **Examiner challenge: code-switching results restricted to Swahili (NEW in v1.5.1)** | §2.3 documents the zero CS coverage for Kikuyu/Kamba/Luo; §10.3.3 limitations bullet 4 frames CS evaluation on these three as "zero-shot CS generalisation from monolingual training"; proposal's CS contribution claim restricted to Swahili |
| **Examiner challenge: synth-CS admission asymmetry (NEW in v1.5.1)** | §2.3 Decisions 1 and 2 document the asymmetric admission rule; §10.3.3 limitations bullet 5 frames it as quality-based not origin-based; the ~95% pipeline-label disagreement evidence is publishable; archived synth-CS manifests at `_excluded/synthetic_cs_2026-05-17/` for audit |
| **Corpus state drift between v1.5 and v1.5.1 (NEW in v1.5.1)** | Four snapshots preserved on `/workspace/` documenting the transition: pre-exclusion (16:25), post-exclusion (16:31), Sheng-relabel (16:37, canonical); §18 spec-hash mechanism ensures runs are auditable to the spec version that governed them |

---

## 21. Spec Version History

- **v1.0** (2026-05-17, 08:00 UTC): Initial specification.
- **v1.1** (2026-05-17, 15:00 UTC): Defining relation, P_baseline,
  negative-PPH resolution, confirmation-run selection, dataset
  hand-off, Flash Attention pinning, preflight, kill-switch, smoke
  test, timeline, hand-off, risk register.
- **v1.2** (2026-05-17, 22:00 UTC): Language-distribution conflict
  resolved; smoke-test line item; Phase 3 = 20 runs; spec version
  lock; Lion pinning.
- **v1.3** (2026-05-17, 23:30 UTC): Sign-convention fix (§6.4); n=5
  Wilcoxon caveat (§10.3); slot-4 paired sensitivity (§3.4);
  environment-verification procedure (§9.2).
- **v1.4** (2026-05-18, 00:15 UTC): Pre-registration of TOST δ
  (§10.3.4 + §7.4 + §15); glossary (§22).
- **v1.5** (2026-05-18, 01:00 UTC, this document): Eight final fixes:
  - **§9.5 + §17 Week 2 + §8.2**: Throughput micro-benchmark
    procedure; 5 GPU-hour line item; calibrates Phase 2 budget before
    launch.
  - **§6.4 + §8.2 + §17 Week 2**: Phase 0 plateau check with uniform
    extension to step 1,500; budget reflects up-to-1,500 max with
    500-hour contingency.
  - **§4.5 + §7.1 + §10.4**: PPH log-ratio as secondary robustness
    metric; logged alongside PPH; rank disagreements flagged.
  - **§9.4 + §7.1 + §10.1**: Per-language gradient norm ratio
    procedure; tracked every 50 steps; reported in Phase 2 evaluation.
  - **§11.1 + §16 + §20**: LLaMA 3.1 license compliance gate for
    public checkpoint release; internal-use checkpoints unaffected.
  - **§3.4 + §17 Week 1**: Bootstrap CIs on confirmation-run
    predictions; 50%-overlap rule triggers manual review;
    `select_confirmation_runs.py` Week 1 deliverable.
  - **§3.1 + §10.3.3 + §16**: Lion LR held constant at AdamW's
    preferred LR; conservatism explicitly documented as a thesis
    limitation.
  - **§14 + §12 + §20**: Flash Attention 2 engagement verification in
    smoke test (VRAM comparison) and preflight (10-step probe); silent
    fallback raises `runtime_failure: silent_fa2_fallback`.
- **v1.5.1** (2026-05-17, 16:40 UTC, this document): Point release
  capturing two corpus-level decisions from the 2026-05-17 inventory
  review. No factorial-design or registry-schema changes:
  - **§0, §2.3 Decision 1**: LughaGen-pipeline-generated code-switched
    data (45,883 docs, 708K words across 4 languages) excluded from
    active training; archived at
    `/workspace/data/corpus/_excluded/synthetic_cs_2026-05-17/`.
    Rationale: pipeline QA found ~95% disagreement between CS labels
    and actual code-switching content. Corpus drops from 1,524,640 →
    1,478,757 documents.
  - **§0, §2.3 Decision 2**: Sheng corpus reclassified from
    `authentic` to `synthetic`. Provenance verification established
    that `syntheticsentences.jsonl` is third-party generator output.
    Manifest relocated from `manifests/authentic_code_switched/` to
    `manifests/synthetic_code_switched/`. Corpus contents unchanged;
    origin distribution shifts: authentic 73.51% → 47.42%, synthetic
    26.49% → 52.58%.
  - **§5.3, §5.4**: Updated dataset metadata example and recipes
    table to reflect post-decision corpus state. `auth_cs_only` recipe
    now returns zero records (no authentic CS exists).
  - **§10.3.3**: Added limitations bullets 4 and 5 — code-switching
    coverage restricted to Swahili; synth-CS admission asymmetry
    documented as quality-based, not origin-based.
  - **§11.1**: Sheng pre-release verification tightened to require
    establishing generator identity, redistribution rights, and
    attribution requirements.
  - **§20**: Sheng-license risk row updated; three new risk rows for
    examiner challenges and corpus state drift.
  - Canonical corpus state:
    `/workspace/corpus_snapshot_sheng_relabeled_2026-05-17T163736Z.tar.gz`.

---

## 22. Glossary

Terms appear in the order they are first used in the spec.

**Continued pre-training (CPT).** Further training of an
already-pretrained language model on a new corpus, using the same
self-supervised next-token-prediction objective as the original
pretraining. Distinct from fine-tuning, which targets specific
downstream tasks; and from instruction tuning, which uses a supervised
instruction-response format. CPT preserves the model's general
language-modelling capability while adapting its representations to the
new corpus.

**LLaMA-3.1-8B.** Open-weight language model from Meta with
approximately 8.03 billion parameters, 128,256-token vocabulary, and
131,072-token maximum context length. The base architecture for all
LughaGen experiments. Selected for being the largest model that fits
the per-run compute envelope when adapted via QLoRA (Appendix A.1 of
the proposal).

**PEFT (Parameter-Efficient Fine-Tuning).** A family of methods that
adapt large pre-trained models by updating only a small fraction of
parameters (typically 0.1–1%) while keeping the base model frozen.
LughaGen compares four PEFT strategies in Phase 3: LoRA, QLoRA,
adapter tuning, and full fine-tuning (the last as a capacity
upper-bound reference, not a PEFT method).

**LoRA (Low-Rank Adaptation).** PEFT method (Hu et al., 2022) that
injects trainable low-rank matrices into the attention layers of a
frozen base model. The rank `r` controls the parameter count of the
adapter.

**QLoRA (Quantised LoRA).** Variant of LoRA (Dettmers et al., 2023)
that loads the base model in 4-bit precision (NF4 quantisation) and
trains LoRA adapters in BF16. Reduces per-GPU memory by ~6.9× compared
to full fine-tuning of LLaMA-3.1-8B (Appendix A.1).

**Adapter tuning.** PEFT method (Houlsby et al., 2019) that inserts
small trainable feedforward modules between the layers of a frozen
base model. An older method than LoRA but with different
parameter-allocation properties.

**Full fine-tuning.** Standard training where all model parameters are
updated. For LLaMA-3.1-8B requires ~106 GB per GPU (Appendix A.1) and
must be sharded across all 8 GPUs via ZeRO-3 to fit in 80 GB A100
memory. Included in Phase 3 as a capacity reference, not as a
recommended deployment path.

**ZeRO-3.** Memory optimisation technique for distributed training
that partitions model parameters, gradients, and optimiser states
across workers. Necessary for full fine-tuning of LLaMA-3.1-8B; not
used for LoRA/QLoRA which fit on a single GPU.

**Resolution V fractional factorial.** Experimental design (Box,
Hunter & Hunter, 2005) where main effects and two-factor interactions
are not aliased with each other or with other main effects. The
2^(5-1) Resolution V design used in LughaGen tests 5 factors at 2
levels each using 16 runs (half-fraction of the full 32-run design),
with clean estimates of all main effects and all 10 two-factor
interactions.

**Defining relation.** For a fractional factorial design, the equation
that specifies which higher-order interaction is confounded with the
identity (and therefore which alias structure the design has).
LughaGen uses `E = ABCD`, meaning factor E's level pattern is
determined by the product of A, B, C, and D levels.

**Alias structure.** The set of effects that cannot be distinguished
from each other in a fractional factorial design. At Resolution V
under `E = ABCD`, each main effect is aliased with a four-factor
interaction (assumed negligible), and each two-factor interaction is
aliased with a three-factor interaction (also assumed negligible).
Main effects and two-factor interactions are thus "clean".

**Confirmation run.** Additional runs performed after a fractional
factorial to test the model's prediction at configurations that were
not in the original design. LughaGen performs 4 confirmation runs
after the 32-run Phase 2 factorial: 3 at the highest-predicted-PPH
untested configurations (with bootstrap CIs for uncertainty
quantification), plus 1 paired sensitivity run at the same factor
settings as confirmation slot 1 but using the
`balanced_25_25_25_25` recipe.

**Perplexity.** Standard language-modelling metric defined as
`exp(mean(cross_entropy_per_token))`. Lower is better. A perplexity of
P means the model is, on average, as uncertain about the next token
as if it were uniformly choosing among P options.

**PPH (Performance-Per-GPU-Hour).** LughaGen's primary optimisation
metric, defined as `(P_baseline − P_config) / GPU-hours_consumed`.
Positive PPH indicates the configuration improves on the zero-shot
baseline; higher PPH indicates greater improvement per GPU-hour
spent.

**PPH log-ratio (NEW in v1.5).** Secondary robustness metric defined
as `log(P_baseline / P_config) / GPU-hours_consumed`. Captures
multiplicative perplexity improvement and is more robust to seed noise
when perplexity deltas are small. Reported alongside PPH in all
leaderboards; rank disagreements between PPH and PPH log-ratio are
flagged in reports.

**P_baseline.** Validation perplexity of zero-shot LLaMA-3.1-8B (no
adaptation, base weights) on the held-out LughaGen corpus, computed
once with the Phase-0-winning tokenizer and stored as a global
constant.

**Token fertility.** Tokens per word when a tokenizer is applied to
text in a given language. Measures how "expensive" a language is for
the tokenizer in terms of sequence length per word. LughaGen
measured: Swahili 2.32, Kikuyu 3.03, Kamba 2.85, Luo 1.82, English
1.32 (Appendix A.2).

**TOST (Two One-Sided Tests).** Statistical procedure (Wellek, 2010)
for testing whether two means are equivalent within a pre-specified
band [−δ, +δ]. Unlike standard significance testing, which only
allows rejection of equivalence (declaring a difference), TOST allows
positive declaration of equivalence. Used in Phase 3 to identify PEFT
methods that produce statistically indistinguishable PPH within the
pre-registered band.

**Equivalence band (δ).** The pre-specified threshold within which two
methods are declared equivalent under TOST. LughaGen pre-registers δ
as 5% of Phase 2 mean PPH (§10.3.4) before Phase 3 begins.

**Pre-registration.** Practice of specifying experimental procedures,
analyses, and decision thresholds before data are collected (in this
context, before Phase 3 runs launch). Prevents post-hoc selection
that could be argued to favour a desired conclusion. Inherited from
clinical trials methodology.

**Bootstrap confidence interval.** Non-parametric method for
estimating the uncertainty of a statistic by resampling the data with
replacement many times (1,000 resamples in LughaGen) and reporting
the percentile range of the resampled statistic. The 95% bootstrap CI
for the PPH difference between two methods is LughaGen's primary
Phase 3 inference tool, and bootstrap CIs are also used on
confirmation-run predictions in Phase 2 (§3.4).

**Wilcoxon signed-rank test.** Non-parametric statistical test for
paired samples. At n=5 (LughaGen's Phase 3 per-method-pair sample
size), the minimum achievable two-tailed p-value is 0.0625, making
the test unsuitable as a primary inference tool. Used diagnostically
only in LughaGen (§10.3).

**Code-switching.** Linguistic phenomenon where speakers alternate
between two or more languages within a single utterance or
conversation. Common in East African contexts (Sheng =
Swahili-English code-switching; similar patterns in Kikuyu-English,
etc.). LughaGen tracks code-switched records separately from
monolingual records.

**Sheng.** Urban Kenyan code-switching register mixing Swahili and
English (and occasionally other languages). Provides authentic
code-switched data for LughaGen's Swahili corpus partition.

**Manifest.** YAML file declaring a source dataset's location,
license, domain, language, register (monolingual/code-switched), and
quality contract. LughaGen has 13 manifests covering synthetic
mono/CS and authentic mono/CS sources for all four target languages.

**Recipe.** Named filter applied to the unified corpus at training
time to produce an experimental subset (e.g., `synth_only`,
`auth_only`, `balanced_25_25_25_25`). The same physical corpus
backs many recipes; the recipe used per run is recorded in the
registry.

**Registry.** Central log of every experimental run, capturing factor
settings, hyperparameters, dataset version, environment, metrics, and
training events. Stored as both CSV and SQLite. Auditable forever via
the `spec_hash` field linking each run to the spec version that
governed it.

**Spec hash.** SHA-256 of the implementation specification document
(this file) at the time a run was executed. Written to every registry
entry's `spec_hash` field. Enables reconstruction of which spec
version governed each run even after the spec is updated.

**Phase 0 / Phase 1 / Phase 2 / Phase 3.** The four sequenced phases
of the LughaGen modelling study. Phase 0 = tokenizer comparison.
Phase 1 = corpus integration sanity check. Phase 2 = factorial design
(the primary scientific contribution). Phase 3 = PEFT method
comparison. Each phase hard-gates the next.

**Preflight.** Programmatic checklist run before every `torchrun`
invocation, verifying GPU count, disk space, corpus file integrity,
environment version match, spec version currency, and Flash Attention
engagement. Refuses launch on any FAIL.

**Kill-switch.** Monitoring sidecar that terminates a training run if
specific failure conditions trigger (gradient explosion, NaN loss,
VRAM near-OOM, stuck worker, disk full). Operates independently of
PPH — poor-quality but valid runs complete to support the
experimental design.

**Environment lock.** YAML file (`environment_lock.yaml`) capturing
the exact versions of all Python packages installed on the cluster at
Phase 0 start. Preflight refuses launch if installed versions drift
from the lock.

**Smoke test.** Short (≤30 minute) end-to-end training run exercising
every code path in the training framework. Run once after Phase 1 and
again after any material code, corpus, or environment change.
Allocated a dedicated 25-GPU-hour budget line item. Verifies FA2
engagement via VRAM comparison.

**Flash Attention silent fallback (NEW in v1.5).** Failure mode where
FlashAttention 2 fails to initialise (typically due to CUDA version
mismatch or DeepSpeed/ZeRO-3 incompatibility) and PyTorch silently
falls back to scaled dot-product attention (SDPA). The fallback works
correctly but uses ~30% more memory and runs ~25% slower. LughaGen
detects this via VRAM excess in the smoke test (§14) and via a
10-step probe in preflight (§12).

**Throughput micro-benchmark (NEW in v1.5).** Single 1,000-step
training run performed in Week 2 immediately after the smoke test, to
empirically measure tokens/sec on the actual cluster configuration.
Used to calibrate Phase 2's 18,000-hour budget forecast before the
first factorial run launches. Allocated 5 GPU-hours from the main
budget (§8.2). Mandatory hard gate: Phase 2 cannot launch until the
micro-benchmark records `phase2_throughput_calibration` in
`global_constants.yaml`.

**Per-language gradient norm ratio (NEW in v1.5).** Diagnostic metric
quantifying whether the smaller languages (Kikuyu, Kamba, Luo) receive
proportionate gradient updates given Swahili's 75% share of the
corpus. Computed every 50 training steps by partitioning each batch by
source language, computing per-partition gradient norms, and reporting
Swahili-to-other-language ratios. A ratio close to 1.0 indicates
balanced gradient signal; substantially > 1.0 indicates Swahili
dominance.

**Lion LR conservatism (NEW in v1.5).** Methodological choice to hold
peak learning rate constant at 2 × 10⁻⁴ across all 32 base factorial
runs regardless of optimiser choice (AdamW or Lion). Lion typically
prefers a 3-10× smaller LR, so Lion is being evaluated at a
non-optimal LR by design. This preserves the factorial's
orthogonality at the cost of potentially underestimating Lion's peak
performance. The thesis Limitations chapter documents this explicitly
(§10.3.3).

---

*End of LughaGen Training Framework Implementation Specification v1.5.1*
