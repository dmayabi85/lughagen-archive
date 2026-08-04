# Training Loop Amendments

Chronological record of modifications to `src/lughagen/training/training_loop.py` since the framework spec v1.5.1 was finalized. Each amendment includes rationale, mechanism, and file locations.

## Amendment 8 (2026-05-27) — Token-aware embedding remap

**Problem**: `resize_token_embeddings()` silently keeps the first `len(tokenizer)` base-model rows, but Phase 0 token IDs do not correspond to Llama token IDs. Pre-Amendment 8 sweep produced val_pph ~11700 (only 16% baseline reduction) via LoRA compensation through garbled input embeddings.

**Fix**: Load pre-computed embedding tensors that map `strategy_c` 8K vocab to base-model embedding space via exact-match + subword-decompose fallback (Buzaaba et al., 2025 methodology).

**Metrics**: 34.4% exact match, 65.6% subword decomposition, 0.05% mean fallback.

**Validation**: 73% perplexity reduction in 50 steps (vs old 16% in 6000 steps).

**Location**: `src/lughagen/training/training_loop.py` around line 385.

## Amendment 10 (2026-08-01) — Configurable remap directory

**Problem**: Phase 3.5 Gemma 27B needs different remap tensors than Phase 3 Llama 8B. Hardcoded path prevented model-family swap.

**Fix**: Read `LUGHAGEN_REMAP_DIR` env var to locate remap tensors. Default preserves Llama behavior for Phase 3 runs.

**Location**: `src/lughagen/training/training_loop.py` around line 395.

## Amendment 11 (2026-08-02) — Softcap disable + SDPA routing for Gemma

**Problem**: Gemma 2 27B config carries `attn_logit_softcapping=50.0` which BOTH FlashAttention 2 and SDPA refuse in transformers 5.8.1. Requesting FA2 silently downgrades to eager (3-5x slowdown, would extend 4-day run to 12-15 days).

**Fix**: Three-part patch:
1. Peek at model config via `AutoConfig.from_pretrained()` before load.
2. If model_type starts with "gemma", disable `attn_logit_softcapping` on the config and route to SDPA.
3. Otherwise use requested attention (FA2 for Llama).
4. Spec §12 check compares against the routed impl, not hardcoded FA2.

**Trade-off**: Disabling softcapping is a small architectural modification to the base Gemma 2 model. The CPT run adapts the model to operate without softcap. Documented in thesis Chapter 4.

**Location**: `src/lughagen/training/training_loop.py` lines 356-364, 374, 389, 442-451.

## Amendment 12 (2026-08-03) — Pre-save checkpoint prune

**Problem**: `TopKCheckpointManager.register()` prunes old checkpoints AFTER new save completes. On 27B models with 337 GB checkpoints, peak disk usage doubles during save window (682 GB). Filled 1.1 TB workspace at step 400 of Phase 3.5 CPT on 2026-08-02, triggering NCCL timeout and SIGABRT crash.

**Root cause**: `CheckpointPolicyCallback.on_save` fires after HF Trainer's save completes. On 27B ZeRO-3 checkpoints, the save takes long enough for two full checkpoints (old + new) to coexist on disk.

**Fix**: New `PreSavePruneCallback` fires on `on_step_end` when `state.global_step % save_steps == 0`:
1. Deletes oldest checkpoint on disk BEFORE HF Trainer initiates new save.
2. Asserts `>=400 GB` free before allowing save to proceed (fail-fast, no mid-save NCCL timeout).
3. Registered before `CheckpointPolicyCallback`; the older `register()` prune becomes a no-op.

**Effect**: Peak disk = 1x checkpoint (~341 GB), not 2x (~682 GB).

**Location**: `src/lughagen/training/training_loop.py` around line 690 (new class) and line 889 (registration).
