# Training Crash Post-Mortems

Root-cause analyses of production training crashes and the fixes applied.

## 2026-07-30 — CPT-S2 checkpoint retention race (Phase 3)

**Symptom**: Save at step 5600 failed. Both `save_total_limit=1` in HF Trainer and `top_k=1` in `TopKCheckpointManager` were active, and they deleted each other's kept checkpoint mid-save.

**Root cause**: Redundant retention policies interfering.

**Fix**: Set `save_total_limit=None` in `TrainingArguments`, delegate retention entirely to `TopKCheckpointManager`. Documented at line 612 of training_loop.py.

**Impact**: CPT-S2 required rerun. Registry has both entries (Jul 29 buggy 19.568 + Jul 30 correct 22.291).

## 2026-08-02 17:41 UTC — Phase 3.5 CPT step-400 disk-full crash

**Symptom**: NCCL timeout at step 400 of Phase 3.5 Gemma 27B CPT. Ranks crashed with SIGABRT (ranks 2/3/4/6) and SIGTERM (others). Disk hit 100% (3 MB free out of 19 TB).

**Trajectory before crash was excellent**: Loss dropped 15.77 to 4.535 across 400 steps at only 30% through warmup. Recovery of this trajectory was not possible because HF Trainer's save wrote model shards but did not complete trainer_state.json before disk filled — resume-from-checkpoint would have failed.

**Root cause**: `CheckpointPolicyCallback.register()` fires on `on_save` AFTER save completes. On 27B ZeRO-3 models with sharded safetensors, one save writes ~337 GB. Between save-at-step-400 completing and register()'s prune deleting checkpoint-200, disk held two full checkpoints (677 GB). At 953 GB starting free, minus dataset caches and other buffers, this pushed over the 100% boundary.

**Fix**: Amendment 12 — new `PreSavePruneCallback` deletes old checkpoint BEFORE HF Trainer initiates save. Peak disk usage: 1x checkpoint. Also adds 400 GB disk pre-flight assertion; if disk margin gets thin, training fails cleanly with a Python exception rather than a mysterious 30-minute NCCL timeout.

**Recovery**: Full restart from step 0 (faster than reconstructing missing trainer_state.json). Amendment 12 confirmed working at step 400 of restart.

**Cleanup performed**: Deleted RUN-P2-C* checkpoints (128 GB), RUN-P3-FFT-S2/S3 optimizer state, RUN-P3-CPT-S3 optimizer state. Per-language eval JSONs preserved. Disk freed from 505 GB to 842 GB.
