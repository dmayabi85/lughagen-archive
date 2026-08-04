# Storage and Cleanup Plan — Gemma 2 27B FFT

## Current disk state (2026-07-28)

- Total: 19 TB
- Used: 18.6 TB
- Free: 399 GB
- Utilization: 98 percent

Chain currently reserves ~500 GB projected for FFT-S2, CPT-S2, FFT-S3, CPT-S3 checkpoints.

## Storage requirements

- Gemma 2 27B base download: 54 GB
- Gemma 2 27B remapped weights (strategy_c 8K vocab): 45 GB approx (smaller embeddings)
- Training checkpoint retention top-K=1: 45 GB
- Training state (optimizer, scheduler, RNG): 30 GB estimated
- Logs and eval outputs: 5 GB
- Total peak: 179 GB

## Timeline of disk usage

- Now to Friday: LughaGen chain runs, consumes checkpoints as it goes
- Friday: LughaGen chain completes, ~500 GB checkpoints exist
- Friday to Gemma launch: cleanup phase (see below)
- Gemma download: adds 54 GB
- Gemma remap: adds 45 GB (base retained for reference)
- Gemma FFT training: adds up to 75 GB peak (checkpoint + training state)
- Gemma FFT complete: reduces to 45 GB (final checkpoint only)

## Cleanup before Gemma launch

After LughaGen chain completes, evaluate what to keep from Phase 3:

Definitely keep:

- Phase 3 seed 1 checkpoints (RUN-P3-FFT, RUN-P3-CPT, RUN-P3-CLEAN-QLORA)
- Phase 3 seed 2 and 3 final checkpoints (for reproducibility)
- All per-language eval JSONs
- Base Llama model at /workspace/models/llama-3.1-8b-base

Candidates for deletion to free space:

- Phase 3 intermediate checkpoints (step-1200, step-2400 etc.) — keep only final and best
- Phase 2 factorial cells not in top 5 by val_pph
- Old smoke test artifacts if any remain

Target: free 100-150 GB before Gemma launch, leaving 250 GB free for Gemma work.

## Cleanup rules

- Never delete a checkpoint without confirming its eval results are saved
- Never delete anything referenced by runs.csv registry entries flagged production
- Always dry-run rm commands with ls first
- Log deletions to /workspace/gemma2_27b_prep/logs/pre_gemma_cleanup.log

## During Gemma FFT training

- Monitor disk every 30 min via existing watchdog
- Existing watchdog threshold at 99 percent — same rule applies
- If disk crosses 96 percent during Gemma training: alert, no automatic action

## Post-Gemma cleanup

Keep after experiment completes:

- Final Gemma FFT checkpoint (45 GB)
- Baseline eval results
- Per-language eval results
- Training log
- Comparison analysis document

Delete after experiment completes:

- Intermediate training checkpoints
- Optimizer state (only needed if resuming)
- Remapped model weights (can be regenerated from base + remap script)

Approximate final storage footprint after all cleanup: 55 GB retained for Gemma experiment.
