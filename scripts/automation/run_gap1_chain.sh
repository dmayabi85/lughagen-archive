#!/bin/bash
# ==============================================================================
# LughaGen Gap 1 Chain Runner
# ==============================================================================
# Sequentially runs:
#   1. Waits for FFT-S2 (currently running) to complete
#   2. Runs per-language eval on FFT-S2 best checkpoint
#   3. Launches CPT-S2, waits, evals
#   4. Launches FFT-S3, waits, evals
#   5. Launches CPT-S3, waits, evals
#
# Failure policy: halt on any run or eval failure, write FAILED flag.
# ==============================================================================

set -u

source /workspace/.venv/bin/activate
export PATH="/workspace/.venv/bin:$PATH"

NOTIFY="/workspace/automation/notify.sh"
LOG_DIR="/workspace/logs"
CHAIN_LOG="/workspace/automation/logs/chain.log"
RESULTS_DIR="/workspace/results"

log() {
    local ts=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    echo "[$ts] $*" | tee -a "$CHAIN_LOG"
}

# ---- Utility: wait for a torchrun run to complete ----
wait_for_run() {
    local run_id="$1"
    local log_pattern="${LOG_DIR}/phase3_${run_id}_*.log"
    log "Waiting for ${run_id} to complete..."

    # Poll every 60s
    while true; do
        # Check if training process is still alive
        local workers=$(ps aux | grep train_orchestrator | grep -v grep | wc -l)

        if [ "$workers" -eq 0 ]; then
            # Process gone — check if completed successfully via registry
            sleep 15  # let registry write settle
            local val_ppl=$(python3 -c "
import csv
for r in csv.DictReader(open('/workspace/registry/runs.csv')):
    if r.get('factorial_run_id') == '${run_id}':
        v = r.get('metrics.validation_perplexity', '')
        if v: print(v)
        break
" 2>/dev/null)

            if [ -n "$val_ppl" ]; then
                # Sanity check on val_ppl (should be < 100 for successful runs)
                local ok=$(awk "BEGIN {print ($val_ppl < 100) ? 1 : 0}")
                if [ "$ok" -eq 1 ]; then
                    log "${run_id} completed successfully. val_ppl=${val_ppl}"
                    return 0
                else
                    log "${run_id} completed but val_ppl=${val_ppl} looks broken (>100)"
                    return 1
                fi
            else
                log "${run_id} process exited but no registry entry found — treating as FAILURE"
                return 1
            fi
        fi

        sleep 60
    done
}

# ---- Utility: run per-language eval on best checkpoint ----
run_per_lang_eval() {
    local run_id="$1"
    local ckpt_dir="/workspace/checkpoints/${run_id}"

    log "Locating best checkpoint for ${run_id}..."

    # Find best checkpoint (highest step number in the dir — assumes top-K retention keeps the best)
    local best_ckpt=$(ls -d ${ckpt_dir}/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)

    if [ -z "$best_ckpt" ]; then
        log "ERROR: no checkpoint found in ${ckpt_dir}"
        return 1
    fi

    log "Running per-language eval on ${best_ckpt}..."
    local output="${RESULTS_DIR}/P3_${run_id}_per_language_eval.json"

    python3 /workspace/scripts/eval_per_language_5tracks.py \
        --checkpoint "${best_ckpt}" \
        --output "${output}" \
        >> "$CHAIN_LOG" 2>&1

    if [ $? -eq 0 ] && [ -f "$output" ]; then
        log "Per-language eval complete: ${output}"
        # Extract pooled val_ppl for the notification
        local pooled=$(python3 -c "
import json
d = json.load(open('${output}'))
print(d.get('pooled', {}).get('perplexity', 'N/A'))
" 2>/dev/null)
        log "  pooled val_ppl: ${pooled}"
        return 0
    else
        log "ERROR: per-language eval failed for ${run_id}"
        return 1
    fi
}

# ---- Utility: prune intermediate checkpoints, keep only final ----
cleanup_intermediate_checkpoints() {
    local run_id="$1"
    local ckpt_dir="/workspace/checkpoints/${run_id}"
    if [ ! -d "$ckpt_dir" ]; then
        log "cleanup: no checkpoint dir for ${run_id} — skip"
        return 0
    fi
    local final_ckpt=$(ls -d ${ckpt_dir}/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)
    if [ -z "$final_ckpt" ]; then
        log "cleanup: no checkpoints in ${ckpt_dir} — skip"
        return 0
    fi
    log "cleanup: keeping ${final_ckpt}, removing intermediates for ${run_id}"
    local removed=0
    for ckpt in $(ls -d ${ckpt_dir}/checkpoint-* 2>/dev/null); do
        if [ "$ckpt" != "$final_ckpt" ]; then
            log "cleanup:   removing ${ckpt}"
            rm -rf "$ckpt"
            removed=$((removed + 1))
        fi
    done
    log "cleanup: removed ${removed} intermediate checkpoint(s) for ${run_id}"
    df -h /workspace | tail -1 | awk '{print "cleanup: disk now " $4 " free (" $5 " used)"}' | tee -a "$CHAIN_LOG"
    return 0
}

# ---- Utility: launch a new run ----
launch_run() {
    local launcher="$1"
    local run_id="$2"

    log "Launching ${run_id} via ${launcher}..."
    "$NOTIFY" "$run_id" "RUNNING" "chain step launch"

    # Fire and forget — the launcher itself blocks until torchrun exits
    "$launcher" > /workspace/logs/wrapper_${run_id}.log 2>&1 &
    local wrapper_pid=$!

    sleep 60

    # Verify torchrun started
    local workers=$(ps aux | grep train_orchestrator | grep -v grep | wc -l)
    if [ "$workers" -lt 4 ]; then
        log "ERROR: ${run_id} failed to start torchrun workers (found $workers)"
        return 1
    fi

    log "${run_id} torchrun workers active: $workers"
    return 0
}

# ---- Utility: process one run in the chain ----
process_run() {
    local run_id="$1"
    local launcher="$2"

    log "==================================================================="
    log "Chain step: ${run_id}"
    log "==================================================================="

    "$NOTIFY" "$run_id" "RUNNING" "chain step start"

    if [ -n "$launcher" ]; then
        # We need to launch it
        launch_run "$launcher" "$run_id" || {
            "$NOTIFY" "$run_id" "FAILED" "launch failed"
            "$NOTIFY" "CHAIN" "FAILED" "${run_id} launch failed"
            return 1
        }
    fi

    # Wait for it (whether we just launched or FFT-S2 was already running)
    wait_for_run "$run_id" || {
        "$NOTIFY" "$run_id" "FAILED" "training failed or produced bad val_ppl"
        "$NOTIFY" "CHAIN" "FAILED" "${run_id} training failed"
        return 1
    }

    # Run per-language eval
    run_per_lang_eval "$run_id" || {
        "$NOTIFY" "$run_id" "FAILED" "per-language eval failed"
        "$NOTIFY" "CHAIN" "FAILED" "${run_id} eval failed"
        return 1
    }

    # Cleanup intermediate checkpoints now that eval JSON is safely saved
    cleanup_intermediate_checkpoints "$run_id"
    "$NOTIFY" "$run_id" "DONE" "training + eval complete"
    return 0
}

# ==============================================================================
# MAIN CHAIN
# ==============================================================================

log "###################################################################"
log "# Gap 1 Chain started at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
log "# Sequence: FFT-S2 → CPT-S2 → FFT-S3 → CPT-S3"
log "###################################################################"

"$NOTIFY" "CHAIN" "RUNNING" "Gap 1 chain started"

# Step 1: FFT-S2 (already running) — no launcher, just wait
process_run "RUN-P3-FFT-S2" "" || exit 1

# Step 2: CPT-S2
# SKIPPED 2026-07-31 — pooled 22.29 captured in registry, checkpoint lost to retention race, per-language unrecoverable
# # SKIPPED 2026-07-31 — pooled 22.29 captured in registry (July 30 entry), per-language lost to retention race
# process_run "RUN-P3-CPT-S2" "/workspace/scripts/phase3_cpt_seed2.sh" || exit 1

# Step 3: FFT-S3
process_run "RUN-P3-FFT-S3" "/workspace/scripts/phase3_fft_seed3.sh" || exit 1

# Step 4: CPT-S3
process_run "RUN-P3-CPT-S3" "/workspace/scripts/phase3_cpt_seed3.sh" || exit 1

log "###################################################################"
log "# ALL GAP 1 RUNS COMPLETE at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
log "###################################################################"

"$NOTIFY" "CHAIN" "DONE" "All 4 Gap 1 seeds complete"
