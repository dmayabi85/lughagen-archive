#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────
# show_pilot_evidence.sh
#
# Runs inside the LughaGen Brev container (sudo docker exec -it ... bash)
# and produces the operational evidence cited in the proposal:
#   1. Checkpoint listings    — proves pilot artefacts exist
#   2. Tokenizer config       — proves SentencePiece BPE 32K choice
#   3. GPU memory during load — proves QLoRA fits at ~16 GB
#   4. Training-log excerpts  — proves pilot loss/perplexity figures
#   5. Distributed-training warnings & operational issues encountered
#
# Capture the output to a file:
#   bash show_pilot_evidence.sh 2>&1 | tee pilot_evidence_$(date +%F).log
# ─────────────────────────────────────────────────────────────────────

set -u
OUT_SEP="================================================================="

# ─── 1. Checkpoint listings ──────────────────────────────────────────
echo "$OUT_SEP"
echo "  [1] PILOT CHECKPOINT ARTEFACTS"
echo "$OUT_SEP"
for dir in \
    /workspace/checkpoints/afro-xlmr-swahili/final \
    /workspace/checkpoints/afro-xlmr-swahili-v2/final \
    /workspace/checkpoints/llama-3.1-8b-swahili/final
do
    echo
    echo "→ $dir"
    if [ -d "$dir" ]; then
        ls -lh "$dir" 2>/dev/null | head -20
    else
        echo "  (directory not found)"
    fi
done

# ─── 2. Tokenizer configuration ──────────────────────────────────────
echo
echo "$OUT_SEP"
echo "  [2] SAVED TOKENIZER CONFIG"
echo "$OUT_SEP"
for cfg in \
    /workspace/checkpoints/llama-3.1-8b-swahili/final/tokenizer_config.json \
    /workspace/data/processed/swahili_tokenizer.model
do
    echo
    echo "→ $cfg"
    if [ -f "$cfg" ]; then
        ls -lh "$cfg"
        if [[ "$cfg" == *.json ]]; then
            python3 -c "import json; d=json.load(open('$cfg')); \
                [print(f'  {k:<30} {v}') for k,v in list(d.items())[:10]]"
        fi
    else
        echo "  (not found)"
    fi
done

# ─── 3. Live GPU memory check ────────────────────────────────────────
echo
echo "$OUT_SEP"
echo "  [3] GPU STATUS (current)"
echo "$OUT_SEP"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free \
           --format=csv 2>/dev/null || echo "  nvidia-smi not available"

# ─── 4. Training log highlights ──────────────────────────────────────
echo
echo "$OUT_SEP"
echo "  [4] PILOT TRAINING LOG EXCERPTS"
echo "$OUT_SEP"
for log in \
    /workspace/logs/afro_xlmr_train.log \
    /workspace/logs/afro_xlmr_v2_train.log \
    /workspace/logs/llama_swahili_train.log
do
    echo
    echo "→ $log"
    if [ -f "$log" ]; then
        echo
        echo "  First 5 loss entries:"
        grep -m 5 "'loss':" "$log" 2>/dev/null | head -5
        echo
        echo "  Last 5 loss entries:"
        grep "'loss':" "$log" 2>/dev/null | tail -5
        echo
        echo "  Train runtime / throughput:"
        grep -E "train_runtime|train_samples_per_second|total_flos" "$log" 2>/dev/null | tail -5
    else
        echo "  (log not found)"
    fi
done

# ─── 5. Operational incidents documented in logs ────────────────────
echo
echo "$OUT_SEP"
echo "  [5] OPERATIONAL INCIDENTS RECORDED IN LOGS"
echo "$OUT_SEP"
for log in /workspace/logs/*.log; do
    [ -f "$log" ] || continue
    found=$(grep -E "OutOfMemoryError|CUDA out of memory|destroy_process_group|Shared connection.*closed|barrier" "$log" 2>/dev/null | head -3)
    if [ -n "$found" ]; then
        echo
        echo "→ $log"
        echo "$found"
    fi
done

echo
echo "$OUT_SEP"
echo "  Capture complete. Save this output to attach to the proposal."
echo "$OUT_SEP"
