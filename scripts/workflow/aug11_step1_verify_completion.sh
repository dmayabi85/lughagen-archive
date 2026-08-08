#!/bin/bash
# Aug 11 Step 1: Verify Phase 3.5 FFT completed cleanly
set -e

RUN_NAME="RUN-P3.5-GEMMA2-FFT-S137"
CKPT_DIR="/workspace/checkpoints/$RUN_NAME/checkpoint-6000"
LATEST_LOG=$(ls -t /workspace/logs/phase3_5_${RUN_NAME}_*.log 2>/dev/null | head -1)

echo "=== Aug 11 Step 1: FFT Completion Verification ==="
echo ""

echo "[1] Process check"
if pgrep -f "torchrun.*RUN-P3.5-GEMMA2-FFT" > /dev/null; then
    echo "  ⚠ Training process still running — do not proceed"
    exit 1
else
    echo "  ✓ No training processes running"
fi

echo ""
echo "[2] Checkpoint completeness"
if [ -d "$CKPT_DIR" ]; then
    echo "  ✓ Checkpoint dir exists: $CKPT_DIR"
    ls -la $CKPT_DIR/*.safetensors 2>/dev/null | head -3
    if [ -f "$CKPT_DIR/config.json" ]; then
        echo "  ✓ config.json present"
    fi
    if [ -f "$CKPT_DIR/tokenizer.json" ]; then
        echo "  ✓ tokenizer.json present"
    fi
else
    echo "  ✗ Checkpoint dir missing: $CKPT_DIR"
    exit 1
fi

echo ""
echo "[3] Training log analysis"
if [ -z "$LATEST_LOG" ]; then
    echo "  ✗ Log file not found"
    exit 1
fi
echo "  Log: $LATEST_LOG"

FINAL_STEP=$(grep -oE "'step': [0-9]+" "$LATEST_LOG" | tail -1 | grep -oE "[0-9]+$")
echo "  Final step reached: $FINAL_STEP / 6000"
if [ "$FINAL_STEP" -lt 5900 ]; then
    echo "  ⚠ Training may not have completed fully"
fi

FINAL_TRAIN=$(grep -oE "'loss': '?[0-9.]+" "$LATEST_LOG" | tail -1 | grep -oE "[0-9.]+$")
FINAL_EVAL=$(grep -oE "'eval_loss': '?[0-9.]+" "$LATEST_LOG" | tail -1 | grep -oE "[0-9.]+$")
echo "  Final train loss: $FINAL_TRAIN"
echo "  Final eval loss:  $FINAL_EVAL"

if [ -n "$FINAL_TRAIN" ] && [ -n "$FINAL_EVAL" ]; then
    GAP=$(echo "scale=3; $FINAL_EVAL - $FINAL_TRAIN" | bc)
    echo "  Train-eval gap:   $GAP nats"
    if (( $(echo "$GAP > 1.5" | bc -l) )); then
        echo "  ⚠ Gap > 1.5 nats — may need best-checkpoint selection"
    elif (( $(echo "$GAP > 1.0" | bc -l) )); then
        echo "  ○ Gap > 1.0 nats — monitor, likely acceptable"
    else
        echo "  ✓ Gap < 1.0 nats — no overfitting concern"
    fi
fi

echo ""
echo "[4] NaN check"
NAN_COUNT=$(grep -icE "loss.*nan|loss.*inf" "$LATEST_LOG" 2>/dev/null)
if [ "$NAN_COUNT" -gt 0 ]; then
    echo "  ✗ $NAN_COUNT NaN/Inf events — investigate"
else
    echo "  ✓ No NaN/Inf events"
fi

echo ""
echo "[5] Amendment 12 checkpoint safety fires"
PRUNE_COUNT=$(grep -c "ckpt-preprune" "$LATEST_LOG" 2>/dev/null)
echo "  Fires: $PRUNE_COUNT (expected ~30)"

echo ""
echo "=== Verification complete ==="
