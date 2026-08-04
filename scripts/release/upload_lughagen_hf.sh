#!/bin/bash
# ==============================================================================
# Upload LughaGen models to HuggingFace Hub.
#
# Usage:
#   1. Set HF_USERNAME below (or set HF_USERNAME env var)
#   2. huggingface-cli login  (paste your token from https://huggingface.co/settings/tokens)
#   3. bash /workspace/scripts/upload_lughagen_hf.sh
#
# Uploads (skips if repo not yet ready, e.g., Gemma models pre-training):
#   - LughaGen-Llama-8B-FFT-S137
#   - LughaGen-Llama-8B-CPT-S137
#   - LughaGen-Llama-8B-QLoRA-S137
#   - LughaGen-Gemma2-27B-CPT-S137 (if Phase 3.5 CPT complete)
#   - LughaGen-Gemma2-27B-FFT-S137 (if Phase 3.5 FFT complete)
#   - LughaGen-strategy-c-tokenizer
# ==============================================================================
set -e

HF_USERNAME="${HF_USERNAME:-YOUR_USERNAME}"

if [ "$HF_USERNAME" = "YOUR_USERNAME" ]; then
    echo "ERROR: set HF_USERNAME before running." >&2
    echo "  Either edit the script or run: HF_USERNAME=derekm bash $0" >&2
    exit 1
fi

# Test HF login
if ! /workspace/.venv/bin/huggingface-cli whoami > /dev/null 2>&1; then
    echo "ERROR: not logged in to HuggingFace Hub." >&2
    echo "  Run: /workspace/.venv/bin/huggingface-cli login" >&2
    exit 1
fi

WHOAMI=$(/workspace/.venv/bin/huggingface-cli whoami | head -1)
echo "Logged in as: $WHOAMI"
echo "Uploading to org/user: $HF_USERNAME"
echo ""

# Model definitions: slug|checkpoint_path
MODELS=(
    "LughaGen-Llama-8B-FFT-S137|/workspace/checkpoints/RUN-P3-FFT/checkpoint-5400"
    "LughaGen-Llama-8B-CPT-S137|/workspace/checkpoints/RUN-P3-CPT/checkpoint-5600"
    "LughaGen-Llama-8B-QLoRA-S137|/workspace/checkpoints/RUN-P3-CLEAN-QLORA/merged_bf16"
    "LughaGen-Gemma2-27B-CPT-S137|/workspace/checkpoints/RUN-P3.5-GEMMA2-CPT-S137/checkpoint-6000"
    "LughaGen-Gemma2-27B-FFT-S137|/workspace/checkpoints/RUN-P3.5-GEMMA2-FFT-S137/checkpoint-6000"
    "LughaGen-strategy-c-tokenizer|/workspace/data/processed/tokenizers/strategy_c"
)

RELEASES_DIR="/workspace/gemma2_27b_prep/releases"
UPLOADED_URLS=()
SKIPPED=()
FAILED=()

for entry in "${MODELS[@]}"; do
    IFS='|' read -r SLUG CHECKPOINT_PATH <<< "$entry"
    REPO="${HF_USERNAME}/${SLUG}"
    URL="https://huggingface.co/${REPO}"

    echo "=========================================="
    echo "Model: $SLUG"
    echo "  Checkpoint: $CHECKPOINT_PATH"
    echo "  Target: $URL"

    # Skip if checkpoint doesn't exist (e.g., Gemma models pre-training)
    if [ ! -d "$CHECKPOINT_PATH" ]; then
        echo "  SKIP: checkpoint not yet available"
        SKIPPED+=("$SLUG (checkpoint missing: $CHECKPOINT_PATH)")
        continue
    fi

    # Check for lingering DeepSpeed optimizer state — refuse to upload if present
    if compgen -G "$CHECKPOINT_PATH/global_step*" > /dev/null; then
        echo "  ERROR: DeepSpeed optimizer state (global_step*) still in checkpoint" >&2
        echo "  Strip it first: rm -rf $CHECKPOINT_PATH/global_step*" >&2
        FAILED+=("$SLUG (optimizer state present)")
        continue
    fi

    # Verify model card exists and has YOUR_USERNAME replaced
    CARD_SRC="$RELEASES_DIR/$SLUG/README.md"
    if [ ! -f "$CARD_SRC" ]; then
        echo "  ERROR: model card missing at $CARD_SRC" >&2
        echo "  Run: /workspace/.venv/bin/python /workspace/scripts/generate_model_cards.py" >&2
        FAILED+=("$SLUG (no model card)")
        continue
    fi

    # Substitute username in model card and copy to checkpoint dir for upload
    sed "s|YOUR_USERNAME|${HF_USERNAME}|g" "$CARD_SRC" > "$CHECKPOINT_PATH/README.md"
    echo "  Model card ready ($(wc -l < $CHECKPOINT_PATH/README.md) lines)"

    # Compute size for user info
    SIZE=$(du -sh "$CHECKPOINT_PATH" | awk '{print $1}')
    echo "  Size: $SIZE"

    # Upload
    echo "  Uploading..."
    if /workspace/.venv/bin/huggingface-cli upload \
            "$REPO" \
            "$CHECKPOINT_PATH" \
            . \
            --repo-type model \
            --commit-message "Initial LughaGen release: $SLUG"; then
        echo "  SUCCESS: $URL"
        UPLOADED_URLS+=("$URL")
    else
        echo "  FAILED: upload error" >&2
        FAILED+=("$SLUG (upload error)")
    fi

    echo ""
done

# Summary
echo "=========================================="
echo "UPLOAD SUMMARY"
echo "=========================================="
echo ""
echo "Uploaded (${#UPLOADED_URLS[@]}):"
for url in "${UPLOADED_URLS[@]}"; do
    echo "  $url"
done
echo ""
if [ "${#SKIPPED[@]}" -gt 0 ]; then
    echo "Skipped (${#SKIPPED[@]}):"
    for s in "${SKIPPED[@]}"; do
        echo "  $s"
    done
    echo ""
fi
if [ "${#FAILED[@]}" -gt 0 ]; then
    echo "Failed (${#FAILED[@]}):"
    for f in "${FAILED[@]}"; do
        echo "  $f"
    done
    echo ""
    exit 1
fi

# Write URL list for thesis Chapter 8 (Reproducibility)
URLS_OUT="/workspace/gemma2_27b_prep/releases/uploaded_urls.txt"
{
    echo "# LughaGen model release URLs"
    echo "# Generated: $(date -u '+%Y-%m-%d %H:%M UTC')"
    echo ""
    for url in "${UPLOADED_URLS[@]}"; do
        echo "$url"
    done
} > "$URLS_OUT"
echo "URLs written to $URLS_OUT for thesis inclusion."
