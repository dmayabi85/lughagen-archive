#!/usr/bin/env bash
#
# Gemma 2 27B base model download
#
# Downloads google/gemma-2-27b to /workspace/models/gemma-2-27b/
# Uses low I/O priority so it does not compete with running training.
# Safe to run while LughaGen chain is training.
#
# Prerequisites:
#   - HF token exists at ~/.cache/huggingface/token (verified 2026-07-28)
#   - Gemma license accepted on huggingface.co/google/gemma-2-27b
#   - At least 100 GB free on /workspace
#
# Usage:
#   bash /workspace/gemma2_27b_prep/scripts/download_gemma2_27b.sh
#
# Runs in foreground. To run in background:
#   nohup bash /workspace/gemma2_27b_prep/scripts/download_gemma2_27b.sh > /dev/null 2>&1 &
#

set -euo pipefail

# Configuration
MODEL_ID="google/gemma-2-27b"
TARGET_DIR="/workspace/models/gemma-2-27b"
LOG_DIR="/workspace/gemma2_27b_prep/logs"
TIMESTAMP=$(date -u +%Y%m%dT%H%M%SZ)
LOG_FILE="${LOG_DIR}/download_${TIMESTAMP}.log"
MIN_FREE_GB=100

# Ensure log directory exists
mkdir -p "${LOG_DIR}"

# All output goes to log file and stdout
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "==================================================================="
echo "Gemma 2 27B base model download"
echo "Started: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Model: ${MODEL_ID}"
echo "Target: ${TARGET_DIR}"
echo "Log: ${LOG_FILE}"
echo "==================================================================="

# ============================================================
# Preflight checks
# ============================================================

echo ""
echo "[preflight] Checking disk space..."
FREE_KB=$(df /workspace | awk 'NR==2 {print $4}')
FREE_GB=$((FREE_KB / 1024 / 1024))
echo "[preflight] Free space on /workspace: ${FREE_GB} GB"

if [ "${FREE_GB}" -lt "${MIN_FREE_GB}" ]; then
  echo "[preflight] ERROR: Need at least ${MIN_FREE_GB} GB free, have ${FREE_GB} GB"
  echo "[preflight] Aborting download"
  exit 1
fi
echo "[preflight] Disk check: OK"

echo ""
echo "[preflight] Checking HuggingFace token..."
HF_TOKEN_FILE="${HOME}/.cache/huggingface/token"
if [ ! -f "${HF_TOKEN_FILE}" ]; then
  echo "[preflight] ERROR: No HF token at ${HF_TOKEN_FILE}"
  echo "[preflight] Run: huggingface-cli login"
  exit 1
fi
echo "[preflight] HF token: OK"

echo ""
echo "[preflight] Checking Python environment..."
if ! python3 -c "import huggingface_hub" 2>/dev/null; then
  echo "[preflight] ERROR: huggingface_hub not installed in current Python"
  echo "[preflight] Current Python: $(which python3)"
  echo "[preflight] Activate venv: source /workspace/.venv/bin/activate"
  exit 1
fi
HF_HUB_VERSION=$(python3 -c "import huggingface_hub; print(huggingface_hub.__version__)")
echo "[preflight] huggingface_hub version: ${HF_HUB_VERSION}"

echo ""
echo "[preflight] Checking chain status (informational only)..."
if [ -f /workspace/automation/flags/CHAIN.status ]; then
  CHAIN_STATUS=$(grep '^status:' /workspace/automation/flags/CHAIN.status | awk '{print $2}')
  echo "[preflight] LughaGen chain status: ${CHAIN_STATUS}"
  echo "[preflight] Download will run in parallel with training (low I/O priority)"
else
  echo "[preflight] No chain status file (chain not running or not started)"
fi

echo ""
echo "[preflight] All checks passed"
echo "==================================================================="

# ============================================================
# Download
# ============================================================

mkdir -p "${TARGET_DIR}"

echo ""
echo "[download] Starting download at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "[download] This typically takes 1-3 hours depending on network"
echo "[download] Using low CPU and I/O priority to avoid competing with training"
echo ""

# nice: low CPU priority (19 = lowest)
# ionice: idle I/O class (3 = only run when nothing else needs disk)
nice -n 19 ionice -c 3 python3 << PYEOF
import os
import sys
from huggingface_hub import snapshot_download

target_dir = "${TARGET_DIR}"
model_id = "${MODEL_ID}"

print(f"[download] Fetching {model_id} to {target_dir}")
print(f"[download] Excluding: *.gguf (quantized formats not needed)")
print()

try:
    path = snapshot_download(
        repo_id=model_id,
        local_dir=target_dir,
        local_dir_use_symlinks=False,
        resume_download=True,
        max_workers=4,  # moderate parallelism, not aggressive
        ignore_patterns=["*.gguf", "*.bin"],  # only need safetensors
    )
    print(f"[download] SUCCESS: model at {path}")
except Exception as e:
    print(f"[download] ERROR: {e}", file=sys.stderr)
    sys.exit(1)
PYEOF

DOWNLOAD_EXIT=$?

if [ ${DOWNLOAD_EXIT} -ne 0 ]; then
  echo ""
  echo "[download] FAILED with exit code ${DOWNLOAD_EXIT}"
  echo "[download] Check log: ${LOG_FILE}"
  echo "[download] To resume: re-run this script (resume_download=True)"
  exit ${DOWNLOAD_EXIT}
fi

# ============================================================
# Verification
# ============================================================

echo ""
echo "==================================================================="
echo "[verify] Checking downloaded files..."

REQUIRED_FILES=("config.json" "tokenizer.json" "tokenizer_config.json")
MISSING=0

for f in "${REQUIRED_FILES[@]}"; do
  if [ -f "${TARGET_DIR}/${f}" ]; then
    echo "[verify] OK: ${f}"
  else
    echo "[verify] MISSING: ${f}"
    MISSING=$((MISSING + 1))
  fi
done

# Check for safetensors files (multiple sharded files expected for 27B)
SAFETENSOR_COUNT=$(ls "${TARGET_DIR}"/*.safetensors 2>/dev/null | wc -l)
echo "[verify] Safetensor shards found: ${SAFETENSOR_COUNT}"

if [ "${SAFETENSOR_COUNT}" -eq 0 ]; then
  echo "[verify] ERROR: No safetensors files found"
  MISSING=$((MISSING + 1))
fi

# Total size
TOTAL_SIZE=$(du -sh "${TARGET_DIR}" | cut -f1)
echo "[verify] Total size: ${TOTAL_SIZE}"

if [ ${MISSING} -gt 0 ]; then
  echo "[verify] FAILED: ${MISSING} missing files"
  exit 1
fi

echo "[verify] All required files present"
echo ""
echo "==================================================================="
echo "Download complete at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Model location: ${TARGET_DIR}"
echo "Log: ${LOG_FILE}"
echo "==================================================================="

# Final disk state for reference
echo ""
df -h /workspace | tail -1
