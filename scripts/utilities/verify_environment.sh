#!/usr/bin/env bash
# /workspace/scripts/verify_environment.sh
#
# LughaGen Training Framework v1.5.1 §9.2.1
# Captures cluster versions, GPU inventory, and disk state to
# /workspace/registry/environment_lock.yaml.

# IMPORTANT: do NOT set 'pipefail' at the script level.
# Several diagnostic pipelines use 'head -1' which closes the read end early
# and causes upstream commands (nvidia-smi) to die with SIGPIPE (exit 141).
# This is harmless — we just want the first line of multi-GPU output —
# but pipefail would propagate it and kill the script.
set -u

REGISTRY_DIR=/workspace/registry
LOCK_FILE="${REGISTRY_DIR}/environment_lock.yaml"
HISTORY_DIR="${REGISTRY_DIR}/environment_lock_history"

mkdir -p "${REGISTRY_DIR}" "${HISTORY_DIR}"

# Capture a Python package version. Falls back to importlib.metadata for
# packages (like lion-pytorch) that don't expose __version__ as an attribute.
capture() {
    local pkg="$1"
    python3 -c "
import sys
try:
    mod = __import__('${pkg}')
    v = getattr(mod, '__version__', None)
    if v is None:
        try:
            from importlib.metadata import version
        except ImportError:
            from importlib_metadata import version
        pkg_name = '${pkg}'.replace('_', '-')
        v = version(pkg_name)
    print(v)
except Exception:
    print('MISSING')
    sys.exit(1)
" 2>/dev/null
    local rc=$?
    if [[ $rc -ne 0 ]]; then
        echo "MISSING"
    fi
}

# Read the first line of a pipeline, suppressing SIGPIPE failures from upstream
first_line() {
    # Use awk/sed to read the first non-empty line; do NOT use 'head -1' because
    # head exits as soon as it reads its line, which kills the upstream pipe.
    awk 'NF { print; exit }' 2>/dev/null
}

echo "Capturing environment state for LughaGen Training Framework v1.5.1..."

if [[ -f "${LOCK_FILE}" ]]; then
    archive_name="${HISTORY_DIR}/environment_lock_$(date -u +%Y%m%dT%H%M%SZ).yaml"
    cp "${LOCK_FILE}" "${archive_name}"
    echo "Archived previous lock to ${archive_name}"
fi

# Python and packages
PY_VERSION=$(python3 -c 'import sys; print(sys.version.split()[0])' 2>/dev/null | tr -d '\n')
[[ -z "${PY_VERSION}" ]] && PY_VERSION="MISSING"

TORCH_VERSION=$(capture torch | tr -d '\n')
CUDA_TORCH=$(python3 -c 'import torch; print(torch.version.cuda)' 2>/dev/null | tr -d '\n')
[[ -z "${CUDA_TORCH}" ]] && CUDA_TORCH="MISSING"

CUDA_DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | first_line | tr -d ' ,\n')
[[ -z "${CUDA_DRIVER}" ]] && CUDA_DRIVER="MISSING"

TRANSFORMERS_V=$(capture transformers | tr -d '\n')
PEFT_V=$(capture peft | tr -d '\n')
DATASETS_V=$(capture datasets | tr -d '\n')
DEEPSPEED_V=$(capture deepspeed | tr -d '\n')
BNB_V=$(capture bitsandbytes | tr -d '\n')
FA_V=$(capture flash_attn | tr -d '\n')
LION_V=$(capture lion_pytorch | tr -d '\n')
ACC_V=$(capture accelerate | tr -d '\n')
SPM_V=$(capture sentencepiece | tr -d '\n')
TOK_V=$(capture tokenizers | tr -d '\n')

# GPU inventory
GPU_COUNT=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l 2>/dev/null | tr -d ' \n')
[[ -z "${GPU_COUNT}" ]] && GPU_COUNT=0

GPU_TYPE=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | first_line | tr -d '\n')
[[ -z "${GPU_TYPE}" ]] && GPU_TYPE="UNKNOWN"

VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | first_line | tr -d ' \n')
if [[ -n "${VRAM_MB}" && "${VRAM_MB}" =~ ^[0-9]+$ ]]; then
    VRAM_PER_GPU=$(( VRAM_MB / 1024 ))
else
    VRAM_PER_GPU=0
fi

# Disk space
WORKSPACE_FREE=$(df -BG /workspace 2>/dev/null | awk 'NR==2 {gsub("G",""); print $4}' | tr -d ' \n')
[[ -z "${WORKSPACE_FREE}" ]] && WORKSPACE_FREE=0

HOSTNAME=$(hostname 2>/dev/null | tr -d '\n')
[[ -z "${HOSTNAME}" ]] && HOSTNAME="unknown"

TIMESTAMP=$(date -u +%Y-%m-%dT%H:%M:%SZ)

# Write the lock file
cat > "${LOCK_FILE}" <<EOF
# Captured by verify_environment.sh on ${TIMESTAMP}
# Host: ${HOSTNAME}
# Spec version: 1.5.1
verified_versions:
  python: "${PY_VERSION}"
  torch: "${TORCH_VERSION}"
  cuda_torch: "${CUDA_TORCH}"
  cuda_driver: "${CUDA_DRIVER}"
  transformers: "${TRANSFORMERS_V}"
  peft: "${PEFT_V}"
  datasets: "${DATASETS_V}"
  deepspeed: "${DEEPSPEED_V}"
  bitsandbytes: "${BNB_V}"
  flash_attn: "${FA_V}"
  lion_pytorch: "${LION_V}"
  accelerate: "${ACC_V}"
  sentencepiece: "${SPM_V}"
  tokenizers: "${TOK_V}"
gpu_inventory:
  gpu_count: ${GPU_COUNT}
  gpu_type: "${GPU_TYPE}"
  total_vram_per_gpu_gb: ${VRAM_PER_GPU}
disk:
  workspace_free_gb: ${WORKSPACE_FREE}
captured_at: "${TIMESTAMP}"
hostname: "${HOSTNAME}"
EOF

echo "Wrote environment lock to ${LOCK_FILE}"
echo ""
cat "${LOCK_FILE}"
echo ""

# Acceptance criteria
echo "--- Acceptance criteria check ---"
FAIL=0
check() {
    local name="$1"
    local cond="$2"
    local detail="${3:-}"
    if [[ "${cond}" == "true" ]]; then
        echo "  PASS: ${name}"
    else
        echo "  FAIL: ${name}${detail:+ — ${detail}}"
        FAIL=1
    fi
}

# 1. No package reports MISSING
NO_MISSING=true
for v in "${TORCH_VERSION}" "${TRANSFORMERS_V}" "${PEFT_V}" "${DATASETS_V}" \
         "${DEEPSPEED_V}" "${BNB_V}" "${FA_V}" "${LION_V}" "${ACC_V}" \
         "${SPM_V}" "${TOK_V}"; do
    if [[ "${v}" == "MISSING" ]]; then NO_MISSING=false; fi
done
check "All required packages installed (no MISSING)" "${NO_MISSING}"

# 2. flash_attn >= 2.0
if [[ "${FA_V}" != "MISSING" ]]; then
    FA_MAJOR=$(echo "${FA_V}" | cut -d. -f1)
    if [[ "${FA_MAJOR}" =~ ^[0-9]+$ ]] && (( FA_MAJOR >= 2 )); then
        FA_OK=true
    else
        FA_OK=false
    fi
else
    FA_OK=false
fi
check "flash_attn >= 2.0 (got ${FA_V})" "${FA_OK}"

# 3. lion_pytorch present
if [[ "${LION_V}" != "MISSING" && -n "${LION_V}" ]]; then LION_OK=true; else LION_OK=false; fi
check "lion_pytorch installed (got ${LION_V})" "${LION_OK}"

# 4. CUDA both ways
if [[ "${CUDA_TORCH}" != "MISSING" && "${CUDA_DRIVER}" != "MISSING" ]]; then
    CUDA_OK=true
else
    CUDA_OK=false
fi
check "torch CUDA + nvidia driver both detected (torch=${CUDA_TORCH}, driver=${CUDA_DRIVER})" "${CUDA_OK}"

# 5. GPU count
if [[ "${GPU_COUNT}" =~ ^[0-9]+$ ]] && (( GPU_COUNT == 8 )); then
    GPU_COUNT_OK=true
else
    GPU_COUNT_OK=false
fi
check "GPU count = 8 (got ${GPU_COUNT})" "${GPU_COUNT_OK}"

# 6. VRAM
if [[ "${VRAM_PER_GPU}" =~ ^[0-9]+$ ]] && (( VRAM_PER_GPU >= 70 )); then
    VRAM_OK=true
else
    VRAM_OK=false
fi
check "VRAM per GPU >= 70 GB (got ${VRAM_PER_GPU})" "${VRAM_OK}"

# 7. Disk
if [[ "${WORKSPACE_FREE}" =~ ^[0-9]+$ ]] && (( WORKSPACE_FREE >= 500 )); then
    DISK_OK=true
else
    DISK_OK=false
fi
check "/workspace free >= 500 GB (got ${WORKSPACE_FREE})" "${DISK_OK}"

echo ""
if [[ "${FAIL}" -eq 0 ]]; then
    echo "All acceptance criteria PASSED. Environment locked."
    exit 0
else
    echo "One or more acceptance criteria FAILED. Resolve before Phase 0."
    exit 1
fi
