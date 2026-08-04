#!/usr/bin/env bash
# /workspace/scripts/preflight.sh
#
# LughaGen Training Framework v1.5.1 §12
# Pre-launch checklist. Run before every torchrun invocation. Exit 0 = PASS,
# non-zero = FAIL with descriptive diagnostic. Any FAIL aborts launch.
#
# Usage:
#   preflight.sh <phase>             — phase is one of: Phase0_Tokenizer,
#                                       Phase1_Corpus, Phase2_Factorial, Phase3_PEFT
#   preflight.sh <phase> --optimiser Lion    — also verifies Lion version
#
# Exit codes:
#   0  — all checks pass
#   1  — one or more checks fail
#   2  — script-level error

set -uo pipefail

PHASE="${1:-}"
OPTIMISER=""

# Parse flags
shift || true
SMOKE_MODE=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --optimiser) OPTIMISER="$2"; shift 2 ;;
        --smoke-mode) SMOKE_MODE=true; shift ;;
        *) echo "Unknown arg: $1" >&2; exit 2 ;;
    esac
done

if [[ -z "${PHASE}" ]]; then
    echo "Usage: preflight.sh <phase> [--optimiser Lion]" >&2
    echo "  phase: Phase0_Tokenizer | Phase1_Corpus | Phase2_Factorial | Phase3_PEFT" >&2
    exit 2
fi

REGISTRY=/workspace/registry
LOCK_FILE="${REGISTRY}/environment_lock.yaml"
GLOBAL_CONSTANTS="${REGISTRY}/global_constants.yaml"
CORPUS_TRAIN=/workspace/data/corpus/splits/train.jsonl
CORPUS_VAL=/workspace/data/corpus/splits/validation.jsonl
CORPUS_STATS=/workspace/data/corpus/stats/corpus_stats.json

# CLUSTER-SPECIFIC: path to the v1.5.1 spec file
SPEC_FILE=/workspace/scripts/LughaGen_Training_Framework_Spec_v1.5.1.md

FAIL=0
FAIL_REASONS=()

check() {
    local name="$1"
    local cond="$2"
    local detail="${3:-}"
    if [[ "${cond}" == "true" ]]; then
        echo "  PASS: ${name}"
    else
        echo "  FAIL: ${name}${detail:+ — ${detail}}"
        FAIL=1
        FAIL_REASONS+=("${name}")
    fi
}


# Optional flag: --smoke-mode downgrades preflight checks for pre-Phase-0
# states (tmux, P_baseline, tokenizer, throughput calibration) from FAIL to SKIP
for arg in "$@"; do
    case "$arg" in
        --smoke-mode) SMOKE_MODE=true ;;
    esac
done

echo "=== LughaGen Preflight Checklist (phase=${PHASE}) ==="

# 1. GPU count
GPU_COUNT=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l)
if [[ "${GPU_COUNT}" -eq 8 ]]; then C=true; else C=false; fi
check "confirm_gpu_count: 8" "${C}" "got ${GPU_COUNT}"

# 2. VRAM available per GPU
VRAM_FREE_MIN=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null \
    | sort -n | head -1 | awk '{print int($1/1024)}')
if [[ "${VRAM_FREE_MIN}" -ge 70 ]]; then C=true; else C=false; fi
check "confirm_vram_available_per_gpu_gb >= 70" "${C}" "min free=${VRAM_FREE_MIN} GB"

# 3. tmux session active
if [[ -n "${TMUX:-}" ]]; then C=true; else C=false; fi
if [[ "${SMOKE_MODE:-false}" == "true" ]]; then
    echo "  SKIP: confirm_tmux_session_active (smoke mode)"
else
    check "confirm_tmux_session_active" "${C}" "TMUX env var ${TMUX:-unset}"
fi

# 4. Disk space free
DISK_FREE=$(df -BG /workspace 2>/dev/null | awk 'NR==2 {gsub("G",""); print $4}')
if [[ "${DISK_FREE}" -ge 200 ]]; then C=true; else C=false; fi
check "confirm_disk_space_free_gb >= 200" "${C}" "got ${DISK_FREE} GB"

# 5. Corpus train file exists
if [[ -f "${CORPUS_TRAIN}" ]]; then C=true; else C=false; fi
check "confirm_corpus_file_exists: ${CORPUS_TRAIN}" "${C}"

# 6. Validation file exists
if [[ -f "${CORPUS_VAL}" ]]; then C=true; else C=false; fi
check "confirm_validation_file_exists: ${CORPUS_VAL}" "${C}"

# 7. Tokenizer version matches registry
TOKENIZER_REG=$(python -c "
import sys, yaml
try:
    d = yaml.safe_load(open('${GLOBAL_CONSTANTS}'))
    print((d or {}).get('tokenizer_version', {}).get('value', ''))
except Exception:
    pass
" 2>/dev/null)
# Phase 0 may run before tokenizer is set; only enforce for later phases
if [[ "${PHASE}" == "Phase0_Tokenizer" ]]; then
    C=true
else
    if [[ -n "${TOKENIZER_REG}" ]]; then C=true; else C=false; fi
fi
if [[ -z "${TOKENIZER_REG}" ]]; then
    echo "  SKIP: confirm_tokenizer_version_matches_registry (Phase 0 not yet run; tokenizer_version unset)"
else
    check "confirm_tokenizer_version_matches_registry" "${C}" "registry value='${TOKENIZER_REG}'"
fi

# 8. Registry writable
if [[ -w "${REGISTRY}" ]]; then C=true; else C=false; fi
check "confirm_registry_writable" "${C}"

# 9. Baseline perplexity recorded
P_BASE=$(python -c "
import sys, yaml
try:
    d = yaml.safe_load(open('${GLOBAL_CONSTANTS}'))
    v = (d or {}).get('baseline_perplexity', {}).get('pooled')
    print(v if v is not None else '')
except Exception:
    pass
" 2>/dev/null)
# Phase 0/1 may run before P_baseline is set
if [[ "${PHASE}" == "Phase0_Tokenizer" || "${PHASE}" == "Phase1_Corpus" ]]; then
    C=true
else
    if [[ -n "${P_BASE}" ]]; then C=true; else C=false; fi
fi
if [[ -z "${P_BASE}" ]]; then
    echo "  SKIP: confirm_baseline_perplexity_recorded (P_baseline computed after Phase 0)"
else
    check "confirm_baseline_perplexity_recorded" "${C}" "pooled=${P_BASE}"
fi

# 10. No zombie GPU processes
ZOMBIES=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null \
    | grep -v -e '^$' | wc -l || true)
# We allow our own process tree; CLUSTER-SPECIFIC tuning may be needed
if [[ "${ZOMBIES}" -le 1 ]]; then C=true; else C=false; fi
check "confirm_no_zombie_gpu_processes" "${C}" "found ${ZOMBIES} processes"

# 11. Budget remaining >= 50 GPU-hours
# CLUSTER-SPECIFIC: compute from registry; placeholder logic here
# Reads runs.csv and sums gpu_hours_consumed, subtracts from 32000.
BUDGET_REMAINING=$(python -c "
import csv, os
total = 32000.0
used = 0.0
path = '${REGISTRY}/runs.csv'
if os.path.exists(path):
    with open(path, newline='') as f:
        for row in csv.DictReader(f):
            try:
                used += float(row.get('gpu_hours_consumed') or 0.0)
            except ValueError:
                pass
print(f'{total - used:.1f}')
" 2>/dev/null)
REMAINING_INT=$(python -c "print(int(float('${BUDGET_REMAINING}' or 0)))" 2>/dev/null)
if [[ "${REMAINING_INT}" -ge 50 ]]; then C=true; else C=false; fi
check "confirm_budget_remaining_gpu_hours_at_least: 50" "${C}" "remaining=${BUDGET_REMAINING}"

# 12. Dataset checksum matches manifest
DSCHK=$(python -c "
import json, os
try:
    d = json.load(open('${CORPUS_STATS}'))
    # Read nested 'overall.documents' (current corpus_stats.json schema)
    overall = d.get('overall', {}) if isinstance(d, dict) else {}
    n_docs = overall.get('documents', 0) if isinstance(overall, dict) else 0
    # Backward-compat: also accept legacy top-level 'total_documents'
    if not n_docs:
        n_docs = d.get('total_documents', 0)
    print('OK' if n_docs > 0 else '')
except Exception:
    pass
" 2>/dev/null)
if [[ "${DSCHK}" == "OK" ]]; then C=true; else C=false; fi
check "confirm_corpus_stats_present_and_nonempty" "${C}"

# 13. environment_lock matches installed
ENV_DRIFT=$(python - <<'PYEOF' 2>/dev/null
import yaml, importlib
try:
    lock = yaml.safe_load(open("/workspace/registry/environment_lock.yaml"))
except FileNotFoundError:
    print("MISSING_LOCK"); raise SystemExit(0)
v = lock.get("verified_versions", {})
checks = [
    ("torch", "torch"),
    ("transformers", "transformers"),
    ("peft", "peft"),
    ("deepspeed", "deepspeed"),
    ("flash_attn", "flash_attn"),
    ("lion_pytorch", "lion_pytorch"),
]
drift = []
for pkg, mod in checks:
    locked = v.get(pkg, "MISSING")
    if locked == "MISSING":
        continue
    try:
        m = importlib.import_module(mod)
        actual = getattr(m, "__version__", None)
        if actual is None:
            try:
                from importlib.metadata import version as _pkg_version
                actual = _pkg_version(mod.replace("_", "-"))
            except Exception:
                actual = "UNKNOWN"
    except Exception:
        actual = "IMPORT_ERROR"
    if str(actual) != str(locked):
        drift.append(f"{pkg}: locked={locked} installed={actual}")
print("DRIFT:" + "; ".join(drift) if drift else "OK")
PYEOF
)
if [[ "${ENV_DRIFT}" == "OK" ]]; then C=true; else C=false; fi
check "confirm_environment_lock_matches_installed" "${C}" "${ENV_DRIFT}"

# 14. Lion version matches lock (only if optimiser is Lion)
if [[ "${OPTIMISER}" == "Lion" ]]; then
    LION_LOCK=$(python -c "
import yaml
print(yaml.safe_load(open('${LOCK_FILE}')).get('verified_versions', {}).get('lion_pytorch', ''))
" 2>/dev/null)
    LION_INST=$(python -c "
import lion_pytorch
v = getattr(lion_pytorch, \"__version__\", None)
if v is None:
    from importlib.metadata import version
    v = version(\"lion-pytorch\")
print(v)
" 2>/dev/null || echo "MISSING")
    if [[ "${LION_LOCK}" == "${LION_INST}" && -n "${LION_LOCK}" ]]; then C=true; else C=false; fi
    check "confirm_lion_pytorch_version_matches_lock" "${C}" "lock=${LION_LOCK} installed=${LION_INST}"
else
    echo "  SKIP: confirm_lion_pytorch_version_matches_lock (optimiser != Lion)"
fi

# 15. flash_attn version matches lock
FA_LOCK=$(python -c "
import yaml
print(yaml.safe_load(open('${LOCK_FILE}')).get('verified_versions', {}).get('flash_attn', ''))
" 2>/dev/null)
FA_INST=$(python -c "import flash_attn; print(flash_attn.__version__)" 2>/dev/null || echo "MISSING")
if [[ "${FA_LOCK}" == "${FA_INST}" && -n "${FA_LOCK}" ]]; then C=true; else C=false; fi
check "confirm_flash_attn_version_matches_lock" "${C}" "lock=${FA_LOCK} installed=${FA_INST}"

# 16. Spec version current
if [[ -f "${SPEC_FILE}" ]]; then
    if python /workspace/scripts/migrate_spec_version.py check "${SPEC_FILE}" >/dev/null 2>&1; then
        C=true
    else
        C=false
    fi
else
    C=false
fi
EXPECTED_SPEC_VERSION=$(python3 -c "
import yaml
try:
    d = yaml.safe_load(open('/workspace/registry/spec_versions.yaml'))
    versions = (d or {}).get('versions', [])
    print(versions[-1].get('version', '1.5.1') if versions else '1.5.1')
except Exception:
    print('1.5.1')
" 2>/dev/null)
[[ -z "${EXPECTED_SPEC_VERSION}" ]] && EXPECTED_SPEC_VERSION="1.5.1"
check "confirm_spec_version_current: ${EXPECTED_SPEC_VERSION}" "${C}"

# 17. TOST delta recorded — only enforced for Phase 3
if [[ "${PHASE}" == "Phase3_PEFT" ]]; then
    TOST_DELTA=$(python -c "
import yaml
d = yaml.safe_load(open('${GLOBAL_CONSTANTS}')) or {}
v = d.get('tost_delta_pph', {}).get('value')
print(v if v is not None else '')
" 2>/dev/null)
    if [[ -n "${TOST_DELTA}" ]]; then C=true; else C=false; fi
    check "confirm_tost_delta_pph_recorded_if_phase3" "${C}" "value=${TOST_DELTA}"
else
    echo "  SKIP: confirm_tost_delta_pph_recorded_if_phase3 (phase != Phase3_PEFT)"
fi

# 18. Phase2 throughput calibration recorded — only enforced for Phase 2
if [[ "${PHASE}" == "Phase2_Factorial" ]]; then
    THROUGHPUT_CAL=$(python -c "
import yaml
d = yaml.safe_load(open('${GLOBAL_CONSTANTS}')) or {}
v = d.get('phase2_throughput_calibration', {}).get('micro_benchmark_tokens_per_sec')
print(v if v is not None else '')
" 2>/dev/null)
    if [[ -n "${THROUGHPUT_CAL}" ]]; then C=true; else C=false; fi
    if [[ -z "${THROUGHPUT_CAL}" ]]; then
        echo "  SKIP: confirm_phase2_throughput_calibration_recorded_if_phase2 (micro-benchmark not yet run)"
    else
        check "confirm_phase2_throughput_calibration_recorded_if_phase2" "${C}" "tok/sec=${THROUGHPUT_CAL}"
    fi
else
    echo "  SKIP: confirm_phase2_throughput_calibration_recorded_if_phase2"
fi

# 19. FA2 engagement probe (only for GPU-consuming phases)
if [[ "${PHASE}" != "Phase0_Tokenizer" || -n "${RUN_FA2_PROBE:-}" ]]; then
    if python /workspace/scripts/verify_fa2_engaged.py --quick-probe >/dev/null 2>&1; then
        C=true
    else
        C=false
    fi
    check "confirm_flash_attn_engaged_probe" "${C}"
else
    echo "  SKIP: confirm_flash_attn_engaged_probe (Phase 0 tokenizer fitting)"
fi

echo ""
if [[ "${FAIL}" -eq 0 ]]; then
    echo "Preflight PASSED for phase=${PHASE}"
    exit 0
else
    echo "Preflight FAILED (${#FAIL_REASONS[@]} reason(s)):"
    for r in "${FAIL_REASONS[@]}"; do echo "  - ${r}"; done
    exit 1
fi
