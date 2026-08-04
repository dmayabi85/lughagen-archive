#!/bin/bash
# ==============================================================================
# Stage LughaGen models in /workspace/releases/lughagen for external download.
#
# Uses hard links for large safetensors (saves disk), copies for small configs
# and tokenizer files (portable + resilient).
#
# Idempotent — safe to re-run when new models complete training.
# ==============================================================================
set -e

RELEASE_ROOT="/workspace/releases/lughagen"
TOKENIZER_SRC="/workspace/data/processed/tokenizers/strategy_c"
RESULTS_DIR="/workspace/results"

mkdir -p "$RELEASE_ROOT"

# Model registry: slug|checkpoint_path|eval_json|role
# Empty checkpoint_path or eval_json means "not ready yet, skip silently"
MODELS=(
    "LughaGen-Llama-8B-FFT-S137|/workspace/checkpoints/RUN-P3-FFT/checkpoint-5400|/workspace/results/baseline_bpc_lughagen_fft_s137_per_language_eval.json|Best in-domain specialization (pooled BPC 1.210)"
    "LughaGen-Llama-8B-CPT-S137|/workspace/checkpoints/RUN-P3-CPT/checkpoint-5600|/workspace/results/baseline_bpc_lughagen_cpt_s137_per_language_eval.json|Chapter 6 primary comparison (pooled BPC 1.269)"
    "LughaGen-Llama-8B-QLoRA-S137|/workspace/checkpoints/RUN-P3-CLEAN-QLORA/merged_bf16|/workspace/results/baseline_bpc_lughagen_qlora_merged_s137_per_language_eval.json|Parameter-efficient adaptation (pooled BPC 1.312)"
    "LughaGen-Gemma2-27B-CPT-S137|/workspace/checkpoints/RUN-P3.5-GEMMA2-CPT-S137/checkpoint-6000|/workspace/results/baseline_bpc_lughagen_gemma27b_cpt_s137_per_language_eval.json|Scale + specialization CPT (Gemma 27B base)"
    "LughaGen-Gemma2-27B-FFT-S137|/workspace/checkpoints/RUN-P3.5-GEMMA2-FFT-S137/checkpoint-6000|/workspace/results/baseline_bpc_lughagen_gemma27b_fft_s137_per_language_eval.json|Scale + specialization FFT (Gemma 27B base)"
)

STAGED=()
SKIPPED=()

for entry in "${MODELS[@]}"; do
    IFS='|' read -r SLUG CHECKPOINT EVAL_JSON ROLE <<< "$entry"
    DEST="$RELEASE_ROOT/$SLUG"

    if [ ! -d "$CHECKPOINT" ]; then
        echo "SKIP $SLUG (checkpoint not yet available at $CHECKPOINT)"
        SKIPPED+=("$SLUG")
        continue
    fi

    # Verify checkpoint has model.safetensors (single file, not sharded)
    if [ ! -f "$CHECKPOINT/model.safetensors" ] && [ ! -f "$CHECKPOINT/model.safetensors.index.json" ]; then
        echo "SKIP $SLUG (no model.safetensors or index in $CHECKPOINT)"
        SKIPPED+=("$SLUG (no safetensors)")
        continue
    fi

    echo "STAGE $SLUG"
    mkdir -p "$DEST"

    # ---- Hard-link large model files (saves ~14-54 GB per model) ----
    for f in model.safetensors model.safetensors.index.json; do
        if [ -f "$CHECKPOINT/$f" ] && [ ! -e "$DEST/$f" ]; then
            ln "$CHECKPOINT/$f" "$DEST/$f"
            echo "  hard-linked $f"
        fi
    done

    # Handle sharded safetensors (Gemma 27B is sharded)
    for shard in "$CHECKPOINT"/model-*.safetensors; do
        [ -f "$shard" ] || continue
        base=$(basename "$shard")
        if [ ! -e "$DEST/$base" ]; then
            ln "$shard" "$DEST/$base"
        fi
    done

    # ---- Copy small config files (portable, resilient to source deletion) ----
    for f in config.json generation_config.json; do
        [ -f "$CHECKPOINT/$f" ] && cp "$CHECKPOINT/$f" "$DEST/$f"
    done

    # ---- Copy tokenizer files from strategy_c ----
    cp "$TOKENIZER_SRC/tokenizer.json" "$DEST/tokenizer.json"
    [ -f "$TOKENIZER_SRC/tokenizer_config.json" ] && cp "$TOKENIZER_SRC/tokenizer_config.json" "$DEST/tokenizer_config.json"
    [ -f "$TOKENIZER_SRC/special_tokens_map.json" ] && cp "$TOKENIZER_SRC/special_tokens_map.json" "$DEST/special_tokens_map.json"

    # ---- Copy eval results if available ----
    if [ -f "$EVAL_JSON" ]; then
        cp "$EVAL_JSON" "$DEST/eval_results.json"
    else
        echo "  note: eval results not yet at $EVAL_JSON"
    fi

    # ---- Generate LOAD.py sample script ----
    IS_GEMMA="no"
    [[ "$SLUG" == *"Gemma"* ]] && IS_GEMMA="yes"

    cat > "$DEST/LOAD.py" <<PYEOF
"""
Load and use $SLUG.

Requires: transformers >= 4.44.0, torch >= 2.1.0
Optional: flash-attn (Llama models), or falls back to SDPA
"""
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import torch

# Point at this directory (contains model + tokenizer + configs together)
MODEL_DIR = "."

PYEOF

    if [ "$IS_GEMMA" = "yes" ]; then
        cat >> "$DEST/LOAD.py" <<'PYEOF'
# ---- IMPORTANT for Gemma 2 27B variants ----
# This model was trained with attn_logit_softcapping disabled to permit SDPA
# attention. For consistent behavior at inference, load with softcap disabled
# and use SDPA (do NOT use FA2, do NOT use eager).
config = AutoConfig.from_pretrained(MODEL_DIR)
config.attn_logit_softcapping = None  # matches training-time architecture

model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    config=config,
    torch_dtype=torch.bfloat16,
    attn_implementation="sdpa",  # required
    device_map="auto",
)
PYEOF
    else
        cat >> "$DEST/LOAD.py" <<'PYEOF'
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    torch_dtype=torch.bfloat16,
    attn_implementation="sdpa",  # or "flash_attention_2" if available
    device_map="auto",
)
PYEOF
    fi

    cat >> "$DEST/LOAD.py" <<'PYEOF'

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

# ---- Sample generation ----
prompts = [
    "Habari yako? Karibu",              # Swahili
    "Uugi wakwa niwega, ni",            # Kikuyu
    "Nade maber, ang'o",                # Luo
    "Wailyi vaa, ni",                   # Kamba
    "Vipi mzee, uko",                   # Sheng
]

for prompt in prompts:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=40,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
    print(f"prompt: {prompt}")
    print(f"cont:   {tokenizer.decode(outputs[0], skip_special_tokens=True)}")
    print()
PYEOF

    # ---- Generate per-model README.md ----
    cat > "$DEST/README.md" <<MDEOF
# $SLUG

**$ROLE**

Part of the LughaGen Research Programme at JKUAT.

## What's in this directory

- \`model.safetensors\` (or \`model-*-of-*.safetensors\` for sharded) — model weights
- \`config.json\`, \`generation_config.json\` — model configuration
- \`tokenizer.json\`, \`tokenizer_config.json\`, \`special_tokens_map.json\` — strategy_c 8K tokenizer
- \`LOAD.py\` — ready-to-run sample script (edit MODEL_DIR if moved)
- \`eval_results.json\` — per-language BPC results for verification

## Load and generate

\`\`\`bash
cd /path/to/$SLUG
python LOAD.py
\`\`\`

## Loading from another environment

After copying this directory to a target machine:

\`\`\`python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model = AutoModelForCausalLM.from_pretrained(
    "/path/to/$SLUG",
    torch_dtype=torch.bfloat16,
    attn_implementation="sdpa",
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained("/path/to/$SLUG")
\`\`\`

## Expected results

Verify your setup produces the numbers in \`eval_results.json\`. If BPC values differ by more than ~1% from the reference, check that:
- Tokenizer files loaded from THIS directory (not the base model's tokenizer)
- \`attn_implementation="sdpa"\` (not eager, not FA2 for Gemma)
- bf16 dtype

## License

This model derives from its base model — see the base model's license page linked in the thesis. Users must comply with base model terms.

## Citation

\`\`\`bibtex
@mastersthesis{lughagen2026,
    author = {Mayabi, Derek},
    title  = {LughaGen: Adapting Large Language Models for East African Languages},
    school = {Jomo Kenyatta University of Agriculture and Technology (JKUAT)},
    year   = {2026}
}
\`\`\`
MDEOF

    SIZE=$(du -sh "$DEST" | awk '{print $1}')
    echo "  staged at $DEST ($SIZE)"
    STAGED+=("$SLUG ($SIZE)")
done

# ---- Top-level README with model index ----
cat > "$RELEASE_ROOT/README.md" <<MDEOF
# LughaGen Model Releases

Self-contained model directories for the LughaGen Research Programme.
Each subdirectory contains everything needed to load and run one model.

## Available models

$(for entry in "${MODELS[@]}"; do
    IFS='|' read -r SLUG CHECKPOINT EVAL_JSON ROLE <<< "$entry"
    if [ -d "$RELEASE_ROOT/$SLUG" ]; then
        SIZE=$(du -sh "$RELEASE_ROOT/$SLUG" 2>/dev/null | awk '{print $1}')
        echo "- **$SLUG** ($SIZE) — $ROLE"
    else
        echo "- $SLUG — NOT YET STAGED (training in progress)"
    fi
done)

## Download from Brev

See \`DOWNLOAD.md\` for terminal and Jupyter download instructions.

## Load in another environment

Each subdirectory has its own \`README.md\` and \`LOAD.py\`.
MDEOF

# ---- requirements.txt ----
cat > "$RELEASE_ROOT/requirements.txt" <<'REQEOF'
# LughaGen model requirements (tested versions)
torch>=2.1.0
transformers>=4.44.0
tokenizers>=0.19
safetensors>=0.4.0
accelerate>=0.30.0
sentencepiece>=0.1.99

# Optional (for Llama models only — recommended, ~15% speed-up)
# flash-attn>=2.5.0

# Optional (for evaluation)
datasets>=2.14.0
REQEOF

# ---- DOWNLOAD.md ----
cat > "$RELEASE_ROOT/DOWNLOAD.md" <<'DLEOF'
# Downloading LughaGen models from Brev

## Option 1: SCP from your local machine

```bash
# Single model (14-54 GB depending on model)
scp -r brev-user@your-brev-hostname:/workspace/releases/lughagen/LughaGen-Llama-8B-FFT-S137 ./

# All staged models (larger, resumable with rsync)
rsync -avP brev-user@your-brev-hostname:/workspace/releases/lughagen/ ./lughagen-models/
```

Find your Brev hostname/SSH command in the Brev console (top of the environment page).

## Option 2: Jupyter file browser

1. Open Jupyter on the Brev environment
2. Navigate to `/workspace/releases/lughagen/<model-name>/`
3. Right-click each file → Download
   - You'll need to download files individually — Jupyter doesn't zip directories in-browser

For a full-directory download via Jupyter, first tar it in a terminal cell:

```bash
cd /workspace/releases/lughagen
tar czf LughaGen-Llama-8B-FFT-S137.tar.gz LughaGen-Llama-8B-FFT-S137/
```

Then download the single `.tar.gz` file from the Jupyter file browser.

## Option 3: Brev's built-in `brev port-forward` + local wget/curl

If Brev exposes a file server on the environment:

```bash
brev port-forward your-env-name --port 8000
```

Then on your local machine:

```bash
wget -r http://localhost:8000/workspace/releases/lughagen/
```

(Check current Brev docs for the exact command — this varies by Brev version.)

## Size planning

Approximate download sizes:

| Model | Size |
|---|---:|
| LughaGen-Llama-8B-FFT-S137 | ~14 GB |
| LughaGen-Llama-8B-CPT-S137 | ~14 GB |
| LughaGen-Llama-8B-QLoRA-S137 | ~14 GB |
| LughaGen-Gemma2-27B-CPT-S137 | ~54 GB |
| LughaGen-Gemma2-27B-FFT-S137 | ~54 GB |
| **All 5 (post-Phase-3.5)** | **~150 GB** |

Downloads at ~50 MB/s take about 45 min per Llama model, 3 hours per Gemma model.

## After downloading

```bash
cd LughaGen-Llama-8B-FFT-S137
pip install -r ../requirements.txt   # if you saved requirements.txt too
python LOAD.py                       # runs sample generation
```
DLEOF

# ---- Summary ----
echo ""
echo "========================================"
echo "STAGING COMPLETE"
echo "========================================"
echo ""
echo "Release root: $RELEASE_ROOT"
echo "Total size:   $(du -sh $RELEASE_ROOT | awk '{print $1}')"
echo ""
echo "Staged (${#STAGED[@]}):"
for s in "${STAGED[@]}"; do echo "  - $s"; done
echo ""
if [ "${#SKIPPED[@]}" -gt 0 ]; then
    echo "Skipped (${#SKIPPED[@]}) — re-run this script after training completes:"
    for s in "${SKIPPED[@]}"; do echo "  - $s"; done
fi
