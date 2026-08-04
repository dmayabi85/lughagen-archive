"""
Generate HuggingFace model cards (README.md) for LughaGen model releases.

For each model, produces a README.md file at:
    /workspace/gemma2_27b_prep/releases/<model-slug>/README.md

Usage:
    /workspace/.venv/bin/python /workspace/scripts/generate_model_cards.py

Then review the generated files before running the upload script.
"""
import json
from pathlib import Path
from datetime import datetime

RELEASES_DIR = Path("/workspace/gemma2_27b_prep/releases")
RELEASES_DIR.mkdir(exist_ok=True, parents=True)

# ============================================================
# Model registry — one entry per model to release
# ============================================================
MODELS = [
    {
        "slug": "LughaGen-Llama-8B-FFT-S137",
        "base_model": "meta-llama/Llama-3.1-8B",
        "base_license": "llama3.1",
        "base_license_url": "https://www.llama.com/llama3_1/license/",
        "method": "Full Fine-Tuning (FFT)",
        "params": "8.03B",
        "peak_lr": "2e-5",
        "warmup_steps": 200,
        "weight_decay": 0.01,
        "batch_size": 256,
        "seq_len": 1024,
        "total_steps": 6000,
        "seed": 137,
        "data_ordering": "curriculum (code-switch)",
        "val_ppl_registry": 19.550,
        "val_ppl_eval_script": 19.879,
        "bpc_pooled": 1.210,
        "bpc_per_lang": {"Swahili": 0.950, "Kikuyu": 1.401, "Luo": 1.318, "Kamba": 1.655, "Sheng": 1.369},
        "chapter": "Chapter 5 primary FFT model (best in-domain specialization)",
        "checkpoint_path": "/workspace/checkpoints/RUN-P3-FFT/checkpoint-5400",
        "tokenizer": "LughaGen-strategy-c-tokenizer",
    },
    {
        "slug": "LughaGen-Llama-8B-CPT-S137",
        "base_model": "meta-llama/Llama-3.1-8B",
        "base_license": "llama3.1",
        "base_license_url": "https://www.llama.com/llama3_1/license/",
        "method": "Continued Pre-Training (CPT)",
        "params": "8.03B",
        "peak_lr": "1e-5",
        "warmup_steps": 500,
        "weight_decay": 0.1,
        "batch_size": 256,
        "seq_len": 1024,
        "total_steps": 6000,
        "seed": 137,
        "data_ordering": "mixed (natural)",
        "val_ppl_registry": 22.193,
        "val_ppl_eval_script": 22.812,
        "bpc_pooled": 1.269,
        "bpc_per_lang": {"Swahili": 0.981, "Kikuyu": 1.478, "Luo": 1.419, "Kamba": 1.742, "Sheng": 1.478},
        "chapter": "Chapter 6 primary comparison model (matches Lugha-Llama/AfriqueLlama CPT protocol)",
        "checkpoint_path": "/workspace/checkpoints/RUN-P3-CPT/checkpoint-5600",
        "tokenizer": "LughaGen-strategy-c-tokenizer",
    },
    {
        "slug": "LughaGen-Llama-8B-QLoRA-S137",
        "base_model": "meta-llama/Llama-3.1-8B",
        "base_license": "llama3.1",
        "base_license_url": "https://www.llama.com/llama3_1/license/",
        "method": "QLoRA (4-bit NF4, merged bf16)",
        "params": "8.03B (fully merged)",
        "peak_lr": "2e-5",
        "warmup_steps": 200,
        "weight_decay": 0.01,
        "batch_size": 256,
        "seq_len": 1024,
        "total_steps": 6000,
        "seed": 137,
        "data_ordering": "curriculum",
        "val_ppl_registry": 23.689,
        "val_ppl_eval_script": 25.575,
        "bpc_pooled": 1.312,
        "bpc_per_lang": {"Swahili": 1.022, "Kikuyu": 1.520, "Luo": 1.452, "Kamba": 1.799, "Sheng": 1.632},
        "chapter": "Chapter 5 parameter-efficient method demonstration",
        "checkpoint_path": "/workspace/checkpoints/RUN-P3-CLEAN-QLORA/merged_bf16",
        "tokenizer": "LughaGen-strategy-c-tokenizer",
    },
    {
        "slug": "LughaGen-Gemma2-27B-CPT-S137",
        "base_model": "google/gemma-2-27b",
        "base_license": "gemma",
        "base_license_url": "https://ai.google.dev/gemma/terms",
        "method": "Continued Pre-Training (CPT) with disabled softcap",
        "params": "27.23B",
        "peak_lr": "5e-6",
        "warmup_steps": 500,
        "weight_decay": 0.1,
        "batch_size": 128,
        "seq_len": 1024,
        "total_steps": 6000,
        "seed": 137,
        "data_ordering": "mixed (natural)",
        "val_ppl_registry": None,  # populated post-training
        "val_ppl_eval_script": None,
        "bpc_pooled": None,
        "bpc_per_lang": None,
        "chapter": "Chapter 6 scale × specialization comparison (Gemma 27B base)",
        "checkpoint_path": "/workspace/checkpoints/RUN-P3.5-GEMMA2-CPT-S137/checkpoint-6000",
        "tokenizer": "LughaGen-strategy-c-tokenizer",
        "notes": "attn_logit_softcapping disabled at load to permit SDPA attention (transformers 5.8.1 refuses FA2/SDPA with softcap on Gemma 2). Model was trained without softcap; use SDPA at inference.",
    },
    {
        "slug": "LughaGen-Gemma2-27B-FFT-S137",
        "base_model": "google/gemma-2-27b",
        "base_license": "gemma",
        "base_license_url": "https://ai.google.dev/gemma/terms",
        "method": "Full Fine-Tuning (FFT) with disabled softcap",
        "params": "27.23B",
        "peak_lr": "1e-5",
        "warmup_steps": 200,
        "weight_decay": 0.01,
        "batch_size": 128,
        "seq_len": 1024,
        "total_steps": 6000,
        "seed": 137,
        "data_ordering": "curriculum",
        "val_ppl_registry": None,
        "val_ppl_eval_script": None,
        "bpc_pooled": None,
        "bpc_per_lang": None,
        "chapter": "Chapter 6 paradigm × scale interaction (Gemma 27B base + FFT recipe)",
        "checkpoint_path": "/workspace/checkpoints/RUN-P3.5-GEMMA2-FFT-S137/checkpoint-6000",
        "tokenizer": "LughaGen-strategy-c-tokenizer",
        "notes": "attn_logit_softcapping disabled at load to permit SDPA attention. Use SDPA at inference.",
    },
    # Tokenizer (uploaded as a separate repo)
    {
        "slug": "LughaGen-strategy-c-tokenizer",
        "base_model": None,
        "method": "BPE-8000 vocab tuned for East African languages",
        "params": None,
        "chapter": "Chapter 3 tokenizer contribution",
        "checkpoint_path": "/workspace/data/processed/tokenizers/strategy_c",
        "is_tokenizer": True,
    },
]

# ============================================================
# Model card template
# ============================================================
def render_model_card(m):
    is_tok = m.get("is_tokenizer", False)

    header = f"""---
library_name: transformers
language:
  - sw
  - ki
  - luo
  - kam
tags:
  - lughagen
  - east-african-languages
  - swahili
  - kikuyu
  - luo
  - kamba
  - sheng
"""
    if not is_tok:
        header += f"""base_model: {m['base_model']}
license: {m['base_license']}
"""
    else:
        header += "license: apache-2.0\n"

    header += "---\n\n"

    if is_tok:
        return header + render_tokenizer_body(m)
    else:
        return header + render_model_body(m)


def render_tokenizer_body(m):
    return f"""# {m['slug']}

**A BPE-8000 tokenizer optimized for East African languages.**

Part of the LughaGen Research Programme at JKUAT (Jomo Kenyatta University of Agriculture and Technology).

## Overview

`strategy_c` is a byte-pair encoding tokenizer with 8000 vocabulary items, trained on Kenyan Swahili, Kikuyu, Luo, Kamba, and Sheng corpora. It is the tokenizer used by all LughaGen models and is required to load them.

## Tokenizer efficiency (chars per token, higher is better)

| Language | strategy_c 8K | Llama 3.1 128K | Gemma 2 256K | Efficiency gain |
|---|---:|---:|---:|---:|
| Swahili | 3.94 | 2.78 | 3.34 | +42% vs Llama |
| Kikuyu | 3.38 | 2.01 | 2.16 | +68% vs Llama |
| Luo | 3.26 | 2.72 | 2.96 | +19% vs Llama |
| Kamba | 3.12 | 2.10 | 2.28 | +49% vs Llama |
| Sheng | 2.82 | 2.77 | 3.16 | +2% vs Llama |

## Usage

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("YOUR_USERNAME/{m['slug']}")
tokens = tokenizer("Habari yako?", return_tensors="pt")
```

## Citation

Please cite the LughaGen thesis when using this tokenizer:

```bibtex
@mastersthesis{{lughagen2026,
    author = {{Mayabi, Derek}},
    title = {{LughaGen: Adapting Large Language Models for East African Languages}},
    school = {{Jomo Kenyatta University of Agriculture and Technology (JKUAT)}},
    year = {{2026}},
    type = {{{{MSc}} thesis}},
    address = {{Nairobi, Kenya}}
}}
```

## License

Apache 2.0 for the tokenizer artifacts. The training corpora used to fit this tokenizer are subject to their original licenses.
"""


def render_model_body(m):
    hyperparams_table = f"""| Hyperparameter | Value |
|---|---|
| Base model | `{m['base_model']}` |
| Adaptation method | {m['method']} |
| Parameters | {m['params']} |
| Peak learning rate | {m['peak_lr']} |
| Warmup steps | {m['warmup_steps']} |
| Weight decay | {m['weight_decay']} |
| Batch size | {m['batch_size']} |
| Sequence length | {m['seq_len']} |
| Total steps | {m['total_steps']} |
| Data ordering | {m['data_ordering']} |
| Seed | {m['seed']} |
"""

    if m.get("bpc_pooled") is not None:
        eval_section = f"""
## Evaluation results

Validation perplexity on LughaGen validation set (v3, 5976 records):

| Source | Pooled val_ppl |
|---|---:|
| Training-time (HF Trainer) | {m['val_ppl_registry']:.3f} |
| Post-hoc eval script | {m['val_ppl_eval_script']:.3f} |

Cross-tokenizer-fair bits-per-character (BPC) on validation set:

| Language | BPC |
|---|---:|
| Swahili | {m['bpc_per_lang']['Swahili']:.3f} |
| Kikuyu | {m['bpc_per_lang']['Kikuyu']:.3f} |
| Luo | {m['bpc_per_lang']['Luo']:.3f} |
| Kamba | {m['bpc_per_lang']['Kamba']:.3f} |
| Sheng | {m['bpc_per_lang']['Sheng']:.3f} |
| **Pooled (4 monolingual)** | **{m['bpc_pooled']:.3f}** |

Full comparison against 6 external baselines (AfriqueLlama-8B, Lugha-Llama-8B-wura, Lugha-Llama-8B-wura_edu, Gemma 2 27B untuned, AfriqueGemma-4B, AfriqueGemma-12B) is available in Chapter 6 of the thesis.

"""
    else:
        eval_section = """
## Evaluation results

Evaluation results will be added after training completes. See the thesis Chapter 5/6 for the full comparison.

"""

    notes_section = ""
    if m.get("notes"):
        notes_section = f"""
## Important notes

{m['notes']}

"""

    return f"""# {m['slug']}

**A LughaGen model adapted for East African languages.**

Part of the LughaGen Research Programme at JKUAT (Jomo Kenyatta University of Agriculture and Technology). {m['chapter']}.

## Model description

This model is `{m['base_model']}` adapted via **{m['method']}** on a native East African language corpus (Kenyan Swahili, Kikuyu, Luo, Kamba, and Sheng). It uses the [`YOUR_USERNAME/{m['tokenizer']}`](../{m['tokenizer']}) tokenizer, which has 8,000 vocabulary items optimized for these languages.

The model was trained as part of an MSc research programme investigating whether dedicated native-language corpora can produce competitive language models on under-resourced Kenyan Bantu languages (Kikuyu, Kamba, Luo) that are absent from widely used African language corpora such as WURA.

## Hyperparameters

{hyperparams_table}
{eval_section}
{notes_section}
## Intended use

- Perplexity/BPC benchmarking on East African languages
- Downstream evaluation (translation, classification) via few-shot or additional fine-tuning
- Research on adaptation methods for under-resourced languages

**Not intended for**:
- Direct chatbot deployment (this is a base language model, not instruction-tuned)
- Production use without additional safety evaluation
- Languages other than the 5 covered above

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model = AutoModelForCausalLM.from_pretrained(
    "YOUR_USERNAME/{m['slug']}",
    torch_dtype=torch.bfloat16,
    attn_implementation="sdpa",  # Required for Gemma 2 variants; recommended for all
)
tokenizer = AutoTokenizer.from_pretrained("YOUR_USERNAME/{m['tokenizer']}")

inputs = tokenizer("Habari yako? Karibu kwa", return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=50, do_sample=True, temperature=0.7)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Training data

Filtered path-D corpus (`pathD_v2_filtered`) consisting of native Kenyan text across Swahili (authentic monolingual), Kikuyu, Luo, Kamba, and Sheng (Swahili-English code-switch). Full corpus construction and filtering methodology is described in Chapter 3 of the thesis.

Corpus size: approximately 1.38M records / ~1B tokens.

## Training compute

Trained on 8× NVIDIA A100 80GB (Brev.dev) using PyTorch + DeepSpeed ZeRO-3. Total training compute was on the order of 300-750 GPU-hours per model depending on base model size.

## License

This model is a derivative of `{m['base_model']}` and inherits its base model license:

- Base model license: [{m['base_license']}]({m['base_license_url']})

Users of this model must comply with the base model's license terms.

## Citation

If you use this model, please cite the LughaGen thesis:

```bibtex
@mastersthesis{{lughagen2026,
    author = {{Mayabi, Derek}},
    title = {{LughaGen: Adapting Large Language Models for East African Languages}},
    school = {{Jomo Kenyatta University of Agriculture and Technology (JKUAT)}},
    year = {{2026}},
    type = {{{{MSc}} thesis}},
    address = {{Nairobi, Kenya}}
}}
```

## Acknowledgements

Developed as part of the LughaGen Research Programme under the supervision of Dr. Lawrence Nderu at JKUAT. Compute provided by Brev.dev.
"""


# ============================================================
# Write cards
# ============================================================
for m in MODELS:
    out_dir = RELEASES_DIR / m["slug"]
    out_dir.mkdir(exist_ok=True, parents=True)
    card_path = out_dir / "README.md"
    card_path.write_text(render_model_card(m))
    print(f"  written: {card_path}")

print(f"\n{len(MODELS)} model cards generated in {RELEASES_DIR}")
print("Review before uploading:")
print("  ls /workspace/gemma2_27b_prep/releases/")
print("\nReplace YOUR_USERNAME in each README with your actual HF username before upload.")
