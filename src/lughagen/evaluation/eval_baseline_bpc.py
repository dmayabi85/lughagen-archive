#!/usr/bin/env python3
"""Untuned baseline eval with bits-per-character normalization.

For Gap 3: comparing AfriqueLlama-8B, Lugha-Llama-8B-wura, Gemma 2 27B,
and base Llama-3.1-8B against LughaGen using cross-tokenizer-fair BPC.

Key differences from eval_per_language_5tracks.py:
  - No embedding remap (baseline models keep their native vocab)
  - Uses each model's native tokenizer
  - Computes BPC alongside perplexity (BPC is cross-tokenizer-comparable)
  - Character counting BEFORE tokenization

Usage:
  python3 eval_baseline_bpc.py --model-path /workspace/models/gemma-2-27b \\
      --output /workspace/results/baseline_bpc_gemma2_27b_per_language_eval.json

  python3 eval_baseline_bpc.py --model-path Lugha-Llama/Meta-Llama-3.1-8B-wura \\
      --output /workspace/results/baseline_bpc_lugha_llama_per_language_eval.json
"""
from __future__ import annotations
import argparse, json, math, os, sys, time
from pathlib import Path

# Force HF caches to /data BEFORE importing transformers/datasets
os.environ.setdefault("HF_HOME", "/data/hf_home")
os.environ.setdefault("HF_DATASETS_CACHE", "/data/hf_home/datasets")
os.environ.setdefault("TRANSFORMERS_CACHE", "/data/hf_home/transformers")
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/data/hf_home/hub")
os.environ.setdefault("TMPDIR", "/data/tmp")
os.environ.setdefault("XDG_CACHE_HOME", "/data/cache")
for d in (os.environ["HF_HOME"], os.environ["HF_DATASETS_CACHE"],
          os.environ["TRANSFORMERS_CACHE"], os.environ["HUGGINGFACE_HUB_CACHE"],
          os.environ["TMPDIR"], os.environ["XDG_CACHE_HOME"]):
    Path(d).mkdir(parents=True, exist_ok=True)

import torch
from datasets import load_dataset, Features, Value, Sequence
from transformers import AutoTokenizer, AutoModelForCausalLM

LUGHAGEN_FEATURES = Features({
    "doc_id":             Value("string"),
    "text":               Value("string"),
    "language":           Value("string"),
    "origin":             Value("string"),
    "register":           Value("string"),
    "register_detail":    Value("string"),
    "split":              Value("string"),
    "domain":             Value("string"),
    "license":            Value("string"),
    "source_id":          Value("string"),
    "source_url":         Value("string"),
    "ingested_at":        Value("string"),
    "code_switch_pair":   Value("string"),
    "code_switch_ratio":  Value("float64"),
    "quality_score":      Value("float64"),
    "fluency_score":      Value("float64"),
    "words":              Value("int64"),
    "tokens_llama":       Value("int64"),
    "augmentation":       Sequence(Value("string")),
    "preprocessing":      Sequence(Value("string")),
})

LANGUAGES = ["swahili_authentic", "kikuyu", "luo", "kamba", "sheng"]
SHENG_SOURCE_IDS = {"swahili_sheng_v1", "sheng_external_v1"}
SHENG_VAL_JSONL = "/workspace/data/corpus/splits_v2/sheng_val.jsonl"

LN2 = math.log(2.0)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True,
                   help="Path or HuggingFace ID of the model to evaluate")
    p.add_argument("--tokenizer", default=None,
                   help="Tokenizer path (defaults to model-path)")
    p.add_argument("--val-jsonl",
                   default="/workspace/data/corpus/splits_v2/validation_shared_v3.jsonl")
    p.add_argument("--output", default=None,
                   help="Output JSON path (default: baseline_bpc_<name>_per_language_eval.json)")
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--per-device-batch", type=int, default=4)
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def build_model(args):
    tok_path = args.tokenizer or args.model_path
    print(f"[load] tokenizer from {tok_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        tok_path, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"[load] pad_token set to eos_token ({tokenizer.eos_token})")

    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    dtype = dtype_map[args.dtype]

    print(f"[load] model from {args.model_path} (dtype={args.dtype})")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        dtype=dtype,
        attn_implementation="flash_attention_2",
        device_map="auto",
        trust_remote_code=args.trust_remote_code,
    )
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[load] loaded {n_params:,} params ({n_params/1e9:.2f}B)")
    return model, tokenizer


def load_language_subset(args, language):
    """Filter validation set. Matches LughaGen eval_per_language_5tracks.py filters exactly.
    
    - swahili_authentic: language=="swahili" AND register=="monolingual"
    - kikuyu/luo/kamba: language==<name>
    - sheng: separate file, filtered by source_id
    """
    if language == "sheng":
        val_file = SHENG_VAL_JSONL
    else:
        val_file = args.val_jsonl

    if not Path(val_file).exists():
        print(f"[warn] {val_file} does not exist, skipping {language}")
        return None

    print(f"[data] loading {val_file} (filter={language})")
    ds = load_dataset("json", data_files=val_file, split="train",
                      features=LUGHAGEN_FEATURES)
    n_before = len(ds)

    if language == "sheng":
        ds = ds.filter(lambda r: r.get("source_id") in SHENG_SOURCE_IDS,
                       num_proc=4)
    elif language == "swahili_authentic":
        # Matches LughaGen filter: swahili + monolingual (excludes code-switched)
        ds = ds.filter(
            lambda r: r["language"] == "swahili" and r["register"] == "monolingual",
            num_proc=4
        )
    elif language in ("kikuyu", "luo", "kamba"):
        ds = ds.filter(lambda r: r.get("language") == language, num_proc=4)
    else:
        print(f"[warn] unknown language filter: {language}")
        return None

    if len(ds) == 0:
        print(f"[warn] {language} filter yielded 0 records (of {n_before})")
        return None

    print(f"[data]   filtered {n_before} -> {len(ds)} for {language}")
    return ds


def evaluate_language(model, tokenizer, ds, args, label):
    """Direct forward-pass evaluation with BPC computation."""
    device = model.device
    model.eval()
    batch_size = args.per_device_batch

    total_loss_weighted = 0.0
    total_tokens = 0
    total_chars = 0

    # Pre-count characters (raw text length before tokenization)
    for row in ds:
        total_chars += len(row["text"])

    t0 = time.time()

    for start in range(0, len(ds), batch_size):
        end = min(start + batch_size, len(ds))
        texts = [ds[i]["text"] for i in range(start, end)]

        enc = tokenizer(texts, truncation=True, max_length=args.seq_len,
                        padding="max_length", return_tensors="pt")
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        # Build labels: mask pads with -100
        pad_id = tokenizer.pad_token_id
        labels = input_ids.clone()
        labels[labels == pad_id] = -100

        # n_valid: labels != -100, minus one per row for causal shift
        n_valid = ((labels != -100).sum(dim=1) - 1).clamp(min=0).sum().item()
        if n_valid == 0:
            continue

        with torch.no_grad():
            out = model(input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels)

        total_loss_weighted += out.loss.item() * n_valid
        total_tokens += n_valid

    elapsed = time.time() - t0
    mean_loss = total_loss_weighted / total_tokens if total_tokens > 0 else float("nan")
    pph = math.exp(mean_loss) if mean_loss < 50 else float("inf")

    # BPC: bits per character
    #   total loss in nats = mean_loss * total_tokens
    #   convert to bits: divide by ln(2)
    #   normalize by chars: divide by total_chars
    if total_chars > 0 and total_tokens > 0:
        bpc = (mean_loss * total_tokens) / (total_chars * LN2)
        avg_chars_per_token = total_chars / total_tokens
    else:
        bpc = float("nan")
        avg_chars_per_token = float("nan")

    print(f"[eval] {label:>22}: loss={mean_loss:.4f}  ppl={pph:.4f}  "
          f"bpc={bpc:.4f}  chars/tok={avg_chars_per_token:.2f}  "
          f"n={len(ds)}  tokens={total_tokens:,}  chars={total_chars:,}  "
          f"time={elapsed:.1f}s")

    return {
        "label": label,
        "eval_loss": mean_loss,
        "perplexity": pph,
        "bpc": bpc,
        "avg_chars_per_token": avg_chars_per_token,
        "n_samples": len(ds),
        "n_tokens": total_tokens,
        "n_chars": total_chars,
        "eval_runtime_s": elapsed,
    }


def main():
    args = parse_args()

    # Auto-generate output name from model path if not specified
    if args.output is None:
        model_name = Path(args.model_path).name if "/" not in args.model_path \
                     else args.model_path.replace("/", "_")
        args.output = f"/workspace/results/baseline_bpc_{model_name}_per_language_eval.json"

    print(f"[start] model_path={args.model_path}")
    print(f"[start] val_file={args.val_jsonl}")
    print(f"[start] output={args.output}")
    print(f"[start] seq_len={args.seq_len}, batch={args.per_device_batch}, dtype={args.dtype}")

    if args.dry_run:
        print("[dry-run] exiting before load")
        return

    model, tokenizer = build_model(args)

    results = {}
    for language in LANGUAGES:
        ds = load_language_subset(args, language)
        if ds is None:
            continue
        results[language] = evaluate_language(model, tokenizer, ds, args, language)

    # Pooled aggregate: 4 monolingual languages ONLY (Sheng excluded).
    # Matches LughaGen eval_per_language_5tracks.py convention for backwards
    # comparability with prior LughaGen validation numbers.
    POOLED_LANGS = {"swahili_authentic", "kikuyu", "luo", "kamba"}
    pooled_results = {k: v for k, v in results.items() if k in POOLED_LANGS}
    total_loss = sum(r["eval_loss"] * r["n_tokens"] for r in pooled_results.values())
    total_tokens = sum(r["n_tokens"] for r in pooled_results.values())
    total_chars = sum(r["n_chars"] for r in pooled_results.values())
    total_samples = sum(r["n_samples"] for r in pooled_results.values())

    if total_tokens > 0:
        pooled_loss = total_loss / total_tokens
        pooled_ppl = math.exp(pooled_loss) if pooled_loss < 50 else float("inf")
        pooled_bpc = (pooled_loss * total_tokens) / (total_chars * LN2)
        pooled_cpt = total_chars / total_tokens
    else:
        pooled_loss = float("nan")
        pooled_ppl = float("nan")
        pooled_bpc = float("nan")
        pooled_cpt = float("nan")

    results["pooled"] = {
        "label": "pooled",
        "eval_loss": pooled_loss,
        "perplexity": pooled_ppl,
        "bpc": pooled_bpc,
        "avg_chars_per_token": pooled_cpt,
        "n_samples": total_samples,
        "n_tokens": total_tokens,
        "n_chars": total_chars,
        "eval_runtime_s": sum(r["eval_runtime_s"] for r in pooled_results.values()),
        "note": "pooled = 4 monolingual languages (swahili_authentic, kikuyu, luo, kamba); Sheng excluded to match LughaGen convention",
    }

    print(f"\n[eval] {'POOLED':>22}: loss={pooled_loss:.4f}  ppl={pooled_ppl:.4f}  "
          f"bpc={pooled_bpc:.4f}  chars/tok={pooled_cpt:.2f}  tokens={total_tokens:,}")

    # Save output
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out = {
        "model_path": args.model_path,
        "tokenizer": args.tokenizer or args.model_path,
        "val_file": args.val_jsonl,
        "seq_len": args.seq_len,
        "per_device_batch": args.per_device_batch,
        "dtype": args.dtype,
        "results": results,
    }
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[done] wrote {args.output}")


if __name__ == "__main__":
    main()
