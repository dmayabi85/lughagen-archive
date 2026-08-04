#!/usr/bin/env python3
"""Per-language validation perplexity for a saved LughaGen checkpoint.

Replays training_loop's eval path on validation_shared.jsonl, filtering
by `language` before each evaluate() call. Guarantees commensurability
with the pooled val_pph already in the registry.
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
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,
)
from torch.utils.data import DataLoader
from peft import PeftModel

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

LANGUAGES = ["swahili", "luo", "kikuyu", "kamba"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--base-model", default="/workspace/models/llama-3.1-8b-base")
    p.add_argument("--tokenizer", default="/workspace/data/processed/tokenizers/strategy_c")
    p.add_argument("--embedding-remap",
                   default="/workspace/restart_backup_20260527T052144Z/remapped_embeddings")
    p.add_argument("--val-jsonl", required=True)
    p.add_argument("--output", default=None)
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--per-device-batch", type=int, default=4)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def build_model(args):
    print(f"[load] tokenizer from {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[load] base model {args.base_model} (4-bit NF4)")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map={"": 0},
    )

    print(f"[load] resize embeddings to {len(tokenizer)}")
    model.resize_token_embeddings(len(tokenizer))

    remap_path = Path(args.embedding_remap)
    if remap_path.exists():
        embed_file = remap_path / "embed_tokens.pt"
        lm_head_file = remap_path / "lm_head.pt"
        if embed_file.exists() and lm_head_file.exists():
            print(f"[load] applying remapped embeddings from {remap_path}")
            embed_state = torch.load(embed_file, map_location="cpu")
            lm_head_state = torch.load(lm_head_file, map_location="cpu")
            with torch.no_grad():
                model.get_input_embeddings().weight.copy_(embed_state.to(
                    model.get_input_embeddings().weight.dtype))
                model.get_output_embeddings().weight.copy_(lm_head_state.to(
                    model.get_output_embeddings().weight.dtype))
        else:
            print(f"[warn] missing embed_tokens.pt or lm_head.pt in {remap_path}")
            print(f"[warn] files present: {list(remap_path.iterdir())}")

    print(f"[load] LoRA adapter from {args.checkpoint}")
    model = PeftModel.from_pretrained(model, args.checkpoint, is_trainable=False)
    model.eval()
    print(f"[load] model ready")
    return model, tokenizer


def build_eval_dataset(args, tokenizer, language_filter=None):
    print(f"[data] loading {args.val_jsonl} (filter={language_filter})")
    ds = load_dataset("json", data_files=args.val_jsonl,
                      features=LUGHAGEN_FEATURES, split="train")
    if language_filter is not None:
        n_before = len(ds)
        ds = ds.filter(lambda r: r["language"] == language_filter, num_proc=4)
        print(f"[data]   filtered {n_before} -> {len(ds)} for language={language_filter}")
    if args.dry_run:
        n_keep = min(10, len(ds))
        ds = ds.select(range(n_keep))
        print(f"[data]   dry-run: kept {n_keep}")

    seq_len = args.seq_len
    pad_id = tokenizer.pad_token_id

    def tokenize_fn(example):
        enc = tokenizer(example["text"], truncation=True,
                        max_length=seq_len, padding="max_length")
        enc["labels"] = [(tid if tid != pad_id else -100)
                          for tid in enc["input_ids"]]
        return enc

    ds = ds.map(tokenize_fn,
                remove_columns=[c for c in ds.column_names if c != "language"],
                num_proc=4)
    print(f"[data]   tokenized: {len(ds)} examples")
    return ds


def evaluate_subset(model, tokenizer, ds, args, label):
    """Direct forward-pass evaluation. No HF Trainer.

    Iterates over ds in batches, computes mean loss per token (masking pads
    via labels=-100), exponentiates to get perplexity. Equivalent to what
    HF Trainer's evaluate() does internally for a causal LM, but without
    the column-removal/collator complications that broke our first attempt.
    """
    import torch
    device = model.device
    model.eval()

    # Build batches manually. ds rows have input_ids/attention_mask/labels as lists.
    batch_size = args.per_device_batch
    total_loss_weighted = 0.0  # sum of (loss_per_token * n_valid_tokens)
    total_tokens = 0

    t0 = time.time()
    n_batches = (len(ds) + batch_size - 1) // batch_size
    for batch_idx in range(n_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, len(ds))
        rows = [ds[i] for i in range(start, end)]

        input_ids = torch.tensor([r["input_ids"] for r in rows],
                                  dtype=torch.long, device=device)
        attention_mask = torch.tensor([r["attention_mask"] for r in rows],
                                       dtype=torch.long, device=device)
        labels = torch.tensor([r["labels"] for r in rows],
                               dtype=torch.long, device=device)

        # n_valid: count of label positions that are not -100, MINUS one per row
        # (HF causal-LM shifts labels by 1 internally; the last label is always
        # ignored, so each row contributes (n_real_tokens - 1) to the count)
        n_valid = ((labels != -100).sum(dim=1) - 1).clamp(min=0).sum().item()
        if n_valid == 0:
            continue

        with torch.no_grad():
            out = model(input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels)
        # out.loss is mean loss per valid token in this batch
        total_loss_weighted += out.loss.item() * n_valid
        total_tokens += n_valid

    elapsed = time.time() - t0
    mean_loss = total_loss_weighted / total_tokens if total_tokens > 0 else float("nan")
    pph = math.exp(mean_loss) if mean_loss < 50 else float("inf")
    print(f"[eval] {label:>10}: loss={mean_loss:.6f}  pph={pph:.4f}  "
          f"n={len(ds)}  tokens={total_tokens:,}  time={elapsed:.1f}s")
    return {"label": label, "eval_loss": mean_loss, "perplexity": pph,
            "n_samples": len(ds), "n_tokens": total_tokens, "eval_runtime_s": elapsed}


def main():
    args = parse_args()
    if args.output is None:
        args.output = str(Path(args.checkpoint).parent / "per_language_val_pph.json")

    print(f"[start] checkpoint={args.checkpoint}")
    print(f"[start] val={args.val_jsonl}")
    print(f"[start] output={args.output}")
    print(f"[start] dry_run={args.dry_run}")

    model, tokenizer = build_model(args)

    results = {}
    for lang in LANGUAGES:
        ds = build_eval_dataset(args, tokenizer, language_filter=lang)
        results[lang] = evaluate_subset(model, tokenizer, ds, args, label=lang)

    print()
    print("[check] running pooled eval for consistency...")
    ds_pooled = build_eval_dataset(args, tokenizer, language_filter=None)
    results["pooled"] = evaluate_subset(
        model, tokenizer, ds_pooled, args, label="pooled")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "checkpoint": args.checkpoint,
        "val_file": args.val_jsonl,
        "seq_len": args.seq_len,
        "per_device_batch": args.per_device_batch,
        "dry_run": args.dry_run,
        "results": results,
    }, indent=2))

    print()
    print(f"[done] wrote {out_path}")
    print()
    print(f"  {'language':<10}  {'n':>5}  {'eval_loss':>10}  {'perplexity':>12}")
    print(f"  {'-'*10}  {'-'*5}  {'-'*10}  {'-'*12}")
    for lang in LANGUAGES + ["pooled"]:
        r = results[lang]
        print(f"  {lang:<10}  {r['n_samples']:>5}  "
              f"{r['eval_loss']:>10.6f}  {r['perplexity']:>12.4f}")


if __name__ == "__main__":
    main()
