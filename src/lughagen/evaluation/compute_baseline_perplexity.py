#!/usr/bin/env python3
"""
/workspace/scripts/compute_baseline_perplexity.py

LughaGen Training Framework v1.5 §4.2
Computes P_baseline: validation perplexity of zero-shot LLaMA-3.1-8B (no
adaptation) on /workspace/data/corpus/splits/validation.jsonl under the
Phase-0-winning tokenizer.

This is executed ONCE per study (Week 3 in the timeline). Outputs are written
to /workspace/registry/global_constants.yaml:
  - baseline_perplexity.pooled        (single scalar)
  - baseline_perplexity.per_language  (swahili, kikuyu, kamba, luo)
  - baseline_perplexity.computed_at
  - baseline_perplexity.tokenizer_version

Procedure (§4.2 verbatim):
  1. Load LLaMA-3.1-8B base weights with no adaptation
  2. Tokenize validation.jsonl using the Phase-0-winning tokenizer
  3. Compute perplexity as exp(mean(cross_entropy_per_token)) with
     max_seq_length=2048, no padding loss contribution, BF16 precision
  4. Report pooled scalar + four per-language scalars

If the Phase-0 tokenizer changes later, this script must be re-run and all
downstream PPH values invalidated.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable

# Heavy imports are deferred so the CLI parses cleanly on CPU-only nodes
sys.path.insert(0, "/workspace/scripts")
from registry_csv import write_global_constant, get_global_constant, utc_now_iso


VALIDATION_FILE = Path("/workspace/data/corpus/splits/validation.jsonl")
LANGUAGES = ("swahili", "kikuyu", "kamba", "luo")


def stream_records(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def compute_perplexity(
    model,
    tokenizer,
    records: list[dict],
    max_seq_length: int,
    device: str,
    batch_size: int = 1,
) -> dict[str, float]:
    """
    Compute pooled and per-language perplexity.

    Returns dict with keys: pooled, swahili, kikuyu, kamba, luo.

    Implementation notes (§4.2):
      - exp(mean(cross_entropy_per_token))
      - max_seq_length=2048
      - no padding loss contribution: we use attention_mask to exclude padding
      - BF16 precision: model is loaded in BF16 by caller
    """
    import torch
    import torch.nn.functional as F

    # Accumulate token-level cross-entropy and token counts, per language and pooled
    nll_sum_pooled = 0.0
    tok_count_pooled = 0
    nll_sum_lang = defaultdict(float)
    tok_count_lang = defaultdict(int)

    model.eval()

    with torch.no_grad():
        for i, rec in enumerate(records):
            text = rec.get("text", "")
            lang = rec.get("language", "")
            if not text or lang not in LANGUAGES:
                continue

            enc = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_seq_length,
                padding=False,
            )
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)

            if input_ids.shape[1] < 2:
                continue  # need at least 2 tokens to compute LM loss

            # Forward pass; labels = input_ids with -100 where mask is 0 to
            # exclude padding from loss (no padding here since batch_size=1
            # but kept explicit for §4.2 'no padding loss contribution')
            labels = input_ids.clone()
            labels[attention_mask == 0] = -100

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits  # [1, T, V]

            # Shift for next-token prediction
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()

            # Compute per-token CE
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
                reduction="sum",
            )
            n_tokens = (shift_labels != -100).sum().item()

            if n_tokens == 0:
                continue

            loss_val = loss.item()
            nll_sum_pooled += loss_val
            tok_count_pooled += n_tokens
            nll_sum_lang[lang] += loss_val
            tok_count_lang[lang] += n_tokens

            if (i + 1) % 500 == 0:
                pooled_pp = math.exp(nll_sum_pooled / max(tok_count_pooled, 1))
                print(f"  processed {i+1} records; running pooled PPL = {pooled_pp:.3f}")

    out = {}
    out["pooled"] = math.exp(nll_sum_pooled / max(tok_count_pooled, 1))
    for lang in LANGUAGES:
        if tok_count_lang[lang] > 0:
            out[lang] = math.exp(nll_sum_lang[lang] / tok_count_lang[lang])
        else:
            out[lang] = None
            print(f"  WARNING: no validation tokens for {lang}", file=sys.stderr)

    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute P_baseline (§4.2)")
    parser.add_argument(
        "--model-path", type=str,
        # CLUSTER-SPECIFIC: path to LLaMA-3.1-8B base weights on Brev
        default="/workspace/models/llama-3.1-8b-base",
        help="Path to LLaMA-3.1-8B base weights (no adaptation).",
    )
    parser.add_argument(
        "--tokenizer-path", type=str, default=None,
        help="Path to Phase-0-winning tokenizer. "
             "Defaults to value in global_constants.yaml.",
    )
    parser.add_argument(
        "--validation-file", type=Path, default=VALIDATION_FILE,
    )
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int, default=None,
                        help="Optional cap on validation records (for smoke testing).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Parse and validate inputs without loading the model.")
    args = parser.parse_args()

    # Resolve tokenizer
    tokenizer_path = args.tokenizer_path or get_global_constant("tokenizer_version.value")
    if not tokenizer_path:
        print("ERROR: no tokenizer specified and tokenizer_version.value is unset "
              "in global_constants.yaml. Run Phase 0 first or pass --tokenizer-path.",
              file=sys.stderr)
        return 1

    # Read validation records
    print(f"Loading validation records from {args.validation_file}")
    records = list(stream_records(args.validation_file))
    if args.limit is not None:
        records = records[:args.limit]
    print(f"  {len(records)} records loaded")

    if args.dry_run:
        print("Dry run; skipping model load and perplexity computation.")
        return 0

    # Heavy imports
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading tokenizer from {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model from {args.model_path} (BF16, no adaptation)")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",  # FA2 per §9.1
        device_map=args.device,
    )

    # Resize embeddings to match Strategy C tokenizer (8000 vocab) so that
    # P_baseline measures the model state Phase 2 will START from
    # (training_loop.py line 368 does the same resize). Without this, the
    # 128K LLaMA embeddings are looked up by 8K Strategy C token IDs, producing
    # near-random perplexity (~17000). Per spec §4.2 line 347: "If the Phase-0
    # tokenizer choice changes later, P_baseline must be recomputed" — this
    # confirms P_baseline is tokenizer-coupled, not native-tokenizer-based.
    original_vocab_size = model.get_input_embeddings().num_embeddings
    target_vocab_size = len(tokenizer)
    if original_vocab_size != target_vocab_size:
        print(f"Resizing embeddings: {original_vocab_size} -> {target_vocab_size}")
        # Standard practice: init new rows from mean of existing embedding matrix
        # so new tokens start at "average" position in embedding space rather
        # than random. This is the same init Phase 2 will use.
        old_embeddings = model.get_input_embeddings().weight.data
        embed_mean = old_embeddings.mean(dim=0)
        embed_std = old_embeddings.std(dim=0).mean().item()

        model.resize_token_embeddings(target_vocab_size)
        new_embeddings = model.get_input_embeddings().weight.data

        # Overwrite ALL rows of the new embedding with mean+small_noise.
        # This is the "warm-start" init that's standard for tokenizer-swap baselines.
        import torch
        with torch.no_grad():
            noise = torch.randn_like(new_embeddings) * (embed_std * 0.02)
            new_embeddings.copy_(embed_mean.unsqueeze(0).expand_as(new_embeddings) + noise)

        # Also resize output (LM head) if not tied
        if hasattr(model, "lm_head") and not getattr(model.config, "tie_word_embeddings", False):
            # Force a fresh init of lm_head — same procedure
            old_lm_head = model.lm_head.weight.data
            lm_head_mean = old_lm_head.mean(dim=0)
            lm_head_std = old_lm_head.std(dim=0).mean().item()
            with torch.no_grad():
                new_lm_head = model.lm_head.weight.data
                noise = torch.randn_like(new_lm_head) * (lm_head_std * 0.02)
                new_lm_head.copy_(lm_head_mean.unsqueeze(0).expand_as(new_lm_head) + noise)

        print(f"  embedding mean init: |mean|={embed_mean.abs().mean():.4f}, "
              f"std_scale={embed_std * 0.02:.4f}")
    else:
        print(f"Embedding size already matches tokenizer ({target_vocab_size}), no resize")

    print("Computing perplexity...")
    ppl = compute_perplexity(
        model=model,
        tokenizer=tokenizer,
        records=records,
        max_seq_length=args.max_seq_length,
        device=args.device,
    )

    print("\n=== P_baseline ===")
    print(f"  pooled  = {ppl['pooled']:.4f}")
    for lang in LANGUAGES:
        v = ppl.get(lang)
        print(f"  {lang:8s}= {v:.4f}" if v is not None else f"  {lang:8s}= (no data)")

    # Persist to global_constants.yaml (immutable-write)
    write_global_constant("baseline_perplexity.pooled", float(ppl["pooled"]))
    write_global_constant("baseline_perplexity.per_language", {
        lang: (float(ppl[lang]) if ppl.get(lang) is not None else None)
        for lang in LANGUAGES
    })
    write_global_constant("baseline_perplexity.computed_at", utc_now_iso())
    write_global_constant("baseline_perplexity.tokenizer_version", tokenizer_path)

    print(f"\nWritten to /workspace/registry/global_constants.yaml")
    return 0


if __name__ == "__main__":
    sys.exit(main())
