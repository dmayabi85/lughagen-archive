#!/usr/bin/env python3
"""
Token-aware embedding remap: Gemma 2 27B -> LughaGen strategy_c 8K vocab.

v1.1 (2026-08-01): tokenizer-normalized matching + no_grad hygiene.
  - Removed .strip() that was destroying whitespace prefix markers
  - Added SPM (U+2581) / BPE (U+0120) whitespace-marker normalization for
    cross-tokenizer-family exact-match lookup
  - Wrapped remap loop in torch.no_grad() for defensive memory hygiene
"""
import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoTokenizer

SPM_SPACE = "\u2581"
BPE_SPACE = "\u0120"


def normalize_whitespace_markers(token):
    if BPE_SPACE in token:
        return token, token.replace(BPE_SPACE, SPM_SPACE)
    if SPM_SPACE in token:
        return token.replace(SPM_SPACE, BPE_SPACE), token
    return token, token


def find_embed_tokens_shard(gemma_dir):
    for shard in sorted(gemma_dir.glob("*.safetensors")):
        with safe_open(shard, framework="pt", device="cpu") as f:
            for key in f.keys():
                if "embed_tokens" in key and key.endswith("weight"):
                    return f.get_tensor(key), f"{shard.name}::{key}"
    raise RuntimeError(f"embed_tokens.weight not found under {gemma_dir}")


def build_remap(strategy_c_tokenizer_path, gemma_model_path, output_dir):
    print(f"[remap] strategy_c: {strategy_c_tokenizer_path}")
    print(f"[remap] gemma model: {gemma_model_path}")
    print(f"[remap] output: {output_dir}")

    strat = AutoTokenizer.from_pretrained(strategy_c_tokenizer_path)
    gemma = AutoTokenizer.from_pretrained(gemma_model_path)
    n_strat_vocab = len(strat)
    print(f"[remap] strategy_c vocab: {n_strat_vocab}")
    print(f"[remap] gemma vocab: {len(gemma)}")

    embed_tensor, source_ref = find_embed_tokens_shard(gemma_model_path)
    n_gemma_vocab, hidden = embed_tensor.shape
    print(f"[remap] loaded {source_ref}: shape ({n_gemma_vocab}, {hidden})")

    row_norms = embed_tensor.abs().sum(dim=1)
    nonzero_mask = row_norms > 0
    n_nonzero = int(nonzero_mask.sum().item())
    print(f"[remap] source nonzero: {n_nonzero} / {n_gemma_vocab}")
    if n_nonzero == 0:
        raise RuntimeError("All Gemma rows zero. Model load failed?")

    mean_row = embed_tensor[nonzero_mask].mean(dim=0)

    gemma_vocab_raw = gemma.get_vocab()
    gemma_vocab_norm = {}
    for tok, tid in gemma_vocab_raw.items():
        gemma_vocab_norm.setdefault(tok, tid)
        if SPM_SPACE in tok:
            bpe_form = tok.replace(SPM_SPACE, BPE_SPACE)
            gemma_vocab_norm.setdefault(bpe_form, tid)

    output = torch.zeros(n_strat_vocab, hidden, dtype=embed_tensor.dtype)
    stats = {
        "exact_match_kept": 0,
        "exact_match_fallback_mean": 0,
        "decompose_kept": 0,
        "decompose_fallback_mean": 0,
        "mean_fallback_original": 0,
    }

    with torch.no_grad():
        for strat_id in range(n_strat_vocab):
            strat_token = strat.convert_ids_to_tokens(strat_id)
            if strat_token is None:
                output[strat_id] = mean_row
                stats["mean_fallback_original"] += 1
                continue

            bpe_form, spm_form = normalize_whitespace_markers(strat_token)
            source_id = None
            for candidate in (strat_token, bpe_form, spm_form):
                if candidate in gemma_vocab_norm:
                    source_id = gemma_vocab_norm[candidate]
                    break

            if source_id is not None:
                source_row = embed_tensor[source_id]
                if source_row.abs().sum() > 0:
                    output[strat_id] = source_row
                    stats["exact_match_kept"] += 1
                else:
                    output[strat_id] = mean_row
                    stats["exact_match_fallback_mean"] += 1
                continue

            try:
                token_str = strat.decode([strat_id])
                if not token_str:
                    output[strat_id] = mean_row
                    stats["decompose_fallback_mean"] += 1
                    continue

                gemma_ids = gemma.encode(token_str, add_special_tokens=False)
                if len(gemma_ids) == 0:
                    output[strat_id] = mean_row
                    stats["decompose_fallback_mean"] += 1
                    continue

                source_rows = embed_tensor[gemma_ids]
                nonzero = source_rows.abs().sum(dim=1) > 0
                if nonzero.any():
                    output[strat_id] = source_rows[nonzero].mean(dim=0)
                    stats["decompose_kept"] += 1
                else:
                    output[strat_id] = mean_row
                    stats["decompose_fallback_mean"] += 1
            except Exception as e:
                print(f"[remap] warn strat_id={strat_id} ('{strat_token}'): {e}")
                output[strat_id] = mean_row
                stats["mean_fallback_original"] += 1

    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(output, output_dir / "embed_tokens.pt")
    torch.save(output, output_dir / "lm_head.pt")

    metadata = {
        "version": "gemma2-v1.1",
        "source_model": str(gemma_model_path),
        "source_ref": source_ref,
        "source_vocab_size": n_gemma_vocab,
        "target_vocab_size": n_strat_vocab,
        "hidden_size": hidden,
        "source_nonzero_rows": n_nonzero,
        "tie_word_embeddings": True,
        "stats": stats,
        "notes": ("Gemma 2 27B -> strategy_c 8K vocab. lm_head.pt = embed_tokens.pt (tied). "
                  "v1.1 fixes: no strip(), SPM/BPE whitespace-marker normalization, no_grad wrap.")
    }
    with (output_dir / "remap_metadata.json").open("w") as f:
        json.dump(metadata, f, indent=2)

    total = sum(stats.values())
    print(f"\n[remap] SUCCESS")
    print(f"[remap] Total: {total} (expected {n_strat_vocab})")
    for k, v in stats.items():
        pct = 100 * v / total if total else 0
        print(f"[remap]   {k}: {v} ({pct:.2f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy-c-tokenizer",
                        default="/workspace/data/processed/tokenizers/strategy_c")
    parser.add_argument("--gemma-model",
                        default="/workspace/models/gemma-2-27b")
    parser.add_argument("--output-dir",
                        default="/workspace/gemma2_27b_prep/remapped_embeddings")
    args = parser.parse_args()

    build_remap(
        Path(args.strategy_c_tokenizer),
        Path(args.gemma_model),
        Path(args.output_dir),
    )
