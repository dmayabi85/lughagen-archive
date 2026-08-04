#!/usr/bin/env python3
"""
/workspace/scripts/tokenizer_strategy_a.py

LughaGen Training Framework v1.5.1 §6.1 — Strategy A
Train a SentencePiece tokenizer on the four target languages and ADD its new
tokens to the LLaMA-3.1-8B base tokenizer (vocabulary extension).

Rationale: keeps LLaMA's existing vocabulary intact so the base model's
weights remain meaningful for English/code; adds Bantu-language-specific
tokens to reduce fertility on Swahili/Kikuyu/Kamba/Luo.

Output:
  /workspace/data/processed/tokenizers/strategy_a/
    ├── tokenizer.json              ← extended tokenizer
    ├── tokenizer_config.json       ← config for transformers
    ├── added_tokens.json           ← the new tokens (audit)
    ├── special_tokens_map.json
    ├── lugha_extension.model       ← SentencePiece artifact
    ├── lugha_extension.vocab
    └── step_fertility.jsonl        ← per-step fertility log for §6.4
"""

from __future__ import annotations

import json
import math
import random
import tempfile
from pathlib import Path
from typing import Iterable, Optional

LANGUAGES = ("swahili", "kikuyu", "kamba", "luo")

# CLUSTER-SPECIFIC: path to base LLaMA tokenizer
LLAMA_BASE_PATH = "/workspace/models/llama-3.1-8b-base"

# Phase 0 sampling: per-language word cap for tokenizer training corpus
PER_LANG_TOKEN_BUDGET = 2_000_000


def stream_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def sample_corpus_for_tokenizer(
    corpus_path: Path,
    output_text_file: Path,
    languages: tuple = LANGUAGES,
    per_lang_token_budget: int = PER_LANG_TOKEN_BUDGET,
    seed: int = 42,
) -> dict[str, int]:
    """
    Sample text from the unified corpus by language, writing to a flat text file
    for SentencePiece training. Returns words-written per language.
    """
    rng = random.Random(seed)
    lang_buckets: dict[str, list[str]] = {lang: [] for lang in languages}
    lang_words: dict[str, int] = {lang: 0 for lang in languages}

    print(f"  Sampling tokenizer training corpus from {corpus_path}")
    for rec in stream_jsonl(corpus_path):
        lang = rec.get("language") or rec.get("lang")
        if lang not in languages:
            continue
        if lang_words[lang] >= per_lang_token_budget:
            continue
        text = rec.get("text", "")
        if not text:
            continue
        word_count = len(text.split())
        lang_buckets[lang].append(text)
        lang_words[lang] += word_count

    # Shuffle within each language to mix sources
    for lang in languages:
        rng.shuffle(lang_buckets[lang])

    # Write interleaved
    with output_text_file.open("w", encoding="utf-8") as f:
        all_text: list[str] = []
        for lang in languages:
            all_text.extend(lang_buckets[lang])
        rng.shuffle(all_text)
        for line in all_text:
            f.write(line.replace("\n", " ") + "\n")

    print(f"  Sampled words: {lang_words}")
    return lang_words


def train(
    corpus_path: Path,
    output_dir: Path,
    target_step: int = 1000,
    vocab_size_extension: int = 8000,
) -> Path:
    """
    Train Strategy A: SentencePiece extension of LLaMA tokenizer.

    Returns: path to the output tokenizer directory.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # ----- Step 1: sample tokenizer corpus -----
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, dir=output_dir
    ) as f:
        text_file = Path(f.name)

    try:
        sample_corpus_for_tokenizer(
            corpus_path=corpus_path,
            output_text_file=text_file,
            languages=LANGUAGES,
            per_lang_token_budget=PER_LANG_TOKEN_BUDGET,
        )

        # ----- Step 2: train a SentencePiece BPE on the sampled text -----
        import sentencepiece as spm
        sp_model_prefix = output_dir / "lugha_extension"
        spm_args = {
            "input": str(text_file),
            "model_prefix": str(sp_model_prefix),
            "vocab_size": vocab_size_extension,
            "model_type": "bpe",
            "character_coverage": 0.9995,
            "input_sentence_size": 0,    # use all
            "shuffle_input_sentence": True,
            "byte_fallback": True,        # match LLaMA's BPE
            "split_digits": True,
            "allow_whitespace_only_pieces": True,
            "remove_extra_whitespaces": False,
            "normalization_rule_name": "identity",
            "pad_id": -1,
            "unk_id": 0,
            "bos_id": -1,
            "eos_id": -1,
            "num_threads": 16,
        }
        print(f"  Training SentencePiece BPE: vocab_size={vocab_size_extension}")
        spm.SentencePieceTrainer.train(**spm_args)

        # ----- Step 3: load LLaMA tokenizer and extend with new tokens -----
        from transformers import AutoTokenizer
        print(f"  Loading LLaMA base tokenizer from {LLAMA_BASE_PATH}")
        base_tokenizer = AutoTokenizer.from_pretrained(LLAMA_BASE_PATH)
        existing_vocab = set(base_tokenizer.get_vocab().keys())

        # Read SentencePiece vocab
        sp_processor = spm.SentencePieceProcessor()
        sp_processor.load(str(sp_model_prefix) + ".model")
        sp_pieces = [sp_processor.id_to_piece(i) for i in range(sp_processor.get_piece_size())]

        # Identify new tokens not already in LLaMA
        new_tokens = []
        for piece in sp_pieces:
            # SentencePiece uses "▁" (U+2581) for word boundaries; convert to LLaMA "Ġ"
            llama_form = piece.replace("▁", "Ġ")
            if llama_form not in existing_vocab and piece not in existing_vocab:
                new_tokens.append(llama_form)
        print(f"  New tokens to add: {len(new_tokens)} (target {vocab_size_extension})")

        # Add to LLaMA tokenizer
        num_added = base_tokenizer.add_tokens(new_tokens)
        print(f"  Added {num_added} tokens to LLaMA tokenizer "
              f"(new total vocab: {len(base_tokenizer)})")

        # Save extended tokenizer
        base_tokenizer.save_pretrained(output_dir)

        # Audit log: which tokens were added
        with (output_dir / "lughagen_added_tokens_audit.json").open("w") as f:
            json.dump({"added_tokens": new_tokens, "n_added": num_added}, f, indent=2)

        # ----- Step 4: write per-step fertility log -----
        # NOTE: Strategy A is single-shot tokenizer training (no iterative loop).
        # We synthesise a 'fertility curve' by re-evaluating at sampled
        # checkpoints during training. For now, write only the final point as
        # a single-entry log; per-step monitoring is a future extension.
        val_path = Path("/workspace/data/corpus/splits/validation.jsonl")
        if val_path.exists():
            final_fert = evaluate_fertility(output_dir, val_path, LANGUAGES)
            with (output_dir / "step_fertility.jsonl").open("w") as f:
                f.write(json.dumps({"step": target_step, "fertility": final_fert}) + "\n")
        print(f"  Strategy A complete. Output: {output_dir}")
        return output_dir

    finally:
        text_file.unlink(missing_ok=True)


def evaluate_fertility(
    tokenizer_path: Path,
    val_path: Path,
    languages: tuple = LANGUAGES,
    max_samples_per_lang: int = 5000,
) -> dict[str, float]:
    """
    Compute average tokens-per-word on the validation set, per language.

    Fertility := mean(tokens / words). Lower is better — closer to 1.0 means
    the tokenizer is well-matched to the language.
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    counts = {lang: {"tokens": 0, "words": 0, "n_samples": 0} for lang in languages}
    for rec in stream_jsonl(val_path):
        lang = rec.get("language") or rec.get("lang")
        if lang not in languages or counts[lang]["n_samples"] >= max_samples_per_lang:
            continue
        text = rec.get("text", "")
        if not text:
            continue
        n_words = len(text.split())
        if n_words == 0:
            continue
        encoded = tokenizer(text, add_special_tokens=False)
        n_tokens = len(encoded["input_ids"])
        counts[lang]["tokens"] += n_tokens
        counts[lang]["words"] += n_words
        counts[lang]["n_samples"] += 1

    return {
        lang: counts[lang]["tokens"] / max(counts[lang]["words"], 1)
        for lang in languages
    }


# ----- CLI for standalone testing -----
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Strategy A: SentencePiece extension")
    p.add_argument("--corpus-path", type=Path, required=True)
    p.add_argument("--val-path", type=Path,
                   default=Path("/workspace/data/corpus/splits/validation.jsonl"))
    p.add_argument("--output-dir", type=Path,
                   default=Path("/workspace/data/processed/tokenizers/strategy_a"))
    p.add_argument("--vocab-extension", type=int, default=8000)
    p.add_argument("--target-step", type=int, default=1000)
    args = p.parse_args()

    tokenizer_dir = train(
        corpus_path=args.corpus_path,
        output_dir=args.output_dir,
        target_step=args.target_step,
        vocab_size_extension=args.vocab_extension,
    )
    fert = evaluate_fertility(tokenizer_dir, args.val_path, LANGUAGES)
    print(f"\nFinal fertility: {fert}")
