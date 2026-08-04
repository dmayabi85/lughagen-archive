#!/usr/bin/env python3
"""
/workspace/scripts/tokenizer_strategy_c.py

LughaGen Training Framework v1.5.1 §6.1 — Strategy C
Train a Byte-Pair Encoding (BPE) tokenizer FROM SCRATCH using the HuggingFace
`tokenizers` library (not SentencePiece).

Rationale: differs from Strategy B in tokenizer implementation (HF tokenizers
library uses ByteLevelBPE which matches GPT-style preprocessing exactly).
This is the standard tokenizer for Llama-family models in newer releases.

Output:
  /workspace/data/processed/tokenizers/strategy_c/
    ├── tokenizer.json              ← HF tokenizers native format
    ├── tokenizer_config.json
    ├── special_tokens_map.json
    └── step_fertility.jsonl
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Iterable

LANGUAGES = ("swahili", "kikuyu", "kamba", "luo")
PER_LANG_TOKEN_BUDGET = 4_000_000
TARGET_VOCAB_SIZE = 48_000


def stream_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def sample_corpus(
    corpus_path: Path,
    output_file: Path,
    per_lang_budget: int = PER_LANG_TOKEN_BUDGET,
    seed: int = 42,
) -> dict[str, int]:
    """Sample corpus to a flat text file."""
    import random
    rng = random.Random(seed)
    buckets: dict[str, list[str]] = {lang: [] for lang in LANGUAGES}
    word_counts: dict[str, int] = {lang: 0 for lang in LANGUAGES}

    print(f"  Sampling {per_lang_budget:,} words/language from {corpus_path}")
    for rec in stream_jsonl(corpus_path):
        lang = rec.get("language") or rec.get("lang")
        if lang not in LANGUAGES or word_counts[lang] >= per_lang_budget:
            continue
        text = rec.get("text", "")
        if not text:
            continue
        buckets[lang].append(text)
        word_counts[lang] += len(text.split())

    with output_file.open("w", encoding="utf-8") as f:
        all_lines = []
        for lang in LANGUAGES:
            all_lines.extend(buckets[lang])
        rng.shuffle(all_lines)
        for line in all_lines:
            f.write(line.replace("\n", " ") + "\n")

    print(f"  Words sampled: {word_counts}")
    return word_counts


def train(
    corpus_path: Path,
    output_dir: Path,
    target_step: int = 1000,
    vocab_size_extension: int = TARGET_VOCAB_SIZE,
) -> Path:
    """Train Strategy C: ByteLevelBPE from-scratch via HuggingFace tokenizers."""
    output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, dir=output_dir
    ) as f:
        text_file = Path(f.name)

    try:
        sample_corpus(corpus_path, text_file, per_lang_budget=PER_LANG_TOKEN_BUDGET)

        from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
        from tokenizers.processors import TemplateProcessing

        # ----- Construct a ByteLevelBPE tokenizer -----
        tokenizer = Tokenizer(models.BPE())
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
        tokenizer.decoder = decoders.ByteLevel()

        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size_extension,
            min_frequency=2,
            special_tokens=["<s>", "</s>", "<unk>", "<pad>"],
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            show_progress=True,
        )

        print(f"  Training ByteLevelBPE from scratch: vocab_size={vocab_size_extension}")
        tokenizer.train(files=[str(text_file)], trainer=trainer)

        # Post-processor adds BOS/EOS tokens for causal LM training
        tokenizer.post_processor = TemplateProcessing(
            single="<s> $A </s>",
            special_tokens=[
                ("<s>",  tokenizer.token_to_id("<s>")),
                ("</s>", tokenizer.token_to_id("</s>")),
            ],
        )

        # Save the raw tokenizer
        tokenizer_json_path = output_dir / "tokenizer.json"
        tokenizer.save(str(tokenizer_json_path))

        # ----- Wrap as HuggingFace PreTrainedTokenizerFast -----
        from transformers import PreTrainedTokenizerFast
        hf_tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=str(tokenizer_json_path),
            unk_token="<unk>",
            bos_token="<s>",
            eos_token="</s>",
            pad_token="<pad>",
            clean_up_tokenization_spaces=False,
        )
        hf_tokenizer.save_pretrained(output_dir)

        # Final fertility log
        val_path = Path("/workspace/data/corpus/splits/validation.jsonl")
        if val_path.exists():
            final_fert = evaluate_fertility(output_dir, val_path, LANGUAGES)
            with (output_dir / "step_fertility.jsonl").open("w") as f:
                f.write(json.dumps({"step": target_step, "fertility": final_fert}) + "\n")

        print(f"  Strategy C complete. Output: {output_dir}")
        return output_dir

    finally:
        text_file.unlink(missing_ok=True)


def evaluate_fertility(
    tokenizer_path: Path,
    val_path: Path,
    languages: tuple = LANGUAGES,
    max_samples_per_lang: int = 5000,
) -> dict[str, float]:
    """Compute tokens-per-word per language on the validation set."""
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    counts = {lang: {"tokens": 0, "words": 0, "n": 0} for lang in languages}
    for rec in stream_jsonl(val_path):
        lang = rec.get("language") or rec.get("lang")
        if lang not in languages or counts[lang]["n"] >= max_samples_per_lang:
            continue
        text = rec.get("text", "")
        if not text:
            continue
        n_words = len(text.split())
        if n_words == 0:
            continue
        enc = tokenizer(text, add_special_tokens=False)
        counts[lang]["tokens"] += len(enc["input_ids"])
        counts[lang]["words"] += n_words
        counts[lang]["n"] += 1

    return {
        lang: counts[lang]["tokens"] / max(counts[lang]["words"], 1)
        for lang in languages
    }


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Strategy C: ByteLevelBPE from-scratch")
    p.add_argument("--corpus-path", type=Path, required=True)
    p.add_argument("--val-path", type=Path,
                   default=Path("/workspace/data/corpus/splits/validation.jsonl"))
    p.add_argument("--output-dir", type=Path,
                   default=Path("/workspace/data/processed/tokenizers/strategy_c"))
    p.add_argument("--vocab-size", type=int, default=TARGET_VOCAB_SIZE)
    p.add_argument("--target-step", type=int, default=1000)
    args = p.parse_args()

    tokenizer_dir = train(
        corpus_path=args.corpus_path,
        output_dir=args.output_dir,
        target_step=args.target_step,
        vocab_size_extension=args.vocab_size,
    )
    fert = evaluate_fertility(tokenizer_dir, args.val_path, LANGUAGES)
    print(f"\nFinal fertility: {fert}")
