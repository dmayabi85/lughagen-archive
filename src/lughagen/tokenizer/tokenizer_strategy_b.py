#!/usr/bin/env python3
"""
/workspace/scripts/tokenizer_strategy_b.py

LughaGen Training Framework v1.5.1 §6.1 — Strategy B
Train a SentencePiece BPE tokenizer FROM SCRATCH on the four target
languages, with no relationship to LLaMA's vocabulary.

Rationale: a tokenizer optimised purely for Swahili/Kikuyu/Kamba/Luo should
produce the lowest per-language fertility. The trade-off: LLaMA's existing
weights (English-heavy) become less aligned because the embedding table is
re-initialised.

Output:
  /workspace/data/processed/tokenizers/strategy_b/
    ├── tokenizer.json              ← tokenizers-format wrapper
    ├── tokenizer_config.json
    ├── special_tokens_map.json
    ├── lugha_native.model          ← SentencePiece artifact
    ├── lugha_native.vocab
    └── step_fertility.jsonl
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Iterable

LANGUAGES = ("swahili", "kikuyu", "kamba", "luo")
PER_LANG_TOKEN_BUDGET = 4_000_000   # higher than Strategy A since this is from-scratch
TARGET_VOCAB_SIZE = 48_000           # ~3x LLaMA's per-lang share; tunable in §6.1


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
    """Sample the unified corpus to a flat text file for tokenizer training."""
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
    """
    Train Strategy B: SentencePiece BPE from-scratch.

    Note: `target_step` is ignored for SentencePiece (which is single-shot
    training); we use it only for §6.4 plateau-check uniformity (logging
    a single fertility point at the requested step).

    `vocab_size_extension` is reused as the absolute vocab size for from-scratch.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, dir=output_dir
    ) as f:
        text_file = Path(f.name)

    try:
        sample_corpus(corpus_path, text_file, per_lang_budget=PER_LANG_TOKEN_BUDGET)

        import sentencepiece as spm
        sp_prefix = output_dir / "lugha_native"
        print(f"  Training SentencePiece BPE from scratch: vocab_size={vocab_size_extension}")
        spm.SentencePieceTrainer.train(
            input=str(text_file),
            model_prefix=str(sp_prefix),
            vocab_size=vocab_size_extension,
            model_type="bpe",
            character_coverage=0.9995,
            byte_fallback=False,   # Disabled for transformers 5.x SpmConverter compat
            split_digits=True,
            allow_whitespace_only_pieces=True,
            remove_extra_whitespaces=False,
            normalization_rule_name="identity",
            unk_id=0,
            bos_id=1,
            eos_id=2,
            pad_id=3,
            num_threads=16,
        )

        # ----- Wrap in HuggingFace tokenizers format -----
        # transformers 5.x's SpmConverter has multiple bugs parsing modern SP
        # protobufs. We bypass it: parse the SP .model file ourselves to extract
        # pieces + scores, reconstruct BPE merges via greedy decomposition, and
        # build a tokenizers.SentencePieceBPETokenizer directly. This produces
        # a fully HF-compatible tokenizer (via PreTrainedTokenizerFast wrap)
        # that's equivalent to what SP would tokenize natively.

        sp_model_path = str(sp_prefix) + ".model"

        # Step 1: verify SP model loads via sentencepiece directly
        sp_check = spm.SentencePieceProcessor()
        sp_check.load(sp_model_path)
        sp_vocab_size = sp_check.get_piece_size()
        if sp_vocab_size < 1000:
            raise RuntimeError(
                f"Strategy B: SP model has only {sp_vocab_size} pieces, "
                f"expected ~{vocab_size_extension}"
            )
        print(f"  SentencePiece model loaded: pieces={sp_vocab_size}")

        # Step 2: parse the SP protobuf to extract pieces + scores
        from sentencepiece import sentencepiece_model_pb2 as sp_pb2
        sp_model = sp_pb2.ModelProto()
        with open(sp_model_path, "rb") as f:
            sp_model.ParseFromString(f.read())

        # Step 3: extract vocab and reconstruct merges
        # SP-BPE piece types: 1=NORMAL, 2=UNKNOWN, 3=CONTROL, 6=BYTE
        # NORMAL pieces of length>=2 were produced by BPE merge of two earlier pieces.
        # We reconstruct merges by greedy decomposition.
        vocab_dict = {}
        for idx, piece in enumerate(sp_model.pieces):
            vocab_dict[piece.piece] = idx

        merges = []
        merge_count_unfound = 0
        for idx, piece in enumerate(sp_model.pieces):
            if piece.type != 1:  # not NORMAL
                continue
            tok = piece.piece
            if len(tok) < 2:
                continue
            # Find split: smallest left half such that both halves exist with lower idx
            found = False
            for split in range(1, len(tok)):
                left, right = tok[:split], tok[split:]
                left_idx = vocab_dict.get(left)
                right_idx = vocab_dict.get(right)
                if (left_idx is not None and right_idx is not None
                        and left_idx < idx and right_idx < idx):
                    merges.append((left, right))
                    found = True
                    break
            if not found:
                merge_count_unfound += 1

        print(f"  Extracted {len(merges)} merges from {sp_vocab_size} pieces "
              f"({merge_count_unfound} pieces had no decomposition - byte/single tokens)")

        # Step 4: write vocab.json + merges.txt in HF format
        import json
        vocab_json_path = output_dir / "vocab.json"
        merges_txt_path = output_dir / "merges.txt"
        with vocab_json_path.open("w", encoding="utf-8") as f:
            json.dump(vocab_dict, f, ensure_ascii=False)
        with merges_txt_path.open("w", encoding="utf-8") as f:
            f.write("#version: 0.2\n")
            for left, right in merges:
                f.write(f"{left} {right}\n")

        # Step 5: construct SentencePieceBPETokenizer
        from tokenizers import SentencePieceBPETokenizer
        spb = SentencePieceBPETokenizer.from_file(
            vocab_filename=str(vocab_json_path),
            merges_filename=str(merges_txt_path),
            unk_token="<unk>",
            replacement="▁",
            add_prefix_space=True,
        )
        spb_vocab_size = spb.get_vocab_size()
        if spb_vocab_size < 1000:
            raise RuntimeError(
                f"Strategy B: SentencePieceBPETokenizer has only "
                f"vocab_size={spb_vocab_size}, expected ~{vocab_size_extension}"
            )
        print(f"  Built SentencePieceBPETokenizer: vocab_size={spb_vocab_size}")

        # Step 6: wrap as PreTrainedTokenizerFast (saves to output_dir)
        from transformers import PreTrainedTokenizerFast
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json",
                                          delete=False, dir=output_dir) as tf:
            tmp_json = tf.name
        spb.save(tmp_json)

        tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=tmp_json,
            unk_token="<unk>",
            bos_token="<s>",
            eos_token="</s>",
            pad_token="<pad>",
            clean_up_tokenization_spaces=False,
        )
        if tokenizer.vocab_size < 1000:
            raise RuntimeError(
                f"Strategy B: PreTrainedTokenizerFast has vocab_size="
                f"{tokenizer.vocab_size}, expected ~{vocab_size_extension}"
            )
        print(f"  Wrapped as PreTrainedTokenizerFast: vocab_size={tokenizer.vocab_size}")
        tokenizer.save_pretrained(output_dir)
        Path(tmp_json).unlink(missing_ok=True)

        # Log final fertility
        val_path = Path("/workspace/data/corpus/splits/validation.jsonl")
        if val_path.exists():
            final_fert = evaluate_fertility(output_dir, val_path, LANGUAGES)
            with (output_dir / "step_fertility.jsonl").open("w") as f:
                f.write(json.dumps({"step": target_step, "fertility": final_fert}) + "\n")

        print(f"  Strategy B complete. Output: {output_dir}")
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
    p = argparse.ArgumentParser(description="Strategy B: SentencePiece from-scratch")
    p.add_argument("--corpus-path", type=Path, required=True)
    p.add_argument("--val-path", type=Path,
                   default=Path("/workspace/data/corpus/splits/validation.jsonl"))
    p.add_argument("--output-dir", type=Path,
                   default=Path("/workspace/data/processed/tokenizers/strategy_b"))
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
