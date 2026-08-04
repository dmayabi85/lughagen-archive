"""
Tokenizer fertility measurement on actual LughaGen corpus samples.

Reads from /workspace/data/cleaned_final/*.jsonl, samples N sentences per
language, and reports a full distribution (mean, std, quartiles, min, max)
of tokens-per-word ratios under the LLaMA-3.1 tokenizer.

This is the proper measurement to anchor §1.2 / §5.5 / §6.1 fertility claims.

Run inside the Brev container:
    python tokenizer_fertility_corpus.py

Output:
    fertility_corpus_results.csv  — full per-language statistics
    fertility_corpus_sample.csv   — per-sentence values for further analysis
"""

import json
import random
import statistics
from pathlib import Path

from transformers import AutoTokenizer


CORPUS_DIR = Path("/workspace/data/cleaned_final")
LANGUAGES  = ["swahili", "kikuyu", "luo", "kamba"]
SAMPLE_N   = 5000     # sentences per language; reduce if files are smaller
SEED       = 42

# Optional English reference text (one of these will be tried in order)
ENGLISH_FALLBACKS = [
    "/workspace/data/english_reference.jsonl",
    "/workspace/data/cleaned_final/english.jsonl",
]


def load_sentences(jsonl_path: Path, n: int, seed: int = SEED):
    """Reservoir-sample n sentences from a JSONL file with a 'text' field."""
    random.seed(seed)
    reservoir = []
    with open(jsonl_path) as f:
        for i, line in enumerate(f):
            try:
                obj = json.loads(line)
                txt = obj.get("text") or obj.get("content") or obj.get("sentence")
                if not txt or not isinstance(txt, str):
                    continue
                txt = txt.strip()
                if len(txt.split()) < 5:        # skip very short lines
                    continue
                if i < n:
                    reservoir.append(txt)
                else:
                    j = random.randint(0, i)
                    if j < n:
                        reservoir[j] = txt
            except Exception:
                continue
    return reservoir


def per_sentence_fertility(sentences, tok):
    """Return list of (tokens, words, fertility) per sentence."""
    results = []
    for s in sentences:
        toks = tok.encode(s, add_special_tokens=False)
        words = s.split()
        if not words:
            continue
        results.append((len(toks), len(words), len(toks) / len(words)))
    return results


def summarise(results, label):
    fertilities = [r[2] for r in results]
    total_tokens = sum(r[0] for r in results)
    total_words = sum(r[1] for r in results)

    if not fertilities:
        print(f"  [{label}] empty")
        return None

    fertilities.sort()
    n = len(fertilities)
    summary = {
        "language"      : label,
        "n_sentences"   : n,
        "total_tokens"  : total_tokens,
        "total_words"   : total_words,
        "mean"          : statistics.mean(fertilities),
        "stdev"         : statistics.stdev(fertilities) if n > 1 else 0,
        "min"           : fertilities[0],
        "q1"            : fertilities[n // 4],
        "median"        : statistics.median(fertilities),
        "q3"            : fertilities[3 * n // 4],
        "max"           : fertilities[-1],
        "corpus_level"  : total_tokens / total_words,
    }
    return summary


def main():
    print("Loading LLaMA-3.1-8B tokenizer ...")
    tok = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B", use_fast=True)

    all_summaries = []
    all_per_sentence = []

    print(f"\nSampling {SAMPLE_N} sentences per language from {CORPUS_DIR}\n")

    for lang in LANGUAGES:
        # Try a few common filename patterns
        candidates = [
            CORPUS_DIR / f"{lang}.jsonl",
            CORPUS_DIR / f"{lang}_clean.jsonl",
            CORPUS_DIR / f"{lang}_cleaned.jsonl",
        ]
        path = next((c for c in candidates if c.exists()), None)
        if not path:
            print(f"  [{lang}] no corpus file found in {CORPUS_DIR}")
            continue

        sents = load_sentences(path, SAMPLE_N)
        if not sents:
            print(f"  [{lang}] no usable sentences found in {path}")
            continue

        per_sent = per_sentence_fertility(sents, tok)
        s = summarise(per_sent, lang)
        all_summaries.append(s)
        for r in per_sent:
            all_per_sentence.append((lang, r[0], r[1], r[2]))
        print(f"  [{lang:<10}] {s['n_sentences']:>5} sentences  "
              f"mean={s['mean']:.3f}  median={s['median']:.3f}  "
              f"corpus-level={s['corpus_level']:.3f}")

    # English reference
    english_path = next((Path(p) for p in ENGLISH_FALLBACKS if Path(p).exists()), None)
    if english_path:
        sents = load_sentences(english_path, SAMPLE_N)
        if sents:
            per_sent = per_sentence_fertility(sents, tok)
            s = summarise(per_sent, "english")
            all_summaries.append(s)
            for r in per_sent:
                all_per_sentence.append(("english", r[0], r[1], r[2]))
            print(f"  [{'english':<10}] {s['n_sentences']:>5} sentences  "
                  f"mean={s['mean']:.3f}  median={s['median']:.3f}  "
                  f"corpus-level={s['corpus_level']:.3f}")
    else:
        print(f"  [english   ] no reference file at any of {ENGLISH_FALLBACKS}")

    # Print formatted table
    print("\n" + "=" * 96)
    print("  LLaMA-3.1 Tokenizer Fertility — measured on LughaGen cleaned corpus")
    print("=" * 96)
    print(f"\n  {'Language':<10} {'N':>6}  {'Tokens':>10}  {'Words':>10}  "
          f"{'Mean':>7} {'Median':>7} {'Q1':>6} {'Q3':>6} {'Max':>6}  Corpus-level")
    print("  " + "-" * 92)
    for s in all_summaries:
        print(f"  {s['language']:<10} {s['n_sentences']:>6}  "
              f"{s['total_tokens']:>10,}  {s['total_words']:>10,}  "
              f"{s['mean']:>7.3f} {s['median']:>7.3f} "
              f"{s['q1']:>6.3f} {s['q3']:>6.3f} {s['max']:>6.3f}  "
              f"{s['corpus_level']:>7.3f}")

    # Multiplier vs English (if available)
    en = next((s for s in all_summaries if s["language"] == "english"), None)
    if en:
        print("\n  Multiplier vs English (mean fertility ratio):")
        for s in all_summaries:
            if s["language"] == "english":
                continue
            mult = s["mean"] / en["mean"]
            print(f"    {s['language']:<10} {mult:.2f}×")

    # Save CSVs
    import csv
    with open("fertility_corpus_results.csv", "w", newline="") as f:
        if all_summaries:
            w = csv.DictWriter(f, fieldnames=list(all_summaries[0].keys()))
            w.writeheader()
            for s in all_summaries:
                w.writerow(s)

    with open("fertility_corpus_sample.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["language", "tokens", "words", "fertility"])
        for row in all_per_sentence:
            w.writerow(row)

    print("\n  Results saved to fertility_corpus_results.csv (summary)")
    print("                   fertility_corpus_sample.csv  (per-sentence)")
    print("=" * 96)


if __name__ == "__main__":
    main()
