"""
recipe_loader.py
─────────────────────────────────────────────────────────────────────────
Filter the unified corpus into a training-ready subset based on a named
"recipe". The same physical splits/train.jsonl backs many experiments,
the recipe just defines which records to include.

Recipes are JSON-defined. A recipe is one of:
    1. "all": no filtering
    2. {filter dict}: include only records matching all key/value pairs
    3. {"exclude": {...}}: include all records EXCEPT those matching
    4. {"origin_ratio": {"synthetic": 0.7, "authentic": 0.3}}:
       sample to a target origin distribution (downsamples larger pool)
    5. {"and": [filter, ...]}: intersection of sub-filters
    6. {"or": [filter, ...]}: union of sub-filters

The recipe library is read from splits/recipes.json. Example file:

{
  "recipes": {
    "full":             "all",
    "synth_only":       {"origin": "synthetic"},
    "auth_only":        {"origin": "authentic"},
    "mono_only":        {"register": "monolingual"},
    "cs_only":          {"register": "code_switched"},
    "auth_cs_emphasis": {"or": [
      {"register": "monolingual"},
      {"and": [{"register": "code_switched"}, {"origin": "authentic"}]}
    ]},
    "no_synth_cs":      {"exclude": {"register": "code_switched", "origin": "synthetic"}},
    "balanced_70_30":   {"origin_ratio": {"synthetic": 0.70, "authentic": 0.30}}
  }
}

Usage from a training script:

    from recipe_loader import iter_recipe
    for record in iter_recipe(
        recipe_name="auth_cs_emphasis",
        recipes_file="/workspace/data/corpus/splits/recipes.json",
        split_file="/workspace/data/corpus/splits/train.jsonl",
    ):
        # train on record["text"]
        ...

Command-line preview:
    python recipe_loader.py \\
        --recipe auth_cs_emphasis \\
        --recipes-file /workspace/data/corpus/splits/recipes.json \\
        --split-file   /workspace/data/corpus/splits/train.jsonl \\
        --stats-only
"""

import argparse
import json
import random
from collections import Counter
from pathlib import Path


def _match(record, criteria):
    """True if record matches a flat {field: value} criteria dict."""
    for k, v in criteria.items():
        if record.get(k) != v:
            return False
    return True


def _filter_pass(record, recipe):
    """Recursive filter evaluator."""
    if recipe == "all":
        return True
    if isinstance(recipe, dict):
        if "exclude" in recipe:
            return not _match(record, recipe["exclude"])
        if "and" in recipe:
            return all(_filter_pass(record, sub) for sub in recipe["and"])
        if "or" in recipe:
            return any(_filter_pass(record, sub) for sub in recipe["or"])
        if "origin_ratio" in recipe:
            return True   # ratio handled in iter_recipe (needs two-pass)
        return _match(record, recipe)
    raise ValueError(f"Unknown recipe shape: {recipe!r}")


def iter_recipe(recipe_name, recipes_file, split_file, seed=42):
    """Yield records from split_file that match the named recipe."""
    with open(recipes_file) as f:
        recipes = json.load(f)["recipes"]
    if recipe_name not in recipes:
        raise KeyError(f"Recipe '{recipe_name}' not in {recipes_file}")
    recipe = recipes[recipe_name]

    # Special case: origin_ratio requires a sampling pass
    if isinstance(recipe, dict) and "origin_ratio" in recipe:
        yield from _iter_origin_ratio(recipe["origin_ratio"], split_file, seed)
        return

    with open(split_file) as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if _filter_pass(r, recipe):
                yield r


def _iter_origin_ratio(target_ratios, split_file, seed):
    """Two-pass sampler to hit a target origin distribution by words."""
    rng = random.Random(seed)

    # Pass 1: count words per origin
    pool_words = Counter()
    with open(split_file) as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            pool_words[r.get("origin", "unknown")] += r.get("words", 0)

    # Compute keep-probabilities to hit target ratios
    target_origin = max(target_ratios, key=lambda k: target_ratios[k])
    target_origin_words = pool_words[target_origin]
    target_total = target_origin_words / target_ratios[target_origin]
    keep_prob = {}
    for o, ratio in target_ratios.items():
        want = ratio * target_total
        have = pool_words.get(o, 0)
        keep_prob[o] = min(1.0, want / have) if have else 0.0

    print(f"  origin pool words: {dict(pool_words)}")
    print(f"  target ratios:     {target_ratios}")
    print(f"  keep probability:  {keep_prob}")

    # Pass 2: yield with keep probability
    with open(split_file) as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            p = keep_prob.get(r.get("origin"), 0)
            if rng.random() < p:
                yield r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recipe",       required=True)
    ap.add_argument("--recipes-file", required=True)
    ap.add_argument("--split-file",   required=True)
    ap.add_argument("--output",       help="Write filtered records to JSONL (optional)")
    ap.add_argument("--stats-only",   action="store_true",
                    help="Print composition stats without writing output")
    args = ap.parse_args()

    docs = words = 0
    by_origin = Counter(); by_register = Counter(); by_language = Counter()
    out_f = open(args.output, "w") if args.output else None

    for r in iter_recipe(args.recipe, args.recipes_file, args.split_file):
        docs += 1
        w = r.get("words", 0)
        words += w
        by_origin[r.get("origin", "unknown")]   += w
        by_register[r.get("register", "unknown")] += w
        by_language[r.get("language", "unknown")] += w
        if out_f:
            out_f.write(json.dumps(r, ensure_ascii=False) + "\n")

    if out_f:
        out_f.close()
        print(f"  wrote {docs:,} records to {args.output}")

    print("\n" + "=" * 64)
    print(f"  Recipe: {args.recipe}")
    print("=" * 64)
    print(f"  Documents : {docs:,}")
    print(f"  Words     : {words:,}")
    for label, c in [("By origin", by_origin), ("By register", by_register), ("By language", by_language)]:
        total = sum(c.values())
        print(f"\n  {label}:")
        for k, v in c.most_common():
            print(f"    {k:<18}{v:>12,} words  ({100*v/total:>5.2f}%)")
    print("=" * 64)


if __name__ == "__main__":
    main()
