#!/usr/bin/env python3
"""
Extended categorization: domain distribution, code-switch intensity,
augmentation/preprocessing tags, quality/fluency histograms.

Complements corpus_stats_per_language.py.
"""
import json
from collections import Counter, defaultdict
from pathlib import Path

CORPUS = Path("/workspace/data/corpus/splits_v2/pathD_v2_filtered/train.jsonl")
OUT_MD = Path("/workspace/results/corpus_stats_extended.md")

total = 0
domain_by_lang = defaultdict(Counter)
cs_ratio_by_lang = defaultdict(list)
quality_by_lang = defaultdict(list)
fluency_by_lang = defaultdict(list)
aug_by_lang = defaultdict(Counter)
prep_by_lang = defaultdict(Counter)
lang_totals = Counter()
license_counter = Counter()

print(f"[read] {CORPUS}")
with open(CORPUS) as f:
    for line in f:
        r = json.loads(line)
        total += 1
        lang = r.get("language", "UNKNOWN")
        lang_totals[lang] += 1

        domain_by_lang[lang][r.get("domain", "UNKNOWN")] += 1

        csr = r.get("code_switch_ratio")
        if csr is not None:
            cs_ratio_by_lang[lang].append(csr)

        q = r.get("quality_score")
        if q is not None:
            quality_by_lang[lang].append(q)

        f_ = r.get("fluency_score")
        if f_ is not None:
            fluency_by_lang[lang].append(f_)

        aug = r.get("augmentation") or []
        prep = r.get("preprocessing") or []
        for a in aug:
            aug_by_lang[lang][a] += 1
        for p in prep:
            prep_by_lang[lang][p] += 1

        license_counter[r.get("license", "UNKNOWN")] += 1

        if total % 200000 == 0:
            print(f"  {total:,} records processed...")
print(f"[done] {total:,} records read")

def pct(n, d): return f"{100*n/d:.2f}%" if d else "0%"
def hist(vals, buckets=(0, 0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 1.01)):
    """Count values into buckets [b0,b1), [b1,b2), ..."""
    if not vals: return {}
    out = {}
    for i in range(len(buckets)-1):
        lo, hi = buckets[i], buckets[i+1]
        c = sum(1 for v in vals if lo <= v < hi)
        out[f"[{lo:.2f}, {hi:.2f})"] = c
    return out

langs_sorted = sorted(lang_totals, key=lambda k: -lang_totals[k])

lines = ["# Corpus Extended Categorization: pathD_v2_filtered", "",
         f"Total records: **{total:,}**", "",
         "## Domain distribution per language", ""]
for lang in langs_sorted:
    lines.append(f"### {lang} (n={lang_totals[lang]:,})")
    lines.append("")
    lines.append("| Domain | Count | Share |")
    lines.append("|---|---:|---:|")
    for d, c in domain_by_lang[lang].most_common():
        lines.append(f"| {d} | {c:,} | {pct(c, lang_totals[lang])} |")
    lines.append("")

lines += ["## Code-switch ratio distribution",
          "(For records tagged code_switched, how English-mixed is the text?)", ""]
for lang in langs_sorted:
    vals = cs_ratio_by_lang[lang]
    nonzero = [v for v in vals if v > 0]
    if not nonzero:
        continue
    h = hist(nonzero)
    lines.append(f"### {lang} — {len(nonzero):,} records with code_switch_ratio > 0")
    lines.append("")
    lines.append("| Ratio bucket | Count |")
    lines.append("|---|---:|")
    for k, v in h.items():
        lines.append(f"| {k} | {v:,} |")
    lines.append("")

lines += ["## Quality score coverage per language", "",
          "| Language | Records with quality_score | Mean | Min | Max |",
          "|---|---:|---:|---:|---:|"]
for lang in langs_sorted:
    vals = quality_by_lang[lang]
    if not vals:
        lines.append(f"| {lang} | 0 | - | - | - |")
    else:
        lines.append(f"| {lang} | {len(vals):,} | {sum(vals)/len(vals):.3f} | "
                     f"{min(vals):.3f} | {max(vals):.3f} |")

lines += ["", "## Fluency score coverage per language", "",
          "(nulls indicate Tier 3 fluency scoring was not applied to this record)", "",
          "| Language | With fluency_score | Without | % scored |",
          "|---|---:|---:|---:|"]
for lang in langs_sorted:
    scored = len(fluency_by_lang[lang])
    unscored = lang_totals[lang] - scored
    lines.append(f"| {lang} | {scored:,} | {unscored:,} | {pct(scored, lang_totals[lang])} |")

lines += ["", "## Augmentation tags per language", ""]
for lang in langs_sorted:
    if not aug_by_lang[lang]:
        continue
    lines.append(f"### {lang}")
    lines.append("")
    lines.append("| Tag | Count |")
    lines.append("|---|---:|")
    for tag, c in aug_by_lang[lang].most_common(10):
        lines.append(f"| {tag} | {c:,} |")
    lines.append("")

lines += ["## Preprocessing tags per language", ""]
for lang in langs_sorted:
    if not prep_by_lang[lang]:
        continue
    lines.append(f"### {lang}")
    lines.append("")
    lines.append("| Tag | Count |")
    lines.append("|---|---:|")
    for tag, c in prep_by_lang[lang].most_common(10):
        lines.append(f"| {tag} | {c:,} |")
    lines.append("")

lines += ["## License distribution", "",
          "| License | Count | Share |",
          "|---|---:|---:|"]
for lic, c in license_counter.most_common():
    lines.append(f"| {lic} | {c:,} | {pct(c, total)} |")

OUT_MD.write_text("\n".join(lines))
print(f"[wrote] {OUT_MD}")

# Console summary
print()
print("=" * 60)
print("QUICK SUMMARY")
print("=" * 60)
for lang in langs_sorted:
    print(f"\n{lang} ({lang_totals[lang]:,} records)")
    print(f"  domains: {dict(domain_by_lang[lang].most_common(3))}")
    print(f"  code_switch>0: {sum(1 for v in cs_ratio_by_lang[lang] if v>0):,}")
    print(f"  fluency scored: {len(fluency_by_lang[lang]):,}")
    print(f"  augmentations: {sum(aug_by_lang[lang].values()):,}")
