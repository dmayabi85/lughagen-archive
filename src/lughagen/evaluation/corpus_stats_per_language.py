#!/usr/bin/env python3
"""
Categorize the filtered corpus by language, origin, and register.

Reads /workspace/data/corpus/splits_v2/pathD_v2_filtered/train.jsonl and
produces a stratified breakdown suitable for a thesis data card.

Outputs:
  - /workspace/results/corpus_stats_pathD_v2_filtered.json
  - /workspace/results/corpus_stats_pathD_v2_filtered.md (human-readable)
"""
from __future__ import annotations
import json
from collections import Counter, defaultdict
from pathlib import Path

CORPUS = Path("/workspace/data/corpus/splits_v2/pathD_v2_filtered/train.jsonl")
OUT_JSON = Path("/workspace/results/corpus_stats_pathD_v2_filtered.json")
OUT_MD = Path("/workspace/results/corpus_stats_pathD_v2_filtered.md")

def word_count(text: str) -> int:
    return len(text.split()) if text else 0

# Counters
total = 0
lang_counter: Counter[str] = Counter()
origin_counter: Counter[str] = Counter()
register_counter: Counter[str] = Counter()
source_counter: Counter[str] = Counter()

# Cross-tabs
lang_origin: dict[tuple[str, str], int] = defaultdict(int)
lang_register: dict[tuple[str, str], int] = defaultdict(int)
lang_source: dict[tuple[str, str], int] = defaultdict(int)

# Per-language stats
lang_words: dict[str, int] = defaultdict(int)
lang_records: dict[str, int] = defaultdict(int)
lang_lengths: dict[str, list[int]] = defaultdict(list)

print(f"[read] {CORPUS}")
with open(CORPUS) as f:
    for line in f:
        r = json.loads(line)
        total += 1

        lang = r.get("language", "UNKNOWN")
        origin = r.get("origin", "UNKNOWN")
        register = r.get("register", "UNKNOWN")
        source = r.get("source_id", r.get("source", "UNKNOWN"))
        text = r.get("text", "")
        wc = r.get("words") or word_count(text)

        lang_counter[lang] += 1
        origin_counter[origin] += 1
        register_counter[register] += 1
        source_counter[source] += 1

        lang_origin[(lang, origin)] += 1
        lang_register[(lang, register)] += 1
        lang_source[(lang, source)] += 1

        lang_records[lang] += 1
        lang_words[lang] += wc
        # Sample lengths (reservoir would be safer but this is fine at ~1.4M)
        if len(lang_lengths[lang]) < 10000:
            lang_lengths[lang].append(wc)

        if total % 200000 == 0:
            print(f"  {total:,} records processed...")

print(f"[done] {total:,} records read")

# ---------- Build stats ----------
def pct(n: int, d: int) -> str:
    return f"{100*n/d:.2f}%" if d else "0%"

per_lang = {}
for lang in sorted(lang_counter, key=lambda k: -lang_counter[k]):
    ls = sorted(lang_lengths[lang])
    n = lang_records[lang]
    stats = {
        "records": n,
        "share_of_corpus": pct(n, total),
        "total_words": lang_words[lang],
        "avg_words_per_record": round(lang_words[lang] / n, 1) if n else 0,
        "length_pctiles_words": {
            "p10": ls[len(ls)*10//100] if ls else None,
            "p50": ls[len(ls)//2] if ls else None,
            "p90": ls[len(ls)*90//100] if ls else None,
            "p99": ls[len(ls)*99//100] if ls else None,
        } if ls else {},
        "by_origin": {},
        "by_register": {},
        "top_sources": [],
    }
    for (l, o), c in lang_origin.items():
        if l == lang:
            stats["by_origin"][o] = {"count": c, "share": pct(c, n)}
    for (l, reg), c in lang_register.items():
        if l == lang:
            stats["by_register"][reg] = {"count": c, "share": pct(c, n)}
    lang_srcs = [(s, c) for (l, s), c in lang_source.items() if l == lang]
    lang_srcs.sort(key=lambda t: -t[1])
    stats["top_sources"] = [{"source": s, "count": c, "share": pct(c, n)}
                            for s, c in lang_srcs[:8]]
    per_lang[lang] = stats

result = {
    "corpus_path": str(CORPUS),
    "total_records": total,
    "totals": {
        "by_language": {k: {"count": v, "share": pct(v, total)}
                        for k, v in lang_counter.most_common()},
        "by_origin": {k: {"count": v, "share": pct(v, total)}
                      for k, v in origin_counter.most_common()},
        "by_register": {k: {"count": v, "share": pct(v, total)}
                        for k, v in register_counter.most_common()},
    },
    "per_language": per_lang,
}

OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
OUT_JSON.write_text(json.dumps(result, indent=2))
print(f"[wrote] {OUT_JSON}")

# ---------- Human-readable Markdown ----------
lines = [
    "# Corpus Composition: pathD_v2_filtered",
    "",
    f"Path: `{CORPUS}`  ",
    f"Total records: **{total:,}**",
    "",
    "## By language",
    "",
    "| Language | Records | Share | Total words | Avg words/record |",
    "|---|---:|---:|---:|---:|",
]
for lang, ls in per_lang.items():
    lines.append(f"| {lang} | {ls['records']:,} | {ls['share_of_corpus']} | "
                 f"{ls['total_words']:,} | {ls['avg_words_per_record']} |")

lines += ["", "## Origin split", "",
          "| Origin | Count | Share |",
          "|---|---:|---:|"]
for k, v in origin_counter.most_common():
    lines.append(f"| {k} | {v:,} | {pct(v, total)} |")

lines += ["", "## Register split", "",
          "| Register | Count | Share |",
          "|---|---:|---:|"]
for k, v in register_counter.most_common():
    lines.append(f"| {k} | {v:,} | {pct(v, total)} |")

lines += ["", "## Per-language × origin",
          "", "| Language | Authentic | Synthetic | Other |",
          "|---|---:|---:|---:|"]
langs_sorted = sorted(lang_counter, key=lambda k: -lang_counter[k])
origins_seen = list(origin_counter)
for lang in langs_sorted:
    row = [f"| {lang} |"]
    for o in ["authentic", "synthetic"]:
        c = lang_origin.get((lang, o), 0)
        row.append(f" {c:,} ({pct(c, lang_records[lang])}) |")
    others = sum(c for (l, o), c in lang_origin.items()
                 if l == lang and o not in ("authentic", "synthetic"))
    row.append(f" {others:,} |")
    lines.append("".join(row))

lines += ["", "## Per-language × register",
          "", "| Language | Monolingual | Code-switched | Other |",
          "|---|---:|---:|---:|"]
for lang in langs_sorted:
    row = [f"| {lang} |"]
    for reg in ["monolingual", "code_switched"]:
        c = lang_register.get((lang, reg), 0)
        row.append(f" {c:,} ({pct(c, lang_records[lang])}) |")
    others = sum(c for (l, r), c in lang_register.items()
                 if l == lang and r not in ("monolingual", "code_switched"))
    row.append(f" {others:,} |")
    lines.append("".join(row))

lines += ["", "## Per-language length distribution (words)",
          "", "| Language | p10 | p50 | p90 | p99 |",
          "|---|---:|---:|---:|---:|"]
for lang in langs_sorted:
    p = per_lang[lang]["length_pctiles_words"]
    lines.append(f"| {lang} | {p.get('p10')} | {p.get('p50')} | "
                 f"{p.get('p90')} | {p.get('p99')} |")

lines += ["", "## Top sources per language",
          ""]
for lang in langs_sorted:
    lines.append(f"### {lang}")
    lines.append("")
    lines.append("| Source | Count | Share of language |")
    lines.append("|---|---:|---:|")
    for src in per_lang[lang]["top_sources"]:
        lines.append(f"| {src['source']} | {src['count']:,} | {src['share']} |")
    lines.append("")

OUT_MD.write_text("\n".join(lines))
print(f"[wrote] {OUT_MD}")

# ---------- Console summary ----------
print()
print("=" * 60)
print("SUMMARY: per-language record counts")
print("=" * 60)
for lang, count in lang_counter.most_common():
    print(f"  {lang:20s}  {count:>10,}  {pct(count, total)}")
