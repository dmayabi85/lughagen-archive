"""
corpus_full_audit.py — full kitchen-sink audit of the LughaGen corpus.

Inputs:
  /workspace/data/corpus/splits/{train,validation,test}.jsonl
  /home/nvidia/synthetic/tier2/{swahili,luo,kikuyu,kamba}_tier2.jsonl

Per-record (every record, every split):
  - GlotLID top-3 + decision (STRICT_PASS / RELAXED_PASS / QUARANTINE)
  - Phase 0 token count (on-the-fly tokenization)
  - Char/word/token length
  - Token fertility (Phase 0 tokens / words)

Aggregations:
  per_language, per_origin, per_register, per_source_id, per_domain, per_license,
  per_split, length_tiers, quality_x_glotlid, fluency_x_glotlid,
  contamination_matrix, code_switch_distribution, token_fertility,
  source_url_topN, synth_vs_authentic, today_vs_prior_synth_audit.

Outputs (reports only, no materialized cleaned files):
  /workspace/logs/corpus_audit_v1/
    corpus_audit_report.json
    corpus_audit_report.md
    per_language.csv
    per_origin_x_language.csv
    per_register.csv
    per_source_id.csv
    per_domain.csv
    per_license.csv
    per_split.csv
    length_tiers.csv
    quality_x_glotlid.csv
    fluency_x_glotlid.csv
    contamination_matrix.csv
    code_switch_distribution.csv
    token_fertility.csv
    source_url_top.csv
"""
import json, time, csv, sys
from pathlib import Path
from collections import Counter, defaultdict
from statistics import mean, median, pstdev

# ── CONFIG ───────────────────────────────────────────────────────
GLOTLID_PATH    = "/home/nvidia/models/glotlid.bin"
TOKENIZER_PATH  = "/workspace/data/processed/tokenizers/strategy_c"
GLOTLID_MIN_P   = 0.50
MIN_TEXT_LEN    = 10
REPORT_DIR      = Path("/workspace/logs/corpus_audit_v1")
REPORT_DIR.mkdir(parents=True, exist_ok=True)
PROGRESS_EVERY  = 100_000

SPLIT_FILES = {
    "train":      "/workspace/data/corpus/splits/train.jsonl",
    "validation": "/workspace/data/corpus/splits/validation.jsonl",
    "test":       "/workspace/data/corpus/splits/test.jsonl",
}
SYNTH_FILES = {
    "swahili": "/home/nvidia/synthetic/tier2/swahili_tier2.jsonl",
    "luo":     "/home/nvidia/synthetic/tier2/luo_tier2.jsonl",
    "kikuyu":  "/home/nvidia/synthetic/tier2/kikuyu_tier2.jsonl",
    "kamba":   "/home/nvidia/synthetic/tier2/kamba_tier2.jsonl",
}
GLOTLID_TARGETS = {
    "swahili": {"strict": {"swh_Latn"},      "relaxed": {"swh_Latn", "swc_Latn"}},
    "luo":     {"strict": {"luo_Latn"},      "relaxed": {"luo_Latn"}},
    "kikuyu":  {"strict": {"kik_Latn"},      "relaxed": {"kik_Latn"}},
    "kamba":   {"strict": {"kam_Latn"},      "relaxed": {"kam_Latn"}},
}

LENGTH_TIERS = [
    ("very_short", 0, 30),
    ("short",     30, 100),
    ("medium",   100, 500),
    ("long",     500, 2000),
    ("very_long", 2000, 10**9),
]

def length_tier(n_chars):
    for name, lo, hi in LENGTH_TIERS:
        if lo <= n_chars < hi:
            return name
    return "very_long"

print(f"[{time.strftime('%H:%M:%S')}] Loading GlotLID...")
import fasttext
gl = fasttext.load_model(GLOTLID_PATH)
print(f"[{time.strftime('%H:%M:%S')}] Loading Phase 0 tokenizer...")
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
print(f"[{time.strftime('%H:%M:%S')}] Ready.\n")

def decide(text, lang):
    cfg = GLOTLID_TARGETS.get(lang)
    if cfg is None:
        return "UNK_LANG", "lang_not_in_targets", None, None, []
    if not text or len(text.strip()) < MIN_TEXT_LEN:
        return "QUARANTINE", "text_too_short", None, None, []
    labels, probs = gl.predict(text.replace("\n", " ").strip(), k=3)
    top3 = [(l.replace('__label__',''), float(p)) for l, p in zip(labels, probs)]
    t1, p1 = top3[0]
    if t1 in cfg["strict"] and p1 >= GLOTLID_MIN_P:
        return "STRICT_PASS", None, t1, p1, top3
    if any(l in cfg["relaxed"] for l, _ in top3[:3]):
        return "RELAXED_PASS", None, t1, p1, top3
    return "QUARANTINE", "glotlid_rejected", t1, p1, top3

def qbucket(q):
    """Quality-score bucket for cross-tabs."""
    if q is None: return "missing"
    if q < 0.5:   return "lt_0.5"
    if q < 0.75:  return "0.5_to_0.75"
    if q < 0.9:   return "0.75_to_0.9"
    if q < 0.95:  return "0.9_to_0.95"
    return "ge_0.95"

# ── ACCUMULATORS ─────────────────────────────────────────────────
agg = {
    "by_split":      defaultdict(lambda: Counter()),
    "by_lang":       defaultdict(lambda: Counter()),
    "by_origin_lang": defaultdict(lambda: Counter()),
    "by_register":   defaultdict(lambda: Counter()),
    "by_source_id":  defaultdict(lambda: Counter()),
    "by_domain":     defaultdict(lambda: Counter()),
    "by_license":    defaultdict(lambda: Counter()),
    "by_length_tier": defaultdict(lambda: Counter()),
    "by_quality":    defaultdict(lambda: Counter()),
    "by_fluency":    defaultdict(lambda: Counter()),
    "contam_by_lang": defaultdict(lambda: Counter()),  # quarantined samples' top1 label per declared-lang
    "fertility_samples": defaultdict(list),  # tokens/word per language
    "len_chars": defaultdict(list),
    "len_words": defaultdict(list),
    "len_tokens": defaultdict(list),
    "code_switch_ratio": defaultdict(list),
    "source_url": defaultdict(Counter),  # top-N source_url per language
}

def update_agg(rec, decision, top1, source_label):
    """source_label is e.g. 'train' or 'synth_swahili' — for split tagging."""
    lang = rec.get("language", "?")
    origin = rec.get("origin", "?")
    register = rec.get("register", "?")
    src = rec.get("source_id", "?")
    domain = rec.get("domain", "?")
    lic = rec.get("license", "?")
    quality = rec.get("quality_score")
    fluency = rec.get("fluency_score")
    text = rec.get("text", "")
    words = rec.get("words") or 0
    url = rec.get("source_url", "?")
    cs_ratio = rec.get("code_switch_ratio")

    agg["by_split"][source_label][decision] += 1
    agg["by_lang"][lang][decision] += 1
    agg["by_origin_lang"][f"{origin}_{lang}"][decision] += 1
    agg["by_register"][register][decision] += 1
    agg["by_source_id"][src][decision] += 1
    agg["by_domain"][domain][decision] += 1
    agg["by_license"][lic][decision] += 1
    agg["by_quality"][qbucket(quality)][decision] += 1
    agg["by_fluency"][qbucket(fluency)][decision] += 1

    n_chars = len(text)
    agg["by_length_tier"][length_tier(n_chars)][decision] += 1
    agg["len_chars"][lang].append(n_chars)
    agg["len_words"][lang].append(words)

    if decision == "QUARANTINE" and top1 is not None:
        agg["contam_by_lang"][lang][top1] += 1

    if cs_ratio is not None:
        agg["code_switch_ratio"][lang].append(cs_ratio)

    if url and url != "?":
        agg["source_url"][lang][url] += 1

# ── MAIN PASS ─────────────────────────────────────────────────────
def process_file(path, label, origin_override=None, is_synthetic=False):
    """Stream one file, classify each record, update aggregators."""
    p = Path(path)
    if not p.exists():
        print(f"  ✗ missing: {path}")
        return 0
    print(f"[{time.strftime('%H:%M:%S')}] processing {label}: {p.name}")
    t0 = time.time()
    n = 0
    n_decisions = Counter()
    with open(p) as f:
        for line in f:
            try:
                rec = json.loads(line)
            except:
                continue
            n += 1
            text = rec.get("text", "")
            lang = rec.get("language", "?")
            if is_synthetic and origin_override:
                rec["origin"] = origin_override
                rec["language"] = lang if lang != "?" else origin_override.split("_")[-1]
                lang = rec["language"]

            decision, reason, top1, p1, top3 = decide(text, lang)
            n_decisions[decision] += 1
            update_agg(rec, decision, top1, label)

            # Phase 0 token count (full precision)
            if text and decision != "UNK_LANG":
                tok_ids = tok(text, add_special_tokens=False, truncation=True,
                              max_length=4096)["input_ids"]
                n_tokens = len(tok_ids)
                agg["len_tokens"][lang].append(n_tokens)
                w = rec.get("words") or 0
                if w > 0:
                    agg["fertility_samples"][lang].append(n_tokens / w)

            if n % PROGRESS_EVERY == 0:
                rate = n / (time.time() - t0)
                print(f"   ... {n:>9,} done  ({rate:.0f}/s)")
    print(f"   ✓ {n:,} records  decisions={dict(n_decisions)}  "
          f"elapsed={time.time()-t0:.1f}s")
    return n

total = 0
for split_name, path in SPLIT_FILES.items():
    total += process_file(path, label=split_name)

print(f"\n[{time.strftime('%H:%M:%S')}] Re-auditing synthetic tier2 files...\n")
for lang, path in SYNTH_FILES.items():
    process_file(path, label=f"synth_{lang}", origin_override="synthetic_reaudit",
                 is_synthetic=True)

print(f"\n[{time.strftime('%H:%M:%S')}] All passes done. Total split records: {total:,}")
print(f"[{time.strftime('%H:%M:%S')}] Aggregating + writing reports...\n")

# ── REPORT WRITING ────────────────────────────────────────────────
def stats(xs):
    if not xs:
        return {"n": 0, "mean": None, "median": None, "p5": None, "p95": None, "stdev": None}
    xs2 = sorted(xs)
    n = len(xs2)
    return {
        "n": n,
        "mean": round(mean(xs2), 4),
        "median": round(median(xs2), 4),
        "p5": round(xs2[int(0.05 * (n-1))], 4),
        "p95": round(xs2[int(0.95 * (n-1))], 4),
        "stdev": round(pstdev(xs2), 4) if n > 1 else 0,
    }

def cnt_to_pct(c):
    tot = sum(c.values())
    if tot == 0: return {}
    return {k: round(v/tot*100, 3) for k, v in c.items()}

def write_csv(path, header, rows):
    with open(REPORT_DIR / path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"  wrote {path}  ({len(rows)} rows)")

# per_split.csv
rows = []
for split, c in agg["by_split"].items():
    tot = sum(c.values())
    p_strict = c["STRICT_PASS"]
    p_relax  = c["RELAXED_PASS"]
    q = c["QUARANTINE"]
    rows.append([split, tot, p_strict, p_relax, q,
                 round((p_strict + p_relax) / tot * 100, 2) if tot else 0,
                 round(p_strict / tot * 100, 2) if tot else 0])
write_csv("per_split.csv",
          ["split", "n_total", "n_strict", "n_relaxed", "n_quarantine", "pass_pct", "strict_pct"],
          sorted(rows))

# per_language.csv
rows = []
for lang, c in agg["by_lang"].items():
    tot = sum(c.values())
    p_strict = c["STRICT_PASS"]
    p_relax  = c["RELAXED_PASS"]
    q = c["QUARANTINE"]
    rows.append([lang, tot, p_strict, p_relax, q,
                 round((p_strict + p_relax) / tot * 100, 2) if tot else 0,
                 round(p_strict / tot * 100, 2) if tot else 0])
write_csv("per_language.csv",
          ["language", "n_total", "n_strict", "n_relaxed", "n_quarantine", "pass_pct", "strict_pct"],
          sorted(rows))

# per_origin_x_language.csv
rows = []
for key, c in agg["by_origin_lang"].items():
    tot = sum(c.values())
    if tot == 0: continue
    origin, _, lang = key.partition("_")
    rows.append([origin, lang, tot, c["STRICT_PASS"], c["RELAXED_PASS"], c["QUARANTINE"],
                 round((c["STRICT_PASS"]+c["RELAXED_PASS"])/tot*100, 2)])
write_csv("per_origin_x_language.csv",
          ["origin", "language", "n_total", "n_strict", "n_relaxed", "n_quarantine", "pass_pct"],
          sorted(rows))

# per_source_id.csv (top 200 by volume)
src_rows = []
for src, c in agg["by_source_id"].items():
    tot = sum(c.values())
    src_rows.append((src, tot, c["STRICT_PASS"], c["RELAXED_PASS"], c["QUARANTINE"],
                     round((c["STRICT_PASS"]+c["RELAXED_PASS"])/tot*100, 2) if tot else 0))
src_rows.sort(key=lambda r: -r[1])
write_csv("per_source_id.csv",
          ["source_id", "n_total", "n_strict", "n_relaxed", "n_quarantine", "pass_pct"],
          src_rows[:200])

# per_domain.csv
rows = []
for d, c in agg["by_domain"].items():
    tot = sum(c.values())
    rows.append([d, tot, c["STRICT_PASS"], c["RELAXED_PASS"], c["QUARANTINE"],
                 round((c["STRICT_PASS"]+c["RELAXED_PASS"])/tot*100, 2) if tot else 0])
rows.sort(key=lambda r: -r[1])
write_csv("per_domain.csv",
          ["domain", "n_total", "n_strict", "n_relaxed", "n_quarantine", "pass_pct"],
          rows)

# per_license.csv
rows = []
for l, c in agg["by_license"].items():
    tot = sum(c.values())
    rows.append([l, tot, c["STRICT_PASS"], c["RELAXED_PASS"], c["QUARANTINE"],
                 round((c["STRICT_PASS"]+c["RELAXED_PASS"])/tot*100, 2) if tot else 0])
rows.sort(key=lambda r: -r[1])
write_csv("per_license.csv",
          ["license", "n_total", "n_strict", "n_relaxed", "n_quarantine", "pass_pct"],
          rows)

# per_register.csv
rows = []
for r, c in agg["by_register"].items():
    tot = sum(c.values())
    rows.append([r, tot, c["STRICT_PASS"], c["RELAXED_PASS"], c["QUARANTINE"],
                 round((c["STRICT_PASS"]+c["RELAXED_PASS"])/tot*100, 2) if tot else 0])
write_csv("per_register.csv",
          ["register", "n_total", "n_strict", "n_relaxed", "n_quarantine", "pass_pct"],
          sorted(rows))

# length_tiers.csv
rows = []
for tier_name, lo, hi in LENGTH_TIERS:
    c = agg["by_length_tier"].get(tier_name, Counter())
    tot = sum(c.values())
    rows.append([tier_name, f"{lo}-{hi if hi < 10**8 else 'inf'}", tot,
                 c["STRICT_PASS"], c["RELAXED_PASS"], c["QUARANTINE"],
                 round((c["STRICT_PASS"]+c["RELAXED_PASS"])/tot*100, 2) if tot else 0])
write_csv("length_tiers.csv",
          ["tier", "char_range", "n_total", "n_strict", "n_relaxed", "n_quarantine", "pass_pct"],
          rows)

# quality_x_glotlid.csv
rows = []
for q_bucket, c in agg["by_quality"].items():
    tot = sum(c.values())
    rows.append([q_bucket, tot, c["STRICT_PASS"], c["RELAXED_PASS"], c["QUARANTINE"],
                 round((c["STRICT_PASS"]+c["RELAXED_PASS"])/tot*100, 2) if tot else 0])
write_csv("quality_x_glotlid.csv",
          ["quality_bucket", "n_total", "n_strict", "n_relaxed", "n_quarantine", "pass_pct"],
          sorted(rows))

# fluency_x_glotlid.csv (same shape)
rows = []
for f_bucket, c in agg["by_fluency"].items():
    tot = sum(c.values())
    rows.append([f_bucket, tot, c["STRICT_PASS"], c["RELAXED_PASS"], c["QUARANTINE"],
                 round((c["STRICT_PASS"]+c["RELAXED_PASS"])/tot*100, 2) if tot else 0])
write_csv("fluency_x_glotlid.csv",
          ["fluency_bucket", "n_total", "n_strict", "n_relaxed", "n_quarantine", "pass_pct"],
          sorted(rows))

# contamination_matrix.csv — what each language's quarantined samples were classified AS
rows = []
for lang, c in agg["contam_by_lang"].items():
    top10 = c.most_common(15)
    for label, n in top10:
        rows.append([lang, label, n])
write_csv("contamination_matrix.csv",
          ["declared_language", "glotlid_top1_label", "count"], rows)

# token_fertility.csv
rows = []
for lang in sorted(set(agg["fertility_samples"].keys())):
    s_fert  = stats(agg["fertility_samples"][lang])
    s_chars = stats(agg["len_chars"][lang])
    s_words = stats(agg["len_words"][lang])
    s_tokens = stats(agg["len_tokens"][lang])
    rows.append([lang, s_fert["n"], s_fert["mean"], s_fert["median"],
                 s_fert["p5"], s_fert["p95"], s_chars["mean"], s_words["mean"],
                 s_tokens["mean"]])
write_csv("token_fertility.csv",
          ["language", "n", "fertility_mean", "fertility_median",
           "fertility_p5", "fertility_p95", "char_mean", "word_mean", "phase0_token_mean"],
          rows)

# code_switch_distribution.csv
rows = []
for lang, xs in agg["code_switch_ratio"].items():
    s = stats(xs)
    rows.append([lang, s["n"], s["mean"], s["median"], s["p5"], s["p95"]])
write_csv("code_switch_distribution.csv",
          ["language", "n", "cs_ratio_mean", "cs_ratio_median", "cs_ratio_p5", "cs_ratio_p95"],
          sorted(rows))

# source_url_top.csv — top 20 URLs per language
rows = []
for lang, c in agg["source_url"].items():
    for url, n in c.most_common(20):
        rows.append([lang, url, n])
write_csv("source_url_top.csv", ["language", "source_url", "count"], rows)

# Full JSON report
def serialize(o):
    if isinstance(o, Counter):
        return dict(o)
    if isinstance(o, defaultdict):
        return {k: serialize(v) for k, v in o.items()}
    if isinstance(o, dict):
        return {k: serialize(v) for k, v in o.items()}
    if isinstance(o, list):
        return [serialize(x) for x in o]
    return o

# Trim large lists (length samples) before JSON dump — keep stats only
slim = {
    "audit_version": "corpus_full_v1",
    "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "config": {
        "glotlid_path": GLOTLID_PATH,
        "glotlid_min_prob": GLOTLID_MIN_P,
        "tokenizer_path": TOKENIZER_PATH,
        "min_text_len": MIN_TEXT_LEN,
        "length_tiers": [[n, lo, hi] for n, lo, hi in LENGTH_TIERS],
    },
    "by_split": {k: dict(v) for k, v in agg["by_split"].items()},
    "by_language": {k: dict(v) for k, v in agg["by_lang"].items()},
    "by_origin_x_language": {k: dict(v) for k, v in agg["by_origin_lang"].items()},
    "by_register": {k: dict(v) for k, v in agg["by_register"].items()},
    "by_domain": {k: dict(v) for k, v in agg["by_domain"].items()},
    "by_license": {k: dict(v) for k, v in agg["by_license"].items()},
    "by_length_tier": {k: dict(v) for k, v in agg["by_length_tier"].items()},
    "by_quality_bucket": {k: dict(v) for k, v in agg["by_quality"].items()},
    "by_fluency_bucket": {k: dict(v) for k, v in agg["by_fluency"].items()},
    "contamination_top15_per_language": {
        lang: dict(c.most_common(15)) for lang, c in agg["contam_by_lang"].items()
    },
    "token_fertility": {
        lang: stats(agg["fertility_samples"][lang]) for lang in agg["fertility_samples"]
    },
    "length_stats_chars": {lang: stats(xs) for lang, xs in agg["len_chars"].items()},
    "length_stats_words": {lang: stats(xs) for lang, xs in agg["len_words"].items()},
    "length_stats_tokens": {lang: stats(xs) for lang, xs in agg["len_tokens"].items()},
    "code_switch_ratio_stats": {
        lang: stats(xs) for lang, xs in agg["code_switch_ratio"].items()
    },
    "source_id_top200": [
        {"source_id": r[0], "n_total": r[1], "pass_pct": r[5]}
        for r in src_rows[:200]
    ],
}
with open(REPORT_DIR / "corpus_audit_report.json", "w") as f:
    json.dump(slim, f, indent=2, ensure_ascii=False)
print(f"\n  wrote corpus_audit_report.json")

# Human-readable markdown
md = []
md.append(f"# LughaGen Corpus Full Audit Report\n")
md.append(f"Generated: {slim['generated_at']}  •  GlotLID-authoritative\n")
md.append("## Per-language overview\n")
md.append("| Language | Total | Strict% | Relaxed% | Quarantine% |")
md.append("|---|---:|---:|---:|---:|")
for lang in ["swahili", "luo", "kikuyu", "kamba"]:
    c = agg["by_lang"].get(lang, Counter())
    tot = sum(c.values()) or 1
    md.append(f"| {lang} | {tot:,} | "
              f"{c['STRICT_PASS']/tot*100:.2f}% | "
              f"{c['RELAXED_PASS']/tot*100:.2f}% | "
              f"{c['QUARANTINE']/tot*100:.2f}% |")
md.append("\n## Synthetic vs Authentic per language\n")
md.append("| Origin × Lang | Total | Pass% |")
md.append("|---|---:|---:|")
for key in sorted(agg["by_origin_lang"]):
    c = agg["by_origin_lang"][key]
    tot = sum(c.values()) or 1
    pp = (c["STRICT_PASS"] + c["RELAXED_PASS"]) / tot * 100
    md.append(f"| {key} | {tot:,} | {pp:.2f}% |")
md.append("\n## Token fertility (Phase 0 tokens per word) per language\n")
md.append("| Language | N | Mean | Median | p5 | p95 |")
md.append("|---|---:|---:|---:|---:|---:|")
for lang in ["swahili", "luo", "kikuyu", "kamba"]:
    s = slim["token_fertility"].get(lang, {})
    md.append(f"| {lang} | {s.get('n','?'):,} | {s.get('mean','?')} | "
              f"{s.get('median','?')} | {s.get('p5','?')} | {s.get('p95','?')} |")
md.append("\n## Top contamination labels per declared language\n")
for lang in ["swahili", "luo", "kikuyu", "kamba"]:
    md.append(f"\n### {lang}")
    top = slim["contamination_top15_per_language"].get(lang, {})
    md.append("| GlotLID Label | Count |")
    md.append("|---|---:|")
    for label, n in list(top.items())[:10]:
        md.append(f"| {label} | {n:,} |")
md.append("\n---\nSee CSVs in the same directory for full breakdowns.\n")
with open(REPORT_DIR / "corpus_audit_report.md", "w") as f:
    f.write("\n".join(md))
print(f"  wrote corpus_audit_report.md")

print(f"\n[{time.strftime('%H:%M:%S')}] ALL DONE.")
print(f"Reports in: {REPORT_DIR}")
