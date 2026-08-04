"""
Full corpus GlotLID audit — per-source, per-language passed/quarantined files.

Mirrors the methodology of tier2_quarantine.py (synthetic audit, May 2026):
  - GlotLID-authoritative
  - Strict / relaxed / quarantine decisions
  - GLOTLID_MIN_PROB = 0.50

Audits 4 separate data sources:
  1. Synthetic tier2 (4 per-language files at /home/nvidia/synthetic/tier2/)
  2. Authentic raw_residue (4 per-language files at /workspace/data/corpus/staged/auth_mono/)
  3. Validation split (/workspace/data/corpus/splits/validation.jsonl)
  4. Test split (/workspace/data/corpus/splits/test.jsonl)

Outputs per-source, per-language passed + quarantined files.
"""
import sys
sys.stdout.reconfigure(line_buffering=True)  # so progress is visible during tee

import json, time
from pathlib import Path
from collections import Counter, defaultdict
import fasttext

# ── CONFIG ───────────────────────────────────────────────────────
GLOTLID_PATH = "/home/nvidia/models/glotlid.bin"
GLOTLID_MIN_PROB = 0.50
MIN_TEXT_LEN = 10
PROGRESS_EVERY = 50_000

OUT_ROOT = Path("/workspace/data/corpus/audited_v2")
REPORT_DIR = Path("/workspace/logs/corpus_audit_v2")
for p in [OUT_ROOT, REPORT_DIR]:
    p.mkdir(parents=True, exist_ok=True)

LANGS = ["swahili", "luo", "kikuyu", "kamba"]

GLOTLID_TARGETS = {
    "swahili": {"strict": {"swh_Latn"},  "relaxed": {"swh_Latn", "swc_Latn"}},
    "luo":     {"strict": {"luo_Latn"},  "relaxed": {"luo_Latn"}},
    "kikuyu":  {"strict": {"kik_Latn"},  "relaxed": {"kik_Latn"}},
    "kamba":   {"strict": {"kam_Latn"},  "relaxed": {"kam_Latn"}},
}

# Each source: (name, kind: "per_lang"|"single_file", files)
SOURCES = {
    "synthetic": {
        "kind": "per_lang",
        "files": {
            "swahili": "/home/nvidia/synthetic/tier2/swahili_tier2.jsonl",
            "luo":     "/home/nvidia/synthetic/tier2/luo_tier2.jsonl",
            "kikuyu":  "/home/nvidia/synthetic/tier2/kikuyu_tier2.jsonl",
            "kamba":   "/home/nvidia/synthetic/tier2/kamba_tier2.jsonl",
        },
    },
    "authentic": {
        "kind": "per_lang",
        "files": {
            "swahili": "/workspace/data/corpus/staged/auth_mono/swahili_raw_residue_v1.jsonl",
            "luo":     "/workspace/data/corpus/staged/auth_mono/luo_raw_residue_v1.jsonl",
            "kikuyu":  "/workspace/data/corpus/staged/auth_mono/kikuyu_raw_residue_v1.jsonl",
            "kamba":   "/workspace/data/corpus/staged/auth_mono/kamba_raw_residue_v1.jsonl",
        },
    },
    "validation": {
        "kind": "single_file",
        "file": "/workspace/data/corpus/splits/validation.jsonl",
    },
    "test": {
        "kind": "single_file",
        "file": "/workspace/data/corpus/splits/test.jsonl",
    },
}

print(f"[{time.strftime('%H:%M:%S')}] Loading GlotLID...")
gl = fasttext.load_model(GLOTLID_PATH)
print(f"[{time.strftime('%H:%M:%S')}] ✓ loaded\n")


def decide(text, lang):
    cfg = GLOTLID_TARGETS.get(lang)
    if cfg is None:
        return "UNK_LANG", "lang_not_in_targets", None, None, []
    if not text or len(text.strip()) < MIN_TEXT_LEN:
        return "QUARANTINE", "text_too_short", None, None, []
    labels, probs = gl.predict(text.replace("\n", " ").strip(), k=3)
    top3 = [(l.replace('__label__',''), float(p)) for l, p in zip(labels, probs)]
    t1, p1 = top3[0]
    if t1 in cfg["strict"] and p1 >= GLOTLID_MIN_PROB:
        return "STRICT_PASS", None, t1, p1, top3
    if any(l in cfg["relaxed"] for l, _ in top3[:3]):
        return "RELAXED_PASS", None, t1, p1, top3
    return "QUARANTINE", "glotlid_rejected", t1, p1, top3


def open_writers(source, langs):
    """For one source, open the 4 passed + 4 quarantined writers (per-language)."""
    src_dir = OUT_ROOT / source
    src_dir.mkdir(parents=True, exist_ok=True)
    writers = {}
    for lang in langs:
        writers[lang] = {
            "passed":   open(src_dir / f"{lang}_{source}_passed.jsonl", "w"),
            "quarant":  open(src_dir / f"{lang}_{source}_quarantined.jsonl", "w"),
        }
    return writers


def close_writers(writers):
    for lang_w in writers.values():
        for w in lang_w.values():
            w.close()


def audit_records(source_name, records_iter, writers, lang_override=None):
    """Stream records, classify each, write to appropriate per-language writer.
       lang_override: if set, force this language for every record (synthetic/authentic per-file case).
    """
    counts = defaultdict(lambda: Counter())   # per_lang -> decision -> count
    top1_dist = defaultdict(Counter)           # per_lang -> top1_label -> count
    contam = defaultdict(Counter)              # per_lang -> contam_label (for quar) -> count
    n_total = 0

    for rec in records_iter:
        text = rec.get("text", "")
        lang = lang_override if lang_override else rec.get("language", "?")

        decision, reason, top1, p1, top3 = decide(text, lang)
        n_total += 1

        # Add audit metadata
        rec["audit_glotlid_top1"] = top1
        rec["audit_glotlid_top1_p"] = p1
        rec["audit_glotlid_top3"] = top3
        rec["audit_decision"] = decision
        rec["audit_quarantine_reason"] = reason
        rec["audit_source"] = source_name

        # Route to writer
        if lang not in writers:
            # Unknown language — log but skip, shouldn't happen for our 4 targets
            continue

        if decision in ("STRICT_PASS", "RELAXED_PASS"):
            writers[lang]["passed"].write(json.dumps(rec, ensure_ascii=False) + "\n")
        else:
            writers[lang]["quarant"].write(json.dumps(rec, ensure_ascii=False) + "\n")

        # Stats
        counts[lang][decision] += 1
        if top1:
            top1_dist[lang][top1] += 1
        if decision == "QUARANTINE" and top1:
            contam[lang][top1] += 1

        if n_total % PROGRESS_EVERY == 0:
            print(f"    ... {n_total:>8,}", flush=True)

    return n_total, counts, top1_dist, contam


def iter_jsonl(path):
    with open(path) as f:
        for line in f:
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


# ── MAIN ─────────────────────────────────────────────────────────
all_results = {}   # source -> lang -> stats

for source_name, source_cfg in SOURCES.items():
    print(f"\n{'='*72}")
    print(f"SOURCE: {source_name}")
    print(f"{'='*72}")

    writers = open_writers(source_name, LANGS)
    t0_src = time.time()
    all_results[source_name] = {}

    if source_cfg["kind"] == "per_lang":
        # Process each per-language file separately
        for lang, path in source_cfg["files"].items():
            print(f"\n  [{time.strftime('%H:%M:%S')}] {lang}: {path}")
            p = Path(path)
            if not p.exists():
                print(f"     ✗ missing — skipping")
                continue
            t0 = time.time()
            n, counts, top1_dist, contam = audit_records(
                source_name, iter_jsonl(p), writers, lang_override=lang
            )
            elapsed = time.time() - t0
            c = counts[lang]
            tot = sum(c.values()) or 1
            all_results[source_name][lang] = {
                "input_file": str(p),
                "n_total": tot,
                "n_strict_pass": c["STRICT_PASS"],
                "n_relaxed_pass": c["RELAXED_PASS"],
                "n_quarantined": c["QUARANTINE"],
                "strict_pct": round(c["STRICT_PASS"] / tot * 100, 2),
                "relaxed_pct": round(c["RELAXED_PASS"] / tot * 100, 2),
                "quarantine_pct": round(c["QUARANTINE"] / tot * 100, 2),
                "pass_pct": round((c["STRICT_PASS"] + c["RELAXED_PASS"]) / tot * 100, 2),
                "top10_glotlid_labels": dict(top1_dist[lang].most_common(10)),
                "top15_contamination_labels": dict(contam[lang].most_common(15)),
                "elapsed_seconds": round(elapsed, 1),
            }
            print(f"     Total {tot:,}  strict={c['STRICT_PASS']:,}  relaxed={c['RELAXED_PASS']:,}  "
                  f"quar={c['QUARANTINE']:,}  ({elapsed:.0f}s)", flush=True)
    else:
        # Single file (validation, test) — records have language field
        path = source_cfg["file"]
        print(f"  [{time.strftime('%H:%M:%S')}] processing {path}")
        p = Path(path)
        if not p.exists():
            print(f"     ✗ missing — skipping")
            continue
        t0 = time.time()
        n, counts, top1_dist, contam = audit_records(
            source_name, iter_jsonl(p), writers, lang_override=None
        )
        elapsed = time.time() - t0
        for lang in LANGS:
            c = counts[lang]
            tot = sum(c.values()) or 1
            if tot == 0 and not c:
                continue
            all_results[source_name][lang] = {
                "input_file": str(p),
                "n_total": tot,
                "n_strict_pass": c["STRICT_PASS"],
                "n_relaxed_pass": c["RELAXED_PASS"],
                "n_quarantined": c["QUARANTINE"],
                "strict_pct": round(c["STRICT_PASS"] / tot * 100, 2),
                "relaxed_pct": round(c["RELAXED_PASS"] / tot * 100, 2),
                "quarantine_pct": round(c["QUARANTINE"] / tot * 100, 2),
                "pass_pct": round((c["STRICT_PASS"] + c["RELAXED_PASS"]) / tot * 100, 2),
                "top10_glotlid_labels": dict(top1_dist[lang].most_common(10)),
                "top15_contamination_labels": dict(contam[lang].most_common(15)),
                "elapsed_seconds": round(elapsed, 1),
            }
        print(f"     Total {n:,}  ({elapsed:.0f}s)", flush=True)

    close_writers(writers)
    print(f"  [{time.strftime('%H:%M:%S')}] {source_name} complete in {time.time()-t0_src:.0f}s")

# ── REPORTS ──────────────────────────────────────────────────────
print(f"\n[{time.strftime('%H:%M:%S')}] Writing reports...\n")

# JSON
report = {
    "audit_version": "corpus_full_glotlid_v2",
    "methodology": "GlotLID-authoritative per-record audit, mirroring tier2_quarantine.py methodology",
    "glotlid_path": GLOTLID_PATH,
    "glotlid_min_prob": GLOTLID_MIN_PROB,
    "min_text_len": MIN_TEXT_LEN,
    "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "output_dir": str(OUT_ROOT),
    "sources": all_results,
}
report_json = REPORT_DIR / "audit_report.json"
report_json.write_text(json.dumps(report, indent=2, ensure_ascii=False))
print(f"  wrote {report_json}")

# CSV summary
import csv
csv_path = REPORT_DIR / "summary_table.csv"
with open(csv_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["source", "language", "n_total", "n_strict", "n_relaxed", "n_quarantine",
                "pass_pct", "strict_pct", "relaxed_pct", "quarantine_pct"])
    for src in ["synthetic", "authentic", "validation", "test"]:
        for lang in LANGS:
            r = all_results.get(src, {}).get(lang)
            if not r: continue
            w.writerow([src, lang, r["n_total"], r["n_strict_pass"], r["n_relaxed_pass"],
                        r["n_quarantined"], r["pass_pct"], r["strict_pct"],
                        r["relaxed_pct"], r["quarantine_pct"]])
print(f"  wrote {csv_path}")

# Markdown report
md = []
md.append("# Corpus Full GlotLID Audit Report\n")
md.append(f"Generated: {report['generated_at']}\n")
md.append(f"Methodology: {report['methodology']}\n")
md.append(f"GlotLID threshold: {GLOTLID_MIN_PROB}\n\n")

md.append("## Summary Table\n")
md.append("| Source | Language | N | Pass% | Strict% | Quarantine% |")
md.append("|---|---|---:|---:|---:|---:|")
for src in ["synthetic", "authentic", "validation", "test"]:
    for lang in LANGS:
        r = all_results.get(src, {}).get(lang)
        if not r: continue
        md.append(f"| {src} | {lang} | {r['n_total']:,} | "
                  f"{r['pass_pct']:.2f}% | {r['strict_pct']:.2f}% | "
                  f"{r['quarantine_pct']:.2f}% |")

# Per-source per-language drill-down
for src in ["synthetic", "authentic", "validation", "test"]:
    md.append(f"\n## {src.upper()}\n")
    for lang in LANGS:
        r = all_results.get(src, {}).get(lang)
        if not r: continue
        md.append(f"\n### {src} × {lang}")
        md.append(f"- Total: {r['n_total']:,}")
        md.append(f"- Strict pass: {r['n_strict_pass']:,} ({r['strict_pct']:.2f}%)")
        md.append(f"- Relaxed pass: {r['n_relaxed_pass']:,} ({r['relaxed_pct']:.2f}%)")
        md.append(f"- Quarantine: {r['n_quarantined']:,} ({r['quarantine_pct']:.2f}%)")
        md.append(f"\n**Top GlotLID labels in this bucket:**")
        md.append("| Label | Count |")
        md.append("|---|---:|")
        for label, count in list(r["top10_glotlid_labels"].items())[:10]:
            md.append(f"| {label} | {count:,} |")
        md.append(f"\n**Top contamination (what quarantined samples were classified as):**")
        md.append("| Label | Count |")
        md.append("|---|---:|")
        for label, count in list(r["top15_contamination_labels"].items())[:10]:
            md.append(f"| {label} | {count:,} |")

md_path = REPORT_DIR / "audit_report.md"
md_path.write_text("\n".join(md))
print(f"  wrote {md_path}")

print(f"\n[{time.strftime('%H:%M:%S')}] ALL DONE.")
print(f"  Cleaned files: {OUT_ROOT}")
print(f"  Reports:       {REPORT_DIR}")
