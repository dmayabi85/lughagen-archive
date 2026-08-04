"""
Extract all quarantined records from the v2 audit to two CSV files:
  1. Full archive: every quarantined record, all metadata (~170k rows)
  2. Stratified review CSV: 25 random per (source x language) for human tagging
"""
import sys
sys.stdout.reconfigure(line_buffering=True)

import json, csv, random, time
from pathlib import Path
from collections import defaultdict

random.seed(2026)

INPUT_DIR = Path("/workspace/data/corpus/audited_v2")
OUT_DIR = Path("/workspace/logs/corpus_audit_v2")
OUT_DIR.mkdir(parents=True, exist_ok=True)

ARCHIVE_CSV = OUT_DIR / "quarantined_full_archive.csv"
REVIEW_CSV  = OUT_DIR / "quarantined_review_sample.csv"

REVIEW_PER_BUCKET = 25

SOURCES = ["synthetic", "authentic", "validation", "test"]
LANGS = ["swahili", "luo", "kikuyu", "kamba"]


def clean_text(s):
    if s is None:
        return ""
    return str(s).replace("\n", "\\n").replace("\r", "\\r")


def stringify(v):
    if v is None:
        return ""
    if isinstance(v, (dict, list)):
        return json.dumps(v, ensure_ascii=False)
    return clean_text(v)


# Phase 1: discover schema
print(f"[{time.strftime('%H:%M:%S')}] Phase 1: discovering schema across all quarantined files...")
all_fields = set()
file_summary = {}
total_records = 0

for src in SOURCES:
    for lang in LANGS:
        path = INPUT_DIR / src / f"{lang}_{src}_quarantined.jsonl"
        if not path.exists():
            file_summary[(src, lang)] = (path, 0, False)
            print(f"  MISSING: {path}")
            continue
        n = 0
        with open(path) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    n += 1
                    all_fields.update(rec.keys())
                except json.JSONDecodeError:
                    continue
        file_summary[(src, lang)] = (path, n, True)
        total_records += n
        print(f"  {src}/{lang}: {n:,} records")

print(f"\n[{time.strftime('%H:%M:%S')}] Total quarantined: {total_records:,}")
print(f"  Schema union: {len(all_fields)} unique fields")

priority = [
    "audit_source", "language", "audit_decision", "audit_quarantine_reason",
    "audit_glotlid_top1", "audit_glotlid_top1_p", "audit_glotlid_top3",
    "text", "words", "source_id", "doc_id", "origin", "register",
    "register_detail", "domain", "source_url", "license", "quality_score",
    "fluency_score", "code_switch_ratio", "code_switch_pair", "augmentation",
    "preprocessing", "tokens_llama", "ingested_at", "split",
]
priority_present = [f for f in priority if f in all_fields]
rest = sorted(all_fields - set(priority_present))
columns = priority_present + rest
print(f"  ordered columns ({len(columns)})")

# Phase 2: full archive
print(f"\n[{time.strftime('%H:%M:%S')}] Phase 2: writing full archive CSV...")
t0 = time.time()
n_archive = 0
review_pool = defaultdict(list)

with open(ARCHIVE_CSV, "w", newline="") as fout:
    writer = csv.writer(fout, quoting=csv.QUOTE_ALL)
    writer.writerow(columns)
    for src in SOURCES:
        for lang in LANGS:
            path, n_total, exists = file_summary[(src, lang)]
            if not exists:
                continue
            with open(path) as fin:
                for line in fin:
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    rec.setdefault("audit_source", src)
                    rec.setdefault("language", lang)
                    row = [stringify(rec.get(col, "")) for col in columns]
                    writer.writerow(row)
                    n_archive += 1
                    review_pool[(src, lang)].append(rec)
                    if n_archive % 25000 == 0:
                        rate = n_archive / max(0.01, time.time() - t0)
                        print(f"  ... {n_archive:,} archived ({rate:.0f}/s)")

archive_size_mb = ARCHIVE_CSV.stat().st_size / 1024 / 1024
print(f"  archive written: {n_archive:,} rows, {archive_size_mb:.1f} MB, elapsed {time.time()-t0:.1f}s")

# Phase 3: stratified review CSV
print(f"\n[{time.strftime('%H:%M:%S')}] Phase 3: writing stratified review CSV...")
review_columns = ["human_verdict"] + columns
n_review = 0
with open(REVIEW_CSV, "w", newline="") as fout:
    writer = csv.writer(fout, quoting=csv.QUOTE_ALL)
    writer.writerow(review_columns)
    for src in SOURCES:
        for lang in LANGS:
            pool = review_pool.get((src, lang), [])
            if not pool:
                continue
            n_take = min(REVIEW_PER_BUCKET, len(pool))
            chosen = random.sample(pool, n_take)
            for rec in chosen:
                row = [""] + [stringify(rec.get(col, "")) for col in columns]
                writer.writerow(row)
                n_review += 1
            print(f"  {src}/{lang}: sampled {n_take} from {len(pool)} quarantined")

review_size_kb = REVIEW_CSV.stat().st_size / 1024
print(f"  review written: {n_review:,} rows, {review_size_kb:.1f} KB")

print(f"\n[{time.strftime('%H:%M:%S')}] DONE")
print(f"\nOutputs:")
print(f"  Archive: {ARCHIVE_CSV}  ({n_archive:,} rows, {archive_size_mb:.1f} MB)")
print(f"  Review:  {REVIEW_CSV}   ({n_review:,} rows, {review_size_kb:.1f} KB)")
