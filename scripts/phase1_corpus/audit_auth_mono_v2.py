"""
Stage 3: Audit auth_mono_v2/ — GlotLID classification + MinHash dedup.

Reads: /workspace/data/corpus/staged/auth_mono_v2/{lang}_authentic_clean_v1.jsonl
Writes:
  /workspace/data/corpus/staged/auth_mono_v2_audited/{lang}_v2_passed.jsonl
  /workspace/data/corpus/staged/auth_mono_v2_audited/{lang}_v2_quarantined.jsonl
  /workspace/data/corpus/staged/auth_mono_v2_audited/{lang}_v2_neardup_within.jsonl
  /workspace/data/corpus/staged/auth_mono_v2_audited/{lang}_v2_neardup_cross.jsonl
  /workspace/logs/auth_mono_v2_audit/audit_report.json
  /workspace/logs/auth_mono_v2_audit/audit_report.md
  /workspace/logs/auth_mono_v2_audit/run.log  (when launched via tee)

Phases:
  1. GlotLID per-record classification + decision (STRICT_PASS / RELAXED_PASS / QUARANTINE)
  2. Within-language MinHash near-dedup at 0.95 threshold (256 perms)
  3. Cross-language MinHash near-dedup at 0.95 threshold (256 perms)
  4. Write outputs + reports

DRY_RUN environment variable (1=enable): limits each language to LIMIT records for testing.

Honest design notes:
  - GlotLID is single-threaded fasttext (you opted out of parallelization; ~110min on 11M Swahili)
  - MinHash uses datasketch.MinHash + MinHashLSH (in-memory index, single-process)
  - Cross-language pass insert keyed by f"{lang}:{doc_id}" so we know which language each near-dup comes from
  - Records get an additional decision tag: 'NEAR_DUP_WITHIN', 'NEAR_DUP_CROSS', or no tag
  - Records flagged as near-dup move to the dedup files, NOT quarantined — they're informational
"""

import sys
import os
import json
import time
from pathlib import Path
from collections import Counter, defaultdict

sys.stdout.reconfigure(line_buffering=True)

# ── CONFIG ────────────────────────────────────────────────────────
INPUT_DIR  = Path("/workspace/data/corpus/staged/auth_mono_v2")
OUTPUT_DIR = Path("/workspace/data/corpus/staged/auth_mono_v2_audited")
REPORT_DIR = Path("/workspace/logs/auth_mono_v2_audit")
for p in (OUTPUT_DIR, REPORT_DIR):
    p.mkdir(parents=True, exist_ok=True)

GLOTLID_PATH = "/home/nvidia/models/glotlid.bin"
GLOTLID_MIN_PROB = 0.50

# MinHash configuration
MINHASH_PERMS = 256
MINHASH_THRESHOLD = 0.95
SHINGLE_SIZE = 5     # 5-word shingles for sentence-level dedup

LANGS = ["swahili", "luo", "kikuyu", "kamba"]

GLOTLID_TARGETS = {
    "swahili": {"strict": {"swh_Latn"}, "relaxed": {"swh_Latn", "swc_Latn"}},
    "luo":     {"strict": {"luo_Latn"}, "relaxed": {"luo_Latn"}},
    "kikuyu":  {"strict": {"kik_Latn"}, "relaxed": {"kik_Latn"}},
    "kamba":   {"strict": {"kam_Latn"}, "relaxed": {"kam_Latn"}},
}

DRY_RUN = os.environ.get("DRY_RUN", "0") == "1"
DRY_RUN_LIMIT = int(os.environ.get("DRY_RUN_LIMIT", "1000"))
PROGRESS_EVERY = 50_000 if not DRY_RUN else 100


# ── HELPERS ───────────────────────────────────────────────────────
def now():
    return time.strftime("%H:%M:%S")


def shingles(text, k=SHINGLE_SIZE):
    """Generate k-word shingles from text for MinHash."""
    words = text.split()
    if len(words) < k:
        return [" ".join(words)] if words else []
    return [" ".join(words[i:i+k]) for i in range(len(words) - k + 1)]


def make_minhash(text, num_perm=MINHASH_PERMS):
    from datasketch import MinHash
    m = MinHash(num_perm=num_perm)
    for sh in shingles(text):
        m.update(sh.encode('utf8'))
    return m


# ── PHASE 1: GlotLID classification ───────────────────────────────
def phase1_glotlid(input_paths_by_lang):
    """Run GlotLID on every record. Return classified records per language."""
    print(f"\n[{now()}] PHASE 1: GlotLID per-record classification")
    
    import fasttext
    print(f"  loading GlotLID from {GLOTLID_PATH}")
    gl = fasttext.load_model(GLOTLID_PATH)
    print(f"  ✓ GlotLID loaded\n")
    
    classified = {}   # lang -> list of (record, decision, top1, top1_p, top3, reason)
    counts_per_lang = {}
    
    for lang, path in input_paths_by_lang.items():
        cfg = GLOTLID_TARGETS[lang]
        recs_out = []
        counts = Counter()
        
        print(f"  [{now()}] {lang}: {path}")
        t0 = time.time()
        n_processed = 0
        
        with open(path) as f:
            for line in f:
                if DRY_RUN and n_processed >= DRY_RUN_LIMIT:
                    break
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = rec.get("text", "").strip()
                if not text or len(text) < 10:
                    counts["QUARANTINE_SHORT"] += 1
                    rec["audit_decision"] = "QUARANTINE"
                    rec["audit_quarantine_reason"] = "text_too_short"
                    rec["audit_glotlid_top1"] = None
                    rec["audit_glotlid_top1_p"] = None
                    rec["audit_glotlid_top3"] = []
                    recs_out.append(rec)
                    n_processed += 1
                    continue
                
                # Single-line text for fasttext
                line_text = text.replace("\n", " ")
                labels, probs = gl.predict(line_text, k=3)
                top3 = [(l.replace("__label__", ""), float(p)) 
                        for l, p in zip(labels, probs)]
                t1, p1 = top3[0]
                
                # Decide
                if t1 in cfg["strict"] and p1 >= GLOTLID_MIN_PROB:
                    decision = "STRICT_PASS"
                    reason = None
                elif any(l in cfg["relaxed"] for l, _ in top3[:3]):
                    decision = "RELAXED_PASS"
                    reason = None
                else:
                    decision = "QUARANTINE"
                    reason = "glotlid_rejected"
                
                rec["audit_decision"] = decision
                rec["audit_quarantine_reason"] = reason
                rec["audit_glotlid_top1"] = t1
                rec["audit_glotlid_top1_p"] = p1
                rec["audit_glotlid_top3"] = top3
                
                counts[decision] += 1
                recs_out.append(rec)
                n_processed += 1
                
                if n_processed % PROGRESS_EVERY == 0:
                    rate = n_processed / max(0.01, time.time() - t0)
                    print(f"      ... {n_processed:>9,} ({rate:.0f}/s)")
        
        elapsed = time.time() - t0
        rate = n_processed / max(0.01, elapsed)
        print(f"      done {lang}: n={n_processed:,}  "
              f"strict={counts['STRICT_PASS']:,}  relaxed={counts['RELAXED_PASS']:,}  "
              f"quar={counts['QUARANTINE']:,}  short={counts['QUARANTINE_SHORT']:,}  "
              f"({elapsed:.0f}s @ {rate:.0f}/s)\n")
        
        classified[lang] = recs_out
        counts_per_lang[lang] = counts
    
    return classified, counts_per_lang


# ── PHASE 2: Within-language MinHash near-dedup ───────────────────
def phase2_within_dedup(classified):
    """Build MinHash signatures + LSH index per language; mark within-lang dups."""
    from datasketch import MinHashLSH
    
    print(f"\n[{now()}] PHASE 2: Within-language MinHash near-dedup "
          f"(threshold={MINHASH_THRESHOLD}, perms={MINHASH_PERMS})")
    
    minhashes_per_lang = {}  # lang -> dict of doc_id -> MinHash (for cross-lang phase)
    
    for lang, recs in classified.items():
        print(f"\n  [{now()}] {lang}: {len(recs):,} records")
        t0 = time.time()
        
        # Only fingerprint records that passed GlotLID (don't waste effort on quarantined)
        # but DO keep their MinHash for cross-language reference if quarantine was language-mismatch
        sigs = {}  # doc_id -> MinHash
        n_fingerprinted = 0
        for rec in recs:
            if rec["audit_decision"] in ("STRICT_PASS", "RELAXED_PASS"):
                doc_id = rec["doc_id"]
                sigs[doc_id] = make_minhash(rec["text"])
                n_fingerprinted += 1
                if n_fingerprinted % PROGRESS_EVERY == 0:
                    rate = n_fingerprinted / max(0.01, time.time() - t0)
                    print(f"      ... {n_fingerprinted:>9,} fingerprinted ({rate:.0f}/s)")
        
        print(f"      fingerprinting done: {n_fingerprinted:,} sigs in "
              f"{time.time()-t0:.0f}s")
        
        # Build LSH index and query
        t1 = time.time()
        lsh = MinHashLSH(threshold=MINHASH_THRESHOLD, num_perm=MINHASH_PERMS)
        for doc_id, sig in sigs.items():
            lsh.insert(doc_id, sig)
        print(f"      LSH index built in {time.time()-t1:.0f}s")
        
        # Find near-dups: build doc_id->rec lookup, query LSH for each record,
        # mark all matches except the first as near-dup of the first.
        t2 = time.time()
        rec_by_id = {r["doc_id"]: r for r in recs}
        seen_first = set()
        n_marked = 0
        for rec in recs:
            if rec["audit_decision"] not in ("STRICT_PASS", "RELAXED_PASS"):
                continue
            doc_id = rec["doc_id"]
            if doc_id in seen_first:
                continue
            seen_first.add(doc_id)
            matches = lsh.query(sigs[doc_id])
            for m in matches:
                if m != doc_id and m not in seen_first:
                    seen_first.add(m)
                    rec_by_id[m]["audit_neardup_within"] = doc_id  # mark with primary's id
                    n_marked += 1
        
        print(f"      queried LSH + marked dups: {n_marked:,} marked "
              f"in {time.time()-t2:.0f}s")
        print(f"      {lang} within-dedup TOTAL: {time.time()-t0:.0f}s")
        
        minhashes_per_lang[lang] = sigs
    
    return minhashes_per_lang


# ── PHASE 3: Cross-language MinHash near-dedup ────────────────────
def phase3_cross_dedup(classified, minhashes_per_lang):
    """Combine all language signatures into one LSH. Find cross-lang dups."""
    from datasketch import MinHashLSH
    
    print(f"\n[{now()}] PHASE 3: Cross-language MinHash near-dedup")
    
    t0 = time.time()
    lsh = MinHashLSH(threshold=MINHASH_THRESHOLD, num_perm=MINHASH_PERMS)
    total_inserted = 0
    for lang, sigs in minhashes_per_lang.items():
        for doc_id, sig in sigs.items():
            key = f"{lang}:{doc_id}"
            lsh.insert(key, sig)
            total_inserted += 1
    print(f"  cross-lang LSH built: {total_inserted:,} sigs in {time.time()-t0:.0f}s")
    
    # Query each sig; flag where matches come from a different language
    t1 = time.time()
    cross_pairs = []  # list of (lang_a, doc_a, lang_b, doc_b)
    seen_in_pair = set()
    for lang, sigs in minhashes_per_lang.items():
        for doc_id, sig in sigs.items():
            key = f"{lang}:{doc_id}"
            if key in seen_in_pair:
                continue
            matches = lsh.query(sig)
            cross_matches = [m for m in matches if m != key and m.split(":")[0] != lang]
            for cm in cross_matches:
                lang_b, doc_b = cm.split(":", 1)
                cross_pairs.append((lang, doc_id, lang_b, doc_b))
                seen_in_pair.add(cm)
            seen_in_pair.add(key)
    print(f"  cross-lang query done in {time.time()-t1:.0f}s")
    print(f"  {len(cross_pairs):,} cross-language near-dup pairs found")
    
    # Mark records in `classified`
    rec_lookup = {(lang, r["doc_id"]): r for lang, recs in classified.items() for r in recs}
    for (lang_a, doc_a, lang_b, doc_b) in cross_pairs:
        r_b = rec_lookup.get((lang_b, doc_b))
        if r_b is not None:
            r_b["audit_neardup_cross"] = f"{lang_a}:{doc_a}"
    
    return cross_pairs


# ── PHASE 4: Write outputs + reports ──────────────────────────────
def phase4_write_outputs(classified, counts_per_lang, cross_pairs):
    print(f"\n[{now()}] PHASE 4: Writing outputs + reports")
    
    summary = {}
    
    for lang, recs in classified.items():
        out_passed       = OUTPUT_DIR / f"{lang}_v2_passed.jsonl"
        out_quarantined  = OUTPUT_DIR / f"{lang}_v2_quarantined.jsonl"
        out_neardup_win  = OUTPUT_DIR / f"{lang}_v2_neardup_within.jsonl"
        out_neardup_cross = OUTPUT_DIR / f"{lang}_v2_neardup_cross.jsonl"
        
        n_passed = n_quar = n_dup_win = n_dup_cross = 0
        with open(out_passed, "w") as fp, open(out_quarantined, "w") as fq, \
             open(out_neardup_win, "w") as fd, open(out_neardup_cross, "w") as fc:
            for r in recs:
                line = json.dumps(r, ensure_ascii=False) + "\n"
                if r["audit_decision"] == "QUARANTINE":
                    fq.write(line)
                    n_quar += 1
                elif r.get("audit_neardup_within"):
                    fd.write(line)
                    n_dup_win += 1
                elif r.get("audit_neardup_cross"):
                    fc.write(line)
                    n_dup_cross += 1
                else:
                    fp.write(line)
                    n_passed += 1
        
        c = counts_per_lang[lang]
        tot = sum(c.values()) or 1
        summary[lang] = {
            "n_input": tot,
            "n_strict_pass": c["STRICT_PASS"],
            "n_relaxed_pass": c["RELAXED_PASS"],
            "n_quarantine_glotlid": c["QUARANTINE"],
            "n_quarantine_short": c["QUARANTINE_SHORT"],
            "n_final_passed": n_passed,
            "n_final_quarantined": n_quar,
            "n_neardup_within": n_dup_win,
            "n_neardup_cross": n_dup_cross,
            "final_pass_pct": round(n_passed / tot * 100, 2),
        }
    
    # Per-source-id contamination
    contam_by_lang = defaultdict(Counter)
    top1_dist_by_lang = defaultdict(Counter)
    for lang, recs in classified.items():
        for r in recs:
            t1 = r.get("audit_glotlid_top1")
            if t1:
                top1_dist_by_lang[lang][t1] += 1
                if r["audit_decision"] == "QUARANTINE" and r.get("audit_quarantine_reason") == "glotlid_rejected":
                    contam_by_lang[lang][t1] += 1
    
    # Write JSON report
    report = {
        "audit_version": "auth_mono_v2_glotlid_minhash",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": {
            "glotlid_min_prob": GLOTLID_MIN_PROB,
            "minhash_threshold": MINHASH_THRESHOLD,
            "minhash_perms": MINHASH_PERMS,
            "shingle_size": SHINGLE_SIZE,
            "dry_run": DRY_RUN,
            "dry_run_limit": DRY_RUN_LIMIT if DRY_RUN else None,
        },
        "input_dir": str(INPUT_DIR),
        "output_dir": str(OUTPUT_DIR),
        "summary": summary,
        "n_cross_language_neardup_pairs": len(cross_pairs),
        "contamination_top15_per_lang": {
            lang: dict(c.most_common(15)) for lang, c in contam_by_lang.items()
        },
        "top10_glotlid_labels_per_lang": {
            lang: dict(c.most_common(10)) for lang, c in top1_dist_by_lang.items()
        },
    }
    (REPORT_DIR / "audit_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False))
    print(f"  wrote {REPORT_DIR/'audit_report.json'}")
    
    # Markdown report
    md = ["# auth_mono_v2 Audit Report\n",
          f"Generated: {report['generated_at']}\n",
          f"Config: GlotLID min p={GLOTLID_MIN_PROB}, MinHash threshold={MINHASH_THRESHOLD}, "
          f"perms={MINHASH_PERMS}, shingles={SHINGLE_SIZE}-word\n"]
    if DRY_RUN:
        md.append(f"**DRY-RUN MODE**: each language limited to {DRY_RUN_LIMIT} records\n")
    md.append("\n## Summary table\n")
    md.append("| Lang | Input | Passed | Quar (lid) | Quar (short) | Neardup-Within | Neardup-Cross | Final pass% |")
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for lang in LANGS:
        s = summary.get(lang, {})
        if not s: continue
        md.append(f"| {lang} | {s['n_input']:,} | {s['n_final_passed']:,} | "
                  f"{s['n_quarantine_glotlid']:,} | {s['n_quarantine_short']:,} | "
                  f"{s['n_neardup_within']:,} | {s['n_neardup_cross']:,} | "
                  f"{s['final_pass_pct']:.2f}% |")
    md.append(f"\n**Cross-language near-dup pairs found**: {len(cross_pairs):,}\n")
    md.append("\n## Top-15 contamination labels per language (quarantine bucket)\n")
    for lang in LANGS:
        md.append(f"\n### {lang}")
        contam = report["contamination_top15_per_lang"].get(lang, {})
        if not contam:
            md.append("  (no contamination)")
            continue
        md.append("| Label | Count |")
        md.append("|---|---:|")
        for k, v in contam.items():
            md.append(f"| {k} | {v:,} |")
    (REPORT_DIR / "audit_report.md").write_text("\n".join(md))
    print(f"  wrote {REPORT_DIR/'audit_report.md'}")
    
    return report


# ── MAIN ───────────────────────────────────────────────────────────
def main():
    if DRY_RUN:
        print(f"[{now()}] *** DRY RUN MODE: limit={DRY_RUN_LIMIT} per language ***\n")
    
    print(f"[{now()}] Starting auth_mono_v2 audit pipeline")
    print(f"  Input:  {INPUT_DIR}")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"  Report: {REPORT_DIR}")
    
    input_paths = {
        lang: INPUT_DIR / f"{lang}_authentic_clean_v1.jsonl" for lang in LANGS
    }
    for lang, p in input_paths.items():
        if not p.exists():
            print(f"  ERROR: missing {p}")
            sys.exit(1)
    
    classified, counts = phase1_glotlid(input_paths)
    minhashes = phase2_within_dedup(classified)
    cross_pairs = phase3_cross_dedup(classified, minhashes)
    report = phase4_write_outputs(classified, counts, cross_pairs)
    
    print(f"\n[{now()}] ALL DONE")
    print(f"  Summary table:")
    for lang in LANGS:
        s = report["summary"].get(lang, {})
        if s:
            print(f"    {lang:<10} input={s['n_input']:>9,}  "
                  f"passed={s['n_final_passed']:>9,}  "
                  f"final_pass_pct={s['final_pass_pct']:.2f}%")


if __name__ == "__main__":
    main()
