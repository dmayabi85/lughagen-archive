"""
tier2_quarantine.py — LughaGen Tier 2 Per-Sample Language Audit
                       Multi-language: Swahili, Kikuyu, Kamba, Luo

GATING POLICY: GlotLID is authoritative. AfroLID prediction is recorded
as metadata only.

Rationale: AfroLID v1.5 shows systematic Bantu→Swahili bias on closely-
related low-resource Bantu languages (Kikuyu, Kamba). Manual spot-checks
verified that GlotLID-high-confidence Kikuyu/Kamba samples flagged by
AfroLID as 'swh' are in fact authentic Kikuyu/Kamba. GlotLID is well-
calibrated across the language families used in this project.

Per-sample output fields added:
    audit_glotlid_top1       — top-1 GlotLID prediction
    audit_glotlid_top1_p     — its probability
    audit_glotlid_top3       — top-3 GlotLID predictions
    audit_afrolid            — AfroLID prediction (metadata)
    audit_afrolid_p          — its probability (metadata)
    audit_decision           — STRICT_PASS | RELAXED_PASS | QUARANTINE (GlotLID-driven)
    audit_quarantine_reason  — null | glotlid_rejected | text_too_short_or_empty
    audit_dual_tool_agreement — STRICT_AGREE | RELAXED_AGREE | AFROLID_DISAGREES |
                                GLOTLID_REJECTED | null   (analytical use)

Resume-safe via SQLite. Idempotent. Robust to malformed JSON lines.
"""

import argparse
import json
import logging
import sqlite3
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import jsonlines

logger = logging.getLogger("tier2.quarantine")


# ── PATHS ────────────────────────────────────────────────────────
TIER2_DIR        = Path("/workspace/synthetic/tier2")
AUDIT_REPORT     = TIER2_DIR / "tier2_quarantine_report.json"
PROGRESS_DB      = TIER2_DIR / "quarantine_progress.db"
GLOTLID_PATH     = "/workspace/models/glotlid.bin"

# ── PER-LANGUAGE TARGETS ─────────────────────────────────────────
LANG_TARGETS: Dict[str, dict] = {
    "swahili": {
        "input":    TIER2_DIR / "swahili_tier2.jsonl",
        "passed":   TIER2_DIR / "swahili_audited_passed.jsonl",
        "quarant":  TIER2_DIR / "swahili_audited_quarantined.jsonl",
        "glotlid_strict":   {"swh_Latn"},
        "glotlid_relaxed":  {"swh_Latn", "swc_Latn"},
        "afrolid_strict":   {"swa", "swh", "swahili"},
        "afrolid_relaxed":  {"swa", "swh", "swc", "swahili", "swahili_congo"},
    },
    "kikuyu": {
        "input":    TIER2_DIR / "kikuyu_tier2.jsonl",
        "passed":   TIER2_DIR / "kikuyu_audited_passed.jsonl",
        "quarant":  TIER2_DIR / "kikuyu_audited_quarantined.jsonl",
        "glotlid_strict":   {"kik_Latn"},
        "glotlid_relaxed":  {"kik_Latn"},
        "afrolid_strict":   {"kik", "kikuyu"},
        "afrolid_relaxed":  {"kik", "kikuyu", "gikuyu"},
    },
    "kamba": {
        "input":    TIER2_DIR / "kamba_tier2.jsonl",
        "passed":   TIER2_DIR / "kamba_audited_passed.jsonl",
        "quarant":  TIER2_DIR / "kamba_audited_quarantined.jsonl",
        "glotlid_strict":   {"kam_Latn"},
        "glotlid_relaxed":  {"kam_Latn"},
        "afrolid_strict":   {"kam", "kamba"},
        "afrolid_relaxed":  {"kam", "kamba"},
    },
    "luo": {
        "input":    TIER2_DIR / "luo_tier2.jsonl",
        "passed":   TIER2_DIR / "luo_audited_passed.jsonl",
        "quarant":  TIER2_DIR / "luo_audited_quarantined.jsonl",
        "glotlid_strict":   {"luo_Latn"},
        "glotlid_relaxed":  {"luo_Latn"},
        "afrolid_strict":   {"luo", "lwo", "dholuo"},
        "afrolid_relaxed":  {"luo", "lwo", "dholuo"},
    },
}

GLOTLID_MIN_PROB  = 0.50
AFROLID_MIN_PROB  = 0.50
CHECKPOINT_EVERY  = 5_000


# ── PROGRESS TRACKING ────────────────────────────────────────────

def init_db():
    PROGRESS_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(PROGRESS_DB)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS quarantine_progress (
            language   TEXT PRIMARY KEY,
            n_done     INTEGER NOT NULL,
            n_passed   INTEGER NOT NULL,
            n_quarant  INTEGER NOT NULL,
            updated_at TEXT NOT NULL
        )
    """)
    conn.commit()
    return conn


def get_progress(conn, lang: str):
    row = conn.execute(
        "SELECT n_done, n_passed, n_quarant FROM quarantine_progress WHERE language=?",
        (lang,),
    ).fetchone()
    return row if row else (0, 0, 0)


def set_progress(conn, lang: str, n_done, n_passed, n_quarant):
    conn.execute("""
        INSERT INTO quarantine_progress(language, n_done, n_passed, n_quarant, updated_at)
        VALUES(?, ?, ?, ?, ?)
        ON CONFLICT(language) DO UPDATE SET
            n_done=excluded.n_done, n_passed=excluded.n_passed,
            n_quarant=excluded.n_quarant, updated_at=excluded.updated_at
    """, (lang, n_done, n_passed, n_quarant, datetime.utcnow().isoformat() + "Z"))
    conn.commit()


# ── ROBUST JSONL ITERATOR ────────────────────────────────────────

def safe_jsonl_iter(path: Path, lang: str, skip_count: int = 0):
    """
    Yield records from a JSONL file, skipping malformed lines with logging.
    Does its own file handling (not jsonlines.open) so we can recover from
    individual line errors without losing the rest of the file.
    """
    n_bad = 0
    n_yielded = 0
    n_skipped_for_resume = 0
    with open(path, "r", encoding="utf-8") as f:
        for line_num, raw in enumerate(f, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                rec = json.loads(raw)
            except json.JSONDecodeError as e:
                n_bad += 1
                if n_bad <= 5:
                    logger.warning("[%s] line %d: malformed JSON (%s)",
                                   lang, line_num, str(e)[:100])
                continue
            if n_skipped_for_resume < skip_count:
                n_skipped_for_resume += 1
                continue
            yield rec
            n_yielded += 1
    if n_bad:
        logger.warning("[%s] skipped %d malformed lines total (yielded %d records)",
                       lang, n_bad, n_yielded)


# ── MODEL LOADING ────────────────────────────────────────────────

def load_glotlid():
    import fasttext
    logger.info("Loading GlotLID from %s", GLOTLID_PATH)
    return fasttext.load_model(GLOTLID_PATH)


def load_afrolid():
    """Returns callable text->(label, prob), or None if unloadable."""
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch

        model_id = "UBC-NLP/afrolid_1.5"
        logger.info("Loading AfroLID from %s", model_id)
        tok = AutoTokenizer.from_pretrained(model_id)
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        model = AutoModelForSequenceClassification.from_pretrained(model_id).to(device).eval()
        id2label = model.config.id2label

        @torch.inference_mode()
        def predict(text: str):
            enc = tok(text, return_tensors="pt", truncation=True, max_length=256).to(device)
            out = model(**enc)
            probs = torch.softmax(out.logits[0], dim=-1)
            top_p, top_i = probs.max(dim=-1)
            return id2label[top_i.item()].lower(), float(top_p.item())

        return predict
    except Exception as e:
        logger.warning("AfroLID unavailable (%s) — proceeding with GlotLID only", e)
        return None


# ── CLASSIFICATION (GlotLID-authoritative) ───────────────────────

def classify_sample(text, glotlid, afrolid, targets):
    text_clean = (text or "").replace("\n", " ").strip()

    if not text_clean or len(text_clean) < 10:
        return {
            "audit_glotlid_top1":         None,
            "audit_glotlid_top1_p":       0.0,
            "audit_glotlid_top3":         [],
            "audit_afrolid":              None,
            "audit_afrolid_p":            0.0,
            "audit_decision":             "QUARANTINE",
            "audit_quarantine_reason":    "text_too_short_or_empty",
            "audit_dual_tool_agreement":  None,
        }

    # GlotLID
    labels, probs = glotlid.predict(text_clean, k=3)
    gl_top3 = [(l.replace("__label__", ""), float(p)) for l, p in zip(labels, probs)]
    gl_top1, gl_top1_p = gl_top3[0]
    gl_top3_labels = [l for l, _ in gl_top3]
    gl_top3_records = [{"lang": l, "p": round(p, 4)} for l, p in gl_top3]

    # AfroLID (metadata only)
    if afrolid is not None:
        af_label, af_prob = afrolid(text_clean)
    else:
        af_label, af_prob = None, 0.0

    # GLOTLID-AUTHORITATIVE GATING
    gl_strict  = gl_top1 in targets["glotlid_strict"] and gl_top1_p >= GLOTLID_MIN_PROB
    gl_relaxed = bool(set(gl_top3_labels) & targets["glotlid_relaxed"])

    if gl_strict:
        decision, reason = "STRICT_PASS", None
    elif gl_relaxed:
        decision, reason = "RELAXED_PASS", None
    else:
        decision, reason = "QUARANTINE", "glotlid_rejected"

    # AfroLID agreement (analytical only — does NOT affect decision)
    if afrolid is None:
        dual_agreement = None
    elif decision == "QUARANTINE":
        dual_agreement = "GLOTLID_REJECTED"
    else:
        af_strict  = af_label in targets["afrolid_strict"] and af_prob >= AFROLID_MIN_PROB
        af_relaxed = af_label in targets["afrolid_relaxed"]
        if af_strict:
            dual_agreement = "STRICT_AGREE"
        elif af_relaxed:
            dual_agreement = "RELAXED_AGREE"
        else:
            dual_agreement = "AFROLID_DISAGREES"

    return {
        "audit_glotlid_top1":         gl_top1,
        "audit_glotlid_top1_p":       round(gl_top1_p, 4),
        "audit_glotlid_top3":         gl_top3_records,
        "audit_afrolid":              af_label,
        "audit_afrolid_p":            round(af_prob, 4),
        "audit_decision":             decision,
        "audit_quarantine_reason":    reason,
        "audit_dual_tool_agreement":  dual_agreement,
    }


# ── PER-LANGUAGE PROCESSING ──────────────────────────────────────

def process_language(lang, glotlid, afrolid, conn):
    targets = LANG_TARGETS[lang]
    input_path = targets["input"]

    if not input_path.exists():
        logger.error("[%s] Input not found: %s — skipping", lang, input_path)
        return {"skipped": True, "reason": f"input_missing: {input_path}"}

    n_done, n_passed, n_quarant = get_progress(conn, lang)
    logger.info("[%s] Resuming from offset %d (passed=%d, quarantined=%d)",
                lang, n_done, n_passed, n_quarant)

    decision_counts = Counter()
    reason_counts   = Counter()
    agreement_counts = Counter()
    glotlid_top1_counter = Counter()

    passed_writer  = jsonlines.open(targets["passed"], mode="a")
    quarant_writer = jsonlines.open(targets["quarant"], mode="a")
    n_done_total = n_done

    try:
        for rec in safe_jsonl_iter(input_path, lang, skip_count=n_done):
            text = rec.get("text") or rec.get("output") or rec.get("content") or ""
            audit = classify_sample(text, glotlid, afrolid, targets)
            rec.update(audit)
            rec["language"] = lang

            glotlid_top1_counter[audit["audit_glotlid_top1"] or "EMPTY"] += 1
            decision_counts[audit["audit_decision"]] += 1
            if audit["audit_quarantine_reason"]:
                reason_counts[audit["audit_quarantine_reason"]] += 1
            if audit["audit_dual_tool_agreement"]:
                agreement_counts[audit["audit_dual_tool_agreement"]] += 1

            if audit["audit_decision"] in ("STRICT_PASS", "RELAXED_PASS"):
                passed_writer.write(rec)
                n_passed += 1
            else:
                quarant_writer.write(rec)
                n_quarant += 1

            n_done_total += 1

            if n_done_total % CHECKPOINT_EVERY == 0:
                set_progress(conn, lang, n_done_total, n_passed, n_quarant)
                pct = n_passed / n_done_total * 100
                logger.info("[%s]  %d processed  passed=%d (%.1f%%)  quarantined=%d",
                            lang, n_done_total, n_passed, pct, n_quarant)
    finally:
        passed_writer.close()
        quarant_writer.close()
        set_progress(conn, lang, n_done_total, n_passed, n_quarant)

    total = n_done_total
    return {
        "skipped":            False,
        "input_file":         str(input_path),
        "passed_file":        str(targets["passed"]),
        "quarantine_file":    str(targets["quarant"]),
        "n_total":            total,
        "n_passed":           n_passed,
        "n_quarantined":      n_quarant,
        "pass_rate_pct":      round(n_passed / total * 100, 2) if total else 0,
        "decisions":          dict(decision_counts),
        "quarantine_reasons": dict(reason_counts),
        "dual_tool_agreement": dict(agreement_counts),
        "glotlid_top1_top10": dict(glotlid_top1_counter.most_common(10)),
    }


# ── MAIN ─────────────────────────────────────────────────────────

def run(languages=None, skip_afrolid=False):
    if languages is None:
        languages = list(LANG_TARGETS.keys())

    unknown = [l for l in languages if l not in LANG_TARGETS]
    if unknown:
        logger.error("Unknown languages: %s", unknown)
        sys.exit(1)

    conn = init_db()
    glotlid = load_glotlid()
    afrolid = None if skip_afrolid else load_afrolid()

    results = {}
    for lang in languages:
        logger.info("═" * 60)
        logger.info("PROCESSING %s", lang.upper())
        logger.info("═" * 60)
        results[lang] = process_language(lang, glotlid, afrolid, conn)

    report = {
        "audit_version":   "tier2_quarantine_v3.0_glotlid_authoritative",
        "generated_at":    datetime.utcnow().isoformat() + "Z",
        "gating_policy":   "GlotLID is authoritative; AfroLID is analytical metadata",
        "afrolid_used":    afrolid is not None,
        "thresholds": {
            "glotlid_min_prob": GLOTLID_MIN_PROB,
            "afrolid_min_prob": AFROLID_MIN_PROB,
        },
        "languages":       results,
    }
    AUDIT_REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    print("\n" + "═" * 70)
    print(f"  TIER 2 QUARANTINE COMPLETE — GlotLID-authoritative")
    print("═" * 70)
    for lang, r in results.items():
        if r.get("skipped"):
            print(f"\n  {lang.upper():<10}  SKIPPED — {r['reason']}")
            continue
        n = r["n_total"]; p = r["n_passed"]; q = r["n_quarantined"]; pct = r["pass_rate_pct"]
        print(f"\n  {lang.upper():<10}  n={n:>7,}   ✅ {p:>7,} ({pct:5.2f}%)   ⚠ {q:>5,}")
        if r.get("dual_tool_agreement"):
            for tag, count in r["dual_tool_agreement"].items():
                if count:
                    print(f"               • {tag:<28} {count:>7,}")
    print("\n" + "═" * 70)
    print(f"\n  Report: {AUDIT_REPORT}")
    print("═" * 70 + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--languages", nargs="+",
                   choices=list(LANG_TARGETS.keys()), default=None)
    p.add_argument("--skip-afrolid", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level="INFO",
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    run(languages=args.languages, skip_afrolid=args.skip_afrolid)
