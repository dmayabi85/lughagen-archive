"""
normalise_ingest.py
─────────────────────────────────────────────────────────────────────────
Read every YAML manifest under manifests/, load the referenced source file,
normalise records to the canonical schema, and write to staged/<bucket>/.

Each output record carries the full metadata required by SCHEMA.md.

Usage:
    python normalise_ingest.py \\
        --manifests-dir /workspace/data/corpus/manifests \\
        --staged-dir    /workspace/data/corpus/staged \\
        --audit-log     /workspace/data/corpus/audit/ingest_log.jsonl

Dependencies:  pyyaml, transformers (for tokeniser, optional)
"""

import argparse
import hashlib
import json
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import yaml


# Reuse the CS detector from separate_synthetic_cs.py
EN_MARKERS = {
    "the","and","of","to","a","in","is","that","for","it","with","as","was",
    "on","are","this","by","be","at","an","have","from","but","what","when",
    "which","who","will","would","can","could","has","had","were","been",
    "they","their","there","not","or","also","because","about","into","than",
    "very","more","some","any","other","such","like","just","only","you",
    "your","we","our","my","me","he","she","his","her","them","us",
}


def normalise_text(text: str) -> str:
    """Unicode NFC + collapse whitespace + strip."""
    if not isinstance(text, str):
        return ""
    t = unicodedata.normalize("NFC", text)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def doc_id(text: str) -> str:
    """SHA-1 of normalised text — stable cross-source identifier."""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def code_switch_ratio(text: str) -> float:
    words = re.findall(r"[A-Za-z']+", text.lower())
    if not words:
        return 0.0
    en = sum(1 for w in words if w in EN_MARKERS)
    return en / len(words)


def deterministic_split(doc_id_hex: str, ratios=(0.97, 0.015, 0.015)) -> str:
    """Deterministic train/val/test assignment from the doc_id hash."""
    bucket = int(doc_id_hex[:8], 16) / 0xFFFFFFFF   # [0, 1)
    if bucket < ratios[0]:
        return "train"
    elif bucket < ratios[0] + ratios[1]:
        return "validation"
    else:
        return "test"


def evaluate_quality_contract(rec, contract):
    """Return (passes: bool, reason: str|None) based on the manifest contract."""
    if not contract:
        return True, None
    words = len(rec["text"].split())
    if "min_words" in contract and words < contract["min_words"]:
        return False, f"min_words {words} < {contract['min_words']}"
    if "max_words" in contract and words > contract["max_words"]:
        return False, f"max_words {words} > {contract['max_words']}"
    if "min_code_switch_ratio" in contract:
        if rec.get("code_switch_ratio", 0) < contract["min_code_switch_ratio"]:
            return False, "code_switch_ratio too low for CS source"
    if "max_code_switch_ratio" in contract:
        if rec.get("code_switch_ratio", 1) > contract["max_code_switch_ratio"]:
            return False, "code_switch_ratio too high"
    if "max_repetition_ratio" in contract:
        toks = rec["text"].split()
        if toks:
            unique_ratio = 1 - (len(set(toks)) / len(toks))
            if unique_ratio > contract["max_repetition_ratio"]:
                return False, f"repetition_ratio {unique_ratio:.2f} > {contract['max_repetition_ratio']}"
    return True, None


def load_manifest(path: Path):
    """Load a single-document or multi-document YAML manifest file."""
    docs = []
    with open(path) as f:
        for doc in yaml.safe_load_all(f):
            if doc:
                docs.append(doc)
    return docs


def iter_source_records(manifest):
    """Yield raw records from the source file declared in a manifest."""
    loc = manifest["location"]
    path = Path(loc["path"])
    text_field = loc.get("text_field", "text")
    field_filter = loc.get("filter")          # {field, equals}
    qual_filter  = loc.get("quality_threshold")  # {field, min}

    if not path.exists():
        print(f"  [warn] source file missing: {path}")
        return

    with open(path) as f:
        for line in f:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if field_filter:
                if obj.get(field_filter["field"]) != field_filter["equals"]:
                    continue
            if qual_filter:
                v = obj.get(qual_filter["field"])
                if v is None or v < qual_filter["min"]:
                    continue
            text = obj.get(text_field)
            if not text:
                continue
            yield obj, text


def ingest_manifest(manifest, staged_root: Path, audit_handle, bucket_dir_map):
    """Process one manifest, writing normalised records to the right bucket."""
    src_id   = manifest["source_id"]
    origin   = manifest["origin"]
    register = manifest["register"]
    language = manifest["language"]
    defaults = manifest.get("defaults", {})
    contract = manifest.get("quality_contract", {})
    license_ = manifest.get("license", {}).get("type", "unknown")
    source_url = manifest.get("collection", {}).get("source_url")

    bucket_key = f"{origin[:5]}_{'mono' if register == 'monolingual' else 'cs'}"
    out_dir = staged_root / bucket_dir_map[bucket_key]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{src_id}.jsonl"

    n_in = n_out = n_dropped = 0
    now_iso = datetime.now(timezone.utc).isoformat()

    with open(out_path, "w") as out_f:
        for raw, text in iter_source_records(manifest):
            n_in += 1
            text = normalise_text(text)
            if not text:
                n_dropped += 1
                continue

            cs_r = code_switch_ratio(text)
            words = len(text.split())

            rec = {
                "doc_id":             doc_id(text),
                "text":               text,
                "language":           language,
                "register":           register,
                "origin":             origin,
                "source_id":          src_id,
                "source_url":         source_url,
                "license":            license_,
                "domain":             defaults.get("domain"),
                "register_detail":    defaults.get("register_detail"),
                "code_switch_pair":   defaults.get("code_switch_pair"),
                "code_switch_ratio":  round(cs_r, 4),
                "tokens_llama":       None,    # filled by a later optional pass
                "words":              words,
                "quality_score":      raw.get("quality") or raw.get("q"),
                "fluency_score":      raw.get("fluency") or raw.get("score"),
                "preprocessing":      defaults.get("preprocessing", []),
                "augmentation":       defaults.get("augmentation", []),
                "split":              None,    # filled below
                "ingested_at":        now_iso,
            }
            rec["split"] = deterministic_split(rec["doc_id"])

            passes, reason = evaluate_quality_contract(rec, contract)
            if not passes:
                n_dropped += 1
                continue

            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_out += 1

    audit_handle.write(json.dumps({
        "ts":           now_iso,
        "source_id":    src_id,
        "manifest":     manifest.get("display_name", src_id),
        "language":     language,
        "register":     register,
        "origin":       origin,
        "records_in":   n_in,
        "records_out":  n_out,
        "records_dropped": n_dropped,
        "output_path":  str(out_path),
    }) + "\n")

    print(f"  [{src_id}] {n_in:,} read \u2192 {n_out:,} written, {n_dropped:,} dropped")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifests-dir", required=True)
    ap.add_argument("--staged-dir",    required=True)
    ap.add_argument("--audit-log",     required=True)
    args = ap.parse_args()

    manifests_root = Path(args.manifests_dir)
    staged_root    = Path(args.staged_dir)
    audit_path     = Path(args.audit_log)
    audit_path.parent.mkdir(parents=True, exist_ok=True)

    bucket_dir_map = {
        "synth_mono": "synth_mono",
        "synth_cs":   "synth_cs",
        "authe_mono": "auth_mono",
        "authe_cs":   "auth_cs",
    }

    yamls = sorted(manifests_root.rglob("*.yaml")) + sorted(manifests_root.rglob("*.yml"))
    print(f"Found {len(yamls)} manifest file(s) under {manifests_root}")

    with open(audit_path, "a") as audit:
        for y in yamls:
            print(f"\n--- {y.relative_to(manifests_root)} ---")
            for manifest in load_manifest(y):
                try:
                    ingest_manifest(manifest, staged_root, audit, bucket_dir_map)
                except Exception as e:
                    print(f"  [error] {manifest.get('source_id','?')}: {e}")
                    audit.write(json.dumps({
                        "ts": datetime.now(timezone.utc).isoformat(),
                        "source_id": manifest.get("source_id"),
                        "error": str(e),
                    }) + "\n")


if __name__ == "__main__":
    main()
