"""
Stage 5: Build Path A and Path D training splits + shared validation/test.

Design (locked):
  Validation/test: 1500 records per language each, equal-balanced (6000 each total)
                   Stratified by origin (auth vs synth) within each language
                   Sampled from authentic + synthetic mix
                   Held out so neither train ever sees these doc_ids
  
  Path A: Swahili capped at 2,000,000 records, all other languages used fully
  Path D: Swahili capped at   500,000 records, all other languages used fully
  
  Same training step count (6000) will be used in both downstream pilots.

Inputs:
  /workspace/data/corpus/staged/auth_mono/{lang}_authentic_v2.jsonl       [4 files]
  /workspace/data/corpus/staged/synth_mono/{lang}_lughagen_synth_mono_v2.jsonl  [4 files]
  /workspace/data/corpus/staged/synth_cs/sheng_synthetic_v1.jsonl         [1 file]

Outputs:
  /workspace/data/corpus/splits_v2/
    validation_shared.jsonl
    test_shared.jsonl
    pathA/train.jsonl
    pathD/train.jsonl
    manifest.json
    build.log (when launched with tee)

Random seed: 2026 (for full reproducibility)

DRY_RUN env var (1=enable) limits each source to LIMIT records.
"""
import sys, os, json, random, time, hashlib
from pathlib import Path
from collections import Counter, defaultdict

sys.stdout.reconfigure(line_buffering=True)

# ── CONFIG ──────────────────────────────────────────────────────
SEED = 2026
random.seed(SEED)

INPUT_AUTH = Path("/workspace/data/corpus/staged/auth_mono")
INPUT_SYNTH_MONO = Path("/workspace/data/corpus/staged/synth_mono")
INPUT_SYNTH_CS = Path("/workspace/data/corpus/staged/synth_cs")
OUTPUT_DIR = Path("/workspace/data/corpus/splits_v2")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "pathA").mkdir(exist_ok=True)
(OUTPUT_DIR / "pathD").mkdir(exist_ok=True)

LANGS = ["swahili", "luo", "kikuyu", "kamba"]

# Val/test per language (equal-balanced)
VAL_PER_LANG = 1500
TEST_PER_LANG = 1500

# Path A / D Swahili caps
SWAHILI_CAP_A = 2_000_000
SWAHILI_CAP_D = 500_000

# Input file mapping
SOURCES = {
    "swahili": {
        "authentic":   INPUT_AUTH / "swahili_authentic_v2.jsonl",
        "synth_mono":  INPUT_SYNTH_MONO / "swahili_lughagen_synth_mono_v2.jsonl",
        "sheng":       INPUT_SYNTH_CS / "sheng_synthetic_v1.jsonl",
    },
    "luo": {
        "authentic":  INPUT_AUTH / "luo_authentic_v2.jsonl",
        "synth_mono": INPUT_SYNTH_MONO / "luo_lughagen_synth_mono_v2.jsonl",
    },
    "kikuyu": {
        "authentic":  INPUT_AUTH / "kikuyu_authentic_v2.jsonl",
        "synth_mono": INPUT_SYNTH_MONO / "kikuyu_lughagen_synth_mono_v2.jsonl",
    },
    "kamba": {
        "authentic":  INPUT_AUTH / "kamba_authentic_v2.jsonl",
        "synth_mono": INPUT_SYNTH_MONO / "kamba_lughagen_synth_mono_v2.jsonl",
    },
}

DRY_RUN = os.environ.get("DRY_RUN", "0") == "1"
DRY_RUN_LIMIT = int(os.environ.get("DRY_RUN_LIMIT", "1000"))
PROGRESS_EVERY = 200_000 if not DRY_RUN else 100


def now():
    return time.strftime("%H:%M:%S")


def load_source(path, limit=None):
    """Load jsonl as list of dicts. Returns (records_list, n_total_in_file)."""
    if not path.exists():
        return [], 0
    records = []
    n_total = 0
    with open(path) as f:
        for i, line in enumerate(f):
            try:
                r = json.loads(line)
                records.append(normalize_to_schema(r))
                n_total += 1
            except json.JSONDecodeError:
                continue
            if limit and len(records) >= limit:
                # In dry-run, we cap loading for speed
                break
    return records, n_total


def ensure_doc_id(rec, source_label, idx):
    """Ensure every record has a unique doc_id for tracking held-outs."""
    if rec.get("doc_id"):
        return rec["doc_id"]
    # Generate stable doc_id from text hash + source
    txt = rec.get("text", "")
    h = hashlib.md5((source_label + ":" + txt).encode()).hexdigest()[:16]
    return f"{source_label}_{idx}_{h}"


def add_split_field(rec, split_value):
    rec["split"] = split_value
    return rec


# Schema fields the training_loop.py Features schema requires.
# Every record in the output splits must have these 20 fields with type-correct
# values. Extras (audit_*, afrolid_*, split_path, etc) are preserved — pyarrow
# with explicit Features silently ignores extras at load time.
SCHEMA_FIELDS = {
    # ALWAYS-POPULATED fields (real defaults match OLD corpus convention).
    # These prevent pyarrow null-type inference from breaking the cast.
    "doc_id":            ("str",   None),       # always set by ensure_doc_id()
    "text":              ("str",   ""),         # empty text safer than None
    "language":          ("str",   None),       # always present in sources
    "origin":            ("str",   None),       # always present in sources
    "register":          ("str",   None),       # always present in sources
    "split":             ("str",   None),       # set by add_split_field()
    "source_id":         ("str",   None),       # always present in sources
    "domain":            ("str",   "mixed"),    # default if missing
    "license":           ("str",   "internal_research_only"),
    "ingested_at":       ("str",   "__now__"),  # sentinel: current UTC ISO timestamp
    "code_switch_ratio": ("float", 0.0),        # default for monolingual records
    "quality_score":     ("float", 1.0),        # default high score
    "words":             ("int",   "__words__"),# sentinel: len(text.split())
    "augmentation":      ("list",  []),
    "preprocessing":     ("list",  []),
    # GENUINELY NULLABLE fields (OLD had all-null too, load worked).
    "register_detail":   ("str",   None),
    "source_url":        ("str",   None),
    "code_switch_pair":  ("str",   None),
    "fluency_score":     ("float", None),
    "tokens_llama":      ("int",   None),
}


def normalize_to_schema(rec):
    """Ensure record has all 20 schema fields with type-correct values.

    Option Y: preserves audit and other extra fields alongside schema fields.
    The training_loop.py uses an explicit datasets.Features schema which
    silently ignores extras, so retaining them is safe.

    Missing fields default to None for nullable scalars and [] for Sequence
    fields, matching the OLD corpus's null distribution.
    """
    import datetime

    def coerce(val, typ):
        if val is None:
            return None
        if typ == "str":
            return str(val)
        if typ == "float":
            try:
                return float(val)
            except (TypeError, ValueError):
                return None
        if typ == "int":
            try:
                return int(val)
            except (TypeError, ValueError):
                return None
        if typ == "list":
            if isinstance(val, list):
                return [str(x) for x in val if x is not None]
            return []
        return val

    def resolve_default(default, rec_in):
        """Handle special sentinel defaults that depend on the record."""
        if default == "__now__":
            return datetime.datetime.now(datetime.timezone.utc).isoformat()
        if default == "__words__":
            txt = rec_in.get("text") or ""
            return len(str(txt).split())
        return default

    # Build output with ONLY the 20 schema fields. Extras (audit_*, afrolid_*,
    # split_path, etc.) are dropped because datasets/pyarrow with explicit
    # Features schema rejects records containing undeclared columns.
    # Audit traceability is preserved at project level via
    # /workspace/data/corpus/staged/auth_mono_v2_audited/ (keyed by doc_id).
    out = {}

    # Populate each schema field with type-correct value or default.
    # Defaults that match the OLD corpus convention ensure pyarrow can
    # infer correct types when reading the JSON file.
    for fname, (ftype, default) in SCHEMA_FIELDS.items():
        v = rec.get(fname)
        if v is None and default is not None:
            resolved = resolve_default(default, rec)
            out[fname] = coerce(resolved, ftype)
        else:
            out[fname] = coerce(v, ftype)

    return out


# ─────────────────────────────────────────────────────────────────
# PHASE 1: Load all sources
# ─────────────────────────────────────────────────────────────────
print(f"[{now()}] Stage 5: Build splits v2")
print(f"  Seed: {SEED}")
print(f"  Val/test per language: {VAL_PER_LANG}/{TEST_PER_LANG}")
print(f"  Path A Swahili cap: {SWAHILI_CAP_A:,}")
print(f"  Path D Swahili cap: {SWAHILI_CAP_D:,}")
if DRY_RUN:
    print(f"  *** DRY RUN: limit={DRY_RUN_LIMIT} per source ***")
print()

print(f"[{now()}] PHASE 1: Loading sources")
all_data = {}  # lang -> origin_label -> list of records

t_phase1_start = time.time()
load_limit = DRY_RUN_LIMIT if DRY_RUN else None

for lang in LANGS:
    all_data[lang] = {}
    for origin_label, path in SOURCES[lang].items():
        t0 = time.time()
        recs, n_total = load_source(path, limit=load_limit)
        
        # Ensure doc_id on every record
        for i, r in enumerate(recs):
            r["doc_id"] = ensure_doc_id(r, f"{lang}_{origin_label}", i)
        
        all_data[lang][origin_label] = recs
        elapsed = time.time() - t0
        print(f"  {lang}/{origin_label:<12} loaded {len(recs):>10,} records "
              f"({elapsed:.1f}s)")

print(f"  Phase 1 done in {time.time()-t_phase1_start:.1f}s")


# ─────────────────────────────────────────────────────────────────
# PHASE 2: Sample held-out validation + test (stratified by origin within lang)
# ─────────────────────────────────────────────────────────────────
print(f"\n[{now()}] PHASE 2: Sample shared validation + test (stratified by origin)")

held_out = {"val": defaultdict(list), "test": defaultdict(list)}
held_out_doc_ids = set()

for lang in LANGS:
    origins = list(all_data[lang].keys())
    origin_counts = {o: len(all_data[lang][o]) for o in origins}
    total_avail = sum(origin_counts.values())
    
    print(f"\n  {lang}: stratifying val/test by origin")
    for o in origins:
        print(f"    {o:<12} {origin_counts[o]:>10,} ({origin_counts[o]/total_avail*100:.1f}%)")
    
    # For each origin: pick val + test INDICES disjointly, then use them.
    # This guarantees no overlap, no skip-and-continue, and exact target counts
    # (capped by availability).
    for o in origins:
        pool = all_data[lang][o]
        origin_share = origin_counts[o] / total_avail
        n_val_target = round(origin_share * VAL_PER_LANG)
        n_test_target = round(origin_share * TEST_PER_LANG)
        
        n_val = min(n_val_target, len(pool))
        n_test = min(n_test_target, max(0, len(pool) - n_val))
        
        all_indices = list(range(len(pool)))
        random.shuffle(all_indices)
        val_indices = all_indices[:n_val]
        test_indices = all_indices[n_val:n_val + n_test]
        
        for idx in val_indices:
            rec_copy = dict(pool[idx])
            add_split_field(rec_copy, "validation")
            held_out["val"][lang].append(rec_copy)
            held_out_doc_ids.add(pool[idx]["doc_id"])
        for idx in test_indices:
            rec_copy = dict(pool[idx])
            add_split_field(rec_copy, "test")
            held_out["test"][lang].append(rec_copy)
            held_out_doc_ids.add(pool[idx]["doc_id"])
    
    val_n = len(held_out["val"][lang])
    test_n = len(held_out["test"][lang])
    print(f"    val   sampled: {val_n:,} (target {VAL_PER_LANG:,})")
    print(f"    test  sampled: {test_n:,} (target {TEST_PER_LANG:,})")


# ─────────────────────────────────────────────────────────────────
# PHASE 3: Build train pools (everything except held-out)
# ─────────────────────────────────────────────────────────────────
print(f"\n[{now()}] PHASE 3: Build train pools (exclude held-out)")

train_pools = {}   # lang -> origin -> list of records
for lang in LANGS:
    train_pools[lang] = {}
    for origin, recs in all_data[lang].items():
        kept = [r for r in recs if r["doc_id"] not in held_out_doc_ids]
        train_pools[lang][origin] = kept
        print(f"  {lang}/{origin:<12} {len(kept):>10,} kept (after held-out)")


# ─────────────────────────────────────────────────────────────────
# PHASE 4: Build Path A (Swahili cap 2M)
# ─────────────────────────────────────────────────────────────────
def build_path(name, swahili_cap, train_pools_local):
    print(f"\n[{now()}] Building {name} (Swahili cap = {swahili_cap:,})")
    out_path = OUTPUT_DIR / name / "train.jsonl"
    composition = Counter()
    
    with open(out_path, "w") as f:
        for lang in LANGS:
            for origin, recs in train_pools_local[lang].items():
                # Apply Swahili cap only to swahili authentic + swahili sheng
                # Actually: cap applies to TOTAL Swahili authentic (the dominant source)
                # Sheng is kept fully (it's a different register)
                # Synthetic Swahili is kept fully (small anyway)
                
                if lang == "swahili" and origin == "authentic":
                    if len(recs) > swahili_cap:
                        # Sample swahili_cap records uniformly
                        chosen = random.sample(recs, swahili_cap)
                    else:
                        chosen = recs
                else:
                    chosen = recs
                
                # Shuffle within source for randomness
                random.shuffle(chosen)
                
                for r in chosen:
                    rec_copy = dict(r)
                    add_split_field(rec_copy, "train")
                    f.write(json.dumps(rec_copy, ensure_ascii=False) + "\n")
                    register = rec_copy.get("register", "monolingual")
                    composition[(lang, origin, register)] += 1
    
    # Print composition
    total = sum(composition.values())
    print(f"  {name} written to {out_path}")
    print(f"  Total records: {total:,}")
    print(f"  Composition:")
    for (lang, origin, reg), n in sorted(composition.items()):
        pct = n / total * 100
        print(f"    {lang:<10} {origin:<12} {reg:<14} {n:>10,} ({pct:>5.2f}%)")
    
    return composition, total


composition_A, total_A = build_path("pathA", SWAHILI_CAP_A, train_pools)
composition_D, total_D = build_path("pathD", SWAHILI_CAP_D, train_pools)


# ─────────────────────────────────────────────────────────────────
# PHASE 5: Write validation + test (shared across both paths)
# ─────────────────────────────────────────────────────────────────
print(f"\n[{now()}] PHASE 5: Write shared validation + test")

for split_name, file_name in [("val", "validation_shared.jsonl"), ("test", "test_shared.jsonl")]:
    out_path = OUTPUT_DIR / file_name
    all_recs = []
    for lang in LANGS:
        for r in held_out[split_name][lang]:
            all_recs.append(r)
    random.shuffle(all_recs)
    with open(out_path, "w") as f:
        for r in all_recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    
    print(f"  wrote {out_path}: {len(all_recs):,} records")
    
    # Distribution check
    lang_counter = Counter(r["language"] for r in all_recs)
    for lang, c in sorted(lang_counter.items()):
        print(f"    {lang}: {c}")


# ─────────────────────────────────────────────────────────────────
# PHASE 6: Write manifest
# ─────────────────────────────────────────────────────────────────
print(f"\n[{now()}] PHASE 6: Write manifest")

manifest = {
    "stage": "splits_v2",
    "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "seed": SEED,
    "dry_run": DRY_RUN,
    "dry_run_limit": DRY_RUN_LIMIT if DRY_RUN else None,
    "design": {
        "validation_records_per_lang": VAL_PER_LANG,
        "test_records_per_lang": TEST_PER_LANG,
        "swahili_cap_A": SWAHILI_CAP_A,
        "swahili_cap_D": SWAHILI_CAP_D,
        "stratify_val_test_by_origin": True,
    },
    "input_sources": {
        lang: {origin: str(path) for origin, path in SOURCES[lang].items()}
        for lang in LANGS
    },
    "outputs": {
        "validation_shared": str(OUTPUT_DIR / "validation_shared.jsonl"),
        "test_shared":       str(OUTPUT_DIR / "test_shared.jsonl"),
        "pathA_train":       str(OUTPUT_DIR / "pathA" / "train.jsonl"),
        "pathD_train":       str(OUTPUT_DIR / "pathD" / "train.jsonl"),
    },
    "totals": {
        "pathA_train": total_A,
        "pathD_train": total_D,
        "n_held_out_doc_ids": len(held_out_doc_ids),
    },
    "composition_pathA": {f"{l}__{o}__{r}": n for (l, o, r), n in composition_A.items()},
    "composition_pathD": {f"{l}__{o}__{r}": n for (l, o, r), n in composition_D.items()},
}

with open(OUTPUT_DIR / "manifest.json", "w") as f:
    json.dump(manifest, f, indent=2, ensure_ascii=False)

print(f"  wrote {OUTPUT_DIR/'manifest.json'}")
print()
print(f"[{now()}] STAGE 5 COMPLETE")
print(f"  Path A train: {total_A:,} records  (Swahili cap {SWAHILI_CAP_A:,})")
print(f"  Path D train: {total_D:,} records  (Swahili cap {SWAHILI_CAP_D:,})")
print(f"  Shared val:   {sum(len(v) for v in held_out['val'].values()):,} records")
print(f"  Shared test:  {sum(len(v) for v in held_out['test'].values()):,} records")
print(f"  Total elapsed: {time.time()-t_phase1_start:.0f}s")
