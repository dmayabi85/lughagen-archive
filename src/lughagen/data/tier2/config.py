"""
LughaGen Tier 2 Configuration — v1.2 Final
All 8 critical fixes + subtle improvements applied.
"""

PIPELINE_VERSION = "tier2-v1.2"
PROMPT_VERSION   = "tier2-v1.2"
DATASET_VERSION  = "lughagen-tier2-v1.2"

MODEL_70B = "meta-llama/Meta-Llama-3.1-70B-Instruct"
MODEL_8B  = "meta-llama/Meta-Llama-3.1-8B-Instruct"

QUALITY_MIN    = 0.65
PERPLEXITY_MAX = 160.0   # FIX3: was 250 — tightened

AQT_FLOOR   = 0.45
AQT_CEILING = 0.68
AQT_MARGIN  = 0.08
AQT_WINDOW  = 2000

# FIX8: explicit sampling — all in config not hardcoded
TEMPERATURE_70B    = 0.85
TEMPERATURE_8B     = 0.90
TOP_P              = 0.95
REPETITION_PENALTY = 1.05
MAX_NEW_TOKENS     = 280   # FIX: was 500
MIN_OUTPUT_CHARS   = 20    # FIX: was 20
LOGPROBS           = 5     # FIX1: was 1 — needed for accurate perplexity

BATCH_SIZE_MIN_70B  = 40
BATCH_SIZE_MAX_70B  = 120
BATCH_SIZE_INIT_70B = 60
BATCH_SIZE_MIN_8B   = 40
BATCH_SIZE_MAX_8B   = 120
BATCH_SIZE_INIT_8B  = 60

# FIX4: restored to Tier 1 sensitivity (was 8x2)
LSH_BANDS = 16
LSH_ROWS  = 4

DIVERSITY_MAX_SIM = 0.92
DIVERSITY_WINDOW  = 500

CURRICULUM_WEIGHTS = {
    "dialogue_conversion":  0.25,
    "context_expansion":    0.25,
    "code_switching":       0.20,
    "noise_robust_rewrite": 0.15,
    "domain_specific":      0.15,
}
STYLES  = list(CURRICULUM_WEIGHTS.keys())
WEIGHTS = list(CURRICULUM_WEIGHTS.values())

STYLE_BALANCE_WINDOW     = 1000
STYLE_MIN_FRACTION       = 0.08
CODE_SWITCH_MAX_FRACTION = 0.30
CODE_SWITCH_WORD_MAX     = 0.35

CHECKPOINT_EVERY = 500
SEED_MAX_FAILS   = 5
SEED_FAIL_WINDOW = 10

BASE = "/workspace/data/cleaned_final"
OUT  = "/workspace/synthetic/tier2"
LOGS = "/workspace/synthetic/logs/tier2"
META = "/workspace/data/meta/tier2"
TOK  = "/workspace/data/processed/swahili_tokenizer.model"

LANGUAGE_CONFIGS = {
    "kamba": {
        "inputs": [f"{BASE}/kamba_clean.jsonl"],
        "output": f"{OUT}/kamba_tier2.jsonl",
        "log": f"{LOGS}/kamba.log",
        "telemetry": f"{LOGS}/kamba_telemetry.jsonl",
        "fail_log": f"{LOGS}/kamba_failures.jsonl",
        "checkpoint": f"{OUT}/kamba_checkpoint.json",
        "dedup_db": f"{OUT}/kamba_dedup.db",
        "meta_out": f"{META}/kamba_final.json",
        "tokenizer": TOK, "model_id": MODEL_70B,
        "target": 150_000, "use_70b": True,
    },
    "kikuyu": {
        "inputs": [f"{BASE}/kikuyu_clean.jsonl"],
        "output": f"{OUT}/kikuyu_tier2.jsonl",
        "log": f"{LOGS}/kikuyu.log",
        "telemetry": f"{LOGS}/kikuyu_telemetry.jsonl",
        "fail_log": f"{LOGS}/kikuyu_failures.jsonl",
        "checkpoint": f"{OUT}/kikuyu_checkpoint.json",
        "dedup_db": f"{OUT}/kikuyu_dedup.db",
        "meta_out": f"{META}/kikuyu_final.json",
        "tokenizer": TOK, "model_id": MODEL_70B,
        "target": 150_000, "use_70b": True,
    },
    "luo": {
        "inputs": [f"{BASE}/luo_clean.jsonl"],
        "output": f"{OUT}/luo_tier2.jsonl",
        "log": f"{LOGS}/luo.log",
        "telemetry": f"{LOGS}/luo_telemetry.jsonl",
        "fail_log": f"{LOGS}/luo_failures.jsonl",
        "checkpoint": f"{OUT}/luo_checkpoint.json",
        "dedup_db": f"{OUT}/luo_dedup.db",
        "meta_out": f"{META}/luo_final.json",
        "tokenizer": TOK, "model_id": MODEL_70B,
        "target": 150_000, "use_70b": True,
    },
    "swahili": {
        "inputs": [f"{BASE}/swahili_clean.jsonl"],
        "output": f"{OUT}/swahili_tier2.jsonl",
        "log": f"{LOGS}/swahili.log",
        "telemetry": f"{LOGS}/swahili_telemetry.jsonl",
        "fail_log": f"{LOGS}/swahili_failures.jsonl",
        "checkpoint": f"{OUT}/swahili_checkpoint.json",
        "dedup_db": f"{OUT}/swahili_dedup.db",
        "meta_out": f"{META}/swahili_final.json",
        "tokenizer": TOK, "model_id": MODEL_8B,
        "target": 150_000, "use_70b": False,
    },
}

