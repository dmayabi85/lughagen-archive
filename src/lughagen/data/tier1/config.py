"""
LughaGen Tier 1 Configuration
Clean Parallel Corpus Layer — conservative, high-quality generation
"""

PIPELINE_VERSION = "tier1-v1.0"
PROMPT_VERSION   = "tier1-v1.0"
DATASET_VERSION  = "lughagen-tier1-v1.0"

MODEL_70B = "meta-llama/Meta-Llama-3.1-70B-Instruct"
MODEL_8B  = "meta-llama/Meta-Llama-3.1-8B-Instruct"

QUALITY_MIN      = 0.75
PERPLEXITY_MAX   = 150.0

AQT_FLOOR        = 0.45
AQT_CEILING      = 0.68
AQT_MARGIN       = 0.08
AQT_WINDOW       = 2000

BATCH_SIZE_MIN_70B  = 40
BATCH_SIZE_MAX_70B  = 120
BATCH_SIZE_INIT_70B = 60
BATCH_SIZE_MIN_8B   = 80
BATCH_SIZE_MAX_8B  = 80
BATCH_SIZE_INIT_8B = 40

CHECKPOINT_EVERY = 500
GPU_UTIL_MAX     = 1.01  # Disabled — 100% is target not a ceiling
GPU_UTIL_SLEEP   = 1.0

LSH_BANDS = 8
LSH_ROWS  = 2

DIVERSITY_MAX_SIM = 1.00
DIVERSITY_WINDOW  = 50

CURRICULUM_WEIGHTS = {
    "paraphrase_simple":  0.35,
    "paraphrase_formal":  0.20,
    "spoken":             0.20,
    "idiomatic":          0.15,
    "dialect_variation":  0.10,
}

SEED_MAX_FAILS   = 5
SEED_FAIL_WINDOW = 10

BASE  = "/workspace/data/cleaned_final"
OUT   = "/workspace/synthetic/tier1"
LOGS  = "/workspace/synthetic/logs/tier1"
META  = "/workspace/data/meta/tier1"
TOK   = "/workspace/data/processed/swahili_tokenizer.model"

LANGUAGE_CONFIGS = {
    "kamba": {
        "inputs":      [f"{BASE}/kamba_clean.jsonl"],
        "output":      f"{OUT}/kamba_tier1.jsonl",
        "log":         f"{LOGS}/kamba.log",
        "telemetry":   f"{LOGS}/kamba_telemetry.jsonl",
        "fail_log":    f"{LOGS}/kamba_failures.jsonl",
        "checkpoint":  f"{OUT}/kamba_checkpoint.json",
        "dedup_db":    f"{OUT}/kamba_dedup.db",
        "meta_out":    f"{META}/kamba_final.json",
        "tokenizer":   TOK,
        "model_id":    MODEL_70B,
        "target":      100_000,
        "use_70b":     True,
    },
    "kikuyu": {
        "inputs":      [f"{BASE}/kikuyu_clean.jsonl"],
        "output":      f"{OUT}/kikuyu_tier1.jsonl",
        "log":         f"{LOGS}/kikuyu.log",
        "telemetry":   f"{LOGS}/kikuyu_telemetry.jsonl",
        "fail_log":    f"{LOGS}/kikuyu_failures.jsonl",
        "checkpoint":  f"{OUT}/kikuyu_checkpoint.json",
        "dedup_db":    f"{OUT}/kikuyu_dedup.db",
        "meta_out":    f"{META}/kikuyu_final.json",
        "tokenizer":   TOK,
        "model_id":    MODEL_70B,
        "target":      100_000,
        "use_70b":     True,
    },
    "luo": {
        "inputs":      [f"{BASE}/luo_clean.jsonl"],
        "output":      f"{OUT}/luo_tier1.jsonl",
        "log":         f"{LOGS}/luo.log",
        "telemetry":   f"{LOGS}/luo_telemetry.jsonl",
        "fail_log":    f"{LOGS}/luo_failures.jsonl",
        "checkpoint":  f"{OUT}/luo_checkpoint.json",
        "dedup_db":    f"{OUT}/luo_dedup.db",
        "meta_out":    f"{META}/luo_final.json",
        "tokenizer":   TOK,
        "model_id":    MODEL_70B,
        "target":      100_000,
        "use_70b":     True,
    },
    "swahili": {
        "inputs":      [f"{BASE}/swahili_clean.jsonl"],
        "output":      f"{OUT}/swahili_tier1.jsonl",
        "log":         f"{LOGS}/swahili.log",
        "telemetry":   f"{LOGS}/swahili_telemetry.jsonl",
        "fail_log":    f"{LOGS}/swahili_failures.jsonl",
        "checkpoint":  f"{OUT}/swahili_checkpoint.json",
        "dedup_db":    f"{OUT}/swahili_dedup.db",
        "meta_out":    f"{META}/swahili_final.json",
        "tokenizer":   TOK,
        "model_id":    MODEL_8B,
        "target":      100_000,
        "use_70b":     False,
    },
}
