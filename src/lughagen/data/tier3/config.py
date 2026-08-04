"""
config.py — LughaGen Tier 3 Dataset Intelligence Layer
Shared configuration across all Tier 3 modules.

Edit values here, not in individual modules.
"""

from pathlib import Path

# ── VERSION CONTROL ──────────────────────────────────────────────
TIER3_VERSION    = "v1.0"
DATASET_VERSION  = "lughagen-mixed-tier3-v1.0"

# ── PATHS ────────────────────────────────────────────────────────
WORKSPACE        = Path("/workspace")
TIER1_DIR        = WORKSPACE / "synthetic" / "tier1"
TIER2_DIR        = WORKSPACE / "synthetic" / "tier2"
TIER3_DIR        = WORKSPACE / "synthetic" / "tier3"
LOG_DIR          = WORKSPACE / "logs" / "tier3"
MODEL_DIR        = WORKSPACE / "models"
FINAL_DIR        = WORKSPACE / "data" / "final_training"

# Tier 1 inputs (filtered, post-quality-gate)
TIER1_INPUTS = {
    "kikuyu": TIER2_DIR / "kikuyu_final_dataset_v2.jsonl",
    "kamba":  TIER2_DIR / "kamba_final_dataset_v2.jsonl",
    "luo":    TIER2_DIR / "luo_final_dataset_v2.jsonl",
}

# Tier 2 input (filtered Swahili, post-quality-gate)
TIER2_INPUT = TIER2_DIR / "swahili_final_dataset_v2.jsonl"

# Tier 3 outputs (intermediate + final)
ANALYSIS_REPORT  = TIER3_DIR / "analysis_report.json"
ANALYSIS_PLOTS   = TIER3_DIR / "plots"
TOPICS_OUTPUT    = TIER3_DIR / "topics_classified.jsonl"
FLUENCY_OUTPUT   = TIER3_DIR / "fluency_scored.jsonl"
QUALITY_OUTPUT   = TIER3_DIR / "quality_classified.jsonl"
FINAL_OUTPUT     = FINAL_DIR / "lughagen_train.jsonl"
DATASET_SUMMARY  = FINAL_DIR / "dataset_summary.json"

# Trained classifier checkpoint
CLASSIFIER_DIR   = MODEL_DIR / "swahili_quality_classifier"

# Progress tracking (SQLite)
PROGRESS_DB      = TIER3_DIR / "tier3_progress.db"

# ── MODELS ───────────────────────────────────────────────────────
# Sentence embeddings (used by analyzer + reshaper for diversity)
EMBED_MODEL_ID    = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Fluency scorer (smaller external model — second opinion)
FLUENCY_MODEL_ID  = "Davlan/afro-xlmr-base"   # works for Swahili + many African langs
# Alternative: "castorini/afriberta_base" or "bigscience/bloomz-560m"

# Quality classifier base (fine-tuned by classifier.py)
CLF_BASE_MODEL_ID = "Davlan/afro-xlmr-base"

# Topic classifier — uses sentence embeddings + anchor sentences (no extra model)
# (Cheaper than zero-shot classification with an LLM; good enough for tagging)

# ── TOPIC TAXONOMY ───────────────────────────────────────────────
# Categories used for distribution control. Anchor sentences (Swahili)
# are embedded once; samples are tagged by nearest-anchor cosine similarity.
TOPIC_ANCHORS = {
    "education": [
        "Mwalimu anafundisha wanafunzi darasani.",
        "Mtihani wa shule ulikuwa mgumu sana.",
        "Chuo kikuu kinatoa elimu ya juu.",
    ],
    "health": [
        "Daktari amemchunguza mgonjwa hospitalini.",
        "Chanjo inakinga magonjwa.",
        "Lishe bora ni muhimu kwa afya.",
    ],
    "agriculture": [
        "Mkulima analima shamba lake la mahindi.",
        "Mvua imenyesha vizuri msimu huu wa kilimo.",
        "Mbolea inaongeza mavuno ya mazao.",
    ],
    "business": [
        "Biashara yake imeongeza faida mwaka huu.",
        "Soko la hisa lilipanda jana.",
        "Benki imetoa mkopo wa kuanzisha biashara.",
    ],
    "technology": [
        "Simu ya mkononi imebadilisha mawasiliano.",
        "Programu ya kompyuta inafanya kazi haraka.",
        "Intaneti inawezesha mawasiliano ya haraka.",
    ],
    "news_politics": [
        "Rais ametangaza sera mpya ya serikali.",
        "Bunge limepitisha sheria ya bajeti.",
        "Uchaguzi mkuu utafanyika mwaka ujao.",
    ],
    "culture_society": [
        "Sherehe ya harusi ilihudhuriwa na wengi.",
        "Mila na desturi za kabila zinaheshimika.",
        "Wimbo wa zamani uliimbwa na bibi.",
    ],
    "daily_life": [
        "Asubuhi nilikunywa chai na mkate.",
        "Nilipanda basi kwenda kazini.",
        "Tutakutana sokoni jioni.",
    ],
}

# Maximum proportion of dataset any single topic may occupy.
# 8 topics, equal share = 0.125. We allow some skew to 0.20 for
# naturally over-represented topics (daily_life, news).
MAX_TOPIC_RATIO   = 0.20
MIN_TOPIC_RATIO   = 0.04   # keep at least this many per topic if available

# ── QUALITY THRESHOLDS ───────────────────────────────────────────
HEURISTIC_WEIGHT   = 0.6   # weight for original heuristic quality score
FLUENCY_WEIGHT     = 0.4   # weight for external fluency score
COMBINED_QUALITY_MIN = 0.60  # final hard gate after combining

# Classifier confidence threshold — drop if model says "bad" with this confidence
CLF_BAD_THRESHOLD  = 0.70

# ── DIVERSITY CONTROL ────────────────────────────────────────────
# Cosine similarity above this = considered near-duplicate at semantic level
SEMANTIC_DUP_THRESHOLD = 0.92

# Number of clusters for K-means (used to oversample under-represented regions
# of semantic space when reshaping)
N_SEMANTIC_CLUSTERS = 50

# ── TARGET DATASET COMPOSITION ───────────────────────────────────
# Target sample count per language in final lughagen_train.jsonl.
# If less is available after filtering, take what we have and log it.
TARGET_PER_LANGUAGE = {
    "swahili": 80_000,
    "kikuyu":  60_000,
    "kamba":   60_000,
    "luo":     60_000,
}

# ── BATCHING & PERFORMANCE ───────────────────────────────────────
EMBED_BATCH_SIZE   = 64
CLF_BATCH_SIZE     = 32
FLUENCY_BATCH_SIZE = 16   # small — these models are heavier
CHECKPOINT_EVERY   = 5_000

# ── LOGGING ──────────────────────────────────────────────────────
LOG_LEVEL = "INFO"
LOG_FORMAT_JSON = True   # structured JSONL telemetry alongside human logs

# ── INIT DIRS ────────────────────────────────────────────────────
def ensure_dirs():
    """Create all output dirs idempotently. Call from each entry script."""
    for d in [TIER3_DIR, ANALYSIS_PLOTS, LOG_DIR, MODEL_DIR, FINAL_DIR, CLASSIFIER_DIR]:
        d.mkdir(parents=True, exist_ok=True)
