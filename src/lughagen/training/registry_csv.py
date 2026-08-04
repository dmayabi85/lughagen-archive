#!/usr/bin/env python3
"""
/workspace/scripts/registry_csv.py

LughaGen Training Framework v1.5 §7.1, §7.2 (priority 1: CSV)
CSV backend for the experiment registry. One row per run with the full §7.1
schema flattened to CSV columns. Lives at /workspace/registry/runs.csv.

Also manages /workspace/registry/global_constants.yaml (§7.4) with
immutable-write enforcement: once a value is non-null, it cannot be overwritten.
"""

from __future__ import annotations

import csv
import datetime as dt
import json
import os
import shutil
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional

import yaml


REGISTRY_DIR = Path("/workspace/registry")
RUNS_CSV = REGISTRY_DIR / "runs.csv"
GLOBAL_CONSTANTS_YAML = REGISTRY_DIR / "global_constants.yaml"


# §7.1 — Full per-run schema. Nested fields are flattened with dot notation for CSV.
SCHEMA_FIELDS = [
    "experiment_id",
    "factorial_run_id",
    "phase",                                # Phase0_Tokenizer | Phase1_Corpus | Phase2_Factorial | Phase3_PEFT
    "is_confirmation_run",
    "confirmation_slot_type",               # null | pph_predicted | paired_sensitivity
    "spec_version",
    "spec_hash",

    # factors (§3.1)
    "factors.lr_schedule",
    "factors.batch_size",
    "factors.optimiser",
    "factors.sequence_length",
    "factors.data_ordering",

    "seed",

    # dataset (§5.3)
    "dataset.version",
    "dataset.checksum_sha256",
    "dataset.manifests_used",               # serialised JSON list
    "dataset.language_split",               # serialised JSON dict
    "dataset.origin_split",                 # serialised JSON dict
    "dataset.register_split",               # serialised JSON dict
    "dataset.total_words",
    "dataset.total_tokens_llama",
    "dataset.total_documents",
    "dataset.recipe",

    "tokenizer_version",

    # hyperparameters
    "hyperparameters.peak_lr",
    "hyperparameters.warmup_steps",
    "hyperparameters.weight_decay",
    "hyperparameters.max_grad_norm",
    "hyperparameters.lora_r",
    "hyperparameters.lora_alpha",
    "hyperparameters.lora_dropout",
    "hyperparameters.num_train_epochs",
    "hyperparameters.lion_implementation",

    # gpu_config
    "gpu_config.gpu_count",
    "gpu_config.gpu_type",
    "gpu_config.precision",
    "gpu_config.gradient_checkpointing",
    "gpu_config.flash_attention_version",
    "gpu_config.flash_attention_engaged",   # §7.1 v1.5

    "gpu_hours_consumed",
    "wall_clock_seconds",
    "checkpoint_paths.intermediate",        # serialised JSON list
    "checkpoint_paths.final",

    # confirmation_run_metadata
    "confirmation_run_metadata.predicted_pph_mean",
    "confirmation_run_metadata.predicted_pph_bootstrap_low",
    "confirmation_run_metadata.predicted_pph_bootstrap_high",
    "confirmation_run_metadata.selection_rank",
    "confirmation_run_metadata.flagged_for_manual_review",

    # metrics
    "metrics.baseline_perplexity",
    "metrics.baseline_perplexity_per_lang", # serialised JSON dict
    "metrics.validation_perplexity",
    "metrics.validation_perplexity_per_lang",  # serialised JSON dict
    "metrics.pph_score",
    "metrics.pph_log_ratio",                # §4.5 v1.5
    "metrics.pph_flag_negative",
    "metrics.tokens_per_sec",
    "metrics.peak_vram_gb",
    "metrics.gpu_utilization_pct",
    "metrics.convergence_steps_to_5pct",
    "metrics.token_fertility_ratio",
    "metrics.per_language_grad_norm_ratio_mean",   # §9.4 v1.5
    "metrics.per_language_grad_norm_ratio_per_lang",  # serialised JSON dict

    # training_events (serialised JSON lists where applicable)
    "training_events.gradient_anomalies",
    "training_events.instability_events",
    "training_events.runtime_failures",
    "training_events.early_stopped",
    "training_events.early_stop_reason",

    # environment
    "environment.torch_version",
    "environment.transformers_version",
    "environment.peft_version",
    "environment.deepspeed_version",
    "environment.cuda_version",
    "environment.flash_attn_version",
    "environment.lion_pytorch_version",
    "environment.hostname",
    "environment.datetime_utc",
]


# Fields whose value is a dict but should be stored as a single JSON column
# (NOT flattened further). Derived from SCHEMA_FIELDS by inspection.
_JSON_LEAF_FIELDS = {
    "dataset.language_split",
    "dataset.origin_split",
    "dataset.register_split",
    "metrics.baseline_perplexity_per_lang",
    "metrics.validation_perplexity_per_lang",
    "metrics.per_language_grad_norm_ratio_per_lang",
}


def _flatten(obj: dict, prefix: str = "") -> dict:
    """
    Flatten a nested dict with dot-separated keys.

    Dicts whose dot-path is in _JSON_LEAF_FIELDS are serialised as a single
    JSON string column rather than being flattened further. This keeps the CSV
    schema stable: e.g., `dataset.language_split` is one column containing
    `{"swahili": 0.75, ...}` rather than four columns
    `dataset.language_split.swahili`, etc.
    """
    out: dict[str, Any] = {}
    for k, v in obj.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict) and key in _JSON_LEAF_FIELDS:
            out[key] = json.dumps(v, ensure_ascii=False)
        elif isinstance(v, dict):
            out.update(_flatten(v, key))
        elif isinstance(v, list):
            # Serialise lists as JSON for CSV portability
            out[key] = json.dumps(v, ensure_ascii=False)
        else:
            out[key] = v
    return out


def _serialise_for_csv(record: dict) -> dict:
    """Convert a record to a CSV-writable row (all values strings)."""
    flat = _flatten(record)
    row: dict[str, str] = {}
    for field_name in SCHEMA_FIELDS:
        v = flat.get(field_name)
        if v is None:
            row[field_name] = ""
        elif isinstance(v, bool):
            row[field_name] = "true" if v else "false"
        elif isinstance(v, (dict, list)):
            row[field_name] = json.dumps(v, ensure_ascii=False)
        else:
            row[field_name] = str(v)
    return row


def ensure_registry() -> None:
    """Create the registry directory and CSV header if missing."""
    REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
    if not RUNS_CSV.exists():
        with RUNS_CSV.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=SCHEMA_FIELDS)
            writer.writeheader()


def write_run(record: dict) -> None:
    """
    Append a single run record to runs.csv. Record must be a dict matching the
    §7.1 schema (nested form accepted). Missing fields are written as empty.
    """
    ensure_registry()
    row = _serialise_for_csv(record)
    with RUNS_CSV.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SCHEMA_FIELDS)
        writer.writerow(row)


def read_all_runs() -> list[dict]:
    """Read all runs from runs.csv. Returns a list of flat dicts (strings)."""
    if not RUNS_CSV.exists():
        return []
    with RUNS_CSV.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_runs_by_phase(phase: str) -> list[dict]:
    """Filter runs by phase (e.g. 'Phase2_Factorial')."""
    return [r for r in read_all_runs() if r.get("phase") == phase]


# ─────────────────────────────────────────────────────────────────────────────
# Global constants (§7.4) with immutable-write enforcement
# ─────────────────────────────────────────────────────────────────────────────

def load_global_constants() -> dict:
    """Load /workspace/registry/global_constants.yaml; return empty dict if missing."""
    if not GLOBAL_CONSTANTS_YAML.exists():
        return {}
    with GLOBAL_CONSTANTS_YAML.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def write_global_constant(key_path: str, value: Any) -> None:
    """
    Write a value to global_constants.yaml.

    key_path is a dot-separated path (e.g. 'tost_delta_pph.value').
    If the leaf already has a non-null value, raises RuntimeError — the
    immutable-write rule from §7.4.

    Intermediate dict keys may be overwritten only if their leaves are null.
    """
    constants = load_global_constants()

    parts = key_path.split(".")
    # Walk to parent, creating dicts as needed
    parent = constants
    for p in parts[:-1]:
        if p not in parent or parent[p] is None:
            parent[p] = {}
        elif not isinstance(parent[p], dict):
            raise RuntimeError(
                f"global_constants[{key_path}]: parent {p} is not a dict"
            )
        parent = parent[p]

    leaf = parts[-1]
    existing = parent.get(leaf)
    if existing is not None and existing != value:
        raise RuntimeError(
            f"Refusing to overwrite global_constants.{key_path}: "
            f"existing={existing!r}, new={value!r}. "
            f"§7.4 forbids overwriting non-null constants."
        )

    parent[leaf] = value

    REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
    with GLOBAL_CONSTANTS_YAML.open("w", encoding="utf-8") as f:
        yaml.safe_dump(constants, f, sort_keys=True, default_flow_style=False)


def get_global_constant(key_path: str, default: Any = None) -> Any:
    """Read a value from global_constants.yaml; return default if absent."""
    constants = load_global_constants()
    parts = key_path.split(".")
    node: Any = constants
    for p in parts:
        if not isinstance(node, dict) or p not in node:
            return default
        node = node[p]
    return node


def utc_now_iso() -> str:
    """Current UTC timestamp in ISO 8601 with second precision."""
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def empty_record_template() -> dict:
    """
    Return a fresh empty record matching the §7.1 schema. All optional fields
    are None; nested groupings present so callers can dot-set without KeyError.
    """
    return {
        "experiment_id": None,
        "factorial_run_id": None,
        "phase": None,
        "is_confirmation_run": False,
        "confirmation_slot_type": None,
        "spec_version": None,
        "spec_hash": None,
        "factors": {
            "lr_schedule": None,
            "batch_size": None,
            "optimiser": None,
            "sequence_length": None,
            "data_ordering": None,
        },
        "seed": None,
        "dataset": {
            "version": None,
            "checksum_sha256": None,
            "manifests_used": [],
            "language_split": {},
            "origin_split": {},
            "register_split": {},
            "total_words": None,
            "total_tokens_llama": None,
            "total_documents": None,
            "recipe": None,
        },
        "tokenizer_version": None,
        "hyperparameters": {
            "peak_lr": None,
            "warmup_steps": None,
            "weight_decay": None,
            "max_grad_norm": None,
            "lora_r": None,
            "lora_alpha": None,
            "lora_dropout": None,
            "num_train_epochs": None,
            "lion_implementation": None,
        },
        "gpu_config": {
            "gpu_count": None,
            "gpu_type": None,
            "precision": None,
            "gradient_checkpointing": None,
            "flash_attention_version": None,
            "flash_attention_engaged": None,
        },
        "gpu_hours_consumed": 0.0,
        "wall_clock_seconds": 0,
        "checkpoint_paths": {
            "intermediate": [],
            "final": None,
        },
        "confirmation_run_metadata": {
            "predicted_pph_mean": None,
            "predicted_pph_bootstrap_low": None,
            "predicted_pph_bootstrap_high": None,
            "selection_rank": None,
            "flagged_for_manual_review": False,
        },
        "metrics": {
            "baseline_perplexity": None,
            "baseline_perplexity_per_lang": {
                "swahili": None, "luo": None, "kikuyu": None, "kamba": None,
            },
            "validation_perplexity": None,
            "validation_perplexity_per_lang": {
                "swahili": None, "luo": None, "kikuyu": None, "kamba": None,
            },
            "pph_score": None,
            "pph_log_ratio": None,
            "pph_flag_negative": False,
            "tokens_per_sec": None,
            "peak_vram_gb": None,
            "gpu_utilization_pct": None,
            "convergence_steps_to_5pct": None,
            "token_fertility_ratio": None,
            "per_language_grad_norm_ratio_mean": None,
            "per_language_grad_norm_ratio_per_lang": {
                "swahili": None, "luo": None, "kikuyu": None, "kamba": None,
            },
        },
        "training_events": {
            "gradient_anomalies": [],
            "instability_events": [],
            "runtime_failures": [],
            "early_stopped": False,
            "early_stop_reason": "",
        },
        "environment": {
            "torch_version": "",
            "transformers_version": "",
            "peft_version": "",
            "deepspeed_version": "",
            "cuda_version": "",
            "flash_attn_version": "",
            "lion_pytorch_version": "",
            "hostname": "",
            "datetime_utc": "",
        },
    }


def populate_environment_from_lock(record: dict) -> dict:
    """
    Populate the record's environment block from environment_lock.yaml.
    Called at run start by train_orchestrator.py.
    """
    lock_path = REGISTRY_DIR / "environment_lock.yaml"
    if not lock_path.exists():
        raise FileNotFoundError(
            f"{lock_path} missing; run verify_environment.sh first"
        )
    with lock_path.open("r") as f:
        lock = yaml.safe_load(f)

    versions = lock.get("verified_versions", {})
    record["environment"]["torch_version"] = versions.get("torch", "")
    record["environment"]["transformers_version"] = versions.get("transformers", "")
    record["environment"]["peft_version"] = versions.get("peft", "")
    record["environment"]["deepspeed_version"] = versions.get("deepspeed", "")
    record["environment"]["cuda_version"] = versions.get("cuda_torch", "")
    record["environment"]["flash_attn_version"] = versions.get("flash_attn", "")
    record["environment"]["lion_pytorch_version"] = versions.get("lion_pytorch", "")
    record["environment"]["hostname"] = lock.get("hostname", "")
    record["environment"]["datetime_utc"] = utc_now_iso()
    return record


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test / CLI
# ─────────────────────────────────────────────────────────────────────────────

def _smoke_test() -> int:
    """Round-trip test: write a record, read it back, verify schema compliance."""
    print("Running registry_csv.py round-trip smoke test...")

    # Use a temp directory to avoid polluting the real registry
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        global REGISTRY_DIR, RUNS_CSV, GLOBAL_CONSTANTS_YAML
        saved = (REGISTRY_DIR, RUNS_CSV, GLOBAL_CONSTANTS_YAML)
        REGISTRY_DIR = Path(tmp)
        RUNS_CSV = REGISTRY_DIR / "runs.csv"
        GLOBAL_CONSTANTS_YAML = REGISTRY_DIR / "global_constants.yaml"

        try:
            rec = empty_record_template()
            rec["experiment_id"] = "EXP-SMOKE-001"
            rec["factorial_run_id"] = "RUN-P2-C01-S1"
            rec["phase"] = "Phase2_Factorial"
            rec["spec_version"] = "1.5"
            rec["factors"]["lr_schedule"] = "cosine"
            rec["factors"]["batch_size"] = 128
            rec["factors"]["optimiser"] = "AdamW"
            rec["factors"]["sequence_length"] = 1024
            rec["factors"]["data_ordering"] = "curriculum"
            rec["seed"] = 42
            rec["dataset"]["language_split"] = {
                "swahili": 0.7521, "luo": 0.0997, "kikuyu": 0.0776, "kamba": 0.0706
            }
            rec["dataset"]["manifests_used"] = ["swahili_sheng_v1", "luo_raw_residue_v1"]
            rec["metrics"]["validation_perplexity"] = 5.71
            rec["metrics"]["pph_score"] = 0.123
            rec["metrics"]["pph_log_ratio"] = 0.0987

            write_run(rec)
            rows = read_all_runs()
            assert len(rows) == 1, f"Expected 1 row, got {len(rows)}"
            assert rows[0]["experiment_id"] == "EXP-SMOKE-001"
            assert rows[0]["factors.batch_size"] == "128"
            assert rows[0]["seed"] == "42"

            # Verify language_split round-trips as JSON
            ls = json.loads(rows[0]["dataset.language_split"])
            assert abs(ls["swahili"] - 0.7521) < 1e-9

            # Test global constants immutability
            write_global_constant("baseline_perplexity.pooled", 9.84)
            assert get_global_constant("baseline_perplexity.pooled") == 9.84
            try:
                write_global_constant("baseline_perplexity.pooled", 10.0)
                print("  FAIL: overwrite of non-null constant should have raised")
                return 1
            except RuntimeError:
                pass  # expected

            # Writing the same value is idempotent (allowed)
            write_global_constant("baseline_perplexity.pooled", 9.84)

            print("  PASS: round-trip + immutable-write checks")
            return 0
        finally:
            REGISTRY_DIR, RUNS_CSV, GLOBAL_CONSTANTS_YAML = saved


if __name__ == "__main__":
    import sys
    sys.exit(_smoke_test())
