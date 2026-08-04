#!/usr/bin/env python3
"""
/workspace/scripts/train_orchestrator.py

LughaGen Training Framework v1.5.1 (updated Week 4)
Top-level entry point for any training run. Responsibilities:

  1. Parse CLI args matching the §3.1 factor schema
  2. Run preflight checks (§12) — fail fast if env not ready
  3. Build RunConfig from CLI + factorial schedule
  4. Stamp run record with spec_version, environment_lock_hash, factorial_run_id
  5. Invoke training_loop.run_training()
  6. Receive metrics dict; merge with run-level metadata
  7. Write completed record to runs.csv via registry_csv.write_run()
  8. (optional) Mirror to SQLite via registry_sqlite.sync()

This module is intentionally thin. The substantive training code lives in
training_loop.py so this file remains stable across spec changes.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Optional

sys.path.insert(0, "/workspace/scripts")
from registry_csv import (
    write_run, REGISTRY_DIR, utc_now_iso, get_global_constant,
)
from training_loop import RunConfig, run_training
from pph_calculator import compute_pph


def hash_file(path: Path) -> str:
    """SHA-256 of a file's contents."""
    if not path.exists():
        return ""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def load_p_baseline(sequence_length: int = 1024) -> Optional[float]:
    """
    Load P_baseline from §4.2 output, matched to the cell's sequence_length.

    Reads /workspace/registry/baseline_perplexity.json. If the JSON contains
    a ``by_sequence_length`` map, looks up the baseline for the given
    ``sequence_length`` (V2 methodology supports per-seq-length baselines).
    Falls back to ``pooled_perplexity`` for backward compatibility.

    Returns None if no baseline is recorded (e.g. during smoke tests before
    Phase 0).
    """
    bpath = REGISTRY_DIR / "baseline_perplexity.json"
    if not bpath.exists():
        return None
    try:
        data = json.loads(bpath.read_text())
        # Prefer per-seq-length lookup if available
        by_seq = data.get("by_sequence_length", {})
        seq_key = str(sequence_length)
        if seq_key in by_seq:
            return float(by_seq[seq_key])
        # Backward compatibility: single pooled value
        return float(data["pooled_perplexity"])
    except (KeyError, ValueError, json.JSONDecodeError):
        return None


def build_run_record(args, config: RunConfig, metrics_dict: dict) -> dict:
    """Assemble the full §7.1 schema record from inputs and training output."""

    env_lock_path = REGISTRY_DIR / "environment_lock.yaml"
    spec_versions_path = REGISTRY_DIR / "spec_versions.yaml"

    record = {
        # Identity
        "experiment_id": config.experiment_id,
        "factorial_run_id": config.factorial_run_id,
        "phase": config.phase,
        "is_confirmation_run": False,
        "spec_version": config.spec_version,
        "environment_lock_hash": hash_file(env_lock_path),
        "spec_versions_hash": hash_file(spec_versions_path),
        "started_at": config.started_at,
        "completed_at": utc_now_iso(),
        "seed": config.seed,

        # Factors (§3.1)
        "factors.lr_schedule": config.lr_schedule,
        "factors.batch_size": config.batch_size,
        "factors.optimiser": config.optimiser,
        "factors.sequence_length": config.sequence_length,
        "factors.data_ordering": config.data_ordering,

        # Hyperparameters
        "hyperparameters.peak_lr": config.peak_lr,
        "hyperparameters.warmup_steps": config.warmup_steps,
        "hyperparameters.weight_decay": config.weight_decay,
        "hyperparameters.max_grad_norm": config.max_grad_norm,
        "hyperparameters.lora_r": config.lora_r,
        "hyperparameters.lora_alpha": config.lora_alpha,
        "hyperparameters.lora_dropout": config.lora_dropout,
        "hyperparameters.num_train_epochs": 1,

        # GPU config
        "gpu_config.gpu_count": int(os.environ.get("WORLD_SIZE", "1")),
        "gpu_config.gradient_checkpointing": config.gradient_checkpointing,
        "gpu_config.attn_implementation": config.attn_implementation,

        # Dataset (filled in by orchestrator from corpus stats)
        "dataset.recipe": "full",
        "dataset.snapshot_id": "corpus_snapshot_sheng_relabeled_2026-05-17T163736Z",

        # Tokenizer
        "tokenizer.source": metrics_dict.pop("tokenizer_source", "unknown"),

        # Metrics from training_loop
        **{k: v for k, v in metrics_dict.items() if k.startswith("metrics.")},
        **{k: v for k, v in metrics_dict.items() if k.startswith("training_events.")},
        **{k: v for k, v in metrics_dict.items() if k in (
            "gpu_hours_consumed", "wall_clock_seconds"
        )},
    }

    # Carry the FA2 verification result
    fa2 = metrics_dict.get("gpu_config.flash_attention_engaged")
    if fa2 is not None:
        record["gpu_config.flash_attention_engaged"] = fa2

    # ===== §7.1 registry field wiring =====
    # Wire all required audit / provenance / metric fields that don't come
    # from metrics_dict directly. Each helper is defensive — failures here
    # write None/empty rather than crashing the run.

    import socket
    import datetime as _dt
    import yaml as _yaml

    # ---- Environment versions (importlib.metadata, with fallbacks) ----
    def _pkg_version(name, import_name=None):
        try:
            from importlib.metadata import version
            return version(name)
        except Exception:
            try:
                mod = __import__(import_name or name.replace("-", "_"))
                return getattr(mod, "__version__", None)
            except Exception:
                return None

    record["environment.torch_version"] = _pkg_version("torch")
    record["environment.transformers_version"] = _pkg_version("transformers")
    record["environment.peft_version"] = _pkg_version("peft")
    record["environment.deepspeed_version"] = _pkg_version("deepspeed")
    record["environment.flash_attn_version"] = _pkg_version("flash-attn", "flash_attn")
    record["environment.lion_pytorch_version"] = _pkg_version("lion-pytorch", "lion_pytorch")
    try:
        import torch as _torch
        record["environment.cuda_version"] = _torch.version.cuda
    except Exception:
        record["environment.cuda_version"] = None
    record["environment.hostname"] = socket.gethostname()
    record["environment.datetime_utc"] = _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

    # ---- Baseline perplexity (from §7.4 pinned global_constants) ----
    # IMPORTANT: V2 supports per-seq-length baselines (1024 vs 2048).
    # Use config.p_baseline (already seq-length-matched via load_p_baseline)
    # to keep the registry self-consistent with PPH calculations.
    try:
        _gc = _yaml.safe_load(open("/workspace/registry/global_constants.yaml"))
        _bp = _gc.get("baseline_perplexity", {})
        # Use the seq-length-matched value actually used in PPH calc
        record["metrics.baseline_perplexity"] = (
            config.p_baseline if config.p_baseline is not None else _bp.get("pooled")
        )
        record["metrics.baseline_perplexity_per_lang"] = _bp.get("per_language")

        _tv = _gc.get("tokenizer_version", {})
        record["tokenizer_version"] = _tv.get("value")
    except Exception as _e:
        rank0_print(f"[record] could not read global_constants: {_e}")

    # ---- Spec hash (from spec_versions.yaml) ----
    try:
        _spec = _yaml.safe_load(open("/workspace/registry/spec_versions.yaml"))
        _current = _spec.get("current") or _spec.get("active")
        if _current:
            record["spec_hash"] = _current.get("sha256") or _current.get("hash")
    except Exception:
        pass

    # ---- Dataset provenance from corpus_stats.json ----
    try:
        import json as _json
        _cs = _json.load(open("/workspace/data/corpus/stats/corpus_stats.json"))
        _overall = _cs.get("overall", {})
        record["dataset.total_words"]     = _overall.get("words")
        record["dataset.total_documents"] = _overall.get("documents")
        record["dataset.total_tokens_llama"] = _overall.get("tokens_llama")
        # by_language structure for dataset.language_split
        record["dataset.language_split"] = _json.dumps(_cs.get("by_language", {})) if _cs.get("by_language") else None
        record["dataset.origin_split"]   = _json.dumps(_cs.get("by_origin", {})) if _cs.get("by_origin") else None
        record["dataset.register_split"] = _json.dumps(_cs.get("by_register", {})) if _cs.get("by_register") else None
    except Exception as _e:
        rank0_print(f"[record] could not read corpus_stats: {_e}")

    # ---- Tokens/sec from throughput calibration (§7.4 pinned) ----
    try:
        _gc = _yaml.safe_load(open("/workspace/registry/global_constants.yaml"))
        _tc = _gc.get("phase2_throughput_calibration", {})
        record["metrics.tokens_per_sec"] = _tc.get("micro_benchmark_tokens_per_sec")
    except Exception:
        pass

    # ---- pph_flag_negative for easy filtering downstream ----
    if record.get("metrics.pph_score") is not None:
        try:
            record["metrics.pph_flag_negative"] = float(record["metrics.pph_score"]) < 0
        except (TypeError, ValueError):
            pass

    return record


def main() -> int:
    parser = argparse.ArgumentParser(
        description="LughaGen training orchestrator (§3.1 factor entry)"
    )
    # Identity
    parser.add_argument("--phase", required=True,
                        choices=["Phase0_Tokenizer", "Phase1_Recipe",
                                 "Phase2_Factorial", "Phase3_PEFT"])
    parser.add_argument("--factorial-run-id", required=True,
                        help="From factorial_schedule.csv, e.g. RUN-P2-C01-S1")
    parser.add_argument("--seed", type=int, required=True)

    # Factor levels
    parser.add_argument("--lr-schedule", choices=["cosine", "linear_warmup_constant"],
                        required=True)
    parser.add_argument("--batch-size", type=int, choices=[128, 256], required=True)
    parser.add_argument("--optimiser", choices=["AdamW", "Lion"], required=True)
    parser.add_argument("--sequence-length", type=int, choices=[1024, 2048], required=True)
    parser.add_argument("--data-ordering", choices=["curriculum", "mixed"], required=True)

    # Phase 3 specifics (only used when phase=Phase3_PEFT)
    parser.add_argument("--peft-method",
                        choices=["qlora_r16", "qlora_r64", "ia3", "prefix_tuning", "full_ft", "cpt"],
                        default="qlora_r16",
                        help="Phase 3 PEFT method")

    parser.add_argument("--peak-lr", type=float, default=None)
    parser.add_argument("--warmup-steps", type=int, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--deepspeed-config", type=str, default=None)
    # Recipe + budget
    parser.add_argument("--recipe", default="full",
                        help="Corpus recipe (default: full)")
    parser.add_argument("--max-steps", type=int, default=6000)

    # Mode flags
    parser.add_argument("--smoke-test", action="store_true",
                        help="Short training run to validate plumbing")
    parser.add_argument("--no-eval", action="store_true",
                        help="Skip evaluation pass during/after training (for benchmarking)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print plan and exit without training")
    parser.add_argument("--skip-preflight", action="store_true",
                        help="Skip preflight.sh (NOT recommended; for debug only)")

    # Paths
    parser.add_argument("--output-dir", type=Path,
                        default=None,
                        help="Override checkpoint output; defaults to /workspace/checkpoints/<run_id>")
    parser.add_argument("--model-path", default="/workspace/models/llama-3.1-8b-base")
    parser.add_argument("--train-jsonl", default=None,
                        help="Override training data path. If unset, RunConfig "
                             "default (splits/train.jsonl) is used.")
    parser.add_argument("--val-jsonl", default=None,
                        help="Override validation data path. If unset, RunConfig "
                             "default (splits/validation.jsonl) is used.")

    args = parser.parse_args()

    # ----- Preflight (§12) -----
    if not args.skip_preflight and not args.dry_run:
        print(f"[preflight] running checks for phase={args.phase}")
        try:
            result = subprocess.run(
                ["bash", "/workspace/scripts/preflight.sh", args.phase],
                check=True, capture_output=False,
            )
        except subprocess.CalledProcessError as e:
            print(f"ERROR: preflight failed (exit {e.returncode})", file=sys.stderr)
            return 2

    # ----- Build experiment ID -----
    experiment_id = (
        f"EXP-{dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
        f"-{uuid.uuid4().hex[:8].upper()}"
    )
    started_at = utc_now_iso()

    # ----- Phase 3 PEFT method routing -----
    use_qlora_4bit = True
    lora_r = 16
    if args.phase == "Phase3_PEFT":
        if args.peft_method == "full_ft":
            use_qlora_4bit = False
        elif args.peft_method == "cpt":
            use_qlora_4bit = False
        elif args.peft_method == "qlora_r64":
            lora_r = 64
        elif args.peft_method in ("ia3", "prefix_tuning"):
            # These don't use QLoRA but also don't need DeepSpeed; supported
            # but the actual peft_config swap is left to a Week 6 update.
            print(f"WARNING: peft_method={args.peft_method} not yet fully wired; "
                  f"falling back to qlora_r16 for now.", file=sys.stderr)

    # ----- Build RunConfig -----
    # Compute the output_dir default if not explicitly provided.
    # Help text promises: "defaults to /workspace/checkpoints/<run_id>".
    if args.output_dir is None:
        args.output_dir = Path("/workspace/checkpoints") / args.factorial_run_id

    # Build optional path overrides (only set if user supplied --train-jsonl/--val-jsonl;
    # otherwise RunConfig uses its dataclass defaults from training_loop.py)
    path_overrides = {}
    if args.train_jsonl:
        path_overrides["train_file"] = args.train_jsonl
    if args.val_jsonl:
        path_overrides["val_file"] = args.val_jsonl

    hp_overrides = {}
    if args.peak_lr is not None: hp_overrides["peak_lr"] = args.peak_lr
    if args.warmup_steps is not None: hp_overrides["warmup_steps"] = args.warmup_steps
    if args.weight_decay is not None: hp_overrides["weight_decay"] = args.weight_decay
    if args.deepspeed_config is not None: hp_overrides["deepspeed_config"] = args.deepspeed_config

    config = RunConfig(
        factorial_run_id=args.factorial_run_id,
        experiment_id=experiment_id,
        phase=args.phase,
        spec_version="1.5.1",
        seed=args.seed,
        lr_schedule=args.lr_schedule,
        batch_size=args.batch_size,
        optimiser=args.optimiser,
        sequence_length=args.sequence_length,
        data_ordering=args.data_ordering,
        max_steps=args.max_steps,
        smoke_test=args.smoke_test,
        no_eval=args.no_eval,
        dry_run=args.dry_run,
        output_dir=args.output_dir,
        model_path=args.model_path,
        use_qlora_4bit=use_qlora_4bit,
        lora_r=lora_r,
        p_baseline=load_p_baseline(sequence_length=args.sequence_length),
        started_at=started_at,
        **path_overrides,
        **hp_overrides,
    )

    # ----- Run training -----
    metrics_dict = run_training(config)

    # ----- Write registry record -----
    # Rank guard: only rank 0 writes the registry (distributed: 8 processes else duplicate rows)
    if int(os.environ.get("RANK", "0")) != 0:
        return 0
    if not args.dry_run and not args.smoke_test:
        record = build_run_record(args, config, metrics_dict)
        write_run(record)
        print(f"[registry] wrote run record for {experiment_id}")

        # Optional: sync SQLite mirror
        try:
            from registry_sqlite import sync as sqlite_sync
            sqlite_sync(verbose=False)
        except Exception as e:
            print(f"[registry] SQLite sync skipped: {e}", file=sys.stderr)
    elif args.smoke_test:
        print(f"[smoke] skipping registry write (smoke tests don't persist to runs.csv)")
    elif args.dry_run:
        print(f"[dry-run] skipping registry write")

    return 0


if __name__ == "__main__":
    sys.exit(main())
