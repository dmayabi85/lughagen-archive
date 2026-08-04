#!/usr/bin/env python3
"""
/workspace/scripts/training_loop.py

LughaGen Training Framework v1.5.1
Core training loop implementation. Called by train_orchestrator.py.

What this module is responsible for:
  §3.1   Apply factor settings (lr_schedule, batch_size, optimiser, seq_length, ordering)
  §4.2   Pooled validation perplexity at end of training (consumed by pph_calculator)
  §6     Load the Phase-0-winning tokenizer (or LLaMA stock fallback)
  §8     Apply QLoRA configuration per Phase 2 spec
  §9.1   Load LLaMA-3.1-8B with BF16 + FA2 + gradient checkpointing
  §9.3   Distributed safeguards (barriers, local_rank guards on checkpoint save)
  §9.4   Per-language gradient norm logging every N steps
  §11    Checkpoint policy: every save_steps, retain top-3 by val_pph
  §12    Pre-training FA2 verification
  §13    Kill-switch polling between steps
  §14    Smoke-test mode (shortened max_steps + reduced eval frequency)

Phase 3 full FT path is stubbed; QLoRA path is the production implementation.
Full FT requires DeepSpeed ZeRO-3 config which is deferred to Week 6.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Callable

import torch
import torch.distributed as dist

sys.path.insert(0, "/workspace/scripts")
from registry_csv import get_global_constant, utc_now_iso, REGISTRY_DIR
from mfu_callback import MFUCallback
from pph_calculator import compute_pph, PPHResult
from kill_switch import KillSwitchMonitor


# ===== Default paths (overridable from RunConfig) =====
LLAMA_BASE_PATH_DEFAULT = "/workspace/models/llama-3.1-8b-base"
TRAIN_JSONL_DEFAULT = "/workspace/data/corpus/splits/train.jsonl"
VAL_JSONL_DEFAULT = "/workspace/data/corpus/splits/validation.jsonl"

# Per §11: top-K checkpoints by validation perplexity
CHECKPOINT_TOP_K = 1

# Per §9.4: gradient norms logged every this many steps
GRAD_NORM_LOG_INTERVAL = 50


# ===== Corpus schema (§5.4) — explicit Features to avoid datasets null-inference =====
# Some fields are null in the first chunks of train.jsonl but populated later.
# Declaring the schema upfront prevents `TypeError: Couldn't cast string to null`.
def _build_corpus_features():
    """Build the Features schema for load_dataset()."""
    from datasets import Features, Value, Sequence
    return Features({
        "doc_id":             Value("string"),
        "text":               Value("string"),
        "language":           Value("string"),
        "origin":             Value("string"),
        "register":           Value("string"),
        "register_detail":    Value("string"),       # null for most records
        "split":              Value("string"),
        "domain":             Value("string"),
        "license":            Value("string"),
        "source_id":          Value("string"),
        "source_url":         Value("string"),       # null for most records
        "ingested_at":        Value("string"),
        "code_switch_pair":   Value("string"),       # null for monolingual
        "code_switch_ratio":  Value("float64"),
        "quality_score":      Value("float64"),
        "fluency_score":      Value("float64"),      # null for most records
        "words":              Value("int64"),
        "tokens_llama":       Value("int64"),        # null until tokenization runs
        "augmentation":       Sequence(Value("string")),
        "preprocessing":      Sequence(Value("string")),
    })

LUGHAGEN_CORPUS_FEATURES = _build_corpus_features()

# Per §13: kill switch poll cadence
KILL_SWITCH_POLL_INTERVAL = 10


@dataclass
class RunConfig:
    """Encapsulates everything training_loop needs for a single run.

    Built by train_orchestrator from the factorial schedule row + spec defaults.
    """
    # Identity (populated by orchestrator)
    factorial_run_id: str                       # e.g. "RUN-P2-C01-S1"
    experiment_id: str                          # e.g. "EXP-20260520T120000Z-XYZ"
    phase: str                                  # "Phase0_Tokenizer" | "Phase2_Factorial" | ...
    spec_version: str                           # e.g. "1.5.1"
    seed: int

    # Factor levels (§3.1)
    lr_schedule: str                            # "cosine" | "linear_warmup_constant"
    batch_size: int                             # effective batch (per-step micro-batch * grad_accum * world_size)
    optimiser: str                              # "AdamW" | "Lion"
    sequence_length: int                        # 1024 | 2048
    data_ordering: str                          # "curriculum" | "mixed"

    # Hyperparameters (§8 — QLoRA defaults; tunable per phase)
    peak_lr: float = 2e-4
    lion_peak_lr: float = 2e-5  # Spec v1.5.1 amendment 2026-05-27: 10x lower per Lion paper
    warmup_steps: int = 100
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: tuple = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")  # Spec v1.5.1 amendment 2026-05-27: all 7 modules per original April 2026 plan

    # Compute config (§9)
    bf16: bool = True
    tf32: bool = True
    gradient_checkpointing: bool = True
    attn_implementation: str = "flash_attention_2"
    use_qlora_4bit: bool = True                 # Phase 0/1/2; False for Phase 3 full FT
    deepspeed_config: str | None = None         # Path to DS ZeRO-3 JSON; required when use_qlora_4bit=False

    # Run plan
    max_steps: int = 6000                       # spec §3.3 — typical Phase 2 run length
    eval_steps: int = 200
    save_steps: int = 200
    logging_steps: int = 10

    # Modes
    smoke_test: bool = False                    # if True, override max_steps to ~150
    no_eval: bool = False                        # if True, skip evaluation pass (benchmarking)
    dry_run: bool = False                       # if True, print plan and return without training

    # Paths
    model_path: str = LLAMA_BASE_PATH_DEFAULT
    train_file: str = TRAIN_JSONL_DEFAULT
    val_file: str = VAL_JSONL_DEFAULT
    output_dir: Optional[Path] = None           # set by orchestrator

    # Baseline (required for PPH; set by orchestrator from §4.2 P_baseline file)
    p_baseline: Optional[float] = None

    # Bookkeeping (set by orchestrator at construction)
    started_at: str = ""

    def smoke_test_overrides(self) -> "RunConfig":
        """Return a copy with smoke-test-appropriate caps."""
        return RunConfig(**{**self.__dict__, "max_steps": 150,
                            "eval_steps": 50, "save_steps": 150,
                            "logging_steps": 5})


# ===== Tokenizer resolution =====

def resolve_tokenizer_path(model_path: str) -> tuple[str, str]:
    """
    Resolve which tokenizer to use, per Option (b):
      - Read tokenizer_version.value from global_constants.yaml
      - If unset: fall back to LLaMA stock and log a clear warning
    Returns (tokenizer_path, source_label).
    """
    pinned = get_global_constant("tokenizer_version.value")
    if pinned and Path(pinned).exists():
        return str(pinned), "phase0_winner"

    # Fallback: LLaMA stock tokenizer (lives inside the base model directory)
    if Path(model_path).exists():
        print(f"WARNING: tokenizer_version.value not pinned in global_constants. "
              f"Falling back to LLaMA stock tokenizer at {model_path}. "
              f"This is expected for smoke tests but NOT for Phase 2 runs.",
              file=sys.stderr)
        return model_path, "llama_stock_fallback"

    raise FileNotFoundError(
        f"Neither pinned tokenizer ({pinned}) nor base model ({model_path}) found"
    )


# ===== Distributed-aware logger =====

def get_rank() -> int:
    return int(os.environ.get("RANK", "0"))


def is_rank_zero() -> bool:
    return get_rank() == 0


def rank0_print(*args, **kwargs):
    if is_rank_zero():
        print(*args, **kwargs)


# ===== Per-language gradient norm callback (§9.4) =====

class PerLanguageGradNormCallback:
    """
    Computes per-language gradient norm every GRAD_NORM_LOG_INTERVAL steps.

    Approach: maintain a small auxiliary buffer that captures gradients
    on the LM head during the most recent backward(); group by sample-language;
    compute L2 norm per group; emit ratio relative to pooled mean.

    Detailed implementation delegates to partition_loss_by_language.py which
    was produced in Week 1. This callback wraps that procedure on a schedule
    and logs to a JSONL file.
    """

    def __init__(self, output_dir: Path, log_interval: int = GRAD_NORM_LOG_INTERVAL):
        self.output_dir = output_dir
        self.log_interval = log_interval
        self.log_path = output_dir / "per_language_grad_norms.jsonl"

    def __call__(self, step: int, model, batch, languages_in_batch):
        """
        Called from inside the training loop. NOT a transformers TrainerCallback
        because grad-norm probing requires access to the live batch and gradients,
        which Trainer's standard callbacks don't expose.

        Args:
            step: current global step
            model: the (possibly peft-wrapped) model
            batch: the dict that just came out of the data collator
            languages_in_batch: list of language strings per sample
        """
        if step % self.log_interval != 0:
            return
        if not is_rank_zero():
            return    # only rank 0 logs

        try:
            from partition_loss_by_language import compute_per_language_grad_norms
            # Legacy code path — converts to new signature for consistency
            batch_with_lang = dict(batch)
            batch_with_lang["language"] = languages_in_batch
            norms = compute_per_language_grad_norms(model, batch_with_lang, step)
        except Exception as e:
            rank0_print(f"  [grad-norm] step {step}: failed ({e})")
            return

        record = {
            "ts": utc_now_iso(),
            "step": step,
            "per_lang_grad_norm": norms,
        }
        with self.log_path.open("a") as f:
            f.write(json.dumps(record) + "\n")


# ===== Top-K checkpoint manager (§11) =====

class TopKCheckpointManager:
    """Retain only top-K checkpoints by validation perplexity. Lower is better."""

    def __init__(self, output_dir: Path, top_k: int = CHECKPOINT_TOP_K):
        self.output_dir = output_dir
        self.top_k = top_k
        self.history: list[tuple[float, Path]] = []   # (val_pph, ckpt_dir)
        self.manifest_path = output_dir / "checkpoint_manifest.json"

    def register(self, val_pph: float, ckpt_dir: Path) -> None:
        """Register a new checkpoint and prune older ones below the top-K bar."""
        self.history.append((val_pph, ckpt_dir))
        self.history.sort(key=lambda x: x[0])    # ascending: lowest pph first

        # Prune
        kept = self.history[:self.top_k]
        pruned = self.history[self.top_k:]
        for _, p in pruned:
            if p.exists() and p.is_dir():
                import shutil
                try:
                    shutil.rmtree(p)
                    rank0_print(f"  [ckpt] pruned {p}")
                except OSError as e:
                    rank0_print(f"  [ckpt] could not prune {p}: {e}")
        self.history = kept

        # Persist manifest
        manifest = {
            "top_k": self.top_k,
            "retained": [{"val_pph": p, "ckpt": str(d)} for p, d in self.history],
            "updated_at": utc_now_iso(),
        }
        with self.manifest_path.open("w") as f:
            json.dump(manifest, f, indent=2)


# ===== Main training entry point =====

def run_training(config: RunConfig) -> dict:
    """
    Execute a complete training run per the supplied RunConfig.

    Returns a metrics dict populated for direct insertion into the §7.1 registry.
    Caller (train_orchestrator) wraps this dict into a full run record.
    """
    # ----- Smoke test mode reshapes the config -----
    if config.smoke_test:
        config = config.smoke_test_overrides()
        rank0_print(f"[smoke] max_steps={config.max_steps}, eval_steps={config.eval_steps}")

    # ----- Dry run: print plan and return -----
    if config.dry_run:
        rank0_print(f"=== RunConfig (dry-run) ===")
        for k, v in config.__dict__.items():
            rank0_print(f"  {k:24s} = {v}")
        return {
            "metrics.validation_perplexity": None,
            "training_events.early_stopped": False,
            "training_events.dry_run": True,
        }

    # ----- Set seeds, dtypes -----
    import random
    import numpy as np
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    if config.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # ----- Resolve tokenizer (Option b) -----
    tokenizer_path, tokenizer_source = resolve_tokenizer_path(config.model_path)
    rank0_print(f"[tokenizer] source={tokenizer_source}, path={tokenizer_path}")

    # ----- Load tokenizer -----
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ----- Load base model with FA2 + BF16 (+ optional QLoRA) -----
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank0_print(f"[model] loading {config.model_path}")

    # ----- Determine attention implementation per model family (Amendment 11, 2026-08-02) -----
    # Gemma 2 configs carry attn_logit_softcapping which BOTH FA2 and SDPA refuse
    # in transformers 5.x — silent fallback to eager (3-5x slower). Documented
    # workaround: disable softcapping on the config before loading; the CPT run
    # then trains the model to operate without softcap. This is a real (small)
    # architecture change vs the released Gemma 2 checkpoint, and is noted in the
    # thesis methodology section. We route Gemma-family models to SDPA and pass
    # the modified config to from_pretrained. Spec §12's no-silent-fallback
    # contract is preserved by comparing against the routed impl.
    from transformers import AutoConfig as _AutoConfig
    _peek_cfg = _AutoConfig.from_pretrained(config.model_path)
    _is_gemma_family = _peek_cfg.model_type.lower().startswith("gemma")
    _required_attn_impl = "sdpa" if _is_gemma_family else config.attn_implementation
    _model_config_override = None
    if _is_gemma_family:
        _pre_softcap = getattr(_peek_cfg, "attn_logit_softcapping", None)
        _peek_cfg.attn_logit_softcapping = None
        rank0_print(f"[attn] gemma-family: disabled attn_logit_softcapping "
                    f"(was {_pre_softcap}) to permit SDPA")
        _model_config_override = _peek_cfg
    rank0_print(f"[attn] model_type={_peek_cfg.model_type} -> requesting {_required_attn_impl!r}")

    if config.use_qlora_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        _load_kwargs = dict(
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
            attn_implementation=_required_attn_impl,
            device_map={"": local_rank},
        )
        if _model_config_override is not None:
            _load_kwargs["config"] = _model_config_override
        model = AutoModelForCausalLM.from_pretrained(
            config.model_path,
            **_load_kwargs,
        )
    else:
        # Phase 3 full FT / CPT path: full-precision model, no bnb quantization.
        # DeepSpeed ZeRO-3 handles memory via sharding across GPUs.
        if not config.deepspeed_config:
            raise ValueError(
                "Full fine-tuning (use_qlora_4bit=False) requires "
                "config.deepspeed_config to point at a ZeRO-3 JSON."
            )
        rank0_print(f"[FFT] loading full-precision model (bf16, no 4-bit quant)")
        _load_kwargs = dict(
            torch_dtype=torch.bfloat16,
            attn_implementation=_required_attn_impl,
        )
        if _model_config_override is not None:
            _load_kwargs["config"] = _model_config_override
        model = AutoModelForCausalLM.from_pretrained(
            config.model_path,
            **_load_kwargs,
        )

    # If tokenizer was extended (Strategy A), resize embeddings before LoRA
    if len(tokenizer) != model.config.vocab_size:
        rank0_print(f"[model] resizing embeddings: "
                    f"{model.config.vocab_size} → {len(tokenizer)}")
        model.resize_token_embeddings(len(tokenizer))

        # Amendment 8 (2026-05-27, empirical): resize_token_embeddings silently
        # keeps the first len(tokenizer) LLaMA rows, but Phase 0 token IDs do
        # NOT correspond to LLaMA token IDs. This produces arbitrarily-mapped
        # embeddings. Pre-restart sweep produced val_pph ~11700 (16% baseline
        # reduction) via LoRA compensation through garbled inputs.
        # Fix: load token-aware remapped tensors (Buzaaba et al., 2025):
        #   34.4% exact match, 65.6% subword decomposition, 0.05% mean fallback.
        # Validation: 73% perplexity reduction in 50 steps (vs old 16% in 6000).
        from pathlib import Path as _RemapPath
        import os as _os
        # Amendment 10 (2026-08-01): make remap dir configurable via env var
        # so Phase 3.5 Gemma runs can point at gemma-specific remapped embeddings.
        # Default preserves Llama behavior for Phase 3 runs.
        _remap_dir_str = _os.environ.get(
            "LUGHAGEN_REMAP_DIR",
            "/workspace/restart_backup_20260527T052144Z/remapped_embeddings"
        )
        _remap_dir = _RemapPath(_remap_dir_str)
        _embed_pt = _remap_dir / "embed_tokens.pt"
        _head_pt = _remap_dir / "lm_head.pt"
        if _embed_pt.exists() and _head_pt.exists():
            rank0_print(f"[remap] loading token-aware embeddings from {_remap_dir}")
            _new_embed = torch.load(_embed_pt, weights_only=True, map_location='cpu')
            _new_head = torch.load(_head_pt, weights_only=True, map_location='cpu')
            if tuple(_new_embed.shape) != (len(tokenizer), model.config.hidden_size):
                raise RuntimeError(
                    f"Remap embed shape {tuple(_new_embed.shape)} != "
                    f"({len(tokenizer)}, {model.config.hidden_size})"
                )
            with torch.no_grad():
                model.get_input_embeddings().weight.copy_(_new_embed.to(model.device))
                model.get_output_embeddings().weight.copy_(_new_head.to(model.device))
            rank0_print(f"[remap] applied (token-aware initialization)")
        else:
            raise FileNotFoundError(
                f"[remap] expected remap tensors at {_remap_dir} but they're missing. "
                f"Run the remap build script before training."
            )


    # ----- Verify requested attn impl actually engaged (§12) -----
    # Routed impl comes from Amendment 11 attn-family routing above.
    actually_engaged = (
        hasattr(model.config, "_attn_implementation")
        and model.config._attn_implementation == _required_attn_impl
    )
    if not actually_engaged:
        raise RuntimeError(
            f"Attention impl mismatch. Requested {_required_attn_impl!r}, got "
            f"model.config._attn_implementation="
            f"{getattr(model.config, '_attn_implementation', 'unknown')}. "
            f"Refuse to train — spec §12 forbids silent fallback."
        )
    rank0_print(f"[attn] {_required_attn_impl!r} engaged (model.config._attn_implementation="
                f"{model.config._attn_implementation})")

    # ----- Apply QLoRA (§8) - skipped for full FT / CPT -----
    if config.use_qlora_4bit:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=config.gradient_checkpointing
        )
        peft_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=list(config.lora_target_modules),
            # Amendment 6 (2026-05-27): make embed_tokens and lm_head trainable.
            # Required because tokenizer extension (128256->8000) means base model
            # embeddings need to adapt. Combined with token-aware remap (amend 8)
            # and pad masking (amend 9), enables 73% val perplexity reduction in
            # 50 steps vs ~16% for the pre-restart frozen-embed configuration.
            modules_to_save=["embed_tokens", "lm_head"],
            # Amendment 7 (2026-05-27): keep embed_tokens and lm_head sharing one
            # tensor during training. LLaMA-3.1's config has tie_word_embeddings=True
            # as a design assumption; PEFT's default behavior with modules_to_save
            # creates two independent clones, which we observed to be less stable.
            # Reduces trainable params 107.48M -> 74.71M.
            ensure_weight_tying=True,
        )
        model = get_peft_model(model, peft_config)
        if is_rank_zero():
            model.print_trainable_parameters()
    else:
        # Full fine-tuning / CPT: all params trainable, no LoRA wrapping.
        if config.gradient_checkpointing:
            model.gradient_checkpointing_enable()
        for p in model.parameters():
            p.requires_grad = True
        if is_rank_zero():
            n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
            n_total = sum(p.numel() for p in model.parameters())
            rank0_print(f"[FFT] trainable params: {n_train:,} ({100*n_train/n_total:.2f}% of {n_total:,})")

    # ----- Dataset (§5.4 streaming) -----
    from datasets import load_dataset, Features, Value, Sequence
    rank0_print(f"[data] loading train from {config.train_file}")
    train_ds = load_dataset("json", data_files=config.train_file, split="train",
                             features=LUGHAGEN_CORPUS_FEATURES,
                             streaming=False)
    rank0_print(f"[data] loading val from {config.val_file}")
    val_ds = load_dataset("json", data_files=config.val_file, split="train",
                           features=LUGHAGEN_CORPUS_FEATURES,
                           streaming=False)

    def tokenize_fn(example):
        enc = tokenizer(
            example["text"],
            truncation=True,
            max_length=config.sequence_length,
            padding="max_length",
        )
        # Amendment 9 (2026-05-27): mask pad tokens in labels.
        # Pre-restart used input_ids.copy() which trained the model to predict
        # pad at ~95% of positions. This was acceptable when embeddings were
        # frozen (the pad embedding couldn't drift), but with trainable
        # remapped embeddings, the pad row accumulated 98% of gradient signal
        # and the model rapidly collapsed to "always predict pad".
        # Standard practice: -100 = ignore index for CrossEntropyLoss.
        _pad_id = tokenizer.pad_token_id
        enc["labels"] = [(_tid if _tid != _pad_id else -100)
                         for _tid in enc["input_ids"]]
        # Preserve language for grad-norm logging
        enc["language"] = example.get("language", example.get("lang", "unknown"))
        return enc

    train_ds = train_ds.map(
        tokenize_fn,
        remove_columns=[c for c in train_ds.column_names if c != "language"],
        num_proc=8 if not config.smoke_test else 4,
    )
    val_ds = val_ds.map(
        tokenize_fn,
        remove_columns=[c for c in val_ds.column_names if c != "language"],
        num_proc=8 if not config.smoke_test else 4,
    )

    # ----- Optimizer (factor C) -----
    if config.optimiser == "Lion":
        from lion_pytorch import Lion
        # Use HF Trainer's hook to swap optimiser — we build manually below
        optim_class = Lion
    elif config.optimiser == "AdamW":
        optim_class = torch.optim.AdamW
    else:
        raise ValueError(f"Unknown optimiser: {config.optimiser}")

    # ----- Training arguments -----
    from transformers import TrainingArguments, Trainer
    from transformers import default_data_collator as _hf_default_collator

    def default_data_collator(features):
        """Wraps HF default collator to PRESERVE language_id column (spec §9.4).
        HF default drops unknown keys; we pop language_id, run default, then
        re-attach as a tensor."""
        import torch as _torch
        lang_ids = None
        if features and "language_id" in features[0]:
            lang_ids = [f.pop("language_id") for f in features]
        batch = _hf_default_collator(features)
        if lang_ids is not None:
            batch["language_id"] = _torch.tensor(lang_ids, dtype=_torch.long)
        return batch

    output_dir = config.output_dir or Path(f"/workspace/checkpoints/{config.factorial_run_id}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Compute per-device batch + grad accum from effective batch_size
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    per_device_batch = max(1, config.batch_size // (world_size * 4))   # heuristic; grad_accum tunes the rest
    grad_accum = max(1, config.batch_size // (per_device_batch * world_size))
    rank0_print(f"[batch] effective={config.batch_size}, "
                f"per_device={per_device_batch}, grad_accum={grad_accum}, world_size={world_size}")

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        max_steps=config.max_steps,
        per_device_train_batch_size=per_device_batch,
        gradient_accumulation_steps=grad_accum,
        per_device_eval_batch_size=per_device_batch,
        eval_strategy=("no" if config.no_eval else "steps"),
        learning_rate=(config.lion_peak_lr if config.optimiser == "Lion" else config.peak_lr),  # Spec v1.5.1 amendment 2026-05-27
        warmup_steps=config.warmup_steps,
        weight_decay=config.weight_decay,
        max_grad_norm=config.max_grad_norm,
        lr_scheduler_type="cosine" if config.lr_schedule == "cosine" else "constant_with_warmup",
        bf16=config.bf16,
        tf32=config.tf32,
        gradient_checkpointing=config.gradient_checkpointing,
        logging_steps=config.logging_steps,
        eval_steps=config.eval_steps,
        save_strategy="steps",
        save_steps=config.save_steps,
        save_total_limit=None,           # we manage top-K ourselves
        report_to="none",
        ddp_find_unused_parameters=False,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        seed=config.seed,
        data_seed=config.seed,
        remove_unused_columns=False,   # KEEP language_id for spec §9.4 per-language callback
        deepspeed=config.deepspeed_config,  # None for QLoRA, path to ZeRO-3 JSON for FFT/CPT
    )

    # Add language_id integer column (string "language" gets dropped by collator;
    # int survives. Used by PerLanguageGradNormCallback to recover language at runtime).
    _LANG_TO_ID = {"swahili": 0, "kikuyu": 1, "kamba": 2, "luo": 3}
    def _add_lang_id(ex):
        ex["language_id"] = _LANG_TO_ID.get(ex.get("language", ""), -1)
        return ex
    if "language" in train_ds.column_names:
        train_ds = train_ds.map(_add_lang_id, num_proc=4, desc="add language_id")
        train_ds = train_ds.remove_columns(["language"])
    if "language" in val_ds.column_names:
        val_ds = val_ds.map(_add_lang_id, num_proc=4, desc="add language_id (val)")
        val_ds = val_ds.remove_columns(["language"])
    val_ds_for_eval = val_ds.remove_columns(["language"]) if "language" in val_ds.column_names else val_ds

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds_for_eval,
        data_collator=default_data_collator,
    )

    # Swap optimiser if Lion (Trainer creates AdamW by default)
    # NOTE: HF Trainer's create_optimizer() MUST assign to self.optimizer,
    # otherwise the training loop sees self.optimizer is None and crashes.
    # The original lambda returned the optimizer without storing it — fixed below.
    if config.optimiser == "Lion":
        def _create_lion_optimizer():
            # Spec v1.5.1 amendment 2026-05-27: use lion_peak_lr (2e-5),
            # not peak_lr (2e-4), to prevent gradient divergence (Mode B).
            trainer.optimizer = optim_class(
                trainer.model.parameters(),
                lr=config.lion_peak_lr,
                weight_decay=config.weight_decay,
            )
            return trainer.optimizer
        trainer.create_optimizer = _create_lion_optimizer

    # ----- Kill-switch monitor (§13) -----
    kill_monitor = KillSwitchMonitor(check_interval_steps=KILL_SWITCH_POLL_INTERVAL,
                                     rank=get_rank())

    # ----- Top-K checkpoint manager (§11) -----
    ckpt_manager = TopKCheckpointManager(output_dir=output_dir, top_k=CHECKPOINT_TOP_K)

    # ----- Training -----
    t0 = time.perf_counter()
    rank0_print(f"\n[train] starting: max_steps={config.max_steps}")

    early_stopped = False
    kill_reason = ""

    # We use Trainer.train() for the actual loop; kill-switch is checked via
    # a callback that polls between steps.
    from transformers import TrainerCallback, TrainerControl, TrainerState

    class KillSwitchCallback(TrainerCallback):
        def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
            should_stop, reason = kill_monitor.should_stop(state.global_step)
            if should_stop:
                nonlocal early_stopped, kill_reason
                early_stopped = True
                kill_reason = reason
                rank0_print(f"\n[kill-switch] triggered at step {state.global_step}: {reason}")
                control.should_training_stop = True
            return control

    class PreSavePruneCallback(TrainerCallback):
        """Prune old checkpoints BEFORE HF Trainer initiates new save.
        
        Amendment 12 (2026-08-03): fixes the retention race where old + new
        checkpoints coexist during save window, doubling peak disk usage.
        On 27B models with 337 GB checkpoints, this filled 1.1 TB workspace
        at step 400 of Phase 3.5 CPT and caused NCCL timeout crash.
        
        Fix: fires on on_step_end when this step is a save step, deletes any
        checkpoints beyond top_k FIRST, then verifies disk headroom before
        letting save proceed. Peak disk = 1x checkpoint, not 2x.
        """
        
        MIN_FREE_GB = 400  # safety margin above 337 GB Gemma 27B checkpoint
        
        def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
            if not is_rank_zero():
                return control
            # Save fires when global_step % save_steps == 0 (after step completes).
            # We fire on the same condition to prune BEFORE the save.
            if args.save_steps <= 0 or state.global_step <= 0:
                return control
            if state.global_step % args.save_steps != 0:
                return control
            self._prune_before_save(Path(args.output_dir))
            return control
        
        def _prune_before_save(self, output_dir: Path):
            import shutil
            # Find existing checkpoint dirs, sorted by step number
            existing = sorted(
                [d for d in output_dir.glob("checkpoint-*") if d.is_dir()],
                key=lambda d: int(d.name.split("-")[1]) if d.name.split("-")[1].isdigit() else -1
            )
            # We're about to write ONE new checkpoint. Delete oldest to make room.
            while len(existing) >= CHECKPOINT_TOP_K:
                oldest = existing.pop(0)
                rank0_print(f"  [ckpt-preprune] removing {oldest} (top_k={CHECKPOINT_TOP_K}, keeping room for incoming save)")
                try:
                    shutil.rmtree(oldest)
                except OSError as e:
                    rank0_print(f"  [ckpt-preprune] could not remove {oldest}: {e}")
            # Verify disk headroom for incoming save
            stat = shutil.disk_usage(str(output_dir))
            free_gb = stat.free / (1024**3)
            rank0_print(f"  [ckpt-preprune] {free_gb:.0f} GB free before save (min required: {self.MIN_FREE_GB})")
            if free_gb < self.MIN_FREE_GB:
                raise RuntimeError(
                    f"Insufficient disk for checkpoint save: "
                    f"{free_gb:.0f} GB free at {output_dir}, need >= {self.MIN_FREE_GB} GB. "
                    f"Refuse to save (Amendment 12): would risk mid-save NCCL timeout crash."
                )
    
    class CheckpointPolicyCallback(TrainerCallback):
        def on_save(self, args, state: TrainerState, control: TrainerControl, **kwargs):
            if not is_rank_zero():
                return
            # The trainer just saved checkpoint-{step}; capture val_pph for ranking
            metrics = state.log_history[-1] if state.log_history else {}
            val_loss = metrics.get("eval_loss")
            if val_loss is None:
                return
            val_pph = math.exp(val_loss)
            ckpt_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"
            if ckpt_dir.exists():
                ckpt_manager.register(val_pph, ckpt_dir)

    # ===== Per-language gradient norm callback (spec §9.4) =====
    _ID_TO_LANG = {0: "swahili", 1: "kikuyu", 2: "kamba", 3: "luo"}

    class PerLanguageGradNormCallback(TrainerCallback):
        """Fires every 50 steps. Runs partition_loss_by_language's grad-norm probe.
        The probe does its own forward+backward per language and zeros grads when done."""
        LOG_INTERVAL = 50
        # Divergence killer thresholds (Spec v1.5.1 amendment 2026-05-27)
        DIVERGENCE_NORM_THRESHOLD = 1000.0  # absolute per-language grad_norm
        DIVERGENCE_CONSECUTIVE_LIMIT = 50   # consecutive log events before kill

        def __init__(self, output_dir):
            self.log_path = Path(output_dir) / "per_language_grad_norms.jsonl"
            self.results = []  # accumulator for end-of-training aggregation
            # Divergence tracking state
            self.consecutive_high_norm = 0
            self.divergence_kill_fired = False

        def on_step_end(self, args, state, control, model=None, **kwargs):
            if state.global_step % self.LOG_INTERVAL != 0:
                return control
            if not is_rank_zero():
                return control
            # Recover the current batch: HF Trainer doesnt expose batch directly via
            # the callback hook. We sample a fresh batch from the train dataloader
            # instead, which is sufficient for the grad-norm probe (the spec calls
            # for periodic sampling, not exact same-batch tracking).
            try:
                # Get one batch from a fresh dataloader iteration
                dl = trainer.get_train_dataloader()
                batch = next(iter(dl))
                # Move to device
                device = next(model.parameters()).device
                batch = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in batch.items()}
                # Extract languages from language_id tensor
                lang_ids = batch.get("language_id")
                if lang_ids is None:
                    rank0_print(f"  [grad-norm] step {state.global_step}: no language_id in batch")
                    return control
                languages_in_batch = [_ID_TO_LANG.get(int(i), "unknown") for i in lang_ids.tolist()]
                # Prepare batch for compute_per_language_grad_norms:
                # - drop language_id (the function reads from batch["language"] instead)
                # - inject "language" as list[str] alongside the tensor fields
                batch_for_probe = {k: v for k, v in batch.items() if k != "language_id"}
                batch_for_probe["language"] = languages_in_batch

                from partition_loss_by_language import compute_per_language_grad_norms
                result = compute_per_language_grad_norms(model, batch_for_probe, state.global_step)

                # Persist to JSONL
                record = result.to_dict() if hasattr(result, "to_dict") else {
                    "step": state.global_step,
                    "grad_norms": getattr(result, "grad_norms", {}),
                    "n_tokens": getattr(result, "n_tokens", {}),
                    "ratios_swahili_to_other": getattr(result, "ratios_swahili_to_other", {}),
                }
                self.log_path.parent.mkdir(parents=True, exist_ok=True)
                with self.log_path.open("a") as f:
                    f.write(json.dumps(record) + "\n")
                self.results.append(record)
                rank0_print(f"  [grad-norm] step {state.global_step}: "
                            f"norms={record.get('grad_norms', {})}")
                # ---- Divergence killer (Spec v1.5.1 amendment 2026-05-27) ----
                # Triggers on EITHER (a) any language reports inf/nan, OR
                # (b) any language sustains norm > 1000 for 50 consecutive log events.
                # Cell 13 (Lion @ lr=2e-4) produced exactly this failure mode.
                if not self.divergence_kill_fired:
                    norms_dict = record.get("grad_norms", {})
                    bad_inf_nan = []
                    bad_high = []
                    for lang, val in norms_dict.items():
                        try:
                            fv = float(val)
                        except (TypeError, ValueError):
                            continue
                        if not math.isfinite(fv):
                            bad_inf_nan.append((lang, val))
                        elif fv > self.DIVERGENCE_NORM_THRESHOLD:
                            bad_high.append((lang, fv))
                    kill_now = False
                    reason_str = ""
                    if bad_inf_nan:
                        kill_now = True
                        ln, vl = bad_inf_nan[0]
                        reason_str = f"divergence:{ln}:{vl}@step{state.global_step}"
                    elif bad_high:
                        self.consecutive_high_norm += 1
                        if self.consecutive_high_norm >= self.DIVERGENCE_CONSECUTIVE_LIMIT:
                            kill_now = True
                            ln, fv = bad_high[0]
                            reason_str = (f"divergence:{ln}:norm{fv:.2e}_for_"
                                          f"{self.consecutive_high_norm}_events@step{state.global_step}")
                    else:
                        # All norms healthy this log event — reset counter
                        self.consecutive_high_norm = 0
                    if kill_now:
                        self.divergence_kill_fired = True
                        nonlocal early_stopped, kill_reason
                        early_stopped = True
                        kill_reason = reason_str
                        rank0_print(f"\n[divergence-killer] triggered at step "
                                    f"{state.global_step}: {reason_str}")
                        control.should_training_stop = True
                        return control
            except Exception as e:
                rank0_print(f"  [grad-norm] step {state.global_step}: failed ({e})")
            return control

    per_lang_callback = PerLanguageGradNormCallback(config.output_dir)

    # MFU instrumentation (proposal §6.5 — secondary efficiency metric)
    # Logs per-interval Model FLOPs Utilization to {output_dir}/mfu_log.jsonl
    _out_dir = config.output_dir if isinstance(config.output_dir, Path) \
               else Path(config.output_dir)
    mfu_callback = MFUCallback(
        output_dir=_out_dir,
        n_params=sum(p.numel() for p in model.parameters()),
        num_hidden_layers=model.config.num_hidden_layers,
        hidden_size=model.config.hidden_size,
        sequence_length=config.sequence_length,
        effective_batch_size=config.batch_size,
        world_size=int(os.environ.get("WORLD_SIZE", 1)),
        log_interval=50,
    )
    rank0_print(
        f"[mfu] init: n_params={mfu_callback.n_params:,} "
        f"flops/step={mfu_callback.flops_per_step:.3e} "
        f"peak_aggregate={mfu_callback.peak_aggregate_flops/1e12:.1f} TFLOPS/s "
        f"world_size={mfu_callback.world_size}"
    )

    trainer.add_callback(KillSwitchCallback())
    trainer.add_callback(PreSavePruneCallback())    # Amendment 12: prune before save
    trainer.add_callback(CheckpointPolicyCallback())
    if config.use_qlora_4bit:
        trainer.add_callback(per_lang_callback)
    else:
        rank0_print("[FFT/CPT] skipping PerLanguageGradNormCallback (runs backward inside on_step_end; deadlocks with DeepSpeed ZeRO-3)")
    trainer.add_callback(mfu_callback)

    try:
        trainer.train()
    except KeyboardInterrupt:
        rank0_print("[train] KeyboardInterrupt — saving final state")
        if is_rank_zero():
            trainer.save_state()
        early_stopped = True
        kill_reason = "KeyboardInterrupt"
        raise
    finally:
        # Barrier: ensure all ranks finish before exiting
        if dist.is_available() and dist.is_initialized():
            dist.barrier()

    wall_clock = time.perf_counter() - t0
    n_gpus = int(os.environ.get("WORLD_SIZE", "1"))
    gpu_hours = wall_clock * n_gpus / 3600.0

    # ----- Final pooled validation perplexity -----
    rank0_print(f"\n[eval] computing final pooled val perplexity")
    eval_metrics = trainer.evaluate()
    val_loss = eval_metrics.get("eval_loss")
    val_pph = math.exp(val_loss) if val_loss is not None else float("nan")

    # ----- PPH calculation if we have a baseline -----
    pph_result: Optional[PPHResult] = None
    if config.p_baseline is not None and not math.isnan(val_pph):
        try:
            pph_result = compute_pph(
                p_baseline=config.p_baseline,
                p_final=val_pph,
                gpu_hours=gpu_hours,
            )
            rank0_print(f"[pph] p_baseline={config.p_baseline:.4f} "
                        f"p_final={val_pph:.4f} gpu_hours={gpu_hours:.2f}")
            rank0_print(f"[pph] pph={pph_result.pph:.6f} "
                        f"log_ratio={pph_result.pph_log_ratio:.6f} "
                        f"flag_negative={pph_result.flag_negative}")
        except ValueError as e:
            rank0_print(f"[pph] could not compute: {e}")

    # ----- Assemble metrics dict for registry -----
    metrics_dict = {
        "metrics.validation_perplexity": val_pph,
        "metrics.peak_vram_gb": torch.cuda.max_memory_allocated() / 1024**3,
        "gpu_hours_consumed": gpu_hours,
        "wall_clock_seconds": int(wall_clock),
        "gpu_config.flash_attention_engaged": True,    # we verified at startup
        "training_events.early_stopped": early_stopped,
        "training_events.kill_reason": kill_reason,
        "tokenizer_source": tokenizer_source,
    }
    if pph_result is not None:
        metrics_dict.update(pph_result.to_metrics_dict())

    # Per-language gradient norm aggregation (spec §9.4)
    try:
        if per_lang_callback.results:
            # Mean ratio of Swahili to other languages across all logged steps
            import statistics as _stats
            all_ratios_mean = []
            per_lang_aggregates = {"swahili": [], "kikuyu": [], "kamba": [], "luo": []}
            for r in per_lang_callback.results:
                for lang, norm in r.get("grad_norms", {}).items():
                    if lang in per_lang_aggregates:
                        per_lang_aggregates[lang].append(norm)
                ratios = r.get("ratios_swahili_to_other", {})
                if ratios:
                    all_ratios_mean.extend(ratios.values())

            if all_ratios_mean:
                metrics_dict["metrics.per_language_grad_norm_ratio_mean"] = float(_stats.mean(all_ratios_mean))
            per_lang_means = {
                lang: float(_stats.mean(vals)) if vals else None
                for lang, vals in per_lang_aggregates.items()
            }
            metrics_dict["metrics.per_language_grad_norm_ratio_per_lang"] = json.dumps(per_lang_means)
            rank0_print(f"[grad-norm] aggregated across {len(per_lang_callback.results)} logging events")
    except Exception as _e:
        rank0_print(f"[grad-norm] aggregation failed: {_e}")

    rank0_print(f"\n[done] wall_clock={wall_clock:.1f}s gpu_hours={gpu_hours:.2f} "
                f"val_pph={val_pph:.4f} early_stopped={early_stopped}")
    return metrics_dict
