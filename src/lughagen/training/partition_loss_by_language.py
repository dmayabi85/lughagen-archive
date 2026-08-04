#!/usr/bin/env python3
"""
/workspace/scripts/partition_loss_by_language.py

LughaGen Training Framework v1.5 §9.4 + §2.3
Per-language gradient norm logging. Called every 50 training steps by the
training loop in Week 4. Partitions the current batch by source language,
runs a SEPARATE diagnostic backward pass per language partition, records each
partition's gradient norm, and zeros gradients between partitions.

CRITICAL §9.4 invariant:
  The diagnostic backward passes MUST NOT contribute to the actual weight
  update. The training loop performs its OWN full-batch backward pass for the
  optimiser step. The per-language passes are diagnostic only — gradients are
  zeroed after each measurement.

This module is imported by train_orchestrator.py (Week 4). The Week 1
deliverable is the module + a CPU-mock test that verifies the API.

Returns Swahili-to-other-language ratios:
    swahili_to_kikuyu, swahili_to_kamba, swahili_to_luo

A ratio close to 1.0 indicates balanced gradient signal.
A ratio substantially > 1.0 indicates Swahili dominance.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from typing import Any, Optional


LANGUAGES = ("swahili", "kikuyu", "kamba", "luo")


@dataclass
class PerLanguageGradResult:
    """Result of one per-language gradient measurement step."""
    step: int
    grad_norms: dict[str, float] = field(default_factory=dict)   # lang -> norm
    n_tokens: dict[str, int] = field(default_factory=dict)        # lang -> tokens in this partition
    ratios_swahili_to_other: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "step": self.step,
            "grad_norms": self.grad_norms,
            "n_tokens": self.n_tokens,
            "ratios_swahili_to_other": self.ratios_swahili_to_other,
        }


def partition_batch_by_language(
    batch: dict[str, Any],
    language_field: str = "language",
) -> dict[str, dict[str, Any]]:
    """
    Split a training batch into per-language sub-batches.

    `batch` is expected to contain:
        - 'input_ids':      tensor [B, T]
        - 'attention_mask': tensor [B, T]
        - 'labels':         tensor [B, T]
        - language_field:   list[str] of length B

    Returns: dict mapping language -> sub-batch dict, only for non-empty
             partitions.
    """
    import torch

    languages = batch[language_field]
    if not isinstance(languages, list):
        raise TypeError(
            f"batch['{language_field}'] must be a list[str]; got {type(languages)}"
        )

    out: dict[str, dict[str, Any]] = {}
    for lang in LANGUAGES:
        indices = [i for i, L in enumerate(languages) if L == lang]
        if not indices:
            continue
        idx = torch.tensor(indices, dtype=torch.long, device=batch["input_ids"].device)
        sub = {
            "input_ids":      batch["input_ids"].index_select(0, idx),
            "attention_mask": batch["attention_mask"].index_select(0, idx),
            "labels":         batch["labels"].index_select(0, idx),
        }
        sub["n_tokens"] = int(sub["attention_mask"].sum().item())
        out[lang] = sub
    return out


def compute_per_language_grad_norms(
    model,
    batch: dict[str, Any],
    step: int,
    language_field: str = "language",
) -> PerLanguageGradResult:
    """
    Run diagnostic backward passes per language partition; record grad norms.

    Procedure (§9.4):
      For each language partition:
        1. Zero all gradients
        2. Forward pass on the partition; compute loss
        3. Backward pass (diagnostic only)
        4. Compute global gradient norm
        5. Record norm and token count
        6. Zero gradients AGAIN (so we don't pollute the real training step)

    The caller's training loop performs its OWN full-batch forward/backward
    and optimiser step AFTER this function returns. This function never calls
    optimiser.step().
    """
    import torch

    parts = partition_batch_by_language(batch, language_field=language_field)
    if not parts:
        return PerLanguageGradResult(step=step)

    grad_norms: dict[str, float] = {}
    n_tokens: dict[str, int] = {}

    # Cache the original training mode and restore at end
    was_training = model.training
    model.train()

    for lang, sub in parts.items():
        # 1. Zero gradients before the diagnostic backward
        for p in model.parameters():
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()

        # 2. Forward
        outputs = model(
            input_ids=sub["input_ids"],
            attention_mask=sub["attention_mask"],
            labels=sub["labels"],
        )
        loss = outputs.loss

        # 3. Diagnostic backward
        loss.backward()

        # 4. Compute global grad norm (L2 across all parameters)
        total_sq = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_sq += float(p.grad.detach().norm(2).item() ** 2)
        norm = total_sq ** 0.5

        grad_norms[lang] = norm
        n_tokens[lang] = sub["n_tokens"]

        # 6. Zero gradients again so the real training step starts clean
        for p in model.parameters():
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()

    if not was_training:
        model.eval()

    # Compute Swahili-to-other ratios
    ratios = {}
    sw_norm = grad_norms.get("swahili")
    if sw_norm is not None and sw_norm > 0:
        for lang in ("kikuyu", "kamba", "luo"):
            other = grad_norms.get(lang)
            if other is not None and other > 0:
                ratios[f"swahili_to_{lang}"] = sw_norm / other

    return PerLanguageGradResult(
        step=step,
        grad_norms=grad_norms,
        n_tokens=n_tokens,
        ratios_swahili_to_other=ratios,
    )


def aggregate_run_means(measurements: list[PerLanguageGradResult]) -> dict:
    """
    Aggregate a list of per-step measurements into run-level means.

    Returns dict suitable for the registry's:
      metrics.per_language_grad_norm_ratio_mean        (scalar — overall mean ratio)
      metrics.per_language_grad_norm_ratio_per_lang    (dict — per-language norm mean)
    """
    if not measurements:
        return {
            "per_language_grad_norm_ratio_mean": None,
            "per_language_grad_norm_ratio_per_lang": {
                "swahili": None, "kikuyu": None, "kamba": None, "luo": None,
            },
        }

    # Per-language norm averages
    sums: dict[str, float] = {lang: 0.0 for lang in LANGUAGES}
    counts: dict[str, int] = {lang: 0 for lang in LANGUAGES}
    for m in measurements:
        for lang, n in m.grad_norms.items():
            sums[lang] += n
            counts[lang] += 1
    per_lang_mean = {
        lang: (sums[lang] / counts[lang]) if counts[lang] else None
        for lang in LANGUAGES
    }

    # Overall ratio: mean of all swahili_to_X ratios across all measurements
    all_ratios: list[float] = []
    for m in measurements:
        all_ratios.extend(m.ratios_swahili_to_other.values())
    overall_mean = sum(all_ratios) / len(all_ratios) if all_ratios else None

    return {
        "per_language_grad_norm_ratio_mean": overall_mean,
        "per_language_grad_norm_ratio_per_lang": per_lang_mean,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Week 1 smoke test (CPU mock)
# ─────────────────────────────────────────────────────────────────────────────

def _cpu_smoke_test() -> int:
    """
    CPU smoke that exercises the partition + ratio computation logic without
    requiring a real LLaMA model. Used to verify the §9.4 API in Week 1.
    """
    print("[partition_loss_by_language] CPU smoke test")

    try:
        import torch
        import torch.nn as nn
    except ImportError:
        print("  FAIL: torch not importable", file=sys.stderr)
        return 1

    # Tiny model: linear -> output logits for fake vocab
    class TinyLM(nn.Module):
        def __init__(self, vocab=100, hidden=16):
            super().__init__()
            self.emb = nn.Embedding(vocab, hidden)
            self.out = nn.Linear(hidden, vocab)
        def forward(self, input_ids, attention_mask=None, labels=None):
            h = self.emb(input_ids)
            logits = self.out(h)
            loss = None
            if labels is not None:
                loss = nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                    ignore_index=-100,
                )
            from types import SimpleNamespace
            return SimpleNamespace(loss=loss, logits=logits)

    torch.manual_seed(0)
    model = TinyLM()

    # Fake batch: 8 samples, 4 per language at most
    B, T = 8, 16
    batch = {
        "input_ids":      torch.randint(0, 100, (B, T)),
        "attention_mask": torch.ones(B, T, dtype=torch.long),
        "labels":         torch.randint(0, 100, (B, T)),
        "language":       ["swahili", "swahili", "swahili", "swahili",
                           "kikuyu", "kikuyu", "kamba", "luo"],
    }

    result = compute_per_language_grad_norms(model, batch, step=50)

    print(f"  Grad norms by language: {result.grad_norms}")
    print(f"  Tokens by language:     {result.n_tokens}")
    print(f"  Ratios (Sw->others):    {result.ratios_swahili_to_other}")

    # Sanity checks
    assert "swahili" in result.grad_norms, "Swahili partition missing"
    assert all(n > 0 for n in result.grad_norms.values()), "All grad norms must be positive"
    assert "swahili_to_kikuyu" in result.ratios_swahili_to_other, "Sw->Ki ratio missing"

    # Verify gradients are zeroed at the end (the function's invariant)
    for p in model.parameters():
        if p.grad is not None:
            assert p.grad.abs().sum().item() == 0.0, \
                "Gradients were NOT zeroed at end of diagnostic pass; §9.4 invariant violated"
    print("  PASS: gradients zeroed after diagnostic passes (§9.4 invariant holds)")

    # Aggregate
    agg = aggregate_run_means([result, result, result])
    print(f"  Aggregated (3 measurements): {agg}")
    assert agg["per_language_grad_norm_ratio_mean"] is not None
    print("  PASS: aggregation produces non-null run-level means")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Per-language gradient norm logging (§9.4)"
    )
    parser.add_argument("--cpu-smoke", action="store_true",
                        help="Run CPU smoke test (Week 1 acceptance).")
    args = parser.parse_args()

    if args.cpu_smoke:
        return _cpu_smoke_test()

    print("This module is imported by train_orchestrator.py at training time.")
    print("Run with --cpu-smoke to verify the §9.4 procedure on CPU.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
