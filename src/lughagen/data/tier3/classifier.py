"""
classifier.py — LughaGen Tier 3 Quality Classifier

Trains a small binary classifier (afro-xlmr-base) to distinguish
"good Swahili/African text" from "bad" using historical generation
data:

  • POSITIVES — samples that passed the heuristic quality gate AND
    have combined_quality >= 0.75
  • NEGATIVES — samples rejected by the gate (saved by Tier 2 v2.0
    in quality_history.db) PLUS samples with combined_quality < 0.40

Two modes:
  python classifier.py train     — fine-tune and save model
  python classifier.py infer     — run inference over fluency_scored.jsonl

Output (infer):
  quality_classified.jsonl — adds fields `clf_good_prob`, `clf_label`
"""

import argparse
import json
import logging
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import jsonlines
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)

import config as cfg

logger = logging.getLogger("tier3.classifier")


# ── DATA COLLECTION ──────────────────────────────────────────────

# Tier 2 v2.0 stored rejected samples here. Adjust if your path differs.
REJECTED_DB = Path("/workspace/synthetic/tier2/quality_history.db")
REJECTED_TABLE = "rejected_samples"   # adjust if different in your schema


def collect_training_data(min_per_class: int = 5_000,
                          max_per_class: int = 20_000) -> Tuple[List[str], List[int]]:
    """Returns (texts, labels). label=1 good, label=0 bad."""
    positives, negatives = [], []

    # Positives: from fluency_scored.jsonl, take high-combined-quality
    if cfg.FLUENCY_OUTPUT.exists():
        with jsonlines.open(cfg.FLUENCY_OUTPUT) as r:
            for rec in r:
                if rec.get("combined_quality", 0) >= 0.75:
                    text = rec.get("text") or rec.get("output") or ""
                    if text.strip():
                        positives.append(text)
                if len(positives) >= max_per_class:
                    break
        logger.info("Collected %d positives from fluency_scored.jsonl", len(positives))

    # Negatives: rejected samples from generation pipeline
    if REJECTED_DB.exists():
        try:
            conn = sqlite3.connect(REJECTED_DB)
            rows = conn.execute(
                f"SELECT text FROM {REJECTED_TABLE} LIMIT ?",
                (max_per_class,),
            ).fetchall()
            conn.close()
            negatives.extend(r[0] for r in rows if r[0] and r[0].strip())
            logger.info("Collected %d rejected from %s", len(negatives), REJECTED_DB)
        except sqlite3.OperationalError as e:
            logger.warning("Could not read rejected DB: %s", e)

    # Supplement negatives: low-combined-quality from current pipeline
    if cfg.FLUENCY_OUTPUT.exists() and len(negatives) < min_per_class:
        with jsonlines.open(cfg.FLUENCY_OUTPUT) as r:
            for rec in r:
                if rec.get("combined_quality", 0.5) < 0.40:
                    text = rec.get("text") or rec.get("output") or ""
                    if text.strip():
                        negatives.append(text)
                if len(negatives) >= max_per_class:
                    break

    if len(positives) < min_per_class or len(negatives) < min_per_class:
        raise RuntimeError(
            f"Not enough training data: pos={len(positives)} neg={len(negatives)} "
            f"(need ≥{min_per_class} each). Run more generation or relax thresholds."
        )

    # Balance classes
    n = min(len(positives), len(negatives), max_per_class)
    rng = np.random.default_rng(seed=42)
    pos_idx = rng.choice(len(positives), size=n, replace=False)
    neg_idx = rng.choice(len(negatives), size=n, replace=False)

    texts  = [positives[i] for i in pos_idx] + [negatives[i] for i in neg_idx]
    labels = [1] * n + [0] * n

    # Shuffle
    perm = rng.permutation(len(texts))
    texts  = [texts[i] for i in perm]
    labels = [labels[i] for i in perm]

    logger.info("Final balanced training set: n=%d (n_pos=%d, n_neg=%d)",
                len(texts), n, n)
    return texts, labels


# ── DATASET ──────────────────────────────────────────────────────

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=192):
        self.texts = texts
        self.labels = labels
        self.tok = tokenizer
        self.max_length = max_length

    def __len__(self): return len(self.texts)

    def __getitem__(self, i):
        enc = self.tok(
            self.texts[i], truncation=True, max_length=self.max_length,
            padding="max_length", return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels":         torch.tensor(self.labels[i], dtype=torch.long),
        }


# ── TRAIN ────────────────────────────────────────────────────────

def train(epochs: int = 3, lr: float = 2e-5, batch_size: int = 32):
    cfg.ensure_dirs()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    texts, labels = collect_training_data()

    # 90/10 split
    rng = np.random.default_rng(seed=42)
    n_val = max(200, len(texts) // 10)
    val_idx = rng.choice(len(texts), size=n_val, replace=False)
    val_set  = set(val_idx.tolist())
    tr_texts  = [t for i, t in enumerate(texts)  if i not in val_set]
    tr_labels = [l for i, l in enumerate(labels) if i not in val_set]
    va_texts  = [texts[i]  for i in val_idx]
    va_labels = [labels[i] for i in val_idx]

    logger.info("Loading base model: %s", cfg.CLF_BASE_MODEL_ID)
    tok = AutoTokenizer.from_pretrained(cfg.CLF_BASE_MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.CLF_BASE_MODEL_ID, num_labels=2,
    ).to(device)

    train_ds = TextDataset(tr_texts, tr_labels, tok)
    val_ds   = TextDataset(va_texts, va_labels, tok)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=2)

    total_steps = len(train_dl) * epochs
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(optim, int(0.1 * total_steps), total_steps)

    best_acc = 0.0
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for step, batch in enumerate(train_dl):
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step(); sched.step(); optim.zero_grad()
            running_loss += out.loss.item()
            if step % 50 == 0:
                logger.info("epoch=%d step=%d/%d loss=%.4f",
                            epoch, step, len(train_dl), out.loss.item())

        # Validate
        model.eval()
        correct, total = 0, 0
        with torch.inference_mode():
            for batch in val_dl:
                batch = {k: v.to(device) for k, v in batch.items()}
                out = model(**batch)
                preds = out.logits.argmax(dim=-1)
                correct += (preds == batch["labels"]).sum().item()
                total   += batch["labels"].size(0)
        acc = correct / total
        logger.info("epoch=%d  val_acc=%.4f  avg_train_loss=%.4f",
                    epoch, acc, running_loss / len(train_dl))

        if acc > best_acc:
            best_acc = acc
            cfg.CLASSIFIER_DIR.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(cfg.CLASSIFIER_DIR)
            tok.save_pretrained(cfg.CLASSIFIER_DIR)
            (cfg.CLASSIFIER_DIR / "training_meta.json").write_text(json.dumps({
                "tier3_version": cfg.TIER3_VERSION,
                "trained_at":    datetime.utcnow().isoformat() + "Z",
                "val_accuracy":  acc,
                "n_train":       len(tr_texts),
                "n_val":         len(va_texts),
                "base_model":    cfg.CLF_BASE_MODEL_ID,
            }, indent=2))
            logger.info("Saved best model (acc=%.4f) to %s", acc, cfg.CLASSIFIER_DIR)

    logger.info("Training complete. Best val_acc=%.4f", best_acc)


# ── INFER ────────────────────────────────────────────────────────

def infer():
    cfg.ensure_dirs()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    if not cfg.CLASSIFIER_DIR.exists() or not (cfg.CLASSIFIER_DIR / "config.json").exists():
        raise FileNotFoundError(
            f"No trained classifier at {cfg.CLASSIFIER_DIR}. Run `classifier.py train` first."
        )

    if not cfg.FLUENCY_OUTPUT.exists():
        raise FileNotFoundError(
            f"Run fluency_scorer.py first — missing {cfg.FLUENCY_OUTPUT}"
        )

    logger.info("Loading classifier from %s", cfg.CLASSIFIER_DIR)
    tok = AutoTokenizer.from_pretrained(cfg.CLASSIFIER_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.CLASSIFIER_DIR, torch_dtype=torch.float16,
    ).to(device).eval()

    n_dropped = 0
    n_kept = 0

    with jsonlines.open(cfg.FLUENCY_OUTPUT) as reader, \
         jsonlines.open(cfg.QUALITY_OUTPUT, mode="w") as writer:

        batch = []
        for rec in reader:
            batch.append(rec)
            if len(batch) >= cfg.CLF_BATCH_SIZE:
                _process_batch(batch, tok, model, device, writer)
                batch = []
        if batch:
            _process_batch(batch, tok, model, device, writer)

    logger.info("Classifier inference complete: %s", cfg.QUALITY_OUTPUT)


@torch.inference_mode()
def _process_batch(batch, tok, model, device, writer):
    texts = [r.get("text") or r.get("output") or r.get("content") or "" for r in batch]
    enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=192).to(device)
    out = model(**enc)
    probs = torch.softmax(out.logits, dim=-1)  # [n, 2] — col 1 = good
    good_probs = probs[:, 1].cpu().numpy()
    for r, p in zip(batch, good_probs):
        r["clf_good_prob"] = float(p)
        r["clf_label"] = "good" if p >= (1 - cfg.CLF_BAD_THRESHOLD) else "bad"
        writer.write(r)


# ── CLI ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=cfg.LOG_LEVEL,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["train", "infer"])
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr",     type=float, default=2e-5)
    p.add_argument("--batch",  type=int, default=32)
    args = p.parse_args()

    if args.mode == "train":
        train(epochs=args.epochs, lr=args.lr, batch_size=args.batch)
    else:
        infer()
