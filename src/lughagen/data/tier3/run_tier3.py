"""
run_tier3.py — LughaGen Tier 3 Pipeline Orchestrator

Runs the full Tier 3 dataset intelligence pipeline:

    analyze → tag topics → score fluency → train classifier → infer classifier → reshape

Each stage is idempotent and resumable, so re-running this script picks up
where it left off. Stages can also be skipped via flags, e.g. when the
classifier is already trained.

Usage:
    python run_tier3.py                  # full pipeline
    python run_tier3.py --skip-train     # skip classifier training
    python run_tier3.py --only analyze   # just run analysis
    python run_tier3.py --from fluency   # start from fluency stage onward

Stages:
    analyze      — distribution diagnostics → analysis_report.json + plots/
    tag-topics   — embed + nearest-anchor → topics_classified.jsonl
    score-fluency— pseudo-PLL via afro-xlmr → fluency_scored.jsonl
    train-clf    — fine-tune binary classifier on accept/reject history
    infer-clf    — apply classifier → quality_classified.jsonl
    reshape      — hard gates + dedup + caps + stratify → lughagen_train.jsonl
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import config as cfg

logger = logging.getLogger("tier3.orchestrator")

STAGES = [
    "analyze", "tag-topics", "score-fluency",
    "train-clf", "infer-clf", "reshape",
]


def run_stage(name: str):
    t0 = time.time()
    logger.info("─" * 60)
    logger.info("▶ Stage: %s", name)
    logger.info("─" * 60)

    if name == "analyze":
        from analyzer import run_analysis
        run_analysis()
    elif name == "tag-topics":
        from topic_balancer import run_topic_tagging
        run_topic_tagging()
    elif name == "score-fluency":
        from fluency_scorer import run_fluency_scoring
        run_fluency_scoring()
    elif name == "train-clf":
        from classifier import train
        train()
    elif name == "infer-clf":
        from classifier import infer
        infer()
    elif name == "reshape":
        from reshaper import run_reshape
        run_reshape()
    else:
        raise ValueError(f"Unknown stage: {name}")

    logger.info("✓ %s done in %.1fs", name, time.time() - t0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--skip-train", action="store_true",
                   help="Skip classifier training (use existing trained model)")
    p.add_argument("--only", choices=STAGES,
                   help="Run only this stage")
    p.add_argument("--from", dest="from_stage", choices=STAGES,
                   help="Start from this stage onward")
    args = p.parse_args()

    cfg.ensure_dirs()

    # Build stage list
    if args.only:
        stages = [args.only]
    elif args.from_stage:
        stages = STAGES[STAGES.index(args.from_stage):]
    else:
        stages = list(STAGES)

    if args.skip_train and "train-clf" in stages:
        stages.remove("train-clf")

    logger.info("Running stages: %s", stages)

    for stage in stages:
        try:
            run_stage(stage)
        except Exception as e:
            logger.exception("✗ Stage %s failed: %s", stage, e)
            sys.exit(1)

    logger.info("═" * 60)
    logger.info("ALL STAGES COMPLETE")
    logger.info("Final dataset: %s", cfg.FINAL_OUTPUT)
    logger.info("Summary:       %s", cfg.DATASET_SUMMARY)
    logger.info("═" * 60)


if __name__ == "__main__":
    logging.basicConfig(
        level=cfg.LOG_LEVEL,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(cfg.LOG_DIR / "tier3_orchestrator.log"
                                if cfg.LOG_DIR.exists() else "/tmp/tier3.log"),
        ],
    )
    main()
