#!/usr/bin/env python3
"""Thin CLI around `segment_classifier.train_model(...)`.

Same entry point the UI Training tab invokes as a subprocess.
"""

import argparse
import asyncio
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[4]
sys.path.append(str(project_root))

from app.ml.segment_classifier.service import segment_classifier
from app.pipelines.training.config import TrainingPipelineConfig


async def main() -> int:
    cfg = TrainingPipelineConfig()
    parser = argparse.ArgumentParser(
        description="Train the temporal behavior segment detector.",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--annotation-key", default=cfg.annotation_cache_key)
    parser.add_argument(
        "--session-id",
        dest="session_ids",
        action="append",
        default=None,
        help="Session chunk ID to include. Repeat to train on multiple sessions.",
    )
    args = parser.parse_args()

    selected_sessions = args.session_ids if args.session_ids is not None else "all"
    print(
        f"[INFO] Starting classifier training: epochs={args.epochs} "
        f"batch_size={args.batch_size} lr={args.lr} val_split={args.val_split} "
        f"annotation_key={args.annotation_key} sessions={selected_sessions}"
    )
    await segment_classifier.train_model(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        val_split=args.val_split,
        annotation_cache_key=args.annotation_key,
        session_ids=args.session_ids,
    )
    print("[INFO] Classifier training complete.")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
