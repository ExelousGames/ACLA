#!/usr/bin/env python3
"""CLI for complete-session boundary cropper training."""

import argparse
import asyncio
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[4]
sys.path.append(str(project_root))

from app.ml.segment_cropper.trainer import segment_cropper_trainer
from app.pipelines.training.config import TrainingPipelineConfig


async def main() -> int:
    cfg = TrainingPipelineConfig()
    parser = argparse.ArgumentParser(description="Train the learned session boundary cropper.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--annotation-key", default=cfg.annotation_cache_key)
    args = parser.parse_args()
    print(
        f"[INFO] Starting cropper training: epochs={args.epochs} "
        f"batch_size={args.batch_size} lr={args.lr} "
        f"annotation_key={args.annotation_key} split=90/10"
    )
    await segment_cropper_trainer.train_model(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        annotation_cache_key=args.annotation_key,
    )
    print("[INFO] Segment cropper training complete.")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
