#!/usr/bin/env python3
"""Thin CLI around `segment_classifier.train_model(...)`.

Same entry point the UI Training tab invokes as a subprocess.
"""

import argparse
import asyncio
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from app.ml.segment_classifier.service import segment_classifier


async def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train the LSTM segment classifier.",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-split", type=float, default=0.1)
    args = parser.parse_args()

    print(
        f"[INFO] Starting classifier training: epochs={args.epochs} "
        f"batch_size={args.batch_size} lr={args.lr} val_split={args.val_split}"
    )
    await segment_classifier.train_model(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        val_split=args.val_split,
    )
    print("[INFO] Classifier training complete.")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
