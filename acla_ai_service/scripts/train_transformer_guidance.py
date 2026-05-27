#!/usr/bin/env python3
"""Thin CLI around `Full_dataset_TelemetryMLService.run_transformer_guidance_training(...)`.

Same entry point the UI Training tab invokes as a subprocess.
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from app.infra.config.pipeline import PipelineConfig
from app.pipelines.training.full_dataset import Full_dataset_TelemetryMLService


async def main() -> int:
    cfg = PipelineConfig()
    parser = argparse.ArgumentParser(
        description="Train the transformer guidance model on annotated segments.",
    )
    parser.add_argument("--annotation-key", default=cfg.annotation_cache_key)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger("train_transformer_guidance")

    service = Full_dataset_TelemetryMLService(logger=logger, pipeline_config=cfg)
    print(
        f"[INFO] Starting transformer guidance training: "
        f"annotation_key={args.annotation_key}"
    )
    result = await service.run_transformer_guidance_training(
        annotation_cache_key=args.annotation_key,
    )

    if not result.get("success"):
        print(f"[ERROR] Transformer training failed: {result.get('error')}")
        return 1

    print("[INFO] Transformer guidance training complete.")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
