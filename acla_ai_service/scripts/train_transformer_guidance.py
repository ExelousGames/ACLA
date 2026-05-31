#!/usr/bin/env python3
"""Thin CLI around `training.pipeline.run_transformer_guidance_training(...)`.

Same entry point the UI Training tab invokes as a subprocess.
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from app.pipelines.training.config import TrainingPipelineConfig
from app.pipelines.training.full_dataset import Full_dataset_TelemetryMLService
from app.pipelines.training.pipeline import run_transformer_guidance_training


async def main() -> int:
    cfg = TrainingPipelineConfig()
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
    result = await run_transformer_guidance_training(
        args.annotation_key,
        telemetry_store=service.telemetry_store,
        config=service.pipeline_config,
        backend_service=service.backend_service,
    )

    if not result.get("success"):
        print(f"[ERROR] Transformer training failed: {result.get('error')}")
        return 1

    print("[INFO] Transformer guidance training complete.")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
