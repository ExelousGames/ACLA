#!/usr/bin/env python3
"""Run classifier and transformer training sequentially in a single subprocess.

Invoked by the UI Training tab's "Run all" card.
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[4]
sys.path.append(str(project_root))

from app.pipelines.training.config import TrainingPipelineConfig
from app.ml.segment_classifier.trainer import segment_classifier_trainer
from app.pipelines.training.full_dataset import Full_dataset_TelemetryMLService
from app.pipelines.training.pipeline import run_transformer_guidance_training


async def main() -> int:
    cfg = TrainingPipelineConfig()
    parser = argparse.ArgumentParser(
        description="Run all telemetry-model trainings sequentially.",
    )
    parser.add_argument("--annotation-key", default=cfg.annotation_cache_key)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger("run_all_trainings")

    print("\n=== [1/2] Segment classifier ===")
    await segment_classifier_trainer.train_model(annotation_cache_key=args.annotation_key)

    print("\n=== [2/2] Transformer guidance ===")
    service = Full_dataset_TelemetryMLService(logger=logger, pipeline_config=cfg)
    transformer_result = await run_transformer_guidance_training(
        args.annotation_key,
        telemetry_store=service.telemetry_store,
        config=service.pipeline_config,
        backend_service=service.backend_service,
    )
    if not transformer_result.get("success"):
        print(f"[ERROR] Transformer training failed: {transformer_result.get('error')}")
        return 1

    print("\n=== All trainings complete ===")
    print(f"  classifier: ok")
    print(f"  transformer: ok")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
