#!/usr/bin/env python3
"""Non-interactive CLI for preparing telemetry data for annotation.

Invoked by the Segment Annotation Pipeline UI as a background subprocess.
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
from app.pipelines.training.pipeline import prepare_training_data


async def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download, process, and enrich telemetry data for annotation.",
    )
    parser.add_argument(
        "--top-laps-count",
        type=int,
        default=1,
        help="Number of top laps to keep per track/car/grip bucket.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger("prepare_training_data")

    cfg = TrainingPipelineConfig()
    service = Full_dataset_TelemetryMLService(logger=logger, pipeline_config=cfg)

    print("[INFO] Starting telemetry data preparation")
    print(f"[INFO] Raw sessions cache: {cfg.session_data_cache_key}")
    print(f"[INFO] Processed sessions cache: {cfg.processed_session_data_cache_key}")
    print(f"[INFO] Top laps cache: {cfg.top_laps_cache_key}")
    print(f"[INFO] Enriched sessions cache: {cfg.enriched_sessions_cache_key}")
    print(f"[INFO] Top laps count: {args.top_laps_count}")

    try:
        result = await prepare_training_data(
            telemetry_store=service.telemetry_store,
            config=service.pipeline_config,
            backend_service=service.backend_service,
            imitate_expert_feature_names=service._imitate_expert_feature_names,
            top_laps_count=args.top_laps_count,
        )
    except Exception as exc:
        logger.exception("Data preparation failed")
        print(f"[ERROR] Data preparation failed: {exc}")
        return 1

    if not result.get("success"):
        print(f"[ERROR] Data preparation failed: {result.get('error')}")
        return 1

    print("[INFO] Data preparation complete.")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
