"""CLI entrypoint for preparing telemetry data for annotation."""

import argparse
import asyncio
import logging
import sys

from training.pipelines.training.config import TrainingPipelineConfig
from training.pipelines.training.full_dataset import Full_dataset_TelemetryMLService
from training.pipelines.training.pipeline import prepare_training_data


async def main() -> int:
    parser = argparse.ArgumentParser(
        description="Clean telemetry, select top laps, and add expert features for annotation.",
    )
    parser.add_argument(
        "--stage", choices=("all", "cleanup", "top_laps", "enrich"), default="all",
        help="Run one preparation stage, or all stages in order (default).",
    )
    parser.add_argument(
        "--workers", type=int, default=4,
        help="Number of worker processes for telemetry preparation (default: 4).",
    )
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1")

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger("prepare_training_data")

    cfg = TrainingPipelineConfig()
    service = Full_dataset_TelemetryMLService(logger=logger, pipeline_config=cfg)

    print(f"[INFO] Starting telemetry data preparation: {args.stage}")
    print(f"[INFO] Workers: {args.workers}")
    print(f"[INFO] Raw sessions cache: {cfg.session_data_cache_key}")
    print(f"[INFO] Processed sessions cache: {cfg.processed_session_data_cache_key}")
    print(f"[INFO] Top laps cache: {cfg.top_laps_cache_key}")
    print(f"[INFO] Enriched sessions cache: {cfg.enriched_sessions_cache_key}")

    try:
        result = await prepare_training_data(
            telemetry_store=service.telemetry_store,
            config=service.pipeline_config,
            backend_service=service.backend_service,
            top_lap_reference_feature_names=(
                service._top_lap_reference_feature_names
            ),
            stage=args.stage,
            max_workers=args.workers,
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
