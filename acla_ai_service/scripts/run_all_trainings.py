#!/usr/bin/env python3
"""Run classifier → transformer → LLM training sequentially in a single subprocess.

Invoked by the UI Training tab's "Run all" card.
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from app.infra.config.pipeline import PipelineConfig
from app.ml.segment_classifier.service import segment_classifier
from app.pipelines.training.full_dataset import Full_dataset_TelemetryMLService
from app.pipelines.training.llm_trainer import DEFAULT_MODEL, run_llm_training


async def main() -> int:
    cfg = PipelineConfig()
    parser = argparse.ArgumentParser(
        description="Run all three trainings sequentially.",
    )
    parser.add_argument("--annotation-key", default=cfg.annotation_cache_key)
    parser.add_argument("--chat-dataset", type=Path, required=True,
                        help="LLM chat-format JSONL path.")
    parser.add_argument("--llm-model", default=DEFAULT_MODEL)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger("run_all_trainings")

    print("\n=== [1/3] Segment classifier ===")
    await segment_classifier.train_model()

    print("\n=== [2/3] Transformer guidance ===")
    service = Full_dataset_TelemetryMLService(logger=logger, pipeline_config=cfg)
    transformer_result = await service.run_transformer_guidance_training(
        annotation_cache_key=args.annotation_key,
    )
    if not transformer_result.get("success"):
        print(f"[ERROR] Transformer training failed: {transformer_result.get('error')}")
        return 1

    print("\n=== [3/3] LLM fine-tune ===")
    llm_result = await run_llm_training(
        args.chat_dataset, model=args.llm_model, project_root=project_root,
    )
    if not llm_result.success:
        print(f"[ERROR] LLM training failed: {llm_result.error}")
        return 1

    print("\n=== All trainings complete ===")
    print(f"  classifier: ok")
    print(f"  transformer: ok")
    print(f"  llm adapter: {llm_result.adapter_directory}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
