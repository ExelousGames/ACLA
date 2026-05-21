#!/usr/bin/env python3
"""Thin CLI around `app.pipelines.training.llm_trainer.run_llm_training`.

Same entry point the UI's "Start training" button invokes as a subprocess.
"""

import argparse
import asyncio
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from app.pipelines.training.llm_trainer import DEFAULT_MODEL, run_llm_training


async def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fine-tune the telemetry LLM from a chat-format JSONL.",
    )
    parser.add_argument(
        "--dataset", type=Path, required=True,
        help="Path to the chat-format JSONL (e.g. models/llm_datasets/foo.chat.jsonl).",
    )
    parser.add_argument(
        "--model", type=str, default=DEFAULT_MODEL,
        help=f"HuggingFace model ID. Default: {DEFAULT_MODEL}",
    )
    args = parser.parse_args()

    result = await run_llm_training(
        args.dataset, model=args.model, project_root=project_root,
    )

    if result.success:
        print("\n=== Training Completed Successfully! ===")
        print(f"Adapter Output Directory: {result.adapter_directory}")
        return 0

    print("\n=== Training Failed ===")
    print(result.error or "(no error message)")
    return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
