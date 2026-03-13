#!/usr/bin/env python3
"""
Script to trigger LLM Orchestrator to train a small language model
on the datasets annotated by the UI.
"""

import sys
import argparse
import asyncio
from pathlib import Path

# Adjust path to find app module
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from app.services.llm.telemetry_llm_orchestrator import TelemetryLLMOrchestrator
from app.services.llm.local_llm_service import LocalLLMConfig

async def main():
    parser = argparse.ArgumentParser(description="Train a local LLM adapter using the telemetry orchestrator.")
    parser.add_argument(
        "--dataset", 
        type=str, 
        required=True, 
        help="Path to the JSONL dataset (e.g., models/llm_datasets/my_dataset.jsonl)"
    )
    # Recommending a small instruct model for ~100 word answers
    parser.add_argument(
        "--model", 
        type=str, 
        default="Qwen/Qwen2.5-1.5B-Instruct", 
        help="HuggingFace model ID. Qwen/Qwen2.5-1.5B-Instruct is a great small model for concise answers."
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"Error: Dataset not found at {dataset_path}")
        sys.exit(1)

    print(f"Initializing orchestrator with model: {args.model}")
    
    # Configure the LocalLLMConfig for a small model fit for this task
    llm_config = LocalLLMConfig()
    llm_config.base_model = args.model
    llm_config.tokenizer_name = args.model
    # Recommended optimizations for fine-tuning on consumer hardware:
    llm_config.load_in_4bit = True   # Quantize to fit in memory
    llm_config.use_lora = True       # Crucial for reusing and saving just the adapter
    llm_config.use_gradient_checkpointing = True 

    orchestrator = TelemetryLLMOrchestrator(
        llm_config=llm_config,
        adapter_directory=project_root / "models" / "llm_adapters",
        dataset_directory=project_root / "models" / "llm_datasets",
    )

    print(f"\nEvaluating dataset: {dataset_path}")
    print("Starting training process...")
    
    result = await orchestrator.train_from_dataset(
        dataset_path=dataset_path,
        cleanup_dataset_file=False
    )
    
    if result.get("success"):
        print("\n=== Training Completed Successfully! ===")
        print(f"Adapter Output Directory: {result.get('adapter_directory')}")
        print("\nTo reuse this model for inference later, the orchestrator will automatically")
        print("deserialize the saved adapter when you call `await orchestrator.get_llm_for_inference(...)`")
    else:
        print("\n=== Training Failed ===")
        print(result)

if __name__ == "__main__":
    asyncio.run(main())
