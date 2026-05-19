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

from app.pipelines.chat.orchestrator import TelemetryLLMOrchestrator
from app.llm.local_llm import LocalLLMConfig

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

    # Split the dataset for training and evaluation
    import json
    import random
    with open(dataset_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    random.seed(42)
    random.shuffle(lines)
    
    split_idx = max(1, int(len(lines) * 0.8)) # 80/20 split
    train_lines = lines[:split_idx]
    eval_lines = lines[split_idx:]
    
    train_dataset_path = dataset_path.with_name(dataset_path.stem + "_train.jsonl")
    eval_dataset_path = dataset_path.with_name(dataset_path.stem + "_eval.jsonl")
    
    with open(train_dataset_path, "w", encoding="utf-8") as f:
        for line in train_lines:
            f.write(line + "\n")
            
    with open(eval_dataset_path, "w", encoding="utf-8") as f:
        for line in eval_lines:
            f.write(line + "\n")
            
    print(f"Split dataset: {len(train_lines)} train, {len(eval_lines)} eval")
    llm_config = LocalLLMConfig()
    llm_config.model.base_model = args.model
    llm_config.model.tokenizer_name = args.model
    # Recommended optimizations for fine-tuning on consumer hardware:
    llm_config.model.load_in_4bit = False   # Quantize to fit in memory
    llm_config.lora.use_lora = True       # Crucial for reusing and saving just the adapter
    llm_config.training.use_gradient_checkpointing = True 

    orchestrator = TelemetryLLMOrchestrator(
        llm_config=llm_config,
        adapter_directory=project_root / "models" / "llm_adapters",
        dataset_directory=project_root / "models" / "llm_datasets",
    )

    print(f"\nEvaluating dataset: {dataset_path}")
    print("Starting training process...")
    
    result = await orchestrator.train_from_dataset(
        dataset_path=train_dataset_path,
        eval_dataset_path=eval_dataset_path,
        cleanup_dataset_file=False
    )
    
    if result.get("success"):
        print("\n=== Training Completed Successfully! ===")
        print(f"Adapter Output Directory: {result.get('adapter_directory')}")
        print("\n=== Running Final Evaluation ===")
        try:
            from app.llm.local_llm import GenerationRequest
            
            print(f"Loading adapter {result.get('adapter_directory')} for evaluation...")
            # Note: For evaluation, simply evaluate on up to 3 eval examples
            eval_samples = [json.loads(line) for line in eval_lines[:3]]
            eval_payload = {
                "adapter_directory_name": result.get('adapter_directory'),
                "adapter_zip_base64": result.get('serialized_adapter', {}).get("adapter_zip_base64")
            }
            llm = orchestrator._deserialize_llm_model(eval_payload)
            
            for i, sample in enumerate(eval_samples):
                sys_prompt = sample.get("system_prompt", "You are an AI assistant.")
                user_prompt = sample.get("prompt", "")
                expected = sample.get("response", sample.get("completion", ""))
                
                req = GenerationRequest(
                    user_prompt=user_prompt,
                    max_new_tokens=100
                )
                output = llm.generate(req)
                print(f"\n--- Eval Sample {i+1} ---")
                print(f"[User]: {user_prompt[:100]}...")
                print(f"[Expected]: {expected[:100]}...")
                print(f"[Generated]: {output[:200]}")
        except Exception as e:
            print(f"Failed to run final evaluation: {e}")

        print("\nTo reuse this model for inference later, the orchestrator will automatically")
        print("deserialize the saved adapter when you call `await orchestrator.get_llm_for_inference(...)`")
    else:
        print("\n=== Training Failed ===")
        print(result)

if __name__ == "__main__":
    asyncio.run(main())
