#!/usr/bin/env python3
"""
Launcher script that runs the LLM Dataset Annotation Streamlit UI,
and upon completion, triggers a second step to train a local language model
using the completed dataset.
"""

import sys
import subprocess
import glob
import os
from pathlib import Path

def main():
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    ui_script = project_root / "ui" / "llm_dataset_annotation.py"
    
    if not ui_script.exists():
        print(f"Error: Could not find UI script at: {ui_script}")
        sys.exit(1)
        
    print("=== Step 1: Starting LLM Annotation UI ===")
    print("Please use the UI to complete your annotations.")
    print("When you are finished and have saved the dataset, stop the UI (Ctrl+C).")
    
    cmd = [sys.executable, "-m", "streamlit", "run", str(ui_script)]
    
    try:
        subprocess.run(cmd, cwd=project_root)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Failed to start UI: {e}")
        sys.exit(1)

    print("\n=== Step 2: Training a Telemetry LLM ===")
    
    dataset_path = project_root / "models" / "llm_datasets" / "telemetry_descriptions_v1.jsonl"
    
    if not dataset_path.exists():
        print(f"Dataset not found at {dataset_path}. Ensure you saved annotations.")
        sys.exit(0)
        
    print(f"Found dataset: {dataset_path}")
    print("We will use it to fine-tune a small LLM fit for generating 100-word concise answers.")
    
    # Qwen2.5-1.5B-Instruct is highly recommended as a small yet powerful instruct model 
    # well-suited for short reasoning and 100-word responses without excessive verbosity.
    # Alternatives: meta-llama/Llama-3.2-1B-Instruct or TinyLlama/TinyLlama-1.1B-Chat-v1.0
    small_model_choice = "Qwen/Qwen2.5-1.5B-Instruct" 
    
    # Trigger the training script (which handles TelemetryLLMOrchestrator config with LoRA)
    train_script = script_dir / "train_telemetry_llm.py"
    if train_script.exists():
        print(f"Starting training with model {small_model_choice} on dataset {dataset_path}...")
        try:
            # Reusing the underlying script using subprocess
            subprocess.run(
                [sys.executable, str(train_script), "--dataset", str(dataset_path), "--model", small_model_choice], 
                cwd=project_root, 
                check=True
            )
            print("\nModel trained successfully as a LoRA adapter!")
            print("Note: The orchestrator automatically saves it in models/llm_adapters/")
            print("To reuse this trained model, the orchestrator will auto-load the adapter on inference when calling get_llm_for_inference(...)")
        except subprocess.CalledProcessError as e:
            print(f"Training failed with error code: {e.returncode}")
        except KeyboardInterrupt:
            print("\nTraining interrupted by user.")
    else:
        print("Training script train_telemetry_llm.py not found. Could not start training step.")

if __name__ == "__main__":
    main()
