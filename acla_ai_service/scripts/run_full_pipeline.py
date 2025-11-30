import asyncio
import sys
import subprocess
from pathlib import Path
import os

# Add parent directory to path to allow imports
# parents[0] = scripts
# parents[1] = acla_ai_service (or /app in docker)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.full_dataset_ml_service import Full_dataset_TelemetryMLService, PipelineConfig
from app.services.segment_classifier_service import segment_classifier

def confirm_step(step_name):
    while True:
        response = input(f"\nDo you want to execute {step_name}? (y/n) [y]: ").strip().lower()
        if response in ["", "y", "yes"]:
            return True
        elif response in ["n", "no"]:
            return False

async def main():
    steps = ["prepare_data", "annotate", "train_classifier", "process_segments", "train_transformer"]
    
    print("\nSelect start step:")
    for i, step in enumerate(steps):
        print(f"{i + 1}. {step}")
    
    while True:
        try:
            choice = input(f"\nEnter step number (1-{len(steps)}) [default 1]: ").strip()
            if not choice:
                start_step = steps[0]
                break
            
            idx = int(choice) - 1
            if 0 <= idx < len(steps):
                start_step = steps[idx]
                break
            else:
                print(f"Please enter a number between 1 and {len(steps)}")
        except ValueError:
            print("Invalid input. Please enter a number.")

    print(f"Initializing services...")
    service = Full_dataset_TelemetryMLService()
    pipeline_config = PipelineConfig()
    
    # Default keys
    processed_sessions_cache_key = pipeline_config.processed_session_data_cache_key
    enriched_sessions_cache_key = pipeline_config.enriched_sessions_cache_key
    segments_cache_key = pipeline_config.segments_cache_key
    max_segment_length = 20 # Default from prepare_training_data

    try:
        start_index = steps.index(start_step)
    except ValueError:
        print(f"Invalid start step: {start_step}")
        return

    # Step 1: Prepare Training Data
    if start_index <= 0:
        if confirm_step("Step 1: Prepare Training Data"):
            print("\n" + "="*50)
            print(" Step 1: Prepare Training Data")
            print("="*50)

            result = await service.prepare_training_data(top_laps_count=20)
            if not result.get("success"):
                print(f"Error in prepare_training_data: {result.get('error')}")
                return
            
            # Update keys from result if available
            if result.get("max_segment_length"):
                max_segment_length = result.get("max_segment_length")
                
            print("Step 1 completed successfully.")
        else:
            print("Skipping Step 1.")

    # Step 2: Run Segment Annotation App
    if start_index <= 1:
        if confirm_step("Step 2: Run Segment Annotation App"):
            print("\n" + "="*50)
            print(" Step 2: Run Segment Annotation App")
            print("="*50)
            print("Launching Streamlit app for segment annotation...")
            print("Please perform your annotations in the browser.")
            print("When finished, close the Streamlit app (Ctrl+C in terminal) to continue to the next step.")
            
            # Locate the annotation app script
            # We are in acla_ai_service/scripts/run_full_pipeline.py
            # App is in acla_ai_service/ui/segment_annotation_app.py
            app_path = Path(__file__).resolve().parents[1] / "ui" / "segment_annotation_app.py"
            
            if not app_path.exists():
                print(f"Error: Could not find segment_annotation_app.py at {app_path}")
                return

            try:
                # Run streamlit as a subprocess
                subprocess.run(
                    [sys.executable, "-m", "streamlit", "run", str(app_path)],
                    check=True
                )
            except KeyboardInterrupt:
                print("\nStreamlit app closed by user. Continuing...")
            except subprocess.CalledProcessError as e:
                # Streamlit might exit with non-zero if killed, but we want to continue if user is done
                print(f"\nStreamlit app exited with code {e.returncode}. Continuing...")
                
            print("Step 2 completed.")
        else:
            print("Skipping Step 2.")

    # Step 3: Train Segment Classifier
    if start_index <= 2:
        if confirm_step("Step 3: Train Segment Classifier"):
            print("\n" + "="*50)
            print(" Step 3: Train Segment Classifier")
            print("="*50)
            await segment_classifier.train_model()
            print("Step 3 completed successfully.")
        else:
            print("Skipping Step 3.")

    # Step 4: Process and Cache Segments
    if start_index <= 3:
        if confirm_step("Step 4: Process and Cache Segments"):
            print("\n" + "="*50)
            print(" Step 4: Process and Cache Segments")
            print("="*50)
            
            print(f"Using enriched sessions key: {enriched_sessions_cache_key}")
            print(f"Target segments key: {segments_cache_key}")
            
            segments_cache_key = await service.process_and_cache_segments(
                enriched_sessions_cache_key=enriched_sessions_cache_key,
                segments_cache_key=segments_cache_key,
                max_segment_length=max_segment_length
            )
            print(f"Segments cached at: {segments_cache_key}")
            print("Step 4 completed successfully.")
        else:
            print("Skipping Step 4.")

    # Step 5: Run Transformer Guidance Training
    if start_index <= 4:
        if confirm_step("Step 5: Run Transformer Guidance Training"):
            print("\n" + "="*50)
            print(" Step 5: Run Transformer Guidance Training")
            print("="*50)
            result = await service.run_transformer_guidance_training(
                segments_cache_key=segments_cache_key,
                processed_sessions_cache_key=processed_sessions_cache_key,
                max_segment_length=max_segment_length
            )
            if not result.get("success"):
                print(f"Error in run_transformer_guidance_training: {result.get('error')}")
                return
            print("Step 5 completed successfully.")
        else:
            print("Skipping Step 5.")
        
    print("\n" + "="*50)
    print(" Full Pipeline Execution Completed")
    print("="*50)

if __name__ == "__main__":
    asyncio.run(main())
