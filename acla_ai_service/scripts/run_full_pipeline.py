import asyncio
import argparse
import sys
import subprocess
from pathlib import Path
import os

# Add workspace root to path to allow imports
# parents[0] = scripts
# parents[1] = acla_ai_service
# parents[2] = ACLA (workspace root)
sys.path.append(str(Path(__file__).resolve().parents[2]))

from acla_ai_service.app.services.full_dataset_ml_service import Full_dataset_TelemetryMLService, PipelineConfig
from acla_ai_service.app.services.segment_classifier_service import segment_classifier

async def main():
    parser = argparse.ArgumentParser(description="Run the full ACLA AI pipeline.")
    parser.add_argument(
        "--start-step",
        type=str,
        choices=["prepare_data", "annotate", "train_classifier", "process_segments", "train_transformer"],
        default="prepare_data",
        help="Step to start execution from."
    )
    parser.add_argument("--track-name", type=str, default="monza", help="Track name for data preparation.")
    args = parser.parse_args()

    print(f"Initializing services... (Track: {args.track_name})")
    service = Full_dataset_TelemetryMLService()
    pipeline_config = PipelineConfig()
    
    # Default keys
    processed_sessions_cache_key = pipeline_config.processed_session_data_cache_key
    enriched_sessions_cache_key = pipeline_config.enriched_sessions_cache_key
    segments_cache_key = pipeline_config.segments_cache_key
    max_segment_length = 20 # Default from prepare_training_data

    steps = ["prepare_data", "annotate", "train_classifier", "process_segments", "train_transformer"]
    try:
        start_index = steps.index(args.start_step)
    except ValueError:
        print(f"Invalid start step: {args.start_step}")
        return

    # Step 1: Prepare Training Data
    if start_index <= 0:
        print("\n" + "="*50)
        print(" Step 1: Prepare Training Data")
        print("="*50)
        result = await service.prepare_training_data(track_name=args.track_name)
        if not result.get("success"):
            print(f"Error in prepare_training_data: {result.get('error')}")
            return
        
        # Update keys from result if available
        if result.get("processed_sessions_cache_key"):
            processed_sessions_cache_key = result.get("processed_sessions_cache_key")
        if result.get("max_segment_length"):
            max_segment_length = result.get("max_segment_length")
            
        print("Step 1 completed successfully.")

    # Step 2: Run Segment Annotation App
    if start_index <= 1:
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

    # Step 3: Train Segment Classifier
    if start_index <= 2:
        print("\n" + "="*50)
        print(" Step 3: Train Segment Classifier")
        print("="*50)
        await segment_classifier.train_model()
        print("Step 3 completed successfully.")

    # Step 4: Process and Cache Segments
    if start_index <= 3:
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

    # Step 5: Run Transformer Guidance Training
    if start_index <= 4:
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
        
    print("\n" + "="*50)
    print(" Full Pipeline Execution Completed")
    print("="*50)

if __name__ == "__main__":
    asyncio.run(main())
