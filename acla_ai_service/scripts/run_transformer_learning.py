#!/usr/bin/env python3
"""
Expert Imitation Transformer Learning Pipeline

This script trains a transformer model to learn how non-expert drivers can progressively
improve their racing performance to reach expert-level driving. The model learns from
telemetry data enriched with contextual features including:

- Expert performance gaps (velocity alignment, speed differences, racing line deviations)
- Track geometry and corner identification features  
- Tire grip and environmental context
- Sequential improvement patterns over time

The trained model can then provide real-time guidance to help non-expert drivers
improve their racing performance through step-by-step action recommendations.

Architecture:
- Input: Non-expert telemetry + expert performance gaps + environmental context
- Output: Improved action sequences (gas, brake, steering, gear) for progression learning
- Uses attention mechanism to focus on relevant improvement patterns

Usage:
    python run_transformer_learning.py [track_name]

Example:
    python run_transformer_learning.py brands_hatch
"""

import argparse
import asyncio
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime
import io
from contextlib import redirect_stdout, redirect_stderr
from typing import List, Optional

# Add the parent directory to the Python path to import modules
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))

from app.services.full_dataset_ml_service import Full_dataset_TelemetryMLService

# Configuration
DEFAULT_TRACK = 'brands_hatch'
SUPPORTED_TRACKS = [
    'brands_hatch', 'spa', 'silverstone', 'monza', 'nurburgring',
    'imola', 'hungaroring', 'paul_ricard', 'barcelona', 'zandvoort'
]


def parse_arguments(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments for the transformer learning pipeline."""

    default_app = Path(__file__).parent.parent / 'ui' / 'telemetry_prompt_annotation_app.py'

    parser = argparse.ArgumentParser(
        description="Expert imitation transformer pipeline with optional annotation UI"
    )
    parser.add_argument('track_name', nargs='?', default=DEFAULT_TRACK, help='Track name to process')
    parser.add_argument(
        '--mode',
        choices=['both', 'annotate', 'train'],
        default='both',
        help='Pipeline mode: annotate only, train using existing annotations, or both sequentially.',
    )
    parser.add_argument(
        '--dataset-path',
        type=Path,
        help='Existing prompt dataset JSONL file to reuse for training mode.',
    )
    parser.add_argument(
        '--skip-ui',
        action='store_true',
        help='Skip launching the Streamlit annotation UI even in annotation modes.',
    )
    parser.add_argument(
        '--ui-port',
        type=int,
        default=8501,
        help='Port for the Streamlit UI (default: 8501).',
    )
    parser.add_argument(
        '--ui-address',
        type=str,
        default='0.0.0.0',
        help="Address for the Streamlit UI to bind to (default: '0.0.0.0').",
    )
    parser.add_argument(
        '--ui-public-host',
        type=str,
        default=None,
        help=(
            "Host/IP to advertise for the Streamlit UI (default resolves to the bind address, "
            "or 'localhost' when binding to 0.0.0.0)."
        ),
    )
    parser.add_argument(
        '--streamlit-app',
        type=Path,
        default=default_app,
        help='Path to the Streamlit annotation application.',
    )
    parser.add_argument(
        '--keep-dataset',
        action='store_true',
        help='Do not delete the dataset file after training completes.',
    )
    parser.add_argument(
        '--no-shuffle',
        action='store_true',
        help='Disable shuffling of windows when generating the dataset.',
    )
    return parser.parse_args(argv)


async def launch_annotation_ui(
    app_path: Path,
    dataset_path: Path,
    dataset_dir: Path,
    *,
    address: Optional[str] = None,
    port: Optional[int] = None,
    public_host: Optional[str] = None,
) -> None:
    """Launch the Streamlit annotation UI and wait for it to exit."""

    # Invoke Streamlit through the active interpreter to avoid PATH issues inside containers.
    cmd = [
        sys.executable,
        '-m',
        'streamlit',
        'run',
        str(app_path),
        '--',
        '--dataset',
        str(dataset_path),
        '--dataset-dir',
        str(dataset_dir),
    ]

    env = os.environ.copy()
    if address:
        env['STREAMLIT_SERVER_ADDRESS'] = address
    if port:
        env['STREAMLIT_SERVER_PORT'] = str(port)

    # Keep the browser target predictable when running inside containers.
    browser_host = public_host
    if not browser_host:
        if address and address != '0.0.0.0':
            browser_host = address
        else:
            browser_host = 'localhost'

    env.setdefault('STREAMLIT_BROWSER_SERVER_ADDRESS', browser_host)
    if port:
        env.setdefault('STREAMLIT_BROWSER_SERVER_PORT', str(port))

    # Silence usage telemetry prompts in container logs unless explicitly overridden.
    env.setdefault('STREAMLIT_BROWSER_GATHERUSAGESTATS', 'false')

    advertised_port = port or env.get('STREAMLIT_BROWSER_SERVER_PORT', '8501')
    print(
        "[INFO] Streamlit annotation UI will be reachable at "
        f"http://{env['STREAMLIT_BROWSER_SERVER_ADDRESS']}:{advertised_port}"
    )

    def _run_streamlit() -> None:
        completed = subprocess.run(cmd, env=env)
        if completed.returncode not in (0, 130):  # 130 == interrupted (Ctrl+C)
            raise RuntimeError(f"Streamlit exited with code {completed.returncode}")

    await asyncio.to_thread(_run_streamlit)

async def main(args: argparse.Namespace):
    """Main entry point for the transformer learning pipeline with flexible modes."""

    start_time = time.time()

    # Create output directory if it doesn't exist
    output_dir = current_dir / 'output'
    output_dir.mkdir(exist_ok=True)

    # Generate timestamped log filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    track_name = args.track_name.lower()
    log_filename = output_dir / f'transformer_learning_{track_name}_{timestamp}.log'
    
    # Create a custom stdout/stderr that writes to both file and console
    class TeeOutput:
        def __init__(self, file_handle, original_stream):
            self.file = file_handle
            self.original = original_stream
        
        def write(self, data):
            self.file.write(data)
            self.file.flush()  # Ensure immediate write
            self.original.write(data)
            self.original.flush()
        
        def flush(self):
            self.file.flush()
            self.original.flush()
    
    # Open log file and redirect output
    with open(log_filename, 'w', encoding='utf-8') as log_file:
        # Create tee outputs
        tee_stdout = TeeOutput(log_file, sys.stdout)
        tee_stderr = TeeOutput(log_file, sys.stderr)
        
        # Redirect stdout and stderr
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = tee_stdout
        sys.stderr = tee_stderr
        
        try:
            print("🚀 Starting Expert Imitation Transformer Learning Pipeline...")
            print("=" * 80)
            print(f"📅 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"📝 Log file: {log_filename}")
            print("=" * 80)
            
            # Initialize ML service
            print("🔧 Initializing ML Service...")
            ml_service = Full_dataset_TelemetryMLService()
            
            print(f"🏁 Target Track: {track_name}")
            
            # Validate track name (optional warning, doesn't block execution)
            if track_name not in SUPPORTED_TRACKS:
                print(f"⚠️  Warning: '{track_name}' is not in the list of commonly supported tracks.")
                print(f"    Supported tracks: {', '.join(SUPPORTED_TRACKS)}")
                print("    Continuing anyway - track name will be passed to backend as-is.")
            
            print("-" * 40)

            print("⏳ Executing pipeline with mode:", args.mode)
            if args.mode in ("annotate", "both"):
                print("   • Preparing dataset for annotation")
            if args.mode in ("both", "train"):
                print("   • Training transformer model")
            print("-" * 40)

            annotation_result = None
            training_result = None
            dataset_path: Optional[Path] = args.dataset_path

            if args.mode in ("annotate", "both"):
                annotation_result = await ml_service.StartImitateExpertPipeline(
                    trackName=track_name,
                    pipeline_mode="annotate",
                    shuffle_dataset=not args.no_shuffle,
                )

                dataset_path_value = annotation_result.get("dataset_path")
                if dataset_path_value:
                    dataset_path = Path(dataset_path_value)
                    print(f"[INFO] Dataset created at {dataset_path}")
                else:
                    raise RuntimeError("Dataset path missing from annotation stage results")

                if dataset_path and not args.skip_ui:
                    app_path = args.streamlit_app
                    if not app_path.exists():
                        raise FileNotFoundError(f"Streamlit app not found at {app_path}")

                    print("[INFO] Launching Streamlit annotation UI. Close the UI when finished annotating.")
                    await launch_annotation_ui(
                        app_path=app_path,
                        dataset_path=dataset_path,
                        dataset_dir=dataset_path.parent,
                        address=args.ui_address,
                        port=args.ui_port,
                        public_host=args.ui_public_host,
                    )
                    refreshed_stats = ml_service._summarize_dataset(dataset_path)
                    annotation_result["dataset_stats"] = refreshed_stats
                else:
                    print("[INFO] Skipping annotation UI launch (per configuration).")

                if args.mode == "annotate":
                    execution_time = time.time() - start_time
                    final_results = {
                        "success": True,
                        "track_name": track_name,
                        "mode": "annotate",
                        "annotation": annotation_result,
                    }
                    display_results(final_results, execution_time)
                    print(f"\n✅ All output has been saved to: {log_filename}")
                    return

            if args.mode in ("both", "train"):
                if dataset_path is None:
                    raise ValueError(
                        "Dataset path required for training. Provide --dataset-path or run annotate mode first."
                    )

                if args.mode == "train" and annotation_result is None:
                    annotation_result = {
                        "dataset_path": str(dataset_path),
                        "dataset_stats": ml_service._summarize_dataset(dataset_path),
                        "mode": "train",
                        "success": True,
                    }

                training_result = await ml_service.StartImitateExpertPipeline(
                    trackName=track_name,
                    pipeline_mode="train-only",
                    dataset_path=dataset_path,
                    cleanup_dataset_file=not args.keep_dataset,
                )

            execution_time = time.time() - start_time

            pipeline_success = True
            if training_result is not None:
                pipeline_success = bool(training_result.get("success", False))

            final_results = {
                "success": pipeline_success,
                "track_name": track_name,
                "mode": args.mode,
                "annotation": annotation_result,
                "training": training_result,
            }

            display_results(final_results, execution_time)
            
            print(f"\n✅ All output has been saved to: {log_filename}")
            
        except KeyboardInterrupt:
            print("\n⚠️  Pipeline interrupted by user")
            return
        except Exception as e:
            execution_time = time.time() - start_time
            print(f"\n❌ Critical Error occurred after {execution_time:.1f}s: {e}")
            print("-" * 80)
            import traceback
            traceback.print_exc()
            print("-" * 80)
            return
        finally:
            # Restore original stdout/stderr
            sys.stdout = old_stdout
            sys.stderr = old_stderr

def display_results(results, execution_time: float):
    """
    Display comprehensive results from the transformer learning pipeline
    
    Args:
        results: Dictionary containing all pipeline results
        execution_time: Total execution time in seconds
    """
    print("\n" + "=" * 80)
    print("🎯 TELEMETRY LLM TRAINING PIPELINE RESULTS")
    print("=" * 80)
    print(f"⏱️  Total Execution Time: {execution_time:.1f}s ({execution_time/60:.1f}m)")
    print(f"📅 Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if not results.get('success', False):
        print(f"\n❌ Pipeline failed with error: {results.get('error', 'Unknown error')}")
        print("-" * 80)
        return
    
    print("\n✅ Pipeline completed successfully!")
    print(f"🏁 Track: {results.get('track_name', 'Unknown')}")

    annotation_info = results.get('annotation') or {}
    training_wrapper = results.get('training') or {}

    dataset_path = annotation_info.get('dataset_path') or training_wrapper.get('dataset_path')
    dataset_stats = annotation_info.get('dataset_stats') or training_wrapper.get('dataset_stats') or {}

    if dataset_path or dataset_stats:
        print("\n📦 DATASET SUMMARY:")
        print("-" * 40)
        if dataset_path:
            print(f"  • Dataset file: {dataset_path}")
        if dataset_stats:
            for key, value in dataset_stats.items():
                if key == 'dataset_path':
                    continue
                print(f"  • {key.replace('_', ' ').title()}: {value}")

    llm_results = training_wrapper.get('llm_training') if training_wrapper else None
    if llm_results and llm_results.get('success'):
        print("\n🤖 LOCAL LLM FINE-TUNING:")
        print("-" * 40)

        training_metrics = llm_results.get('training_metrics', {})
        if training_metrics:
            print("  • Training metrics:")
            for key, value in training_metrics.items():
                print(f"    - {key}: {value}")

        adapter_dir = llm_results.get('adapter_directory')
        if adapter_dir:
            print(f"  • Adapter directory saved: {adapter_dir}")

        print("\n💾 ADAPTER PERSISTENCE:")
        print("-" * 40)
        print("  • ✅ LoRA adapter successfully saved to backend")
        print("  • ✅ Model ready for telemetry guidance inference")

    elif training_wrapper:
        error_message = training_wrapper.get('error') or 'No training metrics available'
        print(f"\n❌ LLM TRAINING FAILED: {error_message}")
        print("-" * 80)
    
    print("\n" + "=" * 80)
    print("🏆 PIPELINE EXECUTION COMPLETED")
    print("=" * 80)
    print("📝 Next Steps:")
    if llm_results and llm_results.get('success'):
        print("  • Model is ready for real-time driving assistance")
        print("  • Use the saved model for non-expert driver progression predictions")
        print("  • Monitor model performance in live racing scenarios")
    else:
        print("  • Review dataset annotations and re-run training when ready")

def print_usage():
    """Print usage information for the script"""
    print("Telemetry LLM Training Pipeline")
    print("==============================================")
    print("")
    print("Usage: python run_transformer_learning.py [track_name]")
    print("")
    print("Arguments:")
    print(f"  track_name    Track name to train on (default: {DEFAULT_TRACK})")
    print("")
    print("Examples:")
    print("  python run_transformer_learning.py brands_hatch")
    print("  python run_transformer_learning.py spa")
    print("  python run_transformer_learning.py silverstone")
    print("")
    print("Commonly supported tracks:")
    print(f"  {', '.join(SUPPORTED_TRACKS)}")
    print("")
    print("Note: Available tracks depend on your telemetry data in the backend.")
    print("      The script will work with any track name that has data available.")

def validate_requirements():
    """
    Validate that all required dependencies are available
    """
    missing_deps = []
    
    try:
        import torch
    except ImportError:
        missing_deps.append("PyTorch (torch)")
    
    try:
        import numpy
    except ImportError:
        missing_deps.append("NumPy")
    
    try:
        import pandas
    except ImportError:
        missing_deps.append("Pandas")
    
    if missing_deps:
        print("❌ Missing required dependencies:")
        for dep in missing_deps:
            print(f"  • {dep}")
        print("\nPlease install missing dependencies and try again.")
        return False
    
    return True

if __name__ == "__main__":
    cli_args = parse_arguments(sys.argv[1:])

    if not validate_requirements():
        sys.exit(1)

    try:
        asyncio.run(main(cli_args))
    except KeyboardInterrupt:
        print("\n⚠️  Script interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
