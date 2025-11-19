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
import importlib.util
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
    """Parse CLI arguments for the unified transformer → LLM training pipeline."""

    parser = argparse.ArgumentParser(
        description="Train the ExpertActionTransformer and fine-tune the guidance LLM on its predictions"
    )
    parser.add_argument('track_name', nargs='?', default=DEFAULT_TRACK, help='Track name to process.')
    parser.add_argument(
        '--transformer-epochs',
        type=int,
        default=24,
        help='Number of epochs for transformer training (default: 24).',
    )
    parser.add_argument(
        '--transformer-patience',
        type=int,
        default=6,
        help='Early stopping patience for transformer training (default: 6).',
    )
    parser.add_argument(
        '--transformer-batch-size',
        type=int,
        default=32,
        help='Number of segments per GPU batch when training the transformer (default: 32).',
    )
    parser.add_argument(
        '--keep-dataset',
        action='store_true',
        help='Keep the generated JSONL dataset after LLM training (default: delete).',
    )
    parser.add_argument(
        '--no-shuffle',
        action='store_true',
        help='Disable shuffling of prompt windows before writing the dataset.',
    )
    return parser.parse_args(argv)


async def main(args: argparse.Namespace):
    """Main entry point for the unified transformer → LLM training workflow."""

    start_time = time.time()

    output_dir = current_dir / 'output'
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    track_name = args.track_name.lower()
    log_filename = output_dir / f'transformer_learning_{track_name}_{timestamp}.log'

    class TeeOutput:
        def __init__(self, file_handle, original_stream):
            self.file = file_handle
            self.original = original_stream

        def write(self, data):
            self.file.write(data)
            self.file.flush()
            self.original.write(data)
            self.original.flush()

        def flush(self):
            self.file.flush()
            self.original.flush()

    with open(log_filename, 'w', encoding='utf-8') as log_file:
        tee_stdout = TeeOutput(log_file, sys.stdout)
        tee_stderr = TeeOutput(log_file, sys.stderr)

        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = tee_stdout
        sys.stderr = tee_stderr

        try:
            print("🚀 Starting Transformer ➜ LLM Guidance Training Pipeline...")
            print("=" * 80)
            print(f"📅 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"📝 Log file: {log_filename}")
            print("=" * 80)

            print("🔧 Initializing ML Service...")
            ml_service = Full_dataset_TelemetryMLService()

            print(f"🏁 Target Track: {track_name}")
            if track_name not in SUPPORTED_TRACKS:
                print(f"⚠️  Warning: '{track_name}' is not in the list of commonly supported tracks.")
                print(f"    Supported tracks: {', '.join(SUPPORTED_TRACKS)}")
                print("    Continuing anyway – backend availability determines success.")

            print("-" * 40)
            print("⏳ Executing unified training stack")
            print("   • Training ExpertActionTransformer")
            print("   • Generating transformer-driven prompt dataset")
            print("   • Fine-tuning guidance LLM on predicted plans")
            print("-" * 40)

            result = await ml_service.run_transformer_guidance_training(
                track_name=track_name,
                shuffle_dataset=not args.no_shuffle,
                cleanup_dataset_file=not args.keep_dataset,
            )

            execution_time = time.time() - start_time
            display_results(result, execution_time, log_filename)

        except KeyboardInterrupt:
            print("\n⚠️  Pipeline interrupted by user")
            return
        except Exception as pipeline_error:
            execution_time = time.time() - start_time
            print(f"\n❌ Critical error occurred after {execution_time:.1f}s: {pipeline_error}")
            print("-" * 80)
            import traceback
            traceback.print_exc()
            print("-" * 80)
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            sys.exit(1)
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

def display_results(results, execution_time: float, log_path: Path):
    """
    Display comprehensive results from the transformer learning pipeline
    
    Args:
        results: Dictionary containing all pipeline results
        execution_time: Total execution time in seconds
        log_path: Path to the pipeline execution log file
    """
    print("\n" + "=" * 80)
    print("🎯 TELEMETRY LLM TRAINING PIPELINE RESULTS")
    print("=" * 80)
    print(f"⏱️  Total Execution Time: {execution_time:.1f}s ({execution_time/60:.1f}m)")
    print(f"📅 Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📝 Full log: {Path(log_path).resolve()}")
    
    if not results.get('success', False):
        error_msg = results.get('error', 'Unknown error')
        print(f"\n❌ Pipeline failed with error: {error_msg}")
        print("-" * 80)
        print("\n💡 Common causes:")
        print("  • Insufficient training data (need at least 10 annotated examples)")
        print("  • Dataset annotation was incomplete or skipped")
        print("  • Backend service connection issues")
        print("\n📝 Check the full log above for detailed error information")
        return
    
    print("\n✅ Pipeline completed successfully!")
    print(f"🏁 Track: {results.get('track_name', 'Unknown')}")

    dataset_path = results.get('dataset_path')
    dataset_stats = results.get('dataset_stats') or {}
    if dataset_path or dataset_stats:
        print("\n📦 DATASET SUMMARY:")
        print("-" * 48)
        if dataset_path:
            print(f"  • Dataset file: {dataset_path}")
        if dataset_stats:
            for key, value in dataset_stats.items():
                if key == 'dataset_path':
                    continue
                label = key.replace('_', ' ').title()
                print(f"  • {label}: {value}")

    transformer_training = results.get('transformer_training') or {}
    transformer_metrics = transformer_training.get('training_metrics') or {}
    transformer_metadata = transformer_training.get('metadata') or {}

    if transformer_metrics or transformer_metadata:
        print("\n🛠️  TRANSFORMER TRAINING SUMMARY:")
        print("-" * 48)
        if transformer_metrics:
            print("  • Metrics:")
            for key, value in transformer_metrics.items():
                print(f"    - {key}: {value}")
        if transformer_metadata:
            epochs = transformer_metadata.get('epochs')
            patience = transformer_metadata.get('patience')
            batch_size = transformer_metadata.get('batch_size')
            if any(v is not None for v in (epochs, patience, batch_size)):
                print("  • Hyperparameters:")
                if epochs is not None:
                    print(f"    - epochs: {epochs}")
                if patience is not None:
                    print(f"    - patience: {patience}")
                if batch_size is not None:
                    print(f"    - batch size: {batch_size}")

    llm_training = results.get('llm_training') or {}
    if llm_training.get('success'):
        print("\n🤖 LOCAL LLM FINE-TUNING:")
        print("-" * 48)

        training_metrics = llm_training.get('training_metrics', {})
        if training_metrics:
            print("  • Training metrics:")
            for key, value in training_metrics.items():
                print(f"    - {key}: {value}")

        adapter_dir = llm_training.get('adapter_directory')
        if adapter_dir:
            print(f"  • Adapter directory saved: {adapter_dir}")

        print("\n💾 ADAPTER PERSISTENCE:")
        print("-" * 48)
        print("  • ✅ LoRA adapter successfully saved to backend")
        print("  • ✅ Model ready for telemetry guidance inference")

    elif llm_training:
        error_message = llm_training.get('error') or 'No training metrics available'
        print(f"\n❌ LLM TRAINING FAILED: {error_message}")
        print("-" * 80)
    
    print("\n" + "=" * 80)
    print("🏆 PIPELINE EXECUTION COMPLETED")
    print("=" * 80)
    print("📝 Next Steps:")
    if llm_training and llm_training.get('success'):
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
    required_modules = {
        "torch": "PyTorch (torch)",
        "numpy": "NumPy",
        "pandas": "Pandas",
    }

    missing_deps = [
        friendly_name
        for module_name, friendly_name in required_modules.items()
        if importlib.util.find_spec(module_name) is None
    ]

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
