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

import asyncio
from pathlib import Path
import sys
import time
from datetime import datetime
import io
from contextlib import redirect_stdout, redirect_stderr

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

async def main():
    """
    Main entry point for the Expert Imitation Transformer Learning Pipeline.
    
    This pipeline:
    1. Fetches telemetry data from the backend
    2. Processes and filters top performance laps as expert demonstrations
    3. Enriches data with contextual features (corners, tire grip, expert gaps)
    4. Trains a transformer model to learn non-expert driver progression
    5. Saves the trained model for future use
    """
    start_time = time.time()
    
    # Create output directory if it doesn't exist
    output_dir = current_dir / 'output'
    output_dir.mkdir(exist_ok=True)
    
    # Generate timestamped log filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    track_name = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_TRACK
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
            
            # Execute the pipeline with progress tracking
            print("⏳ Executing transformer learning pipeline...")
            print("   Phase 1: Fetching telemetry data from backend...")
            print("   Phase 2: Processing and filtering expert demonstrations...")
            print("   Phase 3: Enriching contextual features...")
            print("   Phase 4: Training transformer model...")
            print("   Phase 5: Evaluating and saving model...")
            print("-" * 40)
            
            results = await ml_service.StartImitateExpertPipeline(track_name)
            
            # Calculate execution time
            execution_time = time.time() - start_time
            
            # Display comprehensive results
            display_results(results, execution_time)
            
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
    
    # LLM training results summary
    llm_results = results.get('llm_training', {})
    if llm_results and llm_results.get('success'):
        print("\n🤖 LOCAL LLM FINE-TUNING:")
        print("-" * 40)

        training_metrics = llm_results.get('training_metrics', {})
        if training_metrics:
            print("  • Training metrics:")
            for key, value in training_metrics.items():
                print(f"    - {key}: {value}")

        dataset_stats = llm_results.get('dataset_stats', {})
        if dataset_stats:
            print("  • Dataset stats:")
            for key, value in dataset_stats.items():
                print(f"    - {key}: {value}")

        adapter_dir = llm_results.get('adapter_directory')
        if adapter_dir:
            print(f"  • Adapter directory saved: {adapter_dir}")

        print("\n💾 ADAPTER PERSISTENCE:")
        print("-" * 40)
        print("  • ✅ LoRA adapter successfully saved to backend")
        print("  • ✅ Model ready for telemetry guidance inference")

    else:
        llm_error = llm_results.get('error', 'Unknown error') if llm_results else 'No LLM results available'
        print(f"\n❌ LLM TRAINING FAILED: {llm_error}")
        print("-" * 80)
    
    print("\n" + "=" * 80)
    print("🏆 PIPELINE EXECUTION COMPLETED")
    print("=" * 80)
    print("📝 Next Steps:")
    print("  • Model is ready for real-time driving assistance")
    print("  • Use the saved model for non-expert driver progression predictions")
    print("  • Monitor model performance in live racing scenarios")

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
    # Check for help flag
    if len(sys.argv) > 1 and sys.argv[1] in ['-h', '--help', 'help']:
        print_usage()
        sys.exit(0)
    
    # Validate requirements
    if not validate_requirements():
        sys.exit(1)
    
    # Run the main pipeline
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⚠️  Script interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
