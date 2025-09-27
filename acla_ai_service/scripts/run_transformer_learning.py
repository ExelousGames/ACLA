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
    print("🚀 Starting Expert Imitation Transformer Learning Pipeline...")
    print("=" * 80)
    print(f"📅 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    try:
        # Initialize ML service
        print("🔧 Initializing ML Service...")
        ml_service = Full_dataset_TelemetryMLService()
        
        # Get track name from command line args or use default
        track_name = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_TRACK
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

def display_results(results, execution_time: float):
    """
    Display comprehensive results from the transformer learning pipeline
    
    Args:
        results: Dictionary containing all pipeline results
        execution_time: Total execution time in seconds
    """
    print("\n" + "=" * 80)
    print("🎯 TRANSFORMER LEARNING PIPELINE RESULTS")
    print("=" * 80)
    print(f"⏱️  Total Execution Time: {execution_time:.1f}s ({execution_time/60:.1f}m)")
    print(f"📅 Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if not results.get('success', False):
        print(f"\n❌ Pipeline failed with error: {results.get('error', 'Unknown error')}")
        print("-" * 80)
        return
    
    print("\n✅ Pipeline completed successfully!")
    print(f"🏁 Track: {results.get('track_name', 'Unknown')}")
    
    # Transformer Training Results (Updated structure based on actual implementation)
    transformer_results = results.get('transformer_training', {})
    if transformer_results and transformer_results.get('success'):
        print("\n🤖 TRANSFORMER MODEL TRAINING:")
        print("-" * 40)
        
        # Model Information
        model_info = transformer_results.get('model_info', {})
        if model_info:
            print(f"  • Model parameters: {model_info.get('parameters', 0):,}")
            print(f"  • Model size: {model_info.get('model_size_mb', 0.0):.2f} MB")
            print(f"  • Input features: {model_info.get('input_features', 'N/A')}")
            print(f"  • Sequence length: {model_info.get('sequence_length', 'N/A')}")
        
        # Training History Analysis
        training_history = transformer_results.get('training_history', {})
        if training_history:
            training_results = training_history.get('training_results', {})
            if training_results:
                print(f"  • Training completed: {'✅ Yes' if training_results.get('training_completed') else '❌ No'}")
                print(f"  • Total epochs trained: {training_results.get('total_epochs', 0)}")
                
                best_val_loss = training_results.get('best_val_loss')
                if best_val_loss is not None:
                    print(f"  • Best validation loss: {best_val_loss:.6f}")
                
                # Training History Details
                history_list = training_results.get('training_history', [])
                if history_list:
                    print(f"  • Training epochs recorded: {len(history_list)}")
                    
                    # Show first and last epoch metrics
                    first_epoch = history_list[0] if history_list else {}
                    last_epoch = history_list[-1] if history_list else {}
                    
                    if first_epoch and last_epoch:
                        first_train_loss = first_epoch.get('train_metrics', {}).get('total_loss', 0)
                        last_train_loss = last_epoch.get('train_metrics', {}).get('total_loss', 0)
                        
                        if first_train_loss > 0 and last_train_loss > 0:
                            print(f"  • Initial training loss: {first_train_loss:.6f}")
                            print(f"  • Final training loss: {last_train_loss:.6f}")
                            
                            improvement = ((first_train_loss - last_train_loss) / first_train_loss) * 100
                            print(f"  • Training loss improvement: {improvement:.2f}%")
                        
                        # Validation metrics if available
                        first_val = first_epoch.get('val_metrics')
                        last_val = last_epoch.get('val_metrics')
                        if first_val and last_val:
                            first_val_loss = first_val.get('total_loss', 0)
                            last_val_loss = last_val.get('total_loss', 0)
                            if first_val_loss > 0 and last_val_loss > 0:
                                print(f"  • Initial validation loss: {first_val_loss:.6f}")
                                print(f"  • Final validation loss: {last_val_loss:.6f}")
                                
                                val_improvement = ((first_val_loss - last_val_loss) / first_val_loss) * 100
                                print(f"  • Validation loss improvement: {val_improvement:.2f}%")
        
        # Test Metrics
        test_metrics = transformer_results.get('test_metrics', {})
        if test_metrics:
            print("\n📊 MODEL EVALUATION METRICS:")
            print("-" * 40)
            
            # Individual action metrics
            action_metrics = test_metrics.get('individual_action_metrics', {})
            for action, metrics in action_metrics.items():
                if isinstance(metrics, dict):
                    mse = metrics.get('mse', 0)
                    mae = metrics.get('mae', 0)
                    print(f"  • {action.capitalize()}: MSE={mse:.6f}, MAE={mae:.6f}")
            
            # Overall metrics
            overall_mse = test_metrics.get('overall_mse', 0)
            overall_mae = test_metrics.get('overall_mae', 0)
            if overall_mse > 0 or overall_mae > 0:
                print(f"  • Overall: MSE={overall_mse:.6f}, MAE={overall_mae:.6f}")
        
        # Model saved confirmation
        if transformer_results.get('serialized_model'):
            print("\n💾 MODEL PERSISTENCE:")
            print("-" * 40)
            print("  • ✅ Model successfully saved to backend")
            print("  • ✅ Model ready for inference and prediction")
            
    else:
        transformer_error = transformer_results.get('error', 'Unknown error') if transformer_results else 'No transformer results available'
        print(f"\n❌ TRANSFORMER TRAINING FAILED: {transformer_error}")
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
    print("Expert Imitation Transformer Learning Pipeline")
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
