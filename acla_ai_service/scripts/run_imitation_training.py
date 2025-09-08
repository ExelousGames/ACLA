#!/usr/bin/env python3
"""
Standalone script to run train_imitation_model function only

This script initializes the TelemetryMLService and runs the train_imitation_model function
to train imitation learning models from expert driving demonstrations.
"""

import os
import sys
import asyncio
from pathlib import Path

# Add the parent directory to the Python path to import modules
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))

# Import the TelemetryMLService
from app.services.full_dataset_ml_service import Full_dataset_TelemetryMLService


async def main():
    """
    Main function to run imitation learning training
    """
    print("=" * 60)
    print("ACLA Imitation Learning Training Script")
    print("=" * 60)
    
    try:
        # Initialize the TelemetryMLService
        print("[INFO] Initializing TelemetryMLService...")
        models_directory = os.path.join(parent_dir, "models")
        ml_service = Full_dataset_TelemetryMLService(models_directory=models_directory)
        
        print("[INFO] Starting imitation learning training...")
        print("[INFO] This will:")
        print("       - Retrieve all racing sessions from the backend")
        print("       - Extract telemetry data from each session")
        print("       - Train behavior and trajectory learning models")
        print("       - Serialize and save the trained models")
        print("       - Save results to the backend")
        print()
        
        # Run the train_imitation_model function
        results = await ml_service.train_imitation_model('brands_hatch', 'porsche_991ii_gt3_r')

        print("\n" + "=" * 60)
        print("TRAINING COMPLETED")
        print("=" * 60)
        
        # Display results summary
        if results:
            print(f"[SUCCESS] Imitation learning training completed successfully!")
            
            # Check for behavior learning results
            if 'behavior_learning' in results:
                behavior_results = results['behavior_learning']
                print(f"\n[BEHAVIOR LEARNING]")
                print(f"  - Model trained: {'Yes' if 'model' in behavior_results else 'No'}")
                if 'performance' in behavior_results:
                    perf = behavior_results['performance']
                    print(f"  - Accuracy: {perf.get('accuracy', 'N/A')}")
                    print(f"  - Precision: {perf.get('precision', 'N/A')}")
                    print(f"  - Recall: {perf.get('recall', 'N/A')}")
                    print(f"  - F1 Score: {perf.get('f1_score', 'N/A')}")
            
            # Check for trajectory learning results
            if 'trajectory_learning' in results:
                trajectory_results = results['trajectory_learning']
                print(f"\n[TRAJECTORY LEARNING]")
                print(f"  - Models trained: {'Yes' if 'trajectory_model' in trajectory_results else 'No'}")
                if 'performance' in trajectory_results:
                    perf = trajectory_results['performance']
                    for action, metrics in perf.items():
                        if isinstance(metrics, dict):
                            print(f"  - {action}: R² = {metrics.get('r2_score', 'N/A')}, MAE = {metrics.get('mae', 'N/A')}")
            
            # Check for data statistics
            if 'data_statistics' in results:
                stats = results['data_statistics']
                print(f"\n[DATA STATISTICS]")
                print(f"  - Total telemetry records: {stats.get('total_records', 'N/A')}")
                print(f"  - Training samples: {stats.get('training_samples', 'N/A')}")
                print(f"  - Validation samples: {stats.get('validation_samples', 'N/A')}")
                print(f"  - Features used: {stats.get('feature_count', 'N/A')}")
            
        else:
            print("[WARNING] Training completed but no results returned")
            
    except Exception as e:
        print(f"\n[ERROR] Training failed: {str(e)}")
        print(f"[ERROR] Error type: {type(e).__name__}")
        import traceback
        print(f"[ERROR] Traceback:\n{traceback.format_exc()}")
        return 1
    
    print("\n" + "=" * 60)
    print("Script execution completed.")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    # Run the async main function
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
