#!/usr/bin/env python3
"""
Standalone script to run train_track_cornering_model function only

This script initializes the TelemetryMLService and runs the train_track_cornering_model function
to train corner-specific models based on previously analyzed track cornering data.
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
    Main function to run track cornering model training
    """
    print("=" * 60)
    print("ACLA Track Cornering Model Training Script")
    print("=" * 60)
    
    try:
        # Initialize the TelemetryMLService
        print("[INFO] Initializing TelemetryMLService...")
        models_directory = os.path.join(parent_dir, "models")
        ml_service = Full_dataset_TelemetryMLService(models_directory=models_directory)
        
        print("[INFO] Starting track cornering model training...")
        print("[INFO] This will:")
        print("       - Load previously analyzed cornering data from the backend")
        print("       - Train corner-specific machine learning models")
        print("       - Optimize cornering speed predictions")
        print("       - Save the trained models to the backend")
        print()
        
        # Run the train_track_cornering_model function
        # You can change 'brands_hatch' to any track you want to train on
        track_name = 'brands_hatch'
        
        print(f"[INFO] Training cornering model for track: {track_name}")
        results = await ml_service.train_track_cornering_model(track_name)

        print("\n" + "=" * 60)
        print("TRAINING COMPLETED")
        print("=" * 60)
        
        # Display results summary
        if results and "error" not in results:
            print(f"[SUCCESS] Track cornering model training completed successfully!")
            
            # Check for model training results
            if 'models' in results:
                models = results['models']
                print(f"\n[CORNERING MODELS]")
                print(f"  - Models trained: {len(models) if models else 0}")
                
                for model_name, model_info in models.items():
                    if isinstance(model_info, dict):
                        print(f"  - {model_name}:")
                        if 'performance' in model_info:
                            perf = model_info['performance']
                            print(f"    * R² Score: {perf.get('r2_score', 'N/A')}")
                            print(f"    * MAE: {perf.get('mae', 'N/A')}")
                            print(f"    * RMSE: {perf.get('rmse', 'N/A')}")
            
            # Check for corner analysis results
            if 'corner_analysis' in results:
                corner_analysis = results['corner_analysis']
                print(f"\n[CORNER ANALYSIS]")
                print(f"  - Corners identified: {corner_analysis.get('total_corners', 'N/A')}")
                print(f"  - Analysis completed: {'Yes' if corner_analysis else 'No'}")
            
            # Check for performance metrics
            if 'overall_performance' in results:
                overall = results['overall_performance']
                print(f"\n[OVERALL PERFORMANCE]")
                print(f"  - Training accuracy: {overall.get('training_accuracy', 'N/A')}")
                print(f"  - Validation accuracy: {overall.get('validation_accuracy', 'N/A')}")
                print(f"  - Model confidence: {overall.get('confidence_score', 'N/A')}")
            
            # Check for data statistics
            if 'data_statistics' in results:
                stats = results['data_statistics']
                print(f"\n[DATA STATISTICS]")
                print(f"  - Total cornering samples: {stats.get('total_samples', 'N/A')}")
                print(f"  - Training samples: {stats.get('training_samples', 'N/A')}")
                print(f"  - Validation samples: {stats.get('validation_samples', 'N/A')}")
                print(f"  - Features used: {stats.get('feature_count', 'N/A')}")
            
        elif results and "error" in results:
            print(f"[ERROR] Training failed: {results['error']}")
            return 1
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
