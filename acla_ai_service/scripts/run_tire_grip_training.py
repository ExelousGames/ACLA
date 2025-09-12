#!/usr/bin/env python3
"""
Standalone script to run tire grip analysis model training

This script initializes the TelemetryMLService and runs the train_tire_grip_model function
to train models that estimate tire grip, friction circle utilization, weight transfer,
and predictive load calculations.
"""

import os
import sys
import asyncio
import json
from pathlib import Path

# Add the parent directory to the Python path to import modules
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))

# Import the TelemetryMLService
from app.services.full_dataset_ml_service import Full_dataset_TelemetryMLService


async def main():
    """
    Main function to run tire grip analysis training
    """
    print("=" * 70)
    print("ACLA Tire Grip and Friction Circle Analysis Training Script")
    print("=" * 70)
    
    try:
        # Initialize the TelemetryMLService
        print("[INFO] Initializing TelemetryMLService...")
        models_directory = os.path.join(parent_dir, "models")
        ml_service = Full_dataset_TelemetryMLService(models_directory=models_directory)
        
        print("\n[INFO] Starting tire grip analysis model training...")
        print("[INFO] This will:")
        print("       - Retrieve all racing sessions from the backend")
        print("       - Extract physics telemetry data (G-forces, slip angles, temperatures)")
        print("       - Calculate friction circle utilization and weight transfer")
        print("       - Train multiple ML models for different grip metrics")
        print("       - Output new telemetry features for enhanced analysis")
        print("       - Save trained models to backend and local storage")
        print()
        
        # Define training parameters
        track_name = 'brands_hatch'  # You can change this
        car_name = 'porsche_991ii_gt3_r'  # You can change this or set to None for all cars
        
        print(f"[INFO] Training for track: {track_name}")
        print(f"[INFO] Training for car: {car_name if car_name else 'All Cars'}")
        print()
        
        # Run the tire grip analysis training
        print("[INFO] Training tire grip analysis models...")
        results = await ml_service.train_tire_grip_model(track_name, car_name)

        print("\n" + "=" * 70)
        print("TIRE GRIP ANALYSIS TRAINING COMPLETED")
        print("=" * 70)
        
        if "error" in results:
            print(f"[ERROR] Training failed: {results['error']}")
            return
        
        # Display results
        print(f"[SUCCESS] Training completed successfully!")
        print(f"[INFO] Track: {results.get('track_name', 'N/A')}")
        print(f"[INFO] Car: {results.get('car_name', 'N/A')}")
        print(f"[INFO] Models trained: {results.get('models_trained', 0)}/{results.get('total_targets', 0)}")
        print(f"[INFO] Average R² Score: {results.get('average_r2_score', 0):.4f}")
        print(f"[INFO] Training samples: {results.get('training_samples', 0)}")
        print(f"[INFO] Features used: {len(results.get('feature_names', []))}")
        
        print(f"\n[INFO] Individual Model Performance:")
        models_results = results.get('models_results', {})
        for target_name, model_result in models_results.items():
            if 'error' not in model_result:
                r2 = model_result.get('r2_score', 0)
                mae = model_result.get('mae', 0)
                print(f"       - {target_name}: R² = {r2:.4f}, MAE = {mae:.4f}")
            else:
                print(f"       - {target_name}: FAILED - {model_result['error']}")
        
        print(f"\n[INFO] New telemetry features that will be available:")
        tire_grip_features = [
            "friction_circle_utilization",
            "friction_circle_utilization_front", 
            "friction_circle_utilization_rear",
            "longitudinal_weight_transfer",
            "lateral_weight_transfer", 
            "dynamic_weight_distribution",
            "optimal_grip_window",
            "slip_angle_efficiency",
            "slip_ratio_efficiency",
            "overall_tire_grip",
            "tire_saturation_level",
            "estimated_tire_grip_fl",
            "estimated_tire_grip_fr", 
            "estimated_tire_grip_rl",
            "estimated_tire_grip_rr",
            "predictive_load_fl",
            "predictive_load_fr",
            "predictive_load_rl", 
            "predictive_load_rr"
        ]
        
        for i, feature in enumerate(tire_grip_features, 1):
            print(f"       {i:2d}. {feature}")
        
        print(f"\n[INFO] These features will be automatically added to telemetry data")
        print(f"[INFO] and can be used for:")
        print(f"       - Real-time driving analysis")
        print(f"       - Setup optimization recommendations") 
        print(f"       - Driver coaching and feedback")
        print(f"       - Performance comparison and benchmarking")
        print(f"       - Predictive tire degradation analysis")
        
        # Test feature extraction with a sample
        print(f"\n[INFO] Testing feature extraction...")
        try:
            # Get model summary
            summary = ml_service.get_tire_grip_model_summary(track_name, car_name)
            print(f"[INFO] Model Summary:")
            print(f"       - Available models: {len(summary.get('available_models', []))}")
            print(f"       - Cached models: {summary.get('cached_models', 0)}")
            print(f"       - Models directory: {summary.get('models_directory', 'N/A')}")
            
        except Exception as e:
            print(f"[WARNING] Failed to get model summary: {str(e)}")
        
        print(f"\n[SUCCESS] Training script completed successfully!")
        print(f"[INFO] Models are ready for use in feature extraction")
        
    except Exception as e:
        print(f"\n[ERROR] An error occurred during training: {str(e)}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    """
    Run the tire grip analysis training script
    
    Usage:
        python scripts/run_tire_grip_training.py
        
    Or from Docker:
        docker exec -it acla_ai_service_c python scripts/run_tire_grip_training.py
    """
    asyncio.run(main())
