#!/usr/bin/env python3
"""
Standalone script to demonstrate tire grip feature extraction

This script shows how to use trained tire grip models to extract new features
from telemetry data and insert them back into the dataset.
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
    Main function to demonstrate tire grip feature extraction
    """
    print("=" * 75)
    print("ACLA Tire Grip Feature Extraction Demonstration")
    print("=" * 75)
    
    try:
        # Initialize the TelemetryMLService
        print("[INFO] Initializing TelemetryMLService...")
        models_directory = os.path.join(parent_dir, "models")
        ml_service = Full_dataset_TelemetryMLService(models_directory=models_directory)
        
        # Define parameters
        track_name = 'brands_hatch'
        car_name = 'porsche_991ii_gt3_r'
        
        print(f"\n[INFO] Extracting tire grip features for:")
        print(f"       - Track: {track_name}")
        print(f"       - Car: {car_name if car_name else 'All Cars'}")
        print()
        
        # First, let's check if we have trained models
        print("[INFO] Checking for trained models...")
        try:
            summary = ml_service.get_tire_grip_model_summary(track_name, car_name)
            available_models = summary.get('available_models', [])
            
            if not available_models:
                print("[WARNING] No trained tire grip models found!")
                print("[INFO] Running training first...")
                
                # Train models first
                train_results = await ml_service.train_tire_grip_model(track_name, car_name)
                if "error" in train_results:
                    print(f"[ERROR] Training failed: {train_results['error']}")
                    return
                    
                print(f"[SUCCESS] Training completed! {train_results.get('models_trained', 0)} models trained")
            else:
                print(f"[INFO] Found {len(available_models)} trained models:")
                for model in available_models:
                    print(f"       - {model.get('target_name', 'unknown')}")
        
        except Exception as e:
            print(f"[WARNING] Could not check models: {str(e)}")
        
        # Get sample telemetry data from backend
        print(f"\n[INFO] Retrieving sample telemetry data...")
        try:
            from app.services.backend_service import backend_service
            sessions = await backend_service.get_all_racing_sessions(track_name, car_name)
            
            # Get first session data as sample
            sample_telemetry_data = []
            for session in sessions.get("sessions", [])[:1]:  # Just first session
                session_data = session.get("data", [])
                if session_data:
                    # Take a sample of records (first 100 for demo)
                    sample_telemetry_data = session_data[:100]
                    break
            
            if not sample_telemetry_data:
                print("[ERROR] No telemetry data found for demonstration")
                return
                
            print(f"[INFO] Retrieved {len(sample_telemetry_data)} telemetry records for demonstration")
            
            # Show original data structure (first record)
            if sample_telemetry_data:
                original_keys = list(sample_telemetry_data[0].keys())
                print(f"\n[INFO] Original telemetry data contains {len(original_keys)} fields")
                print("[INFO] Sample original fields:")
                for i, key in enumerate(sorted(original_keys)[:10]):
                    print(f"       {i+1:2d}. {key}")
                if len(original_keys) > 10:
                    print(f"       ... and {len(original_keys) - 10} more fields")
            
        except Exception as e:
            print(f"[ERROR] Failed to retrieve telemetry data: {str(e)}")
            return
        
        # Extract tire grip features
        print(f"\n[INFO] Extracting tire grip features from {len(sample_telemetry_data)} records...")
        try:
            enhanced_data = await ml_service.extract_tire_grip_features(
                sample_telemetry_data, 
                track_name, 
                car_name
            )
            
            if not enhanced_data:
                print("[ERROR] Feature extraction returned no data")
                return
                
            print(f"[SUCCESS] Successfully extracted features for {len(enhanced_data)} records")
            
            # Show enhanced data structure
            if enhanced_data:
                enhanced_keys = list(enhanced_data[0].keys())
                original_keys_set = set(sample_telemetry_data[0].keys())
                new_keys = [k for k in enhanced_keys if k not in original_keys_set]
                
                print(f"\n[INFO] Enhanced telemetry data now contains {len(enhanced_keys)} fields")
                print(f"[INFO] Added {len(new_keys)} new tire grip features:")
                
                for i, key in enumerate(sorted(new_keys), 1):
                    # Get sample values for this feature
                    sample_values = [record.get(key, 0) for record in enhanced_data[:5]]
                    avg_value = sum(sample_values) / len(sample_values) if sample_values else 0
                    print(f"       {i:2d}. {key:<35} (avg: {avg_value:.4f})")
                
                # Show detailed analysis of first few records
                print(f"\n[INFO] Detailed analysis of first 3 records:")
                tire_grip_features = [k for k in new_keys if 'grip' in k.lower() or 'friction' in k.lower() or 'weight' in k.lower()]
                
                for i in range(min(3, len(enhanced_data))):
                    record = enhanced_data[i]
                    print(f"\n       Record {i+1}:")
                    print(f"         Speed: {record.get('Physics_speed_kmh', 0):.1f} km/h")
                    print(f"         G-Force X: {record.get('Physics_g_force_x', 0):.3f}")
                    print(f"         G-Force Y: {record.get('Physics_g_force_y', 0):.3f}")
                    print(f"         Friction Circle Util: {record.get('friction_circle_utilization', 0):.3f}")
                    print(f"         Overall Tire Grip: {record.get('overall_tire_grip', 0):.3f}")
                    print(f"         Long. Weight Transfer: {record.get('longitudinal_weight_transfer', 0):.3f}")
                    print(f"         Lat. Weight Transfer: {record.get('lateral_weight_transfer', 0):.3f}")
                    print(f"         Optimal Grip Window: {record.get('optimal_grip_window', 0):.3f}")
                
                # Statistical summary
                print(f"\n[INFO] Statistical Summary of Key Features:")
                key_features = [
                    'friction_circle_utilization', 
                    'overall_tire_grip', 
                    'longitudinal_weight_transfer',
                    'lateral_weight_transfer',
                    'optimal_grip_window'
                ]
                
                for feature in key_features:
                    if feature in new_keys:
                        values = [record.get(feature, 0) for record in enhanced_data]
                        if values:
                            min_val = min(values)
                            max_val = max(values)
                            avg_val = sum(values) / len(values)
                            print(f"       {feature:<30}: Min={min_val:.3f}, Max={max_val:.3f}, Avg={avg_val:.3f}")
                
        except Exception as e:
            print(f"[ERROR] Feature extraction failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return
        
        # Demonstrate practical applications
        print(f"\n[INFO] Practical Applications of These Features:")
        print(f"       1. Real-time driver coaching:")
        print(f"          - Monitor friction circle utilization to optimize cornering")
        print(f"          - Alert when tire grip drops below optimal levels")
        print(f"          - Suggest brake/throttle adjustments based on weight transfer")
        print(f"       ")
        print(f"       2. Setup optimization:")
        print(f"          - Analyze weight transfer patterns for suspension tuning")
        print(f"          - Optimize tire pressures using grip window analysis")
        print(f"          - Balance aero settings based on friction utilization")
        print(f"       ")
        print(f"       3. Performance analysis:")
        print(f"          - Compare drivers' grip utilization efficiency")
        print(f"          - Identify optimal racing lines through grip analysis")
        print(f"          - Predict tire degradation and pit strategy")
        print(f"       ")
        print(f"       4. AI training enhancement:")
        print(f"          - Use as input features for advanced ML models")
        print(f"          - Improve lap time prediction accuracy")
        print(f"          - Enable more sophisticated driver behavior analysis")
        
        print(f"\n[SUCCESS] Tire grip feature extraction demonstration completed!")
        print(f"[INFO] Features are now ready for integration into your AI pipeline")
        
    except Exception as e:
        print(f"\n[ERROR] An error occurred during demonstration: {str(e)}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 75)


if __name__ == "__main__":
    """
    Run the tire grip feature extraction demonstration
    
    Usage:
        python scripts/demo_tire_grip_features.py
        
    Or from Docker:
        docker exec -it acla_ai_service_c python scripts/demo_tire_grip_features.py
    """
    asyncio.run(main())
