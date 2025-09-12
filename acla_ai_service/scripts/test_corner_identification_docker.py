#!/usr/bin/env python3
"""
Docker-compatible test script for Corner Identification Unsupervised Service

Run this script inside the Docker container:
docker exec -it acla_ai_service_c python scripts/test_corner_identification_docker.py
"""

import asyncio
import sys
import os
from pathlib import Path

# The working directory in Docker is /app, and we need to add the app subdirectory to path
sys.path.insert(0, '/app/app')  # Add the app directory to Python path

# Now import the service
try:
    from services.corner_identification_unsupervised_service import CornerIdentificationUnsupervisedService
    print("✅ Successfully imported CornerIdentificationUnsupervisedService")
except ImportError as e:
    print(f"❌ Failed to import CornerIdentificationUnsupervisedService: {e}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Python paths: {sys.path[:5]}")
    sys.exit(1)


async def test_corner_identification_docker():
    """Test the corner identification service in Docker environment"""
    
    print("=== Corner Identification Service Test (Docker) ===\n")
    
    # Initialize the service
    try:
        # Create models directory inside container
        models_dir = "/tmp/corner_models"
        corner_service = CornerIdentificationUnsupervisedService(models_dir)
        print(f"✅ Service initialized with models directory: {models_dir}")
    except Exception as e:
        print(f"❌ Failed to initialize service: {e}")
        return
    
    # Test parameters - use tracks that might exist in the database
    test_configs = [
        {"track_name": "monza", "car_name": "porsche_991ii_gt3_r"},
        {"track_name": "spa", "car_name": None},  # Any car
        {"track_name": "silverstone", "car_name": "bmw_m4_gt3"}
    ]
    
    for config in test_configs:
        track_name = config["track_name"]
        car_name = config["car_name"]
        
        print(f"\n--- Testing Track: {track_name}, Car: {car_name or 'Any'} ---")
        
        # Test 1: Learn corner patterns
        print("1. Attempting to learn corner patterns...")
        try:
            corner_results = await corner_service.learn_track_corner_patterns(
                trackName=track_name,
                carName=car_name
            )
            
            if corner_results.get("success"):
                print(f"✅ Successfully learned corner patterns!")
                print(f"   - Total corners identified: {corner_results.get('total_corners_identified', 0)}")
                
                corner_patterns = corner_results.get("corner_patterns", [])
                if corner_patterns:
                    print(f"   - First corner example:")
                    first_corner = corner_patterns[0]["characteristics"]
                    print(f"     * Type: {first_corner.get('corner_type', 'unknown')}")
                    print(f"     * Duration: {first_corner.get('total_corner_duration', 0):.1f}")
                    print(f"     * Max steering: {first_corner.get('apex_max_steering', 0):.3f}")
                
                # Test successful case - break after first success
                break
                
            else:
                print(f"ℹ️  No data found for this track/car combination")
                print(f"   Error: {corner_results.get('error', 'Unknown')}")
                
        except Exception as e:
            print(f"❌ Error during corner pattern learning: {str(e)}")
            continue
    
    # Test 2: Feature extraction with synthetic data
    print(f"\n--- Testing Feature Extraction ---")
    try:
        # Create synthetic telemetry data that simulates a corner
        synthetic_telemetry = []
        
        # Simulate approaching a corner
        for i in range(50):
            progress = i / 49.0  # 0 to 1
            
            # Simulate corner entry, apex, exit
            if progress < 0.3:  # Entry
                steering = progress * 0.8  # Gradually increase steering
                speed = 200 - progress * 80  # Slow down
                brake = progress * 0.6
                throttle = 1.0 - progress * 1.0
            elif progress < 0.7:  # Apex
                steering = 0.8 + (progress - 0.3) * 0.4  # Max steering
                speed = 120 + (progress - 0.3) * 0  # Constant slow speed
                brake = 0.6 - (progress - 0.3) * 1.5  # Release brake
                throttle = 0.0
            else:  # Exit
                steering = 1.2 - (progress - 0.7) * 4.0  # Unwind steering
                speed = 120 + (progress - 0.7) * 200  # Accelerate
                brake = 0.0
                throttle = (progress - 0.7) * 3.33  # Apply throttle
            
            # Clamp values
            steering = max(0, min(1.5, steering))
            speed = max(50, min(250, speed))
            brake = max(0, min(1.0, brake))
            throttle = max(0, min(1.0, throttle))
            
            telemetry_point = {
                "Physics_steer_angle": steering,
                "Physics_speed_kmh": speed,
                "Physics_brake": brake,
                "Physics_gas": throttle,
                "Physics_g_force_x": -steering * 2.0,  # Lateral G
                "Physics_g_force_z": brake * 1.2 - throttle * 0.8,  # Longitudinal G
                "Graphics_normalized_car_position": progress,
                "timestamp": i
            }
            synthetic_telemetry.append(telemetry_point)
        
        print(f"Created {len(synthetic_telemetry)} synthetic telemetry points")
        
        # Test feature extraction
        enhanced_telemetry = await corner_service.extract_corner_features_for_telemetry(
            synthetic_telemetry, "test_track", "test_car"
        )
        
        print(f"✅ Enhanced {len(enhanced_telemetry)} telemetry records")
        
        # Check if corner features were added
        if enhanced_telemetry:
            sample_record = enhanced_telemetry[25]  # Middle of the corner
            corner_features = [k for k in sample_record.keys() if k.startswith('corner_')]
            
            print(f"   - Added {len(corner_features)} corner features per record")
            
            # Show some key features
            key_features = [
                'is_in_corner', 'corner_id', 'corner_phase', 'corner_type_numeric',
                'corner_apex_max_steering', 'corner_speed_efficiency'
            ]
            
            print("   - Key corner features for middle record:")
            for feature in key_features:
                if feature in sample_record:
                    value = sample_record[feature]
                    print(f"     * {feature}: {value}")
    
    except Exception as e:
        print(f"❌ Error during feature extraction test: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # Test 3: Service methods
    print(f"\n--- Testing Service Methods ---")
    try:
        # Test summary method
        summary = corner_service.get_corner_identification_summary("test_track", "test_car")
        print(f"✅ Summary method works: {summary.get('success', False)}")
        
        # Test cache clearing
        corner_service.clear_corner_cache("test_track", "test_car")
        print("✅ Cache clearing method works")
        
    except Exception as e:
        print(f"❌ Error testing service methods: {str(e)}")
    
    print("\n=== Docker Test Complete ===")


if __name__ == "__main__":
    print("Starting Corner Identification Service Test in Docker...")
    print(f"Python path: {sys.path[:3]}...")  # Show first 3 entries
    print(f"Current working directory: {os.getcwd()}")
    
    try:
        asyncio.run(test_corner_identification_docker())
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
