#!/usr/bin/env python3
"""
Test script using real AC Competizione telemetry data structure
"""

import requests
import json

def create_test_data():
    """Create test data based on the provided AC Competizione structure"""
    
    # Sample data points simulating a racing session
    base_data = {
        "Physics_packed_id": 3254953,
        "Physics_gas": 0.8,
        "Physics_brake": 0.0,
        "Physics_fuel": 62,
        "Physics_gear": 4,
        "Physics_rpm": 6500,
        "Physics_steer_angle": 0.1,
        "Physics_speed_kmh": 125.5,
        "Physics_velocity_x": 35.0,
        "Physics_velocity_y": -0.1,
        "Physics_velocity_z": 15.2,
        "Physics_g_force_x": 0.5,
        "Physics_g_force_y": 0.0,
        "Physics_g_force_z": 1.2,
        "Physics_wheel_slip_front_left": 0.017,
        "Physics_wheel_slip_front_right": 0.011,
        "Physics_wheel_slip_rear_left": 0.009,
        "Physics_wheel_slip_rear_right": 0.006,
        "Physics_wheel_pressure_front_left": 24.6,
        "Physics_wheel_pressure_front_right": 25.7,
        "Physics_wheel_pressure_rear_left": 24.4,
        "Physics_wheel_pressure_rear_right": 25.6,
        "Physics_tyre_core_temp_front_left": 85.4,
        "Physics_tyre_core_temp_front_right": 83.1,
        "Physics_tyre_core_temp_rear_left": 88.7,
        "Physics_tyre_core_temp_rear_right": 90.0,
        "Physics_suspension_travel_front_left": 0.017,
        "Physics_suspension_travel_front_right": 0.017,
        "Physics_suspension_travel_rear_left": 0.017,
        "Physics_suspension_travel_rear_right": 0.017,
        "Physics_tc": 3,
        "Physics_heading": -0.91,
        "Physics_pitch": -0.03,
        "Physics_roll": 0.03,
        "Physics_car_damage_front": 0,
        "Physics_car_damage_rear": 0,
        "Physics_car_damage_left": 0,
        "Physics_car_damage_right": 0,
        "Physics_car_damage_center": 0,
        "Physics_pit_limiter_on": False,
        "Physics_abs": 3,
        "Physics_autoshifter_on": False,
        "Physics_brake_bias": 0.77,
        "Physics_brake_temp_front_left": 133.3,
        "Physics_brake_temp_front_right": 160.7,
        "Physics_brake_temp_rear_left": 150.1,
        "Physics_brake_temp_rear_right": 150.0,
        "Physics_clutch": 0,
        "Physics_is_ai_controlled": False,
        "Physics_ignition_on": True,
        "Physics_starter_engine_on": False,
        "Physics_is_engine_running": True,
        "Graphics_packed_id": 521366,
        "Graphics_current_time_str": "1:35:250",
        "Graphics_last_time_str": "1:34:890",
        "Graphics_best_time_str": "1:33:125",
        "Graphics_completed_lap": 5,
        "Graphics_position": 1,
        "Graphics_current_time": 95250,  # milliseconds
        "Graphics_last_time": 94890,   # milliseconds - this is our lap time target
        "Graphics_best_time": 93125,   # milliseconds
        "Graphics_session_time_left": -1,
        "Graphics_distance_traveled": 7718.2,
        "Graphics_is_in_pit": False,
        "Graphics_current_sector_index": 1,
        "Graphics_last_sector_time": 31250,
        "Graphics_number_of_laps": 0,
        "Graphics_tyre_compound": "dry_compound",
        "Graphics_normalized_car_position": 0.472,
        "Graphics_active_cars": 1,
        # Simplified coordinates - will be handled by complex field processing
        "Graphics_car_coordinates": [
            {"x": 243.8, "y": -7.47, "z": 66.69}
        ],
        "Graphics_car_id": [0, 0, 0, 0, 0],  # Simplified array
        "Graphics_player_car_id": 0,
        "Graphics_penalty_time": 0,
        "Graphics_ideal_line_on": True,
        "Graphics_is_in_pit_lane": False,
        "Graphics_tc_level": 3,
        "Graphics_tc_cut_level": 4,
        "Graphics_engine_map": 7,
        "Graphics_abs_level": 3,
        "Graphics_fuel_per_lap": 2.9,
        "Graphics_rain_light": False,
        "Graphics_flashing_light": False,
        "Graphics_exhaust_temp": 148.6,
        "Graphics_mfd_tyre_pressure_front_left": 27.65,
        "Graphics_mfd_tyre_pressure_front_right": 27.65,
        "Graphics_mfd_tyre_pressure_rear_left": 27.65,
        "Graphics_mfd_tyre_pressure_rear_right": 27.65,
        "Static_sm_version": "1.9",
        "Static_ac_version": "1.7",
        "Static_number_of_session": 0,
        "Static_num_cars": 1,
        "Static_car_model": "porsche_991ii_gt3_r",
        "Static_track": "brands_hatch",
        "Static_player_name": "Giorgio",
        "Static_player_surname": "Roda",
        "Static_sector_count": 3,
        "Static_max_rpm": 9250,
        "Static_max_fuel": 120,
        "Static_penalty_enabled": False,
        "Static_aid_fuel_rate": 0,
        "Static_aid_tyre_rate": 0,
        "Static_aid_mechanical_damage": 0,
        "Static_aid_stability": 0,
        "Static_aid_auto_clutch": True,
        "Static_is_online": False,
        "Static_dry_tyres_name": "DHD2",
        "Static_wet_tyres_name": "WH"
    }
    
    # Generate variations for a realistic session
    session_data = []
    
    for i in range(20):  # 20 data points simulating telemetry samples
        data_point = base_data.copy()
        
        # Vary some key parameters to simulate real telemetry
        import random
        
        # Vary speed and related parameters
        speed_variation = random.uniform(0.9, 1.1)
        data_point["Physics_speed_kmh"] = base_data["Physics_speed_kmh"] * speed_variation
        data_point["Physics_rpm"] = int(base_data["Physics_rpm"] * random.uniform(0.8, 1.2))
        
        # Vary lap times
        if i > 0:  # First sample keeps base time
            time_variation = random.uniform(0.98, 1.05)  # ±5% variation
            data_point["Graphics_last_time"] = int(base_data["Graphics_last_time"] * time_variation)
            data_point["Graphics_current_time"] = data_point["Graphics_last_time"] + random.randint(1000, 5000)
        
        # Vary control inputs
        data_point["Physics_gas"] = random.uniform(0.0, 1.0)
        data_point["Physics_brake"] = random.uniform(0.0, 0.8) if random.random() < 0.3 else 0.0
        data_point["Physics_steer_angle"] = random.uniform(-0.5, 0.5)
        data_point["Physics_gear"] = random.choice([3, 4, 5, 6])
        
        # Vary temperatures
        temp_variation = random.uniform(0.95, 1.05)
        for temp_field in ["Physics_tyre_core_temp_front_left", "Physics_tyre_core_temp_front_right",
                          "Physics_tyre_core_temp_rear_left", "Physics_tyre_core_temp_rear_right"]:
            data_point[temp_field] = base_data[temp_field] * temp_variation
        
        # Vary position on track
        data_point["Graphics_normalized_car_position"] = (i / 20.0) % 1.0
        data_point["Graphics_distance_traveled"] = base_data["Graphics_distance_traveled"] + (i * 100)
        
        session_data.append(data_point)
    
    return session_data

def test_model_training_with_real_data():
    """Test model training with realistic AC Competizione data"""
    
    print("=== Testing Model Training with AC Competizione Data ===")
    
    # Generate test data
    session_data = create_test_data()
    print(f"Generated {len(session_data)} telemetry samples")
    
    # Test different model types
    model_types = ["lap_time_prediction", "sector_analysis", "setup_optimization"]
    
    for model_type in model_types:
        print(f"\n--- Testing {model_type} ---")
        
        payload = {
            "session_data": session_data,
            "model_type": model_type,
            "training_parameters": {
                "n_estimators": 20,  # Small for testing
                "random_state": 42
            }
        }
        
        try:
            response = requests.post(
                "http://localhost:8000/model/train",
                json=payload,
                timeout=120  # Longer timeout for training
            )
            
            print(f"   Response status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print(f"   ✅ {model_type} training successful!")
                print(f"   Model status: {result.get('status')}")
                
                # Print key metrics
                perf_metrics = result.get('performance_metrics', {})
                print(f"   Performance metrics: {perf_metrics}")
                
                # Print feature info
                features = result.get('model_metadata', {}).get('features', [])
                print(f"   Features used: {len(features)} ({features[:5]}...)")
                
            else:
                print(f"   ❌ {model_type} training failed!")
                print(f"   Error: {response.text[:500]}")
                
                # Check for the specific AttributeError we fixed
                if "'int' object has no attribute 'lower'" in response.text:
                    print("   🔥 AttributeError still present!")
                else:
                    print("   🤔 Different error occurred")
                    
        except Exception as e:
            print(f"   ❌ Request failed: {str(e)}")

def test_telemetry_upload():
    """Test the telemetry upload endpoint"""
    
    print("\n=== Testing Telemetry Upload ===")
    
    session_data = create_test_data()
    
    # Convert list to dict format for telemetry upload
    telemetry_dict = {}
    for i, record in enumerate(session_data):
        for key, value in record.items():
            if key not in telemetry_dict:
                telemetry_dict[key] = []
            telemetry_dict[key].append(value)
    
    payload = {
        "session_id": "test_session_001",
        "telemetry_data": telemetry_dict,
        "metadata": {
            "track": "brands_hatch",
            "car": "porsche_991ii_gt3_r"
        }
    }
    
    try:
        response = requests.post(
            "http://localhost:8000/telemetry/upload",
            json=payload,
            timeout=60
        )
        
        print(f"Response status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Telemetry upload successful!")
            print(f"Feature validation: {result.get('feature_validation', {})}")
            print(f"Available analyses: {result.get('available_analyses', [])}")
        else:
            print(f"❌ Telemetry upload failed: {response.text}")
            
    except Exception as e:
        print(f"❌ Telemetry upload error: {str(e)}")

if __name__ == "__main__":
    print("Testing ACLA AI Service with Real AC Competizione Data")
    print("=" * 60)
    
    # Test model training
    test_model_training_with_real_data()
    
    # Test telemetry upload
    test_telemetry_upload()
    
    print("\n" + "=" * 60)
    print("Testing completed!")
