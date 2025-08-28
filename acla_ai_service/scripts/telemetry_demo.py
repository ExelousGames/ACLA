"""
Example script demonstrating how to integrate and use the comprehensive AC Competizione telemetry features
"""

import requests
import json
import pandas as pd
from typing import Dict, List, Any

class ACLATelemetryClient:
    """Client for interacting with ACLA AI Service telemetry endpoints"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
    
    def upload_telemetry_data(self, session_id: str, telemetry_data: Dict[str, Any], 
                            metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Upload comprehensive telemetry data to AI service"""
        payload = {
            "session_id": session_id,
            "telemetry_data": telemetry_data,
            "metadata": metadata or {}
        }
        
        response = requests.post(f"{self.base_url}/telemetry/upload", json=payload)
        response.raise_for_status()
        return response.json()
    
    def analyze_telemetry(self, session_id: str, analysis_type: str = "comprehensive",
                         features: List[str] = None) -> Dict[str, Any]:
        """Perform telemetry analysis"""
        payload = {
            "session_id": session_id,
            "analysis_type": analysis_type,
            "features": features
        }
        
        response = requests.post(f"{self.base_url}/telemetry/analyze", json=payload)
        response.raise_for_status()
        return response.json()
    
    def get_available_features(self) -> Dict[str, Any]:
        """Get information about available telemetry features"""
        response = requests.get(f"{self.base_url}/telemetry/features")
        response.raise_for_status()
        return response.json()
    
    def validate_telemetry(self, telemetry_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate telemetry data structure"""
        response = requests.post(f"{self.base_url}/telemetry/validate", json=telemetry_data)
        response.raise_for_status()
        return response.json()

def create_sample_telemetry_data() -> Dict[str, List[Any]]:
    """
    Create sample telemetry data with your input features
    This demonstrates the structure expected by the AI service
    """
    
    # Your input features from the request
    features = [
        "Physics_pad_life_front_left", "Static_sector_count", "Physics_wheel_angular_s_front_left",
        "Physics_brake_pressure_rear_left", "Graphics_ideal_line_on", "Graphics_is_valid_lap",
        "Graphics_packed_id", "Graphics_delta_lap_time_str", "Graphics_mfd_tyre_pressure_rear_left",
        "Static_pit_window_start", "Physics_starter_engine_on", "Graphics_mfd_tyre_pressure_front_right",
        "Graphics_rain_light", "Graphics_current_tyre_set", "Physics_is_engine_running",
        "Physics_tyre_contact_point_rear_right_z", "Physics_tyre_contact_normal_rear_right_x",
        "Physics_slip_angle_rear_left", "Graphics_flashing_light", "Physics_tyre_core_temp_front_left",
        "Graphics_wiper_stage", "Physics_suspension_damage_rear_left", "Graphics_mfd_tyre_pressure_rear_right",
        "Physics_tyre_contact_heading_rear_left_x", "Physics_rear_brake_compound", "Static_max_rpm",
        "Physics_local_angular_vel_x", "Physics_final_ff", "Static_pit_window_end",
        "Physics_disc_life_rear_right", "Physics_tyre_core_temp_front_right", "Graphics_missing_mandatory_pits",
        "Physics_tyre_contact_normal_front_right_z", "Static_aid_auto_clutch", "Physics_g_vibration",
        "Physics_brake_bias", "Physics_tyre_contact_point_front_right_x", "Static_track",
        "Physics_pad_life_front_right", "Graphics_best_time_str", "Graphics_player_car_id",
        "Physics_local_velocity_x", "Physics_brake_temp_rear_left", "Physics_tyre_contact_point_rear_left_y",
        "Physics_heading", "Physics_tyre_contact_heading_rear_right_z", "Graphics_is_delta_positive",
        "Graphics_mfd_fuel_to_add", "Static_number_of_session", "Static_aid_stability", "Physics_fuel",
        "Graphics_driver_stint_total_time_left", "Physics_tyre_contact_heading_front_left_z",
        "Graphics_tyre_compound", "Physics_slip_vibration", "Physics_disc_life_front_left",
        "Physics_suspension_travel_front_right", "Physics_disc_life_rear_left", "Graphics_session_index",
        "Graphics_driver_stint_time_left", "Physics_slip_angle_front_right", "Graphics_global_green",
        "Physics_g_force_x", "Static_max_fuel", "Physics_rpm", "Physics_g_force_z", "Physics_car_damage_rear",
        "Physics_slip_ratio_front_left", "Physics_tyre_contact_heading_front_left_y", "Static_ac_version",
        "Static_num_cars", "Physics_tyre_contact_point_rear_right_y", "Graphics_is_in_pit_lane",
        "Physics_velocity_x", "Graphics_global_chequered", "Graphics_global_red", "Static_aid_tyre_rate",
        "Physics_tc", "Physics_wheel_pressure_front_right", "Physics_suspension_travel_front_left",
        "Physics_tyre_contact_heading_rear_right_y", "Physics_clutch", "Graphics_current_sector_index",
        "Graphics_direction_light_right", "Physics_road_temp", "Physics_wheel_pressure_front_left",
        "Physics_local_velocity_z", "Graphics_gap_ahead", "Graphics_global_white",
        "Physics_wheel_angular_s_rear_right", "Physics_brake_temp_front_right",
        "Physics_tyre_contact_point_rear_left_x", "Graphics_last_time", "Physics_tyre_contact_heading_front_left_x",
        "Graphics_clock", "Graphics_last_time_str", "Physics_air_temp", "Graphics_wind_direction",
        "Graphics_gap_behind", "Physics_g_force_y", "Physics_autoshifter_on", "Graphics_abs_level",
        "Graphics_delta_lap_time", "Physics_brake_temp_rear_right", "Physics_abs_vibration", "Physics_gear",
        "Physics_wheel_pressure_rear_right", "Graphics_used_fuel", "Graphics_global_yellow_s3",
        "Physics_tyre_contact_point_rear_left_z", "Graphics_car_coordinates", "Graphics_mfd_tyre_set",
        "Physics_tyre_contact_heading_front_right_y", "Physics_suspension_travel_rear_right",
        "Physics_local_angular_vel_z", "Physics_tyre_contact_point_front_left_z",
        "Graphics_normalized_car_position", "Physics_brake_pressure_rear_right", "Physics_kerb_vibration",
        "Graphics_wind_speed", "Physics_tyre_contact_heading_rear_right_x", "Static_sm_version",
        "Physics_tyre_contact_heading_front_right_z", "Physics_tyre_contact_heading_rear_left_z",
        "Graphics_current_time_str", "Physics_wheel_slip_rear_left", "Graphics_last_sector_time_str",
        "Graphics_mfd_tyre_pressure_front_left", "Graphics_penalty_time", "Physics_slip_ratio_front_right",
        "Graphics_mandatory_pit_done", "Graphics_tc_level", "Physics_tyre_contact_point_front_right_y",
        "Graphics_strategy_tyre_set", "Physics_steer_angle", "Physics_is_ai_controlled",
        "Physics_car_damage_left", "Static_aid_fuel_rate", "Physics_wheel_pressure_rear_left",
        "Physics_wheel_angular_s_rear_left", "Physics_pad_life_rear_right", "Physics_ignition_on",
        "Physics_car_damage_right", "Physics_tyre_contact_normal_rear_right_z", "Physics_velocity_z",
        "Physics_wheel_slip_rear_right", "Graphics_last_sector_time", "Physics_tyre_contact_point_front_left_y",
        "Physics_tyre_core_temp_rear_left", "Physics_tyre_contact_point_front_right_z", "Physics_brake",
        "Physics_gas", "Physics_speed_kmh", "Static_player_name", "Static_penalty_enabled",
        "Static_dry_tyres_name", "Graphics_fuel_estimated_laps", "Physics_slip_angle_front_left",
        "Physics_slip_ratio_rear_right", "Graphics_direction_light_left", "Graphics_session_time_left",
        "Physics_brake_pressure_front_right", "Physics_abs", "Graphics_fuel_per_lap", "Graphics_track_status",
        "Physics_pitch", "Physics_tyre_contact_normal_rear_left_z", "Physics_roll", "Graphics_number_of_laps",
        "Physics_tyre_contact_normal_rear_left_x", "Graphics_is_setup_menu_visible", "Physics_pad_life_rear_left",
        "Graphics_position", "Static_player_surname", "Physics_tyre_contact_normal_front_right_y",
        "Graphics_rain_tyres", "Physics_local_angular_vel_y", "Graphics_global_yellow_s2",
        "Physics_tyre_contact_normal_front_left_x", "Graphics_car_id", "Graphics_best_time",
        "Graphics_is_in_pit", "Physics_suspension_travel_rear_left", "Physics_brake_temp_front_left",
        "Physics_slip_angle_rear_right", "Graphics_exhaust_temp", "Physics_slip_ratio_rear_left",
        "Physics_wheel_slip_front_right", "Physics_tyre_contact_heading_front_right_x",
        "Physics_suspension_damage_rear_right", "Graphics_estimated_lap_time", "Physics_tyre_core_temp_rear_right",
        "Graphics_secondary_display_index", "Physics_tyre_contact_normal_rear_right_y",
        "Physics_tyre_contact_heading_rear_left_y", "Physics_disc_life_front_right",
        "Physics_wheel_angular_s_front_right", "Physics_tyre_contact_point_front_left_x",
        "Physics_tyre_contact_normal_front_right_x", "Graphics_global_yellow_s1", "Graphics_completed_lap",
        "Static_is_online", "Physics_car_damage_front", "Graphics_distance_traveled", "Physics_turbo_boost",
        "Graphics_main_display_index", "Physics_local_velocity_y", "Static_car_model", "Physics_water_temp",
        "Graphics_light_stage", "Physics_tyre_contact_normal_front_left_z", "Physics_car_damage_center",
        "Physics_suspension_damage_front_left", "Physics_velocity_y", "Graphics_global_yellow",
        "Graphics_engine_map", "Physics_pit_limiter_on", "Physics_tyre_contact_normal_front_left_y",
        "Physics_packed_id", "Static_aid_mechanical_damage", "Physics_wheel_slip_front_left",
        "Physics_front_brake_compound", "Static_wet_tyres_name", "Graphics_active_cars",
        "Graphics_tc_cut_level", "Graphics_estimated_lap_time_str", "Physics_suspension_damage_front_right",
        "Physics_brake_pressure_front_left", "Physics_tyre_contact_point_rear_right_x",
        "Physics_tyre_contact_normal_rear_left_y", "Graphics_current_time"
    ]
    
    # Generate sample data (in real usage, this would come from AC Competizione)
    import random
    import numpy as np
    
    num_samples = 1000  # Simulate 1000 telemetry samples
    
    sample_data = {}
    
    for feature in features:
        if "Physics_speed_kmh" in feature:
            # Speed data: 0-300 km/h
            sample_data[feature] = [random.uniform(0, 300) for _ in range(num_samples)]
        elif "Physics_rpm" in feature:
            # RPM data: 1000-8000
            sample_data[feature] = [random.uniform(1000, 8000) for _ in range(num_samples)]
        elif "Physics_gear" in feature:
            # Gear data: 0-6
            sample_data[feature] = [random.randint(0, 6) for _ in range(num_samples)]
        elif "Physics_brake" in feature or "Physics_gas" in feature:
            # Brake/Gas: 0-1
            sample_data[feature] = [random.uniform(0, 1) for _ in range(num_samples)]
        elif "temp" in feature.lower():
            # Temperature data: 20-120°C
            sample_data[feature] = [random.uniform(20, 120) for _ in range(num_samples)]
        elif "pressure" in feature.lower():
            # Pressure data: 20-40 PSI
            sample_data[feature] = [random.uniform(20, 40) for _ in range(num_samples)]
        elif "Physics_g_force" in feature:
            # G-force data: -3 to 3
            sample_data[feature] = [random.uniform(-3, 3) for _ in range(num_samples)]
        elif any(word in feature.lower() for word in ["on", "enabled", "valid", "running"]):
            # Boolean features
            sample_data[feature] = [random.choice([True, False]) for _ in range(num_samples)]
        elif "Graphics_last_time" in feature:
            # Lap times in seconds (80-120 seconds)
            sample_data[feature] = [random.uniform(80, 120) for _ in range(num_samples)]
        elif "Graphics_position" in feature:
            # Position: 1-20
            sample_data[feature] = [random.randint(1, 20) for _ in range(num_samples)]
        else:
            # Generic numeric data
            sample_data[feature] = [random.uniform(0, 100) for _ in range(num_samples)]
    
    return sample_data

def main():
    """Demonstrate telemetry integration usage"""
    
    # Initialize client
    client = ACLATelemetryClient()
    
    print("🏁 ACLA Telemetry Integration Demo")
    print("=" * 50)
    
    try:
        # 1. Get available features
        print("\n1. Getting available telemetry features...")
        features_info = client.get_available_features()
        print(f"   Total features supported: {features_info['total_features']}")
        print(f"   Physics features: {features_info['feature_categories']['physics']['count']}")
        print(f"   Graphics features: {features_info['feature_categories']['graphics']['count']}")
        print(f"   Static features: {features_info['feature_categories']['static']['count']}")
        
        # 2. Create sample telemetry data
        print("\n2. Creating sample telemetry data...")
        telemetry_data = create_sample_telemetry_data()
        print(f"   Generated data with {len(telemetry_data)} features and {len(list(telemetry_data.values())[0])} samples")
        
        # 3. Validate telemetry data
        print("\n3. Validating telemetry data...")
        validation_result = client.validate_telemetry(telemetry_data)
        print(f"   Feature coverage: {validation_result['validation_result']['coverage_percentage']:.1f}%")
        print(f"   Data quality: {validation_result['data_quality']['total_records']} records, {validation_result['data_quality']['total_columns']} columns")
        
        # 4. Upload telemetry data
        print("\n4. Uploading telemetry data...")
        session_id = "demo_session_2024"
        upload_result = client.upload_telemetry_data(
            session_id=session_id,
            telemetry_data=telemetry_data,
            metadata={
                "track": "spa_francorchamps",
                "car": "mercedes_amg_gt3",
                "conditions": "dry",
                "session_type": "practice"
            }
        )
        print(f"   Upload successful! Session ID: {upload_result['session_id']}")
        print(f"   Available analyses: {', '.join(upload_result['available_analyses'])}")
        
        # 5. Perform comprehensive analysis
        print("\n5. Performing comprehensive telemetry analysis...")
        analysis_result = client.analyze_telemetry(session_id, "comprehensive")
        
        if "result" in analysis_result:
            result = analysis_result["result"]
            
            # Display performance score
            if "performance_score" in result:
                score = result["performance_score"]
                print(f"   Overall Performance Score: {score['overall_score']:.1f}% (Grade: {score['grade']})")
                print(f"   Analysis Confidence: {score['analysis_confidence']}")
                
                print("\n   Score Components:")
                for component, value in score["components"].items():
                    print(f"     • {component.replace('_', ' ').title()}: {value}")
                
                print("\n   Recommendations:")
                for rec in score["recommendations"]:
                    print(f"     • {rec}")
            
            # Display telemetry summary
            if "telemetry_summary" in result:
                tel_summary = result["telemetry_summary"]
                feature_val = tel_summary["feature_validation"]
                print(f"\n   Telemetry Coverage: {feature_val['coverage_percentage']:.1f}%")
                print(f"   Physics: {feature_val['physics_coverage']}, Graphics: {feature_val['graphics_coverage']}, Static: {feature_val['static_coverage']}")
        
        # 6. Perform specific analyses
        print("\n6. Performing specific analyses...")
        
        # Performance analysis
        perf_result = client.analyze_telemetry(session_id, "performance")
        print("   ✓ Performance analysis completed")
        
        # Setup analysis
        setup_result = client.analyze_telemetry(session_id, "setup")
        print("   ✓ Setup analysis completed")
        
        print("\n🎉 Telemetry integration demo completed successfully!")
        print("\nNext steps:")
        print("1. Replace sample data with real AC Competizione telemetry")
        print("2. Integrate with your data collection system")
        print("3. Set up real-time analysis for live sessions")
        print("4. Customize analysis parameters for your specific needs")
        
    except requests.exceptions.ConnectionError:
        print("❌ Error: Could not connect to ACLA AI Service")
        print("   Make sure the service is running on http://localhost:8000")
    except Exception as e:
        print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    main()
