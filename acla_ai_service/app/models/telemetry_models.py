"""
Telemetry data models for Assetto Corsa Competizione telemetry features
"""

from pydantic import BaseModel
from typing import Dict, List, Any, Optional
import pandas as pd
import math
import numpy as np
import json
import ast

def _safe_float(value):
    """Convert value to float, handling NaN and infinity"""
    try:
        float_val = float(value)
        if math.isnan(float_val) or math.isinf(float_val):
            return 0.0
        return float_val
    except (ValueError, TypeError):
        return 0.0

def _parse_car_coordinates(value):
    """
    Safely parse car coordinates which might be stored as:
    - Actual Python list
    - String representation of a list  
    - JSON string
    - Numpy array
    """
    if isinstance(value, list):
        return value
    
    # Handle numpy arrays
    if isinstance(value, np.ndarray):
        try:
            # Convert numpy array to list and return
            return value.tolist()
        except Exception:
            pass
    
    if isinstance(value, str):
        # Remove any extra whitespace and newlines
        value = value.strip()
        
        try:
            # Try JSON parsing first
            return json.loads(value)
        except (json.JSONDecodeError, ValueError):
            pass
            
        try:
            # Try ast.literal_eval for Python literal strings
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            pass
        
        try:
            # Handle case where it's formatted like the debug output
            # Replace the formatted string with proper JSON
            if 'x\': ' in value and 'y\': ' in value and 'z\': ' in value:
                # This looks like the formatted output, try to extract coordinates
                import re
                coord_pattern = r"{'x': ([-\d.]+), 'y': ([-\d.]+), 'z': ([-\d.]+)}"
                matches = re.findall(coord_pattern, value)
                if matches:
                    coords = []
                    for match in matches:
                        coords.append({
                            'x': float(match[0]),
                            'y': float(match[1]), 
                            'z': float(match[2])
                        })
                    return coords
        except Exception:
            pass
    
    return None

    

class TelemetryFeatures:
    """
    Complete set of AC Competizione telemetry features for AI analysis
    """
    
    # Physics telemetry features
    PHYSICS_FEATURES = [
        "Physics_pad_life_front_left",
        "Physics_wheel_angular_s_front_left", 
        "Physics_brake_pressure_rear_left",
        "Physics_starter_engine_on",
        "Physics_is_engine_running",
        "Physics_tyre_contact_point_rear_right_z",
        "Physics_tyre_contact_normal_rear_right_x",
        "Physics_slip_angle_rear_left",
        "Physics_tyre_core_temp_front_left",
        "Physics_suspension_damage_rear_left",
        "Physics_tyre_contact_heading_rear_left_x",
        "Physics_rear_brake_compound",
        "Physics_local_angular_vel_x",
        "Physics_final_ff",
        "Physics_disc_life_rear_right",
        "Physics_tyre_core_temp_front_right",
        "Physics_tyre_contact_normal_front_right_z",
        "Physics_g_vibration",
        "Physics_brake_bias",
        "Physics_tyre_contact_point_front_right_x",
        "Physics_pad_life_front_right",
        "Physics_local_velocity_x",
        "Physics_brake_temp_rear_left",
        "Physics_tyre_contact_point_rear_left_y",
        "Physics_heading",
        "Physics_tyre_contact_heading_rear_right_z",
        "Physics_fuel",
        "Physics_tyre_contact_heading_front_left_z",
        "Physics_slip_vibration",
        "Physics_disc_life_front_left",
        "Physics_suspension_travel_front_right",
        "Physics_disc_life_rear_left",
        "Physics_slip_angle_front_right",
        "Physics_g_force_x",
        "Physics_rpm",
        "Physics_g_force_z",
        "Physics_car_damage_rear",
        "Physics_slip_ratio_front_left",
        "Physics_tyre_contact_heading_front_left_y",
        "Physics_tyre_contact_point_rear_right_y",
        "Physics_velocity_x",
        "Physics_tc",
        "Physics_wheel_pressure_front_right",
        "Physics_suspension_travel_front_left",
        "Physics_tyre_contact_heading_rear_right_y",
        "Physics_clutch",
        "Physics_road_temp",
        "Physics_wheel_pressure_front_left",
        "Physics_local_velocity_z",
        "Physics_wheel_angular_s_rear_right",
        "Physics_brake_temp_front_right",
        "Physics_tyre_contact_point_rear_left_x",
        "Physics_tyre_contact_heading_front_left_x",
        "Physics_air_temp",
        "Physics_g_force_y",
        "Physics_autoshifter_on",
        "Physics_brake_temp_rear_right",
        "Physics_abs_vibration",
        "Physics_gear",
        "Physics_wheel_pressure_rear_right",
        "Physics_tyre_contact_point_rear_left_z",
        "Physics_tyre_contact_heading_front_right_y",
        "Physics_suspension_travel_rear_right",
        "Physics_local_angular_vel_z",
        "Physics_tyre_contact_point_front_left_z",
        "Physics_brake_pressure_rear_right",
        "Physics_kerb_vibration",
        "Physics_tyre_contact_heading_rear_right_x",
        "Physics_tyre_contact_heading_front_right_z",
        "Physics_tyre_contact_heading_rear_left_z",
        "Physics_wheel_slip_rear_left",
        "Physics_slip_ratio_front_right",
        "Physics_tyre_contact_point_front_right_y",
        "Physics_steer_angle", #absolute value from -1.0 to 1.0.
        "Physics_is_ai_controlled",
        "Physics_car_damage_left",
        "Physics_wheel_pressure_rear_left",
        "Physics_wheel_angular_s_rear_left",
        "Physics_pad_life_rear_right",
        "Physics_ignition_on",
        "Physics_car_damage_right",
        "Physics_tyre_contact_normal_rear_right_z",
        "Physics_velocity_z",
        "Physics_wheel_slip_rear_right",
        "Physics_tyre_contact_point_front_left_y",
        "Physics_tyre_core_temp_rear_left",
        "Physics_tyre_contact_point_front_right_z",
        "Physics_brake",
        "Physics_gas",
        "Physics_speed_kmh",
        "Physics_slip_angle_front_left",
        "Physics_slip_ratio_rear_right",
        "Physics_brake_pressure_front_right",
        "Physics_abs",
        "Physics_pitch",
        "Physics_tyre_contact_normal_rear_left_z",
        "Physics_roll",
        "Physics_tyre_contact_normal_rear_left_x",
        "Physics_pad_life_rear_left",
        "Physics_tyre_contact_normal_front_right_y",
        "Physics_local_angular_vel_y",
        "Physics_tyre_contact_normal_front_left_x",
        "Physics_suspension_travel_rear_left",
        "Physics_brake_temp_front_left",
        "Physics_slip_angle_rear_right",
        "Physics_slip_ratio_rear_left",
        "Physics_wheel_slip_front_right",
        "Physics_tyre_contact_heading_front_right_x",
        "Physics_suspension_damage_rear_right",
        "Physics_tyre_core_temp_rear_right",
        "Physics_tyre_contact_normal_rear_right_y",
        "Physics_tyre_contact_heading_rear_left_y",
        "Physics_disc_life_front_right",
        "Physics_wheel_angular_s_front_right",
        "Physics_tyre_contact_point_front_left_x",
        "Physics_tyre_contact_normal_front_right_x",
        "Physics_car_damage_front",
        "Physics_turbo_boost",
        "Physics_local_velocity_y",
        "Physics_water_temp",
        "Physics_tyre_contact_normal_front_left_z",
        "Physics_car_damage_center",
        "Physics_suspension_damage_front_left",
        "Physics_velocity_y",
        "Physics_pit_limiter_on",
        "Physics_tyre_contact_normal_front_left_y",
        "Physics_packed_id",
        "Physics_wheel_slip_front_left",
        "Physics_front_brake_compound",
        "Physics_suspension_damage_front_right",
        "Physics_brake_pressure_front_left",
        "Physics_tyre_contact_point_rear_right_x",
        "Physics_tyre_contact_normal_rear_left_y"
    ]
    
    # Graphics telemetry features  
    GRAPHICS_FEATURES = [
        "Graphics_ideal_line_on",
        "Graphics_is_valid_lap",
        "Graphics_packed_id",
        "Graphics_delta_lap_time_str",
        "Graphics_mfd_tyre_pressure_rear_left",
        "Graphics_mfd_tyre_pressure_front_right",
        "Graphics_rain_light",
        "Graphics_current_tyre_set",
        "Graphics_flashing_light",
        "Graphics_wiper_stage",
        "Graphics_mfd_tyre_pressure_rear_right",
        "Graphics_missing_mandatory_pits",
        "Graphics_best_time_str",
        "Graphics_player_car_id",
        "Graphics_is_delta_positive",
        "Graphics_mfd_fuel_to_add",
        "Graphics_driver_stint_total_time_left",
        "Graphics_tyre_compound",
        "Graphics_session_index",
        "Graphics_driver_stint_time_left",
        "Graphics_global_green",
        "Graphics_is_in_pit_lane",
        "Graphics_global_chequered",
        "Graphics_global_red",
        "Graphics_current_sector_index",
        "Graphics_direction_light_right",
        "Graphics_gap_ahead",
        "Graphics_global_white",
        "Graphics_last_time",
        "Graphics_clock",
        "Graphics_last_time_str",
        "Graphics_wind_direction",
        "Graphics_gap_behind",
        "Graphics_abs_level",
        "Graphics_delta_lap_time",
        "Graphics_used_fuel",
        "Graphics_global_yellow_s3",
        "Graphics_car_coordinates",
        "Graphics_mfd_tyre_set",
        "Graphics_normalized_car_position",
        "Graphics_wind_speed",
        "Graphics_current_time_str",
        "Graphics_last_sector_time_str",
        "Graphics_mfd_tyre_pressure_front_left",
        "Graphics_penalty_time",
        "Graphics_mandatory_pit_done",
        "Graphics_tc_level",
        "Graphics_strategy_tyre_set",
        "Graphics_last_sector_time",
        "Graphics_fuel_estimated_laps",
        "Graphics_direction_light_left",
        "Graphics_session_time_left",
        "Graphics_fuel_per_lap",
        "Graphics_track_status",
        "Graphics_number_of_laps",
        "Graphics_is_setup_menu_visible",
        "Graphics_position",
        "Graphics_rain_tyres",
        "Graphics_global_yellow_s2",
        "Graphics_car_id",
        "Graphics_best_time",
        "Graphics_is_in_pit",
        "Graphics_exhaust_temp",
        "Graphics_estimated_lap_time",
        "Graphics_secondary_display_index",
        "Graphics_global_yellow_s1",
        "Graphics_completed_lap",
        "Graphics_distance_traveled",
        "Graphics_main_display_index",
        "Graphics_light_stage",
        "Graphics_global_yellow",
        "Graphics_engine_map",
        "Graphics_active_cars",
        "Graphics_tc_cut_level",
        "Graphics_estimated_lap_time_str",
        "Graphics_current_time"
    ]
    
    # Static telemetry features
    STATIC_FEATURES = [
        "Static_sector_count",
        "Static_pit_window_start",
        "Static_max_rpm",
        "Static_pit_window_end",
        "Static_aid_auto_clutch",
        "Static_track",
        "Static_number_of_session",
        "Static_aid_stability",
        "Static_max_fuel",
        "Static_ac_version",
        "Static_num_cars",
        "Static_aid_tyre_rate",
        "Static_sm_version",
        "Static_player_name",
        "Static_penalty_enabled",
        "Static_dry_tyres_name",
        "Static_player_surname",
        "Static_is_online",
        "Static_car_model",
        "Static_aid_mechanical_damage",
        "Static_wet_tyres_name",
        "Static_aid_fuel_rate"
    ]
    
    @classmethod
    def get_all_features(cls) -> List[str]:
        """Get all telemetry features combined"""
        return cls.PHYSICS_FEATURES + cls.GRAPHICS_FEATURES + cls.STATIC_FEATURES
    
    @classmethod
    def get_performance_critical_features(cls) -> List[str]:
        """Get features most critical for performance analysis"""
        return [
            "Physics_speed_kmh",
            "Physics_gear",
            "Physics_rpm", 
            "Physics_brake",
            "Physics_gas",
            "Physics_steer_angle",
            "Physics_slip_angle_front_left",
            "Physics_slip_angle_front_right", 
            "Physics_slip_angle_rear_left",
            "Physics_slip_angle_rear_right",
            "Physics_tyre_core_temp_front_left",
            "Physics_tyre_core_temp_front_right",
            "Physics_tyre_core_temp_rear_left", 
            "Physics_tyre_core_temp_rear_right",
            "Physics_brake_temp_front_left",
            "Physics_brake_temp_front_right",
            "Physics_brake_temp_rear_left",
            "Physics_brake_temp_rear_right",
            "Physics_g_force_x",
            "Physics_g_force_y",
            "Physics_g_force_z",
            "Graphics_delta_lap_time",
            "Graphics_last_time",
            "Graphics_best_time",
            "Graphics_current_sector_index",
            "Graphics_position",
            "Graphics_last_time",
            "Graphics_best_time",
            "Graphics_current_time"
        ]
    
    @classmethod
    def get_setup_features(cls) -> List[str]:
        """Get features related to car setup"""
        return [
            "Physics_brake_bias",
            "Physics_tc",
            "Physics_abs",
            "Physics_final_ff",
            "Graphics_tc_level",
            "Graphics_abs_level",
            "Graphics_engine_map",
            "Physics_front_brake_compound",
            "Physics_rear_brake_compound",
            "Graphics_tyre_compound"
        ]
    
    @classmethod
    def get_damage_features(cls) -> List[str]:
        """Get features related to car damage"""
        return [
            "Physics_car_damage_front",
            "Physics_car_damage_rear", 
            "Physics_car_damage_left",
            "Physics_car_damage_right",
            "Physics_car_damage_center",
            "Physics_suspension_damage_front_left",
            "Physics_suspension_damage_front_right",
            "Physics_suspension_damage_rear_left",
            "Physics_suspension_damage_rear_right"
        ]
    
    @classmethod
    def get_fuel_consumption_features(cls) -> List[str]:
        """Get features related to fuel consumption and engine performance"""
        return [
            "Physics_fuel",
            "Physics_gas",
            "Physics_rpm",
            "Physics_speed_kmh",
            "Physics_gear",
            "Physics_turbo_boost",
            "Physics_water_temp",
            "Graphics_fuel_per_lap",
            "Graphics_fuel_estimated_laps",
            "Graphics_used_fuel",
            "Graphics_engine_map",
            "Graphics_tc_level",
            "Graphics_exhaust_temp"
        ]
    
    @classmethod
    def get_brake_performance_features(cls) -> List[str]:
        """Get features related to brake performance and analysis"""
        return [
            "Physics_brake",
            "Physics_brake_pressure_front_left",
            "Physics_brake_pressure_front_right",
            "Physics_brake_pressure_rear_left",
            "Physics_brake_pressure_rear_right",
            "Physics_brake_temp_front_left",
            "Physics_brake_temp_front_right",
            "Physics_brake_temp_rear_left",
            "Physics_brake_temp_rear_right",
            "Physics_disc_life_front_left",
            "Physics_disc_life_front_right",
            "Physics_disc_life_rear_left",
            "Physics_disc_life_rear_right",
            "Physics_pad_life_front_left",
            "Physics_pad_life_front_right",
            "Physics_pad_life_rear_left",
            "Physics_pad_life_rear_right",
            "Physics_brake_bias",
            "Physics_abs",
            "Physics_abs_vibration",
            "Graphics_abs_level",
            "Physics_speed_kmh",
            "Physics_g_force_x",
            "Physics_g_force_y",
            "Physics_g_force_z"
        ]
    
    @classmethod
    def get_tire_strategy_features(cls) -> List[str]:
        """Get features related to tire strategy and performance"""
        return [
            "Physics_tyre_core_temp_front_left",
            "Physics_tyre_core_temp_front_right",
            "Physics_tyre_core_temp_rear_left",
            "Physics_tyre_core_temp_rear_right",
            "Physics_wheel_pressure_front_left",
            "Physics_wheel_pressure_front_right",
            "Physics_wheel_pressure_rear_left",
            "Physics_wheel_pressure_rear_right",
            "Physics_slip_angle_front_left",
            "Physics_slip_angle_front_right",
            "Physics_slip_angle_rear_left",
            "Physics_slip_angle_rear_right",
            "Physics_slip_ratio_front_left",
            "Physics_slip_ratio_front_right",
            "Physics_slip_ratio_rear_left",
            "Physics_slip_ratio_rear_right",
            "Physics_wheel_slip_front_left",
            "Physics_wheel_slip_front_right",
            "Physics_wheel_slip_rear_left",
            "Physics_wheel_slip_rear_right",
            "Graphics_tyre_compound",
            "Graphics_current_tyre_set",
            "Graphics_mfd_tyre_set",
            "Graphics_strategy_tyre_set",
            "Graphics_mfd_tyre_pressure_front_left",
            "Graphics_mfd_tyre_pressure_front_right",
            "Graphics_mfd_tyre_pressure_rear_left",
            "Graphics_mfd_tyre_pressure_rear_right",
            "Graphics_rain_tyres",
            "Physics_road_temp",
            "Physics_air_temp"
        ]
    
    @classmethod
    def get_overtaking_opportunity_features(cls) -> List[str]:
        """Get features related to overtaking opportunities"""
        return [
            "Physics_speed_kmh",
            "Physics_gas",
            "Physics_brake",
            "Physics_steer_angle",
            "Physics_g_force_x",
            "Physics_g_force_y",
            "Graphics_position",
            "Graphics_gap_ahead",
            "Graphics_gap_behind",
            "Graphics_normalized_car_position",
            "Graphics_current_sector_index",
            "Graphics_distance_traveled",
            "Graphics_track_status",
            "Graphics_global_yellow",
            "Graphics_global_yellow_s1",
            "Graphics_global_yellow_s2",
            "Graphics_global_yellow_s3",
            "Graphics_is_in_pit_lane",
            "Physics_tc",
            "Physics_abs"
        ]
    
    @classmethod
    def get_racing_line_optimization_features(cls) -> List[str]:
        """Get features for racing line optimization"""
        return [
            "Physics_speed_kmh",
            "Physics_steer_angle",
            "Physics_gas",
            "Physics_brake",
            "Physics_g_force_x",
            "Physics_g_force_y",
            "Physics_slip_angle_front_left",
            "Physics_slip_angle_front_right",
            "Physics_heading",
            "Physics_pitch",
            "Physics_roll",
            "Graphics_normalized_car_position",
            "Graphics_current_sector_index",
            "Graphics_distance_traveled",
            "Graphics_delta_lap_time",
            "Graphics_last_sector_time",
            "Physics_local_velocity_x",
            "Physics_local_velocity_y",
            "Physics_local_velocity_z",
            "Physics_velocity_x",
            "Physics_velocity_y",
            "Physics_velocity_z"
        ]
    
    @classmethod
    def get_weather_adaptation_features(cls) -> List[str]:
        """Get features for weather adaptation analysis"""
        return [
            "Physics_road_temp",
            "Physics_air_temp",
            "Graphics_wind_speed",
            "Graphics_wind_direction",
            "Graphics_rain_tyres",
            "Graphics_rain_light",
            "Graphics_wiper_stage",
            "Physics_tyre_core_temp_front_left",
            "Physics_tyre_core_temp_front_right",
            "Physics_tyre_core_temp_rear_left",
            "Physics_tyre_core_temp_rear_right",
            "Physics_wheel_slip_front_left",
            "Physics_wheel_slip_front_right",
            "Physics_wheel_slip_rear_left",
            "Physics_wheel_slip_rear_right",
            "Physics_slip_angle_front_left",
            "Physics_slip_angle_front_right",
            "Physics_slip_angle_rear_left",
            "Physics_slip_angle_rear_right",
            "Physics_speed_kmh",
            "Physics_gas",
            "Physics_brake",
            "Physics_tc",
            "Graphics_tc_level"
        ]
    
    @classmethod
    def get_consistency_analysis_features(cls) -> List[str]:
        """Get features for driving consistency analysis"""
        return [
            "Graphics_last_time",
            "Graphics_best_time",
            "Graphics_delta_lap_time",
            "Graphics_last_sector_time",
            "Graphics_current_sector_index",
            "Physics_speed_kmh",
            "Physics_gas",
            "Physics_brake",
            "Physics_steer_angle",
            "Physics_g_force_x",
            "Physics_g_force_y",
            "Physics_slip_angle_front_left",
            "Physics_slip_angle_front_right",
            "Physics_slip_angle_rear_left",
            "Physics_slip_angle_rear_right",
            "Physics_brake_temp_front_left",
            "Physics_brake_temp_front_right",
            "Physics_tyre_core_temp_front_left",
            "Physics_tyre_core_temp_front_right"
        ]
    
    @classmethod
    def get_sector_time_features(cls) -> List[str]:
        """Get features specific to sector time prediction"""
        return [
            "Graphics_last_sector_time",
            "Graphics_current_sector_index",
            "Physics_speed_kmh",
            "Physics_gas",
            "Physics_brake",
            "Physics_steer_angle",
            "Physics_gear",
            "Physics_rpm",
            "Physics_g_force_x",
            "Physics_g_force_y",
            "Physics_slip_angle_front_left",
            "Physics_slip_angle_front_right",
            "Graphics_normalized_car_position",
            "Graphics_distance_traveled",
            "Physics_tyre_core_temp_front_left",
            "Physics_tyre_core_temp_front_right",
            "Physics_brake_temp_front_left",
            "Physics_brake_temp_front_right"
        ]
        
    @classmethod
    def get_driver_behaviour_features(cls) -> List[str]:
        """Get features specific to driver behaviour analysis"""
        return [
            "Graphics_normalized_car_position",
            "Physics_car_damage_rear",
            "Physics_car_damage_front",
            "Physics_car_damage_left",
            "Physics_car_damage_right",
            "Physics_car_damage_center",
            "Physics_speed_kmh",
            "Physics_gas",
            "Physics_brake",
            "Physics_steer_angle",
            "Physics_gear",
            "Physics_rpm",
            "Physics_g_force_x",
            "Physics_g_force_y",
            "Physics_slip_angle_front_left",
            "Physics_slip_angle_front_right",
            "Physics_kerb_vibration",
            "Physics_slip_vibration",
            "Physics_velocity_x",
            "Physics_velocity_y",
            "Physics_velocity_z",
            "Graphics_track_grip_status",
            "Graphics_current_tyre_set"
        ]
        
    @classmethod
    def get_features_for_learning_expert(cls) -> List[str]:
        """Get features specific to imitate expert learning - includes all features used in imitate_expert_learning_service, corner_identification_unsupervised_service, and tire_grip_analysis_service"""
        return [
            "Graphics_normalized_car_position",
            "Graphics_player_pos_x",
            "Graphics_player_pos_y", 
            "Graphics_player_pos_z",
            "Graphics_current_time",
            "Graphics_last_time",
            "Graphics_completed_lap",
            "Physics_speed_kmh",
            "Physics_gas",
            "Physics_brake",
            "Physics_steer_angle",
            "Physics_gear",
            "Physics_rpm",
            "Physics_roll",
            "Physics_pitch",
            "Physics_g_force_x",
            "Physics_g_force_y",
            "Physics_g_force_z",
            "Physics_slip_angle_front_left",
            "Physics_slip_angle_front_right",
            "Physics_slip_angle_rear_left",
            "Physics_slip_angle_rear_right",
            "Physics_slip_ratio_front_left",
            "Physics_slip_ratio_front_right",
            "Physics_slip_ratio_rear_left",
            "Physics_slip_ratio_rear_right",
            "Physics_suspension_travel_front_left",
            "Physics_suspension_travel_front_right",
            "Physics_suspension_travel_rear_left",
            "Physics_suspension_travel_rear_right",
            "Physics_tyre_core_temp_front_left",
            "Physics_tyre_core_temp_front_right",
            "Physics_tyre_core_temp_rear_left",
            "Physics_tyre_core_temp_rear_right",
            "Physics_brake_temp_front_left",
            "Physics_brake_temp_front_right",
            "Physics_brake_temp_rear_left",
            "Physics_brake_temp_rear_right",
            "Physics_wheel_pressure_front_left",
            "Physics_wheel_pressure_front_right",
            "Physics_wheel_pressure_rear_left",
            "Physics_wheel_pressure_rear_right",
            "Physics_velocity_x",
            "Physics_velocity_y",
            "Physics_velocity_z",
            "Graphics_track_grip_status",
            "Graphics_current_tyre_set",
            "Graphics_is_valid_lap",
            "Static_car_model",
            "Static_track",
            "Opponent_1_pos_x", "Opponent_1_pos_y", "Opponent_1_pos_z", "Opponent_1_distance", "Opponent_1_car_id",
            "Opponent_2_pos_x", "Opponent_2_pos_y", "Opponent_2_pos_z", "Opponent_2_distance", "Opponent_2_car_id",
            "Opponent_3_pos_x", "Opponent_3_pos_y", "Opponent_3_pos_z", "Opponent_3_distance", "Opponent_3_car_id",
            "Opponent_4_pos_x", "Opponent_4_pos_y", "Opponent_4_pos_z", "Opponent_4_distance", "Opponent_4_car_id",
            "Opponent_5_pos_x", "Opponent_5_pos_y", "Opponent_5_pos_z", "Opponent_5_distance", "Opponent_5_car_id",
        ]
        
        
    @classmethod
    def get_features_for_model_type(cls, model_type: str) -> List[str]:
        """
        Get recommended features for a specific model type
        
        Args:
            model_type: The type of prediction task
            
        Returns:
            List of recommended feature names for the task
        """
        feature_map = {
            "driver_behaviour": cls.get_driver_behaviour_features(),
            "performance_classification": cls.get_performance_critical_features(),
            "setup_recommendation": cls.get_setup_features(),
            "tire_strategy": cls.get_tire_strategy_features(),
            "fuel_consumption": cls.get_fuel_consumption_features(),
            "brake_performance": cls.get_brake_performance_features(),
            "overtaking_opportunity": cls.get_overtaking_opportunity_features(),
            "racing_line_optimization": cls.get_racing_line_optimization_features(),
            "weather_adaptation": cls.get_weather_adaptation_features(),
            "consistency_analysis": cls.get_consistency_analysis_features(),
            "damage_prediction": cls.get_damage_features()
        }
        
        return feature_map.get(model_type, cls.get_performance_critical_features())

class FeatureProcessor:
    """Process and prepare telemetry features for AI analysis
    1. Data Initialization & Validation
        Takes a pandas DataFrame of telemetry data as input
        Ensures all column names are strings to prevent errors
        Validates which expected telemetry features are present vs missing
    2. Data Preprocessing (prepare_for_analysis)
        Handles complex nested structures from AC Competizione telemetry (arrays, dictionaries)
        Fills missing values with appropriate defaults (numeric columns get 0)
        Converts string boolean values to actual boolean/numeric format
        Cleans problematic data that could cause AI model issues
    3. Complex Field Processing (_handle_complex_fields)
        Car coordinates: Extracts player car position (x, y, z) from complex array data
        Car IDs: Converts car ID arrays into active car counts
        Time strings: Parses time formats like "9:17:920" into numeric milliseconds
        Boolean fields: Standardizes boolean data across different formats
    4. Performance Metrics Extraction
        Generates key racing performance metrics including:
        Speed analysis: max, average, min speeds and consistency
        Lap time analysis: best lap, worst lap, average lap times
        Temperature monitoring: tire and brake temperature ranges
    G-force analysis: lateral, longitudinal, and vertical G-forces
    """

    def __init__(self, df: pd.DataFrame):
        # Ensure all column names are strings to prevent AttributeError
        if any(not isinstance(col, str) for col in df.columns):
            print("[DEBUG] FeatureProcessor: Converting non-string column names to strings")
            df = df.copy()
            df.columns = [str(col) for col in df.columns]
            
        self.df = df
        self.features = TelemetryFeatures()
    
    def validate_features(self) -> Dict[str, Any]:
        """Validate which expected features are present in the data"""
        
        all_features = self.features.get_all_features()
        available_features = list(self.df.columns)
        
        present_features = [f for f in all_features if f in available_features]
        missing_features = [f for f in all_features if f not in available_features]
        
        physics_present = [f for f in self.features.PHYSICS_FEATURES if f in available_features]
        graphics_present = [f for f in self.features.GRAPHICS_FEATURES if f in available_features]
        static_present = [f for f in self.features.STATIC_FEATURES if f in available_features]
        
        return {
            "total_expected": len(all_features),
            "total_present": len(present_features),
            "coverage_percentage": round((len(present_features) / len(all_features)) * 100, 2),
            "present_features": present_features,
            "missing_features": missing_features,
            "physics_coverage": len(physics_present),
            "graphics_coverage": len(graphics_present),
            "static_coverage": len(static_present),
            "categorized_present": {
                "physics": physics_present,
                "graphics": graphics_present,
                "static": static_present
            }
        }
    
    def general_cleaning_for_analysis(self) -> pd.DataFrame:
        """Prepare the DataFrame for AI analysis by cleaning and preprocessing"""
        
        processed_df = self.df.copy()
        
        # Ensure all column names are strings to prevent AttributeError on .lower()
        if any(not isinstance(col, str) for col in processed_df.columns):
            processed_df.columns = [str(col) for col in processed_df.columns]
        
        # Handle complex nested structures from AC Competizione telemetry
        self._handle_complex_fields(processed_df)

        # Handle missing values
        numeric_columns = processed_df.select_dtypes(include=['number']).columns
        processed_df[numeric_columns] = processed_df[numeric_columns].fillna(0)

        # Convert string boolean values to actual booleans
        boolean_features = [col for col in processed_df.columns if
                          isinstance(col, str) and any(keyword in col.lower() for keyword in ['on', 'enabled', 'valid', 'running', 'controlled'])]

        for col in boolean_features:
            if col in processed_df.columns:
                if processed_df[col].dtype == 'object':
                    processed_df[col] = processed_df[col].map({
                        'True': True, 'False': False, 'true': True, 'false': False,
                        '1': True, '0': False, 1: True, 0: False
                    }).fillna(False)

        # Round float columns to six decimal places
        float_columns = processed_df.select_dtypes(include=['float']).columns
        processed_df[float_columns] = processed_df[float_columns].round(6)
        
        # Persist cleaned frame so downstream helpers (e.g., add_time_delta) operate on enriched data
        self.df = processed_df
        return processed_df
    
    def split_into_laps(self, df: Optional[pd.DataFrame] = None) -> List[Dict[str, Any]]:
        """
        Split telemetry into laps preserving original row order.
        
        Returns:
            List of dicts with keys: lap_num, lap_time_ms, dataframe
        """
        target_df = df if df is not None else self.df

        required_columns = {"Graphics_completed_lap", "Graphics_current_time"}
        missing_columns = [col for col in required_columns if col not in target_df.columns]
        if missing_columns:
            raise ValueError(
                "Missing required columns for lap split: " + ", ".join(missing_columns)
            )

        # Create grouper based on consecutive lap number changes
        # We use an external Series for grouping to avoid adding a helper column to the DataFrame
        laps = target_df["Graphics_completed_lap"].astype(int)
        lap_change = laps != laps.shift()
        
        # Also check for normalized position reset if available to handle cases where lap counter doesn't update
        if "Graphics_normalized_car_position" in target_df.columns:
            norm_pos = target_df["Graphics_normalized_car_position"]
            # Detect reset: current < 0.1 and previous > 0.9
            # This handles the case where the car crosses the line (0.99 -> 0.00)
            pos_reset = (norm_pos < 0.1) & (norm_pos.shift(1) > 0.9)
            split_condition = lap_change | pos_reset
        else:
            split_condition = lap_change
            
        lap_grouper = split_condition.cumsum()

        lap_structs: List[Dict[str, Any]] = []

        for _, lap_df in target_df.groupby(lap_grouper, sort=False):
            # Ensure lap has good coverage of the track (0.1 to 0.9)
            if "Graphics_normalized_car_position" in lap_df.columns:
                norm_pos = lap_df["Graphics_normalized_car_position"]
                if not norm_pos.empty and (norm_pos.min() > 0.1 or norm_pos.max() < 0.9):
                    continue

            lap_num = int(lap_df["Graphics_completed_lap"].iloc[-1])

            # Get lap time from max current_time
            time_values = lap_df["Graphics_current_time"].to_numpy(dtype=float)
            lap_time_ms = None
            if time_values.size and not np.isnan(time_values).all():
                max_time = float(np.nanmax(time_values))
                if np.isfinite(max_time) and max_time > 0:
                    lap_time_ms = max_time

            lap_structs.append({
                "lap_num": lap_num,
                "lap_time_ms": lap_time_ms,
                "dataframe": lap_df.copy(),
            })

        return lap_structs

    def _handle_complex_fields(self, df: pd.DataFrame) -> None:
        """Handle complex nested fields from AC Competizione telemetry
        player car coordinates is extracted from array of car coordinates, and named as Graphics_player_pos_x, Graphics_player_pos_y, Graphics_player_pos_z"""
        
        # Handle Graphics_car_coordinates array - extract player car position and opponents
        if 'Graphics_car_coordinates' in df.columns:
            try:
                # Extract first car coordinates (player car) if it's a list
                for idx in df.index:
                    car_coords_raw = df.loc[idx, 'Graphics_car_coordinates']
                    player_car_id = df.loc[idx, 'Graphics_player_car_id']
                    
                    # Get car IDs if available
                    car_ids = []
                    if 'Graphics_car_id' in df.columns:
                        car_ids_raw = df.loc[idx, 'Graphics_car_id']
                        # Use helper to parse car IDs (handles lists and strings)
                        parsed_ids = _parse_car_coordinates(car_ids_raw)
                        if isinstance(parsed_ids, list):
                            car_ids = parsed_ids

                    # Parse car coordinates using the helper function
                    car_coords = _parse_car_coordinates(car_coords_raw)
                    
                    if car_coords is not None and isinstance(car_coords, list) and len(car_coords) > 0:
                        # Ensure player_car_id is valid
                        try:
                            player_car_id = int(player_car_id) if player_car_id is not None else 0
                        except (ValueError, TypeError):
                            player_car_id = 0
                        
                        player_pos = {'x': 0.0, 'y': 0.0, 'z': 0.0}
                        
                        if 0 <= player_car_id < len(car_coords):
                            player_coord = car_coords[player_car_id]
                            if isinstance(player_coord, dict):
                                player_pos['x'] = _safe_float(player_coord.get('x', 0))
                                player_pos['y'] = _safe_float(player_coord.get('y', 0))
                                player_pos['z'] = _safe_float(player_coord.get('z', 0))
                                
                                df.loc[idx, 'Graphics_player_pos_x'] = player_pos['x']
                                df.loc[idx, 'Graphics_player_pos_y'] = player_pos['y']
                                df.loc[idx, 'Graphics_player_pos_z'] = player_pos['z']
                        else:
                            print(f"[DEBUG] Invalid player_car_id {player_car_id} for car_coords length {len(car_coords)}")
                            # Set default values
                            df.loc[idx, 'Graphics_player_pos_x'] = 0.0
                            df.loc[idx, 'Graphics_player_pos_y'] = 0.0
                            df.loc[idx, 'Graphics_player_pos_z'] = 0.0
                        
                        # Calculate distances to other cars
                        opponents = []
                        for i, coord in enumerate(car_coords):
                            if i == player_car_id:
                                continue
                            
                            if isinstance(coord, dict):
                                cx = _safe_float(coord.get('x', 0))
                                cy = _safe_float(coord.get('y', 0))
                                cz = _safe_float(coord.get('z', 0))
                                
                                dist = math.sqrt(
                                    (cx - player_pos['x'])**2 + 
                                    (cy - player_pos['y'])**2 + 
                                    (cz - player_pos['z'])**2
                                )
                                
                                # Get car ID if available
                                c_id = 0
                                if i < len(car_ids):
                                    c_id = car_ids[i]
                                
                                opponents.append({
                                    'dist': dist,
                                    'x': cx,
                                    'y': cy,
                                    'z': cz,
                                    'id': c_id
                                })
                        
                        # Sort by distance
                        opponents.sort(key=lambda x: x['dist'])
                        
                        # Take top 5
                        for i in range(5):
                            prefix = f"Opponent_{i+1}"
                            if i < len(opponents):
                                opp = opponents[i]
                                df.loc[idx, f"{prefix}_pos_x"] = opp['x']
                                df.loc[idx, f"{prefix}_pos_y"] = opp['y']
                                df.loc[idx, f"{prefix}_pos_z"] = opp['z']
                                df.loc[idx, f"{prefix}_distance"] = opp['dist']
                                df.loc[idx, f"{prefix}_car_id"] = opp['id']
                            else:
                                # Fill with "very large" values
                                df.loc[idx, f"{prefix}_pos_x"] = 100000.0
                                df.loc[idx, f"{prefix}_pos_y"] = 100000.0
                                df.loc[idx, f"{prefix}_pos_z"] = 100000.0
                                df.loc[idx, f"{prefix}_distance"] = 100000.0
                                df.loc[idx, f"{prefix}_car_id"] = -1

                    else:
                        print(f"[DEBUG] Could not parse car_coords for row {idx}: {type(car_coords_raw)} -> {car_coords}")
                        # Set default values
                        df.loc[idx, 'Graphics_player_pos_x'] = 0.0
                        df.loc[idx, 'Graphics_player_pos_y'] = 0.0
                        df.loc[idx, 'Graphics_player_pos_z'] = 0.0
                        
                        # Fill opponents with default values
                        for i in range(5):
                            prefix = f"Opponent_{i+1}"
                            df.loc[idx, f"{prefix}_pos_x"] = 100000.0
                            df.loc[idx, f"{prefix}_pos_y"] = 100000.0
                            df.loc[idx, f"{prefix}_pos_z"] = 100000.0
                            df.loc[idx, f"{prefix}_distance"] = 100000.0
                            df.loc[idx, f"{prefix}_car_id"] = -1
                        
                # Remove the complex column after extraction
                df.drop('Graphics_car_coordinates', axis=1, inplace=True)
                
            except Exception as e:
                print(f"[DEBUG] Error processing Graphics_car_coordinates: {str(e)}")
                # If there's an error, just drop the problematic column
                if 'Graphics_car_coordinates' in df.columns:
                    df.drop('Graphics_car_coordinates', axis=1, inplace=True)
        
        # Handle Graphics_car_id array - convert to count of active cars
        if 'Graphics_car_id' in df.columns:
            try:
                for idx in df.index:
                    car_ids = df.loc[idx, 'Graphics_car_id']
                    if isinstance(car_ids, list):
                        # Count non-zero car IDs
                        active_cars = sum(1 for car_id in car_ids if car_id != 0)
                        df.loc[idx, 'Graphics_active_cars_count'] = active_cars
                
                # Remove the complex column after extraction
                df.drop('Graphics_car_id', axis=1, inplace=True)

                
            except Exception as e:
                print(f"[DEBUG] Error processing Graphics_car_id: {str(e)}")
                if 'Graphics_car_id' in df.columns:
                    df.drop('Graphics_car_id', axis=1, inplace=True)
        
        # Handle string time fields - convert to numeric milliseconds
        time_string_fields = [
            'Graphics_current_time_str', 'Graphics_last_time_str', 
            'Graphics_best_time_str', 'Graphics_delta_lap_time_str',
            'Graphics_estimated_lap_time_str'
        ]
        
        for field in time_string_fields:
            if field in df.columns:
                try:
                    # Convert time strings to numeric values where possible
                    df[field + '_numeric'] = df[field].apply(self._parse_time_string)
                except Exception as e:
                    print(f"[DEBUG] Error converting {field}: {str(e)}")
        
        # Handle boolean fields that might be strings
        boolean_fields = [
            'Physics_pit_limiter_on', 'Physics_autoshifter_on', 'Physics_is_ai_controlled',
            'Physics_ignition_on', 'Physics_starter_engine_on', 'Physics_is_engine_running',
            'Graphics_is_in_pit', 'Graphics_ideal_line_on', 'Graphics_is_in_pit_lane',
            'Graphics_mandatory_pit_done', 'Graphics_is_setup_menu_visible',
            'Graphics_rain_light', 'Graphics_flashing_light', 'Graphics_is_delta_positive',
            'Graphics_is_valid_lap', 'Graphics_direction_light_left', 'Graphics_direction_light_right',
            'Graphics_global_yellow', 'Graphics_global_yellow_s1', 'Graphics_global_yellow_s2',
            'Graphics_global_yellow_s3', 'Graphics_global_white', 'Graphics_global_green',
            'Graphics_global_chequered', 'Graphics_global_red', 'Static_penalty_enabled',
            'Static_aid_auto_clutch', 'Static_is_online'
        ]
        
        for field in boolean_fields:
            if field in df.columns:
                try:
                    # Convert boolean fields to numeric (0/1)
                    df[field] = df[field].astype(bool).astype(int)
                except Exception as e:
                    print(f"[DEBUG] Error converting boolean field {field}: {str(e)}")
    
    def _parse_time_string(self, time_str: str) -> float:
        """Parse time string to numeric milliseconds"""
        try:
            if pd.isna(time_str) or time_str in ['-:--:---', '35791:23:647']:
                return 0.0
            
            # Handle format like "9:17:920" (minutes:seconds:milliseconds)
            if ':' in str(time_str):
                parts = str(time_str).split(':')
                if len(parts) == 3:
                    minutes = float(parts[0])
                    seconds = float(parts[1])
                    milliseconds = float(parts[2])
                    return (minutes * 60 + seconds) * 1000 + milliseconds
            
            # If it's already a number, return as is
            return float(time_str)
            
        except:
            return 0.0

    def filter_features_by_list(self, df: pd.DataFrame, feature_list: List[str]) -> pd.DataFrame:
        """
        Filter DataFrame to only include columns that exist in the provided feature list
        
        Args:
            df: Input DataFrame with telemetry data
            feature_list: List of feature names to keep (e.g., from get_racing_line_optimization_features)
            
        Returns:
            Filtered DataFrame containing only the columns that exist in both the DataFrame and feature_list
        """
        if df.empty:
            print("[WARNING] Input DataFrame is empty")
            return df.copy()
        
        if not feature_list:
            print("[WARNING] Feature list is empty, returning empty DataFrame")
            return pd.DataFrame()
        
        # Get the intersection of DataFrame columns and the feature list
        available_features = [col for col in feature_list if col in df.columns]
        missing_features = [col for col in feature_list if col not in df.columns]

        if not available_features:
            print(f"[WARNING] No features from the provided list are available in the DataFrame")
            print(f"[INFO] Requested features: {feature_list[:10]}{'...' if len(feature_list) > 10 else ''}")
            print(f"[INFO] Available columns: {list(df.columns)[:10]}{'...' if len(df.columns) > 10 else ''}")
            return pd.DataFrame()
        
        # Filter DataFrame to include only the available features
        filtered_df = df[available_features].copy()
        
        #print(f"[INFO] Filtered DataFrame to {len(available_features)} features out of {len(feature_list)} requested")
        if missing_features:
            print(f"[INFO] Missing {len(missing_features)} features: {missing_features[:5]}{'...' if len(missing_features) > 5 else ''}")
            raise ValueError(f"Missing features: {missing_features}")
        
        return filtered_df

    def strip_dataframe_by_time_gap(
        self,
        df: pd.DataFrame,
        gap_between: float
    ) -> pd.DataFrame:
        """Down-sample telemetry frames using a fixed time gap in milliseconds.

        Args:
            df: DataFrame containing telemetry data.
            gap_between: Minimum spacing between retained samples (ms).

        Returns:
            DataFrame with rows removed so consecutive samples are at least
            ``gap_between`` milliseconds apart.
        """

        if gap_between <= 0:
            raise ValueError("gap_between must be a positive value in milliseconds")

        if df is None or df.empty:
            return pd.DataFrame()

        if 'Graphics_current_time' not in df.columns:
            return df.copy()

        working = df.copy()
        time_values = pd.to_numeric(
            working['Graphics_current_time'], errors='coerce'
        ).to_numpy(dtype=float)

        valid_mask = ~np.isnan(time_values)
        working = working.iloc[valid_mask].copy()
        time_values = time_values[valid_mask]

        if time_values.size == 0:
            return df.iloc[0:0].copy()

        keep_mask = np.zeros(len(working), dtype=bool)
        last_selected = None

        for idx, current_time in enumerate(time_values):
            if last_selected is None or current_time < last_selected:
                keep_mask[idx] = True
                last_selected = current_time
                continue

            if (current_time - last_selected) >= gap_between:
                keep_mask[idx] = True
                last_selected = current_time

        if not keep_mask.any():
            return df.iloc[0:0].copy()

        stripped = working.iloc[keep_mask].copy()
        return stripped.reset_index(drop=True)

    def flip_y_z_features(self) -> pd.DataFrame:
        """Swap values across *_y and *_z telemetry columns to align axis conventions."""

        if self.df is None or self.df.empty:
            return self.df if self.df is not None else pd.DataFrame()

        swapped_pairs = 0

        for col in list(self.df.columns):
            if not isinstance(col, str) or not col.endswith("_y"):
                continue

            counterpart = f"{col[:-2]}_z"
            if counterpart not in self.df.columns:
                continue

            y_values = self.df[col].copy()
            z_values = self.df[counterpart].copy()
            self.df[col] = z_values
            self.df[counterpart] = y_values
            swapped_pairs += 1

        if swapped_pairs == 0:
            print("[INFO] No *_y/_z feature pairs found to flip.")

        return self.df
