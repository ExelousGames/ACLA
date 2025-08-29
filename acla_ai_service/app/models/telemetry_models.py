"""
Telemetry data models for Assetto Corsa Competizione telemetry features
"""

from pydantic import BaseModel
from typing import Dict, List, Any, Optional
import pandas as pd
import math

def _safe_float(value):
    """Convert value to float, handling NaN and infinity"""
    try:
        float_val = float(value)
        if math.isnan(float_val) or math.isinf(float_val):
            return 0.0
        return float_val
    except (ValueError, TypeError):
        return 0.0

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
        "Physics_steer_angle",
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
            "Graphics_position"
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

class TelemetryDataModel(BaseModel):
    """Pydantic model for telemetry data validation"""
    session_id: str
    timestamp: Optional[float] = None
    physics_data: Dict[str, Any] = {}
    graphics_data: Dict[str, Any] = {}
    static_data: Dict[str, Any] = {}
    
    class Config:
        extra = "allow"  # Allow additional fields

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
class FeatureProcessor:


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
    
    def prepare_for_analysis(self) -> pd.DataFrame:
        """Prepare the DataFrame for AI analysis by cleaning and preprocessing"""
        
        processed_df = self.df.copy()
        
        # Ensure all column names are strings to prevent AttributeError on .lower()
        if any(not isinstance(col, str) for col in processed_df.columns):
            print("[DEBUG] Converting non-string column names to strings")
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
        
        return processed_df
    
    def _handle_complex_fields(self, df: pd.DataFrame) -> None:
        """Handle complex nested fields from AC Competizione telemetry"""
        
        # Handle Graphics_car_coordinates array - extract player car position
        if 'Graphics_car_coordinates' in df.columns:
            try:
                # Extract first car coordinates (player car) if it's a list
                for idx in df.index:
                    car_coords = df.loc[idx, 'Graphics_car_coordinates']
                    if isinstance(car_coords, list) and len(car_coords) > 0:
                        player_coord = car_coords[0]
                        if isinstance(player_coord, dict):
                            df.loc[idx, 'Graphics_player_pos_x'] = player_coord.get('x', 0)
                            df.loc[idx, 'Graphics_player_pos_y'] = player_coord.get('y', 0)
                            df.loc[idx, 'Graphics_player_pos_z'] = player_coord.get('z', 0)
                
                # Remove the complex column after extraction
                df.drop('Graphics_car_coordinates', axis=1, inplace=True)
                print("[DEBUG] Processed Graphics_car_coordinates array")
                
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
                print("[DEBUG] Processed Graphics_car_id array")
                
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
                    print(f"[DEBUG] Converted {field} to numeric")
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
    
    def extract_performance_metrics(self) -> Dict[str, Any]:
        """Extract key performance metrics from telemetry data"""
        metrics = {}
        
        # Speed analysis
        if 'Physics_speed_kmh' in self.df.columns:
            speed_data = self.df['Physics_speed_kmh'].dropna()
            if len(speed_data) > 0:
                metrics['speed'] = {
                    'max_speed': _safe_float(speed_data.max()),
                    'avg_speed': _safe_float(speed_data.mean()),
                    'min_speed': _safe_float(speed_data.min()),
                    'speed_consistency': _safe_float(speed_data.std())
                }
        
        # Lap time analysis (if available)
        if 'Graphics_last_time' in self.df.columns:
            valid_times = self.df[self.df['Graphics_last_time'] > 0]['Graphics_last_time'].dropna()
            if not valid_times.empty:
                metrics['lap_times'] = {
                    'best_lap': _safe_float(valid_times.min()),
                    'worst_lap': _safe_float(valid_times.max()),
                    'avg_lap': _safe_float(valid_times.mean()),
                    'total_laps': len(valid_times)
                }
        
        # Temperature analysis
        temp_features = [col for col in self.df.columns if isinstance(col, str) and 'temp' in col.lower()]
        if temp_features:
            metrics['temperatures'] = {}
            for feature in temp_features:
                temp_data = self.df[feature].dropna()
                if len(temp_data) > 0:
                    metrics['temperatures'][feature] = {
                        'max': _safe_float(temp_data.max()),
                        'avg': _safe_float(temp_data.mean()),
                        'min': _safe_float(temp_data.min())
                    }
        
        # G-Force analysis
        g_features = ['Physics_g_force_x', 'Physics_g_force_y', 'Physics_g_force_z']
        available_g = [f for f in g_features if f in self.df.columns]
        if available_g:
            metrics['g_forces'] = {}
            for feature in available_g:
                g_data = self.df[feature].dropna()
                if len(g_data) > 0:
                    metrics['g_forces'][feature] = {
                        'max': _safe_float(g_data.max()),
                        'min': _safe_float(g_data.min()),
                        'avg': _safe_float(g_data.mean())
                    }
        
        return metrics
