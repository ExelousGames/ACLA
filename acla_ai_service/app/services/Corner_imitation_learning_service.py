"""
Corner-Specific Imitation Learning Service for Assetto Corsa Competizione

This service implements corner-specific imitation learning algorithms to learn optimal
driving behaviors for specific track corners and phases.
"""

import numpy as np
import pandas as pd
import pickle
import warnings
import io
import base64
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime
from pathlib import Path

# Scikit-learn imports for imitation learning
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Import telemetry models
from ..models.telemetry_models import TelemetryFeatures, FeatureProcessor

warnings.filterwarnings('ignore', category=UserWarning)


class CornerSpecificLearner:
    """Learn optimal actions for specific corner phases using normalized car position"""

    def __init__(self, corner_models: Optional[Dict[str, Any]] = None):
        self.corner_models = corner_models or {}
        self.scaler = StandardScaler()
        self.corner_phases = ['entry', 'turn_in', 'apex', 'acceleration', 'exit']
    
    def extract_corner_specific_features(self, df: pd.DataFrame, corner_positions: Dict[str, float]) -> pd.DataFrame:
        """
        Extract features for specific corner phases based on normalized car position
        
        Args:
            df: Telemetry DataFrame
            corner_positions: Dict with corner phases and their normalized positions
                             e.g., {'entry': 0.15, 'turn_in': 0.18, 'apex': 0.20, 'exit': 0.22}
        
        Returns:
            DataFrame with corner-specific features
        """
        features = pd.DataFrame()
        
        # Get position data
        if 'Graphics_normalized_car_position' not in df.columns:
            raise ValueError("Graphics_normalized_car_position is required for corner-specific learning")
        
        positions = df['Graphics_normalized_car_position']
        
        # For each corner phase, extract relevant telemetry windows
        for phase, target_position in corner_positions.items():
            # Find closest data points to target position (within ±0.01 normalized distance)
            position_tolerance = 0.01
            phase_mask = (positions >= target_position - position_tolerance) & (positions <= target_position + position_tolerance)
            phase_data = df[phase_mask]
            
            if len(phase_data) == 0:
                continue
            
            # Extract phase-specific features
            phase_prefix = f"{phase}_"
            
            # Speed features
            if 'Physics_speed_kmh' in phase_data.columns:
                features[f"{phase_prefix}speed_mean"] = [phase_data['Physics_speed_kmh'].mean()]
                features[f"{phase_prefix}speed_min"] = [phase_data['Physics_speed_kmh'].min()]
                features[f"{phase_prefix}speed_max"] = [phase_data['Physics_speed_kmh'].max()]
                features[f"{phase_prefix}speed_change"] = [phase_data['Physics_speed_kmh'].diff().mean()]
            
            # Input features
            if 'Physics_gas' in phase_data.columns:
                features[f"{phase_prefix}throttle_mean"] = [phase_data['Physics_gas'].mean()]
                features[f"{phase_prefix}throttle_max"] = [phase_data['Physics_gas'].max()]
            
            if 'Physics_brake' in phase_data.columns:
                features[f"{phase_prefix}brake_mean"] = [phase_data['Physics_brake'].mean()]
                features[f"{phase_prefix}brake_max"] = [phase_data['Physics_brake'].max()]
            
            # Steering features
            if 'Physics_steer_angle' in phase_data.columns:
                features[f"{phase_prefix}steering_mean"] = [phase_data['Physics_steer_angle'].mean()]
                features[f"{phase_prefix}steering_max"] = [abs(phase_data['Physics_steer_angle']).max()]
                features[f"{phase_prefix}steering_smoothness"] = [phase_data['Physics_steer_angle'].diff().abs().mean()]
            
            # Gear usage
            if 'Physics_gear' in phase_data.columns:
                features[f"{phase_prefix}gear_mode"] = [phase_data['Physics_gear'].mode().iloc[0] if not phase_data['Physics_gear'].mode().empty else 3]
            
            # G-force analysis
            g_cols = ['Physics_g_force_x', 'Physics_g_force_y', 'Physics_g_force_z']
            available_g_cols = [col for col in g_cols if col in phase_data.columns]
            
            if len(available_g_cols) >= 2:
                g_magnitude = np.sqrt(sum(phase_data[col]**2 for col in available_g_cols))
                features[f"{phase_prefix}g_force_mean"] = [g_magnitude.mean()]
                features[f"{phase_prefix}g_force_max"] = [g_magnitude.max()]
        
        return features.fillna(0)
    
    def learn_corner_optimal_actions(self, 
                                   expert_df: pd.DataFrame, 
                                   corner_definitions: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """
        Learn optimal actions for each corner and phase
        
        Args:
            expert_df: Expert telemetry data
            corner_definitions: Dict of corners with their phase positions
                               e.g., {
                                   'corner_1': {'entry': 0.15, 'turn_in': 0.18, 'apex': 0.20, 'exit': 0.22},
                                   'corner_2': {'entry': 0.45, 'turn_in': 0.47, 'apex': 0.49, 'exit': 0.51}
                               }
        
        Returns:
            Dictionary with corner-specific models
        """
        corner_models = {}
        for corner_name, corner_positions in corner_definitions.items():
            print(f"[INFO] Learning optimal actions for {corner_name}")
            print(f"[DEBUG] Corner positions: {corner_positions}")
            
            # Extract corner-specific data for all phases
            corner_data = self._extract_corner_telemetry(expert_df, corner_positions)
            
            if corner_data.empty:
                print(f"[WARNING] No data found for {corner_name}")
                print(f"[DEBUG] Total telemetry records: {len(expert_df)}")
                if 'Graphics_normalized_car_position' in expert_df.columns:
                    pos_range = expert_df['Graphics_normalized_car_position'].min(), expert_df['Graphics_normalized_car_position'].max()
                    print(f"[DEBUG] Position range in data: {pos_range}")
                else:
                    print(f"[DEBUG] Graphics_normalized_car_position column missing from telemetry data")
                continue
            
            # Train models for each phase
            phase_models = {}
            
            for phase in self.corner_phases:
                if phase not in corner_positions:
                    print(f"[DEBUG] Phase {phase} not found in corner positions, skipping")
                    continue
                    
                phase_position = corner_positions[phase]
                print(f"[DEBUG] Training model for phase {phase} at position {phase_position:.4f}")
                phase_model = self._train_phase_model(corner_data, phase, phase_position)
                
                if phase_model:
                    phase_models[phase] = phase_model
                    print(f"[DEBUG] Successfully trained model for phase {phase}")
                else:
                    print(f"[DEBUG] Failed to train model for phase {phase}")
            
            if phase_models:
                corner_models[corner_name] = phase_models
                print(f"[DEBUG] Added {len(phase_models)} phase models for {corner_name}")
            else:
                print(f"[DEBUG] No phase models created for {corner_name}")
        
        self.corner_models = corner_models
        
        print(f"[INFO] Completed learning for {len(corner_models)} corners")
        return corner_models
    
    def _extract_corner_telemetry(self, df: pd.DataFrame, corner_positions: Dict[str, float]) -> pd.DataFrame:
        """Extract telemetry data for an entire corner (from entry to exit)"""
        positions = df['Graphics_normalized_car_position']
        
        # Get the range from entry to exit
        min_pos = min(corner_positions.values()) - 0.02  # Add buffer
        max_pos = max(corner_positions.values()) + 0.02
        
        print(f"[DEBUG] Corner range: {min_pos:.4f} to {max_pos:.4f}")
        
        # Handle track wraparound (if corner crosses start/finish line)
        if max_pos > 1.0:
            corner_mask = (positions >= min_pos) | (positions <= (max_pos - 1.0))
            print(f"[DEBUG] Using wraparound logic: pos >= {min_pos:.4f} OR pos <= {max_pos - 1.0:.4f}")
        else:
            corner_mask = (positions >= min_pos) & (positions <= max_pos)
            print(f"[DEBUG] Using normal range: {min_pos:.4f} <= pos <= {max_pos:.4f}")
        
        filtered_data = df[corner_mask].copy()
        print(f"[DEBUG] Extracted {len(filtered_data)} records for this corner range")
        
        return filtered_data
    
    def _train_phase_model(self, corner_data: pd.DataFrame, phase: str, phase_position: float) -> Dict[str, Any]:
        """Train a model for a specific corner phase"""
        
        # Find data points closest to the phase position
        positions = corner_data['Graphics_normalized_car_position']
        distance_to_phase = np.abs(positions - phase_position)
        
        print(f"[DEBUG] Phase {phase}: corner_data has {len(corner_data)} records")
        print(f"[DEBUG] Phase {phase}: distance_to_phase range: {distance_to_phase.min():.4f} to {distance_to_phase.max():.4f}")
        
        # Select points closest to this phase - use more flexible approach for small datasets
        if len(corner_data) == 0:
            print(f"[DEBUG] Phase {phase}: No corner data available")
            return None
        
        if len(corner_data) <= 10:
            # For small datasets, take the closest 50% of points but at least 2 points
            min_points = min(2, len(corner_data))
            phase_threshold = np.percentile(distance_to_phase, 50)
        else:
            # For larger datasets, take the closest 20% of points
            phase_threshold = np.percentile(distance_to_phase, 20)
        
        print(f"[DEBUG] Phase {phase}: Using threshold: {phase_threshold:.4f}")
        
        phase_mask = distance_to_phase <= phase_threshold
        phase_data = corner_data[phase_mask]
        
        # Ensure we have at least some minimum data points
        if len(phase_data) < 2:
            # If threshold gives us too few points, take the closest N points directly
            min_points = min(3, len(corner_data))
            closest_indices = distance_to_phase.nsmallest(min_points).index
            phase_data = corner_data.loc[closest_indices]
            print(f"[DEBUG] Phase {phase}: Fallback - taking {min_points} closest points")
        
        print(f"[DEBUG] Phase {phase}: Selected {len(phase_data)} data points")
        
        if len(phase_data) < 2:  # Need minimum 2 data points
            print(f"[DEBUG] Phase {phase}: Still insufficient data points ({len(phase_data)} < 2)")
            return None
        
        # Define input features (current state before the phase)
        input_features = []
        feature_data = {}
        
        # Speed approaching the phase
        if 'Physics_speed_kmh' in phase_data.columns:
            feature_data['approach_speed'] = phase_data['Physics_speed_kmh'].iloc[0] if len(phase_data) > 0 else 0
            input_features.append('approach_speed')
        
        # Current gear
        if 'Physics_gear' in phase_data.columns:
            feature_data['current_gear'] = phase_data['Physics_gear'].iloc[0] if len(phase_data) > 0 else 3
            input_features.append('current_gear')
        
        # Position in corner sequence
        feature_data['phase_position'] = phase_position
        input_features.append('phase_position')
        
        # Define target actions (what expert does at this phase)
        target_actions = {}
        
        if 'Physics_gas' in phase_data.columns:
            throttle_value = phase_data['Physics_gas'].mean()
            throttle_change_rate = abs(phase_data['Physics_gas'].diff().mean()) if len(phase_data) > 1 else 0
            target_actions['optimal_throttle'] = {
                'value': throttle_value,
                'description': self._get_throttle_description(throttle_value),
                'change_rate': throttle_change_rate,
                'rapidity': self._get_action_rapidity(throttle_change_rate, 'throttle')
            }
        
        if 'Physics_brake' in phase_data.columns:
            brake_value = phase_data['Physics_brake'].mean()
            brake_change_rate = abs(phase_data['Physics_brake'].diff().mean()) if len(phase_data) > 1 else 0
            target_actions['optimal_brake'] = {
                'value': brake_value,
                'description': self._get_brake_description(brake_value),
                'change_rate': brake_change_rate,
                'rapidity': self._get_action_rapidity(brake_change_rate, 'brake')
            }
        
        if 'Physics_steer_angle' in phase_data.columns:
            steering_value = phase_data['Physics_steer_angle'].mean()
            steering_change_rate = abs(phase_data['Physics_steer_angle'].diff().mean()) if len(phase_data) > 1 else 0
            target_actions['optimal_steering'] = {
                'value': steering_value,
                'description': self._get_steering_description(steering_value),
                'change_rate': steering_change_rate,
                'rapidity': self._get_action_rapidity(steering_change_rate, 'steering')
            }
        
        if 'Physics_speed_kmh' in phase_data.columns:
            speed_value = phase_data['Physics_speed_kmh'].mean()
            speed_change_rate = abs(phase_data['Physics_speed_kmh'].diff().mean()) if len(phase_data) > 1 else 0
            target_actions['optimal_speed'] = {
                'value': speed_value,
                'description': self._get_speed_description(speed_value),
                'change_rate': speed_change_rate,
                'rapidity': self._get_action_rapidity(speed_change_rate, 'speed')
            }
        
        # For now, return the expert averages for this phase
        # In a more complex implementation, you would train ML models here
        return {
            'phase': phase,
            'phase_position': phase_position,
            'input_features': input_features,
            'target_actions': target_actions,
            'data_points': len(phase_data)
        }
    
    def predict_optimal_corner_actions(self, 
                                     current_telemetry: pd.DataFrame, 
                                     corner_name: str) -> Dict[str, Any]:
        """
        Predict optimal actions for a specific corner based on current position
        
        Args:
            current_telemetry: Current telemetry state
            corner_name: Name of the corner to get predictions for
        
        Returns:
            Optimal actions for the current corner phase
        """
        if corner_name not in self.corner_models:
            return {"error": f"No model available for {corner_name}"}
        
        # Get current position
        current_position = current_telemetry['Graphics_normalized_car_position'].iloc[0]
        
        # Find the closest corner phase
        corner_model = self.corner_models[corner_name]
        closest_phase = None
        min_distance = float('inf')
        
        for phase, phase_model in corner_model.items():
            phase_position = phase_model['phase_position']
            distance = abs(current_position - phase_position)
            
            if distance < min_distance:
                min_distance = distance
                closest_phase = phase
        
        if closest_phase and min_distance < 0.05:  # Within 5% track distance
            phase_model = corner_model[closest_phase]
            return {
                'corner': corner_name,
                'phase': closest_phase,
                'optimal_actions': phase_model['target_actions'],
                'confidence': max(0, 1 - min_distance * 20)  # Higher confidence for closer positions
            }
        
        return {"error": "Not currently in a recognized corner phase"}
    
    def predict_actions_by_position(self, normalized_car_position: float) -> Dict[str, Any]:
        """
        Predict optimal actions based on normalized car position across all corners
        
        Args:
            normalized_car_position: Current normalized car position (0.0 to 1.0)
        
        Returns:
            Dictionary containing optimal actions and corner information
        """
        if not self.corner_models:
            return {"error": "No corner models available"}
        
        # Find all phases within range of current position
        nearby_phases = []
        
        for corner_name, corner_model in self.corner_models.items():
            for phase, phase_model in corner_model.items():
                phase_position = phase_model['phase_position']
                distance = abs(normalized_car_position - phase_position)
                
                # Consider phases within 3% track distance
                if distance <= 0.03:
                    confidence = max(0, 1 - distance * 33.33)  # Scale confidence
                    nearby_phases.append({
                        'corner': corner_name,
                        'phase': phase,
                        'phase_model': phase_model,
                        'distance': distance,
                        'confidence': confidence,
                        'phase_position': phase_position
                    })
        
        if not nearby_phases:
            return {
                "message": "No corner phases detected at current position",
                "position": normalized_car_position,
                "status": "free_driving"
            }
        
        # Sort by confidence (highest first)
        nearby_phases.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Return the best match
        best_phase = nearby_phases[0]
        
        return {
            'position': normalized_car_position,
            'corner': best_phase['corner'],
            'phase': best_phase['phase'],
            'phase_position': best_phase['phase_position'],
            'optimal_actions': best_phase['phase_model']['target_actions'],
            'confidence': best_phase['confidence'],
            'distance_to_phase': best_phase['distance'],
            'alternatives_available': len(nearby_phases) - 1
        }
    
    def _get_throttle_description(self, throttle_value: float) -> str:
        """Get descriptive text for throttle input"""
        if throttle_value <= 0.05:
            return "no throttle"
        elif throttle_value <= 0.20:
            return "light throttle"
        elif throttle_value <= 0.40:
            return "moderate throttle"
        elif throttle_value <= 0.65:
            return "medium throttle"
        elif throttle_value <= 0.85:
            return "heavy throttle"
        else:
            return "full throttle"
    
    def _get_brake_description(self, brake_value: float) -> str:
        """Get descriptive text for brake input"""
        if brake_value <= 0.05:
            return "no braking"
        elif brake_value <= 0.20:
            return "light braking"
        elif brake_value <= 0.40:
            return "moderate braking"
        elif brake_value <= 0.65:
            return "medium braking"
        elif brake_value <= 0.85:
            return "heavy braking"
        else:
            return "maximum braking"
    
    def _get_steering_description(self, steering_value: float) -> str:
        """Get descriptive text for steering input"""
        abs_steering = abs(steering_value)
        direction = "left" if steering_value < 0 else "right"
        
        if abs_steering <= 0.1:
            return "straight"
        elif abs_steering <= 0.3:
            return f"slight {direction} turn"
        elif abs_steering <= 0.6:
            return f"moderate {direction} turn"
        elif abs_steering <= 1.0:
            return f"sharp {direction} turn"
        else:
            return f"extreme {direction} turn"
    
    def _get_speed_description(self, speed_value: float) -> str:
        """Get descriptive text for speed"""
        if speed_value <= 50:
            return "very slow"
        elif speed_value <= 80:
            return "slow"
        elif speed_value <= 120:
            return "moderate speed"
        elif speed_value <= 160:
            return "fast"
        elif speed_value <= 200:
            return "very fast"
        else:
            return "extremely fast"
    
    def _get_action_rapidity(self, change_rate: float, action_type: str) -> str:
        """Get descriptive text for how rapidly an action changes"""
        if action_type == 'throttle' or action_type == 'brake':
            # For throttle/brake, change_rate is per frame (0-1 range)
            if change_rate <= 0.001:
                return "steady"
            elif change_rate <= 0.005:
                return "gradual"
            elif change_rate <= 0.015:
                return "moderate"
            elif change_rate <= 0.030:
                return "quick"
            elif change_rate <= 0.050:
                return "rapid"
            else:
                return "aggressive"
        
        elif action_type == 'steering':
            # For steering, change_rate is in radians per frame
            if change_rate <= 0.002:
                return "smooth"
            elif change_rate <= 0.008:
                return "gradual"
            elif change_rate <= 0.020:
                return "moderate"
            elif change_rate <= 0.040:
                return "quick"
            elif change_rate <= 0.070:
                return "sharp"
            else:
                return "aggressive"
        
        elif action_type == 'speed':
            # For speed, change_rate is in km/h per frame
            if change_rate <= 0.5:
                return "steady"
            elif change_rate <= 2.0:
                return "gradual"
            elif change_rate <= 5.0:
                return "moderate"
            elif change_rate <= 10.0:
                return "quick"
            elif change_rate <= 20.0:
                return "rapid"
            else:
                return "aggressive"
        
        return "unknown"

class CornerImitationLearningService:
    """
    Corner-specific imitation learning service that combines corner analysis with expert learning.
    
    This service takes the output from TrackCorneringAnalyzer and uses it to train corner-specific
    imitation learning models. It expects corner analysis results in the format:
    {
        'total_corners_detected': int,
        'corner_ids': [int, ...],
        'corner_details': {
            'corner_0': {
                'corner_start_position': float,
                'corner_end_position': float,
                'total_duration_points': int,
                'phases': {
                    'entry': {'normalized_car_position': float, 'duration_points': int},
                    'turn_in': {'normalized_car_position': float, 'duration_points': int},
                    'apex': {'normalized_car_position': float, 'duration_points': int},
                    'acceleration': {'normalized_car_position': float, 'duration_points': int},
                    'exit': {'normalized_car_position': float, 'duration_points': int}
                }
            },
            ...
        }
    }
    """

    def __init__(self, corner_models: Optional[Dict[str, Any]] = None):
        """
        Initialize the corner-specific imitation learning service
        
        Args:
            models_directory: Directory to save/load trained corner models
        """
        self.corner_learner = CornerSpecificLearner(corner_models)
        self.trained_models = {}

        print(f"[INFO] CornerImitationLearningService initialized.")

    def train_corner_specific_model(self, 
                                   telemetry_data: List[Dict[str, Any]], 
                                   corner_analysis_result: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Train corner-specific imitation learning models
        
        Args:
            telemetry_data: Expert telemetry data
            corner_analysis_result: Result from track cornering analysis service with corner details
        
        Returns:
            Corner-specific trained models
        """

        # Convert corner analysis result to corner definitions format
        corner_definitions = self._convert_corner_analysis_to_definitions(corner_analysis_result)
        
        print(f"[INFO] Training corner-specific models for {len(corner_definitions)} corners")
        
        # Convert to DataFrame and process
        telemetry_df = pd.DataFrame(telemetry_data)
        feature_processor = FeatureProcessor(telemetry_df)
        processed_df = feature_processor.general_cleaning_for_analysis()
        
        # Filter for best laps
        processed_df = self._filter_top_performance_laps(processed_df)
        
        # Train corner models
        corner_models = self.corner_learner.learn_corner_optimal_actions(processed_df, corner_definitions)

        ai_model_data = {
            'corner_models': corner_models,
            'corners_trained': len(corner_models),
            'total_phases': sum(len(phases) for phases in corner_models.values()),
            'corner_definitions_used': corner_definitions
        }
        return self.serialize_object_inside(ai_model_data),ai_model_data

    def predict_corner_optimal_actions(self, 
                                     current_telemetry: pd.DataFrame, 
                                     corner_name: str) -> Dict[str, Any]:
        """
        Get optimal actions for a specific corner
        
        Args:
            current_telemetry: Current telemetry state
            corner_name: Name of the corner
        
        Returns:
            Optimal actions for the corner
        """
        if not hasattr(self, 'corner_learner'):
            return {"error": "No corner-specific models trained"}
        
        return self.corner_learner.predict_optimal_corner_actions(current_telemetry, corner_name)
    
    
    def get_optimal_actions_at_position(self, normalized_car_position: float) -> Dict[str, Any]:
        """
        Simple method to get optimal actions at a specific track position
        
        Args:
            normalized_car_position: Track position from 0.0 to 1.0
        
        Returns:
            Dictionary with optimal actions and corner information
        """
        if not hasattr(self, 'corner_learner') or not self.corner_learner:
            return {"error": "No corner models trained. Train models first using train_corner_specific_model()"}
        
        # Use the corner learner's position-based prediction
        result = self.corner_learner.predict_actions_by_position(normalized_car_position)
        
        # Add service-level context
        if 'optimal_actions' in result:
            # Add human-readable summary
            actions_summary = []
            optimal_actions = result['optimal_actions']
            
            if 'optimal_throttle' in optimal_actions:
                throttle = optimal_actions['optimal_throttle']
                actions_summary.append(f"Throttle: {throttle['description']} ({throttle['value']:.2f})")
            
            if 'optimal_brake' in optimal_actions:
                brake = optimal_actions['optimal_brake']
                actions_summary.append(f"Brake: {brake['description']} ({brake['value']:.2f})")
            
            if 'optimal_steering' in optimal_actions:
                steering = optimal_actions['optimal_steering']
                actions_summary.append(f"Steering: {steering['description']} ({steering['value']:.2f})")
            
            if 'optimal_speed' in optimal_actions:
                speed = optimal_actions['optimal_speed']
                actions_summary.append(f"Speed: {speed['description']} ({speed['value']:.1f} km/h)")
            
            result['actions_summary'] = actions_summary
        
        return result

    def get_all_corner_predictions(self, corner_analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get predictions for all corners detected in the corner analysis result
        
        Args:
            corner_analysis_result: Result from track cornering analysis service with corner details
        
        Returns:
            Dictionary containing predictions for all corners and their phases
            
            Return structure:
            {
                'total_corners': int,                    # Total number of corners analyzed
                'corner_predictions': {
                    'corner_0': {
                        'corner_name': str,              # Name of the corner
                        'phases': {
                            'entry': {
                                'phase_position': float,          # Normalized track position (0.0-1.0)
                                'duration_points': int,           # Number of telemetry points in this phase
                                'optimal_actions': {
                                    'optimal_throttle': {
                                        'value': float,           # Throttle value (0.0-1.0)
                                        'description': str,       # Human-readable description
                                        'change_rate': float,     # Rate of change per frame
                                        'rapidity': str          # Description of change speed
                                    },
                                    'optimal_brake': { ... },    # Same structure as throttle
                                    'optimal_steering': { ... }, # Same structure as throttle
                                    'optimal_speed': {
                                        'value': float,          # Speed in km/h
                                        'description': str,      # Human-readable description
                                        'change_rate': float,    # Rate of change per frame
                                        'rapidity': str         # Description of change speed
                                    }
                                },
                                'confidence': float,             # Prediction confidence (0.0-1.0)
                                'actions_summary': [str, ...],   # Human-readable action summaries
                                'distance_to_phase': float       # Distance from trained phase position
                            },
                            'turn_in': { ... },          # Same structure as entry
                            'apex': { ... },             # Same structure as entry
                            'acceleration': { ... },     # Same structure as entry
                            'exit': { ... }              # Same structure as entry
                        },
                        'corner_summary': {
                            'total_phases': int,                 # Total phases in this corner
                            'phases_with_predictions': int,      # Phases that have valid predictions
                            'corner_start_position': float,      # Start of corner (normalized position)
                            'corner_end_position': float,        # End of corner (normalized position)
                            'average_confidence': float          # Average confidence across all phases
                        }
                    },
                    # ... more corners
                },
                'summary': {
                    'corners_with_predictions': int,         # Number of corners with valid predictions
                    'total_phases_predicted': int,           # Total number of phases with predictions
                    'corners_without_predictions': [str],    # List of corner names without predictions
                    'prediction_coverage': float             # Percentage of corners with predictions (0.0-1.0)
                }
            }
        """
        if not hasattr(self, 'corner_learner') or not self.corner_learner:
            return {"error": "No corner models trained. Train models first using train_corner_specific_model()"}
        
        if 'corner_details' not in corner_analysis_result:
            return {"error": "No corner_details found in analysis result"}
        
        corner_details = corner_analysis_result['corner_details']
        all_predictions = {
            'total_corners': len(corner_details),
            'corner_predictions': {},
            'summary': {
                'corners_with_predictions': 0,
                'total_phases_predicted': 0,
                'corners_without_predictions': []
            }
        }
        
        for corner_name, corner_info in corner_details.items():
            corner_predictions = {
                'corner_name': corner_name,
                'phases': {},
                'corner_summary': {
                    'total_phases': 0,
                    'phases_with_predictions': 0,
                    'corner_start_position': corner_info.get('corner_start_position', 0.0),
                    'corner_end_position': corner_info.get('corner_end_position', 0.0)
                }
            }
            
            if 'phases' not in corner_info:
                corner_predictions['error'] = "No phases found for this corner"
                all_predictions['corner_predictions'][corner_name] = corner_predictions
                all_predictions['summary']['corners_without_predictions'].append(corner_name)
                continue
            
            phases = corner_info['phases']
            corner_predictions['corner_summary']['total_phases'] = len(phases)
            
            # Get predictions for each phase
            for phase_name, phase_info in phases.items():
                phase_position = phase_info.get('normalized_car_position', 0.0)
                
                # Get optimal actions at this phase position
                phase_prediction = self.get_optimal_actions_at_position(phase_position)
                
                # Structure the phase prediction
                if 'error' not in phase_prediction:
                    corner_predictions['phases'][phase_name] = {
                        'phase_position': phase_position,
                        'duration_points': phase_info.get('duration_points', 0),
                        'optimal_actions': phase_prediction.get('optimal_actions', {}),
                        'confidence': phase_prediction.get('confidence', 0.0),
                        'actions_summary': phase_prediction.get('actions_summary', []),
                        'distance_to_phase': phase_prediction.get('distance_to_phase', 0.0)
                    }
                    corner_predictions['corner_summary']['phases_with_predictions'] += 1
                    all_predictions['summary']['total_phases_predicted'] += 1
                else:
                    corner_predictions['phases'][phase_name] = {
                        'phase_position': phase_position,
                        'duration_points': phase_info.get('duration_points', 0),
                        'error': phase_prediction['error']
                    }
            
            # Add corner-level summary
            if corner_predictions['corner_summary']['phases_with_predictions'] > 0:
                all_predictions['summary']['corners_with_predictions'] += 1
                
                # Calculate average confidence for this corner
                confidences = []
                for phase_data in corner_predictions['phases'].values():
                    if 'confidence' in phase_data:
                        confidences.append(phase_data['confidence'])
                
                if confidences:
                    corner_predictions['corner_summary']['average_confidence'] = np.mean(confidences)
            else:
                all_predictions['summary']['corners_without_predictions'].append(corner_name)
            
            all_predictions['corner_predictions'][corner_name] = corner_predictions
        
        # Add overall summary statistics
        all_predictions['summary']['prediction_coverage'] = (
            all_predictions['summary']['corners_with_predictions'] / all_predictions['total_corners']
            if all_predictions['total_corners'] > 0 else 0.0
        )

        return all_predictions
    
    def _convert_corner_analysis_to_definitions(self, corner_analysis_result: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        """
        Convert corner analysis result to corner definitions format expected by the learner
        
        Args:
            corner_analysis_result: Result from track cornering analysis service
        
        Returns:
            Dictionary in the format expected by CornerSpecificLearner
        """
        corner_definitions = {}
        
        if 'corner_details' not in corner_analysis_result:
            print("[WARNING] No corner_details found in analysis result")
            return corner_definitions
        
        corner_details = corner_analysis_result['corner_details']
        
        for corner_name, corner_info in corner_details.items():
            if 'phases' not in corner_info:
                continue
                
            phases = corner_info['phases']
            corner_phases = {}
            
            # Extract all phase positions regardless of duration_points
            for phase_name, phase_info in phases.items():
                position = phase_info.get('normalized_car_position', 0.0)
                corner_phases[phase_name] = float(position)
            
            # Add corner if it has any phases
            if len(corner_phases) > 0:
                corner_definitions[corner_name] = corner_phases
                print(f"[INFO] Added {corner_name} with phases: {list(corner_phases.keys())}")
            else:
                print(f"[WARNING] Skipping {corner_name} - no phase data found")
        
        print(f"[INFO] Converted {len(corner_definitions)} corners from analysis result")
        return corner_definitions
    
    def _filter_top_performance_laps(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter for valid laps and select top 1% fastest laps for training
        
        Args:
            df: Processed telemetry DataFrame
            
        Returns:
            Filtered DataFrame containing only top 1% fastest valid laps
        """
        print(f"[INFO] Starting lap filtering from {len(df)} telemetry records")
        
        # We'll work with all data first, then filter by validity percentage per lap
        working_df = df.copy()
        
        # Check if we have the required columns
        has_valid_lap_column = 'Graphics_is_valid_lap' in working_df.columns
        if not has_valid_lap_column:
            print("[WARNING] Graphics_is_valid_lap column not found, cannot validate lap quality - returning all data")
            return working_df
        else:
            print(f"[INFO] Found Graphics_is_valid_lap column, will filter laps by validity percentage")
        
        # Group by lap and calculate lap times
        # Use both Graphics_completed_lap and Graphics_normalized_car_position together for robust lap detection
        has_completed_lap = 'Graphics_completed_lap' in working_df.columns
        has_position = 'Graphics_normalized_car_position' in working_df.columns
        
        # Only proceed if we have both fields - return empty data otherwise
        if not (has_completed_lap and has_position):
            print("[WARNING] Lap filtering requires both Graphics_completed_lap and Graphics_normalized_car_position - returning empty DataFrame")
            return pd.DataFrame()
        
        # Use both completed_lap counter and position data for most accurate lap detection
        print("[INFO] Using both Graphics_completed_lap and Graphics_normalized_car_position for lap detection")
        
        completed_laps = working_df['Graphics_completed_lap'].fillna(0)
        position = working_df['Graphics_normalized_car_position'].fillna(0)
        
        # Primary method: detect when completed_lap increments (official lap completion)
        completed_lap_changes = completed_laps.diff() > 0
        
        # Use completed_lap changes as primary lap boundary indicator
        lap_boundaries = completed_lap_changes 
        
        # Create cumulative lap ID
        working_df['lap_id'] = lap_boundaries.cumsum()
        
        # Group all telemetry data by these lap ids, allowing the code to process each lap individually
        lap_groups = working_df.groupby('lap_id')
        print(f"[INFO] Detected {len(lap_groups)} individual lap segments using completed_lap changes")
        
        # Calculate lap times for each lap
        lap_times = []
        lap_data = []
        total_laps_processed = 0
        full_laps_found = 0
        
        for lap_id, lap_df in lap_groups:
            total_laps_processed += 1
            
            if len(lap_df) < 10:  # Skip very short laps (likely incomplete)
                continue
            
            # Check validity percentage if is_valid_lap column is available
            if has_valid_lap_column:
                if not self._is_lap_mostly_valid(lap_df, 0.95):  # Require at least 95% valid points
                    continue
            
            # Validate that this is a full lap using normalized_car_position
            if not self._is_full_lap(lap_df):
                continue
            
            full_laps_found += 1
                
            # Calculate lap time
            if 'Graphics_current_time' in lap_df.columns:
                # Use the current lap time at the end of this lap (already in milliseconds)
                lap_time = lap_df['Graphics_current_time'].iloc[-1] / 1000.0  # Convert to seconds
            
            if lap_time > 0:  # Only include laps with valid times
                lap_times.append(lap_time)
                lap_data.append(lap_df)
        
        if not lap_times:
            print(f"[WARNING] No valid full lap times found out of {total_laps_processed} processed laps, returning empty DataFrame")
            return pd.DataFrame()
        
        print(f"[INFO] Processed {total_laps_processed} potential laps")
        if has_valid_lap_column:
            print(f"[INFO] Found {full_laps_found} complete full laps with ≥95% valid data points")
        else:
            print(f"[INFO] Found {full_laps_found} complete full laps (validity checking skipped)")
        print(f"[INFO] Calculated lap times for {len(lap_times)} qualifying laps")
        print(f"[INFO] Best lap time: {min(lap_times):.3f}s, Worst: {max(lap_times):.3f}s")
        
        # Sort laps by time (fastest first)
        sorted_indices = np.argsort(lap_times)

        # Calculate how many laps to keep (top 5%, minimum 1 lap)
        num_laps_to_keep = max(1, int(np.ceil(len(lap_times) * 0.05)))
        print(f"[INFO] Selecting top {num_laps_to_keep} fastest laps out of {len(lap_times)} total laps")
        
        # Select top laps
        top_lap_indices = sorted_indices[:num_laps_to_keep]
        
        # Combine data from selected laps
        filtered_data_frames = [lap_data[i] for i in top_lap_indices]
        filtered_df = pd.concat(filtered_data_frames, ignore_index=True)
        
        # Report selected lap times
        selected_lap_times = [lap_times[i] for i in top_lap_indices]
        print(f"[INFO] Selected lap times: {[f'{t:.3f}s' for t in selected_lap_times]}")
        print(f"[INFO] Filtered to {len(filtered_df)} records from top {num_laps_to_keep} fastest complete full laps")
        
        return filtered_df
    
    def _is_lap_mostly_valid(self, lap_df: pd.DataFrame, min_valid_percentage: float = 0.75) -> bool:
        """
        Check if a lap has a sufficient percentage of valid data points
        
        Args:
            lap_df: DataFrame containing telemetry data for one lap
            min_valid_percentage: Minimum percentage of valid points (default 75% in decimal)
            
        Returns:
            True if the lap has enough valid data points
        """
        if 'Graphics_is_valid_lap' not in lap_df.columns:
            return True  # Assume valid if we can't check
        
        valid_points = lap_df['Graphics_is_valid_lap'].fillna(False)
        total_points = len(valid_points)
        
        if total_points == 0:
            return False
        
        # Count boolean True values, handling different data types
        if valid_points.dtype == 'bool':
            valid_count = valid_points.sum()
        else:
            # Handle string or numeric representations
            valid_count = (
                (valid_points == True) | 
                (valid_points == 'True') | 
                (valid_points == 'true') | 
                (valid_points == 1) | 
                (valid_points == '1')
            ).sum()
        
        valid_percentage = valid_count / total_points
        
        if valid_percentage < min_valid_percentage:
            print(f"[DEBUG] Rejected lap: only {valid_percentage:.1%} valid points (need {min_valid_percentage:.1%})")
            return False
        
        return True
    
    def _is_full_lap(self, lap_df: pd.DataFrame) -> bool:
        """
        Validate that a lap contains a complete track progression from start to finish
        Uses both Graphics_normalized_car_position and Graphics_completed_lap when available
        
        Args:
            lap_df: DataFrame containing telemetry data for one lap
            
        Returns:
            True if the lap contains progression from ~0 to ~1 in normalized_car_position
            and shows consistent completed lap counter behavior
        """
        has_position = 'Graphics_normalized_car_position' in lap_df.columns
        has_completed_lap = 'Graphics_completed_lap' in lap_df.columns
        
        # If we have neither field, assume valid (fallback)
        if not has_position and not has_completed_lap:
            print("[WARNING] No position or completed lap data available, cannot validate full lap")
            return True
        
        # Validate using normalized car position (primary validation)
        position_valid = True
        if has_position:
            positions = lap_df['Graphics_normalized_car_position'].dropna()
            
            if len(positions) == 0:
                position_valid = False
            else:
                min_position = positions.min()
                max_position = positions.max()
                
                # Check if the lap covers most of the track
                # Allow some tolerance: lap should go from close to 0 to close to 1
                starts_near_beginning = min_position <= 0.15  # Starts at or before 15% of track
                ends_near_finish = max_position >= 0.85       # Ends at or after 85% of track
                
                # Additional check: ensure good coverage of the track
                position_range = max_position - min_position
                good_coverage = position_range >= 0.7  # Covers at least 70% of track length
                
                position_valid = starts_near_beginning and ends_near_finish and good_coverage
                
                if not position_valid:
                    print(f"[DEBUG] Position validation failed: min_pos={min_position:.3f}, max_pos={max_position:.3f}, range={position_range:.3f}")
        
        # Validate using completed lap counter (secondary validation)
        completed_lap_valid = True
        if has_completed_lap:
            completed_laps = lap_df['Graphics_completed_lap'].fillna(0)
            
            # For a valid lap, the completed lap counter should either:
            # 1. Stay constant throughout the lap (during lap progress)
            # 2. Show exactly one increment at the end (lap completion)
            unique_values = completed_laps.unique()
            
            if len(unique_values) == 1:
                # Counter stayed constant - lap in progress, this is expected
                completed_lap_valid = True
            elif len(unique_values) == 2:
                # Counter incremented once - should be at the end of the lap
                # Check that the increment happens towards the end of the data
                increment_positions = completed_laps.diff() > 0
                if increment_positions.sum() == 1:  # Exactly one increment
                    # Find where the increment occurred
                    increment_index = increment_positions.idxmax()
                    total_records = len(completed_laps)
                    increment_position_ratio = (increment_index / total_records) if total_records > 0 else 0
                    
                    # Increment should happen in the latter part of the lap (after 70% completion)
                    completed_lap_valid = increment_position_ratio >= 0.7
                    if not completed_lap_valid:
                        print(f"[DEBUG] Completed lap increment too early: {increment_position_ratio:.1%} through lap")
                else:
                    # Multiple increments - suspicious
                    completed_lap_valid = False
                    print(f"[DEBUG] Multiple completed lap increments detected: {increment_positions.sum()}")
            else:
                # Too many different values - suspicious
                completed_lap_valid = False
                print(f"[DEBUG] Too many completed lap values: {len(unique_values)} unique values")
        
        # Combine validations - both must pass if both fields are available
        if has_position and has_completed_lap:
            is_valid = position_valid and completed_lap_valid
            if not is_valid:
                print(f"[DEBUG] Rejected lap: position_valid={position_valid}, completed_lap_valid={completed_lap_valid}")
        elif has_position:
            is_valid = position_valid
        else:  # has_completed_lap only
            is_valid = completed_lap_valid
        
        return is_valid
    
    def serialize_object_inside(self, results: any) -> Dict[str, Any]:
        """
        Serialize corner learning models and related objects
        
        Args:
            results: Dictionary containing corner learning results
            
        Returns:
            Dictionary with serialized models
        """
        # Serialize corner learning models if present
        if 'corner_models' in results:
            print("[INFO] Serializing corner learning models...")
            serialized_corner_models = {}
            
            for corner_name, corner_model in results.get('corner_models', []).items():
                print(f"[INFO] Serializing corner model: {corner_name}")
                try:
                    # Serialize to bytes using pickle
                    buffer = io.BytesIO()
                    pickle.dump(corner_model, buffer)
                    buffer.seek(0)
                    
                    # Encode to base64
                    encoded_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
                    serialized_corner_models[corner_name] = encoded_data
                    
                except Exception as e:
                    print(f"[ERROR] Failed to serialize corner model {corner_name}: {e}")
                    raise e
                
            results['corner_models'] = serialized_corner_models
            
        return results
    
    def receive_serialized_model_data(self, serialized_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deserialize corner learning models that were serialized by serialize_object_inside, if contained corner models, reconstruct the CornerSpecificLearner instance.

        Args:
            serialized_results: Dictionary containing serialized models and metadata
            
        Returns:
            Dictionary with deserialized models and original structure
        """
        results = serialized_data.copy()
        
        # Deserialize corner learning models if present
        if 'corner_models' in results:
            print("[INFO] Deserializing corner learning models...")
            deserialized_corner_models = {}

            for corner_name, serialized_model in results.get('corner_models', []).items():
                print(f"[INFO] Deserializing corner model: {corner_name}")
                try:
                    # Decode from base64
                    decoded_data = base64.b64decode(serialized_model.encode('utf-8'))
                    
                    # Deserialize using pickle
                    buffer = io.BytesIO(decoded_data)
                    deserialized_model = pickle.load(buffer)
                    deserialized_corner_models[corner_name] = deserialized_model
                    results['corner_models'] = deserialized_corner_models
                except Exception as e:
                    print(f"[ERROR] Failed to deserialize corner model {corner_name}: {e}")
                    raise Exception(f"Failed to deserialize corner learning model {corner_name}: {str(e)}")

            self.corner_learner = CornerSpecificLearner(results.get('corner_models', {}))
        return results
    
    def _generate_learning_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a summary of corner learning results"""
        summary = {
            'timestamp': datetime.now().isoformat(),
            'learning_completed': ['corner_specific']
        }
        
        if 'corner_models' in results:
            corner_info = results
            summary['corner_summary'] = {
                'corners_trained': results.get('corners_trained', 0),
                'total_phases': results.get('total_phases', 0),
                'corner_names': list(results.get('corner_models', {}).keys())
            }
        
        return summary