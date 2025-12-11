"""
Imitation Learning Service for Assetto Corsa Competizione Telemetry Analysis

This service implements imitation learning algorithms to learn from expert driving demonstrations.
It focuses on learning optimal racing lines and decision-making patterns from
professional or expert drivers' telemetry data.
"""

import numpy as np
import pandas as pd
import pickle
import warnings
import io
import base64
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime
from pathlib import Path

# Scikit-learn imports for trajectory learning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, f1_score
from sklearn.decomposition import PCA

# Import your telemetry models
from ..models.telemetry_models import TelemetryFeatures, FeatureProcessor
from .tire_grip_analysis_service import TireGripFeatureCatalog

warnings.filterwarnings('ignore', category=UserWarning)

class ExpertFeatureCatalog:
    """Canonical expert feature names for downstream models.
    All expert state feature keys must be declared here and referenced via the Enum
    to avoid drifting string literals across the codebase.
    """

    class ExpertOptimalFeature(str, Enum):
        # Optimal action predictions 
        EXPERT_OPTIMAL_SPEED = 'expert_optimal_speed'
        EXPERT_OPTIMAL_STEERING = 'expert_optimal_steering'
        EXPERT_OPTIMAL_THROTTLE = 'expert_optimal_throttle'
        EXPERT_OPTIMAL_BRAKE = 'expert_optimal_brake'
        EXPERT_OPTIMAL_GEAR = 'expert_optimal_gear'
        EXPERT_OPTIMAL_PLAYER_POS_X = 'expert_optimal_player_pos_x'
        EXPERT_OPTIMAL_PLAYER_POS_Y = 'expert_optimal_player_pos_y'
        EXPERT_OPTIMAL_PLAYER_POS_Z = 'expert_optimal_player_pos_z'
        EXPERT_OPTIMAL_TRACK_POSITION = 'expert_optimal_track_position'
        EXPERT_OPTIMAL_VELOCITY_X = 'expert_optimal_velocity_x'
        EXPERT_OPTIMAL_VELOCITY_Y = 'expert_optimal_velocity_y'
        EXPERT_OPTIMAL_VELOCITY_Z = 'expert_optimal_velocity_z'

    class ContextFeature(str, Enum):
        # Velocity direction alignment with expert
        EXPERT_VELOCITY_ALIGNMENT = 'expert_velocity_alignment' # 1.0 if moving in the expert velocity direction, 0.0 opposite direction
        SPEED_DIFFERENCE = 'speed_difference' # Difference between current speed and expert optimal speed (km/h)
        DISTANCE_TO_EXPERT_LINE = 'distance_to_expert_line' # distance between current position and expert optimal racing line (meters)


    class ExpertFeatures (str, Enum):
        # Optimal action predictions 
        EXPERT_OPTIMAL_PLAYER_POS_X = 'expert_optimal_player_pos_x'
        EXPERT_OPTIMAL_PLAYER_POS_Y = 'expert_optimal_player_pos_y'
        EXPERT_OPTIMAL_PLAYER_POS_Z = 'expert_optimal_player_pos_z'
        EXPERT_OPTIMAL_SPEED = 'expert_optimal_speed'
        EXPERT_OPTIMAL_THROTTLE = 'expert_optimal_throttle'
        EXPERT_OPTIMAL_BRAKE = 'expert_optimal_brake'
        # Context features
        EXPERT_VELOCITY_ALIGNMENT = 'expert_velocity_alignment'
        SPEED_DIFFERENCE = 'speed_difference' 
        DISTANCE_TO_EXPERT_LINE = 'distance_to_expert_line'

    # Flat list for convenience (now only expert optimal + derived)
    CONTEXT_FEATURES: List[str] = [f.value for f in ContextFeature]
    

    
@dataclass(frozen=True)
class SegmentImprovementConfig:
    """Centralized thresholds and heuristics used during segment improvement analysis."""

    expert_velocity_alignment: float = 0.9
    expert_speed_diff_max: float = 5.0
    expert_distance_max: float = 5.0

    driver_push_high_threshold: float = 0.4
    driver_push_trend_min: float = 0.01

    smoothing_window_min: int = 2
    smoothing_window_max: int = 5
    ema_span_min: int = 2


@dataclass
class SegmentImprovementSummary:
    """Structured container for telemetry segment improvement analysis results."""

    velocity_alignment_mean: float = 0.0
    velocity_alignment_trend: float = 0.0
    velocity_consistency_rate: float = 0.0
    velocity_expert_points: int = 0

    speed_difference_mean: float = 0.0
    speed_difference_trend: float = 0.0
    speed_consistency_rate: float = 0.0
    speed_expert_points: int = 0
    distance_to_line_mean: float = 0.0
    distance_to_line_trend: float = 0.0
    distance_consistency_rate: float = 0.0
    distance_expert_points: int = 0

    driver_push_available: bool = False
    driver_push_mean: float = 0.0
    driver_push_trend: float = 0.0
    driver_push_high_rate: float = 0.0

    overall_improvement_rate: float = 0.0
    overall_consistency_rate: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def get(self, item: str, default: Any = None) -> Any:
        return getattr(self, item, default)

    def __getitem__(self, item: str) -> Any:
        return getattr(self, item)


class ExpertPositionLearner:
    """Learn expert actions based on normalized track position from multiple expert laps, per track."""
    
    def __init__(self):
        self.track_models = {} # Dictionary to store models per track: {track_name: model_data}
    
    def extract_position_features(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Extract position-based features for expert learning
        
        Args:
            df: Telemetry DataFrame
            
        Returns:
            Dictionary with input features (normalized position, track) and target features (expert actions/states)
        """
        input_features = pd.DataFrame()
        target_features = pd.DataFrame()
        
        # Input feature: normalized track position (primary input)
        if 'Graphics_normalized_car_position' in df.columns:
            input_features['normalized_position'] = df['Graphics_normalized_car_position']
        else:
            raise ValueError("Graphics_normalized_car_position not found - this is required for position-based learning")

        # Input feature: track name
        if 'Static_track' in df.columns:
            input_features['track'] = df['Static_track']
        else:
            raise ValueError("Static_track not found - this is required for multi-track learning")
        
        # Target features: Expert actions and states
        EO = ExpertFeatureCatalog.ExpertOptimalFeature
        
        # Actions
        if 'Physics_steer_angle' in df.columns:
            target_features[EO.EXPERT_OPTIMAL_STEERING.value] = df['Physics_steer_angle']
        if 'Physics_gas' in df.columns:
            target_features[EO.EXPERT_OPTIMAL_THROTTLE.value] = df['Physics_gas']
        if 'Physics_brake' in df.columns:
            target_features[EO.EXPERT_OPTIMAL_BRAKE.value] = df['Physics_brake']
        if 'Physics_gear' in df.columns:
            target_features[EO.EXPERT_OPTIMAL_GEAR.value] = df['Physics_gear']
        
        # Positions
        if 'Graphics_player_pos_x' in df.columns:
            target_features[EO.EXPERT_OPTIMAL_PLAYER_POS_X.value] = df['Graphics_player_pos_x']
        if 'Graphics_player_pos_y' in df.columns:
            target_features[EO.EXPERT_OPTIMAL_PLAYER_POS_Y.value] = df['Graphics_player_pos_y']
        if 'Graphics_player_pos_z' in df.columns:
            target_features[EO.EXPERT_OPTIMAL_PLAYER_POS_Z.value] = df['Graphics_player_pos_z']
            
        # Velocities
        if 'Physics_velocity_x' in df.columns:
            target_features[EO.EXPERT_OPTIMAL_VELOCITY_X.value] = df['Physics_velocity_x']
        if 'Physics_velocity_y' in df.columns:
            target_features[EO.EXPERT_OPTIMAL_VELOCITY_Y.value] = df['Physics_velocity_y']
        if 'Physics_velocity_z' in df.columns:
            target_features[EO.EXPERT_OPTIMAL_VELOCITY_Z.value] = df['Physics_velocity_z']
            
        # Speed (derived)
        if 'Physics_speed_kmh' in df.columns:
            target_features[EO.EXPERT_OPTIMAL_SPEED.value] = df['Physics_speed_kmh']
            
        # Track position (for consistency)
        if 'Graphics_normalized_car_position' in df.columns:
            target_features[EO.EXPERT_OPTIMAL_TRACK_POSITION.value] = df['Graphics_normalized_car_position']
        
        # Clean data - Drop rows with missing values instead of filling with 0
        # Filling with 0 creates massive outliers for position coordinates (0,0,0)
        # which ruins the model training if there are any gaps in telemetry.
        
        # Combine to ensure we drop rows consistently
        combined = pd.concat([input_features, target_features], axis=1)
        
        # Drop rows where any feature is NaN
        combined_clean = combined.dropna()
        
        if len(combined) != len(combined_clean):
            print(f"[WARNING] Dropped {len(combined) - len(combined_clean)} rows with missing values during feature extraction")
            
        # Split back into input and target
        input_cols = input_features.columns
        target_cols = target_features.columns
        
        input_features = combined_clean[input_cols]
        target_features = combined_clean[target_cols]
        
        return {
            'input_features': input_features,
            'target_features': target_features
        }
    
    def learn_expert_position_mapping(self, expert_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Learn expert actions from normalized track position using multiple expert laps, per track.
        
        Args:
            expert_df: Expert driver telemetry data from multiple laps
            
        Returns:
            Dictionary with learned position-based model and insights
        """
        print(f"[INFO] Learning expert position mapping from {len(expert_df)} expert data points")
        
        # Extract position-based features
        feature_data = self.extract_position_features(expert_df)
        input_features = feature_data['input_features']
        target_features = feature_data['target_features']
        
        if len(input_features) == 0:
            raise ValueError("No input features extracted")
        if len(target_features.columns) == 0:
            raise ValueError("No target features extracted")
            
        print(f"[INFO] Input features shape: {input_features.shape}")
        print(f"[INFO] Target features shape: {target_features.shape}")
        print(f"[INFO] Available targets: {list(target_features.columns)}")
        
        # Get track name (assume single track as per requirement)
        if 'track' not in input_features.columns or input_features['track'].empty:
             raise ValueError("Track information missing in input features")
             
        track = input_features['track'].iloc[0]
        print(f"[INFO] Training models for track: {track}")
        
        overall_metrics = {}
        
        # Prepare input (normalized position)
        X = input_features[['normalized_position']].values
        
        # Create and fit scaler for this track
        position_scaler = StandardScaler()
        X_scaled = position_scaler.fit_transform(X)
        
        # Train models for each target
        models = {}
        performance_metrics = {}
        
        EO = ExpertFeatureCatalog.ExpertOptimalFeature
        
        # Action models (regression)
        action_targets = [
            EO.EXPERT_OPTIMAL_STEERING.value,
            EO.EXPERT_OPTIMAL_THROTTLE.value,
            EO.EXPERT_OPTIMAL_BRAKE.value,
            EO.EXPERT_OPTIMAL_SPEED.value
        ]
        
        for target_name in action_targets:
            if target_name in target_features.columns:
                # print(f"[INFO] Training action model for: {target_name}")
                y = target_features[target_name].values
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X_scaled, y, test_size=0.2, random_state=42
                )
                
                # Train model
                model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
                model.fit(X_train, y_train)
                
                # Evaluate
                y_pred = model.predict(X_test)
                r2 = model.score(X_test, y_test)
                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                
                models[target_name] = model
                performance_metrics[target_name] = {
                    'r2': float(r2),
                    'mse': float(mse),
                    'mae': float(mae),
                    'type': 'regression'
                }
                
                # print(f"[INFO] {target_name} - R²: {r2:.4f}, MSE: {mse:.4f}, MAE: {mae:.4f}")
        
        # Gear model (classification)
        if EO.EXPERT_OPTIMAL_GEAR.value in target_features.columns:
            # print(f"[INFO] Training gear classification model")
            y = target_features[EO.EXPERT_OPTIMAL_GEAR.value].values
            
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42, stratify=y
            )
            
            model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            models[EO.EXPERT_OPTIMAL_GEAR.value] = model
            performance_metrics[EO.EXPERT_OPTIMAL_GEAR.value] = {
                'accuracy': float(accuracy),
                'f1': float(f1),
                'type': 'classification'
            }
            
            # print(f"[INFO] Gear - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
        
        # Position models (regression)
        position_targets = [
            EO.EXPERT_OPTIMAL_PLAYER_POS_X.value,
            EO.EXPERT_OPTIMAL_PLAYER_POS_Y.value,
            EO.EXPERT_OPTIMAL_PLAYER_POS_Z.value
        ]
        
        for target_name in position_targets:
            if target_name in target_features.columns:
                # print(f"[INFO] Training position model for: {target_name}")
                y = target_features[target_name].values
                
                X_train, X_test, y_train, y_test = train_test_split(
                    X_scaled, y, test_size=0.2, random_state=42
                )
                
                model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
                model.fit(X_train, y_train)
                
                y_pred = model.predict(X_test)
                r2 = model.score(X_test, y_test)
                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                
                models[target_name] = model
                performance_metrics[target_name] = {
                    'r2': float(r2),
                    'mse': float(mse),
                    'mae': float(mae),
                    'type': 'regression'
                }
                
                # print(f"[INFO] {target_name} - R²: {r2:.4f}, MSE: {mse:.4f}, MAE: {mae:.4f}")
        
        # Velocity models (regression)
        velocity_targets = [
            EO.EXPERT_OPTIMAL_VELOCITY_X.value,
            EO.EXPERT_OPTIMAL_VELOCITY_Y.value,
            EO.EXPERT_OPTIMAL_VELOCITY_Z.value
        ]
        
        for target_name in velocity_targets:
            if target_name in target_features.columns:
                # print(f"[INFO] Training velocity model for: {target_name}")
                y = target_features[target_name].values
                
                X_train, X_test, y_train, y_test = train_test_split(
                    X_scaled, y, test_size=0.2, random_state=42
                )
                
                model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
                model.fit(X_train, y_train)
                
                y_pred = model.predict(X_test)
                r2 = model.score(X_test, y_test)
                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                
                models[target_name] = model
                performance_metrics[target_name] = {
                    'r2': float(r2),
                    'mse': float(mse),
                    'mae': float(mae),
                    'type': 'regression'
                }
                
                # print(f"[INFO] {target_name} - R²: {r2:.4f}, MSE: {mse:.4f}, MAE: {mae:.4f}")
        
        # Track position model (for consistency)
        if EO.EXPERT_OPTIMAL_TRACK_POSITION.value in target_features.columns:
            # print(f"[INFO] Training track position model")
            y = target_features[EO.EXPERT_OPTIMAL_TRACK_POSITION.value].values
            
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42
            )
            
            model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_test)
            r2 = model.score(X_test, y_test)
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            
            models[EO.EXPERT_OPTIMAL_TRACK_POSITION.value] = model
            performance_metrics[EO.EXPERT_OPTIMAL_TRACK_POSITION.value] = {
                'r2': float(r2),
                'mse': float(mse),
                'mae': float(mae),
                'type': 'regression'
            }
            
            # print(f"[INFO] Track position - R²: {r2:.4f}, MSE: {mse:.4f}, MAE: {mae:.4f}")
        
        # Store the complete position model for this track
        self.track_models[track] = {
            'models': models,
            'position_scaler': position_scaler,
            'performance_metrics': performance_metrics,
            'input_features': ['normalized_position'],
            'target_features': list(target_features.columns)
        }
        
        overall_metrics[track] = performance_metrics
        
        return {
            'modelData': self.track_models,
            'metadata': {
                'performance_metrics': overall_metrics,
                'input_features': ['normalized_position', 'track'],
                'target_features': list(target_features.columns),
                'models_trained': list(self.track_models.keys()),
                'total_training_samples': len(expert_df)
            }
        }
    
    def predict_expert_actions_at_position(self, track_name: str, normalized_positions: Union[float, List[float], np.ndarray]) -> Union[Dict[str, float], List[Dict[str, float]]]:
        """
        Predict expert actions at given normalized track position(s) for a specific track.
        
        Args:
            track_name: Name of the track to predict for
            normalized_positions: Single position or array of positions (0.0 to 1.0)
            
        Returns:
            Dictionary with expert predictions, or list of dictionaries for multiple positions
        """
        if not self.track_models:
            raise ValueError("No track models trained. Call learn_expert_position_mapping() first.")
        
        if track_name not in self.track_models:
            raise ValueError(f"No model trained for track: {track_name}")
            
        track_model_data = self.track_models[track_name]
        
        # Handle single position vs multiple positions
        single_position = isinstance(normalized_positions, (int, float))
        if single_position:
            positions_array = np.array([[normalized_positions]])
        else:
            positions_array = np.array(normalized_positions).reshape(-1, 1)
        
        # Scale positions using track-specific scaler
        positions_scaled = track_model_data['position_scaler'].transform(positions_array)
        
        # Make predictions for all models
        predictions = {}
        for model_name, model in track_model_data['models'].items():
            pred = model.predict(positions_scaled)
            predictions[model_name] = pred
        
        # Format results
        if single_position:
            # Return single dictionary
            result = {}
            for model_name, pred_array in predictions.items():
                result[model_name] = float(pred_array[0])
            return result
        else:
            # Return list of dictionaries
            results = []
            for i in range(len(positions_array)):
                result = {}
                for model_name, pred_array in predictions.items():
                    result[model_name] = float(pred_array[i])
                results.append(result)
            return results 


    def debug_position_model(self) -> Dict[str, Any]:
        """
        Debug method to inspect the current position model state
        
        Returns:
            Dictionary with detailed model debugging information
        """
        if not self.track_models:
            return {
                'status': 'No models available',
                'has_model': False,
                'error': 'Position models not trained yet'
            }
        
        debug_info = {
            'status': 'Models available',
            'has_model': True,
            'tracks': list(self.track_models.keys()),
            'track_details': {}
        }
        
        for track, model_data in self.track_models.items():
            track_info = {}
            # Check model structure
            for key, value in model_data.items():
                if key == 'models':
                    track_info['models'] = {
                        'count': len(value),
                        'model_names': list(value.keys()),
                        'model_types': [type(model).__name__ for model in value.values()]
                    }
                elif key == 'position_scaler':
                    track_info['position_scaler'] = {
                        'type': type(value).__name__,
                        'fitted': hasattr(value, 'mean_')
                    }
                elif key == 'performance_metrics':
                    track_info['performance_metrics'] = {
                        'available_metrics': list(value.keys()),
                        'metric_count': len(value)
                    }
                else:
                    track_info[key] = {
                        'type': type(value).__name__,
                        'value': str(value) if not isinstance(value, (list, dict)) else f"{type(value).__name__} with {len(value)} items"
                    }
            debug_info['track_details'][track] = track_info
        
        return debug_info
    
    def validate_position_input(self, track_name: str, normalized_positions: Union[float, List[float], np.ndarray]) -> Dict[str, Any]:
        """
        Validate normalized position input for prediction
        
        Args:
            track_name: Track to validate for
            normalized_positions: Position(s) to validate
            
        Returns:
            Dictionary with validation results
        """
        validation_results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'input_analysis': {}
        }
        
        # Check if model exists
        if not self.track_models:
            validation_results['valid'] = False
            validation_results['errors'].append("No position models trained")
            return validation_results
            
        if track_name not in self.track_models:
            validation_results['valid'] = False
            validation_results['errors'].append(f"No model for track: {track_name}")
            return validation_results
        
        # Convert input to array for analysis
        if isinstance(normalized_positions, (int, float)):
            positions_array = np.array([normalized_positions])
        else:
            positions_array = np.array(normalized_positions).flatten()
        
        # Analyze input data
        validation_results['input_analysis'] = {
            'shape': positions_array.shape,
            'min_value': float(np.min(positions_array)),
            'max_value': float(np.max(positions_array)),
            'mean_value': float(np.mean(positions_array)),
            'has_nan': bool(np.isnan(positions_array).any()),
            'has_inf': bool(np.isinf(positions_array).any())
        }
        
        # Check for problematic values
        if np.isnan(positions_array).any():
            validation_results['valid'] = False
            validation_results['errors'].append("Input contains NaN values")
            
        if np.isinf(positions_array).any():
            validation_results['valid'] = False
            validation_results['errors'].append("Input contains infinite values")
        
        # Check position range (should be 0.0 to 1.0 for normalized positions)
        if np.any(positions_array < 0.0):
            validation_results['warnings'].append("Some positions are below 0.0 (not normalized)")
            
        if np.any(positions_array > 1.0):
            validation_results['warnings'].append("Some positions are above 1.0 (not normalized)")
        
        return validation_results
    
    def validate_prediction_input(self, current_state: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate input data for prediction without actually making predictions
        
        Args:
            current_state: Current telemetry state to validate
            
        Returns:
            Dictionary with validation results
        """
        validation_results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'input_analysis': {},
            'feature_analysis': {}
        }
        
        # Check if model exists
        if not self.trajectory_model:
            validation_results['valid'] = False
            validation_results['errors'].append("No trajectory model trained")
            return validation_results
        
        # Analyze input data
        validation_results['input_analysis'] = {
            'shape': current_state.shape,
            'columns': list(current_state.columns),
            'dtypes': current_state.dtypes.to_dict(),
            'missing_values': current_state.isnull().sum().to_dict(),
            'infinite_values': np.isinf(current_state.select_dtypes(include=[np.number])).sum().to_dict()
        }
        
        # Check for problematic values
        numeric_cols = current_state.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if current_state[col].isnull().sum() > 0:
                validation_results['warnings'].append(f"Column {col} has {current_state[col].isnull().sum()} missing values")
            if np.isinf(current_state[col]).sum() > 0:
                validation_results['warnings'].append(f"Column {col} has {np.isinf(current_state[col]).sum()} infinite values")
        
        current_state
        try:
            # Extract features
            trajectory_features = self.extract_trajectory_features(current_state)
            
            validation_results['feature_analysis'] = {
                'extracted_features_count': trajectory_features.shape[1],
                'extracted_features': list(trajectory_features.columns),
                'required_features': self.trajectory_model['input_features']
            }
            
            # Check for missing features
            required_features = self.trajectory_model['input_features']
            missing_features = [f for f in required_features if f not in trajectory_features.columns]
            
            if missing_features:
                validation_results['valid'] = False
                validation_results['errors'].append(f"Missing required features: {missing_features}")
            
            # Check feature data quality
            feature_subset = trajectory_features[required_features].fillna(0)
            
            for feature in required_features:
                if feature in trajectory_features.columns:
                    feature_data = trajectory_features[feature]
                    if feature_data.isnull().all():
                        validation_results['warnings'].append(f"Feature {feature} is all null values")
                    elif np.isinf(feature_data).any():
                        validation_results['warnings'].append(f"Feature {feature} contains infinite values")
                    elif feature_data.std() == 0:
                        validation_results['warnings'].append(f"Feature {feature} has zero variance")
            
        except Exception as e:
            validation_results['valid'] = False
            validation_results['errors'].append(f"Feature extraction failed: {str(e)}")
            raise Exception(f"Feature extraction failed: {str(e)}")
        
        return validation_results


class ExpertImitateLearningService:
    """Main imitation learning service that focuses on trajectory optimization"""
    
    def __init__(self, models_directory: str = "imitation_models"):
        """
        Initialize the imitation learning service
        
        Args:
            models_directory: Directory to save/load trained imitation models
        """
        self.models_directory = Path(models_directory)
        self.models_directory.mkdir(exist_ok=True)
        
        self.position_learner = ExpertPositionLearner()
        
        print(f"[INFO] ImitationLearningService initialized. Models directory: {self.models_directory}")
    
    def get_shared_data_cache(self):
        """Get shared data cache instance"""
        from .zarr_telemetry_store import get_shared_zarr_store
        return get_shared_zarr_store()

    async def train_ai_model(self, top_laps_cache_key: str) -> Dict[str, Any]:
        """
        Learn from expert driving demonstrations using cached top laps.
        
        Args:
            top_laps_cache_key: Cache key for top laps data
            
        Returns:
            Dictionary with trained models and learning insights, serialized objects and ready for storage
        """
        
        print(f"[INFO {self.__class__.__name__}] Learning from cached top laps: {top_laps_cache_key}")

        telemetry_store = self.get_shared_data_cache()
        if not telemetry_store.has_cached_data(top_laps_cache_key):
             raise ValueError(f"No cached top laps found at key: {top_laps_cache_key}")

        chunks_iterator = telemetry_store.get_cached_data_chunks(cache_key=top_laps_cache_key, include_ids=True)
        
        total_samples = 0
        
        for chunk_tuple in chunks_iterator:
            chunk_data, chunk_id = chunk_tuple
            
            if not chunk_data:
                continue

            print(f"[INFO] Processing top laps for track: {chunk_id} ({len(chunk_data)} records)")
            
            processed_df = pd.DataFrame(chunk_data)
            
            # Learn expert position mapping (this is the only learning model)
            self.position_learner.learn_expert_position_mapping(processed_df)
            total_samples += len(processed_df)
        
        # Construct results
        overall_metrics = {}
        all_targets = set()
        for track, model_data in self.position_learner.track_models.items():
            if 'performance_metrics' in model_data:
                overall_metrics[track] = model_data['performance_metrics']
            if 'target_features' in model_data:
                 all_targets.update(model_data['target_features'])

        results = {
            'modelData': self.position_learner.track_models,
            'metadata': {
                'performance_metrics': overall_metrics,
                'input_features': ['normalized_position', 'track'],
                'target_features': list(all_targets),
                'models_trained': list(self.position_learner.track_models.keys()),
                'total_training_samples': total_samples
            }
        }

        results['learning_summary'] = self._generate_learning_summary(results)

        return results
    
    def predict_expert_actions(self, 
                             processed_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Predict what an expert would do in the current situation
        
        Args:
            processed_df: Current telemetry DataFrame (may be single row)
            
        Returns:
            Predicted expert actions and recommendations
        """
        predictions = {}
        
        # Check if models exist
        if not self.position_learner.track_models:
            print("[WARNING] No position models available")
            return {"error": "No trained models available"}
        
        if processed_df.empty:
            return {"error": "Empty input dataframe"}
        
        try:
            # Extract track name
            if 'Static_track' in processed_df.columns:
                # Assuming single track for prediction request
                track_name = processed_df['Static_track'].iloc[0]
            else:
                return {"error": "Static_track not found in input data"}
            
            # Extract normalized positions from the input data
            if 'Graphics_normalized_car_position' in processed_df.columns:
                normalized_positions = processed_df['Graphics_normalized_car_position'].values
                
                try:
                    optimal_actions = self.position_learner.predict_expert_actions_at_position(track_name, normalized_positions)
                except ValueError as e:
                    return {"error": str(e)}
                
                # If multiple rows, average the predictions for consistency with old interface
                if isinstance(optimal_actions, list):
                    averaged_actions = {}
                    for key in optimal_actions[0].keys():
                        averaged_actions[key] = np.mean([action[key] for action in optimal_actions])
                    predictions['optimal_actions'] = averaged_actions
                else:
                    predictions['optimal_actions'] = optimal_actions
            else:
                predictions['optimal_actions'] = {"error": "No normalized track position data available"}
                
        except Exception as e:
            raise Exception(f"[WARNING] Could not predict expert actions: {e}")
        
        # If no specific models are available, provide error
        if not predictions or all('error' in v for v in predictions.values() if isinstance(v, dict)):
            raise Exception("[Error] No valid model available for predictions")
        
        return predictions
    
    def _generate_learning_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a summary of learning results"""
        summary = {
            'timestamp': datetime.now().isoformat(),
            'learning_completed': []
        }
        
        # Check if we have metadata
        if 'metadata' in results:
            summary['learning_completed'].append('position_learning')
            
            # Calculate average performance metric, handling both regression (r2) and classification (accuracy) models
            # performance_metrics is now {track_name: {model_name: metrics}}
            all_performance_metrics = results['metadata']['performance_metrics']
            
            r2_scores = []
            accuracy_scores = []
            total_models = 0
            
            for track_metrics in all_performance_metrics.values():
                for metrics in track_metrics.values():
                    if 'r2' in metrics:
                        r2_scores.append(metrics['r2'])
                    if 'accuracy' in metrics:
                        accuracy_scores.append(metrics['accuracy'])
                    total_models += 1
            
            # Calculate average scores
            avg_r2 = np.mean(r2_scores) if r2_scores else 0.0
            avg_accuracy = np.mean(accuracy_scores) if accuracy_scores else 0.0
            
            summary['position_summary'] = {
                'models_trained': total_models,
                'tracks_trained': len(all_performance_metrics),
                'input_features': len(results['metadata']['input_features']),
                'target_features': len(results['metadata']['target_features']),
                'avg_r2_score': avg_r2,
                'avg_accuracy_score': avg_accuracy,
                'regression_models': len(r2_scores),
                'classification_models': len(accuracy_scores),
                'total_training_samples': results['metadata']['total_training_samples']
            }
        
        return summary
 
    def extract_expert_state_for_telemetry(self, telemetry_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract the comparsion between current state and expert optimal state. it helps transformer model to understand the gap between non-expert and expert driver.

        Purpose:
            - Provide a clear comparison between the current telemetry state and the expert's optimal state.
            - Enable the transformer model to learn from the differences and improve non-expert driving behavior.

        Args:
            telemetry_data: List of cleaned telemetry records to predict on

        Returns:
            List of dictionaries, one per record, containing expert targets, delta-to-expert
            context metrics, and enriched driver/expert positional data for visualization.
        """
        
        ExpertFeatures = ExpertFeatureCatalog.ExpertFeatures
        if not telemetry_data:
            return []
        if not self.position_learner.track_models:
            raise ValueError("No trained imitation models available. Train or load models before calling extract_expert_state_for_telemetry().")

        try:
            processed_df = pd.DataFrame(telemetry_data)
        except Exception as e:
            raise Exception(f"Failed to create DataFrame: {e}")

        expert_feature_rows: List[Dict[str, Any]] = []

        # Position models should already be loaded/deserialized
        if not self.position_learner.track_models:
            raise ValueError("Position model not loaded. Call train_ai_model() or deserialize_imitation_model() first.")

        def predict_expert_batch(batch_df: pd.DataFrame) -> List[Dict[str, float]]:
            """Predict expert actions for a batch of normalized positions - much faster than row-by-row"""
            if not self.position_learner.track_models:
                return [{} for _ in range(len(batch_df))]
            try:
                # Extract normalized positions from batch
                if 'Graphics_normalized_car_position' not in batch_df.columns:
                    raise ValueError("Graphics_normalized_car_position not found in batch data")
                
                if 'Static_track' not in batch_df.columns:
                    # If track is missing, we can't predict. Return empty.
                    return [{} for _ in range(len(batch_df))]
                
                # Assuming single track per batch as per optimization
                track_name = batch_df['Static_track'].iloc[0]
                
                if track_name not in self.position_learner.track_models:
                    return [{} for _ in range(len(batch_df))]
                    
                normalized_positions = batch_df['Graphics_normalized_car_position'].values
                return self.position_learner.predict_expert_actions_at_position(track_name, normalized_positions)

            except Exception as e:
                raise Exception(f"Batch prediction failed: {e}")

        total_rows = len(processed_df)

        # OPTIMIZATION: Process in batches instead of row-by-row for massive speedup
        batch_size = min(1000, total_rows)  # Process up to 1000 rows at once

        # Process all data in batches
        for batch_start in range(0, total_rows, batch_size):
            batch_end = min(batch_start + batch_size, total_rows)
            
            try:
                # Get batch DataFrame
                batch_df = processed_df.iloc[batch_start:batch_end]
                
                # Get predictions for entire batch at once
                batch_predictions = predict_expert_batch(batch_df)
                
                # Process each row in the batch
                for i, row_predictions in enumerate(batch_predictions):
                    row_features: Dict[str, Any] = {}
                    
                    # Only calculate velocity alignment with expert (no other features)
                    try:
                        current_row = batch_df.iloc[i]
                        # Current velocity values from telemetry
                        curr_velocity_x = float(current_row.get('Physics_velocity_x', 0.0))
                        curr_velocity_y = float(current_row.get('Physics_velocity_y', 0.0))
                        curr_velocity_z = float(current_row.get('Physics_velocity_z', 0.0))

                        # Expert optimal velocities from predictions (using ExpertOptimalFeature mapping)
                        exp_velocity_x = float(row_predictions.get(ExpertFeatureCatalog.ExpertOptimalFeature.EXPERT_OPTIMAL_VELOCITY_X.value, curr_velocity_x))
                        exp_velocity_y = float(row_predictions.get(ExpertFeatureCatalog.ExpertOptimalFeature.EXPERT_OPTIMAL_VELOCITY_Y.value, curr_velocity_y))
                        exp_velocity_z = float(row_predictions.get(ExpertFeatureCatalog.ExpertOptimalFeature.EXPERT_OPTIMAL_VELOCITY_Z.value, curr_velocity_z))

                        # Calculate velocity alignment (dot product normalized)
                        # If moving in same direction as expert, alignment = 1.0
                        curr_velocity_vector = np.array([curr_velocity_x, curr_velocity_y, curr_velocity_z])
                        exp_velocity_vector = np.array([exp_velocity_x, exp_velocity_y, exp_velocity_z])
                        curr_velocity_magnitude = np.linalg.norm(curr_velocity_vector)
                        exp_velocity_magnitude = np.linalg.norm(exp_velocity_vector)
                        
                        if curr_velocity_magnitude > 1e-6 and exp_velocity_magnitude > 1e-6:
                            # Normalize vectors and calculate dot product
                            curr_velocity_norm = curr_velocity_vector / curr_velocity_magnitude
                            exp_velocity_norm = exp_velocity_vector / exp_velocity_magnitude
                            velocity_alignment = np.dot(curr_velocity_norm, exp_velocity_norm)
                        else:
                            velocity_alignment = 0.0

                        # Persist raw telemetry context for downstream visualization
                        current_pos_x = float(current_row.get('Graphics_player_pos_x', 0.0))
                        current_pos_y = float(current_row.get('Graphics_player_pos_y', 0.0))
                        current_pos_z = float(current_row.get('Graphics_player_pos_z', 0.0))
                        current_speed = float(current_row.get('Physics_speed_kmh', curr_velocity_magnitude))

                        # Store expert optimal predictions for visualization with safe fallbacks
                        expert_pos_x = float(row_predictions.get(ExpertFeatureCatalog.ExpertOptimalFeature.EXPERT_OPTIMAL_PLAYER_POS_X.value, current_pos_x))
                        expert_pos_y = float(row_predictions.get(ExpertFeatureCatalog.ExpertOptimalFeature.EXPERT_OPTIMAL_PLAYER_POS_Y.value, current_pos_y))
                        expert_pos_z = float(row_predictions.get(ExpertFeatureCatalog.ExpertOptimalFeature.EXPERT_OPTIMAL_PLAYER_POS_Z.value, current_pos_z))

                        row_features[ExpertFeatures.EXPERT_OPTIMAL_PLAYER_POS_X.value] = expert_pos_x
                        row_features[ExpertFeatures.EXPERT_OPTIMAL_PLAYER_POS_Y.value] = expert_pos_y
                        row_features[ExpertFeatures.EXPERT_OPTIMAL_PLAYER_POS_Z.value] = expert_pos_z

                        expert_speed = float(row_predictions.get(ExpertFeatureCatalog.ExpertOptimalFeature.EXPERT_OPTIMAL_SPEED.value, exp_velocity_magnitude))
                        row_features[ExpertFeatures.EXPERT_OPTIMAL_SPEED.value] = expert_speed

                        # Add expert throttle and brake predictions
                        expert_throttle = float(row_predictions.get(ExpertFeatureCatalog.ExpertOptimalFeature.EXPERT_OPTIMAL_THROTTLE.value, 0.0))
                        row_features[ExpertFeatures.EXPERT_OPTIMAL_THROTTLE.value] = expert_throttle

                        expert_brake = float(row_predictions.get(ExpertFeatureCatalog.ExpertOptimalFeature.EXPERT_OPTIMAL_BRAKE.value, 0.0))
                        row_features[ExpertFeatures.EXPERT_OPTIMAL_BRAKE.value] = expert_brake

                        # Store only velocity alignment feature
                        row_features[ExpertFeatures.EXPERT_VELOCITY_ALIGNMENT.value] = float(velocity_alignment)

                        # Calculate speed difference
                        speed_difference = expert_speed - current_speed
                        row_features[ExpertFeatures.SPEED_DIFFERENCE.value] = float(speed_difference)

                        # Calculate distance to expert line (negative if off to left, positive if off to right)
                        distance_to_expert_line = np.sqrt(
                            (expert_pos_x - current_pos_x) ** 2 +
                            (expert_pos_y - current_pos_y) ** 2 +
                            (expert_pos_z - current_pos_z) ** 2
                        )
                        row_features[ExpertFeatures.DISTANCE_TO_EXPERT_LINE.value] = float(distance_to_expert_line)

                    except Exception as _e:
                        raise Exception(f"Velocity alignment calculation failed: {_e}")

                    expert_feature_rows.append(row_features)
                    
            except Exception as e:
                raise Exception(f"[WARNING] Failed to process batch {batch_start}-{batch_end}: {e}")

        print(f"[INFO] Completed expert state extraction. Extracted features for {len(expert_feature_rows)} records")
        return expert_feature_rows
    
    def filter_optimal_telemetry_segments(
        self,
        telemetry_data: List[Dict[str, Any]],
        max_segment_length: int = 60,
        improvement_threshold: float = 0.55,
        consistency_threshold: float = 1.0,
        min_segment_length: int = 20,
        min_segments: int = 0,
    ) -> List[List[Dict[str, Any]]]:
        """
        Identify contiguous telemetry slices that demonstrate measurable improvement or
        sustained expert-level consistency.

        Segments are grown dynamically from each starting point until either the
        improvement or consistency criteria stops being satisfied, or the
        ``max_segment_length`` cap is reached. Only the portion of the telemetry
        that meets the selected criteria is returned, eliminating the fixed-length
        constraints used previously.

        Args:
            telemetry_data: Telemetry records enriched with context features.
            max_segment_length: Upper bound on the number of records a single
                segment may contain.
            improvement_threshold: Minimum overall improvement rate required for
                a segment to be accepted.
            consistency_threshold: Minimum overall consistency rate required for
                a segment to be accepted when improvement is below the threshold.
            min_segment_length: Smallest segment length to analyse before
                considering acceptance.
            min_segments: Minimum number of segments required; raises if the
                condition is not met.

        Returns:
            A list of telemetry segments, where each segment is a list of
            dictionaries corresponding to contiguous telemetry samples.
        """

        print(f"[INFO] Filtering optimal telemetry segments from {len(telemetry_data)} records...")
        print(
            "[INFO] Using max_segment_length=%s, min_segment_length=%s, improvement_threshold=%.2f, consistency_threshold=%.2f"
            % (max_segment_length, min_segment_length, improvement_threshold, consistency_threshold)
        )

        if max_segment_length < min_segment_length:
            raise ValueError("max_segment_length must be greater than or equal to min_segment_length")

        if len(telemetry_data) < min_segment_length:
            print(
                f"[WARNING] Insufficient data for segment analysis. Need at least {min_segment_length} records, got {len(telemetry_data)}. Discarding this batch."
            )
            return []

        # Get context feature names from enum
        ContextFeature = ExpertFeatureCatalog.ContextFeature
        required_features = [
            ContextFeature.EXPERT_VELOCITY_ALIGNMENT.value,
            ContextFeature.SPEED_DIFFERENCE.value,
            ContextFeature.DISTANCE_TO_EXPERT_LINE.value,
        ]

        # Validate that required features exist in data
        if not telemetry_data:
            print("[WARNING] Empty telemetry data provided")
            return []

        first_record = telemetry_data[0]
        missing_features = [f for f in required_features if f not in first_record]
        if missing_features:
            raise ValueError(
                f"[ERROR] Missing required context features: {missing_features}, available: {list(first_record.keys())}"
            )

        # Convert to DataFrame for easier analysis
        try:
            df = pd.DataFrame(telemetry_data)
        except Exception as e:
            raise Exception(f"Failed to convert telemetry data to DataFrame: {e}")

        optimal_segments: List[List[Dict[str, Any]]] = []
        num_improvement_segments = 0
        num_consistency_segments = 0
        window_evaluations = 0

        idx = 0
        total_records = len(df)
        while idx < total_records:
            remaining = total_records - idx
            if remaining < min_segment_length:
                break

            last_valid_end: Optional[int] = None
            last_pass_type: Optional[str] = None
            max_end_index = min(total_records, idx + max_segment_length)
            evaluation_started = False

            for end_idx in range(idx + min_segment_length - 1, max_end_index):
                segment = df.iloc[idx : end_idx + 1]
                window_evaluations += 1
                evaluation_started = True

                summary = self._analyze_segment_improvement(segment)
                passes_improvement = summary.overall_improvement_rate >= improvement_threshold
                passes_consistency = summary.overall_consistency_rate >= consistency_threshold

                if passes_improvement or passes_consistency:
                    last_valid_end = end_idx
                    last_pass_type = "improvement" if passes_improvement else "consistency"
                else:
                    if last_valid_end is not None:
                        break
                    idx += 1
                    break
            else:
                if not evaluation_started:
                    idx += 1
                    continue

            if last_valid_end is not None:
                segment_df = df.iloc[idx : last_valid_end + 1]
                pruned_segment_df = self._prune_segment_stagnant_samples(segment_df)
                if len(pruned_segment_df) < len(segment_df):
                    print(
                        f"[DEBUG] Pruned {len(segment_df) - len(pruned_segment_df)} stagnant samples"
                    )

                optimal_segments.append(pruned_segment_df.to_dict("records"))

                if last_pass_type == "improvement":
                    num_improvement_segments += 1
                else:
                    num_consistency_segments += 1

                idx = last_valid_end + 1
            else:
                if evaluation_started:
                    continue
                idx += 1

        print("[INFO] Dynamic segment filtering analysis complete:")
        print(f"[INFO] - Original records: {len(telemetry_data)}")
        print(f"[INFO] - Windows evaluated: {window_evaluations}")
        print(f"[INFO] - Accepted segments: {len(optimal_segments)}")
        print(
            f"[INFO] - Improvement-based passes: {num_improvement_segments}, Consistency-based passes: {num_consistency_segments}"
        )

        # Ensure we have minimum required segments
        if len(optimal_segments) < min_segments:
            raise ValueError(
                f"[WARNING] Only found {len(optimal_segments)} optimal segments, which is less than the minimum required {min_segments}. Adjust parameters or provide more data."
            )

        return optimal_segments
    
    def _prune_segment_stagnant_samples(
        self,
        segment_df: pd.DataFrame,
        *,
        change_thresholds: Optional[Dict[str, float]] = None,
    ) -> pd.DataFrame:
        """Drop timestamps that barely change driver inputs within a segment.

        A row is removed when every monitored control (gas, brake, steer, gear)
        changes less than its threshold compared to the previous row. This keeps
        the segment focused on meaningful driver inputs while preserving at
        least the first and last sample so downstream consumers retain context.
        """

        if segment_df is None or segment_df.empty or len(segment_df) <= 1:
            return segment_df

        thresholds = change_thresholds or {
            "Physics_gas": 0.01,
            "Physics_brake": 0.01,
            "Physics_steer_angle": 0.01,
            "Physics_gear": 0.0,
        }

        available_columns = [col for col in thresholds if col in segment_df.columns]
        if len(available_columns) <= 1:
            return segment_df

        deltas = segment_df[available_columns].diff().abs()
        keep_mask = np.zeros(len(segment_df), dtype=bool)
        keep_mask[0] = True

        for idx in range(1, len(segment_df)):
            keep_sample = False
            for col in available_columns:
                delta = deltas.iloc[idx][col]
                threshold = thresholds[col]
                if col == "Physics_gear":
                    if delta > threshold:
                        keep_sample = True
                        break
                else:
                    if delta > threshold:
                        keep_sample = True
                        break
            keep_mask[idx] = keep_sample

        keep_mask[-1] = True

        pruned_segment = segment_df.loc[keep_mask].reset_index(drop=True)

        if pruned_segment.empty or len(pruned_segment) < 2:
            return segment_df

        return pruned_segment

    def _analyze_segment_improvement(self, segment: pd.DataFrame) -> SegmentImprovementSummary:
        """
        Analyze improvement trends vs consistency within a telemetry segment.

        Returns a structured dataclass that retains the original dictionary keys for
        backwards compatibility while providing attribute access and helper
        utilities.
        """

        config = SegmentImprovementConfig()
        ContextFeature = ExpertFeatureCatalog.ContextFeature
        summary = SegmentImprovementSummary()

        smoothing_window = max(config.smoothing_window_min, min(config.smoothing_window_max, len(segment)))
        ema_span = max(config.ema_span_min, smoothing_window)

        def _smooth_series(values: Union[pd.Series, np.ndarray]) -> np.ndarray:
            series = values if isinstance(values, pd.Series) else pd.Series(values)
            if len(series) <= 1:
                return series.to_numpy()

            median_smoothed = series.rolling(window=smoothing_window, min_periods=1, center=True).median()
            ema_smoothed = median_smoothed.ewm(span=ema_span, adjust=False).mean()
            return ema_smoothed.to_numpy()

        try:
            # Velocity alignment analysis
            velocity_series = segment[ContextFeature.EXPERT_VELOCITY_ALIGNMENT.value]
            velocity_smoothed = _smooth_series(velocity_series)
            if len(velocity_smoothed) > 1:
                summary.velocity_alignment_mean = float(np.mean(velocity_smoothed))
                summary.velocity_alignment_trend = float(np.polyfit(range(len(velocity_smoothed)), velocity_smoothed, 1)[0])
                summary.velocity_expert_points = int(np.sum(velocity_smoothed >= config.expert_velocity_alignment))
                summary.velocity_consistency_rate = summary.velocity_expert_points / len(velocity_smoothed)

            # Speed difference analysis
            speed_diff_raw = segment[ContextFeature.SPEED_DIFFERENCE.value]
            speed_diff_smoothed = _smooth_series(speed_diff_raw)
            speed_has_samples = len(speed_diff_smoothed) > 1
            if speed_has_samples:
                abs_speed_diff = np.abs(speed_diff_smoothed)
                summary.speed_difference_mean = float(np.mean(abs_speed_diff))
                summary.speed_difference_trend = float(np.polyfit(range(len(abs_speed_diff)), abs_speed_diff, 1)[0])
                summary.speed_expert_points = int(np.sum(abs_speed_diff <= config.expert_speed_diff_max))
                summary.speed_consistency_rate = summary.speed_expert_points / len(abs_speed_diff)

            # Distance to line analysis
            distance_series = segment[ContextFeature.DISTANCE_TO_EXPERT_LINE.value]
            distance_smoothed = _smooth_series(distance_series)
            distance_has_samples = len(distance_smoothed) > 1
            if distance_has_samples:
                summary.distance_to_line_mean = float(np.mean(distance_smoothed))
                summary.distance_to_line_trend = float(np.polyfit(range(len(distance_smoothed)), distance_smoothed, 1)[0])
                summary.distance_expert_points = int(np.sum(distance_smoothed <= config.expert_distance_max))
                summary.distance_consistency_rate = summary.distance_expert_points / len(distance_smoothed)

            # Driver push-to-limit analysis (0-1 intensity provided by TireGripAnalysisService)
            tire_feature = TireGripFeatureCatalog.ContextFeature.DRIVER_PUSH_TO_LIMIT.value
            push_series = pd.to_numeric(segment[tire_feature], errors='coerce').fillna(0.0)
            push_smoothed = _smooth_series(push_series)
            if len(push_smoothed) > 1:
                summary.driver_push_available = True
                summary.driver_push_mean = float(np.mean(push_smoothed))
                sample_idx = np.arange(len(push_smoothed))
                summary.driver_push_trend = float(np.polyfit(sample_idx, push_smoothed, 1)[0])
                push_above_threshold_rate = float(np.mean(push_smoothed >= config.driver_push_high_threshold))
                summary.driver_push_high_rate = push_above_threshold_rate

            # Improvement and consistency calculations
            distance_improvement = distance_has_samples and summary.distance_to_line_trend < 0.0
            speed_improvement = speed_has_samples and summary.speed_difference_trend < 0.0
            
            improvement_criteria: List[bool] = [distance_improvement, speed_improvement]

            base_improvement_rate = 0.0
            if improvement_criteria:
                base_improvement_rate = sum(improvement_criteria) / len(improvement_criteria)

            if summary.driver_push_available:
                push_threshold_rate = float(np.clip(summary.driver_push_high_rate, 0.0, 1.0))
                base_improvement_rate *= push_threshold_rate

            summary.overall_improvement_rate = base_improvement_rate

            consistency_rates = [
                summary.velocity_consistency_rate,
                summary.speed_consistency_rate,
                summary.distance_consistency_rate,
            ]

            if consistency_rates:
                summary.overall_consistency_rate = sum(consistency_rates) / len(consistency_rates)

        except Exception as e:
            raise Exception(f"Error analyzing segment improvement: {e}")

        return summary
    
    # Visualization utilities moved to telemetry_segment_visualizer.visualize_optimal_segments

    def serialize_learning_model(self) -> Dict[str, Any]:
        """
        Memory-efficient serialization of trained models stored in the position learner
        
        Returns:
            Dictionary with serialized models ready for storage/transmission
        """
        if not self.position_learner.track_models:
            raise ValueError("No trained models available to serialize. Train models first.")
        
        print("[INFO] Serializing current position models (memory-efficient)...")
        
        try:
            # Build result structure directly without deep copying the entire model
            result = {}
            
            # Serialize models per track
            serialized_tracks = {}
            
            for track_name, track_data in self.position_learner.track_models.items():
                print(f"[INFO] Serializing models for track: {track_name}")
                serialized_track_data = {}
                
                # Serialize models
                if 'models' in track_data:
                    serialized_models = {}
                    for model_name, model in track_data['models'].items():
                        # Serialize directly without copying
                        serialized_model_data = self.serialize_data(model)
                        serialized_models[model_name] = serialized_model_data
                        # Force garbage collection
                        import gc
                        gc.collect()
                    serialized_track_data['models'] = serialized_models
                
                # Serialize scaler
                if 'position_scaler' in track_data:
                    serialized_track_data['position_scaler'] = self.serialize_data(track_data['position_scaler'])
                
                # Copy metadata
                for key in ['performance_metrics', 'input_features', 'target_features']:
                    if key in track_data:
                        serialized_track_data[key] = track_data[key]
                
                serialized_tracks[track_name] = serialized_track_data
            
            result['track_models'] = serialized_tracks
            
            print("[INFO] Serialization completed successfully")
            return result
            
        except Exception as e:
            error_msg = f"Failed to serialize imitation learning models: {str(e)}"
            raise RuntimeError(error_msg) from e
    
    # Deserialize object inside 
    def deserialize_imitation_model(self, serialized_results: Dict[str, Any]) -> 'ExpertImitateLearningService':
        """
        Deserialize the serialized position models and load them directly into the position learner. After deserializing,
        the models are ready for immediate use in predictions.
        Args:
            serialized_results: Dictionary containing serialized models and metadata
            
        Returns:
            Self (ExpertImitateLearningService): The current instance with loaded models
        """
        try:
            print("[INFO] Deserializing imitation models...")
            
            if 'track_models' in serialized_results:
                print("[INFO] Deserializing track models...")
                deserialized_track_models = {}
                
                for track_name, track_data in serialized_results['track_models'].items():
                    print(f"[INFO] Deserializing models for track: {track_name}")
                    deserialized_track_data = {}
                    
                    # Deserialize models
                    if 'models' in track_data:
                        deserialized_models = {}
                        for model_name, serialized_model in track_data['models'].items():
                            deserialized_models[model_name] = self.deserialize_data(serialized_model)
                        deserialized_track_data['models'] = deserialized_models
                    
                    # Deserialize scaler
                    if 'position_scaler' in track_data:
                        deserialized_track_data['position_scaler'] = self.deserialize_data(track_data['position_scaler'])
                    
                    # Copy metadata
                    for key in ['performance_metrics', 'input_features', 'target_features']:
                        if key in track_data:
                            deserialized_track_data[key] = track_data[key]
                            
                    deserialized_track_models[track_name] = deserialized_track_data
                
                self.position_learner.track_models = deserialized_track_models
                print(f"[INFO] Successfully deserialized models for {len(deserialized_track_models)} tracks")
            else:
                raise ValueError("No track_models found in serialized data")
                
            return self
            
        except Exception as e:
            error_msg = f"Failed to deserialize imitation learning models: {str(e)}"
            raise RuntimeError(error_msg) from e
    

    # Memory-efficient serialize models function
    def serialize_data(self, data: any) -> str:
        """
        Memory-efficient serialization of trained models
        
        Args:
            data: Model data to serialize
            
        Returns:
            Serialized model data as base64 encoded string
        """
        try:
            # Use highest compression protocol for smaller output
            buffer = io.BytesIO()
            pickle.dump(data, buffer, protocol=pickle.HIGHEST_PROTOCOL)
            buffer.seek(0)
            
            # Get raw bytes and immediately clear buffer to free memory
            raw_bytes = buffer.getvalue()
            buffer.close()  # Explicitly close to free memory
            
            # Encode to base64 in chunks to avoid memory spikes
            import binascii
            encoded_data = base64.b64encode(raw_bytes).decode('utf-8')
            
            # Clear intermediate data
            del raw_bytes
            import gc
            gc.collect()
            
            return encoded_data
                
        except Exception as e:
            print(f"[ERROR] Failed to serialize models: {e}")
            raise e
    
    def deserialize_data(self, model_data: str) -> Dict[str, Any]:
        """
        Memory-efficient deserialization of models from base64 string
        
        Args:
            model_data: Base64 encoded model data
        
        Returns:
            Dictionary containing deserialized models and metadata
        """
        try:
            # Decode from base64
            decoded_data = base64.b64decode(model_data.encode('utf-8'))
            
            # Deserialize using pickle with memory-efficient buffer
            buffer = io.BytesIO(decoded_data)
            data_result = pickle.load(buffer)
            
            # Explicitly clean up memory
            buffer.close()
            del decoded_data
            import gc
            gc.collect()
            
            return data_result
            
        except Exception as e:
            raise Exception(f"Failed to deserialize imitation learning models: {str(e)}")
    
# Example usage and testing
if __name__ == "__main__":
    
    # Example workflow
    service = ExpertImitateLearningService()
    
    # 1. Train the model (this stores models in the class)
    # serialized_results = service.train_ai_model(expert_telemetry_data)

    # 3. Compare new telemetry with stored expert models
    # comparison = service.compare_telemetry_with_expert(incoming_telemetry)