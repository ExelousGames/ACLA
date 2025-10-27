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
import copy
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

from .telemetry_models import TelemetryFeatures, FeatureProcessor

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

    class ContextFeature(str, Enum):
        # Velocity direction alignment with expert
        EXPERT_VELOCITY_ALIGNMENT = 'expert_velocity_alignment' # 1.0 if moving in the expert velocity direction, 0.0 opposite direction
        SPEED_DIFFERENCE = 'speed_difference' # Difference between current speed and expert optimal speed (km/h)
        DISTANCE_TO_EXPERT_LINE = 'distance_to_expert_line' # distance between current position and expert optimal racing line (meters)
    
    class TrajectoryFeature(str, Enum):
        TRACK_POSITION = 'track_position'
        TRACK_POSITION_RATE = 'track_position_rate'
        PLAYER_POS_X = 'player_pos_x'
        PLAYER_POS_X_VELOCITY = 'player_pos_x_velocity'
        PLAYER_POS_Y = 'player_pos_y'
        PLAYER_POS_Y_VELOCITY = 'player_pos_y_velocity'
        PLAYER_POS_Z = 'player_pos_z'
        PLAYER_POS_Z_VELOCITY = 'player_pos_z_velocity'
        SPEED = 'speed'
        SPEED_CHANGE = 'speed_change'
        ACCELERATION = 'acceleration'
        GEAR = 'gear'
        GEAR_CHANGE = 'gear_change'
        SPEED_PER_GEAR = 'speed_per_gear'
        STEERING_ANGLE = 'steering_angle'
        STEERING_RATE = 'steering_rate'
        CORNERING_FORCE = 'cornering_force'
        THROTTLE = 'throttle'
        THROTTLE_RATE = 'throttle_rate'
        BRAKE = 'brake'
        BRAKE_RATE = 'brake_rate'
        CURRENT_TIME = 'current_time'
        TIME_RATE = 'time_rate'
        VELOCITY_X = 'velocity_x'
        VELOCITY_X_RATE = 'velocity_x_rate'
        VELOCITY_Y = 'velocity_y'
        VELOCITY_Y_RATE = 'velocity_y_rate'
        VELOCITY_Z = 'velocity_z'
        VELOCITY_Z_RATE = 'velocity_z_rate'

    # Flat list for convenience (now only expert optimal + derived)
    CONTEXT_FEATURES: List[str] = [f.value for f in ContextFeature]


class ExpertPositionLearner:
    """Learn expert actions based on normalized track position from multiple expert laps"""
    
    def __init__(self):
        self.position_model = None
        self.scaler = StandardScaler()
        self.position_scaler = StandardScaler()  # Separate scaler for position input
        
        # Models for different output types
        self.action_models = {}  # For steering, gas, brake
        self.gear_model = None   # Classification model for gear
        self.position_models = {} # For position x,y,z predictions
        self.velocity_models = {} # For velocity x,y,z predictions
    
    def extract_position_features(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Extract position-based features for expert learning
        
        Args:
            df: Telemetry DataFrame
            
        Returns:
            Dictionary with input features (normalized position) and target features (expert actions/states)
        """
        input_features = pd.DataFrame()
        target_features = pd.DataFrame()
        
        # Input feature: normalized track position (primary input)
        if 'Graphics_normalized_car_position' in df.columns:
            input_features['normalized_position'] = df['Graphics_normalized_car_position']
        else:
            raise ValueError("Graphics_normalized_car_position not found - this is required for position-based learning")
        
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
            target_features['expert_optimal_velocity_z'] = df['Physics_velocity_z']
            
        # Speed (derived)
        if 'Physics_speed_kmh' in df.columns:
            target_features[EO.EXPERT_OPTIMAL_SPEED.value] = df['Physics_speed_kmh']
            
        # Track position (for consistency)
        if 'Graphics_normalized_car_position' in df.columns:
            target_features[EO.EXPERT_OPTIMAL_TRACK_POSITION.value] = df['Graphics_normalized_car_position']
        
        # Clean data
        input_features = input_features.fillna(0)
        target_features = target_features.fillna(0)
        
        return {
            'input_features': input_features,
            'target_features': target_features
        }
    
    def learn_expert_position_mapping(self, expert_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Learn expert actions from normalized track position using multiple expert laps
        
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
        
        # Prepare input (normalized position)
        X = input_features[['normalized_position']].values
        X_scaled = self.position_scaler.fit_transform(X)
        
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
                print(f"[INFO] Training action model for: {target_name}")
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
                
                print(f"[INFO] {target_name} - R²: {r2:.4f}, MSE: {mse:.4f}, MAE: {mae:.4f}")
        
        # Gear model (classification)
        if EO.EXPERT_OPTIMAL_GEAR.value in target_features.columns:
            print(f"[INFO] Training gear classification model")
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
            
            print(f"[INFO] Gear - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
        
        # Position models (regression)
        position_targets = [
            EO.EXPERT_OPTIMAL_PLAYER_POS_X.value,
            EO.EXPERT_OPTIMAL_PLAYER_POS_Y.value,
            EO.EXPERT_OPTIMAL_PLAYER_POS_Z.value
        ]
        
        for target_name in position_targets:
            if target_name in target_features.columns:
                print(f"[INFO] Training position model for: {target_name}")
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
                
                print(f"[INFO] {target_name} - R²: {r2:.4f}, MSE: {mse:.4f}, MAE: {mae:.4f}")
        
        # Velocity models (regression)
        velocity_targets = [
            EO.EXPERT_OPTIMAL_VELOCITY_X.value,
            EO.EXPERT_OPTIMAL_VELOCITY_Y.value,
            'expert_optimal_velocity_z'  # This one wasn't in the enum
        ]
        
        for target_name in velocity_targets:
            if target_name in target_features.columns:
                print(f"[INFO] Training velocity model for: {target_name}")
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
                
                print(f"[INFO] {target_name} - R²: {r2:.4f}, MSE: {mse:.4f}, MAE: {mae:.4f}")
        
        # Track position model (for consistency)
        if EO.EXPERT_OPTIMAL_TRACK_POSITION.value in target_features.columns:
            print(f"[INFO] Training track position model")
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
            
            print(f"[INFO] Track position - R²: {r2:.4f}, MSE: {mse:.4f}, MAE: {mae:.4f}")
        
        # Store the complete position model
        self.position_model = {
            'models': models,
            'position_scaler': self.position_scaler,
            'performance_metrics': performance_metrics,
            'input_features': ['normalized_position'],
            'target_features': list(target_features.columns)
        }
        
        return {
            'modelData': self.position_model,
            'metadata': {
                'performance_metrics': performance_metrics,
                'input_features': ['normalized_position'],
                'target_features': list(target_features.columns),
                'models_trained': list(models.keys()),
                'total_training_samples': len(expert_df)
            }
        }
    
    def predict_expert_actions_at_position(self, normalized_positions: Union[float, List[float], np.ndarray]) -> Union[Dict[str, float], List[Dict[str, float]]]:
        """
        Predict expert actions at given normalized track position(s)
        
        Args:
            normalized_positions: Single position or array of positions (0.0 to 1.0)
            
        Returns:
            Dictionary with expert predictions, or list of dictionaries for multiple positions
        """
        if not self.position_model:
            raise ValueError("No position model trained. Call learn_expert_position_mapping() first.")
        
        # Handle single position vs multiple positions
        single_position = isinstance(normalized_positions, (int, float))
        if single_position:
            positions_array = np.array([[normalized_positions]])
        else:
            positions_array = np.array(normalized_positions).reshape(-1, 1)
        
        # Scale positions
        positions_scaled = self.position_model['position_scaler'].transform(positions_array)
        
        # Make predictions for all models
        predictions = {}
        for model_name, model in self.position_model['models'].items():
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
        if not self.position_model:
            return {
                'status': 'No model available',
                'has_model': False,
                'error': 'Position model not trained yet'
            }
        
        debug_info = {
            'status': 'Model available',
            'has_model': True,
            'model_structure': {}
        }
        
        # Check model structure
        for key, value in self.position_model.items():
            if key == 'models':
                debug_info['model_structure']['models'] = {
                    'count': len(value),
                    'model_names': list(value.keys()),
                    'model_types': [type(model).__name__ for model in value.values()]
                }
            elif key == 'position_scaler':
                debug_info['model_structure']['position_scaler'] = {
                    'type': type(value).__name__,
                    'fitted': hasattr(value, 'mean_')
                }
            elif key == 'performance_metrics':
                debug_info['model_structure']['performance_metrics'] = {
                    'available_metrics': list(value.keys()),
                    'metric_count': len(value)
                }
            else:
                debug_info['model_structure'][key] = {
                    'type': type(value).__name__,
                    'value': str(value) if not isinstance(value, (list, dict)) else f"{type(value).__name__} with {len(value)} items"
                }
        
        return debug_info
    
    def validate_position_input(self, normalized_positions: Union[float, List[float], np.ndarray]) -> Dict[str, Any]:
        """
        Validate normalized position input for prediction
        
        Args:
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
        if not self.position_model:
            validation_results['valid'] = False
            validation_results['errors'].append("No position model trained")
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
        from .Training_data_cache_service import get_shared_data_cache
        return get_shared_data_cache()

    def train_ai_model(self, telemetry_data: List[Dict[str, Any]], learning_objectives: List[str] = None) -> Tuple[Dict[str, Any]]:
        """
        Learn from expert driving demonstrations
        
        Args:
            telemetry_data: List of expert telemetry data dictionaries
            learning_objectives: List of what to learn ('trajectory')
            
        Returns:
            Dictionary with trained models and learning insights, serialized objects and ready for storage
        """
        if learning_objectives is None:
            learning_objectives = ['trajectory']
        
        print(f"[INFO {self.__class__.__name__}] Learning from {len(telemetry_data)} expert demonstrations")
        print(f"[INFO {self.__class__.__name__}] Learning objectives: {learning_objectives}")

        # Convert to DataFrame
        telemetry_df = pd.DataFrame(telemetry_data)
        feature_processor = FeatureProcessor(telemetry_df)
        # Cleaned data
        processed_df = feature_processor.general_cleaning_for_analysis()
        
        # Learn expert position mapping (this is the only learning model)
        if 'trajectory' in learning_objectives:
            print("[INFO] Learning expert position mapping...")
            results = self.position_learner.learn_expert_position_mapping(processed_df)
            results['learning_summary'] = self._generate_learning_summary(results)
        else:
            raise ValueError("No valid learning objectives provided. Expected 'trajectory'.")

        return results
    
    def predict_expert_actions(self,
                             processed_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Predict what an expert would do in the current situation.

        Args:
            processed_df: Current telemetry DataFrame (expects a single row).

        Returns:
            Dictionary with the predicted expert actions keyed by canonical metric names.
        """
        predictions: Dict[str, Any] = {}

        if not self.position_learner.position_model:
            print("[WARNING] No position model available")
            return {"error": "No trained models available"}

        row_count = len(processed_df)
        if row_count == 0:
            raise ValueError("Telemetry dataframe is empty")
        if row_count > 1:
            # Use the first telemetry sample when multiple rows slip through preprocessing.
            print(f"[INFO] Received {row_count} telemetry rows; using the first row for prediction.")
            processed_df = processed_df.iloc[[0]].copy()
        else:
            processed_df = processed_df.copy()

        if 'Graphics_normalized_car_position' not in processed_df.columns:
            raise ValueError("No normalized track position data available")

        normalized_position = processed_df['Graphics_normalized_car_position'].iloc[0]
        try:
            optimal_actions = self.position_learner.predict_expert_actions_at_position(float(normalized_position))
        except Exception as exc:
            raise Exception(f"[WARNING] Could not predict expert actions: {exc}") from exc

        if not isinstance(optimal_actions, dict):
            raise ValueError("Expert model returned unexpected payload for optimal actions")

        predictions['optimal_actions'] = optimal_actions

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
            performance_metrics = results['metadata']['performance_metrics']
            
            # Separate regression and classification metrics
            r2_scores = [metrics['r2'] for metrics in performance_metrics.values() if 'r2' in metrics]
            accuracy_scores = [metrics['accuracy'] for metrics in performance_metrics.values() if 'accuracy' in metrics]
            
            # Calculate average scores
            avg_r2 = np.mean(r2_scores) if r2_scores else 0.0
            avg_accuracy = np.mean(accuracy_scores) if accuracy_scores else 0.0
            
            summary['position_summary'] = {
                'models_trained': len(results['metadata']['models_trained']),
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
            List of dictionaries, one per record, containing expert targets and delta-to-expert context only.
        """
        
        ContextFeature = ExpertFeatureCatalog.ContextFeature
        if not telemetry_data:
            return []
        if not self.position_learner.position_model:
            raise ValueError("No trained imitation models available. Train or load models before calling extract_expert_state_for_telemetry().")

        try:
            processed_df = pd.DataFrame(telemetry_data)
        except Exception as e:
            raise Exception(f"Failed to create DataFrame: {e}")

        expert_feature_rows: List[Dict[str, Any]] = []

        # Position models should already be loaded/deserialized
        if not self.position_learner.position_model:
            raise ValueError("Position model not loaded. Call train_ai_model() or deserialize_imitation_model() first.")

        def predict_expert_batch(batch_df: pd.DataFrame) -> List[Dict[str, float]]:
            """Predict expert actions for a batch of normalized positions - much faster than row-by-row"""
            if not self.position_learner.position_model:
                return [{} for _ in range(len(batch_df))]
            try:
                # Extract normalized positions from batch
                if 'Graphics_normalized_car_position' not in batch_df.columns:
                    raise ValueError("Graphics_normalized_car_position not found in batch data")
                
                normalized_positions = batch_df['Graphics_normalized_car_position'].values
                batch_predictions = self.position_learner.predict_expert_actions_at_position(normalized_positions)
                return batch_predictions
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
                        EO = ExpertFeatureCatalog.ExpertOptimalFeature
                        exp_velocity_x = float(row_predictions.get(EO.EXPERT_OPTIMAL_VELOCITY_X.value, curr_velocity_x))
                        exp_velocity_y = float(row_predictions.get(EO.EXPERT_OPTIMAL_VELOCITY_Y.value, curr_velocity_y))
                        exp_velocity_z = float(row_predictions.get('expert_optimal_velocity_z', curr_velocity_z))

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

                        # Store only velocity alignment feature
                        row_features[ContextFeature.EXPERT_VELOCITY_ALIGNMENT.value] = float(velocity_alignment)

                        # Calculate speed difference
                        speed_difference = exp_velocity_magnitude - curr_velocity_magnitude
                        row_features[ContextFeature.SPEED_DIFFERENCE.value] = float(speed_difference)

                        # Calculate distance to expert line (negative if off to left, positive if off to right)
                        expert_pos_x = row_predictions.get(EO.EXPERT_OPTIMAL_PLAYER_POS_X.value, 0.0)
                        expert_pos_y = row_predictions.get(EO.EXPERT_OPTIMAL_PLAYER_POS_Y.value, 0.0)
                        expert_pos_z = row_predictions.get(EO.EXPERT_OPTIMAL_PLAYER_POS_Z.value, 0.0)
                        current_pos_x = current_row.get('Graphics_player_pos_x', 0.0)
                        current_pos_y = current_row.get('Graphics_player_pos_y', 0.0)
                        current_pos_z = current_row.get('Graphics_player_pos_z', 0.0)

                        distance_to_expert_line = np.sqrt(
                            (expert_pos_x - current_pos_x) ** 2 +
                            (expert_pos_y - current_pos_y) ** 2 +
                            (expert_pos_z - current_pos_z) ** 2
                        )
                        row_features[ContextFeature.DISTANCE_TO_EXPERT_LINE.value] = float(distance_to_expert_line)

                    except Exception as _e:
                        raise Exception(f"Velocity alignment calculation failed: {_e}")

                    expert_feature_rows.append(row_features)
                    
            except Exception as e:
                raise Exception(f"[WARNING] Failed to process batch {batch_start}-{batch_end}: {e}")

        print(f"[INFO] Completed expert state extraction. Extracted features for {len(expert_feature_rows)} records")
        return expert_feature_rows
    
    def filter_optimal_telemetry_segments(self, telemetry_data: List[Dict[str, Any]], 
                                         segment_length: int = 20, 
                                         improvement_threshold: float = 0.55,
                                         min_segments: int = 0) -> List[List[Dict[str, Any]]]:
        """
        Filter telemetry data using streamlined no-fallback approach.
        
        Streamlined Logic:
        - Calculate both overall improvement rate AND overall consistency rate for each segment
        - Accept segment if EITHER improvement rate OR consistency rate meets the threshold
        - No fallback mechanisms - clean pass/fail based on the higher of the two rates
        
        Expert-level thresholds (determine which metrics use improvement vs consistency):
        - Velocity alignment: ≥90% alignment = expert-level → use consistency (80% of points ≥90% alignment)
        - Speed difference: ≤5km/h from expert = expert-level → use consistency (75% of points ≤5km/h difference)  
        - Distance to expert: ≤1m from racing line = expert-level → use consistency (80% of points ≤1m)
        - Sub-expert performance uses improvement trends instead

        Args:
            telemetry_data: List of telemetry records containing ContextFeature values
            segment_length: Length of each segment to analyze
            improvement_threshold: Minimum rate (0.0-1.0) for either improvement OR consistency to pass
            min_segments: Minimum number of segments required to return results
            
        Returns:
            List[List[Dict[str, Any]]]: List of segments meeting streamlined criteria
        """
        
        print(f"[INFO] Filtering optimal telemetry segments from {len(telemetry_data)} records...")
        print(f"[INFO] Using segment_length={segment_length}, improvement_threshold={improvement_threshold}")

        if len(telemetry_data) < segment_length * 2:
            raise ValueError(f"[WARNING] Insufficient data for segment analysis. Need at least {segment_length * 2} records, got {len(telemetry_data)}")
        
        # Get context feature names from enum
        ContextFeature = ExpertFeatureCatalog.ContextFeature
        required_features = [
            ContextFeature.EXPERT_VELOCITY_ALIGNMENT.value,
            ContextFeature.SPEED_DIFFERENCE.value,
            ContextFeature.DISTANCE_TO_EXPERT_LINE.value
        ]
        
        # Validate that required features exist in data
        if not telemetry_data:
            print("[WARNING] Empty telemetry data provided")
            return []
            
        first_record = telemetry_data[0]
        missing_features = [f for f in required_features if f not in first_record]
        if missing_features:
            raise ValueError(f"[ERROR] Missing required context features: {missing_features}, available: {list(first_record.keys())}")

        # Convert to DataFrame for easier analysis
        try:
            df = pd.DataFrame(telemetry_data)
        except Exception as e:
            raise Exception(f"Failed to convert telemetry data to DataFrame: {e}")

        # Create segments and analyze improvement trends
        optimal_segments = []

        # Calculate total segments ensuring each has exactly segment_length
        total_segments = (len(df) - segment_length) // (segment_length // 2) + 1  # Overlapping segments
        
        print(f"[INFO] Analyzing {total_segments} potential segments with adaptive criteria...")
        
        for start_idx in range(0, len(df) - segment_length + 1, segment_length // 2):  # 50% overlap
            end_idx = start_idx + segment_length  # Fixed length, no min() to ensure exact segment_length
            segment = df.iloc[start_idx:end_idx].copy()
            
            # Ensure segment has exactly the required length
            if len(segment) != segment_length:
                continue
                
            # Analyze improvement trends for each context feature
            improvement_scores = self._analyze_segment_improvement(segment, required_features)
            
            # Streamlined no-fallback approach: use improvement rate if above threshold, otherwise use consistency rate
            segment_passes = False
            if improvement_scores['overall_improvement_rate'] >= improvement_threshold:
                segment_passes = True
                # Use improvement rate criteria
            elif improvement_scores['overall_consistency_rate'] >= 0.90:
                segment_passes = True
                # Use consistency rate criteria
            
            if segment_passes:
                segment_dict = segment.to_dict('records')
                optimal_segments.append(segment_dict)
        
        print(f"[INFO] Dual-rate filtering analysis complete:")
        print(f"[INFO] - Original records: {len(telemetry_data)}")
        print(f"[INFO] - Segments analyzed: {total_segments}")
        print(f"[INFO] - Optimal segments found: {len(optimal_segments)}")
        print(f"[INFO] - Overall pass rate: {len(optimal_segments)/total_segments*100:.1f}%")

        # Ensure we have minimum required segments
        if len(optimal_segments) < min_segments:
            raise ValueError(f"[WARNING] Only found {len(optimal_segments)} optimal segments, which is less than the minimum required {min_segments}. Adjust parameters or provide more data.")
        
        return optimal_segments
    
    def _analyze_segment_improvement(self, segment: pd.DataFrame, required_features: List[str]) -> Dict[str, float]:
        """
        Analyze improvement trends vs consistency within a telemetry segment.
        Calculates BOTH overall improvement rate AND overall consistency rate for ALL metrics.
        
        Dual Rating System:
        - overall_improvement_rate: Calculated by checking trend improvement across ALL 3 metrics
        - overall_consistency_rate: Calculated by checking percentile consistency across ALL 3 metrics
        
        Individual Metric Logic (for detailed analysis):
        - If driver is far from expert level: Uses improvement mode in individual analysis
        - If driver is close to expert level: Uses consistency mode in individual analysis
        
        Args:
            segment: DataFrame containing telemetry segment
            required_features: List of context feature names to analyze
            
        Returns:
            Dictionary with improvement analysis results including:
            - overall_improvement_rate: Rate calculated from ALL metrics using trend analysis
            - overall_consistency_rate: Rate calculated from ALL metrics using percentile analysis
            - Individual metric analysis with detailed mode-specific results
        """
        ContextFeature = ExpertFeatureCatalog.ContextFeature
        improvement_metrics = {}
        
        # Expert-level thresholds (when driver is considered "close to expert")
        EXPERT_VELOCITY_ALIGNMENT = 0.9  # 90% alignment considered expert-level
        EXPERT_SPEED_DIFF_MAX = 5.0     # Within 5 km/h considered expert-level
        EXPERT_DISTANCE_MAX = 1.0        # Within 1 meter considered expert-level

        # Short smoothing window to tame sensor spikes while retaining responsiveness
        smoothing_window = max(2, min(5, len(segment)))
        ema_span = max(2, smoothing_window)

        def _smooth_series(values: Union[pd.Series, np.ndarray]) -> np.ndarray:
            series = values if isinstance(values, pd.Series) else pd.Series(values)
            if len(series) <= 1:
                return series.to_numpy()

            median_smoothed = series.rolling(window=smoothing_window, min_periods=1, center=True).median()
            ema_smoothed = median_smoothed.ewm(span=ema_span, adjust=False).mean()
            return ema_smoothed.to_numpy()
        
        try:
            # Analyze velocity alignment
            velocity_alignment = _smooth_series(segment[ContextFeature.EXPERT_VELOCITY_ALIGNMENT.value])
            if len(velocity_alignment) > 1:
                velocity_mean = np.mean(velocity_alignment)
                velocity_trend = np.polyfit(range(len(velocity_alignment)), velocity_alignment, 1)[0]
                expert_performance_points = np.sum(velocity_alignment >= EXPERT_VELOCITY_ALIGNMENT)
                velocity_consistency_rate = expert_performance_points / len(velocity_alignment)
                
                improvement_metrics['velocity_alignment_mean'] = float(velocity_mean)
                improvement_metrics['velocity_alignment_trend'] = float(velocity_trend)
                improvement_metrics['velocity_consistency_rate'] = float(velocity_consistency_rate)
                improvement_metrics['velocity_expert_points'] = int(expert_performance_points)
            else:
                improvement_metrics['velocity_alignment_mean'] = 0.0
                improvement_metrics['velocity_alignment_trend'] = 0.0
                improvement_metrics['velocity_consistency_rate'] = 0.0
                improvement_metrics['velocity_expert_points'] = 0
            
            # Analyze speed difference
            speed_diff_raw = segment[ContextFeature.SPEED_DIFFERENCE.value]
            speed_diff = _smooth_series(speed_diff_raw)
            if len(speed_diff) > 1:
                abs_speed_diff = np.abs(speed_diff)
                speed_diff_mean = np.mean(abs_speed_diff)
                speed_trend = np.polyfit(range(len(abs_speed_diff)), abs_speed_diff, 1)[0]
                expert_performance_points = np.sum(abs_speed_diff <= EXPERT_SPEED_DIFF_MAX)
                speed_consistency_rate = expert_performance_points / len(abs_speed_diff)
                
                improvement_metrics['speed_difference_mean'] = float(speed_diff_mean)
                improvement_metrics['speed_difference_trend'] = float(speed_trend)
                improvement_metrics['speed_consistency_rate'] = float(speed_consistency_rate)
                improvement_metrics['speed_expert_points'] = int(expert_performance_points)
            else:
                improvement_metrics['speed_difference_mean'] = 0.0
                improvement_metrics['speed_difference_trend'] = 0.0
                improvement_metrics['speed_consistency_rate'] = 0.0
                improvement_metrics['speed_expert_points'] = 0
            
            # Analyze distance to expert line
            distance_to_line = _smooth_series(segment[ContextFeature.DISTANCE_TO_EXPERT_LINE.value])
            if len(distance_to_line) > 1:
                distance_mean = np.mean(distance_to_line)
                distance_trend = np.polyfit(range(len(distance_to_line)), distance_to_line, 1)[0]
                expert_performance_points = np.sum(distance_to_line <= EXPERT_DISTANCE_MAX)
                distance_consistency_rate = expert_performance_points / len(distance_to_line)
                
                improvement_metrics['distance_to_line_mean'] = float(distance_mean)
                improvement_metrics['distance_to_line_trend'] = float(distance_trend)
                improvement_metrics['distance_consistency_rate'] = float(distance_consistency_rate)
                improvement_metrics['distance_expert_points'] = int(expert_performance_points)
            else:
                improvement_metrics['distance_to_line_mean'] = 0.0
                improvement_metrics['distance_to_line_trend'] = 0.0
                improvement_metrics['distance_consistency_rate'] = 0.0
                improvement_metrics['distance_expert_points'] = 0
            
            # Calculate BOTH improvement and consistency rates for ALL metrics regardless of their mode
            
            # For improvement rate - check ALL metrics for improvement (trends)
            velocity_improvement = velocity_trend > 0 if len(velocity_alignment) > 1 else False
            speed_improvement = speed_trend < 0 if len(speed_diff) > 1 else False  # Decreasing difference is improvement
            distance_improvement = distance_trend < 0 if len(distance_to_line) > 1 else False  # Getting closer is improvement
            
            improvement_criteria = [velocity_improvement, speed_improvement, distance_improvement]
            improvement_metrics['overall_improvement_rate'] = sum(improvement_criteria) / len(improvement_criteria)
            
            # For consistency rate - use already calculated individual consistency rates directly
            consistency_rates = [
                improvement_metrics['velocity_consistency_rate'],
                improvement_metrics['speed_consistency_rate'], 
                improvement_metrics['distance_consistency_rate']
            ]
            improvement_metrics['overall_consistency_rate'] = sum(consistency_rates) / len(consistency_rates)
            
        except Exception as e:
            raise Exception(f"Error analyzing segment improvement: {e}")
        
        return improvement_metrics
    
    def serialize_learning_model(self) -> Dict[str, Any]:
        """
        Memory-efficient serialization of trained models stored in the position learner
        
        Returns:
            Dictionary with serialized models ready for storage/transmission
        """
        if not self.position_learner.position_model:
            raise ValueError("No trained models available to serialize. Train models first.")
        
        print("[INFO] Serializing current position models (memory-efficient)...")
        
        try:
            # Build result structure directly without deep copying the entire model
            result = {}
            
            # Serialize models individually to avoid holding multiple copies in memory
            if 'models' in self.position_learner.position_model:
                print("[INFO] Serializing position models...")
                serialized_models = {}
                
                for model_name, model in self.position_learner.position_model['models'].items():
                    print(f"[INFO] Serializing model: {model_name}")
                    # Serialize directly without copying
                    serialized_model_data = self.serialize_data(model)
                    serialized_models[model_name] = serialized_model_data
                    # Force garbage collection of intermediate objects
                    import gc
                    gc.collect()
                
                result['models'] = serialized_models
                
                # Serialize position scaler separately
                if 'position_scaler' in self.position_learner.position_model:
                    print("[INFO] Serializing position scaler...")
                    result['position_scaler'] = self.serialize_data(self.position_learner.position_model['position_scaler'])
            else:
                raise ValueError("Invalid model structure - expected models in position_model")
            
            # Copy only metadata (lightweight) - no deep copy needed
            for key in ['performance_metrics', 'input_features', 'target_features']:
                if key in self.position_learner.position_model:
                    result[key] = self.position_learner.position_model[key]
            
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
            
            # The serialized_results should be the direct position model structure
            if 'models' in serialized_results:
                print("[INFO] Deserializing position models...")
                # Deserialize each model individually
                deserialized_position_models = {}
                for model_name, serialized_model in serialized_results['models'].items():
                    print(f"[INFO] Deserializing model: {model_name}")
                    deserialized_model = self.deserialize_data(serialized_model)
                    deserialized_position_models[model_name] = deserialized_model
                    
                # Deserialize position scaler if present
                deserialized_scaler = None
                if 'position_scaler' in serialized_results:
                    deserialized_scaler = self.deserialize_data(serialized_results['position_scaler'])
                
                # Construct the complete position model structure
                self.position_learner.position_model = {
                    'models': deserialized_position_models,
                    'position_scaler': deserialized_scaler,
                    'performance_metrics': serialized_results.get('performance_metrics', {}),
                    'input_features': serialized_results.get('input_features', ['normalized_position']),
                    'target_features': serialized_results.get('target_features', [])
                }
                
                print(f"[INFO] Successfully deserialized and loaded {len(deserialized_position_models)} models")
                print(f"[INFO] - Model names: {list(deserialized_position_models.keys())}")
            else:
                raise ValueError("No models found in serialized data")
                
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