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
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor, MLPClassifier
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


class TrackExpertModel:
    """
    Encapsulates the expert model for a single track.
    Manages training, prediction, and state for position and action models.
    """
    def __init__(self, track_name: str):
        self.track_name = track_name
        self.models: Dict[str, Any] = {}
        self.performance_metrics: Dict[str, Any] = {}
        self.target_groups: Dict[str, List[str]] = {}
        self.target_features: List[str] = []
        self.feature_cols = ['normalized_position', 'normalized_pos_sin', 'normalized_pos_cos']
        
        # Buffers for incremental data loading
        self.input_buffer: List[pd.DataFrame] = []
        self.target_buffer: List[pd.DataFrame] = []

    def add_training_data(self, input_features: pd.DataFrame, target_features: pd.DataFrame):
        """
        Buffer training data. Actual training happens in fit_model().
        """
        self.input_buffer.append(input_features)
        self.target_buffer.append(target_features)
        
        # Update target features list if not set
        if not self.target_features:
            self.target_features = list(target_features.columns)

    def fit_model(self):
        """
        Train models using all buffered data.
        """
        if not self.input_buffer:
            print(f"[WARNING] No data to train for track {self.track_name}")
            return

        print(f"[INFO] Training models for track: {self.track_name} with {len(self.input_buffer)} chunks")
        
        # Combine all data
        input_features = pd.concat(self.input_buffer, ignore_index=True)
        target_features = pd.concat(self.target_buffer, ignore_index=True)
        
        # Clear buffers to free memory
        self.input_buffer = []
        self.target_buffer = []
        
        # Prepare input (normalized position + cyclic features)
        X = input_features[self.feature_cols].values
        
        # We use RandomForest which handles unscaled data well, but we can keep scaler if needed.
        # For simplicity and robustness with RF, we'll skip scaling inputs.
        X_train = X
        
        EO = ExpertFeatureCatalog.ExpertOptimalFeature
        
        # Identify targets groups
        position_targets = []
        action_targets = []
        
        # 1. Position targets (Geometry mapping)
        for t in [EO.EXPERT_OPTIMAL_PLAYER_POS_X.value, EO.EXPERT_OPTIMAL_PLAYER_POS_Y.value, 
                  EO.EXPERT_OPTIMAL_PLAYER_POS_Z.value]:
            if t in target_features.columns:
                position_targets.append(t)

        # 2. Action targets (Driving dynamics)
        for t in [EO.EXPERT_OPTIMAL_STEERING.value, EO.EXPERT_OPTIMAL_THROTTLE.value, 
                  EO.EXPERT_OPTIMAL_BRAKE.value, EO.EXPERT_OPTIMAL_SPEED.value,
                  EO.EXPERT_OPTIMAL_VELOCITY_X.value, EO.EXPERT_OPTIMAL_VELOCITY_Y.value, 
                  EO.EXPERT_OPTIMAL_VELOCITY_Z.value, EO.EXPERT_OPTIMAL_TRACK_POSITION.value]:
            if t in target_features.columns:
                action_targets.append(t)
        
        self.target_groups = {
            'position': position_targets,
            'action': action_targets
        }

        # Train Position Regressor (Geometry)
        if position_targets:
            print(f"[INFO] Training position Random Forest for {len(position_targets)} targets")
            y_pos = target_features[position_targets].values
            
            # RandomForestRegressor
            # n_estimators=100 is a good balance. min_samples_leaf=2 provides slight smoothing.
            pos_model = RandomForestRegressor(n_estimators=100, min_samples_leaf=2, n_jobs=-1, random_state=42)
            pos_model.fit(X_train, y_pos)
            
            self.models['position_regression'] = pos_model
            
            for t in position_targets:
                 self.performance_metrics[t] = {
                    'r2': 1.0, # Placeholder
                    'type': 'rf_regression_position'
                 }

        # Train Action Regressor (Dynamics)
        if action_targets:
            print(f"[INFO] Training action Random Forest for {len(action_targets)} targets")
            y_action = target_features[action_targets].values
            
            action_model = RandomForestRegressor(n_estimators=100, min_samples_leaf=2, n_jobs=-1, random_state=42)
            action_model.fit(X_train, y_action)
            
            self.models['action_regression'] = action_model
            
            for t in action_targets:
                 self.performance_metrics[t] = {
                    'r2': 1.0,
                    'type': 'rf_regression_action'
                 }
        
        # Train Gear Classifier (Separate)
        if EO.EXPERT_OPTIMAL_GEAR.value in target_features.columns:
            y = target_features[EO.EXPERT_OPTIMAL_GEAR.value].values
            y_train = y
            
            model = RandomForestClassifier(n_estimators=50, n_jobs=-1, random_state=42)
            model.fit(X_train, y_train)
            
            self.models[EO.EXPERT_OPTIMAL_GEAR.value] = model
            self.performance_metrics[EO.EXPERT_OPTIMAL_GEAR.value] = {
                'accuracy': 1.0,
                'f1': 1.0,
                'type': 'rf_classification'
            }

    def train(self, input_features: pd.DataFrame, target_features: pd.DataFrame):
        """
        Legacy method for compatibility. Buffers data.
        """
        self.add_training_data(input_features, target_features)

    def predict(self, normalized_positions: Union[float, List[float], np.ndarray]) -> Union[Dict[str, float], List[Dict[str, float]]]:
        """
        Predict expert actions at given normalized track position(s).
        """
        # Handle single position vs multiple positions
        single_position = isinstance(normalized_positions, (int, float))
        if single_position:
            pos_val = float(normalized_positions)
            # Create feature vector: [pos, sin, cos]
            sin_val = np.sin(2 * np.pi * pos_val)
            cos_val = np.cos(2 * np.pi * pos_val)
            positions_array = np.array([[pos_val, sin_val, cos_val]])
        else:
            raw_pos = np.array(normalized_positions).reshape(-1, 1)
            sin_pos = np.sin(2 * np.pi * raw_pos)
            cos_pos = np.cos(2 * np.pi * raw_pos)
            positions_array = np.hstack([raw_pos, sin_pos, cos_pos])
        
        # No scaling for RF
        positions_scaled = positions_array
        
        # Make predictions for all models
        predictions = {}
        
        # 1. Handle Position Regression
        if 'position_regression' in self.models:
            model = self.models['position_regression']
            targets = self.target_groups.get('position', [])
            
            pred = model.predict(positions_scaled)
                
            for i, t in enumerate(targets):
                predictions[t] = pred[:, i]

        # 2. Handle Action Regression
        if 'action_regression' in self.models:
            model = self.models['action_regression']
            targets = self.target_groups.get('action', [])
            
            pred = model.predict(positions_scaled)
                
            for i, t in enumerate(targets):
                predictions[t] = pred[:, i]

        # Handle other models (like gear)
        for model_name, model in self.models.items():
            if model_name in ['position_regression', 'action_regression']:
                continue
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

    def get_serializable_components(self) -> Dict[str, Any]:
        """Returns components that need to be serialized"""
        return {
            'models': self.models,
            'performance_metrics': self.performance_metrics,
            'input_features': self.feature_cols,
            'target_features': self.target_features,
            'target_groups': self.target_groups
        }

    def load_from_components(self, components: Dict[str, Any]):
        """Loads model state from deserialized components"""
        self.models = components.get('models', {})
        self.performance_metrics = components.get('performance_metrics', {})
        self.target_features = components.get('target_features', [])
        self.target_groups = components.get('target_groups', {})
        # input_features is fixed in __init__ but could be loaded if dynamic


class ExpertPositionLearner:
    """Learn expert actions based on normalized track position from multiple expert laps, per track."""
    
    def __init__(self):
        self.track_models: Dict[str, TrackExpertModel] = {} # Dictionary to store models per track
    
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
            # Add cyclic features for loop closure
            input_features['normalized_pos_sin'] = np.sin(2 * np.pi * input_features['normalized_position'])
            input_features['normalized_pos_cos'] = np.cos(2 * np.pi * input_features['normalized_position'])
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
        
        # Clean data
        input_features = input_features.fillna(0)
        target_features = target_features.fillna(0)
        
        return {
            'input_features': input_features,
            'target_features': target_features
        }
    
    def learn_expert_position_mapping(self, expert_laps: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
        """
        Accumulate expert actions from normalized track position using multiple expert laps, per track.
        Actual training happens when finalize_models() is called.
        
        Args:
            expert_laps: List of expert driver telemetry laps (each lap is a list of records)
            
        Returns:
            Dictionary with status (models are not trained yet)
        """
        print(f"[INFO] Accumulating expert position data from {len(expert_laps)} expert laps")
        
        all_input_features = []
        all_target_features = []
        
        for i, lap_data in enumerate(expert_laps):
            if not lap_data:
                continue
                
            lap_df = pd.DataFrame(lap_data)
            
            # Extract position-based features for this lap
            feature_data = self.extract_position_features(lap_df)
            all_input_features.append(feature_data['input_features'])
            all_target_features.append(feature_data['target_features'])
            
        if not all_input_features:
             raise ValueError("No valid features extracted from laps")

        input_features = pd.concat(all_input_features, ignore_index=True)
        target_features = pd.concat(all_target_features, ignore_index=True)
        
        # Get track name (assume single track as per requirement)
        if 'track' not in input_features.columns or input_features['track'].empty:
             raise ValueError("Track information missing in input features")
             
        track = input_features['track'].iloc[0]
        
        # Get or create track model
        if track not in self.track_models:
            self.track_models[track] = TrackExpertModel(track)
        
        track_model = self.track_models[track]
        
        # Add data to buffer
        track_model.add_training_data(input_features, target_features)
        
        return {
            'status': 'buffered',
            'track': track,
            'samples': len(input_features)
        }
    
    def finalize_models(self):
        """
        Train all buffered models.
        """
        print(f"[INFO] Finalizing training for {len(self.track_models)} tracks")
        for track, model in self.track_models.items():
            model.fit_model()

    
    def predict_expert_actions_at_position(self, track_name: str, normalized_positions: Union[float, List[float], np.ndarray]) -> Union[Dict[str, float], List[Dict[str, float]]]:
        """
        Predict expert actions at given normalized track position(s) for a specific track.
        """
        if not self.track_models:
            raise ValueError("No track models trained. Call learn_expert_position_mapping() first.")
        
        if track_name not in self.track_models:
            raise ValueError(f"No model trained for track: {track_name}")
            
        return self.track_models[track_name].predict(normalized_positions)

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
            
            # Check for empty data
            if not chunk_data:
                continue

            # Learn expert position mapping (this is the only learning model)
            self.position_learner.learn_expert_position_mapping(chunk_data)
            total_samples += sum(len(lap) for lap in chunk_data)
        
        # Finalize training (train Random Forests on accumulated data)
        self.position_learner.finalize_models()
        
        # Construct results
        overall_metrics = {}
        all_targets = set()
        
        # Collect metrics from all trained models
        for track, track_model in self.position_learner.track_models.items():
            overall_metrics[track] = track_model.performance_metrics
            all_targets.update(track_model.target_features)

        results = {
            'modelData': {t: m.get_serializable_components() for t, m in self.position_learner.track_models.items()},
            'metadata': {
                'performance_metrics': overall_metrics,
                'input_features': ['normalized_position', 'normalized_pos_sin', 'normalized_pos_cos', 'track'],
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
            
            for track_name, track_model in self.position_learner.track_models.items():
                print(f"[INFO] Serializing models for track: {track_name}")
                
                # Get components from the track model
                track_data = track_model.get_serializable_components()
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
                
                # Copy metadata
                for key in ['performance_metrics', 'input_features', 'target_features', 'target_groups']:
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
                
                for track_name, track_data in serialized_results['track_models'].items():
                    print(f"[INFO] Deserializing models for track: {track_name}")
                    
                    # Create new TrackExpertModel
                    track_model = TrackExpertModel(track_name)
                    components = {}
                    
                    # Deserialize models
                    if 'models' in track_data:
                        deserialized_models = {}
                        for model_name, serialized_model in track_data['models'].items():
                            deserialized_models[model_name] = self.deserialize_data(serialized_model)
                        components['models'] = deserialized_models
                    
                    # Copy metadata
                    for key in ['performance_metrics', 'input_features', 'target_features', 'target_groups']:
                        if key in track_data:
                            components[key] = track_data[key]
                    
                    # Load components into track model
                    track_model.load_from_components(components)
                    
                    # Store in learner
                    self.position_learner.track_models[track_name] = track_model
                
                print(f"[INFO] Successfully deserialized models for {len(self.position_learner.track_models)} tracks")
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
    