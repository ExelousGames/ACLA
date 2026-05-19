"""Pure imitation-learning model code.

Two classes:
  - ``TrackExpertModel``: per-track interpolation model. Buffers
    expert telemetry frames, then on ``fit_model()`` builds a
    1-D interpolation table keyed on ``normalized_position`` for
    every available expert target feature (speed, throttle, brake,
    positions, velocities, gear, time).
  - ``ExpertPositionLearner``: multi-track manager owning a dict
    of TrackExpertModel keyed by track name; orchestrates the
    add-data / fit / predict cycle across tracks.

Pure leaves: imports only ``numpy``, ``pandas``, and ``app.domain``.
No I/O, no orchestration. ``ExpertImitateLearningService`` (which
owns checkpoint serialisation + the broader training flow) lives in
``app.ml.imitation.service`` and imports these back.

Extracted from app/ml/imitation/service.py in refactor/hexagonal-v4
(Page 5 of acla-ai-service-architecture.drawio).
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from app.domain.expert_features import ExpertFeatureCatalog, ExpertModelTrainingConfig


def _format_debug_message(message: str, debug_data: Optional[Dict[str, Any]] = None) -> str:
    if not debug_data:
        return message
    kv_pairs = ', '.join(f"{key}={value}" for key, value in debug_data.items())
    return f"{message} | {kv_pairs}"


class TrackExpertModel:
    """
    Encapsulates the expert model for a single track.
    Manages training, prediction, and state for position and action models.
    """
    def __init__(self, track_name: str, *, debug: bool = False, debug_logger: Optional[Callable[..., None]] = None, logger: Optional[logging.Logger] = None):
        self.track_name = track_name
        self.models: Dict[str, Any] = {}
        self.performance_metrics: Dict[str, Any] = {}
        self.target_groups: Dict[str, List[str]] = {}
        self.target_features: List[str] = []
        self.feature_cols = ['normalized_position']
        
        # Buffers for incremental data loading
        self.input_buffer: List[pd.DataFrame] = []
        self.target_buffer: List[pd.DataFrame] = []
        self.debug_enabled = debug
        self._debug_logger = debug_logger
        self.logger = logger or logging.getLogger(f"{__name__}.TrackExpertModel")
        self.config = ExpertModelTrainingConfig()

    def _debug(self, message: str, **debug_data: Any) -> None:
        if not self.debug_enabled:
            return
        if self._debug_logger:
            self._debug_logger(message, **debug_data)
        else:
            self.logger.debug(_format_debug_message(message, debug_data))

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
            self.logger.warning("No data to train for track %s", self.track_name)
            return

        num_laps = len(self.input_buffer)
        self.logger.info(
            "Training models for track %s with %d chunks",
            self.track_name,
            num_laps,
        )
        
        # Combine all data
        input_features = pd.concat(self.input_buffer, ignore_index=True)
        target_features = pd.concat(self.target_buffer, ignore_index=True)

        # Clear buffers to free memory
        self.input_buffer = []
        self.target_buffer = []
        
        EO = ExpertFeatureCatalog.ExpertOptimalFeature
        
        # Identify targets groups
        spline_targets = []
        
        # Collect all available targets for spline interpolation
        all_possible_targets = [
            EO.EXPERT_OPTIMAL_PLAYER_POS_X.value, EO.EXPERT_OPTIMAL_PLAYER_POS_Y.value, 
            EO.EXPERT_OPTIMAL_PLAYER_POS_Z.value,
            EO.EXPERT_OPTIMAL_STEERING.value, EO.EXPERT_OPTIMAL_THROTTLE.value, 
            EO.EXPERT_OPTIMAL_BRAKE.value, EO.EXPERT_OPTIMAL_SPEED.value,
            EO.EXPERT_OPTIMAL_VELOCITY_X.value, EO.EXPERT_OPTIMAL_VELOCITY_Y.value, 
            EO.EXPERT_OPTIMAL_VELOCITY_Z.value, EO.EXPERT_OPTIMAL_TRACK_POSITION.value,
            EO.EXPERT_OPTIMAL_GEAR.value, EO.EXPERT_OPTIMAL_TIME.value
        ]
        
        for t in all_possible_targets:
            if t in target_features.columns:
                spline_targets.append(t)
        
        self.target_groups = {
            'decision_tree': spline_targets
        }
        self.spline_targets = spline_targets
        # Clear legacy groups
        self.position_targets = []
        self.nn_targets = []

        # Prepare data: Sort by normalized_position and average duplicates
        # We need 'normalized_position' from input_features
        
        train_df = pd.DataFrame()
        train_df['x'] = input_features['normalized_position']
        
        # Add all targets to train_df for grouping
        for t in self.spline_targets:
            train_df[t] = target_features[t]
            
        # Group by x and mean to handle duplicates and ensure unique x for interpolation
        train_df_grouped = train_df.groupby('x', as_index=False).mean()

        # --- 1. Train Interpolation Model (and Decision Tree as fallback) ---
        if self.spline_targets:
            
            X_train = train_df_grouped['x'].values
            Y_train = train_df_grouped[self.spline_targets].values
            
            # Store data for interpolation
            self.models['interpolation_data'] = {
                'x': X_train,
                'y': Y_train,
                'targets': self.spline_targets
            }
            
            for t in self.spline_targets:
                 self.performance_metrics[t] = {
                    'r2': 1.0,
                    'type': 'interpolation'
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
            x_query = np.array([pos_val])
        else:
            x_query = np.array(normalized_positions)
        
        predictions = {}
        
        # 1. Interpolation Prediction (Preferred)
        if 'interpolation_data' in self.models:
            data = self.models['interpolation_data']
            x_ref = data['x']
            y_ref = data['y']
            targets = data['targets']
            
            for i, target in enumerate(targets):
                # Use interpolation
                pred_values = np.interp(x_query, x_ref, y_ref[:, i])
                
                # Handle categorical targets
                if target == ExpertFeatureCatalog.ExpertOptimalFeature.EXPERT_OPTIMAL_GEAR.value:
                    pred_values = np.round(pred_values)
                
                predictions[target] = pred_values
        
        if self.debug_enabled:
            input_count = len(x_query)
            debug_payload: Dict[str, Any] = {
                'track': self.track_name,
                'input_count': input_count,
                'single_position': single_position
            }
            if input_count:
                debug_payload['min_query'] = float(np.min(x_query))
                debug_payload['max_query'] = float(np.max(x_query))

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
            for i in range(len(x_query)):
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
            'target_groups': self.target_groups,
            'spline_targets': getattr(self, 'spline_targets', [])
        }

    def load_from_components(self, components: Dict[str, Any]):
        """Loads model state from deserialized components"""
        self.models = components.get('models', {})
        self.performance_metrics = components.get('performance_metrics', {})
        self.target_features = components.get('target_features', [])
        self.target_groups = components.get('target_groups', {})
        self.spline_targets = components.get('spline_targets', [])


class ExpertPositionLearner:
    """Learn expert actions based on normalized track position from multiple expert laps, per track."""
    
    def __init__(self, *, debug: bool = False, debug_logger: Optional[Callable[..., None]] = None, logger: Optional[logging.Logger] = None):
        self.track_models: Dict[str, TrackExpertModel] = {} # Dictionary to store models per track
        self.debug_enabled = debug
        self._debug_logger = debug_logger
        self.logger = logger or logging.getLogger(f"{__name__}.ExpertPositionLearner")

    def _debug(self, message: str, **debug_data: Any) -> None:
        if not self.debug_enabled:
            return
        if self._debug_logger:
            self._debug_logger(message, **debug_data)
        else:
            self.logger.debug(_format_debug_message(message, debug_data))
    
    def extract_features(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Extract features for expert learning
        
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

        # Time (lap time at position)
        if 'Graphics_current_time' in df.columns:
             target_features[EO.EXPERT_OPTIMAL_TIME.value] = df['Graphics_current_time']

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
        self.logger.info(
            "Accumulating expert position data from %d expert laps",
            len(expert_laps),
        )
        
        all_input_features = []
        all_target_features = []
        
        for i, lap_data in enumerate(expert_laps):
            if not lap_data:
                continue
                
            lap_df = pd.DataFrame(lap_data)
            
            # Extract position-based features for this lap
            feature_data = self.extract_features(lap_df)
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
            self.track_models[track] = TrackExpertModel(
                track,
                debug=self.debug_enabled,
                debug_logger=self._debug,
                logger=self.logger,
            )
        
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
        self.logger.info(
            "Finalizing training for %d tracks",
            len(self.track_models),
        )
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
            
        predictions = self.track_models[track_name].predict(normalized_positions)

        return predictions

__all__ = [
    "TrackExpertModel",
    "ExpertPositionLearner",
]
