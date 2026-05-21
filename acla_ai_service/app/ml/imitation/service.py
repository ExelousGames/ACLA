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
import logging
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from datetime import datetime
from pathlib import Path

# Scikit-learn imports for trajectory learning
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


warnings.filterwarnings('ignore', category=UserWarning)

# Domain types moved to app/domain/ in refactor/hexagonal-v1 Step 2.
# Re-imported here so internal references and external callers (transformer_model,
# segment_models, telemetry_segment_visualizer) keep working unchanged.
from app.domain.expert_features import (
    ExpertFeatureCatalog,
    ExpertModelTrainingConfig,
    SegmentImprovementConfig,
    SegmentImprovementSummary,
)


# Model classes extracted to app/ml/imitation/model.py in refactor/hexagonal-v4.
# Re-imported here so ExpertImitateLearningService and external callers see
# the same surface.
from app.ml.imitation.model import (
    ExpertPositionLearner,
    TrackExpertModel,
    _format_debug_message,
)


# --- Removed (now in model.py) ---


class ExpertImitateLearningService:
    """Main imitation learning service that focuses on trajectory optimization"""
    
    def __init__(self, models_directory: str = "imitation_models", *, debug: bool = False, logger: Optional[logging.Logger] = None):
        """
        Initialize the imitation learning service
        
        Args:
            models_directory: Directory to save/load trained imitation models
        """
        self.models_directory = Path(models_directory)
        self.models_directory.mkdir(exist_ok=True)
        
        self.logger = logger or logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.debug_enabled = debug
        self.position_learner = ExpertPositionLearner(
            debug=debug,
            debug_logger=self._debug,
            logger=self.logger,
        )
        
        self.logger.info(
            "ImitationLearningService initialized. Models directory: %s",
            self.models_directory,
        )
    
    def get_shared_data_cache(self):
        """Get shared data cache instance"""
        from app.storage import get_shared_telemetry_store
        return get_shared_telemetry_store()

    async def train_ai_model(self, top_laps_cache_key: str) -> Dict[str, Any]:
        """
        Learn from expert driving demonstrations using cached top laps.
        
        Args:
            top_laps_cache_key: Cache key for top laps data
            
        Returns:
            Dictionary with trained models and learning insights, serialized objects and ready for storage
        """
        
        self.logger.info(
            "Learning from cached top laps: %s",
            top_laps_cache_key,
        )

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
            self.logger.warning("No position models available")
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
            if self.debug_enabled:
                self._debug("Prediction failure", error=str(e))
            raise Exception(f"[WARNING] Could not predict expert actions: {e}")
        
        # If no specific models are available, provide error
        if not predictions or all('error' in v for v in predictions.values() if isinstance(v, dict)):
            raise Exception("[Error] No valid model available for predictions")
        
        return predictions

    def _debug(self, message: str, **debug_data: Any) -> None:
        if not self.debug_enabled:
            return
        formatted = _format_debug_message(message, debug_data if debug_data else None)
        self.logger.debug(formatted)
    
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
                        # Current time from telemetry (usually ms)
                        current_time = float(current_row.get('Graphics_current_time', 0.0))

                        # Store expert optimal predictions for visualization with safe fallbacks
                        expert_pos_x = float(row_predictions.get(ExpertFeatureCatalog.ExpertOptimalFeature.EXPERT_OPTIMAL_PLAYER_POS_X.value, current_pos_x))
                        expert_pos_y = float(row_predictions.get(ExpertFeatureCatalog.ExpertOptimalFeature.EXPERT_OPTIMAL_PLAYER_POS_Y.value, current_pos_y))
                        expert_pos_z = float(row_predictions.get(ExpertFeatureCatalog.ExpertOptimalFeature.EXPERT_OPTIMAL_PLAYER_POS_Z.value, current_pos_z))

                        row_features[ExpertFeatures.EXPERT_OPTIMAL_PLAYER_POS_X.value] = expert_pos_x
                        row_features[ExpertFeatures.EXPERT_OPTIMAL_PLAYER_POS_Y.value] = expert_pos_y
                        row_features[ExpertFeatures.EXPERT_OPTIMAL_PLAYER_POS_Z.value] = expert_pos_z

                        expert_speed = float(row_predictions.get(ExpertFeatureCatalog.ExpertOptimalFeature.EXPERT_OPTIMAL_SPEED.value, exp_velocity_magnitude))
                        row_features[ExpertFeatures.EXPERT_OPTIMAL_SPEED.value] = expert_speed

                        # Expert time
                        expert_time = float(row_predictions.get(ExpertFeatureCatalog.ExpertOptimalFeature.EXPERT_OPTIMAL_TIME.value, current_time))
                        row_features[ExpertFeatures.EXPERT_OPTIMAL_TIME.value] = expert_time

                        # Add expert throttle and brake predictions
                        expert_throttle = float(row_predictions.get(ExpertFeatureCatalog.ExpertOptimalFeature.EXPERT_OPTIMAL_THROTTLE.value, 0.0))
                        row_features[ExpertFeatures.EXPERT_OPTIMAL_THROTTLE.value] = expert_throttle

                        expert_brake = float(row_predictions.get(ExpertFeatureCatalog.ExpertOptimalFeature.EXPERT_OPTIMAL_BRAKE.value, 0.0))
                        row_features[ExpertFeatures.EXPERT_OPTIMAL_BRAKE.value] = expert_brake

                        expert_gear = float(row_predictions.get(ExpertFeatureCatalog.ExpertOptimalFeature.EXPERT_OPTIMAL_GEAR.value, 0.0))
                        row_features[ExpertFeatures.EXPERT_OPTIMAL_GEAR.value] = expert_gear

                        # Store only velocity alignment feature
                        row_features[ExpertFeatures.EXPERT_VELOCITY_ALIGNMENT.value] = float(velocity_alignment)

                        # Calculate speed difference
                        speed_difference = expert_speed - current_speed
                        row_features[ExpertFeatures.SPEED_DIFFERENCE.value] = float(speed_difference)

                        # Calculate time difference
                        # If expert_time is greater (player is faster? no, normalized position is same)
                        # Time at position P:
                        # If Player Time < Expert Time -> Player reached P faster -> Good for player
                        # If Player Time > Expert Time -> Player reached P slower -> Bad for player
                        # Difference = Player Time - Expert Time
                        # Negative diff = Player ahead
                        # Positive diff = Player behind
                        time_difference = current_time - expert_time
                        row_features[ExpertFeatures.EXPERT_TIME_DIFFERENCE.value] = float(time_difference)

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

        self.logger.info(
            "Completed expert state extraction for %d records",
            len(expert_feature_rows),
        )
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
        
        self.logger.info("Serializing current position models (memory-efficient)")
        
        try:
            # Build result structure directly without deep copying the entire model
            result = {}
            
            # Serialize models per track
            serialized_tracks = {}
            
            for track_name, track_model in self.position_learner.track_models.items():
                self.logger.info("Serializing models for track: %s", track_name)
                
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
                # Copy metadata
                for key in ['performance_metrics', 'input_features', 'target_features', 'target_groups', 'spline_targets']:
                    if key in track_data:
                        serialized_track_data[key] = track_data[key]
                
                serialized_tracks[track_name] = serialized_track_data
            
            result['track_models'] = serialized_tracks
            
            self.logger.info("Serialization completed successfully")
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
            self.logger.info("Deserializing imitation models")
            
            if 'track_models' in serialized_results:
                self.logger.info("Deserializing track models")
                
                for track_name, track_data in serialized_results['track_models'].items():
                    self.logger.info("Deserializing models for track: %s", track_name)
                    
                    # Create new TrackExpertModel
                    track_model = TrackExpertModel(
                        track_name,
                        debug=self.debug_enabled,
                        debug_logger=self._debug,
                        logger=self.logger,
                    )
                    components = {}
                    
                    # Deserialize models
                    if 'models' in track_data:
                        deserialized_models = {}
                        for model_name, serialized_model in track_data['models'].items():
                            deserialized_models[model_name] = self.deserialize_data(serialized_model)
                        components['models'] = deserialized_models
                    
                    # Copy metadata
                    # Copy metadata
                    for key in ['performance_metrics', 'input_features', 'target_features', 'target_groups', 'spline_targets']:
                        if key in track_data:
                            components[key] = track_data[key]
                    track_model.load_from_components(components)
                    
                    # Store in learner
                    self.position_learner.track_models[track_name] = track_model
                
                self.logger.info(
                    "Successfully deserialized models for %d tracks",
                    len(self.position_learner.track_models),
                )
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
            self.logger.error("Failed to serialize models: %s", e)
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
    