"""
Imitation "learning" service for ACC telemetry.

This is a memory-based registry, not a model. We store the fastest lap per
``(track, car, avg_grip_int)`` bucket (filled by the cleaning stage of the
training pipeline) and answer queries by 1-D interpolating that one lap's
telemetry against ``normalized_position``.
"""

import base64
import io
import logging
import pickle
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore', category=UserWarning)

from app.domain.expert_features import ExpertFeatureCatalog
from app.ml.imitation.model import (
    FastestLapEntry,
    FastestLapStore,
    _compute_avg_grip_int,
    _format_debug_message,
)


def _bucket_key_from_records(records: List[Dict[str, Any]]) -> Tuple[str, str, int]:
    """(track, car, avg_grip_int) for a batch / lap of records."""
    if not records:
        raise ValueError("Cannot derive bucket key from empty records")
    track = records[0].get("Static_track", "unknown_track")
    car = records[0].get("Static_car_model", "unknown_car")
    grip_values = [
        r.get("Graphics_track_grip_status")
        for r in records
        if r.get("Graphics_track_grip_status") is not None
    ]
    avg_grip_int = _compute_avg_grip_int(np.asarray(grip_values, dtype=float))
    return (track, car, avg_grip_int)


def _bucket_key_from_dataframe(df: pd.DataFrame) -> Tuple[str, str, int]:
    if "Static_track" not in df.columns or df["Static_track"].empty:
        raise ValueError("Static_track required to derive bucket key")
    track = str(df["Static_track"].iloc[0])

    if "Static_car_model" in df.columns and not df["Static_car_model"].empty:
        car = str(df["Static_car_model"].iloc[0])
    else:
        car = "unknown_car"

    if "Graphics_track_grip_status" in df.columns:
        grip_arr = pd.to_numeric(df["Graphics_track_grip_status"], errors="coerce").to_numpy(dtype=float)
        avg_grip_int = _compute_avg_grip_int(grip_arr)
    else:
        avg_grip_int = 2

    return (track, car, avg_grip_int)


class ExpertImitateLearningService:
    """Backed by a FastestLapStore; signature-compatible with previous service."""

    def __init__(self, models_directory: str = "imitation_models", *, debug: bool = False, logger: Optional[logging.Logger] = None):
        self.models_directory = Path(models_directory)
        self.models_directory.mkdir(exist_ok=True)

        self.logger = logger or logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.debug_enabled = debug
        self.fastest_lap_store = FastestLapStore(
            debug=debug,
            debug_logger=self._debug,
            logger=self.logger,
        )

        self.logger.info(
            "ImitationLearningService initialized. Models directory: %s",
            self.models_directory,
        )

    def _debug(self, message: str, **debug_data: Any) -> None:
        if not self.debug_enabled:
            return
        self.logger.debug(_format_debug_message(message, debug_data if debug_data else None))

    def get_shared_data_cache(self):
        from app.storage import get_shared_telemetry_store
        return get_shared_telemetry_store()

    async def train_ai_model(self, top_laps_cache_key: str) -> Dict[str, Any]:
        """Load cached top laps and store the fastest one per (track, car, grip)."""
        self.logger.info("Recording fastest laps from cache: %s", top_laps_cache_key)

        telemetry_store = self.get_shared_data_cache()
        if not telemetry_store.has_cached_data(top_laps_cache_key):
            raise ValueError(f"No cached top laps found at key: {top_laps_cache_key}")

        chunks_iterator = telemetry_store.get_cached_data_chunks(
            cache_key=top_laps_cache_key, include_ids=True
        )

        total_samples = 0
        recorded_keys: List[Tuple[str, str, int]] = []

        for chunk_tuple in chunks_iterator:
            chunk_data, _chunk_id = chunk_tuple
            if not chunk_data:
                continue

            for lap_records in chunk_data:
                key = self.fastest_lap_store.record_lap(lap_records)
                if key is not None:
                    recorded_keys.append(key)
                total_samples += len(lap_records)

        all_targets = set()
        for entry in self.fastest_lap_store.entries.values():
            all_targets.update(entry.target_features)

        results = {
            'modelData': {
                f"{t}|{c}|grip{g}": entry.to_components()
                for (t, c, g), entry in self.fastest_lap_store.entries.items()
            },
            'metadata': {
                'input_features': ['normalized_position', 'track', 'car', 'avg_grip_int'],
                'target_features': sorted(all_targets),
                'buckets_recorded': [
                    {'track': t, 'car': c, 'avg_grip_int': g}
                    for (t, c, g) in recorded_keys
                ],
                'total_training_samples': total_samples,
            },
        }
        results['learning_summary'] = self._generate_learning_summary(results)
        return results

    def predict_expert_actions(self, processed_df: pd.DataFrame) -> Dict[str, Any]:
        """Look up the stored fastest lap for the batch's (track, car, grip) and sample it."""
        if not self.fastest_lap_store.entries:
            self.logger.warning("No stored fastest laps available")
            return {"error": "No stored fastest laps available"}

        if processed_df.empty:
            return {"error": "Empty input dataframe"}

        if 'Graphics_normalized_car_position' not in processed_df.columns:
            return {"error": "Graphics_normalized_car_position not found in input data"}

        try:
            track, car, avg_grip_int = _bucket_key_from_dataframe(processed_df)
        except ValueError as e:
            return {"error": str(e)}

        normalized_positions = processed_df['Graphics_normalized_car_position'].values
        try:
            optimal_actions = self.fastest_lap_store.predict(
                track, car, avg_grip_int, normalized_positions
            )
        except KeyError as e:
            return {"error": str(e)}

        if isinstance(optimal_actions, list) and optimal_actions:
            averaged = {
                key: float(np.mean([row[key] for row in optimal_actions]))
                for key in optimal_actions[0].keys()
            }
            return {'optimal_actions': averaged}
        return {'optimal_actions': optimal_actions}

    def extract_expert_state_for_telemetry(self, telemetry_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Enrich each telemetry row with the stored fastest lap's values and deltas."""
        ExpertFeatures = ExpertFeatureCatalog.ExpertFeatures
        EO = ExpertFeatureCatalog.ExpertOptimalFeature

        if not telemetry_data:
            return []
        if not self.fastest_lap_store.entries:
            raise ValueError(
                "No stored fastest laps. Train or deserialize before calling "
                "extract_expert_state_for_telemetry()."
            )

        processed_df = pd.DataFrame(telemetry_data)
        if 'Graphics_normalized_car_position' not in processed_df.columns:
            raise ValueError("Graphics_normalized_car_position required for expert state extraction")

        track, car, avg_grip_int = _bucket_key_from_dataframe(processed_df)
        try:
            batch_predictions = self.fastest_lap_store.predict(
                track, car, avg_grip_int,
                processed_df['Graphics_normalized_car_position'].values,
            )
        except KeyError as e:
            raise ValueError(str(e)) from e

        if not isinstance(batch_predictions, list):
            batch_predictions = [batch_predictions]

        expert_feature_rows: List[Dict[str, Any]] = []
        for i, row_predictions in enumerate(batch_predictions):
            current_row = processed_df.iloc[i]
            row_features: Dict[str, Any] = {}

            curr_velocity = np.array([
                float(current_row.get('Physics_velocity_x', 0.0)),
                float(current_row.get('Physics_velocity_y', 0.0)),
                float(current_row.get('Physics_velocity_z', 0.0)),
            ])
            exp_velocity = np.array([
                float(row_predictions.get(EO.EXPERT_OPTIMAL_VELOCITY_X.value, curr_velocity[0])),
                float(row_predictions.get(EO.EXPERT_OPTIMAL_VELOCITY_Y.value, curr_velocity[1])),
                float(row_predictions.get(EO.EXPERT_OPTIMAL_VELOCITY_Z.value, curr_velocity[2])),
            ])
            curr_mag = float(np.linalg.norm(curr_velocity))
            exp_mag = float(np.linalg.norm(exp_velocity))
            if curr_mag > 1e-6 and exp_mag > 1e-6:
                velocity_alignment = float(np.dot(curr_velocity / curr_mag, exp_velocity / exp_mag))
            else:
                velocity_alignment = 0.0

            current_pos = np.array([
                float(current_row.get('Graphics_player_pos_x', 0.0)),
                float(current_row.get('Graphics_player_pos_y', 0.0)),
                float(current_row.get('Graphics_player_pos_z', 0.0)),
            ])
            current_speed = float(current_row.get('Physics_speed_kmh', curr_mag))
            current_time = float(current_row.get('Graphics_current_time', 0.0))

            expert_pos = np.array([
                float(row_predictions.get(EO.EXPERT_OPTIMAL_PLAYER_POS_X.value, current_pos[0])),
                float(row_predictions.get(EO.EXPERT_OPTIMAL_PLAYER_POS_Y.value, current_pos[1])),
                float(row_predictions.get(EO.EXPERT_OPTIMAL_PLAYER_POS_Z.value, current_pos[2])),
            ])
            expert_speed = float(row_predictions.get(EO.EXPERT_OPTIMAL_SPEED.value, exp_mag))
            expert_time = float(row_predictions.get(EO.EXPERT_OPTIMAL_TIME.value, current_time))
            expert_throttle = float(row_predictions.get(EO.EXPERT_OPTIMAL_THROTTLE.value, 0.0))
            expert_brake = float(row_predictions.get(EO.EXPERT_OPTIMAL_BRAKE.value, 0.0))
            expert_gear = float(row_predictions.get(EO.EXPERT_OPTIMAL_GEAR.value, 0.0))

            row_features[ExpertFeatures.EXPERT_OPTIMAL_PLAYER_POS_X.value] = float(expert_pos[0])
            row_features[ExpertFeatures.EXPERT_OPTIMAL_PLAYER_POS_Y.value] = float(expert_pos[1])
            row_features[ExpertFeatures.EXPERT_OPTIMAL_PLAYER_POS_Z.value] = float(expert_pos[2])
            row_features[ExpertFeatures.EXPERT_OPTIMAL_SPEED.value] = expert_speed
            row_features[ExpertFeatures.EXPERT_OPTIMAL_TIME.value] = expert_time
            row_features[ExpertFeatures.EXPERT_OPTIMAL_THROTTLE.value] = expert_throttle
            row_features[ExpertFeatures.EXPERT_OPTIMAL_BRAKE.value] = expert_brake
            row_features[ExpertFeatures.EXPERT_OPTIMAL_GEAR.value] = expert_gear
            row_features[ExpertFeatures.EXPERT_VELOCITY_ALIGNMENT.value] = velocity_alignment
            row_features[ExpertFeatures.SPEED_DIFFERENCE.value] = float(expert_speed - current_speed)
            row_features[ExpertFeatures.EXPERT_TIME_DIFFERENCE.value] = float(current_time - expert_time)
            row_features[ExpertFeatures.DISTANCE_TO_EXPERT_LINE.value] = float(np.linalg.norm(expert_pos - current_pos))

            expert_feature_rows.append(row_features)

        self.logger.info(
            "Completed expert state extraction for %d records (bucket=%s)",
            len(expert_feature_rows),
            (track, car, avg_grip_int),
        )
        return expert_feature_rows

    def _generate_learning_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        metadata = results.get('metadata', {})
        buckets = metadata.get('buckets_recorded', [])
        tracks = {b['track'] for b in buckets}
        cars = {b['car'] for b in buckets}
        return {
            'timestamp': datetime.now().isoformat(),
            'learning_completed': ['fastest_lap_registry'] if buckets else [],
            'position_summary': {
                'buckets_recorded': len(buckets),
                'tracks_recorded': len(tracks),
                'cars_recorded': len(cars),
                'input_features': len(metadata.get('input_features', [])),
                'target_features': len(metadata.get('target_features', [])),
                'total_training_samples': metadata.get('total_training_samples', 0),
            },
        }

    def serialize_learning_model(self) -> Dict[str, Any]:
        """Serialize the fastest-lap store for backend storage."""
        if not self.fastest_lap_store.entries:
            raise ValueError("No stored fastest laps to serialize. Record laps first.")

        self.logger.info("Serializing fastest-lap store")
        serialized_entries: Dict[str, str] = {}
        for (track, car, grip), entry in self.fastest_lap_store.entries.items():
            key_str = f"{track}|{car}|grip{grip}"
            serialized_entries[key_str] = self.serialize_data(entry.to_components())
        return {'fastest_lap_store': serialized_entries}

    def deserialize_imitation_model(self, serialized_results: Dict[str, Any]) -> 'ExpertImitateLearningService':
        """Rebuild the in-memory fastest-lap store from a serialized payload."""
        self.logger.info("Deserializing fastest-lap store")

        payload = serialized_results.get('fastest_lap_store')
        if payload is None:
            raise ValueError("No fastest_lap_store found in serialized data")

        self.fastest_lap_store.entries.clear()
        for _key_str, serialized_components in payload.items():
            components = self.deserialize_data(serialized_components)
            entry = FastestLapEntry.from_components(components)
            key = (entry.track, entry.car, int(entry.avg_grip_int))
            self.fastest_lap_store.entries[key] = entry

        self.logger.info(
            "Loaded %d fastest-lap entries",
            len(self.fastest_lap_store.entries),
        )
        return self

    def serialize_data(self, data: Any) -> str:
        buffer = io.BytesIO()
        pickle.dump(data, buffer, protocol=pickle.HIGHEST_PROTOCOL)
        return base64.b64encode(buffer.getvalue()).decode('utf-8')

    def deserialize_data(self, model_data: str) -> Dict[str, Any]:
        decoded = base64.b64decode(model_data.encode('utf-8'))
        return pickle.loads(decoded)


if __name__ == "__main__":
    service = ExpertImitateLearningService()
