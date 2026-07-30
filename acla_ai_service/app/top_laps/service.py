"""
Top-lap reference model service for ACC telemetry.

This is a memory-based registry. We store the selected top lap per
``(track, car, avg_grip_int)`` bucket (filled by the cleaning stage of the
training pipeline) and answer queries by 1-D interpolating that one lap's
telemetry against ``normalized_position``.
"""

import logging
import warnings
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore', category=UserWarning)

from app.top_laps.model import (
    TopLapStore,
    _format_debug_message,
)
from app.top_laps.shared import (
    bucket_key_from_dataframe,
    calculate_reference_features,
    deserialize_top_lap_store,
    serialize_top_lap_store,
)


class TopLapReferenceModelService:
    """Build and query a normalized-position top-lap reference store."""

    def __init__(
        self,
        *,
        debug: bool = False,
        logger: Optional[logging.Logger] = None,
    ):
        self.logger = logger or logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.debug_enabled = debug
        self.top_lap_store = TopLapStore(
            debug=debug,
            debug_logger=self._debug,
            logger=self.logger,
        )

        self.logger.info("TopLapReferenceModelService initialized")

    def _debug(self, message: str, **debug_data: Any) -> None:
        if not self.debug_enabled:
            return
        self.logger.debug(_format_debug_message(message, debug_data if debug_data else None))

    def get_shared_data_cache(self):
        from app.storage import get_shared_telemetry_store
        return get_shared_telemetry_store()

    async def build_from_cached_top_laps(
        self,
        top_laps_cache_key: str,
    ) -> Dict[str, Any]:
        """Load cached top laps and store one per (track, car, grip)."""
        self.logger.info(
            "Building top-lap references from cache: %s",
            top_laps_cache_key,
        )

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
                key = self.top_lap_store.record_lap(lap_records)
                if key is not None:
                    recorded_keys.append(key)
                total_samples += len(lap_records)

        all_targets = set()
        for entry in self.top_lap_store.entries.values():
            all_targets.update(entry.target_features)

        results = {
            'modelData': {
                f"{t}|{c}|grip{g}": entry.to_components()
                for (t, c, g), entry in self.top_lap_store.entries.items()
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
        results['reference_summary'] = self._generate_reference_summary(results)
        return results

    def sample_reference_actions(self, processed_df: pd.DataFrame) -> Dict[str, Any]:
        """Sample the stored top lap for the batch's track, car, and grip."""
        if not self.top_lap_store.entries:
            self.logger.warning("No stored top-lap references available")
            return {"error": "No stored top-lap references available"}

        if processed_df.empty:
            return {"error": "Empty input dataframe"}

        if 'Graphics_normalized_car_position' not in processed_df.columns:
            return {"error": "Graphics_normalized_car_position not found in input data"}

        try:
            track, car, avg_grip_int = bucket_key_from_dataframe(processed_df)
        except ValueError as e:
            return {"error": str(e)}

        normalized_positions = processed_df['Graphics_normalized_car_position'].values
        try:
            reference_actions = self.top_lap_store.predict(
                track, car, avg_grip_int, normalized_positions
            )
        except KeyError as e:
            return {"error": str(e)}

        if isinstance(reference_actions, list) and reference_actions:
            averaged = {
                key: float(np.mean([row[key] for row in reference_actions]))
                for key in reference_actions[0].keys()
            }
            return {'optimal_actions': averaged}
        return {'optimal_actions': reference_actions}

    def extract_reference_features(
        self,
        telemetry_data: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Calculate reference fields and deltas for each telemetry row."""
        reference_feature_rows = calculate_reference_features(
            self.top_lap_store,
            telemetry_data,
        )
        self.logger.info(
            "Completed top-lap reference extraction for %d records",
            len(reference_feature_rows),
        )
        return reference_feature_rows

    def _generate_reference_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        metadata = results.get('metadata', {})
        buckets = metadata.get('buckets_recorded', [])
        tracks = {b['track'] for b in buckets}
        cars = {b['car'] for b in buckets}
        return {
            'timestamp': datetime.now().isoformat(),
            'reference_built': ['top_lap_store'] if buckets else [],
            'position_summary': {
                'buckets_recorded': len(buckets),
                'tracks_recorded': len(tracks),
                'cars_recorded': len(cars),
                'input_features': len(metadata.get('input_features', [])),
                'target_features': len(metadata.get('target_features', [])),
                'total_training_samples': metadata.get('total_training_samples', 0),
            },
        }

    def serialize_reference_model(self) -> Dict[str, Any]:
        """Serialize the top-lap store for backend storage."""
        self.logger.info("Serializing top-lap store")
        return serialize_top_lap_store(self.top_lap_store)

    def load_reference_model(
        self,
        serialized_results: Dict[str, Any],
    ) -> 'TopLapReferenceModelService':
        """Rebuild the in-memory top-lap store from a serialized payload."""
        self.logger.info("Loading top-lap store")

        self.top_lap_store = deserialize_top_lap_store(
            serialized_results,
            logger=self.logger,
        )

        self.logger.info(
            "Loaded %d top-lap entries",
            len(self.top_lap_store.entries),
        )
        return self

if __name__ == "__main__":
    service = TopLapReferenceModelService()
