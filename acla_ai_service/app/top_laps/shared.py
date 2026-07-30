"""Shared top-lap payload and reference-feature operations.

This module is intentionally independent of telemetry storage and training so
both the training service and deployed runtime can use the same serialization
and feature calculations.
"""

from __future__ import annotations

import base64
import io
import pickle
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from app.top_laps.model import (
    TopLapEntry,
    TopLapStore,
    _compute_avg_grip_int,
)
from app.shared.expert_features import ExpertFeatureCatalog


def bucket_key_from_dataframe(df: pd.DataFrame) -> Tuple[str, str, int]:
    """Return the exact track/car and averaged grip bucket for telemetry."""

    if "Static_track" not in df.columns or df["Static_track"].empty:
        raise ValueError("Static_track required to derive bucket key")
    track = str(df["Static_track"].iloc[0])

    if "Static_car_model" in df.columns and not df["Static_car_model"].empty:
        car = str(df["Static_car_model"].iloc[0])
    else:
        car = "unknown_car"

    if "Graphics_track_grip_status" in df.columns:
        grip_arr = pd.to_numeric(
            df["Graphics_track_grip_status"], errors="coerce"
        ).to_numpy(dtype=float)
        avg_grip_int = _compute_avg_grip_int(grip_arr)
    else:
        avg_grip_int = 2

    return track, car, avg_grip_int


def encode_components(data: Dict[str, Any]) -> str:
    """Encode one top-lap component dictionary for backend storage."""

    buffer = io.BytesIO()
    pickle.dump(data, buffer, protocol=pickle.HIGHEST_PROTOCOL)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def decode_components(model_data: str) -> Dict[str, Any]:
    """Decode and minimally type-check one backend top-lap entry."""

    if not isinstance(model_data, str):
        raise ValueError("Top-lap entry must be a base64 string")
    try:
        decoded = base64.b64decode(model_data.encode("utf-8"), validate=True)
        components = pickle.loads(decoded)
    except Exception as exc:
        raise ValueError("Top-lap entry is not valid encoded model data") from exc
    if not isinstance(components, dict):
        raise ValueError("Decoded top-lap entry must be an object")
    return components


def _validate_entry(entry: TopLapEntry) -> None:
    if (
        not isinstance(entry.track, str)
        or not entry.track
        or not isinstance(entry.car, str)
        or not entry.car
    ):
        raise ValueError("Top-lap entry requires track and car")
    if not 0 <= entry.avg_grip_int <= 6:
        raise ValueError("Top-lap grip bucket must be between 0 and 6")
    if entry.x.ndim != 1 or entry.x.size == 0:
        raise ValueError("Top-lap entry requires at least one position")
    if np.any(np.diff(entry.x) <= 0):
        raise ValueError("Top-lap positions must be strictly increasing")
    if len(set(entry.target_features)) != len(entry.target_features):
        raise ValueError("Top-lap target features must be unique")
    if entry.y.ndim != 2 or entry.y.shape != (
        entry.x.size,
        len(entry.target_features),
    ):
        raise ValueError("Top-lap entry feature dimensions are invalid")
    if not np.all(np.isfinite(entry.x)) or not np.all(np.isfinite(entry.y)):
        raise ValueError("Top-lap entry contains non-finite values")


def serialize_top_lap_store(store: TopLapStore) -> Dict[str, Any]:
    """Serialize a store using the ``top_lap_store`` payload."""

    if not store.entries:
        raise ValueError("No stored top laps to serialize. Record laps first.")

    serialized_entries: Dict[str, str] = {}
    for (track, car, grip), entry in store.entries.items():
        key_str = f"{track}|{car}|grip{grip}"
        serialized_entries[key_str] = encode_components(entry.to_components())
    return {"top_lap_store": serialized_entries}


def deserialize_top_lap_store(
    payload: Dict[str, Any],
    *,
    logger=None,
) -> TopLapStore:
    """Validate and build a new store without mutating an installed store."""

    if not isinstance(payload, dict):
        raise ValueError("Top-lap reference payload must be an object")
    serialized_entries = payload.get("top_lap_store")
    if not isinstance(serialized_entries, dict) or not serialized_entries:
        raise ValueError("No top_lap_store found in serialized data")

    store = TopLapStore(logger=logger)
    for key_str, serialized_components in serialized_entries.items():
        if not isinstance(key_str, str):
            raise ValueError("Top-lap store keys must be strings")
        components = decode_components(serialized_components)
        try:
            entry = TopLapEntry.from_components(components)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Top-lap entry {key_str!r} has invalid components"
            ) from exc
        _validate_entry(entry)
        key = (entry.track, entry.car, int(entry.avg_grip_int))
        if key in store.entries:
            raise ValueError(f"Duplicate top-lap entry for {key}")
        store.entries[key] = entry

    return store


def calculate_reference_features(
    store: TopLapStore,
    telemetry_data: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Calculate the classifier's reference fields for each telemetry row."""

    ExpertFeatures = ExpertFeatureCatalog.ExpertFeatures
    EO = ExpertFeatureCatalog.ExpertOptimalFeature

    if not telemetry_data:
        return []
    if not store.entries:
        raise ValueError("No stored top-lap references available")

    processed_df = pd.DataFrame(telemetry_data)
    if "Graphics_normalized_car_position" not in processed_df.columns:
        raise ValueError(
            "Graphics_normalized_car_position required for reference extraction"
        )

    track, car, avg_grip_int = bucket_key_from_dataframe(processed_df)
    batch_predictions = store.predict(
        track,
        car,
        avg_grip_int,
        processed_df["Graphics_normalized_car_position"].values,
    )
    if not isinstance(batch_predictions, list):
        batch_predictions = [batch_predictions]

    reference_feature_rows: List[Dict[str, Any]] = []
    for i, row_predictions in enumerate(batch_predictions):
        current_row = processed_df.iloc[i]
        curr_velocity = np.array(
            [
                float(current_row.get("Physics_velocity_x", 0.0)),
                float(current_row.get("Physics_velocity_y", 0.0)),
                float(current_row.get("Physics_velocity_z", 0.0)),
            ]
        )
        exp_velocity = np.array(
            [
                float(
                    row_predictions.get(
                        EO.EXPERT_OPTIMAL_VELOCITY_X.value, curr_velocity[0]
                    )
                ),
                float(
                    row_predictions.get(
                        EO.EXPERT_OPTIMAL_VELOCITY_Y.value, curr_velocity[1]
                    )
                ),
                float(
                    row_predictions.get(
                        EO.EXPERT_OPTIMAL_VELOCITY_Z.value, curr_velocity[2]
                    )
                ),
            ]
        )
        curr_mag = float(np.linalg.norm(curr_velocity))
        exp_mag = float(np.linalg.norm(exp_velocity))
        if curr_mag > 1e-6 and exp_mag > 1e-6:
            velocity_alignment = float(
                np.dot(curr_velocity / curr_mag, exp_velocity / exp_mag)
            )
        else:
            velocity_alignment = 0.0

        current_pos = np.array(
            [
                float(current_row.get("Graphics_player_pos_x", 0.0)),
                float(current_row.get("Graphics_player_pos_y", 0.0)),
                float(current_row.get("Graphics_player_pos_z", 0.0)),
            ]
        )
        current_speed = float(current_row.get("Physics_speed_kmh", curr_mag))
        current_time = float(current_row.get("Graphics_current_time", 0.0))

        expert_pos = np.array(
            [
                float(
                    row_predictions.get(
                        EO.EXPERT_OPTIMAL_PLAYER_POS_X.value, current_pos[0]
                    )
                ),
                float(
                    row_predictions.get(
                        EO.EXPERT_OPTIMAL_PLAYER_POS_Y.value, current_pos[1]
                    )
                ),
                float(
                    row_predictions.get(
                        EO.EXPERT_OPTIMAL_PLAYER_POS_Z.value, current_pos[2]
                    )
                ),
            ]
        )
        expert_speed = float(
            row_predictions.get(EO.EXPERT_OPTIMAL_SPEED.value, exp_mag)
        )
        expert_time = float(
            row_predictions.get(EO.EXPERT_OPTIMAL_TIME.value, current_time)
        )
        expert_throttle = float(
            row_predictions.get(EO.EXPERT_OPTIMAL_THROTTLE.value, 0.0)
        )
        expert_brake = float(
            row_predictions.get(EO.EXPERT_OPTIMAL_BRAKE.value, 0.0)
        )
        expert_gear = float(
            row_predictions.get(EO.EXPERT_OPTIMAL_GEAR.value, 0.0)
        )

        reference_feature_rows.append(
            {
                ExpertFeatures.EXPERT_OPTIMAL_PLAYER_POS_X.value: float(
                    expert_pos[0]
                ),
                ExpertFeatures.EXPERT_OPTIMAL_PLAYER_POS_Y.value: float(
                    expert_pos[1]
                ),
                ExpertFeatures.EXPERT_OPTIMAL_PLAYER_POS_Z.value: float(
                    expert_pos[2]
                ),
                ExpertFeatures.EXPERT_OPTIMAL_SPEED.value: expert_speed,
                ExpertFeatures.EXPERT_OPTIMAL_TIME.value: expert_time,
                ExpertFeatures.EXPERT_OPTIMAL_THROTTLE.value: expert_throttle,
                ExpertFeatures.EXPERT_OPTIMAL_BRAKE.value: expert_brake,
                ExpertFeatures.EXPERT_OPTIMAL_GEAR.value: expert_gear,
                ExpertFeatures.EXPERT_VELOCITY_ALIGNMENT.value: velocity_alignment,
                ExpertFeatures.SPEED_DIFFERENCE.value: float(
                    expert_speed - current_speed
                ),
                ExpertFeatures.EXPERT_TIME_DIFFERENCE.value: float(
                    current_time - expert_time
                ),
                ExpertFeatures.DISTANCE_TO_EXPERT_LINE.value: float(
                    np.linalg.norm(expert_pos - current_pos)
                ),
            }
        )

    return reference_feature_rows


__all__ = [
    "bucket_key_from_dataframe",
    "calculate_reference_features",
    "decode_components",
    "deserialize_top_lap_store",
    "encode_components",
    "serialize_top_lap_store",
]
