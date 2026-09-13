"""Inference-only telemetry preprocessing for segment analysis flows."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from app.shared.telemetry import FeatureProcessor, TelemetryFeatures


INFERENCE_SAMPLE_INTERVAL_MS = 500
RAW_ROW_INDEX_COLUMN = "__acla_raw_row_index"
PLAYER_POSITION_COLUMNS = (
    "Graphics_player_pos_x",
    "Graphics_player_pos_y",
    "Graphics_player_pos_z",
)
CAR_POSITION_RE = re.compile(r"^Car_(\d+)_pos_([xyz])$")
MAX_ABS_TRACK_COORDINATE_M = 100_000.0
MAX_PLAYER_POSITION_JUMP_M = 500.0
MAX_PLAYER_POSITION_SPEED_MPS = 250.0
MAX_OPPONENT_POSITION_JUMP_M = 1_000.0
MAX_OPPONENT_DISTANCE_FROM_PLAYER_M = 20_000.0


@dataclass
class InferenceTelemetryBatch:
    """Clean model records and their corresponding raw input row indices."""

    records: List[Dict[str, Any]]
    raw_indices: List[int]


def _coordinate_invalid_mask(
    dataframe: pd.DataFrame,
    columns: Sequence[str],
    *,
    max_abs_coordinate: float = MAX_ABS_TRACK_COORDINATE_M,
) -> pd.Series:
    invalid = pd.Series(False, index=dataframe.index)
    for column in columns:
        values = pd.to_numeric(dataframe[column], errors="coerce")
        invalid |= (
            ~np.isfinite(values.to_numpy(dtype=float))
            | (values.abs() > max_abs_coordinate)
        )
    return invalid


def _isolated_position_spike_mask(
    dataframe: pd.DataFrame,
    columns: Sequence[str],
    *,
    max_jump_m: float,
    max_speed_mps: Optional[float] = None,
) -> pd.Series:
    if len(dataframe) < 3 or len(columns) < 2:
        return pd.Series(False, index=dataframe.index)

    coordinates = (
        dataframe[list(columns)]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(dtype=float)
    )
    finite_rows = np.isfinite(coordinates).all(axis=1)

    previous_jump = np.zeros(len(dataframe), dtype=bool)
    next_jump = np.zeros(len(dataframe), dtype=bool)
    previous_distance = np.linalg.norm(
        coordinates[1:] - coordinates[:-1],
        axis=1,
    )
    finite_pairs = finite_rows[1:] & finite_rows[:-1]
    previous_jump[1:] = finite_pairs & (previous_distance > max_jump_m)
    next_jump[:-1] = previous_jump[1:]

    if max_speed_mps is not None and "Graphics_current_time" in dataframe:
        time_values = pd.to_numeric(
            dataframe["Graphics_current_time"],
            errors="coerce",
        ).to_numpy(dtype=float)
        time_delta_seconds = (time_values[1:] - time_values[:-1]) / 1000.0
        valid_time_delta = (
            np.isfinite(time_delta_seconds)
            & (time_delta_seconds > 0.0)
        )
        speeds = np.divide(
            previous_distance,
            time_delta_seconds,
            out=np.zeros_like(previous_distance),
            where=valid_time_delta,
        )

        speed_bad = np.zeros(len(dataframe), dtype=bool)
        speed_bad[1:] = (
            finite_pairs
            & valid_time_delta
            & (speeds > max_speed_mps)
        )
        speed_bad_next = np.zeros(len(dataframe), dtype=bool)
        speed_bad_next[:-1] = speed_bad[1:]
        previous_jump |= speed_bad
        next_jump |= speed_bad_next

    return pd.Series(previous_jump & next_jump, index=dataframe.index)


def _car_position_slots(dataframe: pd.DataFrame) -> Dict[int, List[str]]:
    slots: Dict[int, List[str]] = {}
    for column in dataframe.columns:
        match = CAR_POSITION_RE.match(str(column))
        if match:
            slots.setdefault(int(match.group(1)), []).append(str(column))
    return slots


def _remove_position_anomalies(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Remove impossible player rows and clear impossible opponent positions."""
    if dataframe is None or dataframe.empty:
        return (
            dataframe.copy()
            if dataframe is not None
            else pd.DataFrame()
        )

    cleaned = dataframe.copy()
    player_columns = [
        column
        for column in PLAYER_POSITION_COLUMNS
        if column in cleaned.columns
    ]
    car_slots = _car_position_slots(cleaned)
    position_columns = player_columns + [
        column
        for columns in car_slots.values()
        for column in columns
    ]

    for column in position_columns:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    if player_columns:
        player_invalid = _coordinate_invalid_mask(cleaned, player_columns)
        player_spikes = _isolated_position_spike_mask(
            cleaned,
            player_columns,
            max_jump_m=MAX_PLAYER_POSITION_JUMP_M,
            max_speed_mps=MAX_PLAYER_POSITION_SPEED_MPS,
        )
        cleaned = cleaned.loc[~(player_invalid | player_spikes)].copy()

    if cleaned.empty:
        return cleaned.reset_index(drop=True)

    player_xy_available = {
        "Graphics_player_pos_x",
        "Graphics_player_pos_y",
    }.issubset(cleaned.columns)

    for slot, columns in car_slots.items():
        available_columns = [
            column for column in columns if column in cleaned.columns
        ]
        if not available_columns:
            continue

        bad_mask = _coordinate_invalid_mask(cleaned, available_columns)
        bad_mask |= _isolated_position_spike_mask(
            cleaned,
            available_columns,
            max_jump_m=MAX_OPPONENT_POSITION_JUMP_M,
        )

        x_column = f"Car_{slot}_pos_x"
        y_column = f"Car_{slot}_pos_y"
        if (
            player_xy_available
            and x_column in cleaned.columns
            and y_column in cleaned.columns
        ):
            opponent_x = pd.to_numeric(
                cleaned[x_column],
                errors="coerce",
            ).to_numpy(dtype=float)
            opponent_y = pd.to_numeric(
                cleaned[y_column],
                errors="coerce",
            ).to_numpy(dtype=float)
            player_x = pd.to_numeric(
                cleaned["Graphics_player_pos_x"],
                errors="coerce",
            ).to_numpy(dtype=float)
            player_y = pd.to_numeric(
                cleaned["Graphics_player_pos_y"],
                errors="coerce",
            ).to_numpy(dtype=float)
            active = (
                ((opponent_x != 0.0) | (opponent_y != 0.0))
                & np.isfinite(opponent_x)
                & np.isfinite(opponent_y)
            )
            player_finite = np.isfinite(player_x) & np.isfinite(player_y)
            distance = np.sqrt(
                (opponent_x - player_x) ** 2
                + (opponent_y - player_y) ** 2
            )
            bad_mask |= pd.Series(
                active
                & player_finite
                & np.isfinite(distance)
                & (distance > MAX_OPPONENT_DISTANCE_FROM_PLAYER_M),
                index=cleaned.index,
            )

        cleaned.loc[bad_mask, available_columns] = 0.0

    return cleaned.reset_index(drop=True)


def preprocess_inference_telemetry(
    telemetry_data: Sequence[Dict[str, Any]],
) -> InferenceTelemetryBatch:
    """Apply the training-equivalent 500 ms preprocessing contract at inference."""
    if not telemetry_data:
        return InferenceTelemetryBatch(records=[], raw_indices=[])

    dataframe = pd.DataFrame([dict(row) for row in telemetry_data])
    dataframe[RAW_ROW_INDEX_COLUMN] = [
        str(index) for index in range(len(dataframe))
    ]

    processor = FeatureProcessor(dataframe)
    processor.general_cleaning_for_analysis()
    processed = processor.flip_y_z_features()
    processed = _remove_position_anomalies(processed)
    processor.df = processed

    if processed.empty:
        return InferenceTelemetryBatch(records=[], raw_indices=[])

    downsampled = processor.strip_dataframe_by_time_gap(
        processed,
        INFERENCE_SAMPLE_INTERVAL_MS,
    )
    downsampled = _remove_position_anomalies(downsampled)

    if downsampled.empty:
        return InferenceTelemetryBatch(records=[], raw_indices=[])

    raw_indices = [
        int(value)
        for value in downsampled[RAW_ROW_INDEX_COLUMN].tolist()
    ]
    feature_names = TelemetryFeatures.get_features_for_top_lap_reference()
    missing_features = [
        feature
        for feature in feature_names
        if feature not in downsampled.columns
    ]
    if missing_features:
        raise ValueError(f"Missing features: {missing_features}")

    filtered = processor.filter_features_by_list(
        downsampled,
        feature_names,
    )
    filtered = filtered.reset_index(drop=True)

    return InferenceTelemetryBatch(
        records=filtered.to_dict("records"),
        raw_indices=raw_indices,
    )


__all__ = [
    "INFERENCE_SAMPLE_INTERVAL_MS",
    "InferenceTelemetryBatch",
    "preprocess_inference_telemetry",
]
