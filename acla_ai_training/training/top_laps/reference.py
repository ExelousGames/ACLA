"""Training-local reference payload types and decoding for position preparation.

Copied from the AI service top-lap model, shared payload reader, and expert
feature catalog. Keep the backend component format and feature names compatible;
this module does not need the live service package.
"""

from __future__ import annotations

import base64
import logging
import pickle
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


class ExpertFeatureCatalog:
    """Expert column names in published reference-lap payloads."""

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
        EXPERT_OPTIMAL_TIME = 'expert_optimal_time'


def _compute_avg_grip_int(values: np.ndarray) -> int:
    """Average grip status across a lap and clamp to int in [0, 6]."""
    if values is None or len(values) == 0:
        return 2
    arr = np.asarray(values, dtype=float)
    mean = float(np.nanmean(arr)) if arr.size else float("nan")
    if np.isnan(mean):
        return 2
    return max(0, min(6, int(round(mean))))


def _resolve_game(game: Optional[str], record: Dict[str, Any]) -> str:
    # References and telemetry cached before game was recorded were ACC-only.
    value = game if game is not None else record.get("game_recorded_from", "acc")
    if not isinstance(value, str) or not value.strip():
        raise ValueError("game_recorded_from is required to derive the reference game")
    return value


class TopLapEntry:
    """One stored top lap with sorted/deduped x and y for np.interp lookup."""

    def __init__(
        self,
        *,
        game: str = "acc",
        track: str,
        car: str,
        avg_grip_int: int,
        x: np.ndarray,
        y: np.ndarray,
        target_features: List[str],
    ):
        self.game = game
        self.track = track
        self.car = car
        self.avg_grip_int = avg_grip_int
        self.x = np.asarray(x, dtype=float)
        self.y = np.asarray(y, dtype=float)
        self.target_features = list(target_features)

    def to_components(self) -> Dict[str, Any]:
        return {
            "game": self.game,
            "track": self.track,
            "car": self.car,
            "avg_grip_int": int(self.avg_grip_int),
            "x": self.x,
            "y": self.y,
            "target_features": list(self.target_features),
        }

    @classmethod
    def from_components(cls, components: Dict[str, Any]) -> "TopLapEntry":
        return cls(
            game=components.get("game", "acc"),
            track=components["track"],
            car=components["car"],
            avg_grip_int=int(components["avg_grip_int"]),
            x=np.asarray(components["x"], dtype=float),
            y=np.asarray(components["y"], dtype=float),
            target_features=list(components.get("target_features", [])),
        )


class TopLapStore:
    """Registry of top laps keyed by (game, track, car, avg_grip_int)."""

    def __init__(
        self,
        *,
        debug: bool = False,
        debug_logger: Optional[Callable[..., None]] = None,
        logger: Optional[logging.Logger] = None,
    ):
        self.entries: Dict[Tuple[str, str, str, int], TopLapEntry] = {}
        self.debug_enabled = debug
        self._debug_logger = debug_logger
        self.logger = logger or logging.getLogger(f"{__name__}.TopLapStore")


def bucket_key_from_dataframe(
    df: pd.DataFrame, *, game: Optional[str] = None,
) -> Tuple[str, str, str, int]:
    """Return the exact game/track/car and averaged grip bucket for telemetry."""

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

    return _resolve_game(game, df.iloc[0].to_dict()), track, car, avg_grip_int


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
        not isinstance(entry.game, str)
        or not entry.game.strip()
        or not isinstance(entry.track, str)
        or not entry.track
        or not isinstance(entry.car, str)
        or not entry.car
    ):
        raise ValueError("Top-lap entry requires game, track and car")
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
        key = (entry.game, entry.track, entry.car, int(entry.avg_grip_int))
        if key in store.entries:
            raise ValueError(f"Duplicate top-lap entry for {key}")
        store.entries[key] = entry

    return store
