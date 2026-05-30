"""Fastest-lap registry for expert imitation.

Replaces the previous interpolation-based expert model. We store the single
fastest lap per ``(track, car, avg_grip_int)`` combination and answer
position-based queries by 1-D interpolating that one lap's telemetry against
``normalized_position``.

Pure leaves: imports only ``numpy``, ``pandas``, and ``app.domain``.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from app.domain.expert_features import ExpertFeatureCatalog


# EXPERT_OPTIMAL_* expert-feature name -> raw telemetry column it sources from.
_EXPERT_TARGET_FROM_TELEMETRY: Dict[str, str] = {
    ExpertFeatureCatalog.ExpertOptimalFeature.EXPERT_OPTIMAL_STEERING.value: "Physics_steer_angle",
    ExpertFeatureCatalog.ExpertOptimalFeature.EXPERT_OPTIMAL_THROTTLE.value: "Physics_gas",
    ExpertFeatureCatalog.ExpertOptimalFeature.EXPERT_OPTIMAL_BRAKE.value: "Physics_brake",
    ExpertFeatureCatalog.ExpertOptimalFeature.EXPERT_OPTIMAL_GEAR.value: "Physics_gear",
    ExpertFeatureCatalog.ExpertOptimalFeature.EXPERT_OPTIMAL_PLAYER_POS_X.value: "Graphics_player_pos_x",
    ExpertFeatureCatalog.ExpertOptimalFeature.EXPERT_OPTIMAL_PLAYER_POS_Y.value: "Graphics_player_pos_y",
    ExpertFeatureCatalog.ExpertOptimalFeature.EXPERT_OPTIMAL_PLAYER_POS_Z.value: "Graphics_player_pos_z",
    ExpertFeatureCatalog.ExpertOptimalFeature.EXPERT_OPTIMAL_VELOCITY_X.value: "Physics_velocity_x",
    ExpertFeatureCatalog.ExpertOptimalFeature.EXPERT_OPTIMAL_VELOCITY_Y.value: "Physics_velocity_y",
    ExpertFeatureCatalog.ExpertOptimalFeature.EXPERT_OPTIMAL_VELOCITY_Z.value: "Physics_velocity_z",
    ExpertFeatureCatalog.ExpertOptimalFeature.EXPERT_OPTIMAL_SPEED.value: "Physics_speed_kmh",
    ExpertFeatureCatalog.ExpertOptimalFeature.EXPERT_OPTIMAL_TIME.value: "Graphics_current_time",
    ExpertFeatureCatalog.ExpertOptimalFeature.EXPERT_OPTIMAL_TRACK_POSITION.value: "Graphics_normalized_car_position",
}


class NoExpertLapError(KeyError):
    """No fastest lap stored for a (track, car) at any grip bucket."""

    def __init__(self, track: str, car: str, avg_grip_int: int):
        self.track = track
        self.car = car
        self.avg_grip_int = int(avg_grip_int)
        super().__init__(
            f"No fastest lap stored for {(track, car, int(avg_grip_int))}"
        )


def _format_debug_message(message: str, debug_data: Optional[Dict[str, Any]] = None) -> str:
    if not debug_data:
        return message
    kv_pairs = ', '.join(f"{key}={value}" for key, value in debug_data.items())
    return f"{message} | {kv_pairs}"


def _compute_avg_grip_int(values: np.ndarray) -> int:
    """Average grip status across a lap and clamp to int in [0, 6]."""
    if values is None or len(values) == 0:
        return 2
    arr = np.asarray(values, dtype=float)
    mean = float(np.nanmean(arr)) if arr.size else float("nan")
    if np.isnan(mean):
        return 2
    return max(0, min(6, int(round(mean))))


class FastestLapEntry:
    """One stored fastest lap with sorted/deduped x and y for np.interp lookup."""

    def __init__(
        self,
        *,
        track: str,
        car: str,
        avg_grip_int: int,
        x: np.ndarray,
        y: np.ndarray,
        target_features: List[str],
    ):
        self.track = track
        self.car = car
        self.avg_grip_int = avg_grip_int
        self.x = np.asarray(x, dtype=float)
        self.y = np.asarray(y, dtype=float)
        self.target_features = list(target_features)

    @classmethod
    def from_lap_records(
        cls,
        records: List[Dict[str, Any]],
        *,
        track: str,
        car: str,
        avg_grip_int: int,
    ) -> "FastestLapEntry":
        lap_df = pd.DataFrame(records)
        if "Graphics_normalized_car_position" not in lap_df.columns:
            raise ValueError(
                "Graphics_normalized_car_position required to build FastestLapEntry"
            )

        available_targets = [
            (expert_name, raw_col)
            for expert_name, raw_col in _EXPERT_TARGET_FROM_TELEMETRY.items()
            if raw_col in lap_df.columns
        ]

        df = pd.DataFrame(
            {"x": pd.to_numeric(lap_df["Graphics_normalized_car_position"], errors="coerce")}
        )
        for expert_name, raw_col in available_targets:
            df[expert_name] = pd.to_numeric(lap_df[raw_col], errors="coerce")

        df = df.dropna(subset=["x"]).fillna(0.0)
        df = df.groupby("x", as_index=False).mean().sort_values("x")

        target_features = [expert_name for expert_name, _ in available_targets]
        y = df[target_features].to_numpy(dtype=float) if target_features else np.zeros((len(df), 0))

        return cls(
            track=track,
            car=car,
            avg_grip_int=avg_grip_int,
            x=df["x"].to_numpy(dtype=float),
            y=y,
            target_features=target_features,
        )

    def predict(
        self, normalized_positions: Union[float, List[float], np.ndarray]
    ) -> Union[Dict[str, float], List[Dict[str, float]]]:
        single = isinstance(normalized_positions, (int, float))
        if single:
            x_query = np.array([float(normalized_positions)], dtype=float)
        else:
            x_query = np.asarray(normalized_positions, dtype=float)

        gear_value = ExpertFeatureCatalog.ExpertOptimalFeature.EXPERT_OPTIMAL_GEAR.value
        per_target: Dict[str, np.ndarray] = {}
        for i, target in enumerate(self.target_features):
            pred = np.interp(x_query, self.x, self.y[:, i])
            if target == gear_value:
                pred = np.round(pred)
            per_target[target] = pred

        if single:
            return {k: float(v[0]) for k, v in per_target.items()}
        return [
            {k: float(v[i]) for k, v in per_target.items()}
            for i in range(len(x_query))
        ]

    def to_components(self) -> Dict[str, Any]:
        return {
            "track": self.track,
            "car": self.car,
            "avg_grip_int": int(self.avg_grip_int),
            "x": self.x,
            "y": self.y,
            "target_features": list(self.target_features),
        }

    @classmethod
    def from_components(cls, components: Dict[str, Any]) -> "FastestLapEntry":
        return cls(
            track=components["track"],
            car=components["car"],
            avg_grip_int=int(components["avg_grip_int"]),
            x=np.asarray(components["x"], dtype=float),
            y=np.asarray(components["y"], dtype=float),
            target_features=list(components.get("target_features", [])),
        )


class FastestLapStore:
    """Registry of fastest laps keyed by (track, car, avg_grip_int)."""

    def __init__(
        self,
        *,
        debug: bool = False,
        debug_logger: Optional[Callable[..., None]] = None,
        logger: Optional[logging.Logger] = None,
    ):
        self.entries: Dict[Tuple[str, str, int], FastestLapEntry] = {}
        self.debug_enabled = debug
        self._debug_logger = debug_logger
        self.logger = logger or logging.getLogger(f"{__name__}.FastestLapStore")

    def _debug(self, message: str, **debug_data: Any) -> None:
        if not self.debug_enabled:
            return
        if self._debug_logger:
            self._debug_logger(message, **debug_data)
        else:
            self.logger.debug(_format_debug_message(message, debug_data))

    def record_lap(self, records: List[Dict[str, Any]]) -> Optional[Tuple[str, str, int]]:
        """Store one lap; key derived from its records.

        Caller is expected to pass laps already selected as fastest per bucket
        (cleaning.py guarantees this); we overwrite any existing entry.
        """
        if not records:
            return None

        track = records[0].get("Static_track", "unknown_track")
        car = records[0].get("Static_car_model", "unknown_car")
        grip_values = [
            r.get("Graphics_track_grip_status")
            for r in records
            if r.get("Graphics_track_grip_status") is not None
        ]
        avg_grip_int = _compute_avg_grip_int(np.asarray(grip_values, dtype=float))

        entry = FastestLapEntry.from_lap_records(
            records, track=track, car=car, avg_grip_int=avg_grip_int
        )
        key = (track, car, avg_grip_int)
        self.entries[key] = entry
        self._debug(
            "stored fastest lap",
            track=track,
            car=car,
            avg_grip_int=avg_grip_int,
            samples=len(entry.x),
        )
        return key

    def record_laps(self, lap_records_list: List[List[Dict[str, Any]]]) -> List[Tuple[str, str, int]]:
        keys: List[Tuple[str, str, int]] = []
        for lap_records in lap_records_list:
            key = self.record_lap(lap_records)
            if key is not None:
                keys.append(key)
        return keys

    def has(self, track: str, car: str, avg_grip_int: int) -> bool:
        return (track, car, avg_grip_int) in self.entries

    def predict(
        self,
        track: str,
        car: str,
        avg_grip_int: int,
        normalized_positions: Union[float, List[float], np.ndarray],
    ) -> Union[Dict[str, float], List[Dict[str, float]]]:
        key = (track, car, int(avg_grip_int))
        entry = self.entries.get(key)
        if entry is None:
            # Session chunks are arbitrary row slices, so their averaged grip
            # rarely matches a stored fastest lap's grip exactly. Fall back to
            # the closest available grip bucket for the same (track, car).
            candidates = [
                (k, e) for k, e in self.entries.items()
                if k[0] == track and k[1] == car
            ]
            if not candidates:
                raise NoExpertLapError(track, car, avg_grip_int)
            fallback_key, entry = min(
                candidates, key=lambda ke: abs(ke[0][2] - int(avg_grip_int))
            )
            self._debug(
                "grip bucket miss; using nearest grip",
                requested=key,
                fallback=fallback_key,
            )
        return entry.predict(normalized_positions)


__all__ = [
    "FastestLapEntry",
    "FastestLapStore",
    "NoExpertLapError",
    "_compute_avg_grip_int",
    "_format_debug_message",
]
