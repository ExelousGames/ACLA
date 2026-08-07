"""Deterministic runtime ranges copied from the annotation lap splitter.

This module intentionally owns its implementation. Runtime inference must not
depend on the annotation-agent tool surface, so the two splitters may evolve
independently after this copy.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from app.shared.circuit_sections import CIRCUIT_SECTION_RANGES
from app.shared.labels import LABEL_MAPPING, LABEL_NAME_TO_ID
from app.shared.telemetry import MAX_CARS


NORMALIZED_POSITION_COLUMN = "Graphics_normalized_car_position"


class RuntimeSegmentSplitError(ValueError):
    """Raised when telemetry cannot be split into supported runtime ranges."""


def _normalise_circuit_id(raw: Any) -> str:
    value = str(raw or "").strip()
    if not value:
        return ""
    if value in LABEL_MAPPING:
        return value
    mapped = LABEL_NAME_TO_ID.get(value)
    if mapped:
        return mapped
    return value.lower().replace(" ", "_").replace("-", "_")


def _static_track(dataframe: pd.DataFrame) -> Optional[str]:
    if "Static_track" not in dataframe.columns:
        return None
    for value in dataframe["Static_track"]:
        if pd.notna(value) and str(value).strip():
            return str(value)
    return None


def resolve_runtime_circuit(
    dataframe: pd.DataFrame,
    circuit_id: Optional[str] = None,
) -> str:
    """Resolve an explicit circuit first, then the telemetry ``Static_track``."""
    raw_circuit = circuit_id if str(circuit_id or "").strip() else _static_track(dataframe)
    resolved = _normalise_circuit_id(raw_circuit)
    if not resolved:
        raise RuntimeSegmentSplitError("circuit could not be resolved from the request or Static_track")
    if not any(
        section_id.rstrip("0123456789") == resolved
        for section_id in CIRCUIT_SECTION_RANGES
    ):
        raise RuntimeSegmentSplitError(f"unsupported circuit: {raw_circuit}")
    return resolved


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or values.size < window:
        return values.astype(float)
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(values.astype(float), kernel, mode="same")


def _cumulative_path_distance(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return np.array([], dtype=float)
    return np.concatenate(([0.0], np.cumsum(np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2))))


def _player_heading(player_x: np.ndarray, player_y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    window = min(5, len(player_x))
    if window % 2 == 0:
        window = max(1, window - 1)
    smoothed_x = _moving_average(player_x, window)
    smoothed_y = _moving_average(player_y, window)
    dx = np.gradient(smoothed_x)
    dy = np.gradient(smoothed_y)
    norm = np.sqrt(dx * dx + dy * dy)
    safe_norm = np.where(norm > 1e-6, norm, 1e-6)
    return dx / safe_norm, dy / safe_norm


def _project_points_to_local_reference_path(
    point_x: np.ndarray,
    point_y: np.ndarray,
    ref_x: np.ndarray,
    ref_y: np.ndarray,
    *,
    center_indices: np.ndarray,
    search_radius: int = 30,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_points = int(point_x.size)
    s_out = np.full(n_points, np.nan, dtype=float)
    d_out = np.full(n_points, np.nan, dtype=float)
    idx_out = np.full(n_points, -1, dtype=int)
    if n_points == 0 or ref_x.size < 2:
        return s_out, d_out, idx_out

    ref_s = _cumulative_path_distance(ref_x, ref_y)
    vx_all = ref_x[1:] - ref_x[:-1]
    vy_all = ref_y[1:] - ref_y[:-1]
    segment_length_squared = vx_all * vx_all + vy_all * vy_all
    max_segment_index = int(ref_x.size) - 2

    for index in range(n_points):
        px = float(point_x[index])
        py = float(point_y[index])
        center = int(center_indices[index]) if index < len(center_indices) else index
        if not (np.isfinite(px) and np.isfinite(py)) or center < 0:
            continue

        lo = max(0, center - int(search_radius))
        hi = min(max_segment_index + 1, center + int(search_radius) + 1)
        if hi <= lo:
            lo = max(0, min(max_segment_index, center))
            hi = min(max_segment_index + 1, lo + 1)

        vx = vx_all[lo:hi]
        vy = vy_all[lo:hi]
        length_squared = segment_length_squared[lo:hi]
        wx = px - ref_x[lo:hi]
        wy = py - ref_y[lo:hi]
        projection = np.divide(
            wx * vx + wy * vy,
            length_squared,
            out=np.zeros_like(length_squared),
            where=length_squared > 1e-9,
        )
        projection = np.clip(projection, 0.0, 1.0)
        projected_x = ref_x[lo:hi] + projection * vx
        projected_y = ref_y[lo:hi] + projection * vy
        distance_squared = (px - projected_x) ** 2 + (py - projected_y) ** 2
        local_segment = int(np.nanargmin(distance_squared))
        segment_index = lo + local_segment
        segment_length = float(np.sqrt(max(segment_length_squared[segment_index], 0.0)))
        s_out[index] = float(ref_s[segment_index] + projection[local_segment] * segment_length)
        cross = (
            vx_all[segment_index] * (py - projected_y[local_segment])
            - vy_all[segment_index] * (px - projected_x[local_segment])
        )
        sign = 1.0 if cross >= 0.0 else -1.0
        d_out[index] = sign * float(np.sqrt(distance_squared[local_segment]))
        idx_out[index] = segment_index

    return s_out, d_out, idx_out


def _relative_position_frame(
    player_x: np.ndarray,
    player_y: np.ndarray,
    opponent_x: np.ndarray,
    opponent_y: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    if player_x.size >= 3:
        window = min(5, int(player_x.size))
        if window % 2 == 0:
            window = max(1, window - 1)
        ref_x = _moving_average(player_x, window)
        ref_y = _moving_average(player_y, window)
        player_s, player_d, player_indices = _project_points_to_local_reference_path(
            player_x,
            player_y,
            ref_x,
            ref_y,
            center_indices=np.arange(player_x.size),
            search_radius=3,
        )
        if np.isfinite(player_s).any():
            centers = np.where(player_indices >= 0, player_indices, np.arange(player_x.size))
            opponent_s, opponent_d, _ = _project_points_to_local_reference_path(
                opponent_x,
                opponent_y,
                ref_x,
                ref_y,
                center_indices=centers,
                search_radius=30,
            )
            return (
                opponent_s - player_s,
                opponent_d - player_d,
                player_s,
                player_d,
                "player_local_path_projection",
            )

    heading_x, heading_y = _player_heading(player_x, player_y)
    delta_x = opponent_x - player_x
    delta_y = opponent_y - player_y
    player_s = _cumulative_path_distance(player_x, player_y)
    return (
        delta_x * heading_x + delta_y * heading_y,
        delta_x * (-heading_y) + delta_y * heading_x,
        player_s,
        np.zeros_like(player_s),
        "player_heading_projection",
    )


def _active_opponent_mask(
    opponent_x: np.ndarray,
    opponent_y: np.ndarray,
    player_x: np.ndarray,
    player_y: np.ndarray,
    *,
    same_car_tolerance_m: float = 0.25,
) -> np.ndarray:
    active = (
        ((opponent_x != 0.0) | (opponent_y != 0.0))
        & np.isfinite(opponent_x)
        & np.isfinite(opponent_y)
    )
    if not active.any() or player_x.size != opponent_x.size:
        return active
    distance = np.sqrt((opponent_x - player_x) ** 2 + (opponent_y - player_y) ** 2)
    finite = active & np.isfinite(distance)
    if finite.any() and float((distance[finite] <= same_car_tolerance_m).mean()) >= 0.95:
        return np.zeros_like(active, dtype=bool)
    return active


def _relative_long_gap_velocity(
    segment: pd.DataFrame,
    signed_long_gap: np.ndarray,
    finite: np.ndarray,
) -> Tuple[np.ndarray, str]:
    velocity = np.full_like(signed_long_gap, np.nan, dtype=float)
    if signed_long_gap.size < 2:
        return velocity, "m/sample"
    delta_gap = signed_long_gap[1:] - signed_long_gap[:-1]
    valid_pairs = finite[1:] & finite[:-1]
    if "Graphics_current_time" in segment.columns:
        time_ms = pd.to_numeric(segment["Graphics_current_time"], errors="coerce").to_numpy(dtype=float)
        delta_seconds = (time_ms[1:] - time_ms[:-1]) / 1000.0
        valid_time = np.isfinite(delta_seconds) & (delta_seconds > 1e-6)
        np.divide(
            delta_gap,
            delta_seconds,
            out=velocity[1:],
            where=valid_pairs & valid_time,
        )
        return velocity, "m/s"
    velocity[1:] = np.where(valid_pairs, delta_gap, np.nan)
    return velocity, "m/sample"


def _relative_velocity_threshold(
    units: str,
    n_rows: int,
    *,
    min_relative_velocity_mps: float,
    min_role_gain_m: float,
) -> float:
    if units == "m/s":
        return float(min_relative_velocity_mps)
    return float(min_role_gain_m) / max(1, int(n_rows) - 1)


def _nan_percentile(values: np.ndarray, percentile: float) -> Optional[float]:
    finite = values[np.isfinite(values)]
    return float(np.nanpercentile(finite, percentile)) if finite.size else None


def _contiguous_true_runs(mask: np.ndarray) -> List[Tuple[int, int]]:
    runs: List[Tuple[int, int]] = []
    index = 0
    while index < int(mask.size):
        if not bool(mask[index]):
            index += 1
            continue
        end = index + 1
        while end < int(mask.size) and bool(mask[end]):
            end += 1
        runs.append((index, end))
        index = end
    return runs


def _merge_close_ranges(ranges: List[Tuple[int, int]], max_gap: int) -> List[Tuple[int, int]]:
    if not ranges:
        return []
    merged = [sorted(ranges)[0]]
    for start, end in sorted(ranges)[1:]:
        previous_start, previous_end = merged[-1]
        if start - previous_end <= max_gap:
            merged[-1] = (previous_start, max(previous_end, end))
        else:
            merged.append((start, end))
    return merged


def _interaction_candidates(
    dataframe: pd.DataFrame,
    start_index: int,
    end_index: int,
    *,
    close_distance_m: float = 12.0,
    side_by_side_distance_m: float = 4.0,
    side_by_side_min_lateral_m: float = 1.25,
    side_by_side_longitudinal_window_m: float = 6.0,
    close_following_longitudinal_window_m: float = 30.0,
    close_following_lateral_m: float = 6.0,
    longitudinal_window_m: float = 18.0,
    pass_margin_m: float = 1.5,
    min_role_gain_m: float = 4.0,
    min_relative_velocity_mps: float = 0.5,
    min_threat_overlap_ilocs: int = 3,
    min_active_fraction: float = 0.3,
) -> List[Dict[str, Any]]:
    segment = dataframe.iloc[int(start_index):int(end_index)]
    if len(segment) < 2 or not {
        "Graphics_player_pos_x",
        "Graphics_player_pos_y",
    }.issubset(segment.columns):
        return []

    player_x = segment["Graphics_player_pos_x"].to_numpy(dtype=float)
    player_y = segment["Graphics_player_pos_y"].to_numpy(dtype=float)
    n_rows = len(segment)
    candidates: List[Dict[str, Any]] = []
    for slot in range(1, MAX_CARS + 1):
        x_column = f"Car_{slot}_pos_x"
        y_column = f"Car_{slot}_pos_y"
        if x_column not in segment.columns or y_column not in segment.columns:
            continue
        opponent_x = segment[x_column].to_numpy(dtype=float)
        opponent_y = segment[y_column].to_numpy(dtype=float)
        active = _active_opponent_mask(opponent_x, opponent_y, player_x, player_y)
        if not active.any() or float(active.mean()) < min_active_fraction:
            continue

        distance = np.sqrt((opponent_x - player_x) ** 2 + (opponent_y - player_y) ** 2)
        signed_long, lateral, player_s, player_d, frame = _relative_position_frame(
            player_x,
            player_y,
            opponent_x,
            opponent_y,
        )
        signed_long = np.where(active, signed_long, np.nan)
        lateral = np.where(active, lateral, np.nan)
        lateral_abs = np.abs(lateral)
        finite = active & np.isfinite(distance) & np.isfinite(signed_long) & np.isfinite(lateral_abs)
        if not finite.any():
            continue

        active_indices = np.where(finite)[0]
        entry_index = int(active_indices[0])
        exit_index = int(active_indices[-1])
        entry_long = float(signed_long[entry_index])
        exit_long = float(signed_long[exit_index])
        gap_delta = exit_long - entry_long
        min_distance_index = int(np.nanargmin(np.where(finite, distance, np.nan)))
        min_lateral_index = int(np.nanargmin(np.where(finite, lateral_abs, np.nan)))
        min_abs_long_index = int(np.nanargmin(np.where(finite, np.abs(signed_long), np.nan)))
        velocity, velocity_units = _relative_long_gap_velocity(segment, signed_long, finite)
        velocity_threshold = _relative_velocity_threshold(
            velocity_units,
            n_rows,
            min_relative_velocity_mps=min_relative_velocity_mps,
            min_role_gain_m=min_role_gain_m,
        )
        pressure_mask = finite & (
            (distance <= close_distance_m)
            | (np.abs(signed_long) <= longitudinal_window_m)
            | (lateral_abs <= side_by_side_distance_m)
        )
        attack_velocity = _nan_percentile(velocity[pressure_mask], 25)
        defense_velocity = _nan_percentile(velocity[pressure_mask], 75)
        attack_velocity_pressure = (
            attack_velocity is not None
            and attack_velocity <= -velocity_threshold
            and abs(float(signed_long[min_abs_long_index])) <= longitudinal_window_m
        )
        defense_velocity_pressure = (
            defense_velocity is not None
            and defense_velocity >= velocity_threshold
            and abs(float(signed_long[min_abs_long_index])) <= longitudinal_window_m
        )
        side_by_side = finite & (
            (np.abs(signed_long) <= side_by_side_longitudinal_window_m)
            & (lateral_abs >= side_by_side_min_lateral_m)
            & (lateral_abs <= side_by_side_distance_m)
        )
        side_count = int(side_by_side.sum())
        lateral_threat = finite & (
            (lateral_abs >= side_by_side_min_lateral_m)
            & (lateral_abs <= close_following_lateral_m)
            & (np.abs(signed_long) <= longitudinal_window_m)
        )
        threat_count = int(lateral_threat.sum())
        max_threat_long = (
            float(np.nanmax(signed_long[lateral_threat]))
            if threat_count else float(np.nanmax(signed_long[finite]))
        )
        threat_gain = max_threat_long - entry_long if threat_count else 0.0
        fallback_from_peak = max_threat_long - exit_long if threat_count else 0.0
        close_following = finite & (
            (lateral_abs <= close_following_lateral_m)
            & (np.abs(signed_long) <= close_following_longitudinal_window_m)
        )
        following_count = int(close_following.sum())
        trailing_count = int((close_following & (signed_long < -pass_margin_m)).sum())
        leading_count = int((close_following & (signed_long > pass_margin_m)).sum())
        close_enough = bool(
            float(distance[min_distance_index]) <= close_distance_m
            or side_count
            or following_count
        )
        completion_margin = max(pass_margin_m, min_role_gain_m)
        pass_crossing_close = bool(
            float(distance[min_distance_index]) <= close_distance_m
            and abs(float(signed_long[min_abs_long_index])) <= longitudinal_window_m
        )
        min_long = float(np.nanmin(signed_long[finite]))
        max_long = float(np.nanmax(signed_long[finite]))
        passed = bool(
            pass_crossing_close
            and entry_long >= -pass_margin_m
            and exit_long < -completion_margin
            and max_long > pass_margin_m
        )
        passed_by = bool(
            pass_crossing_close
            and entry_long <= pass_margin_m
            and exit_long > completion_margin
            and min_long <= pass_margin_m
        )
        attack_pressure = bool(
            entry_long > pass_margin_m
            and close_enough
            and not passed
            and not passed_by
            and (
                gap_delta <= -min_role_gain_m
                or attack_velocity_pressure
                or (side_count and abs(float(signed_long[min_abs_long_index])) <= pass_margin_m)
            )
        )
        sustained_overlap = side_count >= min_threat_overlap_ilocs
        defense_pressure = bool(
            close_enough
            and not passed_by
            and exit_long <= pass_margin_m
            and (
                sustained_overlap
                or (
                    threat_count
                    and threat_gain >= min_role_gain_m
                    and max_threat_long >= -pass_margin_m
                )
                or (
                    threat_count
                    and defense_velocity_pressure
                    and max_threat_long >= -pass_margin_m
                )
            )
            and (
                fallback_from_peak >= min_role_gain_m
                or (sustained_overlap and exit_long <= -pass_margin_m)
            )
        )

        if passed:
            role, outcome = "attack", "pass_completed"
        elif passed_by:
            role, outcome = "defense", "broken_defense"
        elif attack_pressure:
            role, outcome = "attack", "failed_attack"
        elif defense_pressure:
            role, outcome = "defense", "held_defense"
        elif close_enough and sustained_overlap:
            role, outcome = "side_by_side", "side_by_side"
        elif following_count:
            role, outcome = "following", "close_following"
        elif close_enough:
            role, outcome = "incidental", "incidental"
        else:
            role, outcome = "none", "no_close_interaction"

        candidates.append({
            "slot": slot,
            "role": role,
            "outcome": outcome,
            "min_distance_m": float(distance[min_distance_index]),
            "min_distance_iloc": int(start_index + min_distance_index),
            "entry_signed_long_gap_m": entry_long,
            "exit_signed_long_gap_m": exit_long,
            "min_lateral_offset_m": float(lateral_abs[min_lateral_index]),
            "side_by_side_iloc_count": side_count,
            "lateral_threat_iloc_count": threat_count,
            "close_following_iloc_count": following_count,
            "trailing_pressure_iloc_count": trailing_count,
            "leading_draft_iloc_count": leading_count,
            "relative_velocity_units": velocity_units,
            "coordinate_frame": frame,
            "player_progress_m_at_entry": float(player_s[entry_index]),
            "player_progress_m_at_exit": float(player_s[exit_index]),
            "player_lateral_offset_m_at_entry": float(player_d[entry_index]),
            "player_lateral_offset_m_at_exit": float(player_d[exit_index]),
            "passed_by_player": passed,
            "got_passed_by_opponent": passed_by,
        })

    priority = {
        "pass_completed": 6,
        "broken_defense": 6,
        "failed_attack": 5,
        "held_defense": 5,
        "side_by_side": 3,
        "close_following": 2,
        "incidental": 2,
        "no_close_interaction": 1,
    }
    candidates.sort(
        key=lambda candidate: (
            priority.get(str(candidate["outcome"]), 0),
            -float(candidate["min_distance_m"]),
        ),
        reverse=True,
    )
    return candidates


def _detect_opponent_interaction_windows(
    dataframe: pd.DataFrame,
    start_index: int,
    end_index: int,
    *,
    close_distance_m: float = 12.0,
    side_by_side_distance_m: float = 4.0,
    side_by_side_min_lateral_m: float = 1.25,
    side_by_side_longitudinal_window_m: float = 6.0,
    close_following_longitudinal_window_m: float = 30.0,
    close_following_lateral_m: float = 6.0,
    longitudinal_window_m: float = 18.0,
    pass_margin_m: float = 1.5,
    min_role_gain_m: float = 4.0,
    min_relative_velocity_mps: float = 0.5,
    min_threat_overlap_ilocs: int = 3,
    min_window_ilocs: int = 3,
    context_padding_ilocs: int = 8,
    event_padding_ilocs: int = 16,
    max_event_window_ilocs: int = 80,
    merge_gap_ilocs: int = 10,
    min_active_fraction: float = 0.3,
) -> List[Dict[str, Any]]:
    start = int(start_index)
    segment = dataframe.iloc[start:int(end_index)]
    n_rows = len(segment)
    if n_rows < 2 or not {
        "Graphics_player_pos_x",
        "Graphics_player_pos_y",
    }.issubset(dataframe.columns):
        return []

    player_x = segment["Graphics_player_pos_x"].to_numpy(dtype=float)
    player_y = segment["Graphics_player_pos_y"].to_numpy(dtype=float)
    if not (np.isfinite(player_x).any() and np.isfinite(player_y).any()):
        return []

    windows: List[Dict[str, Any]] = []

    def event_window(run_start: int, run_end: int, focal: int) -> Tuple[int, int]:
        event_start = max(run_start, focal - int(event_padding_ilocs))
        event_end = min(run_end, focal + int(event_padding_ilocs) + 1)
        max_length = max(int(min_window_ilocs), int(max_event_window_ilocs))
        if event_end - event_start > max_length:
            half = max_length // 2
            event_start = max(run_start, focal - half)
            event_end = min(run_end, event_start + max_length)
            event_start = max(run_start, event_end - max_length)
        return event_start, event_end

    def event_ranges_for_run(
        run_start: int,
        run_end: int,
        signed_long: np.ndarray,
        lateral: np.ndarray,
        distance: np.ndarray,
        relative_velocity: np.ndarray,
        velocity_threshold: float,
        finite: np.ndarray,
    ) -> List[Tuple[int, int, str, str]]:
        run_slice = slice(run_start, run_end)
        finite_run = finite[run_slice]
        if not finite_run.any():
            return []
        local_indices = np.where(finite_run)[0] + run_start
        gaps = signed_long[local_indices]
        if not np.isfinite(gaps).any():
            return []

        entry_long = float(gaps[0])
        exit_long = float(gaps[-1])
        gap_delta = exit_long - entry_long
        min_long = float(np.nanmin(gaps))
        max_long = float(np.nanmax(gaps))
        min_run_distance = float(np.nanmin(distance[local_indices]))
        min_abs_long = float(np.nanmin(np.abs(gaps)))
        completion_margin = max(float(pass_margin_m), float(min_role_gain_m))
        pass_crossing_close = bool(
            min_run_distance <= close_distance_m
            and min_abs_long <= longitudinal_window_m
        )
        passed = bool(
            pass_crossing_close
            and entry_long >= -pass_margin_m
            and exit_long < -completion_margin
            and max_long > pass_margin_m
        )
        passed_by = bool(
            pass_crossing_close
            and entry_long <= pass_margin_m
            and exit_long > completion_margin
            and min_long <= pass_margin_m
        )
        attack_velocity = _nan_percentile(relative_velocity[local_indices], 25)
        defense_velocity = _nan_percentile(relative_velocity[local_indices], 75)
        attack_velocity_pressure = (
            attack_velocity is not None
            and attack_velocity <= -velocity_threshold
            and min_abs_long <= longitudinal_window_m
        )
        defense_velocity_pressure = (
            defense_velocity is not None
            and defense_velocity >= velocity_threshold
            and min_abs_long <= longitudinal_window_m
        )
        lateral_abs = np.abs(lateral[local_indices])
        side_by_side = (
            (np.abs(gaps) <= side_by_side_longitudinal_window_m)
            & (lateral_abs >= side_by_side_min_lateral_m)
            & (lateral_abs <= side_by_side_distance_m)
        )
        lateral_threat = (
            (lateral_abs >= side_by_side_min_lateral_m)
            & (lateral_abs <= close_following_lateral_m)
            & (np.abs(gaps) <= longitudinal_window_m)
        )
        threat_count = int(lateral_threat.sum())
        if threat_count:
            max_threat_long = float(np.nanmax(gaps[lateral_threat]))
            defense_threat_gain = max_threat_long - entry_long
            defense_fallback = max_threat_long - exit_long
        else:
            max_threat_long = float(np.nanmax(gaps))
            defense_threat_gain = 0.0
            defense_fallback = 0.0
        sustained_overlap = int(side_by_side.sum()) >= int(min_threat_overlap_ilocs)

        if passed or passed_by:
            crossing_level = -pass_margin_m if passed else pass_margin_m
            crossings = local_indices[:-1][
                (gaps[:-1] - crossing_level) * (gaps[1:] - crossing_level) <= 0
            ]
            focal = (
                int(crossings[0])
                if crossings.size
                else int(local_indices[np.nanargmin(np.abs(gaps))])
            )
            event_start, event_end = event_window(run_start, run_end, focal)
            return [(
                event_start,
                event_end,
                "attack" if passed else "defense",
                "pass_completed" if passed else "broken_defense",
            )]

        attack_threat = sustained_overlap or threat_count >= int(min_threat_overlap_ilocs)
        if (
            entry_long > pass_margin_m
            and attack_threat
            and (gap_delta <= -min_role_gain_m or attack_velocity_pressure)
        ):
            focal = int(local_indices[np.nanargmin(gaps)])
            event_start, event_end = event_window(run_start, run_end, focal)
            return [(event_start, event_end, "attack", "failed_attack")]

        defense_threat = bool(
            sustained_overlap
            or (
                threat_count > 0
                and defense_threat_gain >= min_role_gain_m
                and max_threat_long >= -pass_margin_m
            )
            or (
                threat_count > 0
                and defense_velocity_pressure
                and max_threat_long >= -pass_margin_m
            )
        )
        defense_repelled = bool(
            defense_fallback >= min_role_gain_m
            or (sustained_overlap and exit_long <= -pass_margin_m)
        )
        if defense_threat and defense_repelled and exit_long <= pass_margin_m:
            focal = int(local_indices[np.nanargmax(gaps)])
            event_start, event_end = event_window(run_start, run_end, focal)
            return [(event_start, event_end, "defense", "held_defense")]

        focal = int(run_start + np.nanargmin(distance[run_slice]))
        event_start, event_end = event_window(run_start, run_end, focal)
        side_by_side_run = (
            (np.abs(signed_long[run_slice]) <= side_by_side_longitudinal_window_m)
            & (np.abs(lateral[run_slice]) >= side_by_side_min_lateral_m)
            & (np.abs(lateral[run_slice]) <= side_by_side_distance_m)
        ) & finite_run
        if int(side_by_side_run.sum()) >= int(min_threat_overlap_ilocs):
            return [(event_start, event_end, "side_by_side", "side_by_side")]
        return [(event_start, event_end, "following", "close_following")]

    for slot in range(1, MAX_CARS + 1):
        x_column = f"Car_{slot}_pos_x"
        y_column = f"Car_{slot}_pos_y"
        if x_column not in dataframe.columns or y_column not in dataframe.columns:
            continue
        opponent_x = segment[x_column].to_numpy(dtype=float)
        opponent_y = segment[y_column].to_numpy(dtype=float)
        active = _active_opponent_mask(opponent_x, opponent_y, player_x, player_y)
        if not active.any() or float(active.mean()) < min_active_fraction:
            continue

        distance = np.sqrt((opponent_x - player_x) ** 2 + (opponent_y - player_y) ** 2)
        signed_long, lateral, _, _, coordinate_frame = _relative_position_frame(
            player_x,
            player_y,
            opponent_x,
            opponent_y,
        )
        signed_long = np.where(active, signed_long, np.nan)
        lateral = np.where(active, lateral, np.nan)
        lateral_abs = np.abs(lateral)
        finite = (
            active
            & np.isfinite(distance)
            & np.isfinite(signed_long)
            & np.isfinite(lateral_abs)
        )
        if not finite.any():
            continue
        relative_velocity, velocity_units = _relative_long_gap_velocity(
            segment,
            signed_long,
            finite,
        )
        velocity_threshold = _relative_velocity_threshold(
            velocity_units,
            n_rows,
            min_relative_velocity_mps=min_relative_velocity_mps,
            min_role_gain_m=min_role_gain_m,
        )
        side_by_side = (
            (np.abs(signed_long) <= side_by_side_longitudinal_window_m)
            & (lateral_abs >= side_by_side_min_lateral_m)
            & (lateral_abs <= side_by_side_distance_m)
        )
        close_following = (
            (lateral_abs <= close_following_lateral_m)
            & (np.abs(signed_long) <= close_following_longitudinal_window_m)
        )
        close = finite & ((distance <= close_distance_m) | side_by_side | close_following)
        runs = _merge_close_ranges(_contiguous_true_runs(close), merge_gap_ilocs)
        for run_start, run_end in runs:
            event_ranges = event_ranges_for_run(
                run_start,
                run_end,
                signed_long,
                lateral,
                distance,
                relative_velocity,
                velocity_threshold,
                finite,
            )
            for event_start, event_end, role, outcome in event_ranges:
                padded_start = max(0, event_start - int(context_padding_ilocs))
                padded_end = min(n_rows, event_end + int(context_padding_ilocs))
                if padded_end - padded_start > int(max_event_window_ilocs):
                    focal = (event_start + event_end) // 2
                    half = int(max_event_window_ilocs) // 2
                    padded_start = max(0, focal - half)
                    padded_end = min(n_rows, padded_start + int(max_event_window_ilocs))
                    padded_start = max(0, padded_end - int(max_event_window_ilocs))
                if padded_end - padded_start < int(min_window_ilocs):
                    continue

                window_slice = slice(padded_start, padded_end)
                finite_window = finite[window_slice]
                if not finite_window.any():
                    continue
                local_indices = np.where(finite_window)[0] + padded_start
                entry_index = int(local_indices[0])
                exit_index = int(local_indices[-1])
                min_distance_index = padded_start + int(np.nanargmin(distance[window_slice]))
                min_lateral_index = padded_start + int(np.nanargmin(lateral_abs[window_slice]))
                min_abs_long_index = padded_start + int(
                    np.nanargmin(np.abs(signed_long[window_slice]))
                )
                side_count = int((side_by_side[window_slice] & finite_window).sum())
                following_count = int((close_following[window_slice] & finite_window).sum())
                trailing_count = int((
                    close_following[window_slice]
                    & finite_window
                    & (signed_long[window_slice] < -pass_margin_m)
                ).sum())
                leading_count = int((
                    close_following[window_slice]
                    & finite_window
                    & (signed_long[window_slice] > pass_margin_m)
                ).sum())
                entry_long = float(signed_long[entry_index])
                exit_long = float(signed_long[exit_index])
                pass_crossing_close = bool(
                    float(distance[min_distance_index]) <= close_distance_m
                    and abs(float(signed_long[min_abs_long_index])) <= longitudinal_window_m
                )
                completion_margin = max(float(pass_margin_m), float(min_role_gain_m))
                windows.append({
                    "start_index": int(start + padded_start),
                    "end_index": int(start + padded_end),
                    "slot": int(slot),
                    "event_role": role,
                    "event_outcome": outcome,
                    "min_distance_m": float(distance[min_distance_index]),
                    "min_distance_iloc": int(start + min_distance_index),
                    "entry_signed_long_gap_m": entry_long,
                    "exit_signed_long_gap_m": exit_long,
                    "min_lateral_offset_m": float(lateral_abs[min_lateral_index]),
                    "side_by_side_iloc_count": side_count,
                    "close_following_iloc_count": following_count,
                    "trailing_pressure_iloc_count": trailing_count,
                    "leading_draft_iloc_count": leading_count,
                    "min_relative_long_gap_velocity": _nan_percentile(
                        relative_velocity[window_slice],
                        10,
                    ),
                    "max_relative_long_gap_velocity": _nan_percentile(
                        relative_velocity[window_slice],
                        90,
                    ),
                    "relative_velocity_threshold": float(velocity_threshold),
                    "relative_velocity_units": velocity_units,
                    "coordinate_frame": coordinate_frame,
                    "pass_crossing_close": pass_crossing_close,
                    "passed_by_player": bool(
                        pass_crossing_close
                        and entry_long > 0
                        and exit_long < -completion_margin
                    ),
                    "got_passed_by_opponent": bool(
                        pass_crossing_close
                        and entry_long < 0
                        and exit_long > completion_margin
                    ),
                })

    windows.sort(key=lambda window: (window["start_index"], window["end_index"]))
    merged: List[Dict[str, Any]] = []
    event_priority = {
        "pass_completed": 5,
        "broken_defense": 5,
        "failed_attack": 4,
        "held_defense": 4,
        "side_by_side": 2,
        "close_following": 1,
    }
    for window in windows:
        if not merged or window["start_index"] > merged[-1]["end_index"] + merge_gap_ilocs:
            merged.append(dict(window))
            continue
        current = merged[-1]
        current["start_index"] = min(current["start_index"], window["start_index"])
        current["end_index"] = max(current["end_index"], window["end_index"])
        current["slots"] = sorted(set(current.get("slots", [current["slot"]]) + [window["slot"]]))
        window_rank = event_priority.get(str(window.get("event_outcome")), 0)
        current_rank = event_priority.get(str(current.get("event_outcome")), 0)
        if (
            window_rank > current_rank
            or (
                window_rank == current_rank
                and float(window["min_distance_m"]) < float(current["min_distance_m"])
            )
        ):
            for key in (
                "slot",
                "event_role",
                "event_outcome",
                "min_distance_m",
                "min_distance_iloc",
                "entry_signed_long_gap_m",
                "exit_signed_long_gap_m",
                "min_lateral_offset_m",
                "side_by_side_iloc_count",
                "close_following_iloc_count",
                "trailing_pressure_iloc_count",
                "leading_draft_iloc_count",
                "min_relative_long_gap_velocity",
                "max_relative_long_gap_velocity",
                "relative_velocity_threshold",
                "relative_velocity_units",
                "coordinate_frame",
                "passed_by_player",
                "got_passed_by_opponent",
            ):
                current[key] = window[key]
        else:
            current["passed_by_player"] = bool(current.get("passed_by_player")) or bool(
                window.get("passed_by_player")
            )
            current["got_passed_by_opponent"] = bool(
                current.get("got_passed_by_opponent")
            ) or bool(window.get("got_passed_by_opponent"))
    return merged


def _align_interaction_windows_with_classifier(
    dataframe: pd.DataFrame,
    windows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    aligned: List[Dict[str, Any]] = []
    for window in windows:
        output = dict(window)
        candidates = _interaction_candidates(
            dataframe,
            int(output["start_index"]),
            int(output["end_index"]),
        )
        selected = next(
            (candidate for candidate in candidates if candidate["slot"] == output.get("slot")),
            candidates[0] if candidates else None,
        )
        if selected is not None:
            selected_outcome = str(selected.get("outcome", output.get("event_outcome")))
            preserve_following = (
                output.get("event_outcome") == "close_following"
                and selected_outcome == "failed_attack"
                and not selected.get("passed_by_player")
                and not selected.get("got_passed_by_opponent")
                and int(selected.get("side_by_side_iloc_count") or 0) < 3
                and int(selected.get("lateral_threat_iloc_count") or 0) < 3
            )
            output.update(selected)
            if preserve_following:
                output["event_role"] = "following"
                output["event_outcome"] = "close_following"
            else:
                output["event_role"] = selected["role"]
                output["event_outcome"] = selected["outcome"]
        aligned.append(output)
    return aligned


def _align_interaction_segments_with_classifier(
    dataframe: pd.DataFrame,
    segments: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    for segment in segments:
        interaction = segment.get("opponent_interaction")
        if not isinstance(interaction, dict):
            continue
        windows = interaction.get("windows")
        if not isinstance(windows, list) or not windows:
            continue
        aligned_windows = _align_interaction_windows_with_classifier(dataframe, windows)
        interaction["windows"] = aligned_windows
        first = next(
            (window for window in aligned_windows if isinstance(window, dict)),
            None,
        )
        if first is not None:
            interaction["targeted_car_slot"] = first.get("slot")
            interaction["targeted_car_label"] = (
                f"Car {first['slot']}" if first.get("slot") is not None else None
            )
            interaction["role"] = first.get("event_role")
            interaction["outcome"] = first.get("event_outcome")
    return segments


def _lap_boundary_offsets(
    positions: np.ndarray,
    completed_laps: Optional[np.ndarray] = None,
) -> List[int]:
    boundaries: List[int] = []
    for offset in range(1, int(positions.size)):
        if completed_laps is not None:
            previous_lap = completed_laps[offset - 1]
            current_lap = completed_laps[offset]
            if (
                np.isfinite(previous_lap)
                and np.isfinite(current_lap)
                and current_lap != previous_lap
            ):
                boundaries.append(offset)
                continue
        previous_position = positions[offset - 1]
        current_position = positions[offset]
        if (
            np.isfinite(previous_position)
            and np.isfinite(current_position)
            and previous_position - current_position > 0.5
        ):
            boundaries.append(offset)
    return boundaries


def _split_windows_at_lap_boundaries(
    windows: List[Dict[str, Any]],
    boundaries: List[int],
    end_index: int,
) -> List[Dict[str, Any]]:
    if not windows or not boundaries:
        return windows
    cuts = [0, *sorted(boundaries), int(end_index)]
    split: List[Dict[str, Any]] = []
    for window in windows:
        source_start = int(window["start_index"])
        source_end = int(window["end_index"])
        for cut_start, cut_end in zip(cuts, cuts[1:]):
            start = max(source_start, cut_start)
            end = min(source_end, cut_end)
            if end <= start:
                continue
            clipped = dict(window)
            clipped["start_index"] = start
            clipped["end_index"] = end
            clipped["source_window_range"] = [source_start, source_end]
            split.append(clipped)
    return split


def _has_active_opponent_data(dataframe: pd.DataFrame) -> bool:
    active_slots = 0
    for slot in range(1, MAX_CARS + 1):
        x_column = f"Car_{slot}_pos_x"
        y_column = f"Car_{slot}_pos_y"
        if x_column not in dataframe.columns or y_column not in dataframe.columns:
            continue
        opponent_x = dataframe[x_column].to_numpy(dtype=float)
        opponent_y = dataframe[y_column].to_numpy(dtype=float)
        active = (
            ((opponent_x != 0.0) | (opponent_y != 0.0))
            & np.isfinite(opponent_x)
            & np.isfinite(opponent_y)
        )
        if active.any():
            active_slots += 1
            if active_slots >= 2:
                return True
    return False


def _is_following_only(window: Dict[str, Any]) -> bool:
    role = str(window.get("event_role") or window.get("role") or "")
    outcome = str(window.get("event_outcome") or window.get("outcome") or "")
    return role == "following" or outcome == "close_following"


def _section_segments(
    positions: np.ndarray,
    circuit_id: str,
    lap_boundaries: List[int],
) -> Tuple[List[Dict[str, Any]], int]:
    candidates = [
        {
            "id": section_id,
            "name": LABEL_MAPPING.get(section_id, section_id),
            "lo": float(section_range[0]),
            "hi": float(section_range[1]),
        }
        for section_id, section_range in CIRCUIT_SECTION_RANGES.items()
        if section_id.rstrip("0123456789") == circuit_id
    ]
    boundary_set = set(lap_boundaries)

    def section_for(position: float) -> Optional[Dict[str, Any]]:
        if not np.isfinite(position):
            return None
        position = position - np.floor(position)
        for candidate in candidates:
            lo, hi = candidate["lo"], candidate["hi"]
            if (hi >= lo and lo <= position <= hi) or (
                hi < lo and (position >= lo or position <= hi)
            ):
                return candidate
        return None

    segments: List[Dict[str, Any]] = []
    current: Optional[Dict[str, Any]] = None
    current_start = 0
    matched = 0
    unmatched = 0

    def close_run(end: int) -> None:
        nonlocal current, current_start, matched
        if current is not None and end > current_start:
            length = end - current_start
            segments.append({
                "start_index": current_start,
                "end_index": end,
                "circuit_section_id": current["id"],
                "circuit_section_name": current["name"],
                "normalized_position_range": [current["lo"], current["hi"]],
                "coverage_fraction": matched / length,
                "split_basis": "circuit_section",
                "opponent_interaction": None,
            })
        current = None
        current_start = end
        matched = 0

    for index, position in enumerate(positions):
        section = section_for(position)
        if section is None:
            unmatched += 1
            if current is None:
                current_start = index + 1
            continue
        if index in boundary_set and current is not None:
            close_run(index)
        if current is None:
            current = section
            current_start = index
            matched = 1
        elif section["id"] == current["id"]:
            matched += 1
        else:
            close_run(index)
            current = section
            current_start = index
            matched = 1
    close_run(len(positions))
    return segments, unmatched


def _interaction_segments(
    windows: List[Dict[str, Any]],
    section_segments: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    segments: List[Dict[str, Any]] = []
    for window in windows:
        window_start = int(window["start_index"])
        window_end = int(window["end_index"])
        contexts: List[Dict[str, Any]] = []
        for section in section_segments:
            start = max(window_start, int(section["start_index"]))
            end = min(window_end, int(section["end_index"]))
            if end <= start:
                continue
            contexts.append({
                "circuit_section_id": section["circuit_section_id"],
                "circuit_section_name": section["circuit_section_name"],
                "range": [start, end],
                "normalized_position_range": section["normalized_position_range"],
            })
        ranges = contexts or [{"range": [window_start, window_end]}]
        for context in ranges:
            start, end = context["range"]
            clipped = dict(window)
            if start != window_start or end != window_end:
                clipped["start_index"] = start
                clipped["end_index"] = end
                clipped["source_window_range"] = [window_start, window_end]
            section_context = [context] if contexts else []
            segments.append({
                "start_index": start,
                "end_index": end,
                "circuit_section_id": "interaction_window",
                "circuit_section_name": "Racing interaction",
                "normalized_position_range": None,
                "coverage_fraction": 1.0,
                "split_basis": "opponent_interaction",
                "opponent_interaction": {
                    "targeted_car_slot": clipped.get("slot"),
                    "targeted_car_label": (
                        f"Car {clipped['slot']}" if clipped.get("slot") is not None else None
                    ),
                    "role": clipped.get("event_role"),
                    "outcome": clipped.get("event_outcome"),
                    "windows": [clipped],
                    "section_context": section_context,
                },
            })
    return segments


def split_runtime_segments(dataframe: pd.DataFrame, circuit_id: Optional[str]) -> Dict[str, Any]:
    """Return annotation-style deterministic session ranges for inference."""
    source = dataframe.reset_index(drop=True)
    if NORMALIZED_POSITION_COLUMN not in source.columns:
        raise RuntimeSegmentSplitError(
            f"column '{NORMALIZED_POSITION_COLUMN}' missing from telemetry"
        )
    resolved_circuit = resolve_runtime_circuit(source, circuit_id)
    if source.empty:
        return {
            "circuit_id": resolved_circuit,
            "range": [0, 0],
            "opponent_session": False,
            "split_mode": "circuit_sections",
            "segments": [],
            "interaction_windows": [],
            "following_windows_filtered": 0,
            "unmatched_ilocs": 0,
        }

    positions = pd.to_numeric(
        source[NORMALIZED_POSITION_COLUMN],
        errors="coerce",
    ).to_numpy(dtype=float)
    completed_laps = None
    if "Graphics_completed_lap" in source.columns:
        completed_laps = pd.to_numeric(
            source["Graphics_completed_lap"],
            errors="coerce",
        ).to_numpy(dtype=float)
    lap_boundaries = _lap_boundary_offsets(positions, completed_laps)
    sections, unmatched = _section_segments(positions, resolved_circuit, lap_boundaries)

    opponent_session = _has_active_opponent_data(source)
    interaction_windows = _detect_opponent_interaction_windows(source, 0, len(source))
    interaction_windows = _split_windows_at_lap_boundaries(
        interaction_windows,
        lap_boundaries,
        len(source),
    )
    interaction_windows = _align_interaction_windows_with_classifier(source, interaction_windows)
    following_windows_filtered = sum(_is_following_only(window) for window in interaction_windows)
    interaction_windows = [
        window for window in interaction_windows if not _is_following_only(window)
    ]
    segments = (
        _interaction_segments(interaction_windows, sections)
        if opponent_session else sections
    )
    segments = _align_interaction_segments_with_classifier(source, segments)
    return {
        "circuit_id": resolved_circuit,
        "range": [0, len(source)],
        "opponent_session": opponent_session,
        "split_mode": (
            "opponent_interactions_only" if opponent_session else "circuit_sections"
        ),
        "segments": segments,
        "interaction_windows": interaction_windows,
        "following_windows_filtered": int(following_windows_filtered),
        "unmatched_ilocs": int(unmatched),
    }


__all__ = [
    "RuntimeSegmentSplitError",
    "resolve_runtime_circuit",
    "split_runtime_segments",
]
