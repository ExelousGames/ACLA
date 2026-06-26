"""Detailed-flow statistical preflight events."""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from app.local_annotation_agent.workflow.preflight import (
    SHARED_PREFLIGHT_QUERY_SPECS,
    SHARED_PREFLIGHT_TOOL_IDS,
    SPEED_INVESTIGATION_QUERY_SPECS,
    PreflightContext,
    _preflight_analysis_ids,
    _preflight_query_table,
    _run_queries,
    _run_tools,
    _semantic_tags,
    _semantic_tool_output,
)
from app.shared.contracts import Attachment


TRAJECTORY_ALIGNED_SIMILARITY_THRESHOLD = 0.8
RACING_PARENT_LABELS = {"O", "OD", "MSR"}


DETAILED_PREFLIGHT_TOOL_IDS = (
    *SHARED_PREFLIGHT_TOOL_IDS,
    "classify_opponent_interaction",
    "find_nearest_opponent",
)
_DETAILED_SHARED_PREFLIGHT_QUERY_SPECS = tuple(
    spec
    for spec in SHARED_PREFLIGHT_QUERY_SPECS
    if (spec.get("params") or {}).get("column") != "expert_time_difference"
)
DETAILED_PREFLIGHT_QUERY_SPECS = (
    *_DETAILED_SHARED_PREFLIGHT_QUERY_SPECS,
    *SPEED_INVESTIGATION_QUERY_SPECS,
    {
        "tool_id": "query_telemetry.compute_slope.trajectory_offset",
        "graph_id": "trajectory_offset",
        "query_id": "compute_slope",
        "params": {"column": "trajectory_offset"},
        "tags": [
            "trajectory recovery",
            "wider than expert",
            "tighter than expert",
        ],
    },
    {
        "tool_id": "query_telemetry.find_trend_runs.trajectory_offset",
        "graph_id": "trajectory_offset",
        "query_id": "find_trend_runs",
        "params": {
            "column": "trajectory_offset",
            "smoothing_window": 3,
        },
        "tags": [
            "trajectory offset trend",
            "moving toward positive",
            "moving toward negative",
        ],
    },
    {
        "tool_id": "query_telemetry.measure_trajectory_similarity.driver_expert_path",
        "graph_id": "trajectory_detailed",
        "query_id": "measure_trajectory_similarity",
        "params": {
            "smoothing_window": 3,
        },
        "tags": [
            "trajectory similarity",
            "driver expert path comparison",
            "line separation",
        ],
    },
    {
        "tool_id": "query_telemetry.find_extremum.brake.player.max",
        "graph_id": "brake",
        "query_id": "find_extremum",
        "params": {"column": "Physics_brake", "kind": "max"},
        "tags": ["player peak brake pressure"],
    },
    {
        "tool_id": "query_telemetry.find_extremum.brake.expert.max",
        "graph_id": "brake",
        "query_id": "find_extremum",
        "params": {"column": "expert_optimal_brake", "kind": "max"},
        "tags": ["expert peak brake pressure"],
    },
    {
        "tool_id": "query_telemetry.find_extremum.throttle.player.min",
        "graph_id": "throttle",
        "query_id": "find_extremum",
        "params": {"column": "Physics_gas", "kind": "min"},
        "tags": ["player lowest throttle pressure"],
    },
    {
        "tool_id": "query_telemetry.find_extremum.throttle.expert.min",
        "graph_id": "throttle",
        "query_id": "find_extremum",
        "params": {"column": "expert_optimal_throttle", "kind": "min"},
        "tags": ["expert lowest throttle pressure"],
    },
    {
        "tool_id": "query_telemetry.find_dips_on_main_slope.throttle",
        "graph_id": "throttle",
        "query_id": "find_dips_on_main_slope",
        "params": {
            "column": "Physics_gas",
            "smoothing_window": 3,
            "min_dip_depth": 0.08,
        },
        "tags": ["throttle modulation dip", "release throttle"],
    },
    {
        "tool_id": "query_telemetry.find_extremum.speed_difference.max",
        "graph_id": "speed_delta",
        "query_id": "find_extremum",
        "params": {"column": "speed_difference", "kind": "max"},
    },
    {
        "tool_id": "query_telemetry.find_extremum.speed_difference.min",
        "graph_id": "speed_delta",
        "query_id": "find_extremum",
        "params": {"column": "speed_difference", "kind": "min"},
    },
    {
        "tool_id": "query_telemetry.find_extremum.push_limit.max",
        "graph_id": "push_limit",
        "query_id": "find_extremum",
        "params": {"column": "driver_push_to_limit", "kind": "max"},
        "tags": [
            "over-limit spike",
            "sustained over-limit",
            "grip utilisation",
        ],
    },
)

_BASE_SEGMENT_SHAPE_WORDS = {
    "in_corner": (
        "In the corner",
        "Full segment covers entire corner; driver turning throughout; "
        "single curve hairpin continuous arc",
    ),
    "straight": (
        "On the straight",
        "Full segment on a straight section; minimal steering; full throttle; "
        "minimal curvature",
    ),
    "approach_to_corner": (
        "Approach to corner",
        "Braking zone before a corner; starts before detected corner arc and "
        "ends before apex",
    ),
    "exit_corner_to_straight": (
        "Exit corner leading to straight",
        "Corner exit section leading onto a straight; steering unwinding and "
        "throttle increasing",
    ),
    "between_consecutive_corners": (
        "Between consecutive corners",
        "Short transition between two corners; brief connection not a full straight",
    ),
    "consecutive_corners_no_straight": (
        "Consecutive corners with no straight in between",
        "Multiple corners with no intervening straight; S-shape chicane esses",
    ),
}

_CORNER_REFINEMENT_WORDS = {
    "constant_radius": (
        "Constant-radius corner",
        "Corner keeps a consistent radius from entry through apex to exit; "
        "smooth steady curvature",
    ),
    "increasing_radius": (
        "Increasing-radius corner",
        "Corner opens up after entry or apex; curvature decreases and radius "
        "increases toward exit",
    ),
    "decreasing_radius": (
        "Decreasing-radius corner",
        "Corner tightens from entry toward apex or exit; curvature increases "
        "and radius decreases",
    ),
    "hairpin": (
        "Hairpin corner",
        "Near-180 U-turn identified by detected turn angle",
    ),
    "chicane_or_esses": (
        "Chicane or esses",
        "Linked left-right or right-left direction changes; chicanes esses S-bends",
    ),
}

_ALTITUDE_WORDS = {
    ("entry", "uphill"): (
        "entry altitude uphill; corner entry has uphill slope angle across the entry phase"
    ),
    ("entry", "level"): (
        "entry altitude level; corner entry has near-level slope angle across the entry phase"
    ),
    ("entry", "downhill"): (
        "entry altitude downhill; corner entry has downhill slope angle across the entry phase"
    ),
    ("apex", "uphill"): "apex altitude uphill; uphill slope angle through the apex window",
    ("apex", "level"): (
        "apex altitude level; near-level slope angle through the apex window"
    ),
    ("apex", "downhill"): (
        "apex altitude downhill; downhill slope angle through the apex window"
    ),
    ("exit", "uphill"): (
        "exit altitude uphill; corner exit has uphill slope angle through the exit phase"
    ),
    ("exit", "level"): (
        "exit altitude level; corner exit has near-level slope angle through the exit phase"
    ),
    ("exit", "downhill"): (
        "exit altitude downhill; corner exit has downhill slope angle through the exit phase"
    ),
}

def build_preflight_context(
    *,
    df,
    start: int,
    end: int,
    parent_main_labels: Sequence[str],
    extra_query_terms: Sequence[str],
) -> PreflightContext:
    s, e = int(start), int(end)
    if e <= s:
        raise RuntimeError(f"detailed preflight: invalid range [{s}, {e}]")

    tool_outputs = [
        *_run_tools(df, s, e, DETAILED_PREFLIGHT_TOOL_IDS),
        *_run_queries(df, s, e, DETAILED_PREFLIGHT_QUERY_SPECS),
    ]
    events = _build_detailed_events(
        df,
        s,
        e,
        tool_outputs,
        parent_main_labels=parent_main_labels,
    )
    event_text = _event_text(events, parent_main_labels, extra_query_terms)
    semantic_search_text = _semantic_search_text(
        events,
        parent_main_labels,
        extra_query_terms,
    )
    source_tool_ids = _dedupe(
        source
        for event in events
        for source in event.get("sources", [])
    )
    tags = _dedupe(
        tag
        for tool_id, content in tool_outputs
        for tag in [f"tool:{tool_id}", *_semantic_tags(tool_id, content)]
    )[:160]

    attachments = [
        Attachment(
            name=f"init.preflight_tool.{tool_id}",
            kind="structured",
            label=f"Preflight Tool: {tool_id}",
            content={
                "tool_id": tool_id,
                "range": [s, e],
                "tags": _semantic_tags(tool_id, content),
                "result": _semantic_tool_output(tool_id, content),
            },
            content_schema="annotation_preflight_tool",
        )
        for tool_id, content in tool_outputs
    ]
    attachments.extend([
        Attachment(
            name="init.detailed_preflight_events",
            kind="structured",
            label="Detailed Preflight Statistical Events",
            content={
                "range": [s, e],
                "events": events,
                "event_text": event_text,
                "semantic_search_text": semantic_search_text,
                "source_tool_ids": source_tool_ids,
            },
            content_schema="detailed_preflight_events",
        ),
        Attachment(
            name="init.annotation_preflight_context",
            kind="structured",
            label="Annotation Preflight Context",
            content={
                "flow": "detailed",
                "range": [s, e],
                "required_tools": _preflight_analysis_ids(
                    DETAILED_PREFLIGHT_TOOL_IDS,
                    DETAILED_PREFLIGHT_QUERY_SPECS,
                ),
                "tool_output_tags": tags,
                "statistical_events": [event["event"] for event in events],
                "semantic_evidence_text": event_text,
                "semantic_search_text": semantic_search_text,
            },
            content_schema="annotation_preflight_context",
        ),
    ])

    return PreflightContext(
        prompt_block=_prompt_block(
            s,
            e,
            event_text,
        ),
        attachments=attachments,
        label_candidates=[],
    )


def _build_detailed_events(
    df,
    start: int,
    end: int,
    tool_outputs: Sequence[Tuple[str, Dict[str, Any]]],
    parent_main_labels: Sequence[str] = (),
) -> List[Dict[str, Any]]:
    by_tool = {tool_id: content for tool_id, content in tool_outputs}
    phases = _phase_windows(by_tool)
    racing_context = _is_racing_parent_context(parent_main_labels)
    events: List[Dict[str, Any]] = []

    _extend(events, _phase_marker_events(phases))
    _extend(events, _shape_events(start, end, by_tool, phases))
    if racing_context:
        _extend(events, _opponent_relative_motion_events(df, start, end, by_tool))
        _extend(events, _balance_and_grip_events(df, start, end, by_tool, phases))
        return _dedupe_events(events)

    _extend(events, _peak_comparison_events(df, start, end, by_tool, "brake"))
    _extend(events, _peak_comparison_events(df, start, end, by_tool, "throttle"))
    _extend(events, _input_timing_comparison_events(df, start, end))
    _extend(events, _local_input_shape_events(df, start, end))
    _extend(events, _time_delta_events(df, start, end, by_tool, phases))
    _extend(events, _trajectory_events(df, start, end, by_tool, phases))
    _extend(events, _speed_events(df, start, end, by_tool, phases))
    _extend(events, _gear_and_rpm_events(df, start, end, phases))
    _extend(events, _balance_and_grip_events(df, start, end, by_tool, phases))

    return _dedupe_events(events)


def _is_racing_parent_context(parent_main_labels: Sequence[str]) -> bool:
    return any(str(label_id) in RACING_PARENT_LABELS for label_id in parent_main_labels)


def _phase_marker_events(phases: List[Dict[str, int]]) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    for phase in phases:
        entry = phase.get("entry")
        apex = phase.get("apex")
        exit_ = phase.get("exit")
        if not all(isinstance(value, int) for value in (entry, apex, exit_)):
            continue
        events.append(_event(
            "corner phase markers",
            "whole_range",
            [entry, exit_],
            {
                "entry_start_iloc": entry,
                "apex_iloc": apex,
                "exit_end_iloc": exit_,
                "direction": phase.get("direction"),
            },
            "strong",
            ["compute_expert_phases"],
        ))
    return events


def _shape_events(
    start: int,
    end: int,
    by_tool: Dict[str, Dict[str, Any]],
    phases: List[Dict[str, int]],
) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    shape = by_tool.get("measure_segment_shape") or {}
    base = shape.get("base_segment_shape")
    if isinstance(base, dict):
        shape_key = str(base.get("shape_key") or "").strip()
        event = _shape_event_name(shape_key, _BASE_SEGMENT_SHAPE_WORDS)
        if event:
            events.append(_event(
                event,
                "straight" if shape_key == "straight" else "whole_range",
                [start, end],
                {"shape_key": shape_key, "reason": base.get("reason")},
                "strong",
                ["measure_segment_shape"],
            ))
    refinement = shape.get("corner_shape_refinement")
    if isinstance(refinement, dict):
        shape_key = str(refinement.get("shape_key") or "").strip()
        event = _shape_event_name(shape_key, _CORNER_REFINEMENT_WORDS)
        if event:
            events.append(_event(
                event,
                "whole_range",
                [start, end],
                {
                    "shape_key": shape_key,
                    "reason": refinement.get("reason"),
                    "turn_angle_degrees": refinement.get("turn_angle_degrees"),
                    "is_near_u_turn": refinement.get("is_near_u_turn"),
                    "relative_curvature_change": refinement.get(
                        "relative_curvature_change"
                    ),
                },
                "strong",
                ["measure_segment_shape"],
            ))
    altitude = shape.get("altitude")
    if isinstance(altitude, dict):
        for phase_name in ("entry", "apex", "exit"):
            summary = altitude.get(phase_name)
            if not isinstance(summary, dict):
                continue
            trend = str(summary.get("trend") or "").strip()
            event = _ALTITUDE_WORDS.get((phase_name, trend))
            if not event:
                continue
            events.append(_event(
                event,
                phase_name,
                [summary.get("start_iloc"), summary.get("end_iloc")],
                {
                    "trend": trend,
                    "slope_angle_degrees": summary.get("slope_angle_degrees"),
                    "horizontal_distance_units": summary.get(
                        "horizontal_distance_units"
                    ),
                    "delta_m": summary.get("delta_m"),
                    "start_altitude_m": summary.get("start_altitude_m"),
                    "end_altitude_m": summary.get("end_altitude_m"),
                },
                "moderate",
                ["measure_segment_shape"],
            ))
    return events


def _shape_event_name(
    shape_key: str,
    vocabulary: Dict[str, Tuple[str, str]],
) -> Optional[str]:
    words = vocabulary.get(shape_key)
    if not words:
        return None
    return "; ".join(words).lower()


def _opponent_relative_motion_events(
    df,
    start: int,
    end: int,
    by_tool: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    slot = _primary_opponent_slot(by_tool)
    if slot is None:
        return []

    motion = _opponent_relative_motion(df, start, end, slot)
    if not motion:
        return []

    ilocs = motion["ilocs"]
    signed_long = motion["signed_long_gap_m"]
    lateral = motion["lateral_offset_m"]
    distance = motion["distance_m"]
    events: List[Dict[str, Any]] = []

    start_iloc = int(ilocs[0])
    end_iloc = int(ilocs[-1])
    start_gap = float(signed_long[0])
    end_gap = float(signed_long[-1])
    closest_pos = _min_index(distance)
    closest_iloc = int(ilocs[closest_pos])

    start_name = (
        "opponent started ahead of the driver"
        if start_gap > 1.5
        else "driver started ahead of the opponent"
        if start_gap < -1.5
        else "driver and opponent started nearly level"
    )
    events.append(_event(
        start_name,
        "whole_range",
        [start_iloc, start_iloc],
        {
            "index": start_iloc,
            "signed_gap_m": start_gap,
            "slot": slot,
        },
        "strong",
        ["local_opponent_relative_position"],
    ))

    end_name = (
        "opponent ended ahead of the driver"
        if end_gap > 1.5
        else "driver ended ahead of the opponent"
        if end_gap < -1.5
        else "driver and opponent ended nearly level"
    )
    events.append(_event(
        end_name,
        "whole_range",
        [end_iloc, end_iloc],
        {
            "index": end_iloc,
            "signed_gap_m": end_gap,
            "slot": slot,
        },
        "strong",
        ["local_opponent_relative_position"],
    ))

    if start_gap > 1.5 and end_gap < -1.5:
        events.append(_event(
            "gap flipped from opponent ahead to driver ahead",
            "whole_range",
            [start_iloc, end_iloc],
            {
                "start_index": start_iloc,
                "end_index": end_iloc,
                "start_gap_m": start_gap,
                "end_gap_m": end_gap,
                "slot": slot,
            },
            "strong",
            ["local_opponent_relative_position"],
        ))
    elif start_gap < -1.5 and end_gap > 1.5:
        events.append(_event(
            "gap flipped from driver ahead to opponent ahead",
            "whole_range",
            [start_iloc, end_iloc],
            {
                "start_index": start_iloc,
                "end_index": end_iloc,
                "start_gap_m": start_gap,
                "end_gap_m": end_gap,
                "slot": slot,
            },
            "strong",
            ["local_opponent_relative_position"],
        ))

    if abs(end_gap) < abs(start_gap) - max(1.0, abs(start_gap) * 0.1):
        events.append(_event(
            "gap to the opponent shrank",
            "whole_range",
            [start_iloc, end_iloc],
            {
                "start_index": start_iloc,
                "end_index": end_iloc,
                "start_gap_m": abs(start_gap),
                "end_gap_m": abs(end_gap),
                "slot": slot,
            },
            "strong",
            ["local_opponent_relative_position"],
        ))

    side = "left" if float(lateral[closest_pos]) > 0.0 else "right"
    events.append(_event(
        f"opponent was on the driver's {side} side",
        "whole_range",
        [closest_iloc, closest_iloc],
        {
            "index": closest_iloc,
            "lateral_offset_m": float(lateral[closest_pos]),
            "distance_m": float(distance[closest_pos]),
            "slot": slot,
        },
        "strong",
        ["local_opponent_relative_position"],
    ))

    for run_start, run_end in _alongside_runs(ilocs, signed_long, lateral):
        first = int(run_start)
        last = int(run_end)
        local_start = _index_position(ilocs, first)
        if local_start is None:
            continue
        actor = (
            "driver"
            if float(signed_long[local_start]) >= 0.0
            else "opponent"
        )
        event_name = (
            "driver drew alongside the opponent"
            if actor == "driver"
            else "opponent drew alongside the driver"
        )
        events.append(_event(
            event_name,
            "whole_range",
            [first, last],
            {
                "start_index": first,
                "end_index": last,
                "slot": slot,
            },
            "strong" if last > first else "moderate",
            ["local_opponent_relative_position"],
        ))
        break

    relative_speed = motion.get("relative_speed")
    speed_indices = motion.get("speed_indices")
    if relative_speed and speed_indices and len(relative_speed) >= 2:
        speed_change = float(relative_speed[-1]) - float(relative_speed[0])
        if speed_change > _motion_change_guard(relative_speed, 0.25):
            events.append(_event(
                "driver gained relative speed",
                "whole_range",
                [int(speed_indices[0]), int(speed_indices[-1])],
                {
                    "start_index": int(speed_indices[0]),
                    "end_index": int(speed_indices[-1]),
                    "start_relative_speed": float(relative_speed[0]),
                    "end_relative_speed": float(relative_speed[-1]),
                    "speed_units": motion.get("speed_units"),
                    "slot": slot,
                },
                "strong",
                ["local_opponent_relative_speed"],
            ))

    acceleration_diff = motion.get("acceleration_diff")
    accel_indices = motion.get("acceleration_indices")
    if acceleration_diff and accel_indices:
        median_accel = _median(acceleration_diff)
        if median_accel is not None and median_accel > _motion_change_guard(acceleration_diff, 0.05):
            events.append(_event(
                "driver accelerated better than the opponent",
                "whole_range",
                [int(accel_indices[0]), int(accel_indices[-1])],
                {
                    "start_index": int(accel_indices[0]),
                    "end_index": int(accel_indices[-1]),
                    "median_acceleration_advantage": median_accel,
                    "acceleration_units": motion.get("acceleration_units"),
                    "slot": slot,
                },
                "strong",
                ["local_opponent_relative_acceleration"],
            ))

    deceleration_diff = motion.get("deceleration_diff")
    decel_indices = motion.get("deceleration_indices")
    if deceleration_diff and decel_indices:
        median_decel = _median(deceleration_diff)
        guard = _motion_change_guard(deceleration_diff, 0.05)
        if median_decel is not None and abs(median_decel) > guard:
            event_name = (
                "driver slowed more than the opponent"
                if median_decel > 0
                else "driver slowed less than the opponent"
            )
            events.append(_event(
                event_name,
                "whole_range",
                [int(decel_indices[0]), int(decel_indices[-1])],
                {
                    "start_index": int(decel_indices[0]),
                    "end_index": int(decel_indices[-1]),
                    "median_deceleration_difference": median_decel,
                    "acceleration_units": motion.get("acceleration_units"),
                    "slot": slot,
                },
                "strong",
                ["local_opponent_relative_deceleration"],
            ))

    return events


def _primary_opponent_slot(by_tool: Dict[str, Dict[str, Any]]) -> Optional[int]:
    interaction = by_tool.get("classify_opponent_interaction") or {}
    for key in ("primary_slot_for_role", "targeted_car_slot"):
        value = interaction.get(key)
        if isinstance(value, int):
            return value
    nearest = by_tool.get("find_nearest_opponent") or {}
    candidates = nearest.get("candidates")
    if isinstance(candidates, list):
        for candidate in candidates:
            if isinstance(candidate, dict) and isinstance(candidate.get("slot"), int):
                return int(candidate["slot"])
    return None


def _opponent_relative_motion(
    df,
    start: int,
    end: int,
    slot: int,
) -> Optional[Dict[str, Any]]:
    if df is None:
        return None
    required = {
        "Graphics_player_pos_x",
        "Graphics_player_pos_y",
        f"Car_{slot}_pos_x",
        f"Car_{slot}_pos_y",
    }
    if not required.issubset(set(getattr(df, "columns", []))):
        return None

    try:
        seg = df.loc[int(start): int(end)]
    except Exception:  # noqa: BLE001
        return None
    if len(seg) < 2:
        return None

    try:
        import numpy as np
        from app.shared.annotation_agent_tools import (
            _active_opponent_mask,
            _relative_position_frame,
        )

        player_x = seg["Graphics_player_pos_x"].to_numpy(dtype=float)
        player_y = seg["Graphics_player_pos_y"].to_numpy(dtype=float)
        opponent_x = seg[f"Car_{slot}_pos_x"].to_numpy(dtype=float)
        opponent_y = seg[f"Car_{slot}_pos_y"].to_numpy(dtype=float)
        active = _active_opponent_mask(
            seg,
            int(slot),
            opponent_x,
            opponent_y,
            player_x,
            player_y,
        )
        signed_long, lateral, _player_s, _player_d, frame_name = (
            _relative_position_frame(seg, player_x, player_y, opponent_x, opponent_y)
        )
        distance = np.sqrt((opponent_x - player_x) ** 2 + (opponent_y - player_y) ** 2)
        finite = (
            active
            & np.isfinite(signed_long)
            & np.isfinite(lateral)
            & np.isfinite(distance)
        )
        positions = np.where(finite)[0]
    except Exception:  # noqa: BLE001
        return None

    if positions.size < 2:
        return None

    iloc_index = [int(value) for value in seg.index.to_list()]
    ilocs = [iloc_index[int(pos)] for pos in positions]
    motion: Dict[str, Any] = {
        "slot": int(slot),
        "coordinate_frame": frame_name,
        "ilocs": ilocs,
        "signed_long_gap_m": [float(signed_long[int(pos)]) for pos in positions],
        "lateral_offset_m": [float(lateral[int(pos)]) for pos in positions],
        "distance_m": [float(distance[int(pos)]) for pos in positions],
    }

    speed = _player_opponent_speed_motion(
        seg,
        player_x,
        player_y,
        opponent_x,
        opponent_y,
        finite,
    )
    motion.update(speed)
    return motion


def _player_opponent_speed_motion(
    seg,
    player_x,
    player_y,
    opponent_x,
    opponent_y,
    finite,
) -> Dict[str, Any]:
    try:
        import numpy as np
    except Exception:  # noqa: BLE001
        return {}
    if len(seg) < 3:
        return {}

    dt = np.ones(len(seg) - 1, dtype=float)
    speed_units = "m/sample"
    acceleration_units = "m/sample^2"
    if "Graphics_current_time" in getattr(seg, "columns", []):
        time_ms = seg["Graphics_current_time"].to_numpy(dtype=float)
        raw_dt = (time_ms[1:] - time_ms[:-1]) / 1000.0
        if np.isfinite(raw_dt).any() and np.nanmedian(raw_dt) > 1e-6:
            dt = raw_dt
            speed_units = "m/s"
            acceleration_units = "m/s^2"

    valid_step = finite[1:] & finite[:-1] & np.isfinite(dt) & (dt > 1e-9)
    player_step = np.sqrt((player_x[1:] - player_x[:-1]) ** 2 + (player_y[1:] - player_y[:-1]) ** 2)
    opponent_step = np.sqrt((opponent_x[1:] - opponent_x[:-1]) ** 2 + (opponent_y[1:] - opponent_y[:-1]) ** 2)
    player_speed = np.full(len(seg) - 1, np.nan, dtype=float)
    opponent_speed = np.full(len(seg) - 1, np.nan, dtype=float)
    np.divide(player_step, dt, out=player_speed, where=valid_step)
    np.divide(opponent_step, dt, out=opponent_speed, where=valid_step)
    relative_speed = player_speed - opponent_speed
    speed_positions = np.where(valid_step & np.isfinite(relative_speed))[0] + 1
    iloc_index = [int(value) for value in seg.index.to_list()]

    out: Dict[str, Any] = {
        "speed_units": speed_units,
        "acceleration_units": acceleration_units,
    }
    if speed_positions.size:
        out["speed_indices"] = [iloc_index[int(pos)] for pos in speed_positions]
        out["relative_speed"] = [
            float(relative_speed[int(pos) - 1])
            for pos in speed_positions
        ]

    if relative_speed.size < 2:
        return out

    accel_dt = dt[1:]
    valid_accel = (
        valid_step[1:]
        & valid_step[:-1]
        & np.isfinite(accel_dt)
        & (accel_dt > 1e-9)
    )
    player_accel = np.full(relative_speed.size - 1, np.nan, dtype=float)
    opponent_accel = np.full(relative_speed.size - 1, np.nan, dtype=float)
    np.divide(
        player_speed[1:] - player_speed[:-1],
        accel_dt,
        out=player_accel,
        where=valid_accel,
    )
    np.divide(
        opponent_speed[1:] - opponent_speed[:-1],
        accel_dt,
        out=opponent_accel,
        where=valid_accel,
    )
    accel_diff = player_accel - opponent_accel
    accel_positions = np.where(valid_accel & np.isfinite(accel_diff))[0] + 2
    if accel_positions.size:
        out["acceleration_indices"] = [
            iloc_index[int(pos)]
            for pos in accel_positions
        ]
        out["acceleration_diff"] = [
            float(accel_diff[int(pos) - 2])
            for pos in accel_positions
        ]

    player_decel = np.where(player_accel < 0.0, -player_accel, 0.0)
    opponent_decel = np.where(opponent_accel < 0.0, -opponent_accel, 0.0)
    decel_diff = player_decel - opponent_decel
    decel_mask = valid_accel & np.isfinite(decel_diff) & (
        (player_decel > 0.0) | (opponent_decel > 0.0)
    )
    decel_positions = np.where(decel_mask)[0] + 2
    if decel_positions.size:
        out["deceleration_indices"] = [
            iloc_index[int(pos)]
            for pos in decel_positions
        ]
        out["deceleration_diff"] = [
            float(decel_diff[int(pos) - 2])
            for pos in decel_positions
        ]
    return out


def _alongside_runs(
    ilocs: Sequence[int],
    signed_long: Sequence[float],
    lateral: Sequence[float],
) -> List[Tuple[int, int]]:
    mask = [
        abs(float(long_gap)) <= 6.0
        and 1.25 <= abs(float(side_gap)) <= 6.0
        for long_gap, side_gap in zip(signed_long, lateral)
    ]
    runs: List[Tuple[int, int]] = []
    start: Optional[int] = None
    last: Optional[int] = None
    for iloc, active in zip(ilocs, mask):
        if active and start is None:
            start = int(iloc)
        if active:
            last = int(iloc)
            continue
        if start is not None and last is not None:
            runs.append((start, last))
        start = last = None
    if start is not None and last is not None:
        runs.append((start, last))
    return sorted(runs, key=lambda item: item[1] - item[0], reverse=True)


def _index_position(values: Sequence[int], target: int) -> Optional[int]:
    for pos, value in enumerate(values):
        if int(value) == int(target):
            return pos
    return None


def _min_index(values: Sequence[float]) -> int:
    return min(range(len(values)), key=lambda pos: float(values[pos]))


def _motion_change_guard(values: Sequence[float], floor: float) -> float:
    finite = [abs(float(value)) for value in values if _is_number(value)]
    baseline = float(_median(finite) or 0.0)
    return max(float(floor), baseline * 0.25)


def _peak_comparison_events(
    df,
    start: int,
    end: int,
    by_tool: Dict[str, Dict[str, Any]],
    kind: str,
) -> List[Dict[str, Any]]:
    if kind == "brake":
        player_tool = "query_telemetry.find_extremum.brake.player.max"
        expert_tool = "query_telemetry.find_extremum.brake.expert.max"
        phrase = "peak brake pressure"
    else:
        player_tool = "query_telemetry.find_extremum.throttle.player.min"
        expert_tool = "query_telemetry.find_extremum.throttle.expert.min"
        phrase = "lowest throttle pressure"
    player = _query_result(by_tool.get(player_tool))
    expert = _query_result(by_tool.get(expert_tool))
    if not player or not expert:
        return []
    player_value = player.get("value")
    expert_value = expert.get("value")
    if not isinstance(player_value, (int, float)) or not isinstance(
        expert_value, (int, float)
    ):
        return []
    delta = float(player_value) - float(expert_value)
    player_iloc = player.get("iloc")
    measurements = {
        "player_value": player_value,
        "expert_value": expert_value,
        "delta": delta,
        "player_iloc": player_iloc,
        "expert_iloc": expert.get("iloc"),
    }
    if kind == "brake" and isinstance(player_iloc, int):
        speed_gap_percent = dict(_speed_gap_percent_values(df, start, end)).get(
            player_iloc
        )
        if speed_gap_percent is not None:
            measurements["speed_gap_percent_at_player_peak"] = speed_gap_percent
            measurements["speed_gap_relation_at_player_peak"] = _gap_percent_relation(
                "speed",
                speed_gap_percent,
            )
    if abs(delta) < 0.05:
        return [_event(
            f"{phrase} about same as expert",
            "unknown",
            [player_iloc, player_iloc] if isinstance(player_iloc, int) else None,
            measurements,
            "strong" if abs(delta) <= 0.02 else "moderate",
            [player_tool, expert_tool],
        )]
    return [_event(
        f"{phrase} {'higher' if delta > 0 else 'lower'} than expert",
        "unknown",
        [player_iloc, player_iloc] if isinstance(player_iloc, int) else None,
        measurements,
        "strong" if abs(delta) >= 0.15 else "moderate",
        [player_tool, expert_tool],
    )]


def _input_timing_comparison_events(
    df,
    start: int,
    end: int,
) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    for player_col, expert_col, direction, phrase in (
        (
            "Physics_brake",
            "expert_optimal_brake",
            "increase",
            "brake initiation",
        ),
        (
            "Physics_brake",
            "expert_optimal_brake",
            "decrease",
            "brake release",
        ),
        (
            "Physics_gas",
            "expert_optimal_throttle",
            "increase",
            "throttle application",
        ),
        (
            "Physics_gas",
            "expert_optimal_throttle",
            "decrease",
            "throttle release",
        ),
    ):
        player = _action_profile(df, start, end, player_col, direction)
        expert = _action_profile(df, start, end, expert_col, direction)
        source = f"local_{phrase.replace(' ', '_')}_shape_comparison"
        if not player or not expert:
            for boundary in ("onset", "end"):
                events.append(_event(
                    f"{phrase} {boundary} comparison unavailable",
                    "unknown",
                    None,
                    {
                        "player_action_detected": bool(player),
                        "expert_action_detected": bool(expert),
                        "direction": direction,
                        "boundary": boundary,
                        "decision_basis": "shape_change_comparison",
                    },
                    "weak",
                    [source],
                ))
            continue
        events.extend(_action_boundary_events(phrase, player, expert, source))
    return events


def _action_boundary_events(
    phrase: str,
    player: Dict[str, Any],
    expert: Dict[str, Any],
    source: str,
) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    for boundary, key, delta_key, band_key in (
        ("onset", "start_index", "start_delta_iloc", "start_band"),
        ("end", "end_index", "end_delta_iloc", "end_band"),
    ):
        player_iloc = player.get(key)
        expert_iloc = expert.get(key)
        if not isinstance(player_iloc, int) or not isinstance(expert_iloc, int):
            continue
        delta = player_iloc - expert_iloc
        relation = (
            "earlier" if delta < 0
            else "later" if delta > 0
            else "aligned with"
        )
        event_name = (
            f"{phrase} {boundary} {relation} expert"
            if relation == "aligned with"
            else f"{phrase} {boundary} {relation} than expert"
        )
        events.append(_event(
            event_name,
            "unknown",
            _range_from_values(player_iloc, expert_iloc),
            {
                "player_start_index": player.get("start_index"),
                "expert_start_index": expert.get("start_index"),
                "start_delta_iloc": (
                    player.get("start_index") - expert.get("start_index")
                    if isinstance(player.get("start_index"), int)
                    and isinstance(expert.get("start_index"), int)
                    else None
                ),
                "player_end_index": player.get("end_index"),
                "expert_end_index": expert.get("end_index"),
                "end_delta_iloc": (
                    player.get("end_index") - expert.get("end_index")
                    if isinstance(player.get("end_index"), int)
                    and isinstance(expert.get("end_index"), int)
                    else None
                ),
                delta_key: delta,
                f"player_{band_key}": player.get(band_key),
                f"expert_{band_key}": expert.get(band_key),
                "player_total_movement": player.get("total_movement"),
                "expert_total_movement": expert.get("total_movement"),
                "direction": player.get("direction"),
                "boundary": boundary,
                "decision_basis": "shape_change_comparison",
            },
            "strong" if abs(delta) >= 2 else "moderate",
            [source],
        ))
    return events


def _local_input_shape_events(
    df,
    start: int,
    end: int,
) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    for kind, player_col, expert_col, noun in (
        ("brake", "Physics_brake", "expert_optimal_brake", "brake"),
        ("throttle", "Physics_gas", "expert_optimal_throttle", "throttle"),
    ):
        source = f"local_{kind}_shape_statistics"
        application_timing = (
            "brake initiation" if kind == "brake" else "throttle application"
        )
        for direction, timing_phrase in (
            ("increase", application_timing),
            ("decrease", f"{noun} release"),
        ):
            player = _action_profile(df, start, end, player_col, direction)
            expert = _action_profile(df, start, end, expert_col, direction)
            if not player or not expert:
                continue
            timing = _compare_action_timing(player, expert)
            if timing:
                events.append(_action_timing_event(
                    timing_phrase,
                    player,
                    expert,
                    timing,
                    source,
                ))
    events.extend(_overlap_events(df, start, end))
    return events


def _action_timing_event(
    phrase: str,
    player: Dict[str, Any],
    expert: Dict[str, Any],
    comparison: Dict[str, Any],
    source: str,
) -> Dict[str, Any]:
    delta = comparison["start_delta_iloc"]
    return _event(
        f"{phrase} onset {'earlier' if delta < 0 else 'later'} than expert",
        "unknown",
        _range_from_values(player.get("start_index"), expert.get("start_index")),
        {
            "player_start_index": player.get("start_index"),
            "expert_start_index": expert.get("start_index"),
            "start_delta_iloc": delta,
            "player_start_band": player.get("start_band"),
            "expert_start_band": expert.get("start_band"),
            "boundary_uncertainty_iloc": comparison.get("boundary_uncertainty_iloc"),
            "decision_basis": "fuzzy_change_speed_comparison",
        },
        comparison["confidence"],
        [source],
    )


def _trajectory_events(
    df,
    start: int,
    end: int,
    by_tool: Dict[str, Dict[str, Any]],
    phases: List[Dict[str, int]],
) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    max_result = _query_result(
        by_tool.get("query_telemetry.find_extremum.trajectory_offset.max")
    )
    min_result = _query_result(
        by_tool.get("query_telemetry.find_extremum.trajectory_offset.min")
    )
    max_value = max_result.get("value") if isinstance(max_result, dict) else None
    min_value = min_result.get("value") if isinstance(min_result, dict) else None
    if isinstance(max_value, (int, float)) and max_value >= 0.5:
        iloc = max_result.get("iloc")
        events.append(_event(
            "trajectory wider than expert",
            _phase_for_iloc(iloc, phases) if isinstance(iloc, int) else "unknown",
            [iloc, iloc] if isinstance(iloc, int) else None,
            {"value": max_value, "iloc": iloc},
            "strong" if max_value >= 1.0 else "moderate",
            ["query_telemetry.find_extremum.trajectory_offset.max"],
        ))
    if isinstance(min_value, (int, float)) and min_value <= -0.5:
        iloc = min_result.get("iloc")
        events.append(_event(
            "trajectory tighter than expert",
            _phase_for_iloc(iloc, phases) if isinstance(iloc, int) else "unknown",
            [iloc, iloc] if isinstance(iloc, int) else None,
            {"value": min_value, "iloc": iloc},
            "strong" if min_value <= -1.0 else "moderate",
            ["query_telemetry.find_extremum.trajectory_offset.min"],
        ))
    if (
        isinstance(max_value, (int, float))
        and isinstance(min_value, (int, float))
        and max_value >= 0.5
        and min_value <= -0.5
    ):
        events.append(_event(
            "trajectory crosses line",
            "whole_range",
            [start, end],
            {"min_value": min_value, "max_value": max_value},
            "strong",
            [
                "query_telemetry.find_extremum.trajectory_offset.max",
                "query_telemetry.find_extremum.trajectory_offset.min",
            ],
        ))

    slope = _query_analysis(
        by_tool.get("query_telemetry.compute_slope.trajectory_offset")
    )
    total = slope.get("total_change") if isinstance(slope, dict) else {}
    absolute = slope.get("absolute_offset") if isinstance(slope, dict) else {}
    if isinstance(total, dict):
        domain = total.get("domain_direction")
        if domain == "moving_wider":
            events.append(_event(
                "moving toward positive",
                "whole_range",
                [start, end],
                {"domain_direction": domain, "change": total.get("value")},
                "moderate",
                ["query_telemetry.compute_slope.trajectory_offset"],
            ))
        elif domain == "moving_tighter":
            events.append(_event(
                "moving toward negative",
                "whole_range",
                [start, end],
                {"domain_direction": domain, "change": total.get("value")},
                "moderate",
                ["query_telemetry.compute_slope.trajectory_offset"],
            ))
    if isinstance(absolute, dict) and absolute.get("moves_toward_expert_line") is True:
        events.append(_event(
            "recovery toward expert line",
            "whole_range",
            [start, end],
            absolute,
            "strong",
            ["query_telemetry.compute_slope.trajectory_offset"],
        ))

    events.extend(_trajectory_apex_timing_events(df, phases))
    similarity = _query_result(
        by_tool.get("query_telemetry.measure_trajectory_similarity.driver_expert_path")
    )
    similarity_extra = similarity.get("extra") if isinstance(similarity, dict) else {}
    similarity_score = (
        similarity_extra.get("similarity_score")
        if isinstance(similarity_extra, dict)
        else None
    )
    events.extend(_trajectory_phase_side_events(
        df,
        start,
        end,
        phases,
        similarity_score=similarity_score,
    ))
    return events


def _time_delta_events(
    df,
    start: int,
    end: int,
    _by_tool: Dict[str, Dict[str, Any]],
    phases: List[Dict[str, int]],
) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    events.extend(_time_gap_percent_events(df, start, end, phases))
    return events


def _time_gap_percent_events(
    df,
    start: int,
    end: int,
    phases: List[Dict[str, int]],
) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    for phase, range_ in [
        ("whole_range", [start, end]),
        *_time_gap_slope_ranges(start, end, phases),
    ]:
        values = _time_gap_percent_values(df, range_[0], range_[1])
        analysis = _gap_percent_change(values, kind="time")
        if not analysis:
            continue
        if phase == "whole_range":
            if analysis["relative_gain_percent"] < 0.0:
                event_name = "gap grows"
            else:
                event_name = "gap shrinks"
        else:
            event_name = _time_gap_event_name(analysis["direction"], phase)
        events.append(_event(
            event_name,
            phase,
            range_,
            analysis,
            (
                "strong"
                if analysis.get("threshold_state") == "label_threshold_met"
                else "moderate"
            ),
            ["local_expert_time_difference_percent_gap"],
        ))
        if phase == "whole_range" and analysis["relative_gain_percent"] < 0.0:
            events.append(_event(
                "time loss",
                "whole_range",
                [start, end],
                analysis,
                "moderate",
                ["local_expert_time_difference_percent_gap"],
            ))
    return events


def _time_gap_slope_ranges(
    start: int,
    end: int,
    phases: List[Dict[str, int]],
) -> List[Tuple[str, List[int]]]:
    ranges: List[Tuple[str, List[int]]] = []
    for phase in phases:
        entry = phase.get("entry")
        apex = phase.get("apex")
        exit_ = phase.get("exit")
        if not all(isinstance(value, int) for value in (entry, apex, exit_)):
            continue
        entry_range = [max(start, entry), min(end, apex)]
        if entry_range[1] > entry_range[0]:
            ranges.append(("entry", entry_range))
        apex_half_window = max(2, int(0.05 * max(end - start + 1, 1)))
        apex_range = [
            max(start, apex - apex_half_window),
            min(end, apex + apex_half_window),
        ]
        if apex_range[1] > apex_range[0]:
            ranges.append(("apex", apex_range))
        exit_range = [max(start, apex + 1), min(end, exit_)]
        if exit_range[1] > exit_range[0]:
            ranges.append(("exit", exit_range))
    if ranges:
        return ranges
    return [("whole_range", [start, end])]


def _trajectory_apex_timing_events(
    df,
    phases: List[Dict[str, int]],
) -> List[Dict[str, Any]]:
    if df is None or not phases:
        return []
    required = (
        "Graphics_player_pos_x",
        "Graphics_player_pos_y",
        "expert_optimal_player_pos_x",
        "expert_optimal_player_pos_y",
    )
    if any(column not in getattr(df, "columns", []) for column in required):
        return []

    events: List[Dict[str, Any]] = []
    for phase in phases:
        entry = phase.get("entry")
        expert_apex = phase.get("apex")
        exit_ = phase.get("exit")
        if not all(isinstance(value, int) for value in (entry, expert_apex, exit_)):
            continue
        player_apex = _curve_apex_iloc(
            df,
            entry,
            exit_,
            "Graphics_player_pos_x",
            "Graphics_player_pos_y",
        )
        if player_apex is None:
            continue
        expert_range = _apex_range(expert_apex, entry, exit_)
        player_range = _apex_range(player_apex, entry, exit_)
        delta = player_apex - expert_apex
        if player_range[1] < expert_range[0]:
            relation = "earlier"
            boundary_gap = expert_range[0] - player_range[1]
        elif player_range[0] > expert_range[1]:
            relation = "later"
            boundary_gap = player_range[0] - expert_range[1]
        else:
            continue

        event = _event(
            f"player reaches apex {relation} than expert",
            "apex",
            [
                min(player_range[0], expert_range[0]),
                max(player_range[1], expert_range[1]),
            ],
            {
                "player_apex_iloc": player_apex,
                "expert_apex_iloc": expert_apex,
                "apex_delta_iloc": delta,
                "player_apex_range": player_range,
                "expert_apex_range": expert_range,
                "apex_boundary_gap_iloc": boundary_gap,
                "player_apex_x": _value_at_iloc(
                    df,
                    entry,
                    exit_,
                    "Graphics_player_pos_x",
                    player_apex,
                ),
                "player_apex_y": _value_at_iloc(
                    df,
                    entry,
                    exit_,
                    "Graphics_player_pos_y",
                    player_apex,
                ),
                "expert_apex_x": _value_at_iloc(
                    df,
                    entry,
                    exit_,
                    "expert_optimal_player_pos_x",
                    expert_apex,
                ),
                "expert_apex_y": _value_at_iloc(
                    df,
                    entry,
                    exit_,
                    "expert_optimal_player_pos_y",
                    expert_apex,
                ),
                "decision_basis": "player_curvature_peak_vs_expert_phase_apex",
            },
            "strong" if boundary_gap >= 2 else "moderate",
            ["local_player_expert_apex_curvature_comparison"],
        )
        event["semantic_search_terms"] = [
            (
                "too early compared to expert apex"
                if relation == "earlier"
                else "too late compared to expert apex"
            )
        ]
        events.append(event)
    return events


def _apex_range(apex_iloc: int, entry: int, exit_: int) -> List[int]:
    half_window = max(2, int(0.05 * max(exit_ - entry + 1, 1)))
    return [
        max(entry, int(apex_iloc) - half_window),
        min(exit_, int(apex_iloc) + half_window),
    ]


def _curve_apex_iloc(
    df,
    start: int,
    end: int,
    x_column: str,
    y_column: str,
) -> Optional[int]:
    try:
        segment = df.loc[int(start): int(end), [x_column, y_column]]
    except Exception:  # noqa: BLE001
        return None
    if len(segment) < 5:
        return None

    points: List[Tuple[int, float, float]] = []
    for iloc, row in segment.iterrows():
        try:
            x = float(row[x_column])
            y = float(row[y_column])
        except (TypeError, ValueError):
            continue
        if math.isfinite(x) and math.isfinite(y):
            points.append((int(iloc), x, y))
    if len(points) < 5:
        return None

    best: Optional[Tuple[float, int]] = None
    for pos in range(1, len(points) - 1):
        _, prev_x, prev_y = points[pos - 1]
        iloc, x, y = points[pos]
        _, next_x, next_y = points[pos + 1]
        dx = (next_x - prev_x) / 2.0
        dy = (next_y - prev_y) / 2.0
        ddx = next_x - (2.0 * x) + prev_x
        ddy = next_y - (2.0 * y) + prev_y
        denom = (dx * dx + dy * dy) ** 1.5
        if denom <= 1e-9:
            continue
        curvature = abs((dx * ddy - dy * ddx) / denom)
        if best is None or curvature > best[0]:
            best = (curvature, iloc)
    if best is None or best[0] <= 0.0:
        return None
    return best[1]


def _trajectory_phase_side_events(
    df,
    start: int,
    end: int,
    phases: List[Dict[str, int]],
    *,
    similarity_score: Any = None,
) -> List[Dict[str, Any]]:
    values = _series_values(df, start, end, "trajectory_offset", graph_id="trajectory_offset")
    if not values:
        return []
    events: List[Dict[str, Any]] = []
    similarity = _as_float(similarity_score)
    is_aligned_trajectory = (
        similarity is not None
        and similarity >= TRAJECTORY_ALIGNED_SIMILARITY_THRESHOLD
    )
    for phase in phases:
        spans = (
            ("entry", phase.get("entry"), phase.get("apex")),
            ("apex", phase.get("apex"), phase.get("apex")),
            ("exit", phase.get("apex"), phase.get("exit")),
        )
        for phase_name, lo, hi in spans:
            if not isinstance(lo, int) or not isinstance(hi, int):
                continue
            median = _median([
                value
                for iloc, value in values
                if lo <= iloc <= hi and isinstance(value, (int, float))
            ])
            if median is None:
                continue
            if is_aligned_trajectory:
                events.append(_event(
                    f"{phase_name} trajectory aligned with expert",
                    phase_name,
                    [lo, hi],
                    {
                        "median_offset": median,
                        "similarity_score": similarity,
                    },
                    "strong",
                    [
                        "trajectory_offset_phase_statistics",
                        "query_telemetry.measure_trajectory_similarity.driver_expert_path",
                    ],
                ))
                continue
            if abs(median) < 0.5:
                continue
            events.append(_event(
                f"{phase_name} trajectory {'wider' if median > 0 else 'tighter'} than expert",
                phase_name,
                [lo, hi],
                {
                    "median_offset": median,
                    "similarity_score": similarity,
                },
                "strong" if abs(median) >= 1.0 else "moderate",
                ["trajectory_offset_phase_statistics"],
            ))
    return events


def _speed_events(
    df,
    start: int,
    end: int,
    by_tool: Dict[str, Dict[str, Any]],
    phases: List[Dict[str, int]],
) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    events.extend(_player_speed_extremum_events(by_tool, phases))
    events.extend(_player_speed_local_curve_events(by_tool, phases))

    max_result = _query_result(
        by_tool.get("query_telemetry.find_extremum.speed_difference.max")
    )
    min_result = _query_result(
        by_tool.get("query_telemetry.find_extremum.speed_difference.min")
    )
    percent_by_iloc = dict(_speed_gap_percent_values(df, start, end))
    for result, event_name, threshold, source in (
        (
            max_result,
            "expert faster than player",
            -2.0,
            "query_telemetry.find_extremum.speed_difference.max",
        ),
        (
            min_result,
            "player faster than expert",
            2.0,
            "query_telemetry.find_extremum.speed_difference.min",
        ),
    ):
        if not isinstance(result, dict):
            continue
        iloc = result.get("iloc")
        gap_percent = percent_by_iloc.get(iloc) if isinstance(iloc, int) else None
        if (
            event_name.startswith("expert")
            and isinstance(gap_percent, (int, float))
            and gap_percent <= threshold
        ) or (
            event_name.startswith("player")
            and isinstance(gap_percent, (int, float))
            and gap_percent >= threshold
        ):
            events.append(_event(
                event_name,
                _phase_for_iloc(iloc, phases) if isinstance(iloc, int) else "unknown",
                [iloc, iloc] if isinstance(iloc, int) else None,
                {
                    "gap_percent": gap_percent,
                    "gap_relation": _gap_percent_relation("speed", gap_percent),
                    "iloc": iloc,
                },
                "strong" if abs(float(gap_percent)) >= 10.0 else "moderate",
                [source],
            ))
            if abs(float(gap_percent)) >= 10.0:
                events.append(_event(
                    "large speed percentage gap",
                    _phase_for_iloc(iloc, phases) if isinstance(iloc, int) else "unknown",
                    [iloc, iloc] if isinstance(iloc, int) else None,
                    {
                        "gap_percent": gap_percent,
                        "gap_relation": _gap_percent_relation("speed", gap_percent),
                        "iloc": iloc,
                    },
                    "strong",
                    [source],
                ))

    whole_gap = _gap_percent_change(
        _speed_gap_percent_values(df, start, end),
        kind="speed",
    )
    if whole_gap:
        events.append(_event(
            _speed_gap_event_name(whole_gap["direction"], "whole_range"),
            "whole_range",
            [start, end],
            whole_gap,
            (
                "strong"
                if whole_gap.get("threshold_state") == "label_threshold_met"
                else "moderate"
            ),
            ["local_speed_difference_percent_gap"],
        ))

    events.extend(_speed_gap_phase_events(df, start, end, phases))

    player_speed = _query_analysis(
        by_tool.get("query_telemetry.compute_slope.player_speed")
    )
    speed_total = (
        player_speed.get("total_change")
        if isinstance(player_speed, dict)
        else {}
    )
    if isinstance(speed_total, dict):
        domain = speed_total.get("domain_direction")
        direction = speed_total.get("direction")
        if domain in {"rising", "falling", "stable"}:
            events.append(_event(
                f"speed overall trend {domain}",
                "whole_range",
                [start, end],
                {
                    "change": speed_total.get("value"),
                    "direction": direction,
                    "domain_direction": domain,
                    "slope_shape": player_speed.get("slope_shape"),
                },
                (
                    "strong"
                    if speed_total.get("is_label_significant") is True
                    else "moderate"
                ),
                ["query_telemetry.compute_slope.player_speed"],
            ))
        if domain == "rising":
            events.append(_event(
                "acceleration onset",
                "whole_range",
                [start, end],
                {
                    "change": speed_total.get("value"),
                    "slope_shape": player_speed.get("slope_shape"),
                },
                "moderate",
                ["query_telemetry.compute_slope.player_speed"],
            ))
        elif domain == "falling":
            events.append(_event(
                "deceleration onset",
                "whole_range",
                [start, end],
                {
                    "change": speed_total.get("value"),
                    "slope_shape": player_speed.get("slope_shape"),
                },
                "moderate",
                ["query_telemetry.compute_slope.player_speed"],
            ))
    return events


def _speed_gap_phase_events(
    df,
    start: int,
    end: int,
    phases: List[Dict[str, int]],
) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    for phase, range_ in _time_gap_slope_ranges(start, end, phases):
        analysis = _gap_percent_change(
            _speed_gap_percent_values(df, range_[0], range_[1]),
            kind="speed",
        )
        if not analysis:
            continue
        direction = analysis["direction"]
        events.append(_event(
            _speed_gap_event_name(direction, phase),
            phase,
            range_,
            analysis,
            (
                "strong"
                if analysis.get("threshold_state") == "label_threshold_met"
                else "moderate"
            ),
            ["local_speed_difference_percent_gap"],
        ))
    return events


def _time_gap_percent_values(
    df,
    start: int,
    end: int,
) -> List[Tuple[int, float]]:
    if df is None:
        return []
    if "Graphics_current_time" in getattr(df, "columns", []) and (
        "expert_time_difference" in getattr(df, "columns", [])
    ):
        rows = _paired_series_values(
            df,
            start,
            end,
            "Graphics_current_time",
            "expert_time_difference",
        )
        out: List[Tuple[int, float]] = []
        for iloc, player_time, time_gap in rows:
            expert_time = player_time - time_gap
            percent = _percent_gap(player_time, expert_time)
            if percent is not None:
                out.append((iloc, percent))
        return out
    if "Graphics_current_time" in getattr(df, "columns", []) and (
        "expert_optimal_time" in getattr(df, "columns", [])
    ):
        rows = _paired_series_values(
            df,
            start,
            end,
            "Graphics_current_time",
            "expert_optimal_time",
        )
        return [
            (iloc, percent)
            for iloc, player_time, expert_time in rows
            for percent in [_percent_gap(player_time, expert_time)]
            if percent is not None
        ]
    return []


def _speed_gap_percent_values(
    df,
    start: int,
    end: int,
) -> List[Tuple[int, float]]:
    if df is None:
        return []
    columns = getattr(df, "columns", [])
    if (
        "Physics_speed_kmh" in columns
        and "expert_optimal_speed" in columns
    ):
        rows = _paired_series_values(
            df,
            start,
            end,
            "Physics_speed_kmh",
            "expert_optimal_speed",
        )
        return [
            (iloc, percent)
            for iloc, player_speed, expert_speed in rows
            for percent in [_percent_gap(player_speed, expert_speed)]
            if percent is not None
        ]
    if "expert_optimal_speed" in columns and "speed_difference" in columns:
        rows = _paired_series_values(
            df,
            start,
            end,
            "expert_optimal_speed",
            "speed_difference",
        )
        return [
            (iloc, percent)
            for iloc, expert_speed, speed_difference in rows
            for percent in [
                _percent_gap(expert_speed - speed_difference, expert_speed)
            ]
            if percent is not None
        ]
    if "Physics_speed_kmh" in columns and "speed_difference" in columns:
        rows = _paired_series_values(
            df,
            start,
            end,
            "Physics_speed_kmh",
            "speed_difference",
        )
        return [
            (iloc, percent)
            for iloc, player_speed, speed_difference in rows
            for percent in [
                _percent_gap(player_speed, player_speed + speed_difference)
            ]
            if percent is not None
        ]
    return []


def _gap_percent_change(
    values: Sequence[Tuple[int, float]],
    *,
    kind: str,
) -> Optional[Dict[str, Any]]:
    if len(values) < 2:
        return None
    finite = [
        (iloc, value)
        for iloc, value in values
        if _is_number(value)
    ]
    if len(finite) < 2:
        return None

    start_percent = float(finite[0][1])
    end_percent = float(finite[-1][1])
    gap_percent_change = end_percent - start_percent
    relative_gain_percent = (
        -gap_percent_change
        if kind == "time"
        else gap_percent_change
    )
    if abs(relative_gain_percent) < 2.0:
        return None

    iloc_delta = max(float(finite[-1][0] - finite[0][0]), 1.0)
    abs_values = [abs(float(value)) for _, value in finite]
    abs_gap_percent_change = abs_values[-1] - abs_values[0]
    threshold_state = (
        "label_threshold_met"
        if abs(relative_gain_percent) >= 5.0
        else "below_label_threshold"
    )
    return {
        "direction": _gap_percent_direction(kind, gap_percent_change, abs_gap_percent_change),
        "start_gap_percent": start_percent,
        "end_gap_percent": end_percent,
        "gap_percent_change": gap_percent_change,
        "relative_gain_percent": relative_gain_percent,
        "start_gap_relation": _gap_percent_relation(kind, start_percent),
        "end_gap_relation": _gap_percent_relation(kind, end_percent),
        "start_abs_gap_percent": abs_values[0],
        "end_abs_gap_percent": abs_values[-1],
        "abs_gap_percent_change": abs_gap_percent_change,
        "min_abs_gap_percent": min(abs_values),
        "max_abs_gap_percent": max(abs_values),
        "percent_slope": gap_percent_change / iloc_delta,
        "threshold_state": threshold_state,
    }


def _gap_percent_direction(
    kind: str,
    gap_percent_change: float,
    abs_gap_percent_change: float,
) -> str:
    if kind == "time":
        return "rising" if gap_percent_change > 0.0 else "falling"
    return "growing" if abs_gap_percent_change > 0.0 else "closing"


def _gap_percent_relation(kind: str, value: float) -> str:
    if abs(value) < 0.05:
        return "even"
    if kind == "time":
        return "slower" if value > 0.0 else "faster"
    return "faster" if value > 0.0 else "slower"


def _percent_gap(player_value: float, expert_value: float) -> Optional[float]:
    if not _is_number(player_value) or not _is_number(expert_value):
        return None
    denominator = abs(float(expert_value))
    if denominator <= 1e-9:
        return None
    return ((float(player_value) - float(expert_value)) / denominator) * 100.0


def _paired_series_values(
    df,
    start: int,
    end: int,
    left_column: str,
    right_column: str,
) -> List[Tuple[int, float, float]]:
    if df is None:
        return []
    if left_column not in getattr(df, "columns", []) or right_column not in getattr(
        df,
        "columns",
        [],
    ):
        return []
    try:
        segment = df.loc[int(start): int(end), [left_column, right_column]]
    except Exception:  # noqa: BLE001
        return []
    out: List[Tuple[int, float, float]] = []
    for iloc, row in segment.iterrows():
        left = row[left_column]
        right = row[right_column]
        if _is_number(left) and _is_number(right):
            out.append((int(iloc), float(left), float(right)))
    return out


def _player_speed_extremum_events(
    by_tool: Dict[str, Dict[str, Any]],
    phases: List[Dict[str, int]],
) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    for tool_id, event_name in (
        ("query_telemetry.find_extremum.player_speed.max", "player speed maximum"),
        ("query_telemetry.find_extremum.player_speed.min", "player speed minimum"),
    ):
        result = _query_result(by_tool.get(tool_id))
        if not result:
            continue
        value = result.get("value")
        if not isinstance(value, (int, float)):
            continue
        iloc = result.get("iloc")
        extra = result.get("extra") if isinstance(result.get("extra"), dict) else {}
        events.append(_event(
            event_name,
            _phase_for_iloc(iloc, phases) if isinstance(iloc, int) else "unknown",
            [iloc, iloc] if isinstance(iloc, int) else None,
            {"value": value, "unit": extra.get("unit"), "iloc": iloc},
            "strong",
            [tool_id],
        ))
    return events


def _player_speed_local_curve_events(
    by_tool: Dict[str, Dict[str, Any]],
    phases: List[Dict[str, int]],
) -> List[Dict[str, Any]]:
    trend = _query_analysis(
        by_tool.get("query_telemetry.find_trend_runs.player_speed")
    )
    runs = trend.get("runs") if isinstance(trend, dict) else None
    if not isinstance(runs, list):
        return []

    events: List[Dict[str, Any]] = []
    for run in runs:
        if not isinstance(run, dict):
            continue
        direction = run.get("direction")
        if direction not in {"rising", "falling", "flat"}:
            continue
        event_name = {
            "rising": "player accelerating",
            "falling": "player decelerating",
            "flat": "player maintaining steady speed",
        }[direction]
        start_iloc = run.get("start_iloc")
        end_iloc = run.get("end_iloc")
        phase_iloc = (
            int((start_iloc + end_iloc) / 2)
            if isinstance(start_iloc, int) and isinstance(end_iloc, int)
            else None
        )
        events.append(_event(
            event_name,
            (
                _phase_for_iloc(phase_iloc, phases)
                if phase_iloc is not None
                else "unknown"
            ),
            _range_from_values(start_iloc, end_iloc),
            {
                "start_value": run.get("start_value"),
                "end_value": run.get("end_value"),
                "change": run.get("change"),
                "unit": run.get("unit"),
                "slope": run.get("slope"),
                "domain_direction": run.get("domain_direction"),
                "is_label_significant": run.get("is_label_significant"),
            },
            "strong" if run.get("is_label_significant") is True else "moderate",
            ["query_telemetry.find_trend_runs.player_speed"],
        ))
    return events


def _gear_and_rpm_events(
    df,
    start: int,
    end: int,
    phases: List[Dict[str, int]],
) -> List[Dict[str, Any]]:
    return [
        *_shift_timing_events(df, start, end, phases),
        *_exit_gear_mismatch_events(df, start, end, phases),
    ]


def _shift_timing_events(
    df,
    start: int,
    end: int,
    phases: List[Dict[str, int]],
) -> List[Dict[str, Any]]:
    player_shifts = _gear_shifts(df, start, end, "Physics_gear")
    expert_shifts = _gear_shifts(df, start, end, "expert_optimal_gear")
    if not player_shifts or not expert_shifts:
        return []

    events: List[Dict[str, Any]] = []
    used_experts: set[int] = set()
    for player_shift in player_shifts:
        expert_pos = _matching_shift_pos(player_shift, expert_shifts, used_experts)
        if expert_pos is None:
            continue
        used_experts.add(expert_pos)
        expert_shift = expert_shifts[expert_pos]
        delta = int(player_shift["iloc"]) - int(expert_shift["iloc"])
        direction = str(player_shift["direction"])
        if direction == "up":
            event_name = (
                "player upshift earlier than expert"
                if delta < 0
                else "player upshift later than expert"
                if delta > 0
                else "player upshift aligned with expert"
            )
        else:
            event_name = (
                "player downshift earlier than expert"
                if delta < 0
                else "player downshift later than expert"
                if delta > 0
                else "player downshift aligned with expert"
            )

        player_iloc = int(player_shift["iloc"])
        rpm_context = _rpm_context(df, start, end, player_iloc)
        event = _event(
            event_name,
            _phase_for_iloc(player_iloc, phases),
            _range_from_values(player_iloc, expert_shift.get("iloc")),
            {
                "player_shift_iloc": player_iloc,
                "expert_shift_iloc": expert_shift.get("iloc"),
                "shift_delta_iloc": delta,
                "shift_offset_iloc": abs(delta),
                "player_from_gear": player_shift.get("from_gear"),
                "player_to_gear": player_shift.get("to_gear"),
                "expert_from_gear": expert_shift.get("from_gear"),
                "expert_to_gear": expert_shift.get("to_gear"),
                "player_rpm_at_shift": rpm_context.get("rpm"),
            },
            _timing_confidence(delta),
            ["local_gear_shift_rpm_statistics"],
        )
        if delta != 0:
            event["semantic_search_terms"] = _gear_shift_search_terms(event_name)
        events.append(event)
    return events


def _exit_gear_mismatch_events(
    df,
    start: int,
    end: int,
    phases: List[Dict[str, int]],
) -> List[Dict[str, Any]]:
    player_gear = {
        iloc: int(round(value))
        for iloc, value in _series_values(df, start, end, "Physics_gear")
    }
    expert_gear = {
        iloc: int(round(value))
        for iloc, value in _series_values(df, start, end, "expert_optimal_gear")
    }
    if not player_gear or not expert_gear or not phases:
        return []

    events: List[Dict[str, Any]] = []
    for phase in phases:
        exit_iloc = phase.get("exit")
        if not isinstance(exit_iloc, int):
            continue
        if exit_iloc not in player_gear or exit_iloc not in expert_gear:
            continue
        player = player_gear[exit_iloc]
        expert = expert_gear[exit_iloc]
        diff = player - expert
        if diff == 0:
            continue
        event_name = (
            "player gear low at exit"
            if player < expert
            else "player gear high at exit"
        )
        rpm_context = _rpm_context(df, start, end, exit_iloc)
        event = _event(
            event_name,
            "exit",
            [exit_iloc, exit_iloc],
            {
                "exit_iloc": exit_iloc,
                "player_gear": player,
                "expert_gear": expert,
                "gear_delta": diff,
                "player_rpm_at_exit": rpm_context.get("rpm"),
            },
            "strong",
            ["local_exit_gear_rpm_statistics"],
        )
        event["semantic_search_terms"] = _exit_gear_search_terms(event_name)
        events.append(event)
    return events


def _gear_shifts(
    df,
    start: int,
    end: int,
    column: str,
) -> List[Dict[str, Any]]:
    values = [
        (iloc, int(round(value)))
        for iloc, value in _series_values(df, start, end, column)
    ]
    shifts: List[Dict[str, Any]] = []
    previous: Optional[Tuple[int, int]] = None
    for iloc, gear in values:
        if previous is None:
            previous = (iloc, gear)
            continue
        previous_iloc, previous_gear = previous
        if gear != previous_gear:
            shifts.append({
                "iloc": iloc,
                "previous_iloc": previous_iloc,
                "from_gear": previous_gear,
                "to_gear": gear,
                "direction": "up" if gear > previous_gear else "down",
            })
        previous = (iloc, gear)
    return shifts


def _matching_shift_pos(
    player_shift: Dict[str, Any],
    expert_shifts: Sequence[Dict[str, Any]],
    used_experts: set[int],
) -> Optional[int]:
    direction = player_shift.get("direction")
    exact_candidates: List[Tuple[int, int]] = []
    fallback_candidates: List[Tuple[int, int]] = []
    player_iloc = int(player_shift.get("iloc", 0))
    for pos, expert_shift in enumerate(expert_shifts):
        if pos in used_experts or expert_shift.get("direction") != direction:
            continue
        distance = abs(player_iloc - int(expert_shift.get("iloc", player_iloc)))
        fallback_candidates.append((distance, pos))
        if (
            expert_shift.get("from_gear") == player_shift.get("from_gear")
            and expert_shift.get("to_gear") == player_shift.get("to_gear")
        ):
            exact_candidates.append((distance, pos))
    candidates = exact_candidates or fallback_candidates
    if not candidates:
        return None
    return sorted(candidates)[0][1]


def _rpm_context(df, start: int, end: int, iloc: int) -> Dict[str, Any]:
    rpm = _value_at_iloc(df, start, end, "Physics_rpm", iloc)
    return {"rpm": rpm}


def _gear_shift_search_terms(event_name: str) -> List[str]:
    return {
        "player upshift earlier than expert": [
            "upshift before expert",
            "player upshift earlier than expert",
        ],
        "player upshift later than expert": [
            "upshift after expert",
            "player upshift later than expert",
        ],
        "player downshift earlier than expert": [
            "downshift before expert",
            "player downshift earlier than expert",
        ],
        "player downshift later than expert": [
            "downshift after expert",
            "player downshift later than expert",
        ],
    }.get(event_name, [])


def _exit_gear_search_terms(event_name: str) -> List[str]:
    return {
        "player gear low at exit": [
            "gear too low when accelerating",
            "player gear low at exit",
            "player gear lower than expert gear at corner exit",
        ],
        "player gear high at exit": [
            "gear too high when accelerating",
            "player gear high at exit",
            "player gear higher than expert gear at corner exit",
        ],
    }.get(event_name, [])


def _balance_and_grip_events(
    df,
    start: int,
    end: int,
    by_tool: Dict[str, Dict[str, Any]],
    phases: List[Dict[str, int]],
) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    events.extend(_slip_balance_run_events(df, start, end, phases))

    push = _query_result(by_tool.get("query_telemetry.find_extremum.push_limit.max"))
    if push:
        value = push.get("value")
        iloc = push.get("iloc")
        if isinstance(value, (int, float)) and value >= 1.0:
            events.append(_event(
                "over-limit spike",
                _phase_for_iloc(iloc, phases) if isinstance(iloc, int) else "unknown",
                [iloc, iloc] if isinstance(iloc, int) else None,
                {"value": value, "iloc": iloc},
                "strong",
                ["query_telemetry.find_extremum.push_limit.max"],
            ))
        elif isinstance(value, (int, float)) and value <= 0.5:
            events.append(_event(
                "sustained low grip utilisation",
                "whole_range",
                None,
                {"max_value": value},
                "moderate",
                ["query_telemetry.find_extremum.push_limit.max"],
            ))
    return events


def _slip_balance_run_events(
    df,
    start: int,
    end: int,
    phases: List[Dict[str, int]],
) -> List[Dict[str, Any]]:
    values = _series_values(df, start, end, "slip_balance", graph_id="trajectory_balance")
    if not values:
        return []

    events: List[Dict[str, Any]] = []
    for event_name, predicate, peak_selector in (
        (
            "oversteer",
            lambda value: value >= 0.02,
            lambda run: max(run, key=lambda item: item[1]),
        ),
        (
            "understeer",
            lambda value: value <= -0.02,
            lambda run: min(run, key=lambda item: item[1]),
        ),
    ):
        for run in _value_runs(values, predicate):
            start_iloc = run[0][0]
            end_iloc = run[-1][0]
            peak_iloc, peak_value = peak_selector(run)
            phase_iloc = int((start_iloc + end_iloc) / 2)
            events.append(_event(
                event_name,
                _phase_for_iloc(phase_iloc, phases),
                [start_iloc, end_iloc],
                {
                    "start_value": run[0][1],
                    "end_value": run[-1][1],
                    "peak_value": peak_value,
                    "peak_iloc": peak_iloc,
                    "threshold": 0.02 if event_name == "oversteer" else -0.02,
                },
                "strong" if abs(float(peak_value)) >= 0.05 else "moderate",
                ["derived.slip_balance_threshold_runs"],
            ))
    return events


def _value_runs(
    values: Sequence[Tuple[int, float]],
    predicate: Callable[[float], bool],
) -> List[List[Tuple[int, float]]]:
    runs: List[List[Tuple[int, float]]] = []
    current: List[Tuple[int, float]] = []
    for iloc, value in values:
        if predicate(value):
            current.append((iloc, value))
        elif current:
            runs.append(current)
            current = []
    if current:
        runs.append(current)
    return runs


def _overlap_events(df, start: int, end: int) -> List[Dict[str, Any]]:
    brake = _series_values(df, start, end, "Physics_brake")
    gas = _series_values(df, start, end, "Physics_gas")
    if not brake or not gas:
        return []
    by_iloc = {iloc: value for iloc, value in gas}
    overlap = [
        iloc
        for iloc, brake_value in brake
        if isinstance(brake_value, (int, float))
        and brake_value > 0.05
        and isinstance(by_iloc.get(iloc), (int, float))
        and by_iloc[iloc] > 0.05
    ]
    if len(overlap) < max(3, int(0.15 * max(1, end - start))):
        return []
    return [_event(
        "brake and throttle overlap",
        "whole_range",
        [min(overlap), max(overlap)],
        {"overlap_iloc_count": len(overlap)},
        "strong",
        ["local_brake_throttle_overlap_statistics"],
    )]


def _action_profile(
    df,
    start: int,
    end: int,
    column: str,
    direction: str,
    *,
    smoothing_window: int = 3,
) -> Optional[Dict[str, Any]]:
    values = _series_values(df, start, end, column)
    if len(values) < 4:
        return None
    ilocs = [iloc for iloc, _ in values]
    arr = _rolling_median([value for _, value in values], smoothing_window)
    finite = [(iloc, value) for iloc, value in zip(ilocs, arr) if _is_number(value)]
    episodes = _change_episodes(finite, direction)
    if not episodes:
        return None
    return max(
        episodes,
        key=lambda episode: (
            float(episode.get("total_movement") or 0.0),
            int(episode.get("duration") or 0),
        ),
    )


def _change_episodes(
    values: Sequence[Tuple[int, float]],
    direction: str,
) -> List[Dict[str, Any]]:
    if len(values) < 4:
        return []
    sign = 1.0 if direction == "increase" else -1.0
    diffs: List[Dict[str, float]] = []
    for pos in range(1, len(values)):
        prev_iloc, prev_value = values[pos - 1]
        iloc, value = values[pos]
        iloc_delta = max(1, iloc - prev_iloc)
        raw_slope = (float(value) - float(prev_value)) / iloc_delta
        diffs.append({
            "pos": float(pos),
            "raw_slope": raw_slope,
            "signed_slope": raw_slope * sign,
        })
    if not diffs:
        return []

    abs_diffs = [abs(diff["raw_slope"]) for diff in diffs]
    noise_floor = float(_median(abs_diffs) or 0.0)
    local_values = [float(value) for _, value in values]
    local_range = max(local_values) - min(local_values)
    total_iloc_span = max(1, values[-1][0] - values[0][0])
    average_range_step = local_range / total_iloc_span
    slope_gate = max(noise_floor * 0.5, average_range_step * 0.25)
    movement_gate = max(noise_floor * 4.0, local_range * 0.2)
    if slope_gate <= 0.0 or movement_gate <= 0.0:
        return []

    episodes: List[Dict[str, Any]] = []
    start_pos: Optional[int] = None
    first_active_pos: Optional[int] = None
    last_active_pos: Optional[int] = None
    last_pos: Optional[int] = None
    active_count = 0
    stall_count = 0
    max_stall = 1

    def finish_current() -> None:
        nonlocal start_pos, first_active_pos, last_active_pos, last_pos
        nonlocal active_count, stall_count
        if (
            start_pos is None
            or first_active_pos is None
            or last_active_pos is None
            or last_pos is None
        ):
            start_pos = first_active_pos = last_active_pos = last_pos = None
            active_count = 0
            stall_count = 0
            return
        if last_pos <= start_pos:
            start_pos = first_active_pos = last_active_pos = last_pos = None
            active_count = 0
            stall_count = 0
            return

        start_iloc, start_value = values[start_pos]
        end_iloc, end_value = values[last_pos]
        duration = end_iloc - start_iloc
        total_movement = (float(end_value) - float(start_value)) * sign
        if duration <= 0 or total_movement < movement_gate:
            start_pos = first_active_pos = last_active_pos = last_pos = None
            active_count = 0
            stall_count = 0
            return
        raw_slopes: List[float] = []
        normalized_slopes: List[float] = []
        for pos in range(start_pos + 1, last_pos + 1):
            prev_iloc, prev_value = values[pos - 1]
            iloc, value = values[pos]
            iloc_delta = max(1, iloc - prev_iloc)
            raw_slope = (float(value) - float(prev_value)) / iloc_delta
            signed_slope = raw_slope * sign
            if signed_slope > 0.0:
                raw_slopes.append(raw_slope)
                normalized_slopes.append(signed_slope / total_movement)

        episodes.append({
            "start_index": start_iloc,
            "end_index": end_iloc,
            "start_band": [values[start_pos][0], values[first_active_pos][0]],
            "end_band": [values[last_active_pos][0], values[last_pos][0]],
            "duration": duration,
            "total_movement": total_movement,
            "noise_floor": noise_floor,
            "direction": direction,
            "median_raw_slope": _median(raw_slopes),
            "median_normalized_slope": _median(normalized_slopes),
            "active_step_count": active_count,
            "movement_gate": movement_gate,
            "slope_gate": slope_gate,
        })
        start_pos = first_active_pos = last_active_pos = last_pos = None
        active_count = 0
        stall_count = 0

    for diff in diffs:
        pos = int(diff["pos"])
        signed_slope = diff["signed_slope"]
        if signed_slope > slope_gate:
            if start_pos is None:
                start_pos = pos - 1
                first_active_pos = pos
            active_count += 1
            last_active_pos = pos
            last_pos = pos
            stall_count = 0
        elif start_pos is not None and abs(signed_slope) <= slope_gate and stall_count < max_stall:
            last_pos = pos
            stall_count += 1
        else:
            finish_current()
    finish_current()
    return episodes


def _compare_action_timing(
    player: Dict[str, Any],
    expert: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    player_start = player.get("start_index")
    expert_start = expert.get("start_index")
    if not isinstance(player_start, int) or not isinstance(expert_start, int):
        return None
    delta = player_start - expert_start
    if delta == 0:
        return None
    player_band = player.get("start_band")
    expert_band = expert.get("start_band")
    if _bands_overlap(player_band, expert_band):
        return None
    uncertainty = max(_band_width(player_band), _band_width(expert_band))
    return {
        "start_delta_iloc": delta,
        "boundary_uncertainty_iloc": uncertainty,
        "confidence": "strong" if abs(delta) > max(1, uncertainty) else "moderate",
    }


def _bands_overlap(a: Any, b: Any) -> bool:
    if not (
        isinstance(a, list)
        and isinstance(b, list)
        and len(a) == 2
        and len(b) == 2
        and all(isinstance(v, int) for v in [*a, *b])
    ):
        return False
    return max(a[0], b[0]) <= min(a[1], b[1])


def _band_width(value: Any) -> int:
    if (
        isinstance(value, list)
        and len(value) == 2
        and isinstance(value[0], int)
        and isinstance(value[1], int)
    ):
        return max(0, value[1] - value[0])
    return 0


def _series_values(
    df,
    start: int,
    end: int,
    column: str,
    *,
    graph_id: Optional[str] = None,
) -> List[Tuple[int, float]]:
    if df is None:
        return []
    table = None
    if hasattr(df, "columns") and column in getattr(df, "columns", []):
        table = df
    elif graph_id:
        try:
            from app.shared.annotation_agent_tools import build_graph

            table = build_graph(graph_id, df)
            table = _preflight_query_table(table, start, end)
        except Exception:  # noqa: BLE001
            table = None
    if table is None or column not in getattr(table, "columns", []):
        return []
    try:
        segment = table.loc[int(start): int(end)]
    except Exception:  # noqa: BLE001
        return []
    out: List[Tuple[int, float]] = []
    for iloc, value in segment[column].items():
        if _is_number(value):
            out.append((int(iloc), float(value)))
    return out


def _value_at_iloc(
    df,
    start: int,
    end: int,
    column: str,
    iloc: int,
    *,
    graph_id: Optional[str] = None,
) -> Optional[float]:
    for row_iloc, value in _series_values(
        df,
        start,
        end,
        column,
        graph_id=graph_id,
    ):
        if row_iloc == iloc:
            return value
    return None


def _rolling_median(values: List[float], window: int) -> List[float]:
    if window <= 1:
        return values
    half = window // 2
    out: List[float] = []
    for i in range(len(values)):
        lo = max(0, i - half)
        hi = min(len(values), i + half + 1)
        sample = sorted(v for v in values[lo:hi] if _is_number(v))
        out.append(_median(sample) if sample else float("nan"))
    return out


def _slope_shape_window(length: int) -> int:
    if length <= 1:
        return 1
    return max(1, min(5, (length + 3) // 4))


def _slope_shape_smoothing_window(length: int) -> int:
    window = _slope_shape_window(length)
    if window % 2 == 0:
        window = max(1, window - 1)
    return window


def _phase_windows(by_tool: Dict[str, Dict[str, Any]]) -> List[Dict[str, int]]:
    content = by_tool.get("compute_expert_phases") or {}
    phases = content.get("phases")
    if not isinstance(phases, list):
        shape = by_tool.get("measure_segment_shape") or {}
        phases = shape.get("phases")
    out: List[Dict[str, int]] = []
    if not isinstance(phases, list):
        return out
    for phase in phases:
        if not isinstance(phase, dict):
            continue
        row: Dict[str, int] = {}
        for key in ("entry", "apex", "exit"):
            try:
                row[key] = int(phase[key])
            except (KeyError, TypeError, ValueError):
                pass
        if row:
            if phase.get("direction") is not None:
                row["direction"] = phase.get("direction")  # type: ignore[assignment]
            out.append(row)
    return out


def _phase_for_iloc(iloc: Any, phases: List[Dict[str, int]]) -> str:
    if not isinstance(iloc, int):
        return "unknown"
    for phase in phases:
        entry = phase.get("entry")
        apex = phase.get("apex")
        exit_ = phase.get("exit")
        if not all(isinstance(v, int) for v in (entry, apex, exit_)):
            continue
        if entry <= iloc <= apex:
            return "entry" if iloc < apex else "apex"
        if apex < iloc <= exit_:
            return "exit"
    return "unknown"


def _time_gap_event_name(direction: str, phase: str) -> str:
    base = f"time gap {direction}"
    if phase in {"entry", "apex", "exit"}:
        return f"{base} at {phase}"
    return base


def _speed_gap_event_name(direction: str, phase: str) -> str:
    base = f"speed gap {direction}"
    if phase in {"entry", "apex", "exit"}:
        return f"{base} at {phase}"
    return base


def _query_result(content: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(content, dict):
        return {}
    result = content.get("result")
    return result if isinstance(result, dict) else {}


def _query_analysis(content: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(content, dict):
        return {}
    analysis = content.get("analysis")
    return analysis if isinstance(analysis, dict) else {}


def _event(
    event: str,
    phase: str,
    range_: Optional[List[int]],
    measurements: Dict[str, Any],
    confidence: str,
    sources: List[str],
) -> Dict[str, Any]:
    return {
        "event": event,
        "phase": phase if phase in {"entry", "apex", "exit", "straight", "whole_range", "unknown"} else "unknown",
        "range": range_,
        "measurements": {
            key: value
            for key, value in measurements.items()
            if value is not None
        },
        "confidence": confidence,
        "sources": sources,
    }


def _event_text(
    events: Sequence[Dict[str, Any]],
    parent_main_labels: Sequence[str],
    extra_query_terms: Sequence[str],
) -> str:
    return _sentence_evidence_text(events, parent_main_labels, extra_query_terms)


def _semantic_search_text(
    events: Sequence[Dict[str, Any]],
    parent_main_labels: Sequence[str],
    extra_query_terms: Sequence[str],
) -> str:
    """Embedding query text: sentence-only evidence."""
    return _sentence_evidence_text(events, parent_main_labels, extra_query_terms)


def _sentence_evidence_text(
    events: Sequence[Dict[str, Any]],
    parent_main_labels: Sequence[str],
    extra_query_terms: Sequence[str],
) -> str:
    lines: List[str] = []
    context_terms = _dedupe(
        str(term).strip()
        for term in [*parent_main_labels, *extra_query_terms]
        if str(term).strip()
    )
    if context_terms:
        lines.append(
            "Parent context for detailed label search includes "
            + ", ".join(context_terms)
            + "."
        )
    for event in events:
        sentence = _event_sentence(event)
        if sentence:
            lines.append(sentence)
    return "\n".join(line for line in lines if line.strip())[:12000]


def _event_sentence(event: Dict[str, Any]) -> str:
    event_name = str(event.get("event") or "").strip()
    if not event_name:
        return ""

    phase = _phase_sentence_prefix(str(event.get("phase") or ""))
    if _gap_event_has_phase(event_name):
        phase = ""
    range_text = _range_sentence_fragment(event.get("range"))
    measurements = event.get("measurements")
    if not isinstance(measurements, dict):
        measurements = {}

    fragments = _measurement_sentence_fragments(event_name, measurements)
    confidence = str(event.get("confidence") or "").strip()
    confidence_text = f"with {confidence} confidence" if confidence else ""

    if _is_opponent_relative_fact(event_name):
        subject = event_name
        if subject.startswith(("opponent", "driver", "gap")):
            subject = "the " + subject
        parts = [part for part in [*fragments, confidence_text] if part]
        detail = ", " + "; ".join(parts) if parts else ""
        return f"{phase}{subject}{detail}."

    shape_terms = _event_label_vocabulary_terms(event_name)
    if shape_terms:
        primary, *vocabulary_terms = shape_terms
        subject = _event_label_subject(phase, primary)
        parts = [part for part in [range_text, *fragments] if part]
        sentence = subject
        if parts:
            sentence += ", " + "; ".join(parts)
        if vocabulary_terms:
            sentence += (
                "; this matches label vocabulary for "
                + _join_sentence_list(vocabulary_terms)
            )
        if confidence_text:
            sentence += f", {confidence_text}"
        return sentence + "."

    parts = [part for part in [range_text, *fragments, confidence_text] if part]
    detail = ", " + "; ".join(parts) if parts else ""
    return f"{phase}the evidence shows {event_name}{detail}."


def _is_opponent_relative_fact(event_name: str) -> bool:
    return (
        event_name.startswith("opponent ")
        or event_name.startswith("driver ")
        or event_name.startswith("gap ")
    ) and (
        "opponent" in event_name
        or "driver" in event_name
    )


def _event_label_vocabulary_terms(event_name: str) -> List[str]:
    terms = [term.strip() for term in event_name.split(";") if term.strip()]
    return terms if len(terms) > 1 else []


def _event_label_subject(phase_prefix: str, primary_term: str) -> str:
    if primary_term.startswith(("in ", "on ")):
        return f"{phase_prefix}the segment is {primary_term}"
    return f"{phase_prefix}the segment is classified as {primary_term}"


def _join_sentence_list(items: Sequence[str]) -> str:
    clean = [item for item in items if item]
    if len(clean) <= 1:
        return "".join(clean)
    return ", ".join(clean[:-1]) + f", and {clean[-1]}"


def _phase_sentence_prefix(phase: str) -> str:
    if phase in {"entry", "apex", "exit", "straight"}:
        return f"During {phase}, "
    if phase == "whole_range":
        return "Across the whole range, "
    return ""


def _gap_event_has_phase(event_name: str) -> bool:
    return event_name.startswith(("time gap ", "speed gap ")) and (
        event_name.endswith(" at entry")
        or event_name.endswith(" at apex")
        or event_name.endswith(" at exit")
    )


def _range_sentence_fragment(range_: Any) -> str:
    if (
        isinstance(range_, list)
        and len(range_) == 2
        and isinstance(range_[0], int)
        and isinstance(range_[1], int)
    ):
        if range_[0] == range_[1]:
            return f"detected at iloc {range_[0]}"
        return f"detected from iloc {range_[0]} to {range_[1]}"
    return ""


def _measurement_sentence_fragments(
    event_name: str,
    measurements: Dict[str, Any],
) -> List[str]:
    fragments: List[str] = []

    if _is_opponent_relative_fact(event_name):
        index = measurements.get("index")
        start_index = measurements.get("start_index")
        end_index = measurements.get("end_index")
        signed_gap = measurements.get("signed_gap_m")
        start_gap = measurements.get("start_gap_m")
        end_gap = measurements.get("end_gap_m")
        lateral = measurements.get("lateral_offset_m")
        distance = measurements.get("distance_m")
        slot = measurements.get("slot")
        if slot is not None:
            fragments.append(f"against opponent slot {slot}")
        if index is not None:
            fragments.append(f"at index {index}")
        elif start_index is not None and end_index is not None:
            fragments.append(f"from index {start_index} to {end_index}")
        if signed_gap is not None:
            fragments.append(
                f"signed ahead/behind gap was {_format_value(signed_gap)} m"
            )
        if event_name == "gap to the opponent shrank" and start_gap is not None and end_gap is not None:
            fragments.append(
                "gap to the opponent shrank from "
                f"{_format_value(start_gap)} m to {_format_value(end_gap)} m"
            )
        elif event_name.startswith("gap flipped") and start_gap is not None and end_gap is not None:
            fragments.append(
                "signed ahead/behind gap changed from "
                f"{_format_value(start_gap)} m to {_format_value(end_gap)} m"
            )
        if lateral is not None:
            fragments.append(f"side offset was {_format_value(lateral)} m")
        if distance is not None:
            fragments.append(f"car-to-car distance was {_format_value(distance)} m")
        start_speed = measurements.get("start_relative_speed")
        end_speed = measurements.get("end_relative_speed")
        speed_units = measurements.get("speed_units")
        if start_speed is not None and end_speed is not None:
            unit_text = f" {speed_units}" if speed_units else ""
            fragments.append(
                "driver-minus-opponent speed changed from "
                f"{_format_value(start_speed)} to {_format_value(end_speed)} "
                f"{unit_text}"
            )
        accel = measurements.get("median_acceleration_advantage")
        accel_units = measurements.get("acceleration_units")
        if accel is not None:
            unit_text = f" {accel_units}" if accel_units else ""
            fragments.append(
                "median acceleration advantage was "
                f"{_format_value(accel)}{unit_text}"
            )
        decel = measurements.get("median_deceleration_difference")
        if decel is not None:
            unit_text = f" {accel_units}" if accel_units else ""
            fragments.append(
                "median deceleration difference was "
                f"{_format_value(decel)}{unit_text}"
            )
        return fragments

    boundary = None
    if (
        "onset earlier than expert" in event_name
        or "onset later than expert" in event_name
        or "onset aligned with expert" in event_name
    ):
        boundary = "onset"
        player = measurements.get("player_start_index")
        expert = measurements.get("expert_start_index")
        delta = measurements.get("start_delta_iloc")
    elif (
        "end earlier than expert" in event_name
        or "end later than expert" in event_name
        or "end aligned with expert" in event_name
    ):
        boundary = "end"
        player = measurements.get("player_end_index")
        expert = measurements.get("expert_end_index")
        delta = measurements.get("end_delta_iloc")
    if boundary:
        if player is not None and expert is not None:
            fragments.append(
                f"the player {boundary} was at iloc {player} while the "
                f"expert {boundary} was at iloc {expert}"
            )
        if delta is not None:
            delta_number = _as_float(delta)
            if delta_number is not None:
                if delta_number == 0:
                    fragments.append(
                        f"the player {boundary} timing was aligned with expert"
                    )
                else:
                    direction = "later" if delta_number > 0 else "earlier"
                    fragments.append(
                        f"the player {boundary} timing was "
                        f"{_format_value(abs(delta_number))} ilocs {direction}"
                    )
        return fragments

    if event_name.startswith("player upshift") or event_name.startswith(
        "player downshift"
    ):
        player = measurements.get("player_shift_iloc")
        expert = measurements.get("expert_shift_iloc")
        delta = measurements.get("shift_delta_iloc")
        if player is not None and expert is not None:
            fragments.append(
                f"the player shifted at iloc {player} while the expert "
                f"shifted at iloc {expert}"
            )
        player_from = measurements.get("player_from_gear")
        player_to = measurements.get("player_to_gear")
        expert_from = measurements.get("expert_from_gear")
        expert_to = measurements.get("expert_to_gear")
        if player_from is not None and player_to is not None:
            fragments.append(f"the player gear change was {player_from}->{player_to}")
        if expert_from is not None and expert_to is not None:
            fragments.append(f"the expert gear change was {expert_from}->{expert_to}")
        delta_number = _as_float(delta)
        if delta_number is not None:
            if delta_number == 0:
                fragments.append("the shift timing was aligned with expert")
            else:
                direction = "later" if delta_number > 0 else "earlier"
                fragments.append(
                    "the player shift timing was "
                    f"{_format_value(abs(delta_number))} ilocs {direction} "
                    "than expert"
                )
        rpm = measurements.get("player_rpm_at_shift")
        if rpm is not None:
            fragments.append(f"player RPM at shift was {_format_value(rpm)}")
        return fragments

    if "player reaches apex" in event_name:
        player = measurements.get("player_apex_iloc")
        expert = measurements.get("expert_apex_iloc")
        delta = measurements.get("apex_delta_iloc")
        player_range = measurements.get("player_apex_range")
        expert_range = measurements.get("expert_apex_range")
        boundary_gap = measurements.get("apex_boundary_gap_iloc")
        if player is not None and expert is not None:
            fragments.append(
                f"the player apex was at iloc {player} while the expert apex "
                f"was at iloc {expert}"
            )
        if (
            isinstance(player_range, list)
            and isinstance(expert_range, list)
            and len(player_range) == 2
            and len(expert_range) == 2
        ):
            fragments.append(
                "the player apex range was iloc "
                f"{player_range[0]} to {player_range[1]} while the expert "
                f"apex range was iloc {expert_range[0]} to {expert_range[1]}"
            )
        delta_number = _as_float(delta)
        if delta_number is not None:
            if delta_number == 0:
                fragments.append("the player apex timing was aligned with expert")
            else:
                direction = "later" if delta_number > 0 else "earlier"
                fragments.append(
                    "the player apex timing was "
                    f"{_format_value(abs(delta_number))} ilocs {direction}"
                )
        boundary_gap_number = _as_float(boundary_gap)
        if boundary_gap_number is not None:
            fragments.append(
                "the apex ranges were separated by "
                f"{_format_value(boundary_gap_number)} ilocs"
            )
        player_x = measurements.get("player_apex_x")
        player_y = measurements.get("player_apex_y")
        expert_x = measurements.get("expert_apex_x")
        expert_y = measurements.get("expert_apex_y")
        if all(
            value is not None
            for value in (player_x, player_y, expert_x, expert_y)
        ):
            fragments.append(
                "player apex position was "
                f"({_format_value(player_x)}, {_format_value(player_y)}) and "
                "expert apex position was "
                f"({_format_value(expert_x)}, {_format_value(expert_y)})"
            )
        return fragments

    if (
        "onset comparison unavailable" in event_name
        or "end comparison unavailable" in event_name
    ):
        player_detected = measurements.get("player_action_detected")
        expert_detected = measurements.get("expert_action_detected")
        if player_detected is False and expert_detected is False:
            fragments.append("neither player nor expert had a clear input-change episode")
        elif player_detected is False:
            fragments.append("the player did not have a clear input-change episode")
        elif expert_detected is False:
            fragments.append("the expert did not have a clear input-change episode")
        direction = measurements.get("direction")
        if direction:
            direction_text = {
                "increase": "rising",
                "decrease": "falling",
            }.get(str(direction), _humanize_token(str(direction)))
            fragments.append(f"searched for a {direction_text} episode")
        return fragments

    if "too quickly" in event_name or "too slowly" in event_name:
        player_duration = measurements.get("player_duration")
        expert_duration = measurements.get("expert_duration")
        if player_duration is not None and expert_duration is not None:
            fragments.append(
                "the player action lasted "
                f"{player_duration} ilocs versus {expert_duration} ilocs "
                "for the expert"
            )
        ratio = measurements.get("slope_ratio")
        if ratio is not None:
            fragments.append(f"the input change-rate ratio was {_format_value(ratio)}")
        return fragments

    if "peak brake pressure" in event_name or "lowest throttle pressure" in event_name:
        player = measurements.get("player_value")
        expert = measurements.get("expert_value")
        if player is not None and expert is not None:
            if "lowest throttle pressure" in event_name:
                fragments.append(
                    f"the player lowest was {_format_value(player)} versus "
                    f"expert lowest {_format_value(expert)}"
                )
            else:
                fragments.append(
                    f"the player peak was {_format_value(player)} versus "
                    f"expert peak {_format_value(expert)}"
                )
        if "peak brake pressure" in event_name:
            speed_gap = _as_float(measurements.get("speed_gap_percent_at_player_peak"))
            relation = measurements.get("speed_gap_relation_at_player_peak")
            if speed_gap is not None and relation is not None:
                fragments.append(
                    "the player was "
                    f"{_gap_percent_relation_text(speed_gap, relation)} "
                    "at the player peak"
                )
        iloc = measurements.get("player_iloc")
        if iloc is not None:
            if "lowest throttle pressure" in event_name:
                fragments.append(f"the player lowest occurred at iloc {iloc}")
            else:
                fragments.append(f"the player peak occurred at iloc {iloc}")
        return fragments

    if event_name in {
        "expert faster than player",
        "player faster than expert",
        "large speed percentage gap",
    }:
        gap_percent = measurements.get("gap_percent")
        relation = measurements.get("gap_relation")
        if gap_percent is not None and relation is not None:
            fragments.append(
                "the player was "
                + _gap_percent_relation_text(gap_percent, relation)
            )
        iloc = measurements.get("iloc")
        if iloc is not None:
            fragments.append(f"detected at iloc {iloc}")
        return fragments

    if "trajectory" in event_name or "moving toward" in event_name or "expert line" in event_name:
        value = measurements.get("value")
        median = measurements.get("median_offset")
        change = measurements.get("change")
        if value is not None:
            fragments.append(f"the trajectory offset was {_format_value(value)} m")
        if median is not None:
            fragments.append(f"the median trajectory offset was {_format_value(median)} m")
        if change is not None:
            fragments.append(f"the trajectory offset changed by {_format_value(change)} m")
        if measurements.get("moves_toward_expert_line") is True:
            start = measurements.get("start")
            end = measurements.get("end")
            if start is not None and end is not None:
                fragments.append(
                    "the absolute offset moved from "
                    f"{_format_value(start)} m to {_format_value(end)} m "
                    "toward the expert line"
                )
            else:
                fragments.append("the offset moved toward the expert line")
        return fragments

    if event_name in {
        "speed gap closing",
        "speed gap growing",
    } or event_name.startswith(("speed gap closing at", "speed gap growing at")):
        fragments.extend(_gap_percent_sentence_fragments(measurements, "speed"))
        threshold = measurements.get("threshold_state")
        if threshold:
            fragments.append(_humanize_token(str(threshold)))
        return fragments

    if event_name in {
        "gap grows",
        "gap shrinks",
        "time loss",
    } or event_name.startswith(("time gap rising", "time gap falling")):
        fragments.extend(_gap_percent_sentence_fragments(measurements, "time"))
        threshold = measurements.get("threshold_state")
        if threshold:
            fragments.append(_humanize_token(str(threshold)))
        slope_shape = measurements.get("slope_shape")
        if slope_shape:
            fragments.append(_humanize_token(str(slope_shape)))
        return fragments

    if event_name in {
        "player gear low at exit",
        "player gear high at exit",
    }:
        player_gear = measurements.get("player_gear")
        expert_gear = measurements.get("expert_gear")
        if player_gear is not None and expert_gear is not None:
            fragments.append(
                f"the player was in gear {player_gear} while the expert was "
                f"in gear {expert_gear} at corner exit"
            )
        rpm = measurements.get("player_rpm_at_exit")
        if rpm is not None:
            fragments.append(
                f"player RPM at exit was {_format_value(rpm)}"
            )
        return fragments

    if "speed" in event_name or "acceleration" in event_name or "deceleration" in event_name:
        value = measurements.get("value")
        unit = measurements.get("unit") or "km/h"
        change = measurements.get("change")
        if value is not None:
            fragments.append(f"the speed value was {_format_value(value)} {unit}")
        if change is not None:
            fragments.append(f"the speed changed by {_format_value(change)} {unit}")
        return fragments

    if event_name in {"oversteer", "understeer"}:
        peak = measurements.get("peak_value")
        peak_iloc = measurements.get("peak_iloc")
        if peak is not None:
            text = f"the peak slip-balance value was {_format_value(peak)}"
            if peak_iloc is not None:
                text += f" at iloc {peak_iloc}"
            fragments.append(text)
        return fragments

    if event_name == "corner phase markers":
        entry = measurements.get("entry_start_iloc")
        apex = measurements.get("apex_iloc")
        exit_ = measurements.get("exit_end_iloc")
        if entry is not None:
            fragments.append(f"entry starts at iloc {entry}")
        if apex is not None:
            fragments.append(f"apex is at iloc {apex}")
        if exit_ is not None:
            fragments.append(f"exit ends at iloc {exit_}")
        direction = measurements.get("direction")
        if direction:
            fragments.append(f"corner direction was {direction}")
        return fragments

    if "altitude" in event_name:
        angle = measurements.get("slope_angle_degrees")
        if angle is not None:
            fragments.append(
                f"slope angle was {_format_value(angle)} degrees"
            )
        distance = measurements.get("horizontal_distance_units")
        if distance is not None:
            fragments.append(
                f"horizontal path distance was {_format_value(distance)} telemetry units"
            )
        return fragments

    for key in (
        "outcome",
        "confidence_level",
        "primary_slot_for_role",
        "min_distance_m",
        "side_by_side_iloc_count",
        "overlap_iloc_count",
        "max_value",
    ):
        if key in measurements:
            fragments.append(
                f"{_humanize_token(key)} was {_format_value(measurements[key])}"
            )
    return fragments


def _gap_percent_sentence_fragments(
    measurements: Dict[str, Any],
    kind: str,
) -> List[str]:
    start_percent = _as_float(measurements.get("start_gap_percent"))
    end_percent = _as_float(measurements.get("end_gap_percent"))
    start_relation = measurements.get("start_gap_relation")
    end_relation = measurements.get("end_gap_relation")
    if (
        start_percent is None
        or end_percent is None
        or start_relation is None
        or end_relation is None
    ):
        return []

    subject = "time gap" if kind == "time" else "speed gap"
    fragments = [
        "the player "
        f"{subject} moved from "
        f"{_gap_percent_relation_text(start_percent, start_relation)} to "
        f"{_gap_percent_relation_text(end_percent, end_relation)}"
    ]
    gain = _as_float(measurements.get("relative_gain_percent"))
    if gain is not None:
        outcome = "gained" if gain > 0.0 else "lost" if gain < 0.0 else "held even"
        fragments.append(
            "net relative change was "
            f"{_format_value(abs(gain))} percentage points {outcome}"
        )
    return fragments


def _gap_percent_relation_text(value: Any, relation: Any) -> str:
    number = _as_float(value)
    if number is None:
        return str(value)
    if relation == "even":
        return "even with expert"
    return f"{_format_value(abs(number))}% {relation} than expert"


def _as_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _format_value(value: Any) -> str:
    number = _as_float(value)
    if number is None:
        return str(value)
    text = f"{number:.3f}".rstrip("0").rstrip(".")
    return text or "0"


def _humanize_token(value: str) -> str:
    return value.replace("_", " ")


def _prompt_block(
    start: int,
    end: int,
    event_text: str,
) -> str:
    lines = [
        "#### Required Upfront Detailed Statistical Preflight",
        "The system already ran deterministic tools and converted their "
        "results into human-readable fact sentences.",
        "These preflight sentences do not identify labels. They only provide "
        "facts with indices and values when available. The sub-label catalog "
        "is the only place that judges which label fits.",
        "Use only these preflight fact sentences and the upfront searched "
        "labels for initial detailed-label reasoning. Reuse the same fact "
        "phrases in the final reasoning when they apply.",
        f"The detailed parent range is [{start}, {end}].",
        "",
        "Preflight fact sentences:",
    ]
    if event_text:
        lines.extend(f"- {line}" for line in event_text.splitlines() if line.strip())
    else:
        lines.append("- (none)")
    return "\n".join(lines)


def _extend(events: List[Dict[str, Any]], new_events: Iterable[Dict[str, Any]]) -> None:
    events.extend(event for event in new_events if event.get("event"))


def _dedupe_events(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen: set[Tuple[Any, Any, Any]] = set()
    for event in events:
        range_value = event.get("range")
        key = (
            event.get("event"),
            event.get("phase"),
            tuple(range_value) if isinstance(range_value, list) else None,
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(event)
    return out[:80]


def _dedupe(values: Iterable[Any]) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value).strip()
        if text and text not in seen:
            seen.add(text)
            out.append(text)
    return out


def _range_from_values(a: Any, b: Any) -> Optional[List[int]]:
    if not isinstance(a, int) or not isinstance(b, int):
        return None
    return [min(a, b), max(a, b)]


def _timing_confidence(delta: Any) -> str:
    if not isinstance(delta, (int, float)):
        return "weak"
    abs_delta = abs(float(delta))
    if abs_delta >= 5:
        return "strong"
    if abs_delta >= 2:
        return "moderate"
    return "weak"


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and value == value


def _median(values: List[float]) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(float(v) for v in values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0
