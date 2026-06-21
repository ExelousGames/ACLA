"""Detailed-flow statistical preflight events."""

from __future__ import annotations

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


DETAILED_PREFLIGHT_TOOL_IDS = (
    *SHARED_PREFLIGHT_TOOL_IDS,
    "classify_opponent_interaction",
    "find_nearest_opponent",
)
DETAILED_PREFLIGHT_QUERY_SPECS = (
    *SHARED_PREFLIGHT_QUERY_SPECS,
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
            "smoothing_window": 5,
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
            "smoothing_window": 5,
        },
        "tags": [
            "trajectory similarity",
            "driver expert path comparison",
            "line separation",
        ],
    },
    {
        "tool_id": "query_telemetry.find_threshold_crossing.brake.onset",
        "graph_id": "brake",
        "query_id": "find_threshold_crossing",
        "params": {
            "columns": ["expert_optimal_brake", "Physics_brake"],
            "threshold": 0.05,
            "smoothing_window": 5,
        },
        "tags": [
            "brake initiation onset",
            "brake earlier than expert",
            "brake later than expert",
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
        "tool_id": "query_telemetry.find_threshold_crossing.throttle.onset",
        "graph_id": "throttle",
        "query_id": "find_threshold_crossing",
        "params": {
            "columns": ["expert_optimal_throttle", "Physics_gas"],
            "threshold": 0.05,
            "smoothing_window": 5,
        },
        "tags": [
            "throttle application onset",
            "throttle earlier than expert",
            "throttle later than expert",
        ],
    },
    {
        "tool_id": "query_telemetry.find_extremum.throttle.player.max",
        "graph_id": "throttle",
        "query_id": "find_extremum",
        "params": {"column": "Physics_gas", "kind": "max"},
        "tags": ["player peak throttle pressure"],
    },
    {
        "tool_id": "query_telemetry.find_extremum.throttle.expert.max",
        "graph_id": "throttle",
        "query_id": "find_extremum",
        "params": {"column": "expert_optimal_throttle", "kind": "max"},
        "tags": ["expert peak throttle pressure"],
    },
    {
        "tool_id": "query_telemetry.find_dips_on_main_slope.throttle",
        "graph_id": "throttle",
        "query_id": "find_dips_on_main_slope",
        "params": {
            "column": "Physics_gas",
            "smoothing_window": 5,
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
        "tool_id": "query_telemetry.compute_slope.speed_difference",
        "graph_id": "speed_delta",
        "query_id": "compute_slope",
        "params": {"column": "speed_difference"},
        "tags": [
            "speed gap closing",
            "speed gap growing",
            "large speed gap over 20",
        ],
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
        "entry altitude uphill; corner entry rises uphill across the entry phase"
    ),
    ("entry", "level"): (
        "entry altitude level; corner entry stays broadly level across the entry phase"
    ),
    ("entry", "downhill"): (
        "entry altitude downhill; corner entry falls downhill across the entry phase"
    ),
    ("apex", "uphill"): "apex altitude uphill; altitude rises through the apex window",
    ("apex", "level"): (
        "apex altitude level; altitude stays broadly level through the apex window"
    ),
    ("apex", "downhill"): (
        "apex altitude downhill; altitude falls through the apex window"
    ),
    ("exit", "uphill"): (
        "exit altitude uphill; corner exit rises uphill through the exit phase"
    ),
    ("exit", "level"): (
        "exit altitude level; corner exit stays broadly level through the exit phase"
    ),
    ("exit", "downhill"): (
        "exit altitude downhill; corner exit falls downhill through the exit phase"
    ),
}

_OPPONENT_OUTCOME_SEARCH_WORDS = {
    "pass_completed": [
        "successful overtake player gained a position on a close opponent",
        "pass completed opponent starts ahead and ends behind the player",
        "late-brake attack at corner entry brake initiation later than expert trajectory tightening",
        "outside-line sweep trajectory wider than expert through entry-to-apex",
        "switchback line cross from wider entry to tighter exit earlier throttle pickup",
        "slipstream draft on straight speed greater than expert with throttle at or below expert",
    ],
    "held_defense": [
        "successful defense player held position against a close opponent",
        "held defense opponent threatened from behind or alongside but did not get ahead by exit",
        "inside cover at corner entry brake initiation earlier than expert trajectory tighter than expert",
        "defensive lift on straight throttle drops below expert with no matching brake onset",
    ],
    "failed_attack": [
        "racing mistake failed overtake attempt close opponent caused position or time loss",
        "failed attack player closed or went side-by-side but pass did not complete",
        "failed late-brake attack brake initiation later than expert trajectory tightening but no pass",
        "failed outside-line sweep trajectory wider than expert but no pass",
        "failed switchback line cross attempt but pass did not complete",
        "failed slipstream gain speed greater than expert with throttle at or below expert but no pass",
    ],
    "broken_defense": [
        "racing mistake broken defense opponent got through by exit",
        "defense broken player tried to hold position but opponent passed",
        "broken inside cover brake initiation earlier than expert trajectory tighter than expert",
        "broken defensive lift throttle drops below expert on straight but opponent got through",
    ],
    "close_following": [
        "close-following target-car context opponent in line without decisive attack or defense outcome",
        "draft pressure signed longitudinal gap shrinks but no completed pass",
    ],
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
    events = _build_detailed_events(df, s, e, tool_outputs)
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
) -> List[Dict[str, Any]]:
    by_tool = {tool_id: content for tool_id, content in tool_outputs}
    phases = _phase_windows(by_tool)
    events: List[Dict[str, Any]] = []

    _extend(events, _shape_events(start, end, by_tool, phases))
    _extend(events, _opponent_events(start, end, by_tool))
    _extend(events, _peak_comparison_events(df, start, end, by_tool, phases, "brake"))
    _extend(events, _peak_comparison_events(df, start, end, by_tool, phases, "throttle"))
    _extend(events, _local_input_shape_events(df, start, end, phases))
    _extend(events, _time_delta_events(start, end, by_tool))
    _extend(events, _trajectory_events(df, start, end, by_tool, phases))
    _extend(events, _speed_events(start, end, by_tool, phases))
    _extend(events, _balance_and_grip_events(df, start, end, by_tool, phases))

    return _dedupe_events(events)


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
                    "delta_m": summary.get("delta_m"),
                    "start_altitude_m": summary.get("start_altitude_m"),
                    "end_altitude_m": summary.get("end_altitude_m"),
                },
                "moderate",
                ["measure_segment_shape"],
            ))
    for phase in phases:
        for phase_name in ("entry", "apex", "exit"):
            iloc = phase.get(phase_name)
            if isinstance(iloc, int):
                events.append(_event(
                    f"{phase_name} phase detected",
                    phase_name,
                    [iloc, iloc],
                    {"direction": phase.get("direction")},
                    "strong",
                    ["compute_expert_phases"],
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


def _opponent_events(
    start: int,
    end: int,
    by_tool: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    content = by_tool.get("classify_opponent_interaction") or {}
    outcome = str(content.get("outcome") or "")
    mapped = {
        "pass_completed": "pass completed",
        "held_defense": "held defense",
        "failed_attack": "failed attack",
        "broken_defense": "broken defense",
    }.get(outcome)
    if not mapped:
        return []
    measurements = {
        key: content.get(key)
        for key in (
            "outcome",
            "confidence",
            "confidence_level",
            "primary_slot_for_role",
            "entry_signed_long_gap_m",
            "exit_signed_long_gap_m",
            "min_distance_m",
            "min_lateral_offset_m",
            "side_by_side_iloc_count",
        )
        if key in content
    }
    event = _event(
        mapped,
        "whole_range",
        [start, end],
        measurements,
        _confidence_from_level(content.get("confidence_level")),
        ["classify_opponent_interaction"],
    )
    event["semantic_search_terms"] = _OPPONENT_OUTCOME_SEARCH_WORDS.get(outcome, [])
    return [event]


def _peak_comparison_events(
    df,
    start: int,
    end: int,
    by_tool: Dict[str, Dict[str, Any]],
    phases: List[Dict[str, int]],
    kind: str,
) -> List[Dict[str, Any]]:
    if kind == "brake":
        player_tool = "query_telemetry.find_extremum.brake.player.max"
        expert_tool = "query_telemetry.find_extremum.brake.expert.max"
        phrase = "peak brake pressure"
    else:
        player_tool = "query_telemetry.find_extremum.throttle.player.max"
        expert_tool = "query_telemetry.find_extremum.throttle.expert.max"
        phrase = "peak throttle pressure"
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
    phase = _phase_for_iloc(player_iloc, phases) if isinstance(player_iloc, int) else "unknown"
    measurements = {
        "player_value": player_value,
        "expert_value": expert_value,
        "delta": delta,
        "player_iloc": player_iloc,
        "expert_iloc": expert.get("iloc"),
    }
    if kind == "brake" and isinstance(player_iloc, int):
        speed_gap = _value_at_iloc(
            df,
            start,
            end,
            "speed_difference",
            player_iloc,
            graph_id="speed_delta",
        )
        if speed_gap is not None:
            measurements["speed_gap_at_player_peak"] = speed_gap
    if abs(delta) < 0.05:
        return [_event(
            f"{phrase} about same as expert",
            phase,
            [player_iloc, player_iloc] if isinstance(player_iloc, int) else None,
            measurements,
            "strong" if abs(delta) <= 0.02 else "moderate",
            [player_tool, expert_tool],
        )]
    return [_event(
        f"{phrase} {'higher' if delta > 0 else 'lower'} than expert",
        phase,
        [player_iloc, player_iloc] if isinstance(player_iloc, int) else None,
        measurements,
        "strong" if abs(delta) >= 0.15 else "moderate",
        [player_tool, expert_tool],
    )]


def _local_input_shape_events(
    df,
    start: int,
    end: int,
    phases: List[Dict[str, int]],
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
        for direction, timing_phrase, speed_phrase in (
            ("increase", application_timing, f"{noun} applied"),
            ("decrease", f"{noun} release", f"{noun} release"),
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
                    phases,
                    source,
                ))
            speed = _compare_action_speed(player, expert)
            if speed:
                events.append(_action_speed_event(
                    speed_phrase,
                    player,
                    expert,
                    speed,
                    phases,
                    source,
                ))
    events.extend(_overlap_events(df, start, end))
    return events


def _action_timing_event(
    phrase: str,
    player: Dict[str, Any],
    expert: Dict[str, Any],
    comparison: Dict[str, Any],
    phases: List[Dict[str, int]],
    source: str,
) -> Dict[str, Any]:
    delta = comparison["start_delta_iloc"]
    return _event(
        f"{phrase} onset {'earlier' if delta < 0 else 'later'} than expert",
        _phase_for_iloc(player.get("start_index"), phases),
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


def _action_speed_event(
    phrase: str,
    player: Dict[str, Any],
    expert: Dict[str, Any],
    comparison: Dict[str, Any],
    phases: List[Dict[str, int]],
    source: str,
) -> Dict[str, Any]:
    verdict = comparison["verdict"]
    return _event(
        f"{phrase} too {verdict}",
        _phase_for_iloc(player.get("start_index"), phases),
        _range_from_values(player.get("start_index"), player.get("end_index")),
        {
            "player_duration": player.get("duration"),
            "expert_duration": expert.get("duration"),
            "duration_delta_iloc": comparison.get("duration_delta_iloc"),
            "player_median_raw_slope": player.get("median_raw_slope"),
            "expert_median_raw_slope": expert.get("median_raw_slope"),
            "player_median_normalized_slope": player.get("median_normalized_slope"),
            "expert_median_normalized_slope": expert.get("median_normalized_slope"),
            "slope_ratio": comparison.get("slope_ratio"),
            "player_start_index": player.get("start_index"),
            "player_end_index": player.get("end_index"),
            "expert_start_index": expert.get("start_index"),
            "expert_end_index": expert.get("end_index"),
            "player_start_band": player.get("start_band"),
            "player_end_band": player.get("end_band"),
            "expert_start_band": expert.get("start_band"),
            "expert_end_band": expert.get("end_band"),
            "player_total_movement": player.get("total_movement"),
            "expert_total_movement": expert.get("total_movement"),
            "player_noise_floor": player.get("noise_floor"),
            "expert_noise_floor": expert.get("noise_floor"),
            "direction": player.get("direction"),
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

    events.extend(_trajectory_phase_side_events(df, start, end, phases))
    return events


def _time_delta_events(
    start: int,
    end: int,
    by_tool: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    slope = _query_analysis(
        by_tool.get("query_telemetry.compute_slope.expert_time_difference")
    )
    total = slope.get("total_gap_change") if isinstance(slope, dict) else {}
    if isinstance(total, dict):
        direction = total.get("gap_direction")
        if direction == "time_gap_rising":
            events.append(_event(
                "gap grows",
                "whole_range",
                [start, end],
                {
                    "change": total.get("value"),
                    "threshold_state": total.get("threshold_state"),
                    "slope_shape": slope.get("slope_shape"),
                },
                "strong" if total.get("threshold_state") == "label_threshold_met" else "moderate",
                ["query_telemetry.compute_slope.expert_time_difference"],
            ))
            events.append(_event(
                "time loss",
                "whole_range",
                [start, end],
                {"change": total.get("value")},
                "moderate",
                ["query_telemetry.compute_slope.expert_time_difference"],
            ))
        elif direction == "time_gap_falling":
            events.append(_event(
                "gap shrinks",
                "whole_range",
                [start, end],
                {
                    "change": total.get("value"),
                    "threshold_state": total.get("threshold_state"),
                    "slope_shape": slope.get("slope_shape"),
                },
                "strong" if total.get("threshold_state") == "label_threshold_met" else "moderate",
                ["query_telemetry.compute_slope.expert_time_difference"],
            ))
    trend = _query_analysis(
        by_tool.get("query_telemetry.find_trend_runs.expert_time_difference")
    )
    if isinstance(trend, dict):
        increase = trend.get("selected_gap_increase_run")
        decrease = trend.get("selected_gap_decrease_run")
        if isinstance(increase, dict):
            events.append(_event(
                "time gap rising run",
                "whole_range",
                _range_from_values(
                    increase.get("start_iloc"),
                    increase.get("end_iloc"),
                ),
                increase,
                "strong" if increase.get("threshold_state") == "label_threshold_met" else "moderate",
                ["query_telemetry.find_trend_runs.expert_time_difference"],
            ))
        if isinstance(decrease, dict):
            events.append(_event(
                "time gap falling run",
                "whole_range",
                _range_from_values(
                    decrease.get("start_iloc"),
                    decrease.get("end_iloc"),
                ),
                decrease,
                "strong" if decrease.get("threshold_state") == "label_threshold_met" else "moderate",
                ["query_telemetry.find_trend_runs.expert_time_difference"],
            ))
    return events


def _trajectory_phase_side_events(
    df,
    start: int,
    end: int,
    phases: List[Dict[str, int]],
) -> List[Dict[str, Any]]:
    values = _series_values(df, start, end, "trajectory_offset", graph_id="trajectory_offset")
    if not values:
        return []
    events: List[Dict[str, Any]] = []
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
            if median is None or abs(median) < 0.5:
                continue
            events.append(_event(
                f"{phase_name} trajectory {'wider' if median > 0 else 'tighter'} than expert",
                phase_name,
                [lo, hi],
                {"median_offset": median},
                "strong" if abs(median) >= 1.0 else "moderate",
                ["trajectory_offset_phase_statistics"],
            ))
    return events


def _speed_events(
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
    for result, event_name, threshold, source in (
        (
            max_result,
            "expert faster than player",
            5.0,
            "query_telemetry.find_extremum.speed_difference.max",
        ),
        (
            min_result,
            "player faster than expert",
            -5.0,
            "query_telemetry.find_extremum.speed_difference.min",
        ),
    ):
        if not isinstance(result, dict):
            continue
        value = result.get("value")
        if (
            event_name.startswith("expert")
            and isinstance(value, (int, float))
            and value >= threshold
        ) or (
            event_name.startswith("player")
            and isinstance(value, (int, float))
            and value <= threshold
        ):
            iloc = result.get("iloc")
            events.append(_event(
                event_name,
                _phase_for_iloc(iloc, phases) if isinstance(iloc, int) else "unknown",
                [iloc, iloc] if isinstance(iloc, int) else None,
                {"value": value, "iloc": iloc},
                "strong" if abs(float(value)) >= 20.0 else "moderate",
                [source],
            ))
            if abs(float(value)) > 20.0:
                events.append(_event(
                    "large speed gap over 20",
                    _phase_for_iloc(iloc, phases) if isinstance(iloc, int) else "unknown",
                    [iloc, iloc] if isinstance(iloc, int) else None,
                    {"value": value, "iloc": iloc},
                    "strong",
                    [source],
                ))

    slope = _query_analysis(
        by_tool.get("query_telemetry.compute_slope.speed_difference")
    )
    total = slope.get("total_change") if isinstance(slope, dict) else {}
    if isinstance(total, dict):
        domain = total.get("domain_direction")
        if domain == "speed_gap_decreasing" or total.get("moves_toward_zero") is True:
            events.append(_event(
                "speed gap closing",
                "whole_range",
                None,
                {"change": total.get("value"), "domain_direction": domain},
                "strong",
                ["query_telemetry.compute_slope.speed_difference"],
            ))
        elif domain == "speed_gap_increasing":
            events.append(_event(
                "speed gap growing",
                "whole_range",
                None,
                {"change": total.get("value"), "domain_direction": domain},
                "moderate",
                ["query_telemetry.compute_slope.speed_difference"],
            ))

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
        event_name = (
            "speed local curve stable"
            if direction == "flat"
            else f"speed local curve {direction}"
        )
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
    smoothing_window: int = 5,
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
            or active_count < 2
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


def _compare_action_speed(
    player: Dict[str, Any],
    expert: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    player_duration = player.get("duration")
    expert_duration = expert.get("duration")
    player_slope = player.get("median_normalized_slope")
    expert_slope = expert.get("median_normalized_slope")
    if not isinstance(player_duration, int) or not isinstance(expert_duration, int):
        return None
    if not isinstance(player_slope, (int, float)) or not isinstance(expert_slope, (int, float)):
        return None
    if float(expert_slope) <= 0.0:
        return None
    duration_delta = player_duration - expert_duration
    if abs(duration_delta) < 2:
        return None
    slope_ratio = float(player_slope) / float(expert_slope)
    verdict: Optional[str] = None
    if slope_ratio >= 1.25 and duration_delta <= -2:
        verdict = "quickly"
    elif slope_ratio <= 0.8 and duration_delta >= 2:
        verdict = "slowly"
    if verdict is None:
        return None
    strong_ratio = slope_ratio >= 1.5 if verdict == "quickly" else slope_ratio <= (2.0 / 3.0)
    return {
        "verdict": verdict,
        "duration_delta_iloc": duration_delta,
        "slope_ratio": slope_ratio,
        "confidence": "strong" if abs(duration_delta) >= 5 and strong_ratio else "moderate",
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
    """Embedding query text: sentence-only evidence with label vocabulary."""
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
        terms = event.get("semantic_search_terms")
        if isinstance(terms, list):
            event_name = str(event.get("event") or "").strip()
            for term in terms:
                term_text = str(term).strip()
                if term_text:
                    lines.append(
                        f"The {event_name} evidence also matches label "
                        f"vocabulary for {term_text}."
                    )
    return "\n".join(line for line in lines if line.strip())[:12000]


def _event_sentence(event: Dict[str, Any]) -> str:
    event_name = str(event.get("event") or "").strip()
    if not event_name:
        return ""

    phase = _phase_sentence_prefix(str(event.get("phase") or ""))
    range_text = _range_sentence_fragment(event.get("range"))
    measurements = event.get("measurements")
    if not isinstance(measurements, dict):
        measurements = {}

    fragments = _measurement_sentence_fragments(event_name, measurements)
    confidence = str(event.get("confidence") or "").strip()
    confidence_text = f"with {confidence} confidence" if confidence else ""

    parts = [part for part in [range_text, *fragments, confidence_text] if part]
    detail = ": " + "; ".join(parts) if parts else ""
    return f"{phase}{event_name}{detail}."


def _phase_sentence_prefix(phase: str) -> str:
    if phase in {"entry", "apex", "exit", "straight"}:
        return f"During {phase}, "
    if phase == "whole_range":
        return "Across the whole range, "
    return ""


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

    if (
        "onset earlier than expert" in event_name
        or "onset later than expert" in event_name
    ):
        player = measurements.get("player_start_index")
        expert = measurements.get("expert_start_index")
        delta = measurements.get("start_delta_iloc")
        if player is not None and expert is not None:
            fragments.append(
                f"the player began at iloc {player} while the expert began at iloc {expert}"
            )
        if delta is not None:
            delta_number = _as_float(delta)
            if delta_number is not None:
                direction = "later" if delta_number > 0 else "earlier"
                fragments.append(
                    "the player timing was "
                    f"{_format_value(abs(delta_number))} ilocs {direction}"
                )
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

    if "peak brake pressure" in event_name or "peak throttle pressure" in event_name:
        player = measurements.get("player_value")
        expert = measurements.get("expert_value")
        if player is not None and expert is not None:
            fragments.append(
                f"the player peak was {_format_value(player)} versus "
                f"expert peak {_format_value(expert)}"
            )
        speed_gap = measurements.get("speed_gap_at_player_peak")
        speed_gap_number = _as_float(speed_gap)
        if speed_gap_number is not None:
            if speed_gap_number > 0:
                fragments.append(
                    "the player was "
                    f"{_format_value(speed_gap_number)} km/h slower than expert "
                    "at the player peak"
                )
            elif speed_gap_number < 0:
                fragments.append(
                    "the player was "
                    f"{_format_value(abs(speed_gap_number))} km/h faster at the player peak"
                )
        iloc = measurements.get("player_iloc")
        if iloc is not None:
            fragments.append(f"the player peak occurred at iloc {iloc}")
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
        "gap grows",
        "gap shrinks",
        "time loss",
        "time gap rising run",
        "time gap falling run",
    }:
        change = measurements.get("change")
        if change is not None:
            fragments.append(f"the time gap changed by {_format_value(change)} ms")
        threshold = measurements.get("threshold_state")
        if threshold:
            fragments.append(_humanize_token(str(threshold)))
        slope_shape = measurements.get("slope_shape")
        if slope_shape:
            fragments.append(_humanize_token(str(slope_shape)))
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

    if "altitude" in event_name:
        delta = measurements.get("delta_m")
        if delta is not None:
            fragments.append(f"altitude changed by {_format_value(delta)} m")
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
        "results into evidence sentences.",
        "Use only these preflight evidence sentences and the upfront searched "
        "labels for initial detailed-label reasoning.",
        f"The detailed parent range is [{start}, {end}].",
        "",
        "Preflight evidence sentences:",
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


def _confidence_from_level(value: Any) -> str:
    text = str(value or "").lower()
    if text in {"high", "strong"}:
        return "strong"
    if text in {"medium", "moderate"}:
        return "moderate"
    return "weak" if text else "moderate"


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
