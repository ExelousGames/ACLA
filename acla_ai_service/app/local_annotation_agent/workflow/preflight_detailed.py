"""Detailed-flow statistical preflight events."""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from app.local_annotation_agent.workflow.preflight import (
    SHARED_PREFLIGHT_QUERY_SPECS,
    SHARED_PREFLIGHT_TOOL_IDS,
    PreflightContext,
    _json,
    _preflight_analysis_ids,
    _preflight_query_table,
    _preflight_tool_summary,
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
        "tool_id": "query_telemetry.compute_slope.player_speed",
        "graph_id": "speed",
        "query_id": "compute_slope",
        "params": {"column": "Physics_speed_kmh"},
        "tags": ["player acceleration", "player deceleration"],
    },
    {
        "tool_id": "query_telemetry.find_extremum.trajectory_balance.max",
        "graph_id": "trajectory_balance",
        "query_id": "find_extremum",
        "params": {"column": "slip_balance", "kind": "max"},
        "tags": ["oversteer", "rear slip dominant"],
    },
    {
        "tool_id": "query_telemetry.find_extremum.trajectory_balance.min",
        "graph_id": "trajectory_balance",
        "query_id": "find_extremum",
        "params": {"column": "slip_balance", "kind": "min"},
        "tags": ["understeer", "front slip dominant"],
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
            },
            content_schema="annotation_preflight_context",
        ),
    ])

    return PreflightContext(
        prompt_block=_prompt_block(s, e, tool_outputs, events, event_text),
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
    _extend(events, _input_onset_events(by_tool, phases, "brake"))
    _extend(events, _input_onset_events(by_tool, phases, "throttle"))
    _extend(events, _peak_comparison_events(by_tool, phases, "brake"))
    _extend(events, _peak_comparison_events(by_tool, phases, "throttle"))
    _extend(events, _local_input_shape_events(df, start, end, phases))
    _extend(events, _time_delta_events(start, end, by_tool))
    _extend(events, _trajectory_events(df, start, end, by_tool, phases))
    _extend(events, _speed_events(by_tool, phases))
    _extend(events, _balance_and_grip_events(by_tool, phases))

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
        name = str(base.get("label_name") or "").strip()
        role = str(base.get("segment_type_role") or base.get("shape_key") or "")
        event = _shape_event_name(name, role)
        if event:
            events.append(_event(
                event,
                "straight" if event == "on the straight" else "whole_range",
                [start, end],
                {"label_id": base.get("label_id"), "reason": base.get("reason")},
                "strong",
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


def _shape_event_name(name: str, role: str) -> Optional[str]:
    lowered = name.lower()
    if lowered:
        return lowered
    by_role = {
        "corner": "in the corner",
        "straight": "on the straight",
        "approach_to_corner": "approach to corner",
        "exit_corner_to_straight": "exit corner leading to straight",
        "between_consecutive_corners": "between consecutive corners",
        "consecutive_corners_no_straight": "consecutive corners with no straight in between",
    }
    return by_role.get(role)


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
            "recommended_label",
            "confidence",
            "confidence_level",
            "primary_slot_for_role",
            "entry_signed_long_gap_m",
            "exit_signed_long_gap_m",
            "min_distance_m",
            "min_lateral_offset_m",
            "side_by_side_iloc_count",
            "label_gates",
        )
        if key in content
    }
    return [_event(
        mapped,
        "whole_range",
        [start, end],
        measurements,
        _confidence_from_level(content.get("confidence_level")),
        ["classify_opponent_interaction"],
    )]


def _input_onset_events(
    by_tool: Dict[str, Dict[str, Any]],
    phases: List[Dict[str, int]],
    kind: str,
) -> List[Dict[str, Any]]:
    if kind == "brake":
        tool_id = "query_telemetry.find_threshold_crossing.brake.onset"
        player_col = "Physics_brake"
        expert_col = "expert_optimal_brake"
        phrase = "brake initiation onset"
    else:
        tool_id = "query_telemetry.find_threshold_crossing.throttle.onset"
        player_col = "Physics_gas"
        expert_col = "expert_optimal_throttle"
        phrase = "throttle application onset"
    result = _query_result(by_tool.get(tool_id))
    samples = result.get("samples") if isinstance(result, dict) else None
    if not isinstance(samples, list):
        return []
    by_column = {
        str(sample.get("column") or ""): sample
        for sample in samples
        if isinstance(sample, dict)
    }
    player = by_column.get(player_col)
    expert = by_column.get(expert_col)
    if not player or not expert:
        return []
    player_iloc = player.get("iloc")
    expert_iloc = expert.get("iloc")
    if not isinstance(player_iloc, int) or not isinstance(expert_iloc, int):
        return []
    delta = player_iloc - expert_iloc
    if delta == 0:
        event_name = f"{phrase} aligned with expert"
    else:
        event_name = f"{phrase} {'earlier' if delta < 0 else 'later'} than expert"
    return [_event(
        event_name,
        _phase_for_iloc(player_iloc, phases),
        [min(player_iloc, expert_iloc), max(player_iloc, expert_iloc)],
        {
            "player_iloc": player_iloc,
            "expert_iloc": expert_iloc,
            "delta_iloc": delta,
        },
        _timing_confidence(delta),
        [tool_id],
    )]


def _peak_comparison_events(
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
    if abs(delta) < 0.05:
        return []
    player_iloc = player.get("iloc")
    phase = _phase_for_iloc(player_iloc, phases) if isinstance(player_iloc, int) else "unknown"
    return [_event(
        f"{phrase} {'higher' if delta > 0 else 'lower'} than expert",
        phase,
        [player_iloc, player_iloc] if isinstance(player_iloc, int) else None,
        {
            "player_value": player_value,
            "expert_value": expert_value,
            "delta": delta,
            "player_iloc": player_iloc,
            "expert_iloc": expert.get("iloc"),
        },
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
        player = _input_profile(df, start, end, player_col)
        expert = _input_profile(df, start, end, expert_col)
        if not player or not expert:
            continue
        events.extend(_duration_events(kind, noun, player, expert, phases))
    events.extend(_overlap_events(df, start, end))
    return events


def _duration_events(
    kind: str,
    noun: str,
    player: Dict[str, Any],
    expert: Dict[str, Any],
    phases: List[Dict[str, int]],
) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    source = f"local_{kind}_shape_statistics"
    rise = _duration_delta(player, expert, "rise_duration")
    if rise is not None:
        event = (
            f"{noun} applied too quickly"
            if rise < 0
            else f"{noun} applied too slowly"
        )
        events.append(_event(
            event,
            _phase_for_iloc(player.get("peak_iloc"), phases),
            _range_from_values(player.get("onset_iloc"), player.get("peak_iloc")),
            {
                "player_rise_duration": player.get("rise_duration"),
                "expert_rise_duration": expert.get("rise_duration"),
                "delta_iloc": rise,
            },
            _timing_confidence(rise),
            [source],
        ))
    release_onset = _timing_delta(player, expert, "release_onset_iloc")
    if release_onset is not None:
        events.append(_event(
            f"{noun} release onset {'earlier' if release_onset < 0 else 'later'} than expert",
            _phase_for_iloc(player.get("release_onset_iloc"), phases),
            _range_from_values(
                player.get("release_onset_iloc"),
                expert.get("release_onset_iloc"),
            ),
            {
                "player_release_onset_iloc": player.get("release_onset_iloc"),
                "expert_release_onset_iloc": expert.get("release_onset_iloc"),
                "delta_iloc": release_onset,
            },
            _timing_confidence(release_onset),
            [source],
        ))
    release = _duration_delta(player, expert, "release_duration")
    if release is not None:
        event = (
            f"{noun} release too quickly"
            if release < 0
            else f"{noun} release too slowly"
        )
        events.append(_event(
            event,
            _phase_for_iloc(player.get("release_onset_iloc"), phases),
            _range_from_values(player.get("peak_iloc"), player.get("off_iloc")),
            {
                "player_release_duration": player.get("release_duration"),
                "expert_release_duration": expert.get("release_duration"),
                "delta_iloc": release,
            },
            _timing_confidence(release),
            [source],
        ))
    return events


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
    by_tool: Dict[str, Dict[str, Any]],
    phases: List[Dict[str, int]],
) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
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
    speed_total = player_speed.get("total_change") if isinstance(player_speed, dict) else {}
    if isinstance(speed_total, dict):
        domain = speed_total.get("domain_direction")
        if domain == "rising":
            events.append(_event(
                "acceleration onset",
                "whole_range",
                None,
                {"change": speed_total.get("value")},
                "moderate",
                ["query_telemetry.compute_slope.player_speed"],
            ))
        elif domain == "falling":
            events.append(_event(
                "deceleration onset",
                "whole_range",
                None,
                {"change": speed_total.get("value")},
                "moderate",
                ["query_telemetry.compute_slope.player_speed"],
            ))
    return events


def _balance_and_grip_events(
    by_tool: Dict[str, Dict[str, Any]],
    phases: List[Dict[str, int]],
) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    for tool_id, event_name, predicate in (
        (
            "query_telemetry.find_extremum.trajectory_balance.max",
            "oversteer",
            lambda value: value >= 0.02,
        ),
        (
            "query_telemetry.find_extremum.trajectory_balance.min",
            "understeer",
            lambda value: value <= -0.02,
        ),
    ):
        result = _query_result(by_tool.get(tool_id))
        if not result:
            continue
        value = result.get("value")
        if not isinstance(value, (int, float)) or not predicate(float(value)):
            continue
        iloc = result.get("iloc")
        events.append(_event(
            event_name,
            _phase_for_iloc(iloc, phases) if isinstance(iloc, int) else "unknown",
            [iloc, iloc] if isinstance(iloc, int) else None,
            {"value": value, "iloc": iloc},
            "strong" if abs(float(value)) >= 0.05 else "moderate",
            [tool_id],
        ))

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


def _input_profile(
    df,
    start: int,
    end: int,
    column: str,
    *,
    threshold: float = 0.05,
    smoothing_window: int = 5,
) -> Optional[Dict[str, Any]]:
    values = _series_values(df, start, end, column)
    if len(values) < 3:
        return None
    ilocs = [iloc for iloc, _ in values]
    arr = _rolling_median([value for _, value in values], smoothing_window)
    finite = [(iloc, value) for iloc, value in zip(ilocs, arr) if _is_number(value)]
    if len(finite) < 3:
        return None
    above = [(iloc, value) for iloc, value in finite if value >= threshold]
    if not above:
        return None
    onset_iloc = above[0][0]
    peak_iloc, peak_value = max(finite, key=lambda item: float(item[1]))
    after_peak = [(iloc, value) for iloc, value in finite if iloc > peak_iloc]
    off = next((iloc for iloc, value in after_peak if value <= threshold), None)
    release_onset = None
    for iloc, value in after_peak:
        if value < float(peak_value) - 0.05:
            release_onset = iloc
            break
    out = {
        "onset_iloc": onset_iloc,
        "peak_iloc": peak_iloc,
        "peak_value": peak_value,
        "release_onset_iloc": release_onset,
        "off_iloc": off,
        "rise_duration": peak_iloc - onset_iloc,
    }
    if release_onset is not None and off is not None and off >= release_onset:
        out["release_duration"] = off - release_onset
    return out


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
    lines = [
        "detailed statistical semantic events",
        "parent_main_labels: " + ", ".join(str(label) for label in parent_main_labels),
        "extra_terms: " + " ".join(str(term) for term in extra_query_terms),
    ]
    for event in events:
        parts = [
            str(event.get("event") or ""),
            f"phase={event.get('phase')}",
            f"range={event.get('range')}",
            f"confidence={event.get('confidence')}",
        ]
        measurements = event.get("measurements")
        if isinstance(measurements, dict) and measurements:
            parts.append("measurements=" + json.dumps(measurements, sort_keys=True, default=str))
        lines.append("; ".join(part for part in parts if part))
    return "\n".join(lines)[:12000]


def _prompt_block(
    start: int,
    end: int,
    tool_outputs: Sequence[Tuple[str, Dict[str, Any]]],
    events: Sequence[Dict[str, Any]],
    event_text: str,
) -> str:
    lines = [
        "#### Required Upfront Detailed Statistical Preflight",
        "The system already ran deterministic tools and converted their results into statistical semantic events.",
        "Use these events as the primary evidence package for querying annotation knowledge.",
        "No label IDs are preselected by preflight; call `search_labels` with event phrases and the relevant `parent_id` before submitting any label.",
        f"Flow: detailed",
        f"Range: [{start}, {end}]",
        "",
        "Statistical semantic events:",
    ]
    if events:
        for event in events:
            lines.append(
                "- "
                + str(event.get("event"))
                + f" | phase={event.get('phase')}"
                + f" | range={event.get('range')}"
                + f" | confidence={event.get('confidence')}"
            )
    else:
        lines.append("- (none)")
    lines.extend([
        "",
        "Event text for search_labels:",
        event_text or "(none)",
        "",
        "Required tool outputs:",
    ])
    for tool_id, content in tool_outputs:
        summary = _preflight_tool_summary(tool_id, content)
        if summary:
            lines.append(summary)
        display_content = _semantic_tool_output(tool_id, content)
        lines.append(f"##### {tool_id}\n```json\n{_json(display_content, 2200)}\n```")
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


def _timing_delta(
    player: Dict[str, Any],
    expert: Dict[str, Any],
    key: str,
) -> Optional[int]:
    player_value = player.get(key)
    expert_value = expert.get(key)
    if not isinstance(player_value, int) or not isinstance(expert_value, int):
        return None
    delta = player_value - expert_value
    return delta if delta != 0 else None


def _duration_delta(
    player: Dict[str, Any],
    expert: Dict[str, Any],
    key: str,
) -> Optional[int]:
    return _timing_delta(player, expert, key)


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
