"""Detailed-flow preflight calculation."""

from __future__ import annotations

from typing import Sequence

from app.local_annotation_agent.workflow.preflight import (
    SHARED_PREFLIGHT_QUERY_SPECS,
    SHARED_PREFLIGHT_TOOL_IDS,
    PreflightContext,
    build_preflight_context as build_shared_preflight_context,
)


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
    return build_shared_preflight_context(
        flow="detailed",
        df=df,
        start=start,
        end=end,
        tool_ids=DETAILED_PREFLIGHT_TOOL_IDS,
        query_specs=DETAILED_PREFLIGHT_QUERY_SPECS,
        parent_main_labels=parent_main_labels,
        extra_query_terms=extra_query_terms,
    )
