"""Lap-flow preflight calculation."""

from __future__ import annotations

from typing import Sequence

from app.local_annotation_agent.workflow.preflight import (
    SHARED_PREFLIGHT_QUERY_SPECS,
    SHARED_PREFLIGHT_TOOL_IDS,
    PreflightContext,
    build_preflight_context as build_shared_preflight_context,
)


LAP_PREFLIGHT_TOOL_IDS = (
    *SHARED_PREFLIGHT_TOOL_IDS,
    "split_lap_by_circuit_sections",
    "classify_opponent_interaction",
    "find_nearest_opponent",
)
LAP_PREFLIGHT_QUERY_SPECS = (
    *SHARED_PREFLIGHT_QUERY_SPECS,
    {
        "tool_id": "query_telemetry.compute_slope.trajectory_offset",
        "graph_id": "trajectory_offset",
        "query_id": "compute_slope",
        "params": {"column": "trajectory_offset"},
        "tags": [
            "trajectory recovery",
            "trajectory merge toward expert line",
            "trajectory move away from expert line",
            "wider than expert",
            "tighter than expert",
        ],
    },
    {
        "tool_id": "query_telemetry.compute_slope.speed_difference",
        "graph_id": "speed_delta",
        "query_id": "compute_slope",
        "params": {"column": "speed_difference"},
        "tags": [
            "speed gap derivative",
            "speed gap recovery",
            "speed gap slope shape",
        ],
    },
    {
        "tool_id": "query_telemetry.measure_point_similarity.throttle",
        "graph_id": "throttle",
        "query_id": "measure_point_similarity",
        "params": {
            "player_column": "Physics_gas",
            "expert_column": "expert_optimal_throttle",
            "smoothing_window": 3,
        },
        "tags": [
            "driver expert throttle similarity",
            "throttle similarity score",
        ],
    },
    {
        "tool_id": "query_telemetry.measure_point_similarity.brake",
        "graph_id": "brake",
        "query_id": "measure_point_similarity",
        "params": {
            "player_column": "Physics_brake",
            "expert_column": "expert_optimal_brake",
            "smoothing_window": 3,
        },
        "tags": [
            "driver expert brake similarity",
            "brake similarity score",
        ],
    },
)


def build_preflight_context(
    *,
    df,
    start: int,
    end: int,
    candidate_label_ids: Sequence[str],
    extra_query_terms: Sequence[str],
) -> PreflightContext:
    return build_shared_preflight_context(
        flow="lap",
        df=df,
        start=start,
        end=end,
        tool_ids=LAP_PREFLIGHT_TOOL_IDS,
        query_specs=LAP_PREFLIGHT_QUERY_SPECS,
        candidate_label_ids=candidate_label_ids,
        extra_query_terms=extra_query_terms,
    )
