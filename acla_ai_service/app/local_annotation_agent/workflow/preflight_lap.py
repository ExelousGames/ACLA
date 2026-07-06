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
        "tool_id": "query_telemetry.find_trend_runs.speed_difference",
        "graph_id": "speed_delta",
        "query_id": "find_trend_runs",
        "params": {
            "column": "speed_difference",
            "smoothing_window": 1,
        },
        "tags": [
            "speed gap trend run",
            "speed gap recovery",
            "speed gap closing",
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
