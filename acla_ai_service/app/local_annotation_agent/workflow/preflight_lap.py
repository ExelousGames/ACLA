"""Lap-flow preflight calculation."""

from __future__ import annotations

from typing import Sequence

from app.local_annotation_agent.workflow.preflight import (
    SHARED_PREFLIGHT_QUERY_SPECS,
    SHARED_PREFLIGHT_TOOL_IDS,
    SPEED_INVESTIGATION_QUERY_SPECS,
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
)


def build_preflight_context(
    *,
    df,
    start: int,
    end: int,
    eligible_behavior_label_ids: Sequence[str],
    fixed_label_ids: Sequence[str],
    extra_query_terms: Sequence[str],
) -> PreflightContext:
    return build_shared_preflight_context(
        flow="lap",
        df=df,
        start=start,
        end=end,
        tool_ids=LAP_PREFLIGHT_TOOL_IDS,
        query_specs=LAP_PREFLIGHT_QUERY_SPECS,
        eligible_behavior_label_ids=eligible_behavior_label_ids,
        fixed_label_ids=fixed_label_ids,
        extra_query_terms=extra_query_terms,
    )
