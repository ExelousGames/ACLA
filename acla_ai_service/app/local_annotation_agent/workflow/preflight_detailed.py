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
DETAILED_PREFLIGHT_QUERY_SPECS = SHARED_PREFLIGHT_QUERY_SPECS


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

