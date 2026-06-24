"""Shared upfront analysis package for annotation flows."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from app.internal_knowledge_base.label_search import get_doc, search
from app.local_annotation_agent.workflow.tools import shape_label_doc_for_llm
from app.shared.contracts import Attachment
from app.shared.labels import LABEL_MAPPING


SHARED_PREFLIGHT_TOOL_IDS: Tuple[str, ...] = (
    "get_circuit_id",
    "compute_expert_phases",
    "measure_segment_shape",
    "locate_circuit_section",
)
PREFLIGHT_TOOL_IDS: Tuple[str, ...] = (
    *SHARED_PREFLIGHT_TOOL_IDS,
    "split_lap_by_circuit_sections",
    "classify_opponent_interaction",
    "find_nearest_opponent",
)
SHARED_PREFLIGHT_QUERY_SPECS: Tuple[Dict[str, Any], ...] = (
    {
        "tool_id": "query_telemetry.find_trend_runs.expert_time_difference",
        "graph_id": "time_delta",
        "query_id": "find_trend_runs",
        "params": {
            "column": "expert_time_difference",
            "smoothing_window": 1,
        },
    },
    {
        "tool_id": "query_telemetry.compute_slope.expert_time_difference",
        "graph_id": "time_delta",
        "query_id": "compute_slope",
        "params": {"column": "expert_time_difference"},
    },
    {
        "tool_id": "query_telemetry.find_extremum.trajectory_offset.max",
        "graph_id": "trajectory_offset",
        "query_id": "find_extremum",
        "params": {"column": "trajectory_offset", "kind": "max"},
    },
    {
        "tool_id": "query_telemetry.find_extremum.trajectory_offset.min",
        "graph_id": "trajectory_offset",
        "query_id": "find_extremum",
        "params": {"column": "trajectory_offset", "kind": "min"},
    },
)
PREFLIGHT_QUERY_SPECS: Tuple[Dict[str, Any], ...] = SHARED_PREFLIGHT_QUERY_SPECS
SPEED_INVESTIGATION_QUERY_SPECS: Tuple[Dict[str, Any], ...] = (
    {
        "tool_id": "query_telemetry.find_extremum.player_speed.max",
        "graph_id": "speed",
        "query_id": "find_extremum",
        "params": {"column": "Physics_speed_kmh", "kind": "max"},
        "tags": ["player speed maximum", "top speed"],
    },
    {
        "tool_id": "query_telemetry.find_extremum.player_speed.min",
        "graph_id": "speed",
        "query_id": "find_extremum",
        "params": {"column": "Physics_speed_kmh", "kind": "min"},
        "tags": ["player speed minimum", "minimum speed"],
    },
    {
        "tool_id": "query_telemetry.find_trend_runs.player_speed",
        "graph_id": "speed",
        "query_id": "find_trend_runs",
        "params": {
            "column": "Physics_speed_kmh",
            "smoothing_window": 1,
        },
        "tags": [
            "player speed trend run",
            "player acceleration",
            "player deceleration",
        ],
    },
    {
        "tool_id": "query_telemetry.compute_slope.player_speed",
        "graph_id": "speed",
        "query_id": "compute_slope",
        "params": {"column": "Physics_speed_kmh"},
        "tags": ["speed overall trend", "player acceleration", "player deceleration"],
    },
)

_TAG_KEYS = {
    "annotation_scope",
    "constant_offset_only",
    "confidence_level",
    "data_available",
    "direction",
    "domain_direction",
    "ends_near_zero",
    "is_ambiguous",
    "is_label_significant",
    "label_id",
    "label_name",
    "moves_toward_zero",
    "outcome",
    "parent",
    "role",
    "semantic_tags",
    "slope_shape",
    "segment_type_role",
    "shape_key",
    "significance",
    "starts_near_zero",
    "tags",
    "total_change_direction",
    "total_change_domain_direction",
    "total_change_is_label_significant",
    "total_change_significance",
    "trend",
    "type",
    "verdict",
}
_REQUIRED_PARENTS = {"O", "OD", "PS", "RM", "MSP", "MSR"}


@dataclass(frozen=True)
class PreflightContext:
    prompt_block: str
    attachments: List[Attachment]
    label_candidates: List[Dict[str, Any]]


_GRAPH_SEMANTIC_PROFILES: Dict[str, Dict[str, Any]] = {
    "time_delta": {
        "target": "time gap to expert",
        "columns": ("expert_time_difference",),
        "include_zero_tags": False,
    },
    "trajectory_detailed": {
        "target": "driver/expert trajectory",
        "include_zero_tags": False,
    },
    "trajectory_offset": {
        "target": "trajectory offset",
        "columns": ("trajectory_offset",),
        "include_zero_tags": True,
        "value_tags": {
            "moving_wider": (
                "moving toward positive",
                "widening",
                "trajectory moving wider",
            ),
            "moving_tighter": (
                "moving toward negative",
                "tightening",
                "trajectory moving tighter",
            ),
            "stable": ("aligned",),
            "slope_decreasing_over_section": (
                "trajectory offset slope decreasing over section",
            ),
            "slope_increasing_over_section": (
                "trajectory offset slope increasing over section",
            ),
            "slope_steady_over_section": (
                "trajectory offset slope steady over section",
            ),
            "reversing_to_falling_within_section": (
                "trajectory offset reversing tighter within section",
            ),
            "reversing_to_rising_within_section": (
                "trajectory offset reversing wider within section",
            ),
        },
        "extra_keys": (
            "verdict",
            "total_change_domain_direction",
            "slope_shape",
            "domain_direction",
        ),
        "zero_tags": {
            "starts_near_zero": "trajectory offset starts near expert line",
            "ends_near_zero": "trajectory offset ends near expert line",
            "moves_toward_zero": (
                "trajectory offset moves toward expert line",
                "trajectory offset recovery toward expert line",
            ),
        },
    },
    "speed_delta": {
        "target": "speed delta",
        "columns": ("speed_difference",),
        "include_zero_tags": True,
        "value_tags": {
            "speed_gap_increasing": ("speed gap growing",),
            "speed_gap_decreasing": ("speed gap closing",),
            "stable": ("speed match",),
            "slope_decreasing_over_section": (
                "speed gap slope decreasing over section",
            ),
            "slope_increasing_over_section": (
                "speed gap slope increasing over section",
            ),
            "slope_steady_over_section": (
                "speed gap slope steady over section",
            ),
            "reversing_to_falling_within_section": (
                "speed gap reversing down within section",
            ),
            "reversing_to_rising_within_section": (
                "speed gap reversing up within section",
            ),
        },
        "extra_keys": (
            "total_change_domain_direction",
            "slope_shape",
            "domain_direction",
        ),
        "zero_tags": {
            "starts_near_zero": "speed delta starts near parity",
            "ends_near_zero": "speed delta ends near parity",
            "moves_toward_zero": (
                "speed delta moves toward parity",
                "speed delta recovery toward parity",
            ),
        },
    },
    "speed": {
        "target": "player speed",
        "columns": ("Physics_speed_kmh", "expert_optimal_speed"),
        "include_zero_tags": False,
        "value_tags": {
            "rising": ("acceleration onset",),
            "falling": ("deceleration onset",),
            "stable": ("speed stable",),
        },
        "extra_keys": ("total_change_direction",),
    },
    "brake": {
        "target": "brake input",
        "columns": ("Physics_brake", "expert_optimal_brake"),
        "include_zero_tags": False,
    },
    "throttle": {
        "target": "throttle input",
        "columns": ("Physics_gas", "expert_optimal_throttle"),
        "include_zero_tags": False,
    },
    "trajectory_balance": {
        "target": "slip balance",
        "columns": ("slip_balance",),
        "include_zero_tags": False,
    },
    "push_limit": {
        "target": "grip utilisation",
        "columns": ("driver_push_to_limit",),
        "include_zero_tags": False,
    },
}


def build_preflight_context(
    *,
    flow: str,
    df,
    start: int,
    end: int,
    tool_ids: Optional[Sequence[str]] = None,
    query_specs: Optional[Sequence[Dict[str, Any]]] = None,
    parent_main_labels: Optional[Sequence[str]] = None,
    eligible_behavior_label_ids: Optional[Sequence[str]] = None,
    fixed_label_ids: Optional[Sequence[str]] = None,
    extra_query_terms: Optional[Sequence[str]] = None,
    strict_query_errors: bool = False,
) -> PreflightContext:
    s, e = int(start), int(end)
    if e <= s:
        raise RuntimeError(f"annotation preflight: invalid range [{s}, {e}]")

    selected_tool_ids = tuple(tool_ids or PREFLIGHT_TOOL_IDS)
    selected_query_specs = tuple(query_specs or PREFLIGHT_QUERY_SPECS)
    tool_outputs = [
        *_run_tools(df, s, e, selected_tool_ids),
        *_run_queries(df, s, e, selected_query_specs, strict=strict_query_errors),
    ]
    semantic_summaries = _preflight_semantic_summaries(tool_outputs)
    tags = _dedupe(
        tag
        for tool_id, content in tool_outputs
        for tag in [f"tool:{tool_id}", *_semantic_tags(tool_id, content)]
    )[:160]
    evidence = _evidence_text(
        flow=flow,
        start=s,
        end=e,
        tool_outputs=tool_outputs,
        tags=tags,
        semantic_summaries=semantic_summaries,
        parent_main_labels=list(parent_main_labels or []),
        eligible_behavior_label_ids=list(eligible_behavior_label_ids or []),
        fixed_label_ids=list(fixed_label_ids or []),
        extra_query_terms=list(extra_query_terms or []),
    )
    candidates = _label_candidates(
        evidence,
        parent_main_labels=list(parent_main_labels or []),
        eligible_behavior_label_ids=list(eligible_behavior_label_ids or []),
    )

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
            name="init.preflight_label_candidates",
            kind="structured",
            label="Preflight Semantic Label Candidates",
            content={
                "range": [s, e],
                "tool_output_tags": tags,
                "candidates": candidates,
            },
            content_schema="annotation_preflight_labels",
        ),
        Attachment(
            name="init.annotation_preflight_context",
            kind="structured",
            label="Annotation Preflight Context",
            content={
                "flow": flow,
                "range": [s, e],
                "required_tools": _preflight_analysis_ids(
                    selected_tool_ids,
                    selected_query_specs,
                ),
                "tool_output_tags": tags,
                "semantic_summaries": semantic_summaries,
                "label_candidate_ids": [c["id"] for c in candidates],
                "semantic_evidence_text": evidence,
            },
            content_schema="annotation_preflight_context",
        ),
    ])

    return PreflightContext(
        prompt_block=_prompt_block(
            flow,
            s,
            e,
            tool_outputs,
            tags,
            candidates,
            semantic_summaries,
        ),
        attachments=attachments,
        label_candidates=candidates,
    )


def _preflight_analysis_ids(
    tool_ids: Sequence[str],
    query_specs: Sequence[Dict[str, Any]],
) -> List[str]:
    return [
        *tool_ids,
        *(str(spec["tool_id"]) for spec in query_specs),
    ]


def _run_tools(
    df,
    start: int,
    end: int,
    tool_ids: Sequence[str],
) -> List[Tuple[str, Dict[str, Any]]]:
    from app.shared.annotation_agent_tools import get_circuit_id, get_pipeline_tool

    out: List[Tuple[str, Dict[str, Any]]] = []
    circuit_id: Optional[str] = None
    for tool_id in tool_ids:
        try:
            if tool_id == "get_circuit_id":
                attachment = get_circuit_id(df)
                content = getattr(attachment, "content", None)
                if isinstance(content, dict):
                    circuit_id = content.get("circuit_id")
            else:
                tool_def = get_pipeline_tool(tool_id)
                if tool_def is None:
                    raise RuntimeError(
                        f"annotation preflight: required tool {tool_id!r} missing"
                    )
                if tool_id == "locate_circuit_section":
                    attachment = tool_def["callable"](df, circuit_id, start, end)
                else:
                    attachment = tool_def["callable"](df, start, end)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"annotation preflight: required tool {tool_id!r} failed: {exc}"
            ) from exc
        content = getattr(attachment, "content", None)
        if not isinstance(content, dict):
            raise RuntimeError(
                f"annotation preflight: required tool {tool_id!r} returned "
                "non-structured content"
            )
        out.append((tool_id, content))
    return out


def _run_queries(
    df,
    start: int,
    end: int,
    query_specs: Sequence[Dict[str, Any]] = PREFLIGHT_QUERY_SPECS,
    *,
    strict: bool = False,
) -> List[Tuple[str, Dict[str, Any]]]:
    from app.shared.annotation_agent_tools import build_graph, run_pipeline_query

    out: List[Tuple[str, Dict[str, Any]]] = []
    for spec in query_specs:
        tool_id = str(spec["tool_id"])
        graph_id = str(spec["graph_id"])
        query_id = str(spec["query_id"])
        table = build_graph(graph_id, df)
        if table is None:
            if strict:
                raise RuntimeError(
                    f"annotation preflight: required query {tool_id!r} cannot "
                    f"build `{graph_id}` graph table"
                )
            out.append((
                tool_id,
                {
                    "graph_id": graph_id,
                    "query_id": query_id,
                    "params": {
                        **dict(spec["params"]),
                        "range": [int(start), int(end)],
                    },
                    "error": (
                        f"cannot build `{graph_id}` graph table for "
                        "preflight query"
                    ),
                },
            ))
            continue
        table = _preflight_query_table(table, start, end)
        query_range = _preflight_query_range(table, start, end)
        if query_range is None:
            if strict:
                raise RuntimeError(
                    f"annotation preflight: required query {tool_id!r} has no "
                    f"rows overlapping range [{int(start)}, {int(end)}]"
                )
            out.append((
                tool_id,
                {
                    "graph_id": graph_id,
                    "query_id": query_id,
                    "params": {
                        **dict(spec["params"]),
                        "range": [int(start), int(end)],
                    },
                    "error": (
                        f"`{graph_id}` graph table has no rows overlapping "
                        f"preflight range [{int(start)}, {int(end)}]"
                    ),
                },
            ))
            continue
        params = {
            **dict(spec["params"]),
            "range": query_range,
        }
        payload, error = run_pipeline_query(table, query_id, params)
        content: Dict[str, Any] = {
            "graph_id": graph_id,
            "query_id": query_id,
            "params": params,
            "result": payload,
            "semantic_target": _query_semantic_target(spec),
            "analysis": _query_analysis({
                "graph_id": graph_id,
                "query_id": query_id,
                "params": params,
                "result": payload,
            }),
            "semantic_tags": _query_semantic_tags(spec, payload),
        }
        if error:
            if strict:
                raise RuntimeError(
                    f"annotation preflight: required query {tool_id!r} failed: "
                    f"{error}"
                )
            content["error"] = error
        out.append((tool_id, content))
    return out


def _query_semantic_tags(
    spec: Dict[str, Any],
    payload: Dict[str, Any],
) -> List[str]:
    query_id = str(spec.get("query_id") or "")
    params = spec.get("params") if isinstance(spec.get("params"), dict) else {}
    column = str(params.get("column") or "")
    graph_profile = _query_semantic_profile(spec)
    tags: List[str] = _query_static_tags(spec, query_id, graph_profile)
    result = payload if isinstance(payload, dict) else {}
    if column == "expert_time_difference":
        return _time_delta_query_semantic_tags(tags, query_id, result)
    extra = result.get("extra")
    if isinstance(extra, dict):
        for key in graph_profile.get("extra_keys", ()):
            value = extra.get(key)
            tags.extend(_query_value_tags(graph_profile, value))
        if graph_profile.get("include_zero_tags"):
            _append_zero_tags(tags, extra.get("near_zero_summary"), graph_profile)

    if query_id == "find_extremum":
        tags.extend(_extremum_tags(column, result.get("value")))
    elif query_id == "find_trend_runs":
        tags.extend(
            _trend_run_tags(
                graph_profile,
                extra if isinstance(extra, dict) else {},
            )
        )
    elif query_id == "compute_slope":
        tags.extend(_slope_tags(column, extra if isinstance(extra, dict) else {}))
    elif query_id == "measure_trajectory_similarity":
        tags.extend(
            _trajectory_similarity_tags(extra if isinstance(extra, dict) else {})
        )
    elif query_id == "find_threshold_crossing":
        tags.extend(_threshold_crossing_tags(result.get("samples")))
    elif query_id == "find_dips_on_main_slope":
        samples = result.get("samples") or []
        if isinstance(samples, list) and samples:
            tags.append("modulation dip")
            tags.append(f"{column} dip detected".strip())

    return _dedupe(tags)[:24]


def _time_delta_query_semantic_tags(
    tags: List[str],
    query_id: str,
    result: Dict[str, Any],
) -> List[str]:
    extra = result.get("extra")
    if not isinstance(extra, dict):
        return _dedupe(tags)[:24]
    if query_id == "find_trend_runs":
        verdict = _time_delta_trend_verdict(extra)
        if verdict == "time_gap_rising_and_falling":
            tags.extend([
                "time gap rising",
                "time gap falling",
                "mixed time-gap trend",
            ])
        elif verdict == "time_gap_rising":
            tags.extend(["time gap rising", "gap increasing"])
        elif verdict == "time_gap_falling":
            tags.extend(["time gap falling", "gap decreasing"])
        elif verdict == "constant_carried_time_gap":
            tags.append("constant carried time gap")
        else:
            tags.append("gap holds stable")
        for run in _time_delta_trend_run_analysis(extra).get(
            "significant_gap_runs",
            [],
        ):
            if isinstance(run, dict):
                tags.extend(_time_delta_gap_tags(run.get("gap_direction")))
    elif query_id == "compute_slope":
        tags.extend(
            _time_delta_gap_tags(
                _time_delta_gap_direction(
                    extra.get("total_change_direction"),
                    extra.get("delta_value"),
                )
            )
        )
        tags.extend(_time_delta_slope_shape_tags(extra))
    return _dedupe(tags)[:24]


def _time_delta_slope_shape_tags(extra: Dict[str, Any]) -> List[str]:
    shape = extra.get("slope_shape")
    gap_direction = _time_delta_gap_direction(
        extra.get("total_change_direction"),
        extra.get("delta_value"),
    )
    if shape == "slope_decreasing_over_section":
        if gap_direction == "time_gap_rising":
            return [
                "recovery trend",
                "rate of losing time decreasing",
                "time loss decelerating",
            ]
        if gap_direction == "time_gap_falling":
            return ["recovery accelerating", "time gap falling faster"]
        return ["slope decreasing over section"]
    if shape == "slope_increasing_over_section":
        if gap_direction == "time_gap_rising":
            return ["losing time accelerating", "time gap rising faster"]
        if gap_direction == "time_gap_falling":
            return ["recovery easing", "time gap falling slower"]
        return ["slope increasing over section"]
    if shape == "slope_steady_over_section":
        return ["time gap slope steady over section"]
    if shape == "reversing_to_falling_within_section":
        return ["time gap reversing to recovery within section"]
    if shape == "reversing_to_rising_within_section":
        return ["time gap reversing to loss within section"]
    return []


def _query_semantic_profile(spec: Dict[str, Any]) -> Dict[str, Any]:
    graph_id = str(spec.get("graph_id") or "")
    profile = dict(_GRAPH_SEMANTIC_PROFILES.get(graph_id, {}))
    if profile:
        return profile
    return {"target": graph_id or "telemetry", "include_zero_tags": False}


def _query_semantic_target(spec: Dict[str, Any]) -> str:
    return str(_query_semantic_profile(spec).get("target") or "telemetry")


def _query_static_tags(
    spec: Dict[str, Any],
    query_id: str,
    profile: Dict[str, Any],
) -> List[str]:
    tags = [
        str(tag).strip()
        for tag in spec.get("tags", [])
        if str(tag).strip()
    ]
    if query_id == "find_threshold_crossing":
        return [
            tag
            for tag in tags
            if not any(
                phrase in tag
                for phrase in (
                    " earlier than expert",
                    " later than expert",
                    " aligned with expert",
                )
            )
        ]
    if query_id in {"compute_slope", "find_trend_runs"}:
        outcome_tags = _profile_value_tag_set(profile)
        tags = [tag for tag in tags if tag not in outcome_tags]
    return tags


def _profile_value_tag_set(profile: Dict[str, Any]) -> set[str]:
    return {
        str(tag)
        for mapped in profile.get("value_tags", {}).values()
        for tag in mapped
    }


def _query_value_tags(profile: Dict[str, Any], value: Any) -> List[str]:
    if value is None:
        return []
    mapped = profile.get("value_tags", {}).get(str(value))
    if mapped:
        return [str(tag) for tag in mapped]
    return []


def _trend_run_tags(profile: Dict[str, Any], extra: Dict[str, Any]) -> List[str]:
    runs = extra.get("significant_runs")
    if not isinstance(runs, list):
        return []
    return [
        tag
        for run in runs
        if isinstance(run, dict)
        for tag in _query_value_tags(profile, run.get("domain_direction"))
    ]


def _append_zero_tags(
    tags: List[str],
    summary: Any,
    profile: Dict[str, Any],
) -> None:
    if not isinstance(summary, dict):
        return
    zero_tags = profile.get("zero_tags")
    zero_tags = zero_tags if isinstance(zero_tags, dict) else {}
    if summary.get("starts_near_zero") is True:
        tags.extend(_zero_tag_values(profile, zero_tags, "starts_near_zero"))
    if summary.get("ends_near_zero") is True:
        tags.extend(_zero_tag_values(profile, zero_tags, "ends_near_zero"))
    if summary.get("moves_toward_zero") is True:
        tags.extend(_zero_tag_values(profile, zero_tags, "moves_toward_zero"))


def _zero_tag_values(
    profile: Dict[str, Any],
    zero_tags: Dict[str, Any],
    key: str,
) -> List[str]:
    value = zero_tags.get(key)
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return [str(tag) for tag in value if str(tag).strip()]
    target = str(profile.get("target") or "telemetry").strip() or "telemetry"
    fallback = {
        "starts_near_zero": f"{target} starts near zero",
        "ends_near_zero": f"{target} ends near zero",
        "moves_toward_zero": f"{target} moves toward zero",
    }
    return [fallback[key]]


def _extremum_tags(column: str, value: Any) -> List[str]:
    if not isinstance(value, (int, float)):
        return []
    if column == "trajectory_offset":
        if value >= 0.5:
            return ["wider than expert", "trajectory offset positive"]
        if value <= -0.5:
            return ["tighter than expert", "trajectory offset negative"]
    if column == "speed_difference":
        tags = ["expert faster than player"] if value >= 5 else []
        if value <= -5:
            tags.append("player faster than expert")
        if abs(value) > 20:
            tags.append("large speed gap over 20")
        return tags
    if column == "slip_balance":
        if value >= 0.02:
            return ["oversteer", "rear slip dominant"]
        if value <= -0.02:
            return ["understeer", "front slip dominant"]
    if column == "driver_push_to_limit":
        if value >= 1.0:
            return ["over-limit spike", "tire sustained over peak grip"]
        if value <= 0.5:
            return ["sustained low grip utilisation"]
    if column in {"Physics_brake", "expert_optimal_brake"}:
        return ["peak brake pressure"]
    if column in {"Physics_gas", "expert_optimal_throttle"}:
        return ["peak throttle pressure"]
    if column in {"Physics_gear", "expert_optimal_gear"}:
        return ["gear selection"]
    return []


def _slope_tags(column: str, extra: Dict[str, Any]) -> List[str]:
    domain = str(extra.get("total_change_domain_direction") or "")
    tags: List[str] = []
    if column == "trajectory_offset":
        if domain == "moving_wider":
            tags.extend([
                "moving toward positive",
                "widening",
                "trajectory moving wider",
            ])
        elif domain == "moving_tighter":
            tags.extend([
                "moving toward negative",
                "tightening",
                "trajectory moving tighter",
            ])
    elif column == "speed_difference":
        if domain == "speed_gap_decreasing":
            tags.append("speed gap closing")
        elif domain == "speed_gap_increasing":
            tags.append("speed gap growing")
    elif column in {"Physics_speed_kmh", "expert_optimal_speed"}:
        if domain == "rising":
            tags.append("acceleration onset")
        elif domain == "falling":
            tags.append("deceleration onset")
    return tags


def _trajectory_similarity_tags(extra: Dict[str, Any]) -> List[str]:
    separation_gain = extra.get("line_separation_gain_m")
    mean_separation = extra.get("mean_line_separation_m")
    tags = [
        "trajectory similarity",
        "driver expert path comparison",
        "driver path separates from expert line",
    ]
    if isinstance(separation_gain, (int, float)) and float(separation_gain) > 0.5:
        tags.extend(["line separation increasing", "trajectory divergence"])
    if isinstance(mean_separation, (int, float)) and float(mean_separation) <= 0.5:
        tags.append("driver path closely follows expert")
    return tags


def _threshold_crossing_tags(samples: Any) -> List[str]:
    if not isinstance(samples, list):
        return []
    by_column = {
        str(sample.get("column") or ""): sample
        for sample in samples
        if isinstance(sample, dict)
    }
    pairs = [
        (
            "expert_optimal_brake",
            "Physics_brake",
            "brake initiation onset",
        ),
        (
            "expert_optimal_throttle",
            "Physics_gas",
            "throttle application onset",
        ),
    ]
    tags: List[str] = []
    for expert_col, player_col, phrase in pairs:
        expert = by_column.get(expert_col)
        player = by_column.get(player_col)
        if not expert or not player:
            continue
        expert_iloc = expert.get("iloc")
        player_iloc = player.get("iloc")
        if not isinstance(expert_iloc, int) or not isinstance(player_iloc, int):
            continue
        if player_iloc < expert_iloc:
            tags.append(f"{phrase} earlier than expert")
        elif player_iloc > expert_iloc:
            tags.append(f"{phrase} later than expert")
        else:
            tags.append(f"{phrase} aligned with expert")
    return tags


def _preflight_query_table(table, start: int, end: int):
    """Repair reset-index segment tables before automatic preflight queries."""
    if table is None or getattr(table, "empty", False):
        return table
    query_start = int(start)
    query_end = int(end)
    try:
        idx_min = int(table.index.min())
        idx_max = int(table.index.max())
    except (TypeError, ValueError):
        return table
    if idx_max >= query_start and idx_min <= query_end:
        return table

    length = len(table)
    expected = max(0, query_end - query_start)
    if length not in {expected, expected + 1}:
        return table
    try:
        first = int(table.index[0])
        last = int(table.index[-1])
    except (TypeError, ValueError):
        return table
    if first != 0 or last != length - 1:
        return table

    repaired = table.copy()
    repaired.index = range(query_start, query_start + length)
    return repaired


def _preflight_query_range(table, start: int, end: int) -> Optional[List[int]]:
    if table is None or getattr(table, "empty", False):
        return None
    query_start = int(start)
    query_end = int(end)
    try:
        table_start = int(table.index.min())
        table_end = int(table.index.max())
    except (TypeError, ValueError):
        return [query_start, query_end]
    clipped_start = max(query_start, table_start)
    clipped_end = min(query_end, table_end)
    if clipped_end < clipped_start:
        return None
    return [clipped_start, clipped_end]


def _label_candidates(
    evidence: str,
    *,
    parent_main_labels: List[str],
    eligible_behavior_label_ids: List[str],
) -> List[Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}

    def add(docs: Iterable[Dict[str, Any]]) -> None:
        for doc in docs:
            if not _allowed(doc, eligible_behavior_label_ids):
                continue
            shaped = shape_label_doc_for_llm(doc)
            current = merged.get(shaped["id"])
            if current is None or shaped.get("score", 0.0) > current.get("score", 0.0):
                merged[shaped["id"]] = shaped

    main_parents = [
        label_id
        for label_id in parent_main_labels
        if (get_doc(label_id) or {}).get("type") == "main"
    ]
    add(search(evidence, filters={"type": "segment_type"}, top_k=12))
    if main_parents:
        for parent_id in main_parents:
            add(search(evidence, filters={"parent": parent_id}, top_k=12))
    else:
        add(search(evidence, filters={"type": "main"}, top_k=12))
        for parent_id in eligible_behavior_label_ids:
            add(search(evidence, filters={"parent": parent_id}, top_k=4))

    return sorted(
        merged.values(),
        key=lambda item: float(item.get("score", 0.0)),
        reverse=True,
    )[:16]


def _allowed(doc: Dict[str, Any], eligible: List[str]) -> bool:
    if not eligible:
        return True
    label_id = str(doc.get("id") or "")
    parent_id = str(doc.get("parent") or "")
    return not (
        (label_id in _REQUIRED_PARENTS and label_id not in eligible)
        or (parent_id in _REQUIRED_PARENTS and parent_id not in eligible)
    )


def _evidence_text(
    *,
    flow: str,
    start: int,
    end: int,
    tool_outputs: List[Tuple[str, Dict[str, Any]]],
    tags: List[str],
    semantic_summaries: List[str],
    parent_main_labels: List[str],
    eligible_behavior_label_ids: List[str],
    fixed_label_ids: List[str],
    extra_query_terms: List[str],
) -> str:
    parts = [
        f"flow={flow}",
        f"range=[{start},{end}]",
        "tool_output_tags: " + ", ".join(tags),
        "semantic_summaries: " + " | ".join(semantic_summaries),
        "parent_main_labels: " + _label_text(parent_main_labels),
        "eligible_behavior_labels: " + _label_text(eligible_behavior_label_ids),
        "fixed_labels: " + _label_text(fixed_label_ids),
        "extra_terms: " + " ".join(str(term) for term in extra_query_terms),
        "tool_results_json: " + json.dumps(
            _semantic_tool_outputs(tool_outputs),
            default=str,
        ),
    ]
    return "\n".join(part for part in parts if not part.endswith(": "))[:12000]


def _prompt_block(
    flow: str,
    start: int,
    end: int,
    tool_outputs: List[Tuple[str, Dict[str, Any]]],
    tags: List[str],
    candidates: List[Dict[str, Any]],
    semantic_summaries: Optional[List[str]] = None,
) -> str:
    semantic_summaries = list(semantic_summaries or [])
    lines = [
        "#### Required Upfront Annotation Preflight",
        "The system already ran the required deterministic tool group before this AI step.",
        f"Flow: {flow}",
        f"Range: [{start}, {end}]",
        "Use these tool results and semantic label candidates as the primary analysis context. "
        "Call tools only to resolve a specific missing detail.",
        "",
        "Tool output tags:",
        ", ".join(tags) if tags else "(none)",
        "",
        "Semantic summaries:",
    ]
    if semantic_summaries:
        lines.extend(f"- {summary}" for summary in semantic_summaries)
    else:
        lines.append("- (none)")
    lines.extend([
        "",
        "Semantic label candidates from hybrid search:",
    ])
    lines.extend(_candidate_lines(candidates))
    lines.extend(["", "Required tool outputs:"])
    for tool_id, content in tool_outputs:
        summary = _preflight_tool_summary(tool_id, content)
        if summary:
            lines.append(summary)
        display_content = _semantic_tool_output(tool_id, content)
        lines.append(f"##### {tool_id}\n```json\n{_json(display_content, 2200)}\n```")
    return "\n".join(lines)


def _preflight_semantic_summaries(
    tool_outputs: List[Tuple[str, Dict[str, Any]]],
) -> List[str]:
    summaries = [
        *_preflight_pair_summaries(tool_outputs),
        *[
            summary
            for tool_id, content in tool_outputs
            for summary in [_preflight_tool_summary(tool_id, content)]
            if summary
        ],
    ]
    return _dedupe(summaries)[:40]


def _preflight_pair_summaries(
    tool_outputs: List[Tuple[str, Dict[str, Any]]],
) -> List[str]:
    by_tool = {tool_id: content for tool_id, content in tool_outputs}
    pairs = [
        (
            "brake peak comparison",
            "query_telemetry.find_extremum.brake.player.max",
            "query_telemetry.find_extremum.brake.expert.max",
            "player peak brake pressure",
            "expert peak brake pressure",
        ),
        (
            "throttle peak comparison",
            "query_telemetry.find_extremum.throttle.player.max",
            "query_telemetry.find_extremum.throttle.expert.max",
            "player peak throttle pressure",
            "expert peak throttle pressure",
        ),
    ]
    out: List[str] = []
    for label, player_tool, expert_tool, player_phrase, expert_phrase in pairs:
        player = _query_result(by_tool.get(player_tool))
        expert = _query_result(by_tool.get(expert_tool))
        if not player or not expert:
            continue
        player_value = player.get("value")
        expert_value = expert.get("value")
        if not isinstance(player_value, (int, float)) or not isinstance(
            expert_value, (int, float)
        ):
            continue
        delta = float(player_value) - float(expert_value)
        relation = (
            "higher than" if delta > 0
            else "lower than" if delta < 0
            else "aligned with"
        )
        out.append(
            f"{label}: {player_phrase}={player_value} at {player.get('iloc')}; "
            f"{expert_phrase}={expert_value} at {expert.get('iloc')}; "
            f"player is {relation} expert by {abs(delta):.3g}"
        )
    return out


def _query_result(content: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not isinstance(content, dict):
        return None
    result = content.get("result")
    return result if isinstance(result, dict) else None


def _preflight_tool_summary(tool_id: str, content: Dict[str, Any]) -> Optional[str]:
    if not tool_id.startswith("query_telemetry."):
        return _preflight_named_tool_summary(tool_id, content)
    if content.get("error"):
        return f"{tool_id}: unavailable ({content.get('error')})"
    result = content.get("result")
    if not isinstance(result, dict):
        return None
    query_id = str(content.get("query_id") or "")
    params = content.get("params") if isinstance(content.get("params"), dict) else {}
    column = str(params.get("column") or "")
    if query_id == "find_trend_runs":
        return _preflight_trend_runs_summary(tool_id, result, column)
    if query_id == "compute_slope":
        return _preflight_slope_summary(tool_id, result, column)
    if query_id == "find_extremum":
        return _preflight_extremum_summary(tool_id, result, column, params)
    if query_id == "find_threshold_crossing":
        return _preflight_threshold_summary(tool_id, result)
    if query_id == "find_dips_on_main_slope":
        return _preflight_dips_summary(tool_id, result, column)
    if query_id == "measure_trajectory_similarity":
        return _preflight_trajectory_similarity_summary(tool_id, result)
    return None


def _preflight_named_tool_summary(
    tool_id: str,
    content: Dict[str, Any],
) -> Optional[str]:
    if tool_id == "compute_expert_phases":
        phases = content.get("phases")
        if isinstance(phases, list) and phases:
            spans = [
                f"{p.get('entry')}->{p.get('apex')}->{p.get('exit')}"
                for p in phases[:4]
                if isinstance(p, dict)
            ]
            return "expert phases: " + "; ".join(spans)
        return "expert phases: no corner arc detected"
    if tool_id == "measure_segment_shape":
        base = content.get("base_segment_shape")
        if isinstance(base, dict):
            role = base.get("segment_type_role")
            shape_key = base.get("shape_key")
            reason = base.get("reason")
            return (
                f"segment shape: role={role}; shape_key={shape_key}; "
                f"reason={reason}"
            )
    if tool_id == "locate_circuit_section":
        best = content.get("best_match")
        if isinstance(best, dict):
            return (
                "circuit section: "
                f"{best.get('label_id')} {best.get('name')} "
                f"(overlap={best.get('overlap_fraction')})"
            )
        if content.get("is_ambiguous"):
            return "circuit section: ambiguous; inspect top_matches"
    if tool_id == "classify_opponent_interaction":
        return (
            "opponent interaction: "
            f"outcome={content.get('outcome')}; "
            f"confidence={content.get('confidence_level')}; "
            f"primary_slot={content.get('primary_slot_for_role')}"
        )
    if tool_id == "find_nearest_opponent":
        slot = content.get("slot") or content.get("nearest_slot")
        distance = content.get("min_distance_m")
        iloc = content.get("min_distance_iloc")
        if slot is not None or distance is not None:
            return (
                "nearest opponent: "
                f"slot={slot}; min_distance_m={distance}; iloc={iloc}"
            )
    return None


def _preflight_trend_runs_summary(
    tool_id: str,
    result: Dict[str, Any],
    column: str,
) -> Optional[str]:
    extra = result.get("extra")
    if not isinstance(extra, dict):
        return None
    if column != "expert_time_difference":
        return _preflight_generic_trend_runs_summary(tool_id, extra)
    unit = extra.get("unit")
    selected_gap_increase, selected_gap_decrease = _time_delta_selected_runs(extra)
    parts = [
        f"{tool_id}: verdict={_time_delta_trend_verdict(extra)}",
    ]
    if isinstance(selected_gap_increase, dict):
        parts.append(
            "selected_gap_increase_run="
            f"{selected_gap_increase.get('start_iloc')}->"
            f"{selected_gap_increase.get('end_iloc')} "
            f"delta={selected_gap_increase.get('gap_change')} {unit}"
        )
    if isinstance(selected_gap_decrease, dict):
        parts.append(
            "selected_gap_decrease_run="
            f"{selected_gap_decrease.get('start_iloc')}->"
            f"{selected_gap_decrease.get('end_iloc')} "
            f"delta={selected_gap_decrease.get('gap_change')} {unit}"
        )
    if len(parts) == 1:
        parts.append("no trend run")
    return "; ".join(parts)


def _preflight_generic_trend_runs_summary(
    tool_id: str,
    extra: Dict[str, Any],
) -> Optional[str]:
    unit = extra.get("unit")
    analysis = _trend_run_analysis({"extra": extra})
    verdict = analysis.get("local_curve_verdict") or analysis.get("verdict")
    parts = [
        f"{tool_id}: verdict={verdict}",
    ]
    runs = analysis.get("runs")
    if isinstance(runs, list) and runs:
        run_parts = []
        for run in runs[:6]:
            if not isinstance(run, dict):
                continue
            run_parts.append(
                f"{run.get('start_iloc')}->{run.get('end_iloc')} "
                f"{run.get('direction')} delta={run.get('change')} {unit}"
            )
        if run_parts:
            parts.append("local_curve_runs=" + " | ".join(run_parts))
        if len(runs) > 6:
            parts.append(f"additional_runs={len(runs) - 6}")
    else:
        parts.append("no local curve run")
    selected = analysis.get("selected_run")
    if not isinstance(selected, dict):
        selected = analysis.get("selected_local_run")
    if isinstance(selected, dict):
        parts.append(
            "largest_local_change="
            f"{selected.get('start_iloc')}->{selected.get('end_iloc')} "
            f"delta={selected.get('change')} {unit}"
        )
    return "; ".join(parts)


def _preflight_extremum_summary(
    tool_id: str,
    result: Dict[str, Any],
    column: str,
    params: Dict[str, Any],
) -> Optional[str]:
    if not column:
        return None
    kind = params.get("kind")
    value = result.get("value")
    iloc = result.get("iloc")
    extra = result.get("extra") if isinstance(result.get("extra"), dict) else {}
    unit = extra.get("unit")
    return f"{tool_id}: {column} {kind}={value} {unit} at iloc={iloc}"


def _preflight_slope_summary(
    tool_id: str,
    result: Dict[str, Any],
    column: str,
) -> Optional[str]:
    extra = result.get("extra")
    if not isinstance(extra, dict):
        return None
    zero = extra.get("near_zero_summary")
    moves_toward_zero = (
        zero.get("moves_toward_zero") if isinstance(zero, dict) else None
    )
    if column == "expert_time_difference":
        return (
            f"{tool_id} slope verdict: "
            f"total_gap_change={extra.get('delta_value')} {extra.get('unit')}; "
            "total_gap_direction="
            f"{_time_delta_gap_direction(extra.get('total_change_direction'), extra.get('delta_value'))}; "
            "total_gap_threshold_state="
            f"{_time_delta_threshold_state(extra.get('total_change_is_label_significant'))}; "
            f"moves_toward_zero={moves_toward_zero}; "
            f"slope_shape={extra.get('slope_shape')}; "
            "do not decide mistake/recovery from the raw endpoint difference"
        )
    if column == "trajectory_offset":
        start_abs = zero.get("start_abs") if isinstance(zero, dict) else None
        end_abs = zero.get("end_abs") if isinstance(zero, dict) else None
        min_abs = zero.get("min_abs") if isinstance(zero, dict) else None
        expert_line_relation = (
            "converging_to_expert_line"
            if moves_toward_zero is True
            else "diverging_from_expert_line"
            if moves_toward_zero is False
            else "unknown"
        )
        return (
            f"{tool_id} slope verdict: "
            f"signed_total_change={extra.get('delta_value')} {extra.get('unit')}; "
            f"signed_side_direction={extra.get('total_change_domain_direction')}; "
            f"absolute_offset_start={start_abs} {extra.get('unit')}; "
            f"absolute_offset_end={end_abs} {extra.get('unit')}; "
            f"absolute_offset_min={min_abs} {extra.get('unit')}; "
            f"expert_line_relation={expert_line_relation}; "
            f"slope_shape={extra.get('slope_shape')}"
        )
    return (
        f"{tool_id} slope verdict: "
        f"total_change={extra.get('delta_value')} {extra.get('unit')}; "
        f"total_change_direction={extra.get('total_change_direction')}; "
        "total_change_domain_direction="
        f"{extra.get('total_change_domain_direction')}; "
        f"total_change_is_label_significant={extra.get('total_change_is_label_significant')}; "
        f"moves_toward_zero={moves_toward_zero}; "
        f"slope_shape={extra.get('slope_shape')}"
    )


def _preflight_threshold_summary(
    tool_id: str,
    result: Dict[str, Any],
) -> Optional[str]:
    samples = result.get("samples")
    if not isinstance(samples, list) or not samples:
        return None
    rows = [
        sample
        for sample in samples
        if isinstance(sample, dict)
    ]
    if not rows:
        return None
    with_iloc = [row for row in rows if row.get("iloc") is not None]
    parts = [
        f"{row.get('column')} crosses at {row.get('iloc')}"
        for row in with_iloc
    ]
    player_vs_expert = _threshold_player_vs_expert(rows)
    if player_vs_expert:
        parts.append(player_vs_expert)
    missing = [
        str(row.get("column"))
        for row in rows
        if row.get("iloc") is None
    ]
    if missing:
        parts.append("no crossing for " + ", ".join(missing))
    return f"{tool_id}: " + "; ".join(parts)


def _threshold_player_vs_expert(rows: List[Dict[str, Any]]) -> Optional[str]:
    by_column = {str(row.get("column") or ""): row for row in rows}
    pairs = [
        ("Physics_brake", "expert_optimal_brake", "player brake"),
        ("Physics_gas", "expert_optimal_throttle", "player throttle"),
    ]
    for player_col, expert_col, phrase in pairs:
        player = by_column.get(player_col)
        expert = by_column.get(expert_col)
        if not player or not expert:
            continue
        player_iloc = player.get("iloc")
        expert_iloc = expert.get("iloc")
        if not isinstance(player_iloc, int) or not isinstance(expert_iloc, int):
            continue
        delta = player_iloc - expert_iloc
        relation = (
            "earlier than" if delta < 0
            else "later than" if delta > 0
            else "aligned with"
        )
        return f"{phrase} crosses {relation} expert by {abs(delta)} ilocs"
    return None


def _preflight_dips_summary(
    tool_id: str,
    result: Dict[str, Any],
    column: str,
) -> Optional[str]:
    samples = result.get("samples")
    extra = result.get("extra") if isinstance(result.get("extra"), dict) else {}
    n_dips = extra.get("n_dips")
    if not isinstance(samples, list):
        return None
    if not samples:
        return f"{tool_id}: no {column} modulation dip detected"
    dips = [
        f"iloc={sample.get('iloc')} depth={sample.get('depth')}"
        for sample in samples[:4]
        if isinstance(sample, dict)
    ]
    return (
        f"{tool_id}: {n_dips} {column} modulation dip(s); "
        + "; ".join(dips)
    )


def _preflight_trajectory_similarity_summary(
    tool_id: str,
    result: Dict[str, Any],
) -> Optional[str]:
    extra = result.get("extra")
    if not isinstance(extra, dict):
        return None
    peak = extra.get("peak_line_separation")
    peak = peak if isinstance(peak, dict) else {}
    return (
        f"{tool_id}: similarity_score={extra.get('similarity_score')}; "
        f"line_separation_gain_m={extra.get('line_separation_gain_m')}; "
        f"peak_line_separation_m={peak.get('value_m')} at iloc={peak.get('iloc')}; "
        f"mean_line_separation_m={extra.get('mean_line_separation_m')}; "
        f"longest_widening_run_steps={extra.get('longest_widening_run_steps')}"
    )


def _candidate_lines(candidates: List[Dict[str, Any]]) -> List[str]:
    if not candidates:
        return ["- (no semantic candidates found)"]
    lines: List[str] = []
    for c in candidates:
        desc = str(c.get("description") or "").strip()
        if len(desc) > 220:
            desc = desc[:217] + "..."
        lines.append(
            f"- `{c['id']}` {c.get('name', '')} "
            f"(type={c.get('type')}, score={c.get('score')})"
            + (f": {desc}" if desc else "")
        )
    return lines


def _tags(value: Any) -> List[str]:
    out: List[str] = []

    def walk(node: Any, path: str = "") -> None:
        if isinstance(node, dict):
            for key, child in node.items():
                next_path = f"{path}.{key}" if path else str(key)
                if key in _TAG_KEYS:
                    out.extend(_tag_values(next_path, child))
                walk(child, next_path)
        elif isinstance(node, list):
            for child in node[:20]:
                walk(child, path)

    walk(value)
    return _dedupe(out)[:80]


def _semantic_tags(tool_id: str, content: Dict[str, Any]) -> List[str]:
    if not _is_query_output(content):
        return _tags(content)
    semantic_tags = content.get("semantic_tags")
    semantic_tags = semantic_tags if isinstance(semantic_tags, list) else []
    analysis = content.get("analysis")
    if not isinstance(analysis, dict):
        analysis = _query_analysis(content)
    return _dedupe([
        *(
            str(tag)
            for tag in semantic_tags
            if isinstance(tag, (str, int, float, bool))
        ),
        *_tags(analysis),
    ])[:80]


def _semantic_tool_outputs(
    tool_outputs: List[Tuple[str, Dict[str, Any]]],
) -> List[Tuple[str, Dict[str, Any]]]:
    return [
        (tool_id, _semantic_tool_output(tool_id, content))
        for tool_id, content in tool_outputs
    ]


def _semantic_tool_output(tool_id: str, content: Dict[str, Any]) -> Dict[str, Any]:
    if not _is_query_output(content):
        return content
    out: Dict[str, Any] = {
        key: content[key]
        for key in (
            "graph_id",
            "query_id",
            "params",
            "error",
            "semantic_target",
            "semantic_tags",
            "analysis",
        )
        if key in content
    }
    if "analysis" not in out:
        analysis = _query_analysis(content)
        if analysis:
            out["analysis"] = analysis
    return out


def _is_query_output(content: Dict[str, Any]) -> bool:
    return content.get("graph_id") is not None and content.get("query_id") is not None


def _query_analysis(content: Dict[str, Any]) -> Dict[str, Any]:
    result = content.get("result")
    if not isinstance(result, dict):
        return {}
    params = content.get("params") if isinstance(content.get("params"), dict) else {}
    column = str(params.get("column") or "")
    if column == "expert_time_difference":
        return _time_delta_analysis(content)

    query_id = str(content.get("query_id") or "")
    if query_id == "find_trend_runs":
        return _trend_run_analysis(result)
    if query_id == "compute_slope":
        return _slope_analysis(result, column)
    if query_id == "find_extremum":
        return _extremum_analysis(result, params)
    if query_id == "find_threshold_crossing":
        return _threshold_crossing_analysis(result)
    if query_id == "find_dips_on_main_slope":
        return _dips_analysis(result)
    if query_id == "measure_trajectory_similarity":
        return _trajectory_similarity_analysis(result)
    return _compact_query_result(result)


def _trend_run_analysis(result: Dict[str, Any]) -> Dict[str, Any]:
    extra = result.get("extra")
    if not isinstance(extra, dict):
        return {}
    runs = extra.get("significant_runs")
    all_runs = extra.get("runs")
    local_runs = [
        _generic_trend_run(run, extra.get("unit"))
        for run in all_runs
        if isinstance(run, dict)
    ] if isinstance(all_runs, list) else []
    local_runs = [run for run in local_runs if run]
    significant_runs = [
        _generic_trend_run(run, extra.get("unit"))
        for run in runs
        if isinstance(run, dict)
    ] if isinstance(runs, list) else []
    significant_runs = [run for run in significant_runs if run]
    analysis: Dict[str, Any] = {
        "verdict": _generic_trend_verdict(significant_runs),
        "unit": extra.get("unit"),
        "constant_offset_only": extra.get("constant_offset_only"),
    }
    if local_runs:
        analysis["local_curve_verdict"] = _generic_trend_verdict(local_runs)
        analysis["runs"] = local_runs
        analysis["selected_local_run"] = max(
            local_runs,
            key=lambda run: abs(float(run.get("change") or 0.0)),
        )
    if significant_runs:
        analysis["selected_run"] = max(
            significant_runs,
            key=lambda run: abs(float(run.get("change") or 0.0)),
        )
        analysis["significant_runs"] = significant_runs
    return analysis


def _generic_trend_verdict(runs: List[Dict[str, Any]]) -> str:
    directions = {
        str(run.get("direction"))
        for run in runs
        if run.get("direction") in {"rising", "falling"}
    }
    if directions == {"rising", "falling"}:
        return "mixed_rising_falling_runs"
    if directions == {"rising"}:
        return "rising_run"
    if directions == {"falling"}:
        return "falling_run"
    return "stable"


def _generic_trend_run(value: Dict[str, Any], unit: Any) -> Optional[Dict[str, Any]]:
    return {
        "start_iloc": value.get("start_iloc"),
        "end_iloc": value.get("end_iloc"),
        "start_value": value.get("start_value"),
        "end_value": value.get("end_value"),
        "change": value.get("delta_value"),
        "unit": unit,
        "slope": value.get("slope"),
        "direction": value.get("direction"),
        "domain_direction": value.get("domain_direction"),
        "is_label_significant": value.get("is_label_significant"),
    }


def _slope_analysis(result: Dict[str, Any], column: str = "") -> Dict[str, Any]:
    extra = result.get("extra")
    if not isinstance(extra, dict):
        return {}
    unit = extra.get("unit")
    zero = extra.get("near_zero_summary")
    analysis = {
        "unit": unit,
        "total_change": {
            "value": extra.get("delta_value"),
            "unit": unit,
            "direction": extra.get("total_change_direction"),
            "domain_direction": extra.get("total_change_domain_direction"),
            "is_label_significant": extra.get("total_change_is_label_significant"),
            "moves_toward_zero": (
                zero.get("moves_toward_zero") if isinstance(zero, dict) else None
            ),
        },
        "slope_shape": extra.get("slope_shape"),
    }
    if column == "trajectory_offset" and isinstance(zero, dict):
        analysis["absolute_offset"] = {
            "start": zero.get("start_abs"),
            "end": zero.get("end_abs"),
            "min": zero.get("min_abs"),
            "unit": unit,
            "moves_toward_expert_line": zero.get("moves_toward_zero"),
        }
    return analysis


def _extremum_analysis(
    result: Dict[str, Any],
    params: Dict[str, Any],
) -> Dict[str, Any]:
    extra = result.get("extra") if isinstance(result.get("extra"), dict) else {}
    analysis: Dict[str, Any] = {
        "column": params.get("column"),
        "kind": params.get("kind"),
        "iloc": result.get("iloc"),
        "value": result.get("value"),
        "unit": extra.get("unit"),
    }
    for key in ("abs_min", "abs_max", "abs_mean", "peak_abs_iloc", "peak_abs_value"):
        if key in extra:
            analysis[key] = extra[key]
    return analysis


def _threshold_crossing_analysis(result: Dict[str, Any]) -> Dict[str, Any]:
    samples = result.get("samples")
    if not isinstance(samples, list):
        return {}
    rows = [row for row in samples if isinstance(row, dict)]
    analysis: Dict[str, Any] = {"samples": rows}
    comparison = _threshold_player_vs_expert(rows)
    if comparison:
        analysis["player_vs_expert"] = comparison
    return analysis


def _dips_analysis(result: Dict[str, Any]) -> Dict[str, Any]:
    samples = result.get("samples")
    extra = result.get("extra") if isinstance(result.get("extra"), dict) else {}
    if not isinstance(samples, list):
        return {}
    return {
        "n_dips": extra.get("n_dips"),
        "slope_direction": extra.get("slope_direction"),
        "samples": [sample for sample in samples if isinstance(sample, dict)],
    }


def _trajectory_similarity_analysis(result: Dict[str, Any]) -> Dict[str, Any]:
    extra = result.get("extra")
    if not isinstance(extra, dict):
        return {}
    return {
        "similarity_score": extra.get("similarity_score"),
        "line_separation_start_m": extra.get("line_separation_start_m"),
        "line_separation_end_m": extra.get("line_separation_end_m"),
        "line_separation_gain_m": extra.get("line_separation_gain_m"),
        "mean_line_separation_m": extra.get("mean_line_separation_m"),
        "widening_fraction": extra.get("widening_fraction"),
        "longest_widening_run_steps": extra.get("longest_widening_run_steps"),
        "peak_line_separation": extra.get("peak_line_separation"),
    }


def _compact_query_result(result: Dict[str, Any]) -> Dict[str, Any]:
    return {
        key: result[key]
        for key in ("iloc", "value", "samples")
        if key in result
    }


def _time_delta_analysis(content: Dict[str, Any]) -> Dict[str, Any]:
    result = content.get("result")
    if not isinstance(result, dict):
        return {}
    extra = result.get("extra")
    if not isinstance(extra, dict):
        return {}

    query_id = str(content.get("query_id") or "")
    if query_id == "find_trend_runs":
        return _time_delta_trend_run_analysis(extra)
    if query_id == "compute_slope":
        return _time_delta_slope_analysis(extra)
    return {}


def _time_delta_selected_runs(
    extra: Dict[str, Any],
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    runs = extra.get("significant_runs")
    if not isinstance(runs, list):
        return None, None
    gap_runs = [
        run
        for run in (_time_delta_run(item, extra.get("unit")) for item in runs)
        if run
    ]
    increases = [
        run for run in gap_runs if run.get("gap_direction") == "time_gap_rising"
    ]
    decreases = [
        run for run in gap_runs if run.get("gap_direction") == "time_gap_falling"
    ]
    selected_increase = (
        max(increases, key=_time_delta_run_abs_change) if increases else None
    )
    selected_decrease = (
        max(decreases, key=_time_delta_run_abs_change) if decreases else None
    )
    return selected_increase, selected_decrease


def _time_delta_trend_verdict(extra: Dict[str, Any]) -> str:
    selected_increase, selected_decrease = _time_delta_selected_runs(extra)
    if selected_increase and selected_decrease:
        return "time_gap_rising_and_falling"
    if selected_increase:
        return "time_gap_rising"
    if selected_decrease:
        return "time_gap_falling"
    if extra.get("constant_offset_only") is True:
        return "constant_carried_time_gap"
    return "time_gap_stable"


def _time_delta_gap_direction(direction: Any, delta_value: Any = None) -> str:
    if direction == "rising":
        return "time_gap_rising"
    if direction == "falling":
        return "time_gap_falling"
    if direction == "stable":
        return "time_gap_stable"
    if isinstance(delta_value, (int, float)):
        if delta_value > 0:
            return "time_gap_rising"
        if delta_value < 0:
            return "time_gap_falling"
        return "time_gap_stable"
    return "time_gap_unknown"


def _time_delta_gap_tags(gap_direction: Any) -> List[str]:
    if gap_direction == "time_gap_rising":
        return ["time gap rising", "gap increasing"]
    if gap_direction == "time_gap_falling":
        return ["time gap falling", "gap decreasing"]
    if gap_direction == "time_gap_stable":
        return ["gap holds stable"]
    return []


def _time_delta_threshold_state(is_label_significant: Any) -> str:
    if is_label_significant is True:
        return "label_threshold_met"
    if is_label_significant is False:
        return "below_label_threshold"
    return "threshold_unknown"


def _time_delta_run_abs_change(run: Dict[str, Any]) -> float:
    value = run.get("gap_change")
    return abs(float(value)) if isinstance(value, (int, float)) else 0.0


def _time_delta_trend_run_analysis(extra: Dict[str, Any]) -> Dict[str, Any]:
    runs = extra.get("significant_runs")
    gap_runs = (
        [
            run
            for run in (_time_delta_run(item, extra.get("unit")) for item in runs)
            if run
        ]
        if isinstance(runs, list)
        else []
    )
    selected_gap_increase, selected_gap_decrease = _time_delta_selected_runs(extra)
    analysis: Dict[str, Any] = {
        "verdict": _time_delta_trend_verdict(extra),
        "unit": extra.get("unit"),
        "constant_carried_time_gap": extra.get("constant_offset_only"),
    }
    if selected_gap_increase:
        analysis["selected_gap_increase_run"] = selected_gap_increase
    if selected_gap_decrease:
        analysis["selected_gap_decrease_run"] = selected_gap_decrease
    if gap_runs:
        analysis["significant_gap_runs"] = gap_runs
    return analysis


def _time_delta_slope_analysis(extra: Dict[str, Any]) -> Dict[str, Any]:
    unit = extra.get("unit")
    zero = extra.get("near_zero_summary")
    return {
        "unit": unit,
        "total_gap_change": {
            "value": extra.get("delta_value"),
            "unit": unit,
            "gap_direction": _time_delta_gap_direction(
                extra.get("total_change_direction"),
                extra.get("delta_value"),
            ),
            "threshold_state": _time_delta_threshold_state(
                extra.get("total_change_is_label_significant")
            ),
            "moves_toward_zero": (
                zero.get("moves_toward_zero") if isinstance(zero, dict) else None
            ),
        },
        "slope_shape": extra.get("slope_shape"),
    }


def _time_delta_run(value: Any, unit: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(value, dict):
        return None
    return {
        "start_iloc": value.get("start_iloc"),
        "end_iloc": value.get("end_iloc"),
        "start_gap": value.get("start_value"),
        "end_gap": value.get("end_value"),
        "gap_change": value.get("delta_value"),
        "unit": unit,
        "slope": value.get("slope"),
        "gap_direction": _time_delta_gap_direction(
            value.get("direction"),
            value.get("delta_value"),
        ),
        "threshold_state": _time_delta_threshold_state(
            value.get("is_label_significant")
        ),
    }


def _tag_values(path: str, value: Any) -> List[str]:
    if isinstance(value, dict):
        return [
            f"{path}.{key}:{value[key]}"
            for key in ("label_id", "label_name", "shape_key", "outcome", "role")
            if value.get(key) is not None
        ]
    if isinstance(value, list):
        return [f"{path}:{item}" for item in value if isinstance(item, (str, int, float, bool))]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return [f"{path}:{value}"]
    return []


def _label_text(label_ids: Sequence[str]) -> str:
    return ", ".join(
        f"{label_id} {LABEL_MAPPING.get(label_id, '')}".strip()
        for label_id in label_ids
    )


def _json(value: Any, max_chars: int) -> str:
    text = json.dumps(value, indent=2, sort_keys=True, default=str)
    return text if len(text) <= max_chars else text[: max_chars - 3] + "..."


def _dedupe(values: Iterable[str]) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value).strip()
        if text and text not in seen:
            seen.add(text)
            out.append(text)
    return out
