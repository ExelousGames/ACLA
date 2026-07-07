"""Shared upfront analysis package for annotation flows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from app.internal_knowledge_base import skills
from app.internal_knowledge_base.label_search import get_doc, search
from app.local_annotation_agent.workflow.tools import shape_label_doc_for_llm
from app.shared.contracts import Attachment
from app.shared.labels import LABEL_CATEGORIES, LABEL_MAPPING


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
            "merge_to_expert_line": (
                "trajectory offset merges toward expert line",
                "trajectory offset recovery toward expert line",
            ),
            "move_away_from_expert_line": (
                "trajectory offset moves away from expert line",
                "driver path separates from expert line",
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
    candidate_label_ids: Optional[Sequence[str]] = None,
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
        candidate_label_ids=list(candidate_label_ids or []),
        extra_query_terms=list(extra_query_terms or []),
    )
    candidates = _label_candidates(
        evidence,
        candidate_label_ids=list(candidate_label_ids or []),
    )

    attachments = [
        Attachment(
            name="init.preflight_label_candidates",
            kind="structured",
            label="Preflight label candidates",
            content={
                "range": [s, e],
                "tool_output_tags": tags,
                "candidate_label_ids": list(candidate_label_ids or []),
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
                "candidate_label_ids": list(candidate_label_ids or []),
                "label_candidate_ids": [c["id"] for c in candidates],
                "semantic_evidence_text": evidence,
            },
            content_schema="annotation_preflight_context",
        ),
    ]

    return PreflightContext(
        prompt_block=_prompt_block(
            flow=flow,
            start=s,
            end=e,
            candidates=candidates,
            semantic_summaries=semantic_summaries,
            candidate_label_ids=list(candidate_label_ids or []),
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
        if column == "trajectory_offset" and isinstance(extra, dict):
            tags.extend(_trajectory_abs_offset_derivative_tags(extra))
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
                "gain trend",
                "rate of gaining time decreasing",
                "time gain decelerating",
            ]
        if gap_direction == "time_gap_falling":
            return ["losing time accelerating", "time gap falling faster"]
        return ["slope decreasing over section"]
    if shape == "slope_increasing_over_section":
        if gap_direction == "time_gap_rising":
            return ["gaining time accelerating", "time gap rising faster"]
        if gap_direction == "time_gap_falling":
            return ["loss rate decreasing", "time gap falling slower"]
        return ["slope increasing over section"]
    if shape == "slope_steady_over_section":
        return ["time gap slope steady over section"]
    if shape == "reversing_to_falling_within_section":
        return ["time gap reversing to loss within section"]
    if shape == "reversing_to_rising_within_section":
        return ["time gap reversing to gain within section"]
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


def _trajectory_abs_offset_derivative_tags(extra: Dict[str, Any]) -> List[str]:
    derivative = extra.get("absolute_offset_derivative")
    if not isinstance(derivative, dict):
        return []
    overall = derivative.get("overall")
    overall = overall if isinstance(overall, dict) else {}
    runs = derivative.get("runs")
    runs = runs if isinstance(runs, list) else []
    directions = {
        str(overall.get("direction") or ""),
        *(str(run.get("direction") or "") for run in runs if isinstance(run, dict)),
    }
    tags: List[str] = []
    if "merge_to_expert_line" in directions:
        tags.extend([
            "trajectory offset merges toward expert line",
            "trajectory offset recovery toward expert line",
        ])
    if "move_away_from_expert_line" in directions:
        tags.extend([
            "trajectory offset moves away from expert line",
            "driver path separates from expert line",
        ])
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
    candidate_label_ids: Sequence[str],
) -> List[Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}

    def add(docs: Iterable[Dict[str, Any]]) -> None:
        for doc in docs:
            shaped = shape_label_doc_for_llm(doc)
            current = merged.get(shaped["id"])
            if current is None or shaped.get("score", 0.0) > current.get("score", 0.0):
                merged[shaped["id"]] = shaped

    for label_id in _dedupe(candidate_label_ids):
        doc = get_doc(label_id)
        if _is_main_label(label_id):
            main_doc = doc or _synthetic_main_doc(label_id)
            if main_doc is not None:
                add([main_doc])
            add(search(
                evidence,
                filters={"parent": label_id},
                top_k=_sub_label_search_limit(label_id),
            ))
        elif _category_child_ids(label_id):
            child_ids = set(_category_child_ids(label_id))
            child_types = _category_child_types(child_ids)
            filters = {"type": child_types} if child_types else None
            add(
                doc
                for doc in search(
                    evidence,
                    filters=filters,
                    top_k=_sub_label_search_limit(label_id),
                )
                if str(doc.get("id") or "") in child_ids
            )
        elif doc is not None:
            add([doc])

    return _prune_exclusive_label_candidates(list(merged.values()))


def _prune_exclusive_label_candidates(
    candidates: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    indexed = list(enumerate(candidates))
    kept: List[Tuple[int, Dict[str, Any]]] = []
    for index, candidate in sorted(
        indexed,
        key=lambda item: (_candidate_score(item[1]), -item[0]),
        reverse=True,
    ):
        if any(_exclusive_conflict(candidate, current) for _, current in kept):
            continue
        kept.append((index, candidate))
    return [candidate for _, candidate in sorted(kept, key=lambda item: item[0])]


def _exclusive_conflict(left: Dict[str, Any], right: Dict[str, Any]) -> bool:
    left_id = str(left.get("id") or "").strip()
    right_id = str(right.get("id") or "").strip()
    if not left_id or not right_id:
        return False
    return (
        right_id in _exclusive_label_ids(left)
        or left_id in _exclusive_label_ids(right)
    )


def _exclusive_label_ids(candidate: Dict[str, Any]) -> set[str]:
    value = candidate.get("exclusive_with") or []
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, Iterable):
        return set()
    return {str(label_id).strip() for label_id in value if str(label_id).strip()}


def _candidate_score(candidate: Dict[str, Any]) -> float:
    try:
        return float(candidate.get("score", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _is_main_label(label_id: str) -> bool:
    return str(label_id or "").strip() in _main_label_ids()


def _synthetic_main_doc(label_id: str) -> Optional[Dict[str, Any]]:
    label_id = str(label_id or "").strip()
    if not label_id or not _is_main_label(label_id):
        return None
    return {
        "id": label_id,
        "name": LABEL_MAPPING.get(label_id, label_id),
        "type": "main",
        "score": 0.0,
    }


def _candidate_section_order(
    candidate_label_ids: Sequence[str],
    candidates: Sequence[Dict[str, Any]],
) -> List[str]:
    order: List[str] = []
    for label_id in _dedupe(candidate_label_ids):
        doc = get_doc(label_id)
        if _is_main_label(label_id):
            order.append(label_id)
        elif _category_child_ids(label_id):
            order.append(label_id)
        elif doc is not None and doc.get("parent"):
            order.append(str(doc["parent"]))
        else:
            order.append(label_id)
    category_lookup = _category_parent_lookup(candidate_label_ids)
    for candidate in candidates:
        group_id = _candidate_group_id(candidate, category_lookup)
        if group_id:
            order.append(group_id)
    return _dedupe(order)


def _sub_label_search_limit(parent_id: str) -> int:
    count = len(_category_child_ids(parent_id))
    if count <= 0:
        return 2
    return max(2, min(10, (count + 9) // 10))


def _main_label_ids() -> set[str]:
    return {
        str(label_id).strip()
        for label_id in LABEL_CATEGORIES.get("Main Labels", [])
        if str(label_id).strip()
    }


def _category_child_ids(category_id: str) -> List[str]:
    return [
        str(label_id).strip()
        for label_id in LABEL_CATEGORIES.get(str(category_id or "").strip(), [])
        if str(label_id).strip()
    ]


def _category_child_types(child_ids: Iterable[str]) -> List[str]:
    return _dedupe(
        str(doc.get("type") or "")
        for child_id in child_ids
        for doc in [get_doc(child_id)]
        if doc is not None and str(doc.get("type") or "")
    )


def _category_parent_lookup(candidate_label_ids: Sequence[str]) -> Dict[str, str]:
    lookup: Dict[str, str] = {}
    for category_id in _dedupe(candidate_label_ids):
        if _is_main_label(category_id):
            continue
        for child_id in _category_child_ids(category_id):
            lookup.setdefault(child_id, category_id)
    return lookup


def _candidate_group_id(
    candidate: Dict[str, Any],
    category_lookup: Dict[str, str],
) -> str:
    label_id = str(candidate.get("id") or "")
    if candidate.get("type") == "main":
        return label_id
    parent_id = str(candidate.get("parent") or "")
    return parent_id or category_lookup.get(label_id, "")


def _evidence_text(
    *,
    flow: str,
    start: int,
    end: int,
    tool_outputs: List[Tuple[str, Dict[str, Any]]],
    tags: List[str],
    semantic_summaries: List[str],
    parent_main_labels: List[str],
    candidate_label_ids: List[str],
    extra_query_terms: List[str],
) -> str:
    parts = [
        f"Flow: {flow}",
        f"Range: [{start}, {end}]",
        "Preflight fact sentences: " + " ".join(semantic_summaries),
        "Parent main labels: " + _label_text(parent_main_labels),
        "Candidate labels: " + _label_text(candidate_label_ids),
        "Extra terms: " + " ".join(str(term) for term in extra_query_terms),
    ]
    return "\n".join(part for part in parts if not part.endswith(": "))


def _prompt_block(
    flow: str,
    start: int,
    end: int,
    candidates: List[Dict[str, Any]],
    semantic_summaries: Optional[List[str]] = None,
    candidate_label_ids: Optional[Sequence[str]] = None,
) -> str:
    semantic_summaries = list(semantic_summaries or [])
    lines = [
        "#### Required Upfront Annotation Preflight",
        "The system already ran deterministic tools and converted their "
        "results into human-readable fact sentences.",
        "These preflight sentences do not identify labels. They only provide "
        "facts with indices and values when available. The label catalog is "
        "the only place that judges which label fits.",
        f"The {flow} range is [{start}, {end}].",
        "",
        "Preflight fact sentences:",
    ]
    if semantic_summaries:
        lines.extend(f"- {summary}" for summary in semantic_summaries)
    else:
        lines.append("- (none)")
    lines.extend([
        "",
        "Preflight label candidates from hybrid search:",
    ])
    lines.extend(_candidate_lines(candidates, candidate_label_ids or []))
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
            f"For {label}, the {player_phrase} was "
            f"{_measurement(player_value)} at iloc {player.get('iloc')}, "
            f"while the {expert_phrase} was {_measurement(expert_value)} "
            f"at iloc {expert.get('iloc')}; the player was {relation} "
            f"the expert by {_measurement(abs(delta))}."
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
                f"entry {p.get('entry')}, apex {p.get('apex')}, exit {p.get('exit')}"
                for p in phases[:4]
                if isinstance(p, dict)
            ]
            return "Expert phase markers were found: " + "; ".join(spans) + "."
        return "Expert phase detection found no corner arc in this range."
    if tool_id == "measure_segment_shape":
        base = content.get("base_segment_shape")
        if isinstance(base, dict):
            role = base.get("segment_type_role")
            shape_key = base.get("shape_key")
            reason = base.get("reason")
            return (
                "Segment shape was classified as "
                f"{_humanize_value(shape_key)} with role {_humanize_value(role)}"
                + (f" because {reason}." if reason else ".")
            )
    if tool_id == "locate_circuit_section":
        best = content.get("best_match")
        if isinstance(best, dict):
            return (
                "The best circuit-section match was "
                f"{best.get('label_id')} {best.get('name')} with overlap "
                f"{_measurement(best.get('overlap_fraction'))}."
            )
        if content.get("is_ambiguous"):
            top_matches = content.get("top_matches")
            if isinstance(top_matches, list) and top_matches:
                matches = []
                for match in top_matches[:3]:
                    if not isinstance(match, dict):
                        continue
                    label_id = match.get("label_id")
                    name = match.get("name")
                    overlap = _measurement(match.get("overlap_fraction"))
                    matches.append(
                        f"{label_id} {name} overlap {overlap}"
                    )
                if matches:
                    return (
                        "Circuit-section location was ambiguous; competing "
                        "top matches were " + "; ".join(matches) + "."
                    )
            return (
                "Circuit-section location was ambiguous; inspect the top "
                "matches if section choice matters."
            )
    if tool_id == "classify_opponent_interaction":
        return (
            "Opponent interaction was classified as "
            f"{_humanize_value(content.get('outcome'))} with "
            f"{_humanize_value(content.get('confidence_level'))} confidence; "
            f"the primary opponent slot was {content.get('primary_slot_for_role')}."
        )
    if tool_id == "find_nearest_opponent":
        slot = content.get("slot") or content.get("nearest_slot")
        distance = content.get("min_distance_m")
        iloc = content.get("min_distance_iloc")
        if slot is not None or distance is not None:
            return (
                f"The nearest opponent was slot {slot}, reaching "
                f"{_measurement(distance, 'm')} at iloc {iloc}."
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
        return _preflight_generic_trend_runs_summary(tool_id, extra, column)
    return None


def _preflight_generic_trend_runs_summary(
    tool_id: str,
    extra: Dict[str, Any],
    column: str,
) -> Optional[str]:
    unit = extra.get("unit")
    analysis = _trend_run_analysis({"extra": extra})
    verdict = analysis.get("local_curve_verdict") or analysis.get("verdict")
    parts = [
        f"The {_humanize_value(column)} trend verdict was {_humanize_value(verdict)}.",
    ]
    runs = analysis.get("runs")
    if isinstance(runs, list) and runs:
        run_parts = []
        for run in runs[:6]:
            if not isinstance(run, dict):
                continue
            run_parts.append(
                f"iloc {run.get('start_iloc')} to {run.get('end_iloc')} "
                f"{_humanize_value(run.get('direction'))} by "
                f"{_measurement(run.get('change'), unit)}"
            )
        if run_parts:
            parts.append("Local curve runs were " + "; ".join(run_parts) + ".")
        if len(runs) > 6:
            parts.append(f"{len(runs) - 6} additional local curve runs were omitted.")
    else:
        parts.append("No local curve run was detected.")
    selected = analysis.get("selected_run")
    if not isinstance(selected, dict):
        selected = analysis.get("selected_local_run")
    if isinstance(selected, dict):
        parts.append(
            "The largest local change spans iloc "
            f"{selected.get('start_iloc')} to {selected.get('end_iloc')} "
            f"and changes by {_measurement(selected.get('change'), unit)}."
        )
    return " ".join(parts)


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
    return (
        f"The {_humanize_value(column)} {kind} was "
        f"{_measurement(value, unit)} at iloc {iloc}."
    )


def _preflight_gap_slope_summary(
    column: str,
    extra: Dict[str, Any],
    moves_toward_zero: Any,
) -> str:
    if column == "expert_time_difference":
        return _preflight_time_gap_slope_summary(extra)

    subject = "time gap" if column == "expert_time_difference" else "speed gap"
    unit = extra.get("unit")
    start = extra.get("start_trend")
    overall = extra.get("overall_point_trend")
    overall = overall if isinstance(overall, dict) else {}
    overall_direction = overall.get("direction") or extra.get("total_change_direction")
    runs = extra.get("point_trend_runs")
    runs = runs if isinstance(runs, list) else []

    parts: List[str] = []
    if isinstance(start, dict):
        parts.append(
            f"{_capitalize_sentence(subject)} value starts "
            f"{_gap_value_phrase(start.get('direction'))} from index "
            f"{start.get('start_iloc')} to {start.get('end_iloc')} "
            f"({_measurement(start.get('delta_value'), unit)})."
        )
    else:
        parts.append(f"{_capitalize_sentence(subject)} value start is unknown.")

    parts.append(
        f"{_capitalize_sentence(subject)} value runs: increases "
        f"{_trend_ranges(runs, 'rising', unit)}; decreases "
        f"{_trend_ranges(runs, 'falling', unit)}."
    )
    parts.append(
        f"{_capitalize_sentence(subject)} value overall is {_gap_value_phrase(overall_direction)} "
        f"by {_measurement(extra.get('delta_value'), unit)} "
        f"(mean {_measurement(extra.get('slope'), extra.get('slope_unit'))}). "
        f"{_gap_rate_shape_sentence(column, extra.get('slope_shape'))}; "
        f"toward zero: {_yes_no_unknown(moves_toward_zero)}."
    )
    return " ".join(parts)


def _preflight_time_gap_slope_summary(extra: Dict[str, Any]) -> str:
    unit = extra.get("unit")
    slope_unit = extra.get("slope_unit")
    raw_runs = extra.get("point_trend_runs")
    runs = [
        run for run in raw_runs
        if isinstance(run, dict)
    ] if isinstance(raw_runs, list) else []
    if not runs:
        return "Time gap slope summary is unavailable because no local runs were detected."

    first = runs[0]
    last = runs[-1]
    start_iloc = first.get("start_iloc")
    start_value = first.get("start_value")
    start_slope = extra.get("start_slope", first.get("slope"))
    end_iloc = last.get("end_iloc")
    end_value = last.get("end_value")
    end_slope = extra.get("end_slope", last.get("slope"))

    reversal_pairs = _time_gap_reversal_pairs(runs)
    parts = [
        "Time gap starts at index "
        f"{start_iloc} with value {_measurement(start_value, unit)} "
        f"and starting slope {_measurement(start_slope, slope_unit)}.",
        "Time gap local curve changes: "
        + _time_gap_slope_change_ranges(
            runs,
            slope_unit,
            start_slope,
            end_slope,
            reversal_pairs,
        )
        + ".",
    ]
    reversal_summary = _time_gap_reversal_summary(runs, unit)
    if reversal_summary:
        parts.append(reversal_summary)
    parts.append(
        "Time gap ends at index "
        f"{end_iloc} with value {_measurement(end_value, unit)}, "
        f"{_value_comparison(end_value, start_value, unit)}. "
        "Ending slope is "
        f"{_measurement(end_slope, slope_unit)}, "
        f"{_slope_comparison(end_slope, start_slope, slope_unit)}."
    )
    return " ".join(parts)


def _time_gap_slope_change_ranges(
    runs: List[Dict[str, Any]],
    slope_unit: Any,
    start_slope: Any,
    end_slope: Any,
    suppressed_pairs: set[Tuple[int, int]],
) -> str:
    ranges: List[str] = []
    if len(runs) == 1:
        change = _time_gap_slope_change(start_slope, end_slope)
        if change:
            ranges.append(_time_gap_slope_change_phrase(
                change,
                runs[0].get("start_iloc"),
                runs[0].get("end_iloc"),
                start_slope,
                end_slope,
                slope_unit,
            ))
        return "; ".join(ranges) if ranges else "none"

    omitted = 0
    for index, (before, after) in enumerate(zip(runs, runs[1:])):
        if (index, index + 1) in suppressed_pairs:
            continue
        before_slope = before.get("slope")
        after_slope = after.get("slope")
        change = _time_gap_slope_change(before_slope, after_slope)
        if not change:
            continue
        if len(ranges) >= 6:
            omitted += 1
            continue
        ranges.append(_time_gap_slope_change_phrase(
            change,
            before.get("start_iloc"),
            after.get("end_iloc"),
            before_slope,
            after_slope,
            slope_unit,
        ))
    if omitted:
        ranges.append(f"{omitted} more")
    return "; ".join(ranges) if ranges else "none"


def _time_gap_slope_change(
    before_slope: Any,
    after_slope: Any,
) -> Optional[str]:
    before_number = _as_number(before_slope)
    after_number = _as_number(after_slope)
    if before_number is None or after_number is None:
        return None
    delta = after_number - before_number
    guard = max(max(abs(before_number), abs(after_number)) * 0.25, 1e-9)
    if delta > guard:
        return "raising"
    if delta < -guard:
        return "falling"
    return None


def _time_gap_slope_change_phrase(
    change: str,
    start_iloc: Any,
    end_iloc: Any,
    before_slope: Any,
    after_slope: Any,
    slope_unit: Any,
) -> str:
    before_number = _as_number(before_slope)
    after_number = _as_number(after_slope)
    delta = (
        after_number - before_number
        if before_number is not None and after_number is not None
        else None
    )
    return (
        f"{change} index {start_iloc} to {end_iloc} "
        f"(slope {_measurement(before_slope, slope_unit)} to "
        f"{_measurement(after_slope, slope_unit)}, "
        f"change {_measurement(delta, slope_unit)}, "
        f"rate change {_percent_change_measurement(after_slope, before_slope)})"
    )


def _time_gap_reversal_pairs(
    runs: List[Dict[str, Any]],
) -> set[Tuple[int, int]]:
    pairs: set[Tuple[int, int]] = set()
    for index, (before, after) in enumerate(zip(runs, runs[1:])):
        before_direction = before.get("direction")
        after_direction = after.get("direction")
        if (
            before_direction == "rising" and after_direction == "falling"
        ) or (
            before_direction == "falling" and after_direction == "rising"
        ):
            pairs.add((index, index + 1))
    return pairs


def _time_gap_reversal_summary(
    runs: List[Dict[str, Any]],
    unit: Any,
) -> str:
    events = []
    for before, after in zip(runs, runs[1:]):
        before_direction = before.get("direction")
        after_direction = after.get("direction")
        if before_direction == "rising" and after_direction == "falling":
            events.append(_time_gap_reversal_event("spike", before, after, unit))
        elif before_direction == "falling" and after_direction == "rising":
            events.append(_time_gap_reversal_event("dip", before, after, unit))
    if not events:
        return ""
    return "Time gap reversal events: " + "; ".join(events) + "."


def _time_gap_reversal_event(
    event_type: str,
    before: Dict[str, Any],
    after: Dict[str, Any],
    unit: Any,
) -> str:
    pivot_label = "peak" if event_type == "spike" else "trough"
    return (
        f"{event_type} index {before.get('start_iloc')} to {after.get('end_iloc')} "
        f"(start {_measurement(before.get('start_value'), unit)}, "
        f"{pivot_label} index {before.get('end_iloc')} value "
        f"{_measurement(before.get('end_value'), unit)}, "
        f"end {_measurement(after.get('end_value'), unit)})"
    )

def _value_comparison(end_value: Any, start_value: Any, unit: Any) -> str:
    return _percent_change_comparison(end_value, start_value, "starting value")


def _slope_comparison(end_slope: Any, start_slope: Any, slope_unit: Any) -> str:
    return _percent_change_comparison(end_slope, start_slope, "starting slope")


def _percent_change_comparison(end_value: Any, start_value: Any, subject: str) -> str:
    end_number = _as_number(end_value)
    start_number = _as_number(start_value)
    if end_number is None or start_number is None:
        return f"percentage change from the {subject} is unknown"
    percent_change = _percent_change(end_number, start_number)
    if percent_change is None:
        return f"percentage change from the {subject} is unknown"
    if percent_change > 0:
        return f"higher than the {subject} by {_format_percent(abs(percent_change))}"
    if percent_change < 0:
        return f"lower than the {subject} by {_format_percent(abs(percent_change))}"
    return f"equal to the {subject}"


def _percent_change_measurement(end_value: Any, start_value: Any) -> str:
    end_number = _as_number(end_value)
    start_number = _as_number(start_value)
    if end_number is None or start_number is None:
        return "unknown"
    percent_change = _percent_change(end_number, start_number)
    if percent_change is None:
        return "unknown"
    return _format_percent(percent_change)


def _percent_change(end_number: float, start_number: float) -> Optional[float]:
    if abs(start_number) <= 1e-9:
        return None
    return ((end_number - start_number) / abs(start_number)) * 100.0


def _format_percent(value: float) -> str:
    text = f"{float(value):.3f}".rstrip("0").rstrip(".")
    return f"{text}%"


def _as_number(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _capitalize_sentence(value: str) -> str:
    return value[:1].upper() + value[1:]


def _gap_value_phrase(direction: Any) -> str:
    if direction == "rising":
        return "increasing"
    if direction == "falling":
        return "decreasing"
    if direction in {"flat", "stable"}:
        return "stable"
    return "unknown"


def _gap_rate_shape_sentence(column: str, slope_shape: Any) -> str:
    subject = "Loss-rate shape" if column == "expert_time_difference" else "Speed-gap rate shape"
    if slope_shape == "slope_decreasing_over_section":
        phrase = "gap growth is slowing, which can still mean the gap value is increasing"
    elif slope_shape == "slope_increasing_over_section":
        phrase = "gap growth is accelerating"
    elif slope_shape == "slope_steady_over_section":
        phrase = "gap change rate is steady"
    elif slope_shape == "reversing_to_falling_within_section":
        phrase = "gap value reverses from increasing to decreasing within the section"
    elif slope_shape == "reversing_to_rising_within_section":
        phrase = "gap value reverses from decreasing to increasing within the section"
    else:
        phrase = _humanize_value(slope_shape)
    return f"{subject}: {phrase} (slope shape {_humanize_value(slope_shape)})"


def _trend_phrase(direction: Any) -> str:
    if direction == "rising":
        return "trending up"
    if direction == "falling":
        return "trending down"
    if direction in {"flat", "stable"}:
        return "stable"
    return "unknown"


def _trend_ranges(runs: List[Any], direction: str, unit: Any) -> str:
    selected = [
        run for run in runs
        if isinstance(run, dict) and run.get("direction") == direction
    ]
    if not selected:
        return "none"
    ranges = [
        (
            f"index {run.get('start_iloc')} to {run.get('end_iloc')} "
            f"({_measurement(run.get('delta_value'), unit)})"
        )
        for run in selected[:6]
    ]
    if len(selected) > 6:
        ranges.append(f"{len(selected) - 6} more")
    return "; ".join(ranges)


def _expert_line_trend_phrase(direction: Any) -> str:
    if direction == "merge_to_expert_line":
        return "merging toward the expert line"
    if direction == "move_away_from_expert_line":
        return "moving away from the expert line"
    if direction in {"flat", "stable"}:
        return "stable relative to the expert line"
    return "unknown"


def _trajectory_abs_offset_ranges(runs: List[Any], direction: str, unit: Any) -> str:
    selected = [
        run for run in runs
        if isinstance(run, dict) and run.get("direction") == direction
    ]
    if not selected:
        return "none"
    ranges = [
        (
            f"index {run.get('start_iloc')} to {run.get('end_iloc')} "
            f"({_measurement(run.get('delta_abs'), unit)})"
        )
        for run in selected[:6]
    ]
    if len(selected) > 6:
        ranges.append(f"{len(selected) - 6} more")
    return "; ".join(ranges)


def _trajectory_abs_offset_derivative_summary(extra: Dict[str, Any]) -> Optional[str]:
    derivative = extra.get("absolute_offset_derivative")
    if not isinstance(derivative, dict):
        return None
    overall = derivative.get("overall")
    overall = overall if isinstance(overall, dict) else {}
    runs = derivative.get("runs")
    runs = runs if isinstance(runs, list) else []
    unit = extra.get("unit")
    return (
        "Expert-line distance derivative: merge ranges "
        f"{_trajectory_abs_offset_ranges(runs, 'merge_to_expert_line', unit)}; "
        "move-away ranges "
        f"{_trajectory_abs_offset_ranges(runs, 'move_away_from_expert_line', unit)}; "
        "overall expert-line trend "
        f"{_expert_line_trend_phrase(overall.get('direction'))} with net "
        f"absolute-offset change {_measurement(overall.get('delta_abs'), unit)} "
        "and mean derivative "
        f"{_measurement(overall.get('mean_derivative'), extra.get('slope_unit'))}."
    )


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
        return _preflight_gap_slope_summary(column, extra, moves_toward_zero)
    if column == "speed_difference":
        return _preflight_gap_slope_summary(column, extra, moves_toward_zero)
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
        derivative = _trajectory_abs_offset_derivative_summary(extra)
        return (
            "The trajectory-offset slope shows a signed total change of "
            f"{_measurement(extra.get('delta_value'), extra.get('unit'))}; "
            "the side direction is "
            f"{_humanize_value(extra.get('total_change_domain_direction'))}; "
            "absolute offset starts at "
            f"{_measurement(start_abs, extra.get('unit'))}, ends at "
            f"{_measurement(end_abs, extra.get('unit'))}, and has a minimum of "
            f"{_measurement(min_abs, extra.get('unit'))}; "
            f"the expert-line relation is {_humanize_value(expert_line_relation)}; "
            f"the slope shape is {_humanize_value(extra.get('slope_shape'))}."
            + (f" {derivative}" if derivative else "")
        )
    return (
        f"The {_humanize_value(column)} slope shows a total change of "
        f"{_measurement(extra.get('delta_value'), extra.get('unit'))}; "
        f"the numeric direction is {_humanize_value(extra.get('total_change_direction'))}; "
        "the domain direction is "
        f"{_humanize_value(extra.get('total_change_domain_direction'))}; "
        "label-threshold significance is "
        f"{_yes_no_unknown(extra.get('total_change_is_label_significant'))}; "
        f"movement toward zero is {_yes_no_unknown(moves_toward_zero)}; "
        f"the slope shape is {_humanize_value(extra.get('slope_shape'))}."
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
    return "Threshold crossing evidence found that " + "; ".join(parts) + "."


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
        return f"No {_humanize_value(column)} modulation dip was detected."
    dips = [
        f"iloc {sample.get('iloc')} with depth {_measurement(sample.get('depth'))}"
        for sample in samples[:4]
        if isinstance(sample, dict)
    ]
    return (
        f"{n_dips} {_humanize_value(column)} modulation dip(s) were detected: "
        + "; ".join(dips)
        + "."
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
        "Trajectory similarity scored "
        f"{_measurement(extra.get('similarity_score'))}; line separation gained "
        f"{_measurement(extra.get('line_separation_gain_m'), 'm')}; peak line "
        f"separation was {_measurement(peak.get('value_m'), 'm')} at iloc "
        f"{peak.get('iloc')}; mean line separation was "
        f"{_measurement(extra.get('mean_line_separation_m'), 'm')}; the longest "
        f"widening run lasted {extra.get('longest_widening_run_steps')} steps."
    )


def _candidate_lines(
    candidates: List[Dict[str, Any]],
    candidate_label_ids: Sequence[str] = (),
) -> List[str]:
    if not candidates:
        return ["- (no semantic candidates found)"]
    section_order = _candidate_section_order(candidate_label_ids, candidates)
    if section_order:
        lines: List[str] = []
        by_group: Dict[str, List[Dict[str, Any]]] = {
            group: [] for group in section_order
        }
        other: List[Dict[str, Any]] = []
        category_lookup = _category_parent_lookup(candidate_label_ids)
        for candidate in candidates:
            group_id = _candidate_group_id(candidate, category_lookup)
            if group_id in by_group:
                by_group[group_id].append(candidate)
            else:
                other.append(candidate)
        for group_id in section_order:
            group_candidates = by_group.get(group_id) or []
            if not group_candidates:
                continue
            lines.append(_candidate_section_heading(group_id))
            for c in group_candidates:
                lines.append(_candidate_line(c))
        if other:
            lines.append("##### Other candidates")
            lines.extend(_candidate_line(c) for c in other)
        return lines

    lines: List[str] = []
    for c in candidates:
        lines.append(_candidate_line(c))
    return lines


def _candidate_section_heading(group_id: str) -> str:
    if _is_main_label(group_id):
        return f"##### Main label `{group_id}` {LABEL_MAPPING.get(group_id, '')}".rstrip()
    return f"##### {LABEL_MAPPING.get(group_id, group_id)}".rstrip()


def _candidate_line(candidate: Dict[str, Any]) -> str:
    desc = str(candidate.get("description") or "").strip()
    return (
        f"- `{candidate['id']}` {candidate.get('name', '')} "
        f"({_humanize_value(candidate.get('type'))})"
        + (f": {desc}" if desc else "")
    )


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
    point = _point_trend_analysis(extra, unit)
    if point:
        analysis["point_trend"] = point
    if column == "trajectory_offset" and isinstance(zero, dict):
        analysis["absolute_offset"] = {
            "start": zero.get("start_abs"),
            "end": zero.get("end_abs"),
            "min": zero.get("min_abs"),
            "unit": unit,
            "moves_toward_expert_line": zero.get("moves_toward_zero"),
        }
        derivative = _trajectory_abs_offset_derivative_analysis(extra, unit)
        if derivative:
            analysis["absolute_offset_derivative"] = derivative
    return analysis


def _trajectory_abs_offset_derivative_analysis(
    extra: Dict[str, Any],
    unit: Any,
) -> Dict[str, Any]:
    derivative = extra.get("absolute_offset_derivative")
    if not isinstance(derivative, dict):
        return {}
    overall = derivative.get("overall")
    overall = overall if isinstance(overall, dict) else {}
    runs = derivative.get("runs")
    runs = runs if isinstance(runs, list) else []
    shaped_runs = [
        _trajectory_abs_offset_run(run, unit)
        for run in runs
        if isinstance(run, dict)
    ]
    shaped_runs = [run for run in shaped_runs if run]
    return {
        "overall": {
            "direction": overall.get("direction"),
            "net_abs_change": overall.get("delta_abs"),
            "mean_derivative": overall.get("mean_derivative"),
            "unit": unit,
            "slope_unit": extra.get("slope_unit"),
            "is_label_significant": overall.get("is_label_significant"),
        },
        "merge_runs": [
            run for run in shaped_runs
            if run.get("direction") == "merge_to_expert_line"
        ],
        "move_away_runs": [
            run for run in shaped_runs
            if run.get("direction") == "move_away_from_expert_line"
        ],
        "runs": shaped_runs,
    }


def _trajectory_abs_offset_run(value: Dict[str, Any], unit: Any) -> Optional[Dict[str, Any]]:
    return {
        "start_iloc": value.get("start_iloc"),
        "end_iloc": value.get("end_iloc"),
        "start_abs": value.get("start_abs"),
        "end_abs": value.get("end_abs"),
        "change": value.get("delta_abs"),
        "unit": unit,
        "derivative": value.get("derivative"),
        "direction": value.get("direction"),
        "is_label_significant": value.get("is_label_significant"),
    }


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
    analysis = {
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
    point = _point_trend_analysis(extra, unit)
    if point:
        analysis["point_trend"] = point
    return analysis


def _point_trend_analysis(extra: Dict[str, Any], unit: Any) -> Dict[str, Any]:
    runs = extra.get("point_trend_runs")
    point_runs = [
        _generic_trend_run(run, unit)
        for run in runs
        if isinstance(run, dict)
    ] if isinstance(runs, list) else []
    point_runs = [run for run in point_runs if run]
    overall = extra.get("overall_point_trend")
    overall = overall if isinstance(overall, dict) else {}
    analysis: Dict[str, Any] = {
        "overall": {
            "direction": overall.get("direction"),
            "domain_direction": overall.get("domain_direction"),
            "net_change": extra.get("delta_value"),
            "mean_slope": extra.get("slope"),
            "unit": unit,
            "slope_unit": extra.get("slope_unit"),
        },
        "step_counts": {
            "rising": extra.get("rising_steps"),
            "falling": extra.get("falling_steps"),
            "flat": extra.get("flat_steps"),
        },
    }
    start = extra.get("start_trend")
    if isinstance(start, dict):
        analysis["start_trend"] = _generic_trend_run(start, unit)
    if point_runs:
        analysis["runs"] = point_runs
    return analysis


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


def _measurement(value: Any, unit: Any = None) -> str:
    if isinstance(value, (int, float)):
        text = f"{float(value):.3f}".rstrip("0").rstrip(".")
    elif value is None:
        text = "unknown"
    else:
        text = str(value)
    unit_text = str(unit or "").strip()
    return f"{text} {unit_text}".strip()


def _humanize_value(value: Any) -> str:
    if value is None:
        return "unknown"
    return str(value).replace("_", " ")


def _yes_no_unknown(value: Any) -> str:
    if value is True:
        return "yes"
    if value is False:
        return "no"
    return "unknown"


def _dedupe(values: Iterable[str]) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value).strip()
        if text and text not in seen:
            seen.add(text)
            out.append(text)
    return out
