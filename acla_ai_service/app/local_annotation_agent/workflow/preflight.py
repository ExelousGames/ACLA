"""Required upfront analysis package for annotation flows."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from app.internal_knowledge_base.label_search import get_doc, search
from app.local_annotation_agent.workflow.tools import shape_label_doc_for_llm
from app.shared.contracts import Attachment
from app.shared.labels import LABEL_MAPPING


PREFLIGHT_TOOL_IDS: Tuple[str, ...] = (
    "get_circuit_id",
    "compute_expert_phases",
    "measure_segment_shape",
    "locate_circuit_section",
    "split_lap_by_circuit_sections",
    "classify_opponent_interaction",
    "find_nearest_opponent",
)
PREFLIGHT_QUERY_SPECS: Tuple[Dict[str, Any], ...] = (
    {
        "tool_id": "query_telemetry.find_trend_runs.expert_time_difference",
        "graph_id": "time_delta",
        "query_id": "find_trend_runs",
        "params": {
            "column": "expert_time_difference",
            "smoothing_window": 5,
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
)

_TAG_KEYS = {
    "annotation_scope",
    "constant_offset_only",
    "confidence_level",
    "data_available",
    "direction",
    "domain_direction",
    "end_change_direction",
    "end_change_domain_direction",
    "end_change_is_label_significant",
    "end_change_significance",
    "end_trend_change",
    "ends_near_zero",
    "is_ambiguous",
    "is_label_significant",
    "label_id",
    "label_name",
    "moves_toward_zero",
    "outcome",
    "parent",
    "recommended_label",
    "role",
    "segment_type_role",
    "shape_key",
    "significance",
    "starts_near_zero",
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


def build_preflight_context(
    *,
    flow: str,
    df,
    start: int,
    end: int,
    parent_main_labels: Optional[Sequence[str]] = None,
    eligible_behavior_label_ids: Optional[Sequence[str]] = None,
    fixed_label_ids: Optional[Sequence[str]] = None,
    extra_query_terms: Optional[Sequence[str]] = None,
) -> PreflightContext:
    s, e = int(start), int(end)
    if e <= s:
        raise RuntimeError(f"annotation preflight: invalid range [{s}, {e}]")

    tool_outputs = [
        *_run_tools(df, s, e),
        *_run_queries(df, s, e),
    ]
    tags = _dedupe(
        tag
        for tool_id, content in tool_outputs
        for tag in [f"tool:{tool_id}", *_tags(content)]
    )[:80]
    evidence = _evidence_text(
        flow=flow,
        start=s,
        end=e,
        tool_outputs=tool_outputs,
        tags=tags,
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
                "tags": _tags(content),
                "result": content,
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
                "required_tools": _preflight_analysis_ids(),
                "tool_output_tags": tags,
                "label_candidate_ids": [c["id"] for c in candidates],
                "semantic_evidence_text": evidence,
            },
            content_schema="annotation_preflight_context",
        ),
    ])

    return PreflightContext(
        prompt_block=_prompt_block(flow, s, e, tool_outputs, tags, candidates),
        attachments=attachments,
        label_candidates=candidates,
    )


def _preflight_analysis_ids() -> List[str]:
    return [
        *PREFLIGHT_TOOL_IDS,
        *(str(spec["tool_id"]) for spec in PREFLIGHT_QUERY_SPECS),
    ]


def _run_tools(df, start: int, end: int) -> List[Tuple[str, Dict[str, Any]]]:
    from app.shared.annotation_agent_tools import get_circuit_id, get_pipeline_tool

    out: List[Tuple[str, Dict[str, Any]]] = []
    circuit_id: Optional[str] = None
    for tool_id in PREFLIGHT_TOOL_IDS:
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


def _run_queries(df, start: int, end: int) -> List[Tuple[str, Dict[str, Any]]]:
    from app.shared.annotation_agent_tools import build_graph, run_pipeline_query

    out: List[Tuple[str, Dict[str, Any]]] = []
    for spec in PREFLIGHT_QUERY_SPECS:
        tool_id = str(spec["tool_id"])
        graph_id = str(spec["graph_id"])
        query_id = str(spec["query_id"])
        table = build_graph(graph_id, df)
        if table is None:
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
        }
        if error:
            content["error"] = error
        out.append((tool_id, content))
    return out


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
    parent_main_labels: List[str],
    eligible_behavior_label_ids: List[str],
    fixed_label_ids: List[str],
    extra_query_terms: List[str],
) -> str:
    parts = [
        f"flow={flow}",
        f"range=[{start},{end}]",
        "tool_output_tags: " + ", ".join(tags),
        "parent_main_labels: " + _label_text(parent_main_labels),
        "eligible_behavior_labels: " + _label_text(eligible_behavior_label_ids),
        "fixed_labels: " + _label_text(fixed_label_ids),
        "extra_terms: " + " ".join(str(term) for term in extra_query_terms),
        "tool_results_json: " + json.dumps(tool_outputs, default=str),
    ]
    return "\n".join(part for part in parts if not part.endswith(": "))[:12000]


def _prompt_block(
    flow: str,
    start: int,
    end: int,
    tool_outputs: List[Tuple[str, Dict[str, Any]]],
    tags: List[str],
    candidates: List[Dict[str, Any]],
) -> str:
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
        "Semantic label candidates from hybrid search:",
    ]
    lines.extend(_candidate_lines(candidates))
    lines.extend(["", "Required tool outputs:"])
    for tool_id, content in tool_outputs:
        lines.append(f"##### {tool_id}\n```json\n{_json(content, 2200)}\n```")
    return "\n".join(lines)


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


def _tag_values(path: str, value: Any) -> List[str]:
    if isinstance(value, dict):
        return [
            f"{path}.{key}:{value[key]}"
            for key in ("label_id", "label_name", "shape_key", "recommended_label", "outcome", "role")
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
