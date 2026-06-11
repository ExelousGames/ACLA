"""
Detailed (sub-segment discovery) flow.

Wraps the agent box for the "discover ONE notable sub-segment within a
parent segment" use case. Provides:

    build_request(provider_id, prompt_mode, df, range_, ...) -> AgentRequest
    parse(response, ...) -> AnnotationResult

The prompts and parsing here are racing-specific (parent_main_labels,
sub-label discovery, label_id JSON schema). The box never sees them
directly — it just executes the planner / synth / submit prompts the
caller supplies and returns raw text the parser decodes.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

from app.shared.labels import LABEL_MAPPING
from app.internal_knowledge_base.label_search import get_doc
from app.shared.contracts import (
    AgentCallbacks,
    AgentRequest,
    AgentResponse,
    Attachment,
    NoopCallbacks,
    ProviderConfig,
)
from app.local_annotation_agent.workflow.results import (
    AnnotationResult,
    parse_json_response,
)
from app.local_annotation_agent.workflow.tools import SEARCH_LABELS_TOOL


def _verified_label_ids_from_state(state: Dict[str, Any]) -> List[str]:
    """Pull verified label IDs out of the attachment pool.

    ``label_verifier`` emits ``step_solver.<step_id>.verified_labels``
    after renamespacing. The synth prompt callable reads them from the
    pool — the agent box's state schema is domain-free, so the IDs no
    longer ride as a named state field.
    """
    pool = state.get("attachment_pool", {}) or {}
    out: List[str] = []
    for name in sorted(pool.keys()):
        if not name.endswith(".verified_labels"):
            continue
        att = pool[name]
        content = getattr(att, "content", None)
        if not isinstance(content, list):
            continue
        for entry in content:
            if isinstance(entry, dict):
                lid = entry.get("label_id")
                if isinstance(lid, str):
                    out.append(lid)
    return out


LOGGER = logging.getLogger(__name__)


def _is_full_parent_range(
    start: int,
    end: int,
    parent_start: int,
    parent_end: int,
) -> bool:
    return int(start) == int(parent_start) and int(end) == int(parent_end)


# ---------------------------------------------------------------------------
# Local-backend planner prompt — JSON plan of describe_graphs + label_verifier
# ---------------------------------------------------------------------------


def _local_planner_prompt(
    *,
    parent_start: int,
    parent_end: int,
    parent_main_labels: List[str],
    existing_children: List[dict],
) -> str:
    """Planner prompt the local runner sends to the planner VLM."""
    from app.shared.annotation_agent_tools import (
        AGENT_GRAPH_DEFINITIONS,
        PIPELINE_TOOL_DEFINITIONS,
    )

    graph_catalogue = ", ".join(
        f"`{gdef['id']}` ({gdef['title']})"
        for gdef in AGENT_GRAPH_DEFINITIONS
    )
    tool_catalogue_lines = [
        f"- `{t['id']}` — {t['label']}: {t['description']}"
        for t in PIPELINE_TOOL_DEFINITIONS
    ]

    main_readable = [LABEL_MAPPING.get(l, l) for l in parent_main_labels]
    label_descriptions: List[str] = []
    annotation_guidelines: List[str] = []
    for label_id in parent_main_labels:
        label_def = get_doc(label_id)
        if not label_def:
            continue
        desc = label_def.get("description")
        if desc:
            label_descriptions.append(
                f"  - {label_def['name']} ({label_id}): {desc}"
            )
        guideline = label_def.get("annotation_guideline")
        if guideline:
            annotation_guidelines.append(
                f"  [{label_def['name']}]\n  {guideline}"
            )

    parts = [
        "You are a racing telemetry analyst planning a sub-segment "
        "discovery strategy.",
        "",
        "#### Parent Segment",
        f"Main labels: {', '.join(main_readable)} "
        f"(IDs: {json.dumps(parent_main_labels)})",
        f"Range: [{parent_start}, {parent_end}] "
        f"(length: {parent_end - parent_start} data points)",
        "",
    ]
    if label_descriptions:
        parts.append("#### Label Descriptions (from Label Catalog)")
        parts.extend(label_descriptions)
        parts.append("")
    if annotation_guidelines:
        parts.append("#### Annotation Guidelines")
        parts.append(
            "Follow these steps when planning analysis for the "
            "identified labels:"
        )
        parts.extend(annotation_guidelines)
        parts.append("")
    if existing_children:
        parts.append("#### Already Discovered Sub-Segments")
        for child in existing_children:
            child_labels = [
                LABEL_MAPPING.get(l, l) for l in child.get("labels", [])
            ]
            parts.append(
                f"- Range [{child['start_index']}, {child['end_index']}] "
                f"(length: {child['end_index'] - child['start_index']}): "
                f"{', '.join(child_labels)}"
            )
        parts.append("Find a DIFFERENT region that is not yet covered.")
        parts.append("")
    parts.extend([
        "#### Available Step-Solver Agents",
        "Each plan step is dispatched to ONE sub-agent.",
        "- `describe_graphs` — renders the telemetry graphs listed in "
        "`requested_graphs` and writes a precise observation paragraph "
        "per graph. Pure observation — does not diagnose or assign labels.",
        "- `label_verifier` — embedding-similarity filter over the parent's "
        "candidate labels, using the upstream describe_graphs observations "
        "as evidence. Conclude your plan with one label_verifier step so "
        "the synthesizer sees a verified shortlist.",
        "",
        "#### Available Graph IDs (for `describe_graphs` agent)",
        graph_catalogue,
        "",
        "#### Available Pre-Compute Tools",
        "Invoke per step via the `tools` field. Each invoked tool "
        "produces an attachment that will be attached to that step's "
        "prompt only.",
        *tool_catalogue_lines,
        "",
        "#### Task",
        "Plan analysis steps to help discover ONE notable sub-segment "
        "within the parent segment range. A sub-segment is a contiguous "
        "region where a specific event or behaviour occurs. End your "
        "plan with a single `label_verifier` step over the observations "
        "produced by the preceding describe_graphs steps.",
        "If the parent label or candidate behaviour involves racing, "
        "overtaking, defending, or any close opponent interaction, include "
        "`classify_opponent_interaction`; then use `find_nearest_opponent` "
        "or `query_opponent_trajectory` to identify the specific car slot. "
        "A car tucked directly behind or ahead can be the relevant target "
        "even when it never becomes side-by-side.",
        "",
        "Your plan must be a JSON object with a single key \"steps\". "
        "Each step object must have:",
        "  - \"step_id\": integer (1, 2, 3, ...).",
        "  - \"agent\": one of `describe_graphs` or `label_verifier`.",
        "  - \"description\": string describing the goal of the step.",
        "  - \"requested_graphs\": list of graph IDs (describe_graphs only; "
        "use `[]` for label_verifier).",
        "  - \"tools\": list of pre-compute tool IDs (empty list `[]` for none).",
        "",
        "Example:",
        "```json",
        "{",
        '  "steps": [',
        '    {"step_id": 1, "agent": "describe_graphs", "description": '
        '"Measure entry/apex/exit shape via the trajectory offset trace.", '
        '"requested_graphs": ["trajectory_offset"], "tools": '
        '["compute_expert_phases"]},',
        '    {"step_id": 2, "agent": "describe_graphs", "description": '
        '"Inspect speed and throttle around the apex.", '
        '"requested_graphs": ["speed", "throttle"], "tools": []},',
        '    {"step_id": 3, "agent": "label_verifier", "description": '
        '"Filter candidate labels by embedding similarity to the step observations.", '
        '"requested_graphs": [], "tools": []}',
        "  ]",
        "}",
        "```",
    ])
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Local-backend synthesizer prompt
# ---------------------------------------------------------------------------


def _local_synth_prompts(
    *,
    parent_start: int,
    parent_end: int,
    verified_labels: List[str],
) -> Tuple[str, str]:
    """Build (intro, outro) read by the synth_prompt callable."""
    n_verified = len(verified_labels)
    verified_ids_inline = (
        ", ".join(verified_labels) if verified_labels else "(none)"
    )

    intro = "\n".join([
        "You are a racing telemetry analyst producing label annotations.",
        "",
        "#### Task",
        f"For the parent range [{parent_start}, {parent_end}] "
        f"(length: {parent_end - parent_start} data points), judge "
        "every verified candidate label against the step observations. "
        "For each candidate that is proved, pinpoint its exact "
        "start_index and end_index as a strict child range inside the "
        "parent range.",
    ])

    outro = "\n".join([
        "#### Instructions",
        f"- There are {n_verified} verified candidate label(s): "
        f"{verified_ids_inline}.",
        f"- Emit EXACTLY one entry per verified candidate "
        f"({n_verified} entries total) — neither more nor fewer. Do not "
        "invent labels outside the verified list.",
        "- Judging rule: treat each label's full description as the "
        "definition. The bolded predicate is the headline claim; the "
        "surrounding prose lists its qualifiers. Set \"proved\": true "
        "only if the step observations satisfy the predicate AND every "
        "qualifier the description names. Otherwise set \"proved\": false.",
        f"- start_index and end_index are required only when \"proved\" "
        f"is true; each must satisfy {parent_start} <= start_index < "
        f"end_index <= {parent_end} and the range must contain the cited "
        "evidence. The range must not be identical to the parent range "
        f"[{parent_start}, {parent_end}]. If evidence only supports the "
        "whole parent, set \"proved\": false and explain that no strict "
        "child sub-segment was found. Omit both fields when \"proved\" is "
        "false.",
        "- In \"reasoning\", cite the step observation sentences that "
        "establish (or fail to establish) the predicate + qualifiers. "
        "When \"proved\" is false, name the specific qualifier that is "
        "unmet or contradicted.",
        "- The top-level JSON has a single key \"labels\" whose value "
        "is a flat list of entries. Do not wrap the list in any other "
        "container.",
        "",
        "#### Output Format",
        "Respond with JSON of this exact shape only. Emit one entry per "
        f"verified candidate ({n_verified} entries total). Output strict "
        "JSON only — no comments, no trailing commas, no extra keys.",
        "```json",
        "{",
        '  "labels": [',
        "    {",
        '      "label_id": "<one of the verified label IDs>",',
        '      "proved": true,',
        f'      "start_index": <integer in [{parent_start}, {parent_end}]>,',
        f'      "end_index": <integer in [{parent_start}, {parent_end}]>,',
        '      "reasoning": "..."',
        "    },",
        "    {",
        '      "label_id": "<one of the verified label IDs>",',
        '      "proved": false,',
        '      "reasoning": "..."',
        "    }",
        "  ]",
        "}",
        "```",
    ])

    return intro, outro


# ---------------------------------------------------------------------------
# Tool-agent planner prompt — task description + submit schema
# ---------------------------------------------------------------------------


def _tool_agent_task_prompt(
    *,
    parent_start: int,
    parent_end: int,
    parent_main_labels: List[str],
    existing_children: List[dict],
) -> str:
    """User-message prompt the tool-agent runner sends as the session start."""

    parent_label_blocks = [
        f"  - `{pid}` ({LABEL_MAPPING.get(pid, pid)})"
        for pid in parent_main_labels
    ]

    existing_block = ""
    if existing_children:
        lines = []
        for c in existing_children:
            names = ", ".join(LABEL_MAPPING.get(l, l) for l in c.get("labels", []))
            lines.append(
                f"  - [{c['start_index']}, {c['end_index']}] — {names}"
            )
        existing_block = (
            "\n### Already discovered sub-segments (do NOT re-propose)\n"
            + "\n".join(lines) + "\n"
        )

    return (
        "Discover the most notable sub-segment(s) within the parent "
        "segment below and submit them via `submit_result`. A valid "
        "sub-segment is a strict child range: it may touch one parent "
        "boundary, but it must not be identical to the parent range.\n"
        "\n"
        "### Parent segment\n"
        f"- index range: [{parent_start}, {parent_end}] "
        f"(length {parent_end - parent_start})\n"
        "- parent main label(s):\n"
        + ("\n".join(parent_label_blocks) or "  (none)")
        + "\n"
        f"{existing_block}"
        "\n"
        "### How to work\n"
        "1. Call `search_annotation_guidance` for the parent label behavior "
        "and sub-segment discovery rules that match this range.\n"
        "2. Call `recommend_tools` with the evidence you need to gather "
        "(graphs, exact ilocs/values, corner phases, circuit-section "
        "context, opponent interaction). Execute selected capability IDs "
        "with `run_annotation_tool`.\n"
        "3. Discover candidate labels with `search_labels`. Query by "
        "plain-language observations, use `parent_id` for sub-labels under "
        "the relevant parent label, and use `types=\"segment_type\"` for "
        "segment-shape labels.\n"
        "4. Submit via `submit_result(payload_json, summary)` when evidence "
        "is sufficient, then stop after it returns `ok: true`. If the "
        "evidence only supports the whole parent range, submit an empty "
        "`proposals` list and say no strict child sub-segment was found.\n"
        "\n"
        "### Submit payload shape\n"
        "`payload_json` must be a JSON object of this shape:\n"
        "```json\n"
        "{\n"
        '  "proposals": [\n'
        '    {\n'
        '      "label_id": "<a label_id returned by search_labels>",\n'
        f'      "start_index": <int in [{parent_start}, {parent_end}]>,\n'
        f'      "end_index": <int in [{parent_start}, {parent_end}]>,\n'
        '      "reasoning": "<cite ilocs and values>"\n'
        '    }\n'
        "  ]\n"
        "}\n"
        "```\n"
        "Use `{\"proposals\": []}` when no strict child range is supported.\n"
        "\n"
        "### Hard rules\n"
        f"- Every proposed range must satisfy {parent_start} <= start_index < end_index <= {parent_end}.\n"
        f"- A proposed range must not be identical to the parent range [{parent_start}, {parent_end}].\n"
        "- Parent labels are inherited context only; they are not enough evidence for a child proposal.\n"
        "- Only propose label_ids returned by `search_labels`.\n"
        "- Do not propose ranges that exactly match an already-discovered sub-segment.\n"
        "- After `submit_result` returns `ok: true`, stop calling tools."
    )


# ---------------------------------------------------------------------------
# Public flow API
# ---------------------------------------------------------------------------


def build_request(
    *,
    provider_id: str,
    prompt_mode: str,
    df,
    parent_start: int,
    parent_end: int,
    parent_main_labels: List[str],
    existing_children: Optional[List[dict]] = None,
    config: Optional[ProviderConfig] = None,
    callbacks: Optional[AgentCallbacks] = None,
    session_id: str = "",
) -> AgentRequest:
    """Build the AgentRequest for one detailed-flow run.

    Prompt-mode-aware: local_pipeline gets a JSON-plan prompt; tool_agent
    providers get a task + submit schema and drive via tools.
    """
    existing_children = list(existing_children or [])
    config = config or ProviderConfig(provider_id=provider_id)
    callbacks = callbacks or NoopCallbacks()

    # Seed pool so describe_graphs / label_verifier can read it from their
    # sliced pools.
    parent_segment = Attachment(
        name="init.parent_segment",
        kind="structured",
        content_schema="parent_segment",
        label="Parent Segment",
        content={
            "parent_start": int(parent_start),
            "parent_end": int(parent_end),
            "main_labels": [LABEL_MAPPING.get(l, l) for l in parent_main_labels],
            "existing_children": [
                {
                    "start_index": c.get("start_index"),
                    "end_index": c.get("end_index"),
                    "labels": [
                        LABEL_MAPPING.get(l, l) for l in c.get("labels", [])
                    ],
                }
                for c in existing_children
            ],
        },
    )

    if prompt_mode == "local_pipeline":
        planner_prompt = _local_planner_prompt(
            parent_start=parent_start,
            parent_end=parent_end,
            parent_main_labels=parent_main_labels,
            existing_children=existing_children,
        )
        synth_prompt: Callable[[Dict[str, Any]], Tuple[str, str]] = (
            lambda s: _local_synth_prompts(
                parent_start=parent_start,
                parent_end=parent_end,
                verified_labels=_verified_label_ids_from_state(s),
            )
        )
    else:
        planner_prompt = _tool_agent_task_prompt(
            parent_start=parent_start,
            parent_end=parent_end,
            parent_main_labels=parent_main_labels,
            existing_children=existing_children,
        )
        # Not used by claude runner — kept as a no-op callable so the
        # contract holds.
        synth_prompt = lambda _state: ("", "")

    extra_state: Dict[str, Any] = {
        "root_agent": "annotation_root",
    }
    if prompt_mode == "tool_agent":
        extra_state["tool_agent_extra_tools"] = [SEARCH_LABELS_TOOL]

    return AgentRequest(
        provider_id=provider_id,
        config=config,
        planner_prompt=planner_prompt,
        synth_prompt=synth_prompt,
        df_ref=df,
        parent_start=int(parent_start),
        parent_end=int(parent_end),
        initial_attachments=[parent_segment],
        callbacks=callbacks,
        session_id=session_id,
        extra_state=extra_state,
    )


def parse(
    response: AgentResponse,
    *,
    prompt_mode: str,
    parent_start: int,
    parent_end: int,
) -> AnnotationResult:
    """Decode the agent's raw_response into an AnnotationResult.

    ``prompt_mode="local_pipeline"`` expects a JSON object with a
    ``labels`` key. ``prompt_mode="tool_agent"`` expects a submitted
    JSON object with a ``proposals`` key.
    Both shapes lower to the same AnnotationResult.
    """
    raw = response.raw_response or ""

    if prompt_mode == "tool_agent":
        return _parse_claude(response, raw, parent_start, parent_end)
    return _parse_local(response, raw, parent_start, parent_end)


def _parse_local(
    response: AgentResponse,
    raw: str,
    parent_start: int,
    parent_end: int,
) -> AnnotationResult:
    sub_labels: List[str] = []
    label_proposals: List[dict] = []
    proposed_start = parent_start
    proposed_end = parent_end
    reasoning = raw

    parsed = parse_json_response(raw)
    if parsed:
        label_annotations = parsed.get("labels", [])
        if isinstance(label_annotations, list):
            starts: List[int] = []
            ends: List[int] = []
            for ann in label_annotations:
                lid = ann.get("label_id")
                if not lid or lid not in LABEL_MAPPING:
                    continue
                if not ann.get("proved"):
                    continue
                raw_start = ann.get("start_index")
                raw_end = ann.get("end_index")
                ann_reasoning = ann.get("reasoning", "")
                if not isinstance(ann_reasoning, str):
                    ann_reasoning = str(ann_reasoning)
                ann_start = (
                    int(raw_start) if isinstance(raw_start, (int, float)) else parent_start
                )
                ann_end = (
                    int(raw_end) if isinstance(raw_end, (int, float)) else parent_end
                )
                if not (parent_start <= ann_start < ann_end <= parent_end):
                    continue
                if _is_full_parent_range(ann_start, ann_end, parent_start, parent_end):
                    continue
                sub_labels.append(lid)
                label_proposals.append({
                    "label_id": lid,
                    "start_index": ann_start,
                    "end_index": ann_end,
                    "reasoning": ann_reasoning,
                })
                starts.append(ann_start)
                ends.append(ann_end)

            if label_proposals:
                reasoning = "; ".join(p["reasoning"] for p in label_proposals)
            if starts:
                proposed_start = min(starts)
            if ends:
                proposed_end = max(ends)
    else:
        LOGGER.warning("detailed flow (local): synth response was not valid JSON.")

    return AnnotationResult(
        sub_start=proposed_start,
        sub_end=proposed_end,
        final_labels=list(dict.fromkeys(sub_labels)),
        final_reasoning=reasoning,
        accepted=response.verdict == "pass" and len(label_proposals) > 0,
        iterations=1,
        messages=list(response.messages),
        graph_images=list(response.graph_images),
        label_annotations=label_proposals,
    )


def _parse_claude(
    response: AgentResponse,
    raw: str,
    parent_start: int,
    parent_end: int,
) -> AnnotationResult:
    label_ids: List[str] = []
    label_proposals: List[dict] = []
    proposed_start = parent_start
    proposed_end = parent_end
    reasoning = raw

    parsed = parse_json_response(raw) if raw else None
    if parsed:
        proposals = parsed.get("proposals", [])
        if isinstance(proposals, list):
            starts: List[int] = []
            ends: List[int] = []
            for p in proposals:
                if not isinstance(p, dict):
                    continue
                lid = p.get("label_id")
                if lid not in LABEL_MAPPING:
                    continue
                try:
                    s = int(p.get("start_index"))
                    e = int(p.get("end_index"))
                except (TypeError, ValueError):
                    continue
                if not (parent_start <= s < e <= parent_end):
                    continue
                if _is_full_parent_range(s, e, parent_start, parent_end):
                    continue
                if lid not in label_ids:
                    label_ids.append(lid)
                label_proposals.append({
                    "label_id": lid,
                    "start_index": s,
                    "end_index": e,
                    "reasoning": str(p.get("reasoning", "")),
                })
                starts.append(s)
                ends.append(e)

            if starts:
                proposed_start = min(starts)
            if ends:
                proposed_end = max(ends)

    # Prefer the synthesizer.summary attachment as the high-level
    # reasoning; fall back to the transcript / raw payload.
    summary_att = response.attachments.get("synthesizer.summary")
    if summary_att and isinstance(summary_att.content, str) and summary_att.content:
        reasoning = summary_att.content
    elif label_proposals:
        reasoning = "; ".join(p["reasoning"] for p in label_proposals)

    return AnnotationResult(
        sub_start=proposed_start,
        sub_end=proposed_end,
        final_labels=list(dict.fromkeys(label_ids)),
        final_reasoning=reasoning,
        accepted=response.verdict == "submitted" and len(label_proposals) > 0,
        iterations=1,
        messages=list(response.messages),
        graph_images=list(response.graph_images),
        label_annotations=label_proposals,
    )
