"""
Lap-section excerpter flow.

The caller has rough-split a lap into per-circuit-section iloc ranges via
the deterministic ``split_lap_by_circuit_sections`` tool. One run of this
flow annotates ONE section: the agent inspects telemetry, optionally
shrinks/extends the boundary, and submits a single label proposal.

    build_request(backend, df, lap_start, lap_end, section_id, ...)
    parse(response, backend, ...) -> LapAnnotationResult
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

from app.domain.labels import LABEL_MAPPING
from app.internal_knowledge_base import skills
from app.local_annotation_agent import (
    AgentRequest,
    AgentResponse,
    Attachment,
    BackendConfig,
)
from app.shared.contracts import AgentCallbacks, NoopCallbacks
from app.local_annotation_agent.workflow.results import (
    LapAnnotationResult,
    parse_json_response,
)
from app.local_annotation_agent.workflow.tools import CLAUDE_SEARCH_LABELS_TOOL


# ---------------------------------------------------------------------------
# lap_annotation skill — prompt rendering
# ---------------------------------------------------------------------------


def lap_annotation_prompt() -> str:
    """Per-label `characteristics` block for the lap-flow planner / synthesizer.

    The skill carries only main-label characteristics; circuit and
    circuit_section labels are attached upstream by the splitter and
    never appear here.
    """
    labels = skills.iter("lap_annotation.labels")
    global_rules = skills.get("lap_annotation.global_rules", "")

    lines: List[str] = [
        "#### Lap Annotation Skill — Candidate Label Characteristics",
        "",
        "Each candidate parent label below lists the telemetry pattern "
        "that justifies attaching it. The `global_rules` block at the "
        "end is the per-section detection procedure.",
        "",
    ]

    for entry in labels:
        lid = entry["id"]
        name = entry.get("name", lid)
        applies_when = str(entry.get("applies_when", "")).strip()
        characteristics = str(entry.get("characteristics", "")).strip()

        lines.append(f"##### `{lid}` — {name}")
        if applies_when:
            lines.append(f"_Applies when:_ {applies_when}")
        if characteristics:
            lines.append(characteristics)
        lines.append("")

    if global_rules:
        lines.append("##### Global rules — how to find each label")
        for ln in str(global_rules).rstrip("\n").split("\n"):
            lines.append(ln)
        lines.append("")

    return "\n".join(lines)


def _verified_label_ids_from_state(state: Dict[str, Any]) -> List[str]:
    """Pull verified label IDs out of the attachment pool."""
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


# ---------------------------------------------------------------------------
# Local-backend planner + synth prompts
# ---------------------------------------------------------------------------


def _local_planner_prompt(
    *,
    lap_start: int,
    lap_end: int,
    section_id: str,
    section_start: int,
    section_end: int,
    circuit_id: str,
    existing_section_annotations: List[dict],
) -> str:
    from app.shared.annotation_agent_tools import (
        AGENT_GRAPH_DEFINITIONS,
        PIPELINE_TOOL_DEFINITIONS,
    )

    section_name = LABEL_MAPPING.get(section_id, section_id) if section_id else section_id

    graph_catalogue = ", ".join(
        f"`{gdef['id']}` ({gdef['title']})"
        for gdef in AGENT_GRAPH_DEFINITIONS
    )
    tool_catalogue_lines = [
        f"- `{t['id']}` — {t['label']}: {t['description']}"
        for t in PIPELINE_TOOL_DEFINITIONS
    ]

    lap_skill_block = lap_annotation_prompt()

    existing_block = ""
    if existing_section_annotations:
        rows = []
        for c in existing_section_annotations:
            names = ", ".join(
                LABEL_MAPPING.get(l, l) for l in c.get("labels", [])
            )
            rows.append(
                f"  - [{c.get('start_index')}, {c.get('end_index')}] — {names}"
            )
        existing_block = (
            "\n#### Sections already annotated on this lap "
            "(reference, do NOT re-annotate)\n"
            + "\n".join(rows) + "\n"
        )

    parts = [
        "You are a racing telemetry analyst planning the analysis for "
        "ONE circuit section of a lap. The deterministic splitter handed "
        "you a rough iloc boundary; the synthesizer downstream will pick "
        "the parent labels by matching telemetry against each candidate "
        "label's `characteristics` block in the skill. Your job here is "
        "to plan the steps that gather that evidence.",
        "",
        "#### Lap context",
        f"- Circuit: {circuit_id}",
        f"- Lap range: [{lap_start}, {lap_end}] (length {lap_end - lap_start})",
        "",
        "#### Section under review",
        f"- circuit_section: `{section_id}` ({section_name}) "
        "— located by the splitter; the synthesizer emits it (with the "
        "circuit) as a label.",
        f"- rough iloc boundary: [{section_start}, {section_end}] "
        f"(length {section_end - section_start})",
        existing_block,
        "",
        lap_skill_block,
        "",
        "#### Available Step-Solver Agents",
        "Each plan step is dispatched to ONE sub-agent.",
        "- `describe_graphs` — renders the listed graphs over the rough "
        "boundary and writes one observation paragraph per graph.",
        "- `label_verifier` — embedding-similarity filter against the "
        "candidate labels using the describe_graphs observations. End your "
        "plan with one label_verifier step.",
        "",
        "#### Available Graph IDs",
        graph_catalogue,
        "",
        "#### Available Pre-Compute Tools",
        *tool_catalogue_lines,
        "",
        "#### Task",
        "Plan describe_graphs steps gathering evidence to:",
        "  1. score each main label (EA / MS / RM / PS / O / MD) against "
        "its `characteristics` block in the skill, and",
        "  2. optionally identify the trajectory shape if an ST1-ST6 pick "
        "would be unambiguous.",
        "Keep the plan tight — typically 1-3 describe_graphs steps plus a "
        "label_verifier. `trajectory_offset` + `time_difference_to_expert` "
        "are the two diagnostic graphs called out by the skill.",
        "",
        "Plan format: JSON object with a single key \"steps\". Each step:",
        "  - \"step_id\": integer (1, 2, 3, ...).",
        "  - \"agent\": one of `describe_graphs` or `label_verifier`.",
        "  - \"description\": short string stating the goal of the step.",
        "  - \"requested_graphs\": list of graph IDs (describe_graphs only).",
        "  - \"tools\": list of pre-compute tool IDs (empty `[]` for none).",
        "",
        "Example:",
        "```json",
        "{",
        '  "steps": [',
        '    {"step_id": 1, "agent": "describe_graphs", "description": '
        '"Confirm boundary + check brake/throttle onsets.", "requested_graphs": '
        '["trajectory_offset", "brake", "throttle"], "tools": '
        '["compute_expert_phases"]},',
        '    {"step_id": 2, "agent": "describe_graphs", "description": '
        '"Measure trajectory shape to pick ST1-ST6.", "requested_graphs": '
        '["trajectory_detailed"], "tools": []},',
        '    {"step_id": 3, "agent": "label_verifier", "description": '
        '"Shortlist labels by similarity to observations.", '
        '"requested_graphs": [], "tools": []}',
        "  ]",
        "}",
        "```",
    ]
    return "\n".join(parts)


def _local_synth_prompts(
    *,
    lap_start: int,
    lap_end: int,
    section_id: str,
    section_start: int,
    section_end: int,
    circuit_id: str,
    verified_labels: List[str],
) -> Tuple[str, str]:
    lap_skill_block = lap_annotation_prompt()
    verified_inline = (
        ", ".join(verified_labels) if verified_labels
        else "(none — emit an empty label_ids array)"
    )

    intro = "\n".join([
        "You are a racing telemetry analyst producing the final "
        "annotation for ONE circuit section. The describe_graphs steps "
        "captured the evidence below; pick the parent labels by matching "
        "the section's telemetry against each candidate label's "
        "`characteristics` block in the skill. Revising the boundary is "
        "an escape hatch only — invoke it when one main-label signature "
        "does not hold across the rough range.",
        "",
        "#### Section under review",
        f"- circuit_section id: `{section_id}` "
        "(located by the splitter; include it in label_ids)",
        f"- circuit id: `{circuit_id}` "
        "(from Static_track; include it in label_ids)",
        f"- rough iloc boundary: [{section_start}, {section_end}]",
        f"- lap range: [{lap_start}, {lap_end}]",
        "",
        lap_skill_block,
    ])

    outro = "\n".join([
        "#### Candidate label IDs",
        f"The shortlist retrieved for this section is: {verified_inline}. "
        "Pick the parent label(s) from this shortlist by matching the "
        "section's telemetry against each candidate's `characteristics` "
        "block in the skill. An ST1-ST6 pick is OPTIONAL — include one only "
        "when the trajectory shape is unambiguous. At most ONE of "
        "{EA, MSP, MSR, RM} may be attached.",
        "",
        "#### Output format",
        "Respond with ONE JSON object only — no surrounding prose. Schema:",
        "```json",
        "{",
        '  "revised_range": [start_iloc, end_iloc],',
        '  "revised": <true|false>,',
        '  "revision_reason": "<one short sentence; empty when revised=false>",',
        '  "label_ids": ["<id>", ...],',
        '  "reasoning": "<1-3 sentences citing ilocs / values>"',
        "}",
        "```",
        "Hard rules:",
        f"- revised_range must satisfy {lap_start} <= start < end <= "
        f"{lap_end} and end - start >= 3.",
        "- Every main / ST / sub label_id must come from the shortlist "
        "above; additionally include the circuit id and circuit_section "
        "id listed under 'Section under review'.",
        "- An empty label_ids array is the valid 'drop this section' signal.",
    ])

    return intro, outro


# ---------------------------------------------------------------------------
# Claude-backend task prompt
# ---------------------------------------------------------------------------


def _claude_task_prompt(
    *,
    lap_start: int,
    lap_end: int,
    section_start: int,
    section_end: int,
    existing_section_annotations: List[dict],
) -> str:
    lap_skill_block = lap_annotation_prompt()

    existing_block = ""
    if existing_section_annotations:
        lines = []
        for c in existing_section_annotations:
            names = ", ".join(LABEL_MAPPING.get(l, l) for l in c.get("labels", []))
            lines.append(
                f"  - [{c['start_index']}, {c['end_index']}] — {names}"
            )
        existing_block = (
            "\n### Sections already annotated on this lap "
            "(reference, do NOT re-annotate)\n"
            + "\n".join(lines) + "\n"
        )

    return (
        "Annotate ONE circuit section of a lap. The deterministic "
        "splitter handed you a rough iloc boundary; your job is to label "
        "which circuit + named circuit_section this is AND to pick the "
        "main label (+ optional ST1–ST6, + optional sub-label) by "
        "matching the section's telemetry against each candidate label's "
        "`characteristics` block in the skill.\n"
        "\n"
        "### Lap context\n"
        f"- Lap range: [{lap_start}, {lap_end}] "
        f"(length {lap_end - lap_start})\n"
        f"- Rough section boundary: [{section_start}, {section_end}] "
        f"(length {section_end - section_start})\n"
        f"{existing_block}"
        "\n"
        f"{lap_skill_block}"
        "\n"
        "### How to work\n"
        "Follow the numbered procedure in the skill's `global_rules` block "
        "above. Labels are not pre-loaded into this prompt — you discover "
        "every candidate by querying `search_labels` with a plain-language "
        "description of what you observed:\n"
        "\n"
        "1. **Identify the circuit + circuit_section.** Call "
        "`get_circuit_id` for the circuit id and `locate_circuit_section` "
        "for the named section that the iloc window overlaps. Include both "
        "returned ids in `label_ids`. When `locate_circuit_section` reports "
        "`is_ambiguous` (e.g. pit lane vs. the adjacent straight share a "
        "normalized position range), do NOT guess — read its `top_matches` "
        "and disambiguate with a second signal (pit-limiter speed, "
        "persistent lateral offset, brake pattern); drop the section id "
        "only if you still cannot commit.\n"
        "2. **Pick ONE parent main label.** After inspecting the telemetry, "
        "call `search_labels(query=<what you observed>, types=\"main\")` and "
        "pick the best-matching main label (EA / MSP / MSR / RM / PS / O / "
        "OD / MD), or none when telemetry is too noisy to commit. At most "
        "one of {EA, MSP, MSR, RM} may be attached.\n"
        "3. **Maybe pick ONE segment type.** Call `search_labels(query="
        "<trajectory shape>, types=\"segment_type\")` and attach one ST1-ST6 "
        "only when the shape is unambiguous; skip it otherwise.\n"
        "4. **Check for a sub-label fit.** When a main label was picked, "
        "call `search_labels(query=<what you observed>, parent_id=<main>)` "
        "and read each returned sub-label's `description`. Attach a "
        "sub-label ONLY when its description matches the **entire** "
        "section's telemetry — partial-fit cases stay in the "
        "detailed-annotation flow. Re-query with different wording to "
        "broaden.\n"
        "5. **Submit.** Call `submit_result` with the chosen IDs.\n"
        "\n"
        "Call `revise_range` when one main-label signature does not hold "
        "uniformly across the rough range — shrink to the ilocs where ONE "
        "characteristic block fits cleanly, or extend outward (within the "
        "lap range) when the signature clearly continues past the rough "
        "end. Re-render the diagnostic graphs on the new range before "
        "submitting. After `submit_result` returns `ok: true`, stop.\n"
        "\n"
        "### Submit payload shape\n"
        "`payload_json` must be a JSON object of this shape:\n"
        "```json\n"
        "{\n"
        '  "label_ids": ["<id>", "<id>", ...],\n'
        '  "reasoning": "<1-3 sentences citing ilocs / values>"\n'
        "}\n"
        "```\n"
        "`label_ids` carries the circuit id, the circuit_section id, and "
        "your main / ST / sub picks together. An empty `label_ids` array "
        "is a valid 'drop this section' signal. The runner reports back "
        "the final iloc range (your initial range or whatever "
        "`revise_range` set last).\n"
        "\n"
        "### Hard rules\n"
        f"- Final range must satisfy {lap_start} <= start < end <= {lap_end} and be ≥ 3 ilocs.\n"
        "- Do not invent label IDs — circuit / circuit_section ids come "
        "from `get_circuit_id` / `locate_circuit_section`, every other id "
        "from a `search_labels` response.\n"
        "- One proposal per session — do NOT annotate downstream sections.\n"
        "- Budget tool calls: a typical section needs 7-10 calls total."
    )


# ---------------------------------------------------------------------------
# Public flow API
# ---------------------------------------------------------------------------


def build_request(
    *,
    backend: str,
    df,
    lap_start: int,
    lap_end: int,
    section_id: str,
    section_start: int,
    section_end: int,
    circuit_id: str,
    existing_section_annotations: Optional[List[dict]] = None,
    config: Optional[BackendConfig] = None,
    callbacks: Optional[AgentCallbacks] = None,
    session_id: str = "",
) -> AgentRequest:
    existing_section_annotations = list(existing_section_annotations or [])
    config = config or BackendConfig()
    callbacks = callbacks or NoopCallbacks()

    section_name = LABEL_MAPPING.get(section_id, section_id) if section_id else section_id

    parent_segment = Attachment(
        name="init.parent_segment",
        kind="structured",
        content_schema="parent_segment",
        label=f"Lap Section: {section_id} ({section_name})",
        content={
            "parent_start": int(section_start),
            "parent_end": int(section_end),
            "main_labels": [circuit_id] if circuit_id else [],
            "existing_children": [
                {
                    "start_index": c.get("start_index"),
                    "end_index": c.get("end_index"),
                    "labels": [
                        LABEL_MAPPING.get(l, l) for l in c.get("labels", [])
                    ],
                }
                for c in existing_section_annotations
            ],
        },
    )

    # parent_start/end on the request are the section range — sub-agents
    # like describe_graphs operate over this window.
    parent_start = int(section_start)
    parent_end = int(section_end)

    if backend == "local":
        planner_prompt = _local_planner_prompt(
            lap_start=lap_start,
            lap_end=lap_end,
            section_id=section_id,
            section_start=section_start,
            section_end=section_end,
            circuit_id=circuit_id,
            existing_section_annotations=existing_section_annotations,
        )
        synth_prompt: Callable[[Dict[str, Any]], Tuple[str, str]] = (
            lambda s: _local_synth_prompts(
                lap_start=lap_start,
                lap_end=lap_end,
                section_id=section_id,
                section_start=section_start,
                section_end=section_end,
                circuit_id=circuit_id,
                verified_labels=_verified_label_ids_from_state(s),
            )
        )
        extra_state = {"root_agent": "annotation_root"}
    else:
        # For Claude, parent_start/end on the request bound the
        # working envelope. We use the LAP range so revise_range can
        # extend outward when a shrink/extend rule fires.
        parent_start = int(lap_start)
        parent_end = int(lap_end)
        planner_prompt = _claude_task_prompt(
            lap_start=lap_start,
            lap_end=lap_end,
            section_start=section_start,
            section_end=section_end,
            existing_section_annotations=existing_section_annotations,
        )
        synth_prompt = lambda _state: ("", "")
        extra_state = {
            "root_agent": "annotation_root",
            "claude_extra_tools": [CLAUDE_SEARCH_LABELS_TOOL],
        }

    return AgentRequest(
        backend=backend,  # type: ignore[arg-type]
        config=config,
        planner_prompt=planner_prompt,
        synth_prompt=synth_prompt,
        df_ref=df,
        parent_start=parent_start,
        parent_end=parent_end,
        initial_attachments=[parent_segment],
        callbacks=callbacks,
        session_id=session_id,
        extra_state=extra_state,
    )


def parse(
    response: AgentResponse,
    *,
    backend: str,
    lap_start: int,
    lap_end: int,
    section_id: str,
    section_start: int,
    section_end: int,
) -> LapAnnotationResult:
    """Decode the raw response into a LapAnnotationResult.

    ``backend="local"`` expects the JSON schema with revised_range +
    label_ids + reasoning. ``backend="claude"`` reads the submit payload
    ({label_ids, reasoning}) and the revised range from
    ``response.attachments["claude.revised_range"]``.

    Returns only what the LLM committed to — including the circuit and
    circuit_section ids the LLM picked via ``get_circuit_id`` /
    ``locate_circuit_section``.
    """
    if backend == "claude":
        return _parse_claude(response, lap_start, lap_end, section_id,
                             section_start, section_end)
    return _parse_local(response, lap_start, lap_end, section_id,
                        section_start, section_end)


def _parse_local(
    response: AgentResponse,
    lap_start: int,
    lap_end: int,
    section_id: str,
    section_start: int,
    section_end: int,
) -> LapAnnotationResult:
    raw = response.raw_response or ""
    parsed = parse_json_response(raw)
    if not parsed:
        raise RuntimeError(
            f"lap flow (local): synth response was not valid JSON. "
            f"First 300 chars: {raw[:300]!r}"
        )

    revised_range = parsed.get("revised_range") or [section_start, section_end]
    try:
        new_start = int(revised_range[0])
        new_end = int(revised_range[1])
    except (TypeError, ValueError, IndexError) as exc:
        raise RuntimeError(
            f"lap flow (local): revised_range was not [int, int]: "
            f"{revised_range!r}"
        ) from exc
    if not (lap_start <= new_start < new_end <= lap_end):
        raise RuntimeError(
            f"lap flow (local): revised_range [{new_start}, {new_end}] "
            f"outside lap [{lap_start}, {lap_end}] or start >= end"
        )
    if (new_end - new_start) < 5:
        raise RuntimeError(
            f"lap flow (local): revised_range too short "
            f"({new_end - new_start} ilocs) — minimum 5"
        )

    raw_label_ids = parsed.get("label_ids") or []
    cleaned, rejected = _clean_label_ids(raw_label_ids)

    revised_flag = bool(parsed.get("revised")) or (
        new_start != section_start or new_end != section_end
    )
    reasoning = str(parsed.get("reasoning") or "")
    if parsed.get("revision_reason") and revised_flag:
        reasoning = (
            f"[revision: {parsed.get('revision_reason')}] {reasoning}".strip()
        )

    return LapAnnotationResult(
        section_id=section_id,
        start_index=new_start,
        end_index=new_end,
        label_ids=cleaned,
        reasoning=reasoning or raw or "(no reasoning)",
        revised=revised_flag,
        submitted=True,
        rough_start=int(section_start),
        rough_end=int(section_end),
        rejected_proposals=rejected,
        rendered_images=list(response.graph_images),
        transcript=raw,
        tool_calls=0,
    )


def _parse_claude(
    response: AgentResponse,
    lap_start: int,
    lap_end: int,
    section_id: str,
    section_start: int,
    section_end: int,
) -> LapAnnotationResult:
    raw = response.raw_response or ""
    parsed = parse_json_response(raw) if raw else None

    cleaned: List[str] = []
    rejected: List[Dict[str, Any]] = []
    reasoning = ""
    if parsed:
        raw_label_ids = parsed.get("label_ids") or []
        cleaned, rejected = _clean_label_ids(raw_label_ids)
        reasoning = str(parsed.get("reasoning") or "")

    # Resolve final range — prefer claude.revised_range attachment when
    # revise_range fired; otherwise the section's rough boundary.
    new_start, new_end = int(section_start), int(section_end)
    revised = False
    revised_att = response.attachments.get("claude.revised_range")
    if revised_att and isinstance(revised_att.content, dict):
        new_start = int(revised_att.content.get("start_index", section_start))
        new_end = int(revised_att.content.get("end_index", section_end))
        revised = (new_start, new_end) != (section_start, section_end)

    if not (lap_start <= new_start < new_end <= lap_end):
        raise RuntimeError(
            f"lap flow (claude): final range [{new_start}, {new_end}] "
            f"outside lap [{lap_start}, {lap_end}]"
        )
    if (new_end - new_start) < 5:
        raise RuntimeError(
            f"lap flow (claude): final range too short "
            f"({new_end - new_start} ilocs) — minimum 5"
        )

    # Prefer the submission summary as headline reasoning if present.
    summary_att = response.attachments.get("synthesizer.summary")
    if summary_att and isinstance(summary_att.content, str) and summary_att.content:
        reasoning = summary_att.content

    transcript_att = response.attachments.get("claude.transcript")
    transcript = (
        transcript_att.content
        if transcript_att and isinstance(transcript_att.content, str)
        else ""
    )

    return LapAnnotationResult(
        section_id=section_id,
        start_index=new_start,
        end_index=new_end,
        label_ids=cleaned,
        reasoning=reasoning or transcript or "(no reasoning)",
        revised=revised,
        submitted=response.verdict == "submitted",
        rough_start=int(section_start),
        rough_end=int(section_end),
        rejected_proposals=rejected,
        rendered_images=list(response.graph_images),
        transcript=transcript,
        tool_calls=0,
    )


def _clean_label_ids(
    raw_label_ids: Any,
) -> Tuple[List[str], List[Dict[str, Any]]]:
    cleaned: List[str] = []
    rejected: List[Dict[str, Any]] = []
    if not isinstance(raw_label_ids, list):
        rejected.append({
            "value": raw_label_ids, "reason": "label_ids was not a list",
        })
        return cleaned, rejected
    for i, raw_lid in enumerate(raw_label_ids):
        if not isinstance(raw_lid, str):
            rejected.append({
                "index": i, "value": raw_lid, "reason": "must be string",
            })
            continue
        if raw_lid not in LABEL_MAPPING:
            rejected.append({
                "index": i, "value": raw_lid,
                "reason": f"unknown label_id '{raw_lid}'",
            })
            continue
        if raw_lid in cleaned:
            continue
        cleaned.append(raw_lid)
    return cleaned, rejected
