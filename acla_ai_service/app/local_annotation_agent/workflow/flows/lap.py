"""
Lap-section excerpter flow.

The caller has split a lap via the deterministic
``split_lap_by_circuit_sections`` tool. Solo sessions produce
circuit-section ranges; opponent sessions produce only close overtake
offence / defence engagement ranges. One run of this flow annotates ONE
range: the agent inspects telemetry within that split section and submits
a single label proposal.

    build_request(provider_id, df, lap_start, lap_end, section_id, ...)
    parse(response, ...) -> LapAnnotationResult
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from app.shared.label_hierarchy import normalize_grouped_label_ids
from app.shared.labels import LABEL_MAPPING
from app.internal_knowledge_base import skills
from app.local_annotation_agent import (
    AgentRequest,
    AgentResponse,
    Attachment,
    ProviderConfig,
)
from app.shared.contracts import AgentCallbacks, NoopCallbacks
from app.local_annotation_agent.workflow.results import (
    LapAnnotationResult,
    parse_json_response,
)
from app.local_annotation_agent.workflow.preflight_lap import (
    build_preflight_context,
)
from app.annotation_providers.tool_surface import PREFLIGHT_ONLY_TOOL_NAMES


LOGGER = logging.getLogger(__name__)


def _render_lap_template(template: str, **values: Any) -> str:
    rendered = str(template)
    for key, value in values.items():
        rendered = rendered.replace(f"{{{{{key}}}}}", str(value))
    return rendered.strip()


def _lap_prompt_rule(
    path: str,
    *,
    required: bool = True,
    **values: Any,
) -> str:
    value = skills.get(f"lap_annotation.prompt_rules.{path}", "")
    if not isinstance(value, str):
        raise RuntimeError(
            f"lap_annotation.prompt_rules.{path} must be a string"
        )
    rendered = _render_lap_template(value, **values)
    if required and not rendered:
        raise RuntimeError(f"lap_annotation.prompt_rules.{path} is missing")
    return rendered


def _lap_prompt_rule_list(
    path: str,
    *,
    required: bool = True,
    **values: Any,
) -> List[str]:
    value = skills.get(f"lap_annotation.prompt_rules.{path}", [])
    if not isinstance(value, list):
        raise RuntimeError(f"lap_annotation.prompt_rules.{path} must be a list")
    rules = [
        _render_lap_template(str(rule), **values)
        for rule in value
        if str(rule).strip()
    ]
    if required and not rules:
        raise RuntimeError(f"lap_annotation.prompt_rules.{path} is missing")
    return rules


def _lap_label_id_list(path: str) -> Tuple[str, ...]:
    value = skills.get(f"lap_annotation.behavior_parent_label_ids.{path}", [])
    if not isinstance(value, list):
        raise RuntimeError(
            f"lap_annotation.behavior_parent_label_ids.{path} must be a list"
        )
    label_ids = tuple(
        str(label_id).strip()
        for label_id in value
        if str(label_id).strip()
    )
    if not label_ids:
        raise RuntimeError(
            f"lap_annotation.behavior_parent_label_ids.{path} is missing"
        )
    return label_ids


def _is_racing_context(
    section_split_basis: Optional[str],
    opponent_interaction: Optional[dict],
) -> bool:
    return bool(opponent_interaction) or (
        "interaction" in str(section_split_basis or "")
    )


def _session_context(
    section_split_basis: Optional[str],
    opponent_interaction: Optional[dict],
) -> str:
    return (
        "racing"
        if _is_racing_context(section_split_basis, opponent_interaction)
        else "practice"
    )


def _eligible_behavior_label_ids(session_context: str) -> Tuple[str, ...]:
    return _lap_label_id_list(f"eligible_by_session.{session_context}")


def _required_behavior_label_ids() -> Tuple[str, ...]:
    return _lap_label_id_list("required")


def _label_set_text(label_ids: Tuple[str, ...] | List[str]) -> str:
    return "{" + ", ".join(label_ids) + "}"


def _mode_exclusion_rule(session_context: str) -> str:
    return _lap_prompt_rule(
        f"mode_exclusion_rules.{session_context}",
    )


def _required_behavior_parent_label_rule(eligible_label_ids: List[str]) -> str:
    return _lap_prompt_rule(
        "required_behavior_parent_label_rule",
        eligible_behavior_labels=_label_set_text(eligible_label_ids),
    )


def _interaction_section_context(
    opponent_interaction: Optional[dict],
) -> List[Dict[str, Any]]:
    if not isinstance(opponent_interaction, dict):
        return []
    context = opponent_interaction.get("section_context") or []
    if not isinstance(context, list):
        return []
    return [c for c in context if isinstance(c, dict)]


def _preselected_interaction_section_id(
    opponent_interaction: Optional[dict],
) -> Optional[str]:
    """Return a splitter-provided circuit_section id when it is unambiguous."""
    section_ids: List[str] = []
    for context in _interaction_section_context(opponent_interaction):
        section_id = context.get("circuit_section_id")
        if (
            isinstance(section_id, str)
            and section_id
            and section_id != "interaction_window"
            and section_id in LABEL_MAPPING
            and section_id not in section_ids
        ):
            section_ids.append(section_id)
    return section_ids[0] if len(section_ids) == 1 else None


def _interaction_focus_block(
    section_split_basis: Optional[str],
    opponent_interaction: Optional[dict],
) -> str:
    """Prompt block for opponent-only work units."""
    is_interaction = bool(opponent_interaction) or (
        "interaction" in str(section_split_basis or "")
    )
    if not is_interaction:
        return ""
    windows = []
    if isinstance(opponent_interaction, dict):
        windows = list(opponent_interaction.get("windows") or [])
    manual = any(isinstance(w, dict) and w.get("manual") for w in windows)
    origin = (
        "was manually selected as a racing interaction range"
        if manual else
        "exists because a close opponent engagement was detected"
    )
    target_lines: List[str] = []
    if isinstance(opponent_interaction, dict):
        slot = opponent_interaction.get("targeted_car_slot")
        label = opponent_interaction.get("targeted_car_label")
        if slot is not None or label:
            target_lines.append(
                f"- Preselected target: {label or f'Car {slot}'} "
                f"(slot {slot}). Inspect this slot first."
            )
        for window in windows[:3]:
            if not isinstance(window, dict):
                continue
            details: List[str] = []
            for key, label_text in (
                ("event_role", "role"),
                ("event_outcome", "outcome"),
                ("entry_signed_long_gap_m", "entry gap"),
                ("exit_signed_long_gap_m", "exit gap"),
                ("min_distance_m", "closest"),
                ("close_following_iloc_count", "close-following ilocs"),
                ("trailing_pressure_iloc_count", "trailing-pressure ilocs"),
                ("leading_draft_iloc_count", "leading-draft ilocs"),
            ):
                value = window.get(key)
                if value is not None:
                    details.append(f"{label_text}: {value}")
            if details:
                target_lines.append(
                    f"- Splitter evidence [{window.get('start_index')}, "
                    f"{window.get('end_index')}]: " + "; ".join(details)
                )
        for context in _interaction_section_context(opponent_interaction):
            section_id = context.get("circuit_section_id")
            if not section_id or section_id == "interaction_window":
                continue
            section_name = (
                context.get("circuit_section_name")
                or LABEL_MAPPING.get(section_id, section_id)
            )
            section_range = context.get("range")
            suffix = f" over ilocs {section_range}" if section_range else ""
            target_lines.append(
                f"- Splitter section context: `{section_id}` "
                f"({section_name}){suffix}."
            )
    target_block = (
        "\n"
        + _lap_prompt_rule("interaction_focus.target_heading")
        + "\n"
        + "\n".join(target_lines)
        + "\n"
        if target_lines else ""
    )
    return (
        "\n#### Opponent-session focus\n"
        + _lap_prompt_rule("interaction_focus.body", origin=origin)
        + "\n"
        f"{target_block}"
    )


# ---------------------------------------------------------------------------
# Tool-agent task prompt
# ---------------------------------------------------------------------------


def _tool_agent_task_prompt(
    *,
    lap_start: int,
    lap_end: int,
    circuit_id: str,
    section_id: str,
    section_start: int,
    section_end: int,
    section_split_basis: Optional[str],
    opponent_interaction: Optional[dict],
) -> str:
    session_context = _session_context(section_split_basis, opponent_interaction)
    eligible_labels = list(_eligible_behavior_label_ids(session_context))
    required_label_rule = _required_behavior_parent_label_rule(eligible_labels)
    interaction_focus = _interaction_focus_block(
        section_split_basis, opponent_interaction,
    )
    eligible_labels_text = _label_set_text(eligible_labels)
    selection_notes = "\n".join(
        f"- {rule}"
        for rule in _lap_prompt_rule_list("selection_notes")
    )
    payload_followup = "\n".join(
        _lap_prompt_rule_list(
            "payload_shape.followup",
            eligible_behavior_labels=eligible_labels_text,
        )
    )
    reasoning_placeholder = _lap_prompt_rule("payload_shape.reasoning_placeholder")
    hard_rules = [
        required_label_rule,
        *_lap_prompt_rule_list("hard_rules"),
        _mode_exclusion_rule(session_context),
        _lap_prompt_rule("whole_range_label_rule"),
        _lap_prompt_rule("segment_action_model_rule"),
        _lap_prompt_rule("reasoning_note_rule"),
    ]
    hard_rule_bullets = "\n".join(f"- {rule}" for rule in hard_rules if rule)
    task_intro = _lap_prompt_rule("task_intro")
    preselected_section_id = _preselected_interaction_section_id(opponent_interaction)
    preselected_section_block = ""
    if preselected_section_id:
        preselected_section_block = "- " + _lap_prompt_rule(
            "preselected_section_context",
            preselected_section_id=preselected_section_id,
            preselected_section_name=LABEL_MAPPING.get(
                preselected_section_id,
                preselected_section_id,
            ),
        )
        preselected_section_block += "\n"

    return (
        f"{task_intro}\n"
        "\n"
        "### Lap context\n"
        f"- Detected session mode: {session_context}\n"
        f"- Eligible behavior parent labels: {eligible_labels_text}\n"
        f"- Lap range: [{lap_start}, {lap_end}] "
        f"(length {lap_end - lap_start})\n"
        f"- Split section boundary: [{section_start}, {section_end}] "
        f"(length {section_end - section_start})\n"
        f"{preselected_section_block}"
        f"{interaction_focus}"
        "\n"
        "### Selection notes\n"
        f"{selection_notes}\n"
        "\n"
        "### Submit payload shape\n"
        f"{_lap_prompt_rule('payload_shape.intro')}\n"
        "```json\n"
        "{\n"
        '  "label_ids": ['
        '"<id from Preflight label candidates>", '
        '"<id from Preflight label candidates>", ...],\n'
        f'  "reasoning": "{reasoning_placeholder}"\n'
        "}\n"
        "```\n"
        f"{payload_followup}\n"
        "\n"
        "### Hard rules\n"
        f"{hard_rule_bullets}"
    )


# ---------------------------------------------------------------------------
# Public flow API
# ---------------------------------------------------------------------------


def build_request(
    *,
    provider_id: str,
    df,
    lap_start: int,
    lap_end: int,
    section_id: str,
    section_start: int,
    section_end: int,
    circuit_id: str,
    section_split_basis: Optional[str] = None,
    opponent_interaction: Optional[dict] = None,
    existing_section_annotations: Optional[List[dict]] = None,
    config: Optional[ProviderConfig] = None,
    callbacks: Optional[AgentCallbacks] = None,
    session_id: str = "",
) -> AgentRequest:
    config = config or ProviderConfig(provider_id=provider_id)
    callbacks = callbacks or NoopCallbacks()

    section_name = LABEL_MAPPING.get(section_id, section_id) if section_id else section_id
    session_context = _session_context(section_split_basis, opponent_interaction)
    eligible_labels = list(_eligible_behavior_label_ids(session_context))
    candidate_label_ids = [
        label_id
        for label_id in [circuit_id, *eligible_labels]
        if label_id
    ]

    parent_segment = Attachment(
        name="init.parent_segment",
        kind="structured",
        content_schema="parent_segment",
        label=f"Lap Section: {section_id} ({section_name})",
        content={
            "parent_start": int(section_start),
            "parent_end": int(section_end),
            "split_basis": section_split_basis or "circuit_section",
            "session_context": session_context,
            "eligible_behavior_label_ids": eligible_labels,
            "opponent_interaction": opponent_interaction,
            "preselected_circuit_section_id": (
                _preselected_interaction_section_id(opponent_interaction)
            ),
            "section_context": _interaction_section_context(opponent_interaction),
            "main_labels": [circuit_id] if circuit_id else [],
            "existing_children": [],
        },
    )

    preflight = build_preflight_context(
        df=df,
        start=section_start,
        end=section_end,
        candidate_label_ids=candidate_label_ids,
        extra_query_terms=[
            session_context,
            section_split_basis or "circuit_section",
            LABEL_MAPPING.get(section_id, section_id),
        ],
    )

    parent_start = int(section_start)
    parent_end = int(section_end)

    task_prompt = _tool_agent_task_prompt(
        lap_start=lap_start,
        lap_end=lap_end,
        circuit_id=circuit_id,
        section_id=section_id,
        section_start=section_start,
        section_end=section_end,
        section_split_basis=section_split_basis,
        opponent_interaction=opponent_interaction,
    )
    shared_front_prompt = "\n\n".join([preflight.prompt_block, task_prompt])
    planner_prompt = shared_front_prompt
    synth_prompt = lambda _state: ("", "")
    extra_state = {
        "tool_agent_excluded_tools": sorted(PREFLIGHT_ONLY_TOOL_NAMES),
        "annotation_session_context": session_context,
        "eligible_behavior_label_ids": eligible_labels,
    }

    return AgentRequest(
        provider_id=provider_id,
        config=config,
        planner_prompt=planner_prompt,
        synth_prompt=synth_prompt,
        df_ref=df,
        parent_start=parent_start,
        parent_end=parent_end,
        initial_attachments=[parent_segment, *preflight.attachments],
        callbacks=callbacks,
        session_id=session_id,
        extra_state=extra_state,
    )


def parse(
    response: AgentResponse,
    *,
    lap_start: int,
    lap_end: int,
    section_id: str,
    section_start: int,
    section_end: int,
    circuit_id: Optional[str] = None,
    section_split_basis: Optional[str] = None,
    opponent_interaction: Optional[dict] = None,
) -> LapAnnotationResult:
    """Decode the raw response into a LapAnnotationResult.

    Reads the submitted JSON payload.

    Returns the LLM-committed labels after normalizing ambiguous same-range
    circuit_section ids to one selected section.
    """
    session_context = _session_context(section_split_basis, opponent_interaction)
    eligible_labels = list(_eligible_behavior_label_ids(session_context))

    return _parse_tool_agent(
        response, section_id, section_start, section_end, circuit_id,
        opponent_interaction, eligible_labels,
    )


def _parse_tool_agent(
    response: AgentResponse,
    section_id: str,
    section_start: int,
    section_end: int,
    circuit_id: Optional[str],
    opponent_interaction: Optional[dict],
    eligible_behavior_label_ids: List[str],
) -> LapAnnotationResult:
    raw = response.raw_response or ""
    parsed = parse_json_response(raw) if raw else None

    cleaned: List[str] = []
    rejected: List[Dict[str, Any]] = []
    reasoning = ""
    if parsed:
        _reject_unknown_output_fields(
            parsed,
            "tool_agent",
            extra_allowed_keys={"summary"},
        )
        raw_label_ids = parsed.get("label_ids") or []
        cleaned, rejected = _clean_label_ids(
            raw_label_ids,
            eligible_behavior_label_ids=eligible_behavior_label_ids,
            selected_circuit_section_id=section_id,
        )
        reasoning = str(parsed.get("reasoning") or parsed.get("summary") or "")

    new_start, new_end = int(section_start), int(section_end)
    if (new_end - new_start) < 5:
        raise RuntimeError(
            f"lap flow (tool_agent): split section range too short "
            f"({new_end - new_start} ilocs) — minimum 5"
        )

    # Prefer the submission summary as headline reasoning if present.
    summary_att = response.attachments.get("synthesizer.summary")
    if summary_att and isinstance(summary_att.content, str) and summary_att.content:
        reasoning = summary_att.content

    transcript_att = (
        response.attachments.get("tool_agent.transcript")
        or response.attachments.get("claude.transcript")
    )
    transcript = (
        transcript_att.content
        if transcript_att and isinstance(transcript_att.content, str)
        else ""
    )
    if not transcript:
        transcript = "\n\n".join(
            str(message.get("content", ""))
            for message in response.messages
            if isinstance(message, dict) and message.get("content")
        )

    return LapAnnotationResult(
        section_id=section_id,
        start_index=new_start,
        end_index=new_end,
        label_ids=cleaned,
        reasoning=reasoning or transcript or "(no reasoning)",
        submitted=response.verdict == "submitted",
        rejected_proposals=rejected,
        rendered_images=list(response.graph_images),
        transcript=transcript,
        tool_calls=0,
    )


def _reject_unknown_output_fields(
    parsed: Dict[str, Any],
    source: str,
    *,
    extra_allowed_keys: Optional[set[str]] = None,
) -> None:
    allowed_keys = {"label_ids", "reasoning", *(extra_allowed_keys or set())}
    unknown_keys = sorted(str(key) for key in parsed if key not in allowed_keys)
    if unknown_keys:
        raise RuntimeError(
            f"lap flow ({source}): unsupported output field(s): "
            + ", ".join(unknown_keys)
        )


def _clean_label_ids(
    raw_label_ids: Any,
    *,
    eligible_behavior_label_ids: Optional[List[str]] = None,
    selected_circuit_section_id: Optional[str] = None,
) -> Tuple[List[str], List[Dict[str, Any]]]:
    cleaned, rejected, _ = normalize_grouped_label_ids(raw_label_ids)
    if not cleaned:
        return cleaned, rejected

    from app.internal_knowledge_base.label_lookup import get_label

    required_behavior_label_ids = _required_behavior_label_ids()
    eligible_behavior_label_ids = list(
        eligible_behavior_label_ids or required_behavior_label_ids
    )
    allowed: List[str] = []
    for label_id in cleaned:
        doc = get_label(label_id)
        parent_id = doc.get("parent") if doc else None
        if (
            label_id in required_behavior_label_ids
            and label_id not in eligible_behavior_label_ids
        ) or (
            parent_id in required_behavior_label_ids
            and parent_id not in eligible_behavior_label_ids
        ):
            rejected.append({
                "value": label_id,
                "reason": (
                    "label is not eligible for the detected session mode; "
                    f"eligible behavior parents are "
                    f"{_label_set_text(eligible_behavior_label_ids)}"
                ),
            })
            continue
        if doc and (
            doc.get("lap_parent_allowed") is False
            or doc.get("annotation_scope") == "subsegment_only"
        ):
            rejected.append({
                "value": label_id,
                "reason": (
                    "label catalog marks this label as not allowed on "
                    "lap parent segments"
                ),
            })
            continue
        allowed.append(label_id)
    allowed = _resolve_same_range_circuit_sections(
        allowed,
        rejected,
        selected_circuit_section_id=selected_circuit_section_id,
    )
    if allowed and not any(
        label_id in eligible_behavior_label_ids for label_id in allowed
    ):
        rejected.append({
            "value": allowed,
            "reason": (
                "saved lap segments require at least one behavior parent "
                f"label from {_label_set_text(eligible_behavior_label_ids)}"
            ),
        })
        return [], rejected
    return allowed, rejected


def _resolve_same_range_circuit_sections(
    label_ids: List[str],
    rejected: List[Dict[str, Any]],
    *,
    selected_circuit_section_id: Optional[str],
) -> List[str]:
    from app.internal_knowledge_base.label_lookup import get_label

    section_groups: Dict[Tuple[str, Tuple[float, float]], List[str]] = {}
    label_docs: Dict[str, Dict[str, Any]] = {}
    for label_id in label_ids:
        doc = get_label(label_id)
        if not doc:
            continue
        label_docs[label_id] = doc
        if doc.get("type") != "circuit_section":
            continue
        section_range = doc.get("normalized_position_range")
        parent_id = doc.get("parent")
        if not parent_id or section_range is None:
            continue
        range_key = tuple(float(v) for v in section_range)
        if len(range_key) != 2:
            continue
        section_groups.setdefault((str(parent_id), range_key), []).append(label_id)

    selected_by_group: Dict[str, str] = {}
    has_pit_stop_label = any(
        label_id == "PS" or label_docs.get(label_id, {}).get("parent") == "PS"
        for label_id in label_ids
    )
    for section_ids in section_groups.values():
        if len(section_ids) <= 1:
            continue
        selected_id = _select_same_range_circuit_section(
            section_ids,
            has_pit_stop_label=has_pit_stop_label,
            selected_circuit_section_id=selected_circuit_section_id,
        )
        for section_id in section_ids:
            selected_by_group[section_id] = selected_id
            if section_id != selected_id:
                rejected.append({
                    "value": section_id,
                    "reason": (
                        "same-range circuit_section ambiguity resolved to "
                        f"{selected_id}"
                    ),
                })

    if not selected_by_group:
        return label_ids
    return [
        label_id for label_id in label_ids
        if selected_by_group.get(label_id, label_id) == label_id
    ]


def _select_same_range_circuit_section(
    section_ids: List[str],
    *,
    has_pit_stop_label: bool,
    selected_circuit_section_id: Optional[str],
) -> str:
    pit_sections = [
        section_id for section_id in section_ids
        if "pit" in LABEL_MAPPING.get(section_id, section_id).lower()
    ]
    non_pit_sections = [
        section_id for section_id in section_ids
        if section_id not in pit_sections
    ]
    if has_pit_stop_label and pit_sections:
        return pit_sections[0]
    if selected_circuit_section_id in section_ids:
        return str(selected_circuit_section_id)
    if not has_pit_stop_label and non_pit_sections:
        return non_pit_sections[0]
    return section_ids[0]
