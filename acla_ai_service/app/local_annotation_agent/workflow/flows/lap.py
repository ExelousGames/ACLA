"""
Lap-section excerpter flow.

The caller has split a lap via the deterministic
``split_lap_by_circuit_sections`` tool. Solo sessions produce
circuit-section ranges; opponent sessions produce only close overtake
offence / defence engagement ranges. One run of this flow annotates ONE
range: the agent inspects telemetry within that split section and submits
a single label proposal.

    build_request(provider_id, prompt_mode, df, lap_start, lap_end, section_id, ...)
    parse(response, prompt_mode, ...) -> LapAnnotationResult
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

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
from app.local_annotation_agent.workflow.preflight import build_preflight_context
from app.local_annotation_agent.workflow.tools import SEARCH_LABELS_TOOL


# ---------------------------------------------------------------------------
# lap_annotation skill — prompt rendering
# ---------------------------------------------------------------------------


def lap_annotation_prompt(session_context: str = "practice") -> str:
    """Per-label `characteristics` block for the lap-flow planner / synthesizer.

    The skill carries only main-label characteristics; circuit and
    circuit_section labels are attached upstream by the splitter and
    never appear here.
    """
    eligible_labels = set(_eligible_behavior_label_ids(session_context))
    labels = [
        entry
        for entry in skills.iter("lap_annotation.labels")
        if entry.get("id") in eligible_labels
    ]
    lines: List[str] = [
        "#### Lap Annotation Skill — Candidate Label Characteristics",
        "",
        "Each candidate parent label below lists the telemetry pattern "
        "that justifies attaching it. The session-specific global rules "
        "at the end are the per-section detection procedure.",
        "",
        _session_context_rule(session_context),
        "",
    ]

    action_model = skills.get("lap_annotation.action_model", {})
    if isinstance(action_model, dict):
        model_lines = [
            str(action_model.get("parent_segment", "")).strip(),
            str(action_model.get("action", "")).strip(),
            str(action_model.get("child_segment", "")).strip(),
        ]
        model_lines = [line for line in model_lines if line]
        if model_lines:
            lines.extend([
                "##### Segment / Action Model",
                *model_lines,
                "",
            ])

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

    global_rules = _mode_specific_global_rules(session_context)
    lines.append(f"##### Global rules — {session_context} mode")
    lines.extend(global_rules)
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

WHOLE_RANGE_LABEL_RULE = (
    "Every behavior, segment-type, and sub-label must describe the final "
    "annotation range as a whole. Do not attach a label because it matches "
    "only one phase, one apex moment, or a short slice inside the range; "
    "omit that label or leave it for detailed child sub-segment annotation."
)

LAP_REASONING_NOTE_RULE = (
    "Write `reasoning` as a longer human annotation note: 4-6 concise "
    "sentences covering the final iloc range, selected label fit, key "
    "telemetry values or trends, deterministic tool verdicts, and why "
    "competing labels were omitted."
)

REQUIRED_BEHAVIOR_PARENT_LABEL_IDS = ("O", "OD", "PS", "RM", "MSP", "MSR")
PRACTICE_BEHAVIOR_PARENT_LABEL_IDS = ("PS", "RM", "MSP")
RACING_BEHAVIOR_PARENT_LABEL_IDS = ("O", "OD", "MSR", "PS")


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
    return (
        RACING_BEHAVIOR_PARENT_LABEL_IDS
        if session_context == "racing"
        else PRACTICE_BEHAVIOR_PARENT_LABEL_IDS
    )


def _label_set_text(label_ids: Tuple[str, ...] | List[str]) -> str:
    return "{" + ", ".join(label_ids) + "}"


def _session_context_rule(session_context: str) -> str:
    if session_context == "racing":
        return (
            "Detected session mode: racing / opponent interaction. Only "
            f"behavior parent labels from {_label_set_text(RACING_BEHAVIOR_PARENT_LABEL_IDS)} "
            "are eligible. Do not evaluate or attach practice-session "
            "behavior parents MSP / RM for this work unit; if no "
            "overtake / defense / racing mistake or pit-stop label fits, "
            "submit an empty label_ids array."
        )
    return (
        "Detected session mode: practice / solo section. Only behavior "
        f"parent labels from {_label_set_text(PRACTICE_BEHAVIOR_PARENT_LABEL_IDS)} "
        "are eligible. Do not evaluate or attach racing-session behavior "
        "parents O / OD / MSR for this work unit."
    )


def _mode_specific_global_rules(session_context: str) -> List[str]:
    mode_rules = skills.get(
        f"lap_annotation.global_rules_by_session.{session_context}",
        [],
    )
    common_rules = skills.get("lap_annotation.global_rules_by_session.common", [])
    if not isinstance(mode_rules, list) or not isinstance(common_rules, list):
        raise RuntimeError(
            "lap_annotation.global_rules_by_session must contain list-valued "
            f"{session_context!r} and 'common' rules"
        )
    rules = [
        str(rule).strip()
        for rule in [*mode_rules, *common_rules]
        if str(rule).strip()
    ]
    if not rules:
        raise RuntimeError(
            "lap_annotation.global_rules_by_session missing rules for "
            f"{session_context!r}"
        )
    return rules


def _mode_exclusion_rule(session_context: str) -> str:
    if session_context == "racing":
        return (
            "Pick at most one opponent-aware behavior parent from {O, OD, MSR}; "
            "O + OD, O + MSR, and OD + MSR are contradictions. PS is allowed "
            "only for pit-lane procedure and should not be combined with "
            "O / OD / MSR."
        )
    return (
        "Pick at most one technical/recovery behavior parent from {MSP, RM}; "
        "MSP + RM is a contradiction, and PS is incompatible with MSP / RM."
    )


def _required_behavior_parent_label_rule(eligible_label_ids: List[str]) -> str:
    return (
        "Every saved lap segment must include at least one behavior parent "
        f"label from {_label_set_text(eligible_label_ids)}. Circuit, "
        "circuit_section, segment-type, sub-label, and EA labels do not "
        "satisfy this required-parent rule. If no required behavior parent "
        "label fits the whole split-section range, submit an empty label_ids "
        "array to drop the range instead of saving a parentless segment."
    )


def _planner_opening(session_context: str) -> str:
    if session_context == "racing":
        return (
            "You are a racing telemetry analyst planning the analysis for "
            "ONE opponent-interaction range of a lap. The deterministic "
            "splitter handed you a fixed iloc boundary around a close "
            "engagement. The synthesizer downstream will pick from the "
            "eligible racing behavior labels by matching telemetry against "
            "each candidate label's `characteristics` block in the skill. "
            "Your job here is to plan the steps that gather that evidence."
        )
    return (
        "You are a racing telemetry analyst planning the analysis for "
        "ONE circuit-section range of a solo/practice lap. The deterministic "
        "splitter handed you a fixed iloc boundary. The synthesizer "
        "downstream will pick from the eligible practice behavior labels by "
        "matching telemetry against each candidate label's `characteristics` "
        "block in the skill. Your job here is to plan the steps that gather "
        "that evidence."
    )


def _planner_task_context(session_context: str) -> str:
    if session_context == "racing":
        return (
            "  2. gather full-range opponent context with "
            "`classify_opponent_interaction` as the mathematical label gate, "
            "including confidence-aware `label_gates` so low / weak results "
            "trigger extra opponent-path evidence; use "
            "`find_nearest_opponent` / `query_opponent_trajectory` when the "
            "primary slot's detailed path decides the technique, and"
        )
    return (
        "  2. gather technical driving evidence with `trajectory_offset`, "
        "`expert_time_difference`, brake / throttle / speed, and "
        "deterministic telemetry queries so time-loss, recovery, or pit "
        "labels are supported by the whole range, and"
    )


def _synth_opening(session_context: str) -> str:
    if session_context == "racing":
        return (
            "You are a racing telemetry analyst producing the final "
            "annotation for ONE opponent-interaction lap range. The "
            "describe_graphs steps captured the evidence below; pick the "
            "eligible racing parent label by matching the interaction "
            "window's telemetry against each candidate label's "
            "`characteristics` block in the skill."
        )
    return (
        "You are a racing telemetry analyst producing the final annotation "
        "for ONE practice / solo lap range. The describe_graphs steps "
        "captured the evidence below; pick the eligible practice parent "
        "label by matching the section's telemetry against each candidate "
        "label's `characteristics` block in the skill."
    )


def _segment_type_label_rule() -> str:
    return str(
        skills.get("sub_label_annotation.category_guidelines.Segment Type", "")
    ).strip()


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


def _range_section_line(
    *,
    section_id: str,
    section_name: str,
    section_split_basis: Optional[str],
    opponent_interaction: Optional[dict],
    for_local_synth: bool = False,
) -> str:
    preselected_section_id = _preselected_interaction_section_id(opponent_interaction)
    if preselected_section_id:
        preselected_name = LABEL_MAPPING.get(
            preselected_section_id, preselected_section_id,
        )
        prefix = "- circuit_section id" if for_local_synth else "- circuit_section"
        return (
            f"{prefix}: `{preselected_section_id}` ({preselected_name}) "
            "from the splitter's opponent sub-segment context; include it "
            "in label_ids."
        )
    if "interaction" in str(section_split_basis or ""):
        return (
            "- circuit_section id: not preselected for this interaction "
            "window; call `locate_circuit_section` if a named section label "
            "is needed."
            if for_local_synth
            else "- racing interaction window: circuit_section is not "
            "preselected; use `locate_circuit_section` only as context."
        )
    return (
        f"- circuit_section id: `{section_id}` "
        "(located by the splitter; include it in label_ids)"
        if for_local_synth
        else f"- circuit_section: `{section_id}` ({section_name}) "
        "— located by the splitter; the synthesizer emits it (with the "
        "circuit) as a label."
    )


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
        "\nTarget-car hint from the splitter:\n" + "\n".join(target_lines) + "\n"
        if target_lines else ""
    )
    return (
        "\n#### Opponent-session focus\n"
        f"This split range {origin}. For this work unit, identify the "
        "target car from the splitter evidence first, then do ONLY overtake "
        "offence / defense "
        "annotation: pick O for a successful attacking pass, OD for a held "
        "defense, or MSR for a failed attack / broken defense. If the "
        "opponent evidence is only close-following/draft context, or the "
        "opponent stays tucked directly behind without a lateral/alongside "
        "threat, submit "
        "`label_ids: []` rather than labeling normal practice-driving "
        "telemetry such as EA / MSP / RM. PS is still valid when the "
        "range has pit-lane procedure evidence.\n"
        f"{target_block}"
    )


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
    section_split_basis: Optional[str],
    opponent_interaction: Optional[dict],
) -> str:
    from app.shared.annotation_agent_tools import (
        AGENT_GRAPH_DEFINITIONS,
        PIPELINE_TOOL_DEFINITIONS,
    )

    section_name = LABEL_MAPPING.get(section_id, section_id) if section_id else section_id
    session_context = _session_context(section_split_basis, opponent_interaction)
    eligible_labels = list(_eligible_behavior_label_ids(session_context))

    graph_catalogue = ", ".join(
        f"`{gdef['id']}` ({gdef['title']})"
        for gdef in AGENT_GRAPH_DEFINITIONS
    )
    tool_catalogue_lines = [
        f"- `{t['id']}` — {t['label']}: {t['description']}"
        for t in PIPELINE_TOOL_DEFINITIONS
    ]

    lap_skill_block = lap_annotation_prompt(session_context)
    interaction_focus = _interaction_focus_block(
        section_split_basis, opponent_interaction,
    )

    parts = [
        _planner_opening(session_context),
        "",
        "#### Lap context",
        f"- Circuit: {circuit_id}",
        f"- Lap range: [{lap_start}, {lap_end}] (length {lap_end - lap_start})",
        "",
        "#### Range under review",
        f"- detected session mode: {session_context}",
        f"- eligible behavior parent labels: {_label_set_text(eligible_labels)}",
        _range_section_line(
            section_id=section_id,
            section_name=section_name,
            section_split_basis=section_split_basis,
            opponent_interaction=opponent_interaction,
        ),
        f"- split iloc boundary: [{section_start}, {section_end}] "
        f"(length {section_end - section_start})",
        f"- split basis: {section_split_basis or 'circuit_section'}",
        interaction_focus,
        "",
        lap_skill_block,
        "",
        "#### Available Step-Solver Agents",
        "The Required Upfront Annotation Preflight block above has already "
        "run the standard deterministic tools and hybrid semantic label "
        "search. Plan only graph-description or targeted verifier work "
        "needed to refine that context.",
        "Each plan step is dispatched to ONE sub-agent.",
        "- `describe_graphs` — renders the listed graphs over the split "
        "boundary and writes one observation paragraph per graph.",
        "- `label_verifier` — embedding-similarity filter against the "
        "candidate labels using the describe_graphs observations. End your "
        "plan with one label_verifier step.",
        "",
        "#### Available Graph IDs",
        graph_catalogue,
        "",
        "#### Available Pre-Compute Tools",
        "The standard tool group already ran in preflight. Add tools here "
        "only for a concrete gap discovered after reading that package.",
        *tool_catalogue_lines,
        "",
        "#### Task",
        "Plan describe_graphs steps that start from preflight and gather "
        "only the remaining evidence needed to:",
        "  1. score only the eligible behavior parent labels "
        f"{_label_set_text(eligible_labels)} against "
        "its `characteristics` block in the skill, and",
        _planner_task_context(session_context),
        "  3. optionally identify the base segment shape and corner shape "
        "if the segment-type picks would be unambiguous, while recording "
        "entry/apex/exit altitude only as subsegment evidence.",
        "Keep the plan tight — typically 1-3 describe_graphs steps plus a "
        "label_verifier. `trajectory_offset` + `time_delta` are the two "
        "diagnostic graphs called out by the skill; add `altitude_profile` "
        "and `measure_segment_shape` only when preflight leaves segment "
        "shape or altitude unresolved. For corner-entry or corner-exit segment "
        "completeness, include brake / throttle graphs and deterministic "
        "queries that establish the full driver-vs-expert action bounds "
        "required by `lap_annotation.global_rules_by_session.common`.",
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
        '"Confirm boundary + full entry/exit brake/throttle action bounds.", "requested_graphs": '
        '["trajectory_offset", "brake", "throttle"], "tools": '
        '["compute_expert_phases"]},',
        '    {"step_id": 2, "agent": "describe_graphs", "description": '
        '"Measure trajectory and altitude for segment-type evidence.", "requested_graphs": '
        '["trajectory_detailed", "altitude_profile"], "tools": '
        '["measure_segment_shape"]},',
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
    section_split_basis: Optional[str],
    opponent_interaction: Optional[dict],
    verified_labels: List[str],
) -> Tuple[str, str]:
    session_context = _session_context(section_split_basis, opponent_interaction)
    eligible_labels = list(_eligible_behavior_label_ids(session_context))
    required_label_rule = _required_behavior_parent_label_rule(eligible_labels)
    lap_skill_block = lap_annotation_prompt(session_context)
    interaction_focus = _interaction_focus_block(
        section_split_basis, opponent_interaction,
    )
    verified_inline = (
        ", ".join(verified_labels) if verified_labels
        else "(none — emit an empty label_ids array to drop this range)"
    )

    intro = "\n".join([
        _synth_opening(session_context),
        "",
        "#### Range under review",
        f"- detected session mode: {session_context}",
        f"- eligible behavior parent labels: {_label_set_text(eligible_labels)}",
        _range_section_line(
            section_id=section_id,
            section_name=LABEL_MAPPING.get(section_id, section_id),
            section_split_basis=section_split_basis,
            opponent_interaction=opponent_interaction,
            for_local_synth=True,
        ),
        f"- circuit id: `{circuit_id}` "
        "(from Static_track; include it in label_ids)",
        f"- split iloc boundary: [{section_start}, {section_end}]",
        f"- split basis: {section_split_basis or 'circuit_section'}",
        f"- lap range: [{lap_start}, {lap_end}]",
        interaction_focus,
        "",
        lap_skill_block,
    ])

    outro = "\n".join([
        "#### Candidate label IDs",
        f"The shortlist retrieved for this section is: {verified_inline}. "
        "Pick the parent label(s) from this shortlist by matching the "
        "section's telemetry against each candidate's `characteristics` "
        "block in the skill. Every saved segment must include at least "
        f"one required behavior parent from {_label_set_text(eligible_labels)}; "
        "circuit, circuit_section, segment-type, sub-label, and EA labels "
        "do not satisfy this requirement. Segment-type picks are OPTIONAL — include "
        "only lap-parent-allowed labels whose base shape or corner-shape "
        "evidence is unambiguous. Do not include any label whose catalog "
        "metadata says `lap_parent_allowed: false`. If no required "
        "behavior parent in the shortlist fits the whole final range, "
        "return an empty `label_ids` array to drop this range. "
        f"{_mode_exclusion_rule(session_context)}",
        "",
        "#### Whole-range fit rule",
        WHOLE_RANGE_LABEL_RULE,
        "",
        "#### Output format",
        "Respond with ONE JSON object only — no surrounding prose. Schema:",
        "```json",
        "{",
        '  "label_ids": ["<id>", ...],',
        '  "reasoning": "<4-6 sentence human-readable evidence note citing ilocs, values, trends, tool verdicts, and range-fit rationale>"',
        "}",
        "```",
        "Hard rules:",
        f"- The submitted range is fixed to [{section_start}, {section_end}]; "
        "submit labels only for that exact range.",
        f"- {required_label_rule}",
        "- Every main / segment-type / sub label_id must come from the shortlist "
        "above; additionally include the circuit id. Include a "
        "circuit_section id only when it was listed under 'Range under "
        "review' or returned by `locate_circuit_section`.",
        f"- {WHOLE_RANGE_LABEL_RULE}",
        "- Apply the segment/action model and segment completeness rules "
        "from `lap_annotation.action_model` and "
        "`lap_annotation.global_rules_by_session.common`. Do not treat "
        "complete action-group evidence as optional sub-label evidence.",
        f"- {LAP_REASONING_NOTE_RULE}",
        f"- {_segment_type_label_rule()}",
        "- An empty label_ids array is the valid 'drop this section' signal, "
        "not a saved annotation without a required behavior parent.",
    ])

    return intro, outro


# ---------------------------------------------------------------------------
# Tool-agent task prompt
# ---------------------------------------------------------------------------


def _tool_agent_task_prompt(
    *,
    lap_start: int,
    lap_end: int,
    section_start: int,
    section_end: int,
    section_split_basis: Optional[str],
    opponent_interaction: Optional[dict],
) -> str:
    session_context = _session_context(section_split_basis, opponent_interaction)
    eligible_labels = list(_eligible_behavior_label_ids(session_context))
    required_label_rule = _required_behavior_parent_label_rule(eligible_labels)
    if session_context == "racing":
        mode_submit_rule = (
            "- In opponent-interaction windows, use O / OD / MSR, or PS "
            "when pit-lane procedure evidence fits the whole range; submit "
            "an empty `label_ids` array to drop the range when none fit.\n"
        )
    else:
        mode_submit_rule = (
            "- In practice / solo sections, use MSP / RM / PS or submit "
            "an empty `label_ids` array to drop the range.\n"
        )
    interaction_focus = _interaction_focus_block(
        section_split_basis, opponent_interaction,
    )
    preselected_section_id = _preselected_interaction_section_id(opponent_interaction)
    preselected_section_block = ""
    if preselected_section_id:
        preselected_section_block = (
            "- Splitter section context: "
            f"`{preselected_section_id}` "
            f"({LABEL_MAPPING.get(preselected_section_id, preselected_section_id)}) "
            "from the opponent sub-segment; include it in `label_ids`.\n"
        )

    return (
        "Annotate ONE lap range. The deterministic splitter handed you a "
        "fixed split-section boundary; if this is an opponent interaction window, "
        "the boundary is event-shaped and circuit sections are context only. "
        "Your job is to pick the circuit id, optional circuit_section id, "
        "at least one required behavior parent label for any saved segment, optional "
        "lap-parent-allowed segment-type labels, and an optional matching "
        "sub-label.\n"
        "\n"
        "### Lap context\n"
        f"- Detected session mode: {session_context}\n"
        f"- Eligible behavior parent labels: {_label_set_text(eligible_labels)}\n"
        f"- Lap range: [{lap_start}, {lap_end}] "
        f"(length {lap_end - lap_start})\n"
        f"- Split section boundary: [{section_start}, {section_end}] "
        f"(length {section_end - section_start})\n"
        f"- Split basis: {section_split_basis or 'circuit_section'}\n"
        f"{preselected_section_block}"
        f"{interaction_focus}"
        "\n"
        "### How to work\n"
        "1. Use the Required Upfront Annotation Preflight block as the "
        "primary evidence package. It already contains deterministic tool "
        "outputs, tool output tags, and semantic label candidates from "
        "hybrid search.\n"
        "2. Use `search_annotation_guidance` or extra data tools only for a "
        "specific missing detail, not to rediscover the basic analysis path "
        "already covered by preflight.\n"
        "3. Use `search_labels` only for a targeted semantic re-query when "
        "the preflight candidates miss a specific observation. Include "
        "relevant `tool_output_tags` in the query. Query `types=\"main\"` "
        "for the required behavior parent "
        f"label from {_label_set_text(eligible_labels)}, "
        "`types=\"segment_type\"` for segment-type labels, and "
        "`parent_id` for sub-labels under a chosen main label. Do not "
        "submit labels whose returned catalog metadata says "
        "`lap_parent_allowed: false`.\n"
        "4. Tools are scoped to the split section boundary. When any "
        "main-label, segment-type, or sub-label signature fits only part "
        "of the range, omit that label or submit an empty `label_ids` array "
        "to drop the range.\n"
        "5. Call `submit_result(payload_json, summary)` once with the "
        "chosen IDs, using the same longer evidence note for `summary`, "
        "and stop after it returns `ok: true`.\n"
        "\n"
        "### Whole-range fit rule\n"
        f"{WHOLE_RANGE_LABEL_RULE}\n"
        "\n"
        "### Submit payload shape\n"
        "`payload_json` must be a JSON object of this shape:\n"
        "```json\n"
        "{\n"
        '  "label_ids": ["<id>", "<id>", ...],\n'
        '  "reasoning": "<4-6 sentence human-readable evidence note citing ilocs, values, trends, tool verdicts, and range-fit rationale>"\n'
        "}\n"
        "```\n"
        "`label_ids` carries the circuit id, optional circuit_section id, "
        "and your main / segment-type / sub picks together. Every saved "
        f"segment must contain at least one of {_label_set_text(eligible_labels)}; "
        "otherwise submit an empty `label_ids` array as the valid "
        "'drop this section' signal. The runner reports back "
        "the split section range.\n"
        "\n"
        "### Hard rules\n"
        f"- Final range is fixed to [{section_start}, {section_end}].\n"
        f"- {required_label_rule}\n"
        "- Do not invent label IDs; circuit / circuit_section ids must come "
        "from splitter context or capability results, every other id from "
        "preflight semantic candidates or a targeted `search_labels` response.\n"
        "- Include a circuit_section id only when it is unambiguous.\n"
        f"- {_mode_exclusion_rule(session_context)}\n"
        f"- {WHOLE_RANGE_LABEL_RULE}\n"
        "- Apply the segment/action model and segment completeness rules "
        "from `lap_annotation.action_model` and "
        "`lap_annotation.global_rules_by_session.common`. Do not treat "
        "complete action-group evidence as optional sub-label evidence.\n"
        f"- {LAP_REASONING_NOTE_RULE}\n" +
        mode_submit_rule +
        "- For time-delta and offset evidence, cite deterministic tool "
        "verdict fields (unit, label-significance, end-window trend); do not "
        "create strength judgments from raw numbers.\n"
        f"- {_segment_type_label_rule()}\n"
        "- Sub-labels require their parent main label in `label_ids`.\n"
        "- One proposal per session — do NOT annotate downstream sections.\n"
        "- Budget tool calls: a typical section needs 7-10 calls total."
    )


# ---------------------------------------------------------------------------
# Public flow API
# ---------------------------------------------------------------------------


def build_request(
    *,
    provider_id: str,
    prompt_mode: str,
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

    fixed_label_ids = [label_id for label_id in (circuit_id, section_id) if label_id]
    preselected_section_id = _preselected_interaction_section_id(opponent_interaction)
    if preselected_section_id:
        fixed_label_ids.append(preselected_section_id)
    preflight = build_preflight_context(
        flow="lap",
        df=df,
        start=section_start,
        end=section_end,
        eligible_behavior_label_ids=eligible_labels,
        fixed_label_ids=fixed_label_ids,
        extra_query_terms=[
            session_context,
            section_split_basis or "circuit_section",
            LABEL_MAPPING.get(section_id, section_id),
        ],
    )

    # parent_start/end on the request are the section range — sub-agents
    # like describe_graphs operate over this window.
    parent_start = int(section_start)
    parent_end = int(section_end)

    if prompt_mode == "local_pipeline":
        planner_prompt = _local_planner_prompt(
            lap_start=lap_start,
            lap_end=lap_end,
            section_id=section_id,
            section_start=section_start,
            section_end=section_end,
            circuit_id=circuit_id,
            section_split_basis=section_split_basis,
            opponent_interaction=opponent_interaction,
        )
        planner_prompt = "\n\n".join([preflight.prompt_block, planner_prompt])
        synth_prompt: Callable[[Dict[str, Any]], Tuple[str, str]] = (
            lambda s: _local_synth_prompts(
                lap_start=lap_start,
                lap_end=lap_end,
                section_id=section_id,
                section_start=section_start,
                section_end=section_end,
                circuit_id=circuit_id,
                section_split_basis=section_split_basis,
                opponent_interaction=opponent_interaction,
                verified_labels=_verified_label_ids_from_state(s),
            )
        )
        extra_state = {
            "root_agent": "annotation_root",
            "annotation_session_context": session_context,
            "eligible_behavior_label_ids": eligible_labels,
        }
    else:
        planner_prompt = _tool_agent_task_prompt(
            lap_start=lap_start,
            lap_end=lap_end,
            section_start=section_start,
            section_end=section_end,
            section_split_basis=section_split_basis,
            opponent_interaction=opponent_interaction,
        )
        planner_prompt = "\n\n".join([preflight.prompt_block, planner_prompt])
        synth_prompt = lambda _state: ("", "")
        extra_state = {
            "root_agent": "annotation_root",
            "tool_agent_extra_tools": [SEARCH_LABELS_TOOL],
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
    prompt_mode: str,
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

    ``prompt_mode="local_pipeline"`` expects the JSON schema with label_ids
    + reasoning. ``prompt_mode="tool_agent"`` reads the submit payload.

    Returns only what the LLM committed to — including the circuit and
    circuit_section ids the LLM picked via ``get_circuit_id`` /
    ``locate_circuit_section``.
    """
    session_context = _session_context(section_split_basis, opponent_interaction)
    eligible_labels = list(_eligible_behavior_label_ids(session_context))

    if prompt_mode == "tool_agent":
        return _parse_claude(
            response, section_id, section_start, section_end, circuit_id,
            opponent_interaction, eligible_labels,
        )
    return _parse_local(
        response, section_id, section_start, section_end, circuit_id,
        opponent_interaction, eligible_labels,
    )


def _parse_local(
    response: AgentResponse,
    section_id: str,
    section_start: int,
    section_end: int,
    circuit_id: Optional[str],
    opponent_interaction: Optional[dict],
    eligible_behavior_label_ids: List[str],
) -> LapAnnotationResult:
    raw = response.raw_response or ""
    parsed = parse_json_response(raw)
    if not parsed:
        raise RuntimeError(
            f"lap flow (local): synth response was not valid JSON. "
            f"First 300 chars: {raw[:300]!r}"
        )
    _reject_unknown_output_fields(parsed, "local")

    new_start = int(section_start)
    new_end = int(section_end)
    if (new_end - new_start) < 5:
        raise RuntimeError(
            f"lap flow (local): split section range too short "
            f"({new_end - new_start} ilocs) — minimum 5"
        )

    raw_label_ids = parsed.get("label_ids") or []
    cleaned, rejected = _clean_label_ids(
        raw_label_ids,
        eligible_behavior_label_ids=eligible_behavior_label_ids,
    )
    cleaned = _with_preselected_interaction_labels(
        cleaned, circuit_id, opponent_interaction,
    )

    reasoning = str(parsed.get("reasoning") or "")

    return LapAnnotationResult(
        section_id=section_id,
        start_index=new_start,
        end_index=new_end,
        label_ids=cleaned,
        reasoning=reasoning or raw or "(no reasoning)",
        submitted=True,
        rejected_proposals=rejected,
        rendered_images=list(response.graph_images),
        transcript=raw,
        tool_calls=0,
    )


def _parse_claude(
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
        _reject_unknown_output_fields(parsed, "claude")
        raw_label_ids = parsed.get("label_ids") or []
        cleaned, rejected = _clean_label_ids(
            raw_label_ids,
            eligible_behavior_label_ids=eligible_behavior_label_ids,
        )
        cleaned = _with_preselected_interaction_labels(
            cleaned, circuit_id, opponent_interaction,
        )
        reasoning = str(parsed.get("reasoning") or "")

    new_start, new_end = int(section_start), int(section_end)
    if (new_end - new_start) < 5:
        raise RuntimeError(
            f"lap flow (claude): split section range too short "
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


def _reject_unknown_output_fields(parsed: Dict[str, Any], source: str) -> None:
    allowed_keys = {"label_ids", "reasoning"}
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
) -> Tuple[List[str], List[Dict[str, Any]]]:
    cleaned, rejected, _ = normalize_grouped_label_ids(raw_label_ids)
    if not cleaned:
        return cleaned, rejected

    from app.internal_knowledge_base.label_lookup import get_label

    eligible_behavior_label_ids = list(
        eligible_behavior_label_ids or REQUIRED_BEHAVIOR_PARENT_LABEL_IDS
    )
    allowed: List[str] = []
    for label_id in cleaned:
        doc = get_label(label_id)
        parent_id = doc.get("parent") if doc else None
        if (
            label_id in REQUIRED_BEHAVIOR_PARENT_LABEL_IDS
            and label_id not in eligible_behavior_label_ids
        ) or (
            parent_id in REQUIRED_BEHAVIOR_PARENT_LABEL_IDS
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


def _with_preselected_interaction_labels(
    label_ids: List[str],
    circuit_id: Optional[str],
    opponent_interaction: Optional[dict],
) -> List[str]:
    """Add splitter-provided circuit context to non-empty interaction labels."""
    if not label_ids:
        return label_ids

    section_id = _preselected_interaction_section_id(opponent_interaction)
    if not section_id:
        return label_ids

    prefix: List[str] = []
    if circuit_id and circuit_id in LABEL_MAPPING:
        prefix.append(circuit_id)
    prefix.append(section_id)

    merged: List[str] = []
    for label_id in [*prefix, *label_ids]:
        if label_id not in merged:
            merged.append(label_id)
    return merged
