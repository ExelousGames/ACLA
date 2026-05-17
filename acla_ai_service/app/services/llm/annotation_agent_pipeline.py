"""
Pipeline runners for the ``annotation_root`` agent.

There is ONE annotation agent — ``annotation_root`` in
``app.services.llm.agents.annotation_root``. It is flow-agnostic: it runs
the planner → step_solvers (describe_graphs + label_verifier) →
synthesizer cycle, and emits a raw evaluator-finalised string in
``state['final_synth_response']``.

This module owns the two flow-specific wrappers:

* ``run_annotation_pipeline`` — sub-segment discovery (detailed flow).
* ``run_lap_annotation_pipeline`` — lap-to-segment excerpter (manual flow).

Each wrapper builds the flow's planner prompt + synthesizer intro/outro,
seeds the initial pool attachment, invokes the same compiled graph, and
parses the agent's raw response into its flow-specific result type.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from app.models.label_catalog import LabelCatalog, get_label_catalog
from app.models.lap_annotation_skill import get_lap_skill
from app.models.segment_models import LABEL_CATEGORIES, LABEL_MAPPING
from app.services.llm.agent_framework import AGENT_REGISTRY
from app.services.llm.step_evaluator_agents import (
    PipelineAttachment,
    get_active_stage,
    set_eval_llm,
    set_step_event_callback,
)

# Side-effect import: registers all agents (including annotation_root).
import app.services.llm.agents  # noqa: F401

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


@dataclass
class AnnotationPipelineConfig:
    """Configuration for the annotation pipeline."""

    max_iterations: int = 3
    max_new_tokens: int = 1500
    temperature: float = 0.7

    # Backend selector — "local" (llama-server) or "claude" (claude-agent-sdk).
    backend: str = "local"

    # llama-cpp VLM settings (only used when backend == "local")
    gguf_path: Optional[str] = None
    mmproj_path: Optional[str] = None
    context_size: int = 32768
    n_gpu_layers: int = -1

    # HuggingFace source repo + conversion (only used when backend == "local")
    hf_repo: str = "Qwen/Qwen2.5-VL-72B-Instruct"
    quantization_type: str = "Q4_K_M"

    # Claude backend settings (only used when backend == "claude")
    claude_model: str = "claude-sonnet-4-6"
    claude_use_thinking: bool = False


@dataclass
class AnnotationResult:
    """Result returned by ``run_annotation_pipeline`` (detailed flow)."""

    final_labels: List[str]
    final_reasoning: str
    accepted: bool
    iterations: int
    messages: List[dict]
    graph_images: List[bytes] = field(default_factory=list)  # PNG bytes
    sub_start: Optional[int] = None
    sub_end: Optional[int] = None
    # Per-label proposals from the synthesizer. Each entry:
    #   {label_id, start_index, end_index, reasoning}
    # The UI groups these by (start_index, end_index) to materialise one
    # sub-segment per AI-discovered range, rather than collapsing them all
    # into the union span [sub_start, sub_end].
    label_annotations: List[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Shared backend wiring
# ---------------------------------------------------------------------------


def _wire_vlm_backend(
    config: AnnotationPipelineConfig,
    *,
    vlm_stream_callback: Optional[Callable],
    vlm_prompt_callback: Optional[Callable],
    vlm_reasoning_callback: Optional[Callable],
    step_event_callback: Optional[Callable],
) -> None:
    """Bind the VLM / LLM callables that ``annotation_root`` reads.

    Builds the backend (local llama-server or Claude SDK) and registers
    the per-call closures via ``set_eval_llm`` / ``set_step_event_callback``
    so every node in the LangGraph subgraph shares one VLM client.
    """
    if config.backend == "claude":
        from app.services.llm.claude_agent_backend import (
            get_or_start_claude_backend,
        )
        vlm_service = get_or_start_claude_backend(
            model=config.claude_model,
            use_thinking=config.claude_use_thinking,
        )
    else:
        from app.services.llm.annotation_agent_llm_service import (
            AnnotationAgentLLMConfig,
            get_or_start_service,
        )
        agent_llm_config = AnnotationAgentLLMConfig(
            gguf_path=config.gguf_path,
            mmproj_path=config.mmproj_path,
            context_size=config.context_size,
            n_gpu_layers=config.n_gpu_layers,
            hf_repo=config.hf_repo,
            quantization_type=config.quantization_type,
        )
        vlm_service = get_or_start_service(agent_llm_config)

    def vlm_generate(prompt: str, images: Optional[List[bytes]] = None) -> str:
        if vlm_prompt_callback:
            vlm_prompt_callback(prompt, get_active_stage())
        return vlm_service.generate(
            prompt,
            images=images,
            max_tokens=config.max_new_tokens,
            temperature=config.temperature,
            stream_callback=vlm_stream_callback,
            reasoning_callback=vlm_reasoning_callback,
        )

    def llm_generate(prompt: str) -> str:
        if vlm_prompt_callback:
            vlm_prompt_callback(prompt, get_active_stage())
        return vlm_service.generate(
            prompt,
            images=None,
            max_tokens=config.max_new_tokens,
            temperature=0.1,
            stream_callback=vlm_stream_callback,
            reasoning_callback=vlm_reasoning_callback,
        )

    set_eval_llm(vlm_generate, llm_generate)
    set_step_event_callback(step_event_callback)


def _stream_graph(
    initial_state: Dict[str, Any],
    progress_callback: Optional[Callable],
    progress_detail_fn: Callable[[str, Dict[str, Any]], str],
) -> Dict[str, Any]:
    """Stream the compiled ``annotation_root`` graph and accumulate state."""
    graph = AGENT_REGISTRY["annotation_root"]
    final_state: Dict[str, Any] = dict(initial_state)
    for event in graph.stream(initial_state, config={"recursion_limit": 100}):
        for node_name, node_output in event.items():
            if not isinstance(node_output, dict):
                continue
            final_state.update(node_output)
            if progress_callback:
                detail = progress_detail_fn(node_name, final_state)
                progress_callback(node_name, detail)
    return final_state


def _base_root_state(
    df,
    *,
    parent_start: int,
    parent_end: int,
    parent_main_labels: List[str],
    existing_children: List[dict],
    session_id: str = "",
) -> Dict[str, Any]:
    """Build the framework-level state fields shared by every flow."""
    return {
        "segment_data": {
            "session_id": session_id,
            "start_index": int(parent_start),
            "end_index": int(parent_end),
        },
        "df_ref": df,
        "parent_main_labels": list(parent_main_labels),
        "existing_children": list(existing_children),
        "parent_start": int(parent_start),
        "parent_end": int(parent_end),
        "available_labels": LABEL_CATEGORIES,
        "plan": "",
        "plan_steps": [],
        "current_step_index": 0,
        "step_results": [],
        "all_graph_images": [],
        "all_graph_descriptions": [],
        "attachment_pool": {},
        "verified_labels": [],
        "verified_label_reasoning": {},
        "evaluation": "",
        "final_synth_response": "",
        "messages": [],
        "depth": 0,
        "call_stack": [],
        "spawn_log": [],
        "total_spawns": 0,
    }


# ---------------------------------------------------------------------------
# JSON response parsing helpers
# ---------------------------------------------------------------------------


def _parse_json_response(raw: str) -> Optional[dict]:
    """Best-effort extraction of a JSON object from a synthesizer response."""

    def _try_loads(s: str) -> Optional[dict]:
        s = s.strip()
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            pass
        try:
            fixed = re.sub(
                r'"((?:[^"\\]|\\.)*)"',
                lambda m: '"' + m.group(1).replace('\n', '\\n').replace('\r', '\\r') + '"',
                s,
            )
            return json.loads(fixed)
        except (json.JSONDecodeError, re.error):
            pass
        return None

    try:
        json_str = raw
        if "```json" in json_str:
            json_str = json_str.split("```json")[1].split("```")[0]
        elif "```" in json_str:
            json_str = json_str.split("```")[1].split("```")[0]
        result = _try_loads(json_str)
        if result is not None:
            return result
    except (IndexError, KeyError):
        pass

    brace_match = re.search(r'\{[\s\S]*\}', raw)
    if brace_match:
        result = _try_loads(brace_match.group())
        if result is not None:
            return result

    return None


def _validate_and_fix_hierarchy(
    labels: List[str],
    catalog: Optional[LabelCatalog] = None,
) -> Tuple[List[str], List[str]]:
    """Validate label hierarchy and auto-insert missing parents."""
    cat = catalog or get_label_catalog()
    fixed = list(dict.fromkeys(labels))
    warnings: List[str] = []

    to_add: List[str] = []
    for lid in fixed:
        parent = cat.parent_of.get(lid)
        if parent and parent not in fixed and parent not in to_add:
            to_add.append(parent)
            warnings.append(
                f"Auto-inserted parent '{parent}' "
                f"({LABEL_MAPPING.get(parent, parent)}) for sub-label "
                f"'{lid}' ({LABEL_MAPPING.get(lid, lid)})."
            )
    fixed = to_add + fixed

    rules = cat.get_hierarchy_rules(fixed)
    for conflict in rules["exclusive_conflicts"]:
        a, b = conflict["labels"]
        warnings.append(
            f"Conflict: '{a}' ({LABEL_MAPPING.get(a, a)}) and "
            f"'{b}' ({LABEL_MAPPING.get(b, b)}) are mutually exclusive."
        )
    return list(dict.fromkeys(fixed)), warnings


# ---------------------------------------------------------------------------
# Detailed flow — sub-segment discovery
# ---------------------------------------------------------------------------


def _detailed_planner_prompt(
    *,
    parent_start: int,
    parent_end: int,
    parent_main_labels: List[str],
    existing_children: List[dict],
) -> str:
    """Build the full planner prompt for the detailed (sub-segment) flow."""
    from app.services.llm.annotation_agent_tools import (
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
    catalog = get_label_catalog()
    label_descriptions: List[str] = []
    annotation_guidelines: List[str] = []
    for label_id in parent_main_labels:
        label_def = catalog.get_label(label_id)
        if label_def and label_def.description:
            label_descriptions.append(
                f"  - {label_def.name} ({label_id}): {label_def.description}"
            )
        if label_def and label_def.annotation_guideline:
            annotation_guidelines.append(
                f"  [{label_def.name}]\n  {label_def.annotation_guideline}"
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
        "Each plan step is dispatched to ONE sub-agent. The planner may "
        "only request `describe_graphs` here — a `label_verifier` step "
        "is appended automatically by the framework after your plan.",
        "- `describe_graphs` — renders the telemetry graphs listed in "
        "`requested_graphs` and writes a precise observation paragraph "
        "per graph. Pure observation — does not diagnose or assign labels.",
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
        "region where a specific event or behaviour occurs.",
        "",
        "Your plan must be a JSON object with a single key \"steps\". "
        "Each step object must have:",
        "  - \"step_id\": integer (1, 2, 3, ...).",
        "  - \"agent\": always \"describe_graphs\".",
        "  - \"description\": string describing the goal of the step.",
        "  - \"requested_graphs\": list of graph IDs from the catalogue above.",
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
        '"requested_graphs": ["speed", "throttle"], "tools": []}',
        "  ]",
        "}",
        "```",
    ])
    return "\n".join(parts)


def _detailed_synth_prompts(
    *,
    parent_start: int,
    parent_end: int,
    verified_labels: List[str],
) -> Tuple[str, str]:
    """Build (synth_prompt_intro, synth_prompt_outro) for the detailed flow."""
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
        "start_index and end_index.",
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
        "evidence. Omit both fields when \"proved\" is false.",
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
        f"verified candidate ({n_verified} entries total). The schema "
        "below shows BOTH entry shapes — proved-true (with indices) and "
        "proved-false (without). Output strict JSON only — no comments, "
        "no trailing commas, no extra keys.",
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


def _parse_detailed_response(
    raw: str,
    parent_start: int,
    parent_end: int,
) -> Tuple[List[str], List[dict], int, int, str]:
    """Extract (fixed_labels, label_proposals, sub_start, sub_end, reasoning)."""
    sub_labels: List[str] = []
    label_proposals: List[dict] = []
    proposed_start = parent_start
    proposed_end = parent_end
    reasoning = raw

    parsed = _parse_json_response(raw)
    if not parsed:
        LOGGER.warning("Proposal synthesizer response was not valid JSON.")
        return [], [], proposed_start, proposed_end, reasoning

    label_annotations = parsed.get("labels", [])
    if not isinstance(label_annotations, list):
        return [], [], proposed_start, proposed_end, reasoning

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

    fixed_labels, hierarchy_warnings = _validate_and_fix_hierarchy(sub_labels)
    for w in hierarchy_warnings:
        LOGGER.info("Hierarchy fix: %s", w)

    return fixed_labels, label_proposals, proposed_start, proposed_end, reasoning


def run_annotation_pipeline(
    df,
    start_index: int,
    end_index: int,
    session_id: str,
    parent_main_labels: List[str],
    existing_children: Optional[List[dict]] = None,
    config: Optional[AnnotationPipelineConfig] = None,
    progress_callback=None,
    vlm_stream_callback: Optional[Callable] = None,
    vlm_prompt_callback: Optional[Callable] = None,
    vlm_reasoning_callback: Optional[Callable] = None,
    step_event_callback: Optional[Callable] = None,
) -> AnnotationResult:
    """Execute the detailed (sub-segment discovery) pipeline.

    Builds the detailed-flow planner + synthesizer prompts, seeds the
    initial pool with ``init.parent_segment``, invokes ``annotation_root``,
    and parses the synthesizer's raw response into per-label proposals.
    """
    config = config or AnnotationPipelineConfig()
    existing_children = existing_children or []

    _wire_vlm_backend(
        config,
        vlm_stream_callback=vlm_stream_callback,
        vlm_prompt_callback=vlm_prompt_callback,
        vlm_reasoning_callback=vlm_reasoning_callback,
        step_event_callback=step_event_callback,
    )

    # Build the prompts the agent reads via state.
    planner_prompt = _detailed_planner_prompt(
        parent_start=start_index,
        parent_end=end_index,
        parent_main_labels=parent_main_labels,
        existing_children=existing_children,
    )

    # Seed init.parent_segment so describe_graphs / label_verifier can
    # consume it from their sliced pools.
    parent_segment = PipelineAttachment(
        name="init.parent_segment",
        kind="structured",
        content_schema="parent_segment",
        label="Parent Segment",
        content={
            "parent_start": int(start_index),
            "parent_end": int(end_index),
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

    initial_state = _base_root_state(
        df,
        parent_start=start_index,
        parent_end=end_index,
        parent_main_labels=parent_main_labels,
        existing_children=existing_children,
        session_id=session_id,
    )
    initial_state["planner_prompt"] = planner_prompt
    initial_state["initial_attachments"] = [parent_segment]
    # Defer synth prompt construction to synth time — the synthesizer
    # invokes these callables with the current state (which by then has
    # verified_labels populated by label_verifier).
    initial_state["synth_prompt_intro"] = lambda s: _detailed_synth_prompts(
        parent_start=start_index,
        parent_end=end_index,
        verified_labels=s.get("verified_labels", []),
    )[0]
    initial_state["synth_prompt_outro"] = lambda s: _detailed_synth_prompts(
        parent_start=start_index,
        parent_end=end_index,
        verified_labels=s.get("verified_labels", []),
    )[1]

    def _detailed_progress(node_name: str, state: Dict[str, Any]) -> str:
        if node_name == "planner":
            return f"Planned {len(state.get('plan_steps', []))} step(s)"
        if node_name == "executor":
            idx = state.get("current_step_index", 0)
            total = len(state.get("plan_steps", []))
            steps = state.get("plan_steps", [])
            just_done = idx - 1
            if 0 <= just_done < len(steps):
                step = steps[just_done]
                return (
                    f"Ran step {idx}/{total} via "
                    f"agent '{step.get('agent', '?')}'"
                )
            return f"Ran step {idx}/{total}"
        if node_name == "synthesizer":
            return "Synthesizer emitted final response"
        if node_name == "evaluator":
            return f"Verdict: {state.get('evaluation', '?')}"
        return ""

    final_state = _stream_graph(initial_state, progress_callback, _detailed_progress)

    set_eval_llm(None, None)

    raw_response = final_state.get("final_synth_response", "")
    fixed_labels, proposals, sub_start, sub_end, reasoning = (
        _parse_detailed_response(raw_response, start_index, end_index)
    )

    accepted = final_state.get("evaluation") == "pass"

    return AnnotationResult(
        sub_start=sub_start,
        sub_end=sub_end,
        final_labels=fixed_labels,
        final_reasoning=reasoning,
        accepted=accepted,
        iterations=1,
        messages=final_state.get("messages", []),
        graph_images=final_state.get("all_graph_images", []),
        label_annotations=proposals,
    )


# ---------------------------------------------------------------------------
# Lap-to-segment excerpter pipeline (manual.py)
# ---------------------------------------------------------------------------


def _eligible_lap_labels_lines(circuit_id: str) -> List[str]:
    """Eligible label IDs for the lap flow (circuit + sections + ST + main)."""
    catalog = get_label_catalog()
    lines: List[str] = []
    circuit_entry = catalog.get_label(circuit_id)
    if circuit_entry is not None:
        lines.append(f"  - `{circuit_entry.id}` ({circuit_entry.name})")
    for entry in catalog.entries_by_type("circuit_section"):
        if entry.parent != circuit_id:
            continue
        rng = entry.normalized_position_range
        rng_str = (
            f"[{rng[0]:.3f}, {rng[1]:.3f}]" if rng is not None else "[null, null]"
        )
        lines.append(f"  - `{entry.id}` ({entry.name}) — range {rng_str}")
    for entry in catalog.get_segment_types():
        lines.append(f"  - `{entry.id}` ({entry.name})")
    for entry in catalog.get_main_labels():
        lines.append(f"  - `{entry.id}` ({entry.name})")
    return lines


def _lap_planner_prompt(
    *,
    lap_start: int,
    lap_end: int,
    section_id: str,
    section_start: int,
    section_end: int,
    circuit_id: str,
    existing_section_annotations: List[dict],
) -> str:
    """Build the full planner prompt for the lap excerpter flow."""
    from app.services.llm.annotation_agent_tools import (
        AGENT_GRAPH_DEFINITIONS,
        PIPELINE_TOOL_DEFINITIONS,
    )

    catalog = get_label_catalog()
    section_entry = catalog.get_label(section_id) if section_id else None
    section_name = section_entry.name if section_entry else section_id
    section_desc = (
        (section_entry.description or "").strip() if section_entry else ""
    )
    section_rng = (
        section_entry.normalized_position_range if section_entry else None
    )
    section_rng_str = (
        f"[{section_rng[0]:.3f}, {section_rng[1]:.3f}]"
        if section_rng is not None else "[null, null]"
    )

    graph_catalogue = ", ".join(
        f"`{gdef['id']}` ({gdef['title']})"
        for gdef in AGENT_GRAPH_DEFINITIONS
    )
    tool_catalogue_lines = [
        f"- `{t['id']}` — {t['label']}: {t['description']}"
        for t in PIPELINE_TOOL_DEFINITIONS
    ]

    lap_skill_block = get_lap_skill().build_prompt(circuit_id)

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
        "to plan the describe_graphs steps that gather that evidence.",
        "",
        "#### Lap context",
        f"- Circuit: {circuit_id}",
        f"- Lap range: [{lap_start}, {lap_end}] (length {lap_end - lap_start})",
        "",
        "#### Section under review",
        f"- section_id: `{section_id}` ({section_name})",
        f"- description: {section_desc}",
        f"- normalized_position_range: {section_rng_str}",
        f"- rough iloc boundary: [{section_start}, {section_end}] "
        f"(length {section_end - section_start})",
        existing_block,
        "",
        lap_skill_block,
        "",
        "#### Available Step-Solver Agents",
        "Each plan step is dispatched to ONE sub-agent. The planner may "
        "only request `describe_graphs` here — a `label_verifier` step "
        "is appended automatically after your plan.",
        "- `describe_graphs` — renders the listed graphs over the rough "
        "boundary and writes one observation paragraph per graph. Pure "
        "observation — does not diagnose or assign labels.",
        "",
        "#### Available Graph IDs",
        graph_catalogue,
        "",
        "#### Available Pre-Compute Tools",
        "Invoke per step via the `tools` field. Each invoked tool "
        "produces an attachment attached to that step's prompt.",
        *tool_catalogue_lines,
        "",
        "#### Task",
        "Plan describe_graphs steps that gather the evidence needed to:",
        "  1. score each main label (EA / MS / RM / PS / OV / MD) against "
        "its `characteristics` block in the skill, and",
        "  2. optionally identify the trajectory shape if an ST1-ST6 pick "
        "would be unambiguous (skip when shape is ambiguous).",
        "Keep the plan tight — a typical section needs 1-3 steps. "
        "`trajectory_offset` + `time_difference_to_expert` are the two "
        "diagnostic graphs called out by the skill.",
        "",
        "Your plan must be a JSON object with a single key \"steps\". "
        "Each step object must have:",
        "  - \"step_id\": integer (1, 2, 3, ...).",
        "  - \"agent\": always \"describe_graphs\".",
        "  - \"description\": short string stating the goal of the step.",
        "  - \"requested_graphs\": list of graph IDs from the catalogue.",
        "  - \"tools\": list of pre-compute tool IDs (empty `[]` for none).",
        "",
        "Example:",
        "```json",
        "{",
        '  "steps": [',
        '    {"step_id": 1, "agent": "describe_graphs", "description": '
        '"Confirm the player drove through this section and check '
        'brake/throttle onsets at the boundaries.", "requested_graphs": '
        '["trajectory_offset", "brake", "throttle"], "tools": '
        '["compute_expert_phases"]},',
        '    {"step_id": 2, "agent": "describe_graphs", "description": '
        '"Measure trajectory shape to pick ST1-ST6.", "requested_graphs": '
        '["trajectory_detailed"], "tools": []}',
        "  ]",
        "}",
        "```",
    ]
    return "\n".join(parts)


def _lap_synth_prompts(
    *,
    lap_start: int,
    lap_end: int,
    section_id: str,
    section_start: int,
    section_end: int,
    circuit_id: str,
    verified_labels: List[str],
) -> Tuple[str, str]:
    """Build (synth_prompt_intro, synth_prompt_outro) for the lap flow."""
    lap_skill_block = get_lap_skill().build_prompt(circuit_id)
    eligible_lines = _eligible_lap_labels_lines(circuit_id)
    verified_inline = (
        ", ".join(verified_labels) if verified_labels
        else "(none — fall back to the eligible list)"
    )

    intro = "\n".join([
        "You are a racing telemetry analyst producing the final "
        "annotation for ONE circuit section. The describe_graphs steps "
        "captured the evidence below; pick the parent labels by matching "
        "the section's telemetry against each candidate label's "
        "`characteristics` block in the skill. Revising the boundary is "
        "an escape hatch only — invoke it when `locate_circuit_section` "
        "shows the rough range straddles two catalog sections.",
        "",
        "#### Section under review",
        f"- section_id: `{section_id}`",
        f"- rough iloc boundary: [{section_start}, {section_end}]",
        f"- lap range: [{lap_start}, {lap_end}]",
        f"- circuit: {circuit_id}",
        "",
        lap_skill_block,
    ])

    outro = "\n".join([
        "#### Eligible label IDs",
        "Only IDs from this list will be accepted. Always include the "
        "circuit and the section. An ST1-ST6 pick is OPTIONAL — include "
        "one only when the trajectory shape is unambiguous. Main labels "
        "(EA / MS / RM / PS / OV / MD) follow the skill's `characteristics` "
        "blocks; at most ONE of {EA, MS, RM} may be attached. "
        f"The verified shortlist from label_verifier is: {verified_inline}. "
        "Treat verified IDs as the primary candidates; fall back to the "
        "eligible list when the verified set misses a required parent.",
        *eligible_lines,
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
        "- Every label_id must come from the eligible list above.",
        "- The circuit label is required unless label_ids is empty.",
        "- An empty label_ids array is the valid 'drop this section' signal.",
    ])

    return intro, outro


def _parse_lap_response(
    raw: str,
    *,
    lap_start: int,
    lap_end: int,
    section_start: int,
    section_end: int,
) -> Tuple[int, int, bool, List[str], List[dict], str]:
    """Extract (new_start, new_end, revised, label_ids, rejected, reasoning).

    Raises ``RuntimeError`` on missing / unparseable JSON or out-of-bounds
    range — per the no-fallback policy.
    """
    parsed = _parse_json_response(raw)
    if not parsed:
        raise RuntimeError(
            f"lap synthesizer response was not valid JSON. "
            f"First 300 chars: {raw[:300]!r}"
        )

    revised_range = parsed.get("revised_range") or [section_start, section_end]
    try:
        new_start = int(revised_range[0])
        new_end = int(revised_range[1])
    except (TypeError, ValueError, IndexError) as e:
        raise RuntimeError(
            f"lap synthesizer: revised_range was not [int, int]: "
            f"{revised_range!r}"
        ) from e
    if not (lap_start <= new_start < new_end <= lap_end):
        raise RuntimeError(
            f"lap synthesizer: revised_range [{new_start}, {new_end}] "
            f"outside lap [{lap_start}, {lap_end}] or start >= end"
        )
    if (new_end - new_start) < 3:
        raise RuntimeError(
            f"lap synthesizer: revised_range too short "
            f"({new_end - new_start} ilocs) — minimum 3"
        )

    raw_label_ids = parsed.get("label_ids") or []
    cleaned: List[str] = []
    rejected: List[Dict[str, Any]] = []
    if isinstance(raw_label_ids, list):
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
    else:
        rejected.append({
            "value": raw_label_ids, "reason": "label_ids was not a list",
        })

    revised_flag = bool(parsed.get("revised")) or (
        new_start != section_start or new_end != section_end
    )
    reasoning = str(parsed.get("reasoning") or "")
    if parsed.get("revision_reason") and revised_flag:
        reasoning = (
            f"[revision: {parsed.get('revision_reason')}] {reasoning}".strip()
        )

    return new_start, new_end, revised_flag, cleaned, rejected, reasoning


def run_lap_annotation_pipeline(
    *,
    df,
    lap_start: int,
    lap_end: int,
    section_id: str,
    section_start: int,
    section_end: int,
    circuit_id: str,
    existing_section_annotations: Optional[List[dict]] = None,
    config: Optional[AnnotationPipelineConfig] = None,
    progress_callback: Optional[Callable] = None,
    vlm_stream_callback: Optional[Callable] = None,
    vlm_prompt_callback: Optional[Callable] = None,
    vlm_reasoning_callback: Optional[Callable] = None,
    step_event_callback: Optional[Callable] = None,
):
    """Execute the lap-section excerpter pipeline.

    Builds the lap-flow planner + synthesizer prompts, seeds the initial
    pool with a lap-flavoured ``init.parent_segment``, invokes
    ``annotation_root`` (same graph as the detailed flow), and parses
    the raw synthesizer response into a ``LapAnnotationResult``.
    """
    from app.services.llm.claude_lap_annotation_runner import LapAnnotationResult

    config = config or AnnotationPipelineConfig()
    existing_section_annotations = existing_section_annotations or []

    _wire_vlm_backend(
        config,
        vlm_stream_callback=vlm_stream_callback,
        vlm_prompt_callback=vlm_prompt_callback,
        vlm_reasoning_callback=vlm_reasoning_callback,
        step_event_callback=step_event_callback,
    )

    planner_prompt = _lap_planner_prompt(
        lap_start=lap_start,
        lap_end=lap_end,
        section_id=section_id,
        section_start=section_start,
        section_end=section_end,
        circuit_id=circuit_id,
        existing_section_annotations=existing_section_annotations,
    )

    catalog = get_label_catalog()
    section_entry = catalog.get_label(section_id) if section_id else None
    section_name = section_entry.name if section_entry else section_id

    # Seed init.parent_segment with the lap section's context. Same schema
    # so render_inputs_for_prompt's parent_segment formatter handles it.
    # parent_main_labels = [circuit_id] so label_verifier pulls
    # circuit_section sub-labels + ST1-ST6 as candidates.
    parent_segment = PipelineAttachment(
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

    initial_state = _base_root_state(
        df,
        parent_start=section_start,
        parent_end=section_end,
        parent_main_labels=[circuit_id] if circuit_id else [],
        existing_children=existing_section_annotations,
    )
    initial_state["planner_prompt"] = planner_prompt
    initial_state["initial_attachments"] = [parent_segment]
    initial_state["synth_prompt_intro"] = lambda s: _lap_synth_prompts(
        lap_start=lap_start,
        lap_end=lap_end,
        section_id=section_id,
        section_start=section_start,
        section_end=section_end,
        circuit_id=circuit_id,
        verified_labels=s.get("verified_labels", []),
    )[0]
    initial_state["synth_prompt_outro"] = lambda s: _lap_synth_prompts(
        lap_start=lap_start,
        lap_end=lap_end,
        section_id=section_id,
        section_start=section_start,
        section_end=section_end,
        circuit_id=circuit_id,
        verified_labels=s.get("verified_labels", []),
    )[1]

    def _lap_progress(node_name: str, state: Dict[str, Any]) -> str:
        if node_name == "planner":
            return f"Planned {len(state.get('plan_steps', []))} step(s)"
        if node_name == "executor":
            idx = state.get("current_step_index", 0)
            total = len(state.get("plan_steps", []))
            steps = state.get("plan_steps", [])
            just_done = idx - 1
            if 0 <= just_done < len(steps):
                step = steps[just_done]
                return (
                    f"Ran step {idx}/{total} via "
                    f"agent '{step.get('agent', '?')}'"
                )
            return f"Ran step {idx}/{total}"
        if node_name == "synthesizer":
            return "Synthesizer emitted final response"
        if node_name == "evaluator":
            return f"Verdict: {state.get('evaluation', '?')}"
        return ""

    final_state = _stream_graph(initial_state, progress_callback, _lap_progress)

    set_eval_llm(None, None)

    raw_response = final_state.get("final_synth_response", "")
    new_start, new_end, revised, label_ids, rejected, reasoning = (
        _parse_lap_response(
            raw_response,
            lap_start=lap_start,
            lap_end=lap_end,
            section_start=section_start,
            section_end=section_end,
        )
    )

    return LapAnnotationResult(
        section_id=section_id,
        start_index=new_start,
        end_index=new_end,
        label_ids=label_ids,
        reasoning=reasoning or raw_response or "(no reasoning)",
        revised=revised,
        submitted=True,
        rough_start=int(section_start),
        rough_end=int(section_end),
        rejected_proposals=rejected,
        rendered_images=list(final_state.get("all_graph_images", [])),
        transcript=raw_response,
        tool_calls=0,
    )
