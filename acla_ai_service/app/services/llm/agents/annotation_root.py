"""
annotation_root Agent — the outer annotation pipeline as a uniform Agent.

Topology (inherited from the framework):

    planner ──► step_solvers (loop) ──► proposal_synthesizer ──► evaluator ──► END
                  ▲                                              (no-op:
                  │                                               synthesizer
                  │                                               runs eval
              describe_graphs                                     suite inline)
              describe_graphs
              ...
              label_verifier        ◄── always appended as the last step

The planner emits one ``describe_graphs`` step per analysis goal plus a
trailing ``label_verifier`` step. The synthesizer is the existing
proposal_synthesizer logic — it judges every verified label against the
collected describe_graphs observations and pinpoints their start/end indices.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

from app.models.graph_analysis_skill import get_graph_skill
from app.models.label_catalog import LabelCatalog, get_label_catalog
from app.models.segment_models import LABEL_MAPPING
from app.services.llm.agent_framework import (
    Agent,
    AgentState,
    delegate_step,
)
from app.services.llm.step_evaluator_agents import (
    AttachmentPool,
    EvalPipelineResult,
    PipelineAttachment,
    _eval_llm_holder,
    pool_get_many,
    render_inputs_for_prompt,
    run_evaluator_suite,
    set_active_attachments,
    set_active_stage,
)

LOGGER = logging.getLogger(__name__)

ANNOTATION_ROOT_AGENT_NAME = "annotation_root"
DEFAULT_STEP_AGENT = "describe_graphs"


# ---------------------------------------------------------------------------
# Plan parsing helpers
# ---------------------------------------------------------------------------


def _parse_planner_steps(plan_text: str) -> List[Dict[str, Any]]:
    """Parse planner VLM output into structured step dicts.

    Each returned step dict has keys:
        step_id, agent, description, requested_graphs, tools
    """
    from app.services.llm.annotation_agent_tools import AGENT_GRAPH_DEFINITIONS

    all_graph_ids = [g["id"] for g in AGENT_GRAPH_DEFINITIONS]

    steps_raw: Optional[list] = None
    try:
        json_match = re.search(r"```json\s*(\{.*?\})\s*```", plan_text, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group(1))
        else:
            brace_match = re.search(r"(\{.*\})", plan_text, re.DOTALL)
            parsed = json.loads(brace_match.group(1)) if brace_match else None
        if parsed and isinstance(parsed, dict) and "steps" in parsed:
            steps_raw = parsed["steps"]
    except (json.JSONDecodeError, ValueError):
        steps_raw = None

    if not steps_raw or not isinstance(steps_raw, list):
        LOGGER.warning("Could not parse planner steps; using fallback.")
        return [{
            "step_id": 1,
            "agent": DEFAULT_STEP_AGENT,
            "description": "Analyse all telemetry graphs and propose the most fitting labels.",
            "requested_graphs": list(all_graph_ids),
            "tools": [],
        }]

    structured: List[Dict[str, Any]] = []
    for i, raw_step in enumerate(steps_raw, start=1):
        step_id = raw_step.get("step_id", i)
        desc = raw_step.get("description", f"Step {step_id}")
        # Accept legacy "solver" key from older planners.
        agent_id = raw_step.get("agent") or raw_step.get("solver") or DEFAULT_STEP_AGENT
        if agent_id != "describe_graphs":
            LOGGER.warning(
                "Step %s requested agent '%s'; only 'describe_graphs' is a "
                "valid planner-chosen agent. Falling back to describe_graphs.",
                step_id, agent_id,
            )
            agent_id = DEFAULT_STEP_AGENT

        req_graphs = raw_step.get("requested_graphs", [])
        if not req_graphs:
            desc_lower = desc.lower()
            req_graphs = [g for g in all_graph_ids
                          if g in desc_lower or g.replace("_", " ") in desc_lower]
        req_graphs = [g for g in req_graphs if g in all_graph_ids]

        tools = raw_step.get("tools", [])
        if not isinstance(tools, list):
            tools = []

        structured.append({
            "step_id": step_id,
            "agent": agent_id,
            "description": desc,
            "requested_graphs": req_graphs,
            "tools": tools,
        })

    return structured


def _validate_and_fix_hierarchy(
    labels: List[str],
    catalog: Optional[LabelCatalog] = None,
) -> tuple[List[str], List[str]]:
    """Validate label hierarchy and auto-fix common issues."""
    cat = catalog or get_label_catalog()
    fixed = list(dict.fromkeys(labels))
    warnings: List[str] = []

    to_add: List[str] = []
    for lid in fixed:
        parent = cat.parent_of.get(lid)
        if parent and parent not in fixed and parent not in to_add:
            to_add.append(parent)
            warnings.append(
                f"Auto-inserted parent '{parent}' ({LABEL_MAPPING.get(parent, parent)}) "
                f"for sub-label '{lid}' ({LABEL_MAPPING.get(lid, lid)})."
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


def _parse_json_response(raw: str) -> Optional[dict]:
    """Best-effort extraction of a JSON block from an LLM response."""

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


# ---------------------------------------------------------------------------
# Planner — seeds init.parent_segment, calls VLM, appends label_verifier step
# ---------------------------------------------------------------------------


def _planner(state: AgentState) -> Dict[str, Any]:
    """Analyse the segment and produce an analysis plan for sub-segment discovery.

    Seeds the attachment pool with ``init.parent_segment``, then calls the
    VLM to plan describe_graphs steps. Appends a trailing ``label_verifier``
    step so embedding-similarity filtering always runs after observation.
    """
    from app.services.llm.annotation_agent_tools import (
        AGENT_GRAPH_DEFINITIONS,
        PIPELINE_TOOL_DEFINITIONS,
    )

    parent_main_labels = state.get("parent_main_labels", [])
    existing_children = state.get("existing_children", [])
    parent_start = state.get("parent_start", 0)
    parent_end = state.get("parent_end", 0)

    # Seed init.parent_segment so all downstream agents can consume it.
    parent_segment = PipelineAttachment(
        name="init.parent_segment",
        kind="structured",
        content_schema="parent_segment",
        label="Parent Segment",
        content={
            "parent_start": parent_start,
            "parent_end": parent_end,
            "main_labels": [LABEL_MAPPING.get(l, l) for l in parent_main_labels],
            "existing_children": [
                {
                    "start_index": child.get("start_index"),
                    "end_index": child.get("end_index"),
                    "labels": [LABEL_MAPPING.get(l, l) for l in child.get("labels", [])],
                }
                for child in existing_children
            ],
        },
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

    prompt_parts = [
        "You are a racing telemetry analyst planning a sub-segment discovery strategy.",
        "",
        "#### Parent Segment",
        f"Main labels: {', '.join(main_readable)} (IDs: {json.dumps(parent_main_labels)})",
        f"Range: [{parent_start}, {parent_end}] (length: {parent_end - parent_start} data points)",
        "",
    ]

    if label_descriptions:
        prompt_parts.append("#### Label Descriptions (from Label Catalog)")
        prompt_parts.extend(label_descriptions)
        prompt_parts.append("")

    if annotation_guidelines:
        prompt_parts.append("#### Annotation Guidelines")
        prompt_parts.append("Follow these steps when planning analysis for the identified labels:")
        prompt_parts.extend(annotation_guidelines)
        prompt_parts.append("")

    if existing_children:
        prompt_parts.append("#### Already Discovered Sub-Segments")
        for child in existing_children:
            child_labels = [LABEL_MAPPING.get(l, l) for l in child.get("labels", [])]
            prompt_parts.append(
                f"- Range [{child['start_index']}, {child['end_index']}] "
                f"(length: {child['end_index'] - child['start_index']}): "
                f"{', '.join(child_labels)}"
            )
        prompt_parts.append("Find a DIFFERENT region that is not yet covered.")
        prompt_parts.append("")

    prompt_parts.extend([
        "#### Available Step-Solver Agents",
        "Each plan step is dispatched to ONE sub-agent. The planner may only "
        "request `describe_graphs` here — a `label_verifier` step is appended "
        "automatically by the framework after your plan.",
        "- `describe_graphs` — renders the telemetry graphs listed in "
        "`requested_graphs` and writes a precise observation paragraph per "
        "graph. Pure observation — does not diagnose or assign labels.",
        "",
        "#### Available Graph IDs (for `describe_graphs` agent)",
        graph_catalogue,
        "",
        "#### Available Pre-Compute Tools",
        "Invoke per step via the `tools` field. Each invoked tool produces "
        "an attachment that will be attached to that step's prompt only.",
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
        '{',
        '  "steps": [',
        '    {"step_id": 1, "agent": "describe_graphs", "description": "Measure entry/apex/exit shape via the trajectory offset trace.", "requested_graphs": ["trajectory_offset"], "tools": ["compute_expert_phases"]},',
        '    {"step_id": 2, "agent": "describe_graphs", "description": "Inspect speed and throttle around the apex.", "requested_graphs": ["speed", "throttle"], "tools": []}',
        '  ]',
        '}',
        "```",
    ])

    prompt = "\n".join(prompt_parts)

    set_active_stage("planner", "main")
    set_active_attachments([])
    vlm_fn = _eval_llm_holder.get("vlm")
    if vlm_fn:
        raw_plan = vlm_fn(prompt)
    else:
        raw_plan = (
            "[VLM not available — using passthrough plan] "
            "Examine all telemetry features and propose the most fitting labels."
        )

    suite_result: EvalPipelineResult = run_evaluator_suite(
        parent_prompt=prompt,
        parent_output_text=raw_plan,
        parent_inputs=[],
        step_name="planner",
        parent_start=parent_start,
        parent_end=parent_end,
    )
    evaluated_plan = suite_result.final_result

    parsed_steps = _parse_planner_steps(evaluated_plan)

    # Append label_verifier as the trailing step solver.
    next_step_id = max((s.get("step_id", 0) for s in parsed_steps), default=0) + 1
    parsed_steps.append({
        "step_id": next_step_id,
        "agent": "label_verifier",
        "description": "Filter candidate labels by embedding similarity to step observations.",
    })

    plan_attachment = PipelineAttachment(
        name="planner.plan",
        kind="text",
        label="Planner Plan",
        content=evaluated_plan,
    )

    msg = {"role": "planner", "content": evaluated_plan}
    messages = list(state.get("messages", []))
    messages.append(msg)

    return {
        "plan": evaluated_plan,
        "plan_steps": parsed_steps,
        "current_step_index": 0,
        "step_results": [],
        "all_graph_images": [],
        "all_graph_descriptions": [],
        "attachment_pool": {
            parent_segment.name: parent_segment,
            plan_attachment.name: plan_attachment,
        },
        "messages": messages,
    }


# ---------------------------------------------------------------------------
# Synthesizer — proposal_synthesizer; produces final label proposals
# ---------------------------------------------------------------------------


def _call_vlm(prompt: str, graph_image_bytes: List[bytes]) -> str:
    vlm_fn = _eval_llm_holder.get("vlm")
    if not vlm_fn:
        return ""
    if graph_image_bytes:
        return vlm_fn(prompt, graph_image_bytes)
    return vlm_fn(prompt)


def _synthesizer(state: AgentState) -> Dict[str, Any]:
    """Assemble the final label annotations from verified labels and step evidence.

    Reads every ``step_solver.*.observations`` attachment (the describe_graphs
    outputs) and the label_verifier's verified_labels attachment, prompts the
    VLM to decide which verified labels the evidence supports, and emits the
    final proposal. Runs the evaluator suite inline so the evaluator node
    downstream is a verdict-only pass-through.
    """
    messages = list(state.get("messages", []))
    parent_start = state.get("parent_start", 0)
    parent_end = state.get("parent_end", 0)
    verified_labels = state.get("verified_labels", [])

    pool: AttachmentPool = state.get("attachment_pool", {})

    # Collect inputs: parent_segment, verified labels (under whatever step_id
    # label_verifier ran at), and all describe_graphs observations.
    parent_inputs: List[PipelineAttachment] = []
    if "init.parent_segment" in pool:
        parent_inputs.append(pool["init.parent_segment"])
    for name in sorted(pool.keys()):
        if name.endswith(".verified_labels"):
            parent_inputs.append(pool[name])
    for name in sorted(pool.keys()):
        if name.endswith(".observations"):
            parent_inputs.append(pool[name])

    context_block = render_inputs_for_prompt(parent_inputs)

    n_verified = len(verified_labels)
    verified_ids_inline = ", ".join(verified_labels) if verified_labels else "(none)"

    intro_parts = [
        "You are a racing telemetry analyst producing label annotations.",
        "",
        "#### Task",
        f"For the parent range [{parent_start}, {parent_end}] "
        f"(length: {parent_end - parent_start} data points), judge every "
        "verified candidate label against the step observations. For each "
        "candidate that is proved, pinpoint its exact start_index and end_index.",
    ]

    instructions_parts = [
        "#### Instructions",
        f"- There are {n_verified} verified candidate label(s): {verified_ids_inline}.",
        f"- Emit EXACTLY one entry per verified candidate ({n_verified} entries "
        "total) — neither more nor fewer. Do not invent labels outside the "
        "verified list.",
        "- Judging rule: treat each label's full description as the definition. "
        "The bolded predicate is the headline claim; the surrounding prose lists "
        "its qualifiers. Set \"proved\": true only if the step observations "
        "satisfy the predicate AND every qualifier the description names. "
        "Otherwise set \"proved\": false.",
        f"- start_index and end_index are required only when \"proved\" is true; "
        f"each must satisfy {parent_start} <= start_index < end_index <= {parent_end} "
        "and the range must contain the cited evidence. Omit both fields when "
        "\"proved\" is false.",
        "- In \"reasoning\", cite the step observation sentences that establish "
        "(or fail to establish) the predicate + qualifiers. When \"proved\" is "
        "false, name the specific qualifier that is unmet or contradicted.",
        "- The top-level JSON has a single key \"labels\" whose value is a flat "
        "list of entries. Do not wrap the list in any other container.",
        "",
        "#### Output Format",
        "Respond with JSON of this exact shape only. Emit one entry per verified "
        f"candidate ({n_verified} entries total). The schema below shows BOTH "
        "entry shapes — proved-true (with indices) and proved-false (without). "
        "Output strict JSON only — no comments, no trailing commas, no extra keys.",
        "```json",
        "{",
        '  "labels": [',
        '    {',
        '      "label_id": "<one of the verified label IDs>",',
        '      "proved": true,',
        f'      "start_index": <integer in [{parent_start}, {parent_end}]>,',
        f'      "end_index": <integer in [{parent_start}, {parent_end}]>,',
        '      "reasoning": "..."',
        '    },',
        '    {',
        '      "label_id": "<one of the verified label IDs>",',
        '      "proved": false,',
        '      "reasoning": "..."',
        '    }',
        '  ]',
        "}",
        "```",
    ]

    vlm_prompt = "\n".join(intro_parts + ["", context_block, ""] + instructions_parts)
    eval_prompt = "\n".join(intro_parts + [""] + instructions_parts)

    set_active_stage("proposal_synthesizer", "main")
    set_active_attachments(parent_inputs)
    raw_response = _call_vlm(vlm_prompt, [])
    if not raw_response:
        raise RuntimeError(
            f"proposal_synthesizer: VLM returned empty response "
            f"(parent=[{parent_start}, {parent_end}], "
            f"verified_labels={verified_labels})"
        )

    suite_result: EvalPipelineResult = run_evaluator_suite(
        parent_prompt=eval_prompt,
        parent_output_text=raw_response,
        parent_inputs=parent_inputs,
        step_name="proposal_synthesizer",
        parent_start=parent_start,
        parent_end=parent_end,
    )
    evaluated_response = suite_result.final_result

    sub_labels: List[str] = []
    label_proposals: List[dict] = []
    reasoning = evaluated_response
    proposed_start = parent_start
    proposed_end = parent_end
    parsed = _parse_json_response(evaluated_response)
    if parsed:
        label_annotations = parsed.get("labels", [])
        starts: List[int] = []
        ends: List[int] = []
        for ann in (label_annotations if isinstance(label_annotations, list) else []):
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
            ann_start = int(raw_start) if isinstance(raw_start, (int, float)) else parent_start
            ann_end = int(raw_end) if isinstance(raw_end, (int, float)) else parent_end
            sub_labels.append(lid)
            label_proposals.append({
                "label_id": lid,
                "start_index": ann_start,
                "end_index": ann_end,
                "reasoning": ann_reasoning,
            })
            starts.append(ann_start)
            ends.append(ann_end)
        reasoning = "; ".join(p["reasoning"] for p in label_proposals) if label_proposals else evaluated_response
        if starts:
            proposed_start = min(starts)
        if ends:
            proposed_end = max(ends)
    else:
        LOGGER.warning("Proposal synthesizer evaluated response was not valid JSON.")

    fixed_labels, hierarchy_warnings = _validate_and_fix_hierarchy(sub_labels)
    for w in hierarchy_warnings:
        LOGGER.info("Hierarchy fix: %s", w)

    messages.append({"role": "assistant", "content": evaluated_response})

    proposal_attachment = PipelineAttachment(
        name="proposal_synthesizer.proposal",
        kind="text",
        label="Proposal Synthesizer Output",
        content=evaluated_response,
    )

    return {
        "evaluation": suite_result.final_verdict,
        "final_sub_start": proposed_start,
        "final_sub_end": proposed_end,
        "final_labels": fixed_labels,
        "final_label_annotations": label_proposals,
        "final_reasoning": reasoning,
        "attachment_pool": {proposal_attachment.name: proposal_attachment},
        "messages": messages,
    }


# ---------------------------------------------------------------------------
# Evaluator — no-op pass-through (synthesizer ran evaluator suite inline)
# ---------------------------------------------------------------------------


def _evaluator(state: AgentState) -> Dict[str, Any]:
    return {"evaluation": state.get("evaluation", "pass")}


class AnnotationRoot(Agent):
    """Root agent: orchestrates describe_graphs + label_verifier into
    final label proposals via the proposal_synthesizer.
    """

    name = ANNOTATION_ROOT_AGENT_NAME
    consumes: list = []       # root: starts from raw state inputs
    produces = ["proposal"]
    delegates_to = ["describe_graphs", "label_verifier"]

    def planner(self, state: AgentState) -> Dict[str, Any]:
        return _planner(state)

    def executor(self, state: AgentState, step, registry) -> Dict[str, Any]:
        return delegate_step(state, step, registry)

    def synthesizer(self, state: AgentState) -> Dict[str, Any]:
        return _synthesizer(state)

    def evaluator(self, state: AgentState) -> Dict[str, Any]:
        return _evaluator(state)


ANNOTATION_ROOT_SPEC = AnnotationRoot.register()
