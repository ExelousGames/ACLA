"""
Multi-agent annotation pipeline using LangGraph.

Implements a multi-step planner → executor → describer → synthesis pipeline
for automated telemetry sub-segment annotation.  Each step that produces
LLM output runs a step-appropriate evaluator suite (see step_evaluator_agents.py)
internally before writing to state.  This catches and corrects errors at the
point of origin, eliminating the need for a retry loop.

Tool categories used during execution:

* **generate_telemetry_graphs** – rendered telemetry graphs (PIL images)

The Vision Language Model (VLM) receives rendered graph images at each step,
replicating the visual evidence a human annotator would use.

Graph flow (forward-only — no retry edge):

    planner ──────────► steps_data_fetcher ──► step_describer ─┐
    (eval: format)      (no LLM output,       (eval: format,   │ (repeat per
                         no evaluator)          evidence)       │  plan step)
                        ┌───────────────────────────────────────┘
                        ▼
                label_verifier          ← embedding similarity filter
                        │                 (eval: format, range, consistency)
                proposal_synthesizer    ← VLM determines precise boundaries
                        │                 (eval: ALL FOUR)
                       END

    Every LLM-producing node calls run_evaluator_suite() internally
    before writing its output to state.  steps_data_fetcher is the only
    node that does NOT run evaluators (it fetches data, not LLM output).
"""

from __future__ import annotations

import io
import json
import logging
import operator
import threading
from dataclasses import dataclass, field
from typing import Any, Annotated, Callable, Dict, List, Literal, Optional, TypedDict

from langgraph.graph import StateGraph, END

from app.models.segment_models import (
    LABEL_MAPPING,
    LABEL_CATEGORIES,
)
from app.models.label_catalog import get_label_catalog, LabelCatalog
from app.models.graph_analysis_skill import get_graph_skill
from app.services.llm.step_evaluator_agents import (
    run_evaluator_suite,
    set_eval_llm,
    set_active_stage,
    set_active_iteration,
    get_active_stage,
    _eval_llm_holder,
    EvalPipelineResult,
    PipelineAttachment,
    AttachmentPool,
    merge_pool,
    pool_get_many,
    render_inputs_for_prompt,
)

LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


class AnnotationState(TypedDict, total=False):
    """Typed state flowing through the graph."""
    # --- input ---
    segment_data: dict
    df_ref: Any              # reference to the DataFrame (not serialised)
    parent_main_labels: list # Main Labels from parent segment (hints)
    existing_children: list  # [{id, start_index, end_index, labels}] already-found sub-segments
    parent_start: int        # parent segment start boundary
    parent_end: int          # parent segment end boundary
    available_labels: dict
    # --- planner state ---
    plan: str
    plan_steps: list
    current_step_index: int
    step_results: Annotated[list, operator.add]
    all_graph_images: list
    all_graph_descriptions: list
    # --- named attachment pool — see step_evaluator_agents.PipelineAttachment ---
    # Each pipeline agent declares which attachments it consumes / produces by
    # name.  The pool is keyed by stable namespaced names like
    # ``"init.parent_segment"``, ``"steps_data_fetcher.3.graph_images"``,
    # ``"step_describer.3.observations"``.  Evaluators see ONLY the parent
    # agent's input set, never the whole pool.
    attachment_pool: Annotated[Dict[str, PipelineAttachment], merge_pool]
    # --- label filtering state ---
    verified_labels: list        # label IDs that passed the embedding similarity filter
    verified_label_reasoning: dict  # {label_id: "Similarity <score> — <label text>"}
    # --- agent state ---
    evaluation: str              # final verdict from evaluator suite ("pass"/"fail")
    final_labels: list
    final_label_annotations: list  # [{label_id, start_index, end_index, reasoning}]
    final_reasoning: str
    final_sub_start: int
    final_sub_end: int
    messages: list


def _default_state() -> AnnotationState:
    return {
        "segment_data": {},
        "df_ref": None,
        "parent_main_labels": [],
        "existing_children": [],
        "parent_start": 0,
        "parent_end": 0,
        "available_labels": {},
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
        "final_labels": [],
        "final_label_annotations": [],
        "final_reasoning": "",
        "final_sub_start": 0,
        "final_sub_end": 0,
        "messages": [],
    }


# ---------------------------------------------------------------------------
# LLM dispatch — _eval_llm_holder (defined in step_evaluator_agents) is the
# single source of truth for the VLM/LLM callables and the active stage.
#   "vlm" – generate(prompt, images=None) → str   (vision + text)
#   "llm" – generate(prompt) → str                (text-only, for routing)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Embedding model singleton (lazy-loaded, shared across runs)
# ---------------------------------------------------------------------------

_EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
_SIMILARITY_THRESHOLD = 0.25
_MIN_FILTER_LABELS = 2   # always pass at least this many (highest scoring)
_MAX_FILTER_LABELS = 8   # cap to keep synthesizer prompt focused

_embedder_instance = None
_embedder_lock = threading.Lock()


def _get_embedder():
    """Return the SentenceTransformer singleton, loading on first call."""
    global _embedder_instance
    if _embedder_instance is not None:
        return _embedder_instance
    with _embedder_lock:
        if _embedder_instance is None:
            from sentence_transformers import SentenceTransformer
            LOGGER.info("Loading embedding model '%s' …", _EMBED_MODEL_NAME)
            _embedder_instance = SentenceTransformer(_EMBED_MODEL_NAME)
            LOGGER.info("Embedding model loaded.")
    return _embedder_instance


def _cosine_sim(a, b) -> float:
    """Cosine similarity between two 1-D numpy arrays."""
    import numpy as np
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------


def _validate_and_fix_hierarchy(
    labels: List[str],
    catalog: Optional[LabelCatalog] = None,
) -> tuple[List[str], List[str]]:
    """Validate label hierarchy and auto-fix common issues.

    Returns
    -------
    (fixed_labels, warnings)
        *fixed_labels* has missing parents inserted.
        *warnings* lists human-readable issues (exclusive-with conflicts etc).
    """
    cat = catalog or get_label_catalog()
    fixed = list(dict.fromkeys(labels))  # deduplicate, preserve order
    warnings: list[str] = []

    # Auto-insert missing parents
    to_add: list[str] = []
    for lid in fixed:
        parent = cat.parent_of.get(lid)
        if parent and parent not in fixed and parent not in to_add:
            to_add.append(parent)
            warnings.append(
                f"Auto-inserted parent '{parent}' ({LABEL_MAPPING.get(parent, parent)}) "
                f"for sub-label '{lid}' ({LABEL_MAPPING.get(lid, lid)})."
            )
    fixed = to_add + fixed  # parents first

    # Check exclusive-with conflicts
    rules = cat.get_hierarchy_rules(fixed)
    for conflict in rules["exclusive_conflicts"]:
        a, b = conflict["labels"]
        warnings.append(
            f"Conflict: '{a}' ({LABEL_MAPPING.get(a, a)}) and "
            f"'{b}' ({LABEL_MAPPING.get(b, b)}) are mutually exclusive."
        )

    return list(dict.fromkeys(fixed)), warnings


# ---------------------------------------------------------------------------
# Plan parsing helpers
# ---------------------------------------------------------------------------


def _parse_planner_steps(plan_text: str, available_tools: list) -> list:
    """Parse planner VLM output into structured step dicts.

    Attempts to extract a JSON ``{"steps": [...]}`` block from the plan.
    Falls back to a single catch-all step that requests all available
    statistics and graphs when parsing fails.

    Each returned step dict has keys:
        step_id, description, requested_statistics, requested_graphs
    """
    from .annotation_agent_tools import AGENT_GRAPH_DEFINITIONS

    all_graph_ids = [g["id"] for g in AGENT_GRAPH_DEFINITIONS]

    # Try to extract JSON from the plan text
    steps_raw: list | None = None
    try:
        # Look for ```json ... ``` fenced block first
        import re
        json_match = re.search(r"```json\s*(\{.*?\})\s*```", plan_text, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group(1))
        else:
            # Try parsing the whole text or the first { ... } block
            brace_match = re.search(r"(\{.*\})", plan_text, re.DOTALL)
            if brace_match:
                parsed = json.loads(brace_match.group(1))
            else:
                parsed = None
        if parsed and isinstance(parsed, dict) and "steps" in parsed:
            steps_raw = parsed["steps"]
    except (json.JSONDecodeError, ValueError):
        steps_raw = None

    if not steps_raw or not isinstance(steps_raw, list):
        # Fallback — single step that asks for everything
        LOGGER.warning("Could not parse planner steps; using fallback.")
        return [{
            "step_id": 1,
            "description": "Analyse all telemetry graphs and propose the most fitting labels.",
            "requested_statistics": [],
            "requested_graphs": list(all_graph_ids),
        }]

    structured: list[dict] = []
    for i, raw_step in enumerate(steps_raw, start=1):
        step_id = raw_step.get("step_id", i)
        desc = raw_step.get("description", f"Step {step_id}")

        req_graphs = raw_step.get("requested_graphs", [])

        # If the VLM didn't specify explicit graph requests, try to infer
        # from keywords in the description
        if not req_graphs:
            desc_lower = desc.lower()
            req_graphs = [g for g in all_graph_ids
                          if g in desc_lower or g.replace("_", " ") in desc_lower]

        # Validate against known graph IDs
        req_graphs = [g for g in req_graphs if g in all_graph_ids]

        structured.append({
            "step_id": step_id,
            "description": desc,
            "requested_statistics": [],
            "requested_graphs": req_graphs,
        })

    return structured


# ---------------------------------------------------------------------------
# Attachment naming convention (namespaced)
# ---------------------------------------------------------------------------
#
#   init.parent_segment
#   planner.plan
#   steps_data_fetcher.{k}.graph_images
#   steps_data_fetcher.{k}.graph_descriptions
#   step_describer.{k}.observations
#   label_verifier.verified_labels
#   proposal_synthesizer.proposal
#
# where {k} is the 1-based step index from the planner's plan_steps.


# ---------------------------------------------------------------------------
# Agent node functions
# ---------------------------------------------------------------------------


def init_producer_node(state: AnnotationState) -> Dict[str, Any]:
    """Seed the attachment pool with the parent_segment attachment.

    This is the explicit init producer — it converts raw run inputs
    (parent_start, parent_end, parent_main_labels, existing_children) into
    the first named attachment in the pool, which downstream agents may
    declare as a ``consumes`` dependency.
    """
    parent_main_labels = state.get("parent_main_labels", [])
    existing_children = state.get("existing_children", [])
    parent_start = state.get("parent_start", 0)
    parent_end = state.get("parent_end", 0)

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

    return {
        "attachment_pool": {parent_segment.name: parent_segment},
    }


def planner_node(state: AnnotationState) -> dict:
    """Analyse the segment and produce an analysis plan for sub-segment discovery.

    The planner inspects the telemetry summary, parent Main Labels, and
    existing children to decide *which statistics and graphs* are needed
    to find a new sub-segment within the parent range.

    Runs evaluator suite (format) on its own output before
    writing to state.
    """
    from .annotation_agent_tools import AGENT_GRAPH_DEFINITIONS

    parent_main_labels = state.get("parent_main_labels", [])
    existing_children = state.get("existing_children", [])
    parent_start = state.get("parent_start", 0)
    parent_end = state.get("parent_end", 0)

    graph_catalogue = ", ".join(
        f"`{gdef['id']}` ({gdef['title']})"
        for gdef in AGENT_GRAPH_DEFINITIONS
    )

    main_readable = [LABEL_MAPPING.get(l, l) for l in parent_main_labels]
    catalog = get_label_catalog()
    label_descriptions = []
    annotation_guidelines = []
    for label_id in parent_main_labels:
        label_def = catalog.get_label(label_id)
        if label_def and label_def.description:
            label_descriptions.append(f"  - {label_def.name} ({label_id}): {label_def.description}")
        if label_def and label_def.annotation_guideline:
            annotation_guidelines.append(f"  [{label_def.name}]\n  {label_def.annotation_guideline}")

    prompt_parts = [
        "You are a racing telemetry analyst planning a sub-segment discovery strategy.",
        "",
        "=== Parent Segment ===",
        f"Main labels: {', '.join(main_readable)} (IDs: {json.dumps(parent_main_labels)})",
        f"Range: [{parent_start}, {parent_end}] (length: {parent_end - parent_start} data points)",
        "",
    ]

    if label_descriptions:
        prompt_parts.append("=== Label Descriptions (from Label Catalog) ===")
        prompt_parts.extend(label_descriptions)
        prompt_parts.append("")

    if annotation_guidelines:
        prompt_parts.append("=== Annotation Guidelines ===")
        prompt_parts.append("Follow these steps when planning analysis for the identified labels:")
        prompt_parts.extend(annotation_guidelines)
        prompt_parts.append("")

    # Existing children for duplicate avoidance
    if existing_children:
        prompt_parts.append("=== Already Discovered Sub-Segments ===")
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
        "=== Available Tools ===",
        "1. **generate_telemetry_graphs** — visual telemetry graphs rendered as images.",
        f"   Available graph IDs: {graph_catalogue}",
        "   The solver has a Vision Language Model and CAN analyse the "
        "graph images directly.",
    ])

    prompt_parts.extend([
        "",
        "=== Task ===",
        "Plan which statistics and graphs to generate to help discover "
        "ONE notable sub-segment within the parent segment range. "
        "A sub-segment is a contiguous region where a specific event "
        "or behaviour occurs.",
        "",
        "Your plan must be a JSON object with a single key \"steps\".",
        "The \"steps\" key must contain a list of step objects.",
        "Each step object must have:",
        "  - \"step_id\": An integer (1, 2, 3, ...).",
        "  - \"description\": A string describing the goal of the step.",
        "  - \"requested_graphs\": A list of graph IDs to generate for this step (from the catalogue above). Use an empty list if none are needed.",
        "You can have multiple steps. Each step should request specific graphs to analyze.",
        "",
        "Example of a valid plan:",
        "```json",
        '{',
        '  "steps": [',
        '    {"step_id": 1, "description": "Analyze speed and acceleration to find the core anomaly.", "requested_graphs": ["speed", "speed_delta"]},',
        '    {"step_id": 2, "description": "Investigate braking behavior around the identified anomaly.", "requested_graphs": ["brake"]}',
        '  ]',
        '}',
        "```",
    ])

    prompt = "\n".join(prompt_parts)

    # planner consumes nothing from the pool — it works from raw run inputs
    # (parent_main_labels, existing_children, parent bounds) which are part
    # of the run's scaffolding, not produced by another agent.
    parent_inputs: List[PipelineAttachment] = []

    # 1. Generate output (LLM call)
    set_active_stage("planner", "main")
    vlm_fn = _eval_llm_holder.get("vlm")
    if vlm_fn:
        raw_plan = vlm_fn(prompt)
    else:
        raw_plan = "[VLM not available — using passthrough plan] " \
                   "Examine all telemetry features and propose the most fitting labels."

    # 2. Run evaluator suite — evaluators see only what the planner saw.
    suite_result: EvalPipelineResult = run_evaluator_suite(
        parent_prompt=prompt,
        parent_output_text=raw_plan,
        parent_inputs=parent_inputs,
        step_name="planner",
        parent_start=parent_start,
        parent_end=parent_end,
    )

    evaluated_plan = suite_result.final_result

    # Parse structured tool requests from evaluated planner output
    parsed_steps = _parse_planner_steps(evaluated_plan, [])

    # 3. Emit the named output attachment.
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
        "attachment_pool": {plan_attachment.name: plan_attachment},
        "messages": messages,
    }


# ---------------------------------------------------------------------------
# Tool executor node
# ---------------------------------------------------------------------------


def steps_data_fetcher_node(state: AnnotationState) -> Dict[str, Any]:
    """
    Executes the tools (statistics, graphs) requested for the current step
    in the plan and emits two named attachments per step:
        steps_data_fetcher.{k}.graph_images        (image_set)
        steps_data_fetcher.{k}.graph_descriptions  (structured)

    Where ``k`` is the 1-based step_id of the current plan step.  This node
    is not an LLM agent — it has no evaluators.
    """
    from .annotation_agent_tools import generate_telemetry_graphs
    messages = list(state.get("messages", []))
    plan_steps = state.get("plan_steps", [])
    current_step_index = state.get("current_step_index", 0)
    df = state.get("df_ref")
    start = state.get("parent_start", 0)
    end = state.get("parent_end", 0)

    if not plan_steps or current_step_index >= len(plan_steps):
        LOGGER.warning("Step executor called with no steps or invalid index. Skipping.")
        return {}

    step = plan_steps[current_step_index]
    step_id = step.get("step_id", current_step_index + 1)
    requested_graphs = step.get("requested_graphs", [])
    LOGGER.info(f"Executing step {current_step_index + 1}/{len(plan_steps)}: {step.get('description')}")
    LOGGER.info(f"  - Requested graphs: {requested_graphs}")

    image_bytes_list: List[bytes] = []
    desc_list: List[str] = []
    if requested_graphs:
        graph_results = generate_telemetry_graphs(
            df, start, end,
            graph_ids=requested_graphs
        )
        for img, desc in graph_results:
            desc_list.append(desc)
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            image_bytes_list.append(buf.getvalue())

    if image_bytes_list:
        content = f"Generated {len(image_bytes_list)} graphs: {', '.join(desc_list)}"
        messages.append({
            "role": "tool",
            "content": [{"tool_name": "generate_telemetry_graphs", "content": content}],
        })

    # Emit named attachments for this step into the pool.
    images_att = PipelineAttachment(
        name=f"steps_data_fetcher.{step_id}.graph_images",
        kind="image_set",
        label=f"Step {step_id} — Graph Images",
        content=image_bytes_list,
    )
    descs_att = PipelineAttachment(
        name=f"steps_data_fetcher.{step_id}.graph_descriptions",
        kind="structured",
        content_schema="graph_descriptions",
        label=f"Step {step_id} — Graph Descriptions",
        content=desc_list,
    )

    return {
        "all_graph_images": state.get("all_graph_images", []) + image_bytes_list,
        "all_graph_descriptions": state.get("all_graph_descriptions", []) + desc_list,
        "attachment_pool": {
            images_att.name: images_att,
            descs_att.name: descs_att,
        },
        "messages": messages,
    }


def _call_vlm(
    prompt: str,
    graph_image_bytes: List[bytes],
) -> str:
    """Dispatch a prompt to the VLM, optionally with graph images."""
    vlm_fn = _eval_llm_holder.get("vlm")
    if not vlm_fn:
        return ""

    if graph_image_bytes:
        return vlm_fn(prompt, graph_image_bytes)
    return vlm_fn(prompt)



def _parse_json_response(raw: str) -> Optional[dict]:
    """Best-effort extraction of a JSON block from an LLM response."""
    import re

    def _try_loads(s: str) -> Optional[dict]:
        """Try json.loads, falling back to fixing literal newlines inside strings."""
        s = s.strip()
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            pass
        # VLMs often emit literal newlines inside JSON string values;
        # replace them with \n only inside quoted strings.
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

    # Last resort: find the first { ... } block in the raw text
    brace_match = re.search(r'\{[\s\S]*\}', raw)
    if brace_match:
        result = _try_loads(brace_match.group())
        if result is not None:
            return result

    return None


# --- Step solver ------------------------------------------------------------


def step_describer_node(state: AnnotationState) -> Dict[str, Any]:
    """
    Produces a detailed prose description of the graphs and data gathered for
    the current plan step.  This node only describes what is visible — it does
    not interpret, diagnose, or assign labels.  Downstream nodes
    (label_verifier, proposal_synthesizer) consume these descriptions to make
    decisions.

    Consumes from the pool:
        - init.parent_segment
        - steps_data_fetcher.{k}.graph_images
        - steps_data_fetcher.{k}.graph_descriptions
    Produces:
        - step_describer.{k}.observations
    """
    messages = list(state.get("messages", []))
    plan_steps = state.get("plan_steps", [])
    current_step_index = state.get("current_step_index", 0)
    step = plan_steps[current_step_index]
    step_id = step.get("step_id", current_step_index + 1)
    parent_start = state.get("parent_start", 0)
    parent_end = state.get("parent_end", 0)

    # Declare and fetch named attachments from the pool.
    pool: AttachmentPool = state.get("attachment_pool", {})
    consumes = [
        "init.parent_segment",
        f"steps_data_fetcher.{step_id}.graph_images",
        f"steps_data_fetcher.{step_id}.graph_descriptions",
    ]
    parent_inputs = pool_get_many(pool, consumes)

    # Pull the concrete pieces this node needs from its own input set.
    images_atts = [a for a in parent_inputs if a.name == f"steps_data_fetcher.{step_id}.graph_images"]
    descs_atts = [a for a in parent_inputs if a.name == f"steps_data_fetcher.{step_id}.graph_descriptions"]
    graph_images: List[bytes] = images_atts[0].content if images_atts else []
    graph_descriptions: List[str] = descs_atts[0].content if descs_atts else []

    prompt = (
        f"You are a telemetry graph describer.  Your ONLY job is to produce a "
        f"detailed, precise description of the data and graphs provided.  "
        f"Write in flowing prose paragraphs — do NOT use numbered lists, "
        f"bullet points, or step-by-step formatting.  Do NOT diagnose problems, "
        f"assign labels, or suggest what the observations mean.  Downstream nodes "
        f"in the pipeline will interpret your description — your job is to give "
        f"them accurate raw observations.\n\n"
        f"**Segment Context:**\n"
        f"- Index Range: {parent_start} to {parent_end}\n\n"
        f"**Analysis Goal: {step['description']}**\n\n"
        f"**Tool Outputs:**\n"
    )

    if graph_descriptions:
        prompt += f"- Generated Graphs:\n" + "\n".join([f"  - {desc}" for desc in graph_descriptions]) + "\n"

    # Inject graph analysis skill instructions when graphs are present
    if graph_images:
        requested_graph_ids = step.get("requested_graphs", [])
        if requested_graph_ids:
            graph_skill = get_graph_skill()
            skill_prompt = graph_skill.build_graph_prompt(requested_graph_ids)
            if skill_prompt:
                prompt += f"\n{skill_prompt}\n"

    prompt += (
        "\n**Your Task:**\n"
        "Write one paragraph per graph, following the per-graph guidance above.\n\n"
    )

    # 1. Generate output (VLM call with graph images)
    # Stage first (resets iter/total on node change), then iteration.
    set_active_stage("step_describer", "main")
    set_active_iteration(current_step_index + 1, len(plan_steps))
    raw_response = _call_vlm(prompt, graph_images)

    # 2. Run evaluator suite — evaluators see only this step's parent_inputs,
    # which includes the same image_set attachment the describer consumed,
    # so evidence_evaluator can check claims against the actual graphs.
    suite_result: EvalPipelineResult = run_evaluator_suite(
        parent_prompt=prompt,
        parent_output_text=raw_response,
        parent_inputs=parent_inputs,
        step_name="step_describer",
        parent_start=parent_start,
        parent_end=parent_end,
    )

    evaluated_response = suite_result.final_result
    messages.append({"role": "assistant", "content": evaluated_response})

    next_step_index = current_step_index + 1

    # 3. Emit the named output attachment (step observations) into the pool.
    obs_attachment = PipelineAttachment(
        name=f"step_describer.{step_id}.observations",
        kind="structured",
        content_schema="step_observation",
        label=f"Step {step_id} — {step['description']}",
        content={
            "requested_graphs": step.get("requested_graphs", []),
            "graph_descriptions": graph_descriptions,
            "graph_observations": evaluated_response,
        },
    )

    return {
        "current_step_index": next_step_index,
        "step_results": [{
            "step_id": step_id,
            "description": step["description"],
            "graph_observations": evaluated_response,
            "graph_descriptions": graph_descriptions,
        }],
        "attachment_pool": {obs_attachment.name: obs_attachment},
        "messages": messages,
    }


def step_router(state: AnnotationState) -> Literal["steps_data_fetcher", "label_verifier"]:
    """Routes to the next step executor or to label verification if all steps are complete."""
    if state["current_step_index"] < len(state.get("plan_steps", [])):
        return "steps_data_fetcher"
    return "label_verifier"


# ---------------------------------------------------------------------------
# Label verifier — embedding similarity filter
# ---------------------------------------------------------------------------


def label_verifier_node(state: AnnotationState) -> Dict[str, Any]:
    """Filter candidate labels by embedding similarity to step describer evidence.

    Embeds the concatenated step-describer observations as a query, embeds
    each candidate label's name + description, then keeps only labels whose
    cosine similarity exceeds _SIMILARITY_THRESHOLD.  At least
    _MIN_FILTER_LABELS labels are always passed (highest scoring), and the
    result is capped at _MAX_FILTER_LABELS to keep the synthesizer prompt
    focused.

    Consumes from the pool:
        - step_describer.{k}.observations  (for every k in plan_steps)
    Produces:
        - label_verifier.verified_labels

    No evaluator suite — this is a deterministic embedding-similarity
    filter, not an LLM agent.
    """
    import numpy as np

    messages = list(state.get("messages", []))
    parent_main_labels = state.get("parent_main_labels", [])

    # Build candidate list from parent sub-labels + segment types
    catalog = get_label_catalog()
    candidate_ids: list[str] = []
    for pid in parent_main_labels:
        for entry in catalog.get_sublabels(pid):
            candidate_ids.append(entry.id)
    for entry in catalog.get_segment_types():
        candidate_ids.append(entry.id)

    # Deduplicate while preserving order
    seen: set[str] = set()
    shortlisted: list[str] = []
    for lid in candidate_ids:
        if lid not in seen:
            seen.add(lid)
            shortlisted.append(lid)

    def _emit_verified(payload: list[dict]) -> PipelineAttachment:
        return PipelineAttachment(
            name="label_verifier.verified_labels",
            kind="structured",
            content_schema="verified_labels",
            label="Verified Labels",
            content=payload,
        )

    if not shortlisted:
        LOGGER.info("Label similarity filter: no candidate labels.")
        messages.append({
            "role": "label_verifier",
            "content": "No candidate labels available for parent categories.",
        })
        att = _emit_verified([])
        return {
            "verified_labels": [],
            "verified_label_reasoning": {},
            "attachment_pool": {att.name: att},
            "messages": messages,
        }

    # Consume all step_describer observations from the pool, in step_id order.
    pool: AttachmentPool = state.get("attachment_pool", {})
    plan_steps = state.get("plan_steps", [])
    consumes = [
        f"step_describer.{step.get('step_id', i + 1)}.observations"
        for i, step in enumerate(plan_steps)
    ]
    obs_inputs = pool_get_many(pool, consumes)
    evidence_parts: list[str] = []
    for att in obs_inputs:
        c = att.content if isinstance(att.content, dict) else {}
        obs = c.get("graph_observations")
        if obs:
            evidence_parts.append(str(obs))
    query_text = " ".join(evidence_parts).strip()

    if not query_text:
        LOGGER.warning("Label similarity filter: no evidence text; passing top candidates.")
        fallback = shortlisted[:_MAX_FILTER_LABELS]
        messages.append({
            "role": "label_verifier",
            "content": f"No evidence text; passed top {len(fallback)} candidates unchanged.",
        })
        fallback_payload = []
        for lid in fallback:
            entry = catalog.get_label(lid)
            fallback_payload.append({
                "label_id": lid,
                "name": entry.name if entry else LABEL_MAPPING.get(lid, lid),
                "description": entry.description if entry else "",
                "similarity": None,
            })
        att = _emit_verified(fallback_payload)
        return {
            "verified_labels": fallback,
            "verified_label_reasoning": {lid: "No evidence available." for lid in fallback},
            "attachment_pool": {att.name: att},
            "messages": messages,
        }

    # Build label text corpus: "Name: description" per candidate
    label_texts: list[str] = []
    for lid in shortlisted:
        entry = catalog.get_label(lid)
        if entry and entry.description:
            label_texts.append(f"{entry.name}: {entry.description}")
        elif entry:
            label_texts.append(entry.name)
        else:
            label_texts.append(LABEL_MAPPING.get(lid, lid))

    # Batch-embed query and all candidates
    embedder = _get_embedder()
    query_emb: np.ndarray = embedder.encode(query_text, convert_to_numpy=True)
    label_embs: np.ndarray = embedder.encode(label_texts, convert_to_numpy=True)

    # Score each candidate
    scored: list[tuple[str, float, str]] = [
        (lid, _cosine_sim(query_emb, label_embs[i]), label_texts[i])
        for i, lid in enumerate(shortlisted)
    ]
    scored.sort(key=lambda x: x[1], reverse=True)

    # Apply threshold, guarantee min, cap max
    above = [(lid, sim, txt) for lid, sim, txt in scored if sim >= _SIMILARITY_THRESHOLD]
    if len(above) < _MIN_FILTER_LABELS:
        above = scored[:_MIN_FILTER_LABELS]
    filtered = above[:_MAX_FILTER_LABELS]

    verified: list[str] = [lid for lid, _, _ in filtered]
    reasoning_map: dict[str, str] = {
        lid: f"Similarity {sim:.3f} — {txt}"
        for lid, sim, txt in filtered
    }

    passed_log = "\n".join(
        f"✓ {lid} ({LABEL_MAPPING.get(lid, lid)}): {sim:.3f}"
        for lid, sim, _ in filtered
    )
    rejected_log = "\n".join(
        f"✗ {lid} ({LABEL_MAPPING.get(lid, lid)}): {sim:.3f}"
        for lid, sim, _ in scored
        if lid not in verified
    )

    LOGGER.info(
        "Label similarity filter: %d/%d passed (threshold=%.2f)",
        len(verified), len(shortlisted), _SIMILARITY_THRESHOLD,
    )
    messages.append({
        "role": "label_verifier",
        "content": (
            f"Embedding filter: {len(verified)}/{len(shortlisted)} labels passed "
            f"(threshold={_SIMILARITY_THRESHOLD}):\n{passed_log}"
            + (f"\n\nRejected:\n{rejected_log}" if rejected_log else "")
        ),
    })

    verified_payload: list[dict] = []
    for lid, sim, _txt in filtered:
        entry = catalog.get_label(lid)
        verified_payload.append({
            "label_id": lid,
            "name": entry.name if entry else LABEL_MAPPING.get(lid, lid),
            "description": entry.description if entry else "",
            "similarity": sim,
        })

    att = _emit_verified(verified_payload)
    return {
        "verified_labels": verified,
        "verified_label_reasoning": reasoning_map,
        "attachment_pool": {att.name: att},
        "messages": messages,
    }


def proposal_synthesizer_node(state: AnnotationState) -> Dict[str, Any]:
    """Assemble the final label annotations from verified labels and step evidence.

    Decides which verified labels the evidence actually supports and
    pinpoints each label's start/end indices within the parent range.

    Consumes from the pool:
        - init.parent_segment
        - label_verifier.verified_labels
        - step_describer.{k}.observations  (for every k in plan_steps)
    Produces:
        - proposal_synthesizer.proposal
    """
    messages = list(state.get("messages", []))
    parent_start = state.get("parent_start", 0)
    parent_end = state.get("parent_end", 0)
    verified_labels = state.get("verified_labels", [])

    # Declare and fetch named attachments from the pool.
    pool: AttachmentPool = state.get("attachment_pool", {})
    plan_steps = state.get("plan_steps", [])
    consumes = (
        ["init.parent_segment", "label_verifier.verified_labels"]
        + [
            f"step_describer.{step.get('step_id', i + 1)}.observations"
            for i, step in enumerate(plan_steps)
        ]
    )
    parent_inputs = pool_get_many(pool, consumes)

    context_block = render_inputs_for_prompt(parent_inputs)

    n_verified = len(verified_labels)
    verified_ids_inline = ", ".join(verified_labels) if verified_labels else "(none)"

    # Task + instructions only — the rendered attachments are inserted between
    # them ONLY for the VLM call.  The evaluator receives the same task +
    # instructions but reads the attachments from the pool itself, avoiding the
    # duplicate "=== Verified Labels ===" / step-describer sections that would
    # otherwise appear twice in the evaluator's prompt.
    intro_parts = [
        "You are a racing telemetry analyst producing label annotations.",
        "",
        "=== Task ===",
        f"Annotate the parent range [{parent_start}, {parent_end}] "
        f"(length: {parent_end - parent_start} data points) by selecting which "
        "of the verified labels below are clearly evidenced in the graphs, "
        "and pinpointing each label's exact start_index and end_index.",
    ]

    instructions_parts = [
        "=== Instructions ===",
        f"- There are {n_verified} verified candidate label(s): {verified_ids_inline}.",
        "- Emit one entry in the \"labels\" list for EVERY verified label whose "
        "behaviour is clearly evidenced — this can be up to "
        f"{n_verified} entries. Drop only the verified labels the evidence "
        "does not support. Do NOT cap the output at two entries, and do NOT "
        "invent labels outside the verified list.",
        f"- Each entry must satisfy {parent_start} <= start_index < end_index <= {parent_end}.",
        "- In \"reasoning\", explain why those exact boundaries were chosen, "
        "citing the step observations above.",
        "- The top-level JSON has a single key \"labels\" whose value is a flat "
        "list of entries. Do not wrap the list in any other container.",
        "",
        "=== Output Format ===",
        "Respond with JSON of this exact shape only. The schema below shows "
        "ONE entry as a template — repeat that entry object inside the "
        f'"labels" array once for every supported verified label (up to '
        f"{n_verified} entries). Output strict JSON only — no comments, no "
        "trailing commas, no extra keys.",
        "```json",
        "{",
        '  "labels": [',
        '    {',
        '      "label_id": "<one of the verified label IDs>",',
        f'      "start_index": <integer in [{parent_start}, {parent_end}]>,',
        f'      "end_index": <integer in [{parent_start}, {parent_end}]>,',
        '      "reasoning": "..."',
        '    }',
        '  ]',
        "}",
        "```",
    ]

    vlm_prompt = "\n".join(intro_parts + ["", context_block, ""] + instructions_parts)
    eval_prompt = "\n".join(intro_parts + [""] + instructions_parts)

    # 1. Generate output (VLM call)
    set_active_stage("proposal_synthesizer", "main")
    raw_response = _call_vlm(vlm_prompt, [])
    if not raw_response:
        raise RuntimeError(
            f"proposal_synthesizer: VLM returned empty response "
            f"(parent=[{parent_start}, {parent_end}], "
            f"verified_labels={verified_labels})"
        )

    # 2. Run evaluator suite — evaluators read attachments from the pool, so
    # we pass the inputs-free eval_prompt to avoid duplicating them.
    suite_result: EvalPipelineResult = run_evaluator_suite(
        parent_prompt=eval_prompt,
        parent_output_text=raw_response,
        parent_inputs=parent_inputs,
        step_name="proposal_synthesizer",
        parent_start=parent_start,
        parent_end=parent_end,
    )

    # 3. Parse the fully evaluated final result
    evaluated_response = suite_result.final_result

    sub_labels: list[str] = []
    label_proposals: list[dict] = []
    reasoning = evaluated_response
    proposed_start = parent_start
    proposed_end = parent_end
    parsed = _parse_json_response(evaluated_response)
    if parsed:
        label_annotations = parsed.get("labels", [])
        starts: list[int] = []
        ends: list[int] = []
        for ann in (label_annotations if isinstance(label_annotations, list) else []):
            lid = ann.get("label_id")
            if not lid or lid not in LABEL_MAPPING:
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

    # Validate hierarchy (auto-insert missing parents for sub-labels)
    fixed_labels, hierarchy_warnings = _validate_and_fix_hierarchy(sub_labels)
    for w in hierarchy_warnings:
        LOGGER.info("Hierarchy fix: %s", w)

    messages.append({"role": "assistant", "content": evaluated_response})

    # Emit the synthesizer's named output attachment into the pool.
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
# Graph construction
# ---------------------------------------------------------------------------


def build_annotation_graph(
    vlm_llm: Callable,
    text_llm: Callable,
):
    """Build the annotation agent graph.

    Linear forward-only flow — no retry edge.  Each LLM-producing node
    runs its own evaluator suite internally before writing to state.
    """
    # Register LLMs (single shared holder lives in step_evaluator_agents)
    set_eval_llm(vlm_llm, text_llm)

    graph = StateGraph(AnnotationState)

    graph.add_node("init_producer", init_producer_node)
    graph.add_node("planner", planner_node)
    graph.add_node("steps_data_fetcher", steps_data_fetcher_node)
    graph.add_node("step_describer", step_describer_node)
    graph.add_node("label_verifier", label_verifier_node)
    graph.add_node("proposal_synthesizer", proposal_synthesizer_node)
    # No separate evaluator node — evaluation happens inside each node.

    graph.set_entry_point("init_producer")
    graph.add_edge("init_producer", "planner")
    graph.add_edge("planner", "steps_data_fetcher")
    graph.add_edge("steps_data_fetcher", "step_describer")
    graph.add_conditional_edges(
        "step_describer",
        step_router,
        {
            "steps_data_fetcher": "steps_data_fetcher",
            "label_verifier": "label_verifier",
        },
    )
    graph.add_edge("label_verifier", "proposal_synthesizer")
    graph.add_edge("proposal_synthesizer", END)
    # No retry edge — evaluators fix in-place within each node

    return graph.compile()


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------


@dataclass
class AnnotationPipelineConfig:
    """Configuration for the annotation pipeline."""

    max_iterations: int = 3
    max_new_tokens: int = 512
    temperature: float = 0.7

    # llama-cpp VLM settings
    gguf_path: Optional[str] = None
    mmproj_path: Optional[str] = None
    context_size: int = 32768
    n_gpu_layers: int = -1

    # HuggingFace source repo + conversion
    hf_repo: str = "Qwen/Qwen2.5-VL-72B-Instruct"
    quantization_type: str = "Q4_K_M"


@dataclass
class AnnotationResult:
    """Result returned by the pipeline."""

    final_labels: List[str]
    final_reasoning: str
    accepted: bool
    iterations: int
    messages: List[dict]
    graph_images: List[bytes] = field(default_factory=list)  # PNG bytes
    sub_start: Optional[int] = None
    sub_end: Optional[int] = None


def _prepare_segment_data(
    session_id: str,
    start_index: int,
    end_index: int,
) -> dict:
    """Return a minimal segment metadata dict (no statistics computed)."""
    return {"session_id": session_id, "start_index": start_index, "end_index": end_index}


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
) -> AnnotationResult:
    """Execute the forward-only annotation pipeline.

    The pipeline runs: planner → steps_data_fetcher → step_describer (loop)
    → label_verifier → proposal_synthesizer → END.  Each LLM-producing
    node runs its own evaluator suite internally before writing to state,
    so there is no separate evaluator node or retry loop.

    Parameters
    ----------
    df : pandas.DataFrame
        Full session telemetry DataFrame.
    start_index, end_index : int
        Parent segment boundaries.
    session_id : str
        Current session identifier.
    parent_main_labels : list[str]
        Main-label IDs from the parent segment (used as hints).
    existing_children : list[dict], optional
        Already-existing child sub-segments ``[{start_index, end_index, labels}, ...]``.
    config : AnnotationPipelineConfig, optional
        VLM and pipeline parameters.
    progress_callback : callable, optional
        ``fn(step_name: str, detail: str)``
        called after each node completes.

    Returns
    -------
    AnnotationResult
    """
    config = config or AnnotationPipelineConfig()
    existing_children = existing_children or []

    # Prepare segment data (lightweight stats only — heavy tool work is
    # done by the tool_executor node).
    segment_data = _prepare_segment_data(session_id, start_index, end_index)

    # ------------------------------------------------------------------
    # Build VLM generate function (llama-cpp backed)
    # ------------------------------------------------------------------
    from app.services.llm.annotation_agent_llm_service import (
        get_or_start_service,
        AnnotationAgentLLMConfig,
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

    def vlm_generate(
        prompt: str, images: Optional[List[bytes]] = None,
    ) -> str:
        """Send prompt (with optional images) to the llama-cpp VLM."""
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
        """Text-only call to the same server (no images, lower temperature)."""
        return vlm_service.generate(
            prompt,
            images=None,
            max_tokens=64,
            temperature=0.1,
        )

    # Build graph
    graph = build_annotation_graph(vlm_generate, llm_generate)

    # Initial state
    initial_state = _default_state()
    initial_state.update({
        "segment_data": segment_data,
        "df_ref": df,
        "parent_main_labels": parent_main_labels,
        "existing_children": existing_children,
        "parent_start": start_index,
        "parent_end": end_index,
        "available_labels": LABEL_CATEGORIES,
    })

    # Stream through graph nodes for progress reporting
    final_state = dict(initial_state)
    for event in graph.stream(initial_state, config={"recursion_limit": 100}):
        for node_name, node_output in event.items():
            final_state.update(node_output)
            if progress_callback:
                detail = ""
                if node_name == "planner":
                    n_steps = len(final_state.get("plan_steps", []))
                    detail = f"Planned {n_steps} analysis step(s)"
                elif node_name == "steps_data_fetcher":
                    plan_steps = final_state.get("plan_steps", [])
                    idx = final_state.get("current_step_index", 0)
                    pool = final_state.get("attachment_pool", {})
                    step_id = plan_steps[idx].get("step_id", idx + 1) if 0 <= idx < len(plan_steps) else idx + 1
                    descs_att = pool.get(f"steps_data_fetcher.{step_id}.graph_descriptions")
                    n_graphs = len(descs_att.content) if descs_att and isinstance(descs_att.content, list) else 0
                    detail = f"Gathered {n_graphs} graph(s)"
                elif node_name == "step_describer":
                    idx = final_state.get("current_step_index", 0)
                    total = len(final_state.get("plan_steps", []))
                    detail = f"Described step {idx}/{total}"
                elif node_name == "label_verifier":
                    verified = final_state.get("verified_labels", [])
                    detail = (
                        f"Verified {len(verified)} labels: "
                        f"{', '.join(LABEL_MAPPING.get(l, l) for l in verified)}"
                    )
                elif node_name == "proposal_synthesizer":
                    labels = final_state.get("final_labels", [])
                    ss = final_state.get("final_sub_start", "?")
                    se = final_state.get("final_sub_end", "?")
                    verdict = final_state.get("evaluation", "")
                    detail = (
                        f"Range [{ss}, {se}], "
                        f"labels: {', '.join(LABEL_MAPPING.get(l, l) for l in labels)}, "
                        f"eval: {verdict}"
                    )
                progress_callback(node_name, detail)

    # Clean up VLM/LLM holder
    set_eval_llm(None, None)

    accepted = final_state.get("evaluation") == "pass"

    return AnnotationResult(
        sub_start=final_state.get("final_sub_start"),
        sub_end=final_state.get("final_sub_end"),
        final_labels=final_state.get("final_labels", []),
        final_reasoning=final_state.get("final_reasoning", ""),
        accepted=accepted,
        iterations=1,  # forward-only pipeline, always 1 pass
        messages=final_state.get("messages", []),
        graph_images=final_state.get("all_graph_images", []),
    )
