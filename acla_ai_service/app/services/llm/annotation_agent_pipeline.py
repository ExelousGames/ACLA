"""
Multi-agent annotation pipeline using LangGraph.

Implements a multi-step planner → executor → reasoner loop followed by
label filtering and a synthesis → evaluator cycle for automated
telemetry sub-segment annotation.

Tool categories used during execution:

* **generate_telemetry_graphs** – rendered telemetry graphs (PIL images)

The Vision Language Model (VLM) receives rendered graph images at each step,
replicating the visual evidence a human annotator would use.

Graph flow:

    planner ──► steps_data_fetcher ──► step_reasoner ─┐
       ▲                                          │ (repeat for each plan step)
       │                          ┌───────────────┘
       │                          ▼
       │                  label_verifier      ← embedding similarity filter
       │                          │             (narrows candidates before synthesis)
       │                  proposal_synthesizer ← VLM determines precise boundaries
       │                          │
       │                    evaluator ─────────────────────────┐
       │                          │ (pass)                     │ (fail, retry)
       │                         END                           │
       └──────────────────────────────────────────────────────-┘
"""

from __future__ import annotations

import io
import json
import logging
import operator
import threading
from dataclasses import dataclass, field
from typing import Any, Annotated, Callable, Dict, List, Literal, Optional, Sequence, TypedDict

from langgraph.graph import StateGraph, END
from PIL import Image

from app.models.segment_models import (
    LABEL_MAPPING,
    LABEL_CATEGORIES,
)
from app.models.label_catalog import get_label_catalog, LabelCatalog
from app.models.graph_analysis_skill import get_graph_skill

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
    # --- Per-step transient state ---
    current_step_statistics_text: str
    current_step_graph_images: list
    current_step_graph_descriptions: list
    # --- label filtering state ---
    verified_labels: list        # label IDs that passed the embedding similarity filter
    verified_label_reasoning: dict  # {label_id: "Similarity <score> — <label text>"}
    # --- agent state ---
    proposed_sub_start: int      # VLM-proposed sub-segment start index
    proposed_sub_end: int        # VLM-proposed sub-segment end index
    proposed_labels: list
    proposed_label_annotations: list  # [{label_id, start_index, end_index, reasoning}]
    proposed_reasoning: str
    parse_warnings: list         # JSON parse / structural issues from solver
    raw_step_solver_response: str  # raw VLM output from step solver
    evaluation: str
    evaluation_feedback: str
    iteration: int
    max_iterations: int
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
        "current_step_statistics_text": "",
        "current_step_graph_images": [],
        "current_step_graph_descriptions": [],
        "verified_labels": [],
        "verified_label_reasoning": {},
        "proposed_sub_start": 0,
        "proposed_sub_end": 0,
        "proposed_labels": [],
        "proposed_label_annotations": [],
        "proposed_reasoning": "",
        "parse_warnings": [],
        "raw_step_solver_response": "",
        "evaluation": "",
        "evaluation_feedback": "",
        "iteration": 0,
        "max_iterations": 3,
        "final_labels": [],
        "final_label_annotations": [],
        "final_reasoning": "",
        "final_sub_start": 0,
        "final_sub_end": 0,
        "messages": [],
    }


# ---------------------------------------------------------------------------
# LLM function holders (set per-run, avoids passing callables through state).
#   "vlm" – generate(prompt, images=None) → str   (vision + text)
#   "llm" – generate(prompt) → str                (text-only, for routing)
# ---------------------------------------------------------------------------

_llm_holder: Dict[str, Optional[Callable]] = {"vlm": None, "llm": None}
_llm_lock = threading.Lock()

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


def _build_label_context(current_labels: List[str]) -> str:
    """Build a structured description of available labels and guidelines."""
    lines: list[str] = []

    # Current labels
    if current_labels:
        readable = [LABEL_MAPPING.get(l, l) for l in current_labels]
        lines.append(f"Currently assigned labels: {', '.join(readable)}")
    else:
        lines.append("No labels currently assigned.")

    lines.append("")

    # Available labels (names + IDs only)
    lines.append("=== Available Labels ===")
    for lid in LABEL_MAPPING:
        display = LABEL_MAPPING[lid]
        lines.append(f"- {display} ({lid})")

    return "\n".join(lines)


def _build_subsegment_context(
    parent_main_labels: List[str],
    existing_children: List[dict],
    parent_start: int,
    parent_end: int,
    catalog: Optional[LabelCatalog] = None,
) -> str:
    """Build rich context for the **step solver**.

    Combines in one pass:
    - Parent Main Labels as hints
    - Sub-labels from those parent categories (YAML-backed descriptions)
    - ALL segment types (ST1-ST6)
    - Circuit section labels from parent's circuit category
    - Existing child sub-segments (for duplicate avoidance)
    """
    cat = catalog or get_label_catalog()
    lines: list[str] = []

    # Parent context
    if parent_main_labels:
        readable = [LABEL_MAPPING.get(l, l) for l in parent_main_labels]
        lines.append(f"=== Parent Segment (hints) ===")
        lines.append(f"Main labels: {', '.join(readable)} (IDs: {json.dumps(parent_main_labels)})")
        lines.append(f"Range: [{parent_start}, {parent_end}] (length: {parent_end - parent_start} data points)")
    lines.append("")

    # Existing children (duplicate avoidance)
    if existing_children:
        lines.append("=== Already Discovered Sub-Segments ===")
        lines.append("Do NOT re-propose a sub-segment that overlaps significantly with these:")
        for child in existing_children:
            child_labels = [LABEL_MAPPING.get(l, l) for l in child.get("labels", [])]
            lines.append(
                f"- Range [{child['start_index']}, {child['end_index']}] "
                f"(length: {child['end_index'] - child['start_index']}): "
                f"{', '.join(child_labels)}"
            )
        lines.append("Find a DIFFERENT notable region within the parent range.")
        lines.append("")

    # Sub-labels from parent categories
    parents_with_subs = [pid for pid in parent_main_labels if LABEL_CATEGORIES.get(pid)]
    if parents_with_subs:
        for pid in parents_with_subs:
            parent_entry = cat.get_label(pid)
            parent_name = parent_entry.name if parent_entry else LABEL_MAPPING.get(pid, pid)
            subs = cat.get_sublabels(pid)
            if not subs:
                continue
            lines.append(f"=== Sub-Labels for {parent_name} ({pid}) ===")
            for entry in subs:
                lines.append(f"- {entry.name} ({entry.id}): {entry.description}")
            lines.append("")

    # Segment types (always available)
    lines.append("=== Segment Type (pick one) ===")
    for entry in cat.get_segment_types():
        lines.append(f"- {entry.name} ({entry.id}): {entry.description}")
    lines.append("")

    # Circuit section labels (from parent's circuit category)
    circuit_parents = [pid for pid in parent_main_labels if pid in ("brands_hatch", "silverstone")]
    for cpid in circuit_parents:
        subs = cat.get_sublabels(cpid)
        if subs:
            parent_name = LABEL_MAPPING.get(cpid, cpid)
            lines.append(f"=== Circuit Sections for {parent_name} ({cpid}) ===")
            for entry in subs:
                lines.append(f"- {entry.name} ({entry.id}): {entry.description}")
            lines.append("")

    return "\n".join(lines)


def _build_sublabel_context(
    parent_ids: List[str],
    current_labels: List[str],
    catalog: Optional[LabelCatalog] = None,
) -> str:
    """Build rich context for the **sub-label solver** (phase 2).

    Only includes sub-labels for the *chosen* parent categories so the
    search space stays small.
    """
    cat = catalog or get_label_catalog()
    lines: list[str] = []

    # Context: what was already decided
    if current_labels:
        readable = [LABEL_MAPPING.get(l, l) for l in current_labels]
        lines.append(f"Currently assigned labels: {', '.join(readable)}")
    lines.append("")

    parents_with_subs = [pid for pid in parent_ids if LABEL_CATEGORIES.get(pid)]

    if not parents_with_subs:
        lines.append("No parent labels have sub-labels. Nothing to refine.")
        return "\n".join(lines)

    lines.append("")

    for pid in parents_with_subs:
        parent_entry = cat.get_label(pid)
        parent_name = parent_entry.name if parent_entry else LABEL_MAPPING.get(pid, pid)
        subs = cat.get_sublabels(pid)
        if not subs:
            continue
        lines.append(f"=== Sub-Labels for {parent_name} ({pid}) ===")
        for entry in subs:
            lines.append(f"- {entry.name} ({entry.id}): {entry.description}")
        lines.append("")

    return "\n".join(lines)


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
# Agent node functions
# ---------------------------------------------------------------------------


def planner_node(state: AnnotationState) -> dict:
    """Analyse the segment and produce an analysis plan for sub-segment discovery.

    The planner inspects the telemetry summary, parent Main Labels, and
    existing children to decide *which statistics and graphs* are needed
    to find a new sub-segment within the parent range.
    """
    from .annotation_agent_tools import AGENT_GRAPH_DEFINITIONS

    iteration = state.get("iteration", 0) + 1
    feedback = state.get("evaluation_feedback", "")
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

    if feedback:
        prompt_parts.extend([
            "",
            f"=== Previous Attempt Feedback (iteration {iteration - 1}) ===",
            feedback,
            "Revise your plan to address the evaluator's concerns.",
        ])
        if "JSON format" in feedback or "required key" in feedback.lower():
            prompt_parts.extend([
                "",
                "**FORMAT REMINDER**: The evaluator detected JSON format problems. "
                "In your revised plan, explicitly instruct the solver to output "
                'well-formed JSON inside ```json``` code blocks with a "labels" array '
                "where each entry has: label_id, start_index, end_index, reasoning.",
            ])
        if "outside parent" in feedback.lower() or "exceeds parent" in feedback.lower():
            prompt_parts.extend([
                "",
                f"**RANGE REMINDER**: The proposed sub-segment range must be within "
                f"[{parent_start}, {parent_end}]. The previous proposal violated "
                f"this constraint.",
            ])

    prompt = "\n".join(prompt_parts)

    # Call VLM (text-only for planning — no images needed)
    vlm_fn = _llm_holder.get("vlm")
    if vlm_fn:
        plan = vlm_fn(prompt)
    else:
        plan = "[VLM not available — using passthrough plan] " \
               "Examine all telemetry features and propose the most fitting labels."

    # Parse structured tool requests from planner output
    parsed_steps = _parse_planner_steps(plan, [])

    msg = {"role": "planner", "iteration": iteration, "content": plan}
    messages = list(state.get("messages", []))
    messages.append(msg)

    return {
        "plan": plan,
        "plan_steps": parsed_steps,
        "current_step_index": 0,
        "step_results": [],
        "all_graph_images": [],
        "all_graph_descriptions": [],
        "iteration": iteration,
        "messages": messages,
    }


# ---------------------------------------------------------------------------
# Tool executor node
# ---------------------------------------------------------------------------


def steps_data_fetcher_node(state: AnnotationState) -> Dict[str, Any]:
    """
    Executes the tools (statistics, graphs) requested for the current step in the plan.
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
        return {
            "current_step_statistics_text": "",
            "current_step_graph_images": [],
            "current_step_graph_descriptions": [],
        }

    step = plan_steps[current_step_index]
    requested_graphs = step.get("requested_graphs", [])
    LOGGER.info(f"Executing step {current_step_index + 1}/{len(plan_steps)}: {step.get('description')}")
    LOGGER.info(f"  - Requested graphs: {requested_graphs}")

    stats_text = ""
    image_bytes_list = []
    desc_list = []
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


    # Create a tool message for logging/debugging
    tool_outputs = []
    if image_bytes_list:
        # For logging, just note that images were generated
        content = f"Generated {len(image_bytes_list)} graphs: {', '.join(desc_list)}"
        tool_outputs.append({"tool_name": "generate_telemetry_graphs", "content": content})

    if tool_outputs:
        messages.append({"role": "tool", "content": tool_outputs})

    return {
        "current_step_statistics_text": stats_text,
        "current_step_graph_images": image_bytes_list,
        "current_step_graph_descriptions": desc_list,
        "all_graph_images": state.get("all_graph_images", []) + image_bytes_list,
        "all_graph_descriptions": state.get("all_graph_descriptions", []) + desc_list,
        "messages": messages,
    }


def _call_vlm(
    prompt: str,
    graph_image_bytes: List[bytes],
) -> str:
    """Dispatch a prompt to the VLM, optionally with graph images."""
    vlm_fn = _llm_holder.get("vlm")
    if not vlm_fn:
        return ""

    if graph_image_bytes:
        return vlm_fn(prompt, graph_image_bytes)
    return vlm_fn(prompt)


def _extract_verdict(text: str) -> str:
    """Determine accept/reject from the evaluator's free-text output.

    Uses the text-only LLM to classify the evaluator's intent.  The
    evaluator writes naturally — no magic keywords required.  A short
    LLM call reads the evaluation and answers with a single word.

    Falls back to simple heuristics if the LLM is unavailable.
    """
    llm_fn = _llm_holder.get("llm")
    if llm_fn:
        classification_prompt = (
            "Read the following evaluation of a racing telemetry annotation "
            "proposal. Determine whether the evaluator thinks the proposal "
            "is good enough to accept, or needs changes (reject).\n\n"
            "=== Evaluation ===\n"
            f"{text}\n\n"
            "Does the evaluator accept or reject the proposal? "
            "Answer with exactly one word: ACCEPT or REJECT"
        )
        try:
            answer = llm_fn(classification_prompt).strip().upper()
            if "ACCEPT" in answer:
                return "accept"
            if "REJECT" in answer:
                return "reject"
        except Exception:
            LOGGER.debug("LLM verdict classification failed, using heuristic.")

    # Heuristic fallback: scan the last portion of the text
    tail = text[-500:].lower()
    # Look for strong negative signals
    negative_signals = [
        "missing", "incorrect", "should be", "not appropriate",
        "should change", "need to", "should also", "not present",
        "not fully supported", "inconsisten", "does not",
    ]
    positive_signals = [
        "well-supported", "correctly", "accurate", "appropriate",
        "good", "agree", "consistent", "matches",
    ]
    neg = sum(1 for s in negative_signals if s in tail)
    pos = sum(1 for s in positive_signals if s in tail)
    if pos > neg:
        return "accept"
    return "reject"


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


def _build_graph_prompt_section(
    graph_descriptions: List[str],
    graph_image_bytes: List[bytes],
) -> List[str]:
    """Return prompt lines describing telemetry graph context."""
    if not graph_descriptions:
        return []
    parts: list[str] = ["=== Telemetry Graphs ==="]
    if graph_image_bytes:
        parts.append(
            "The following graphs are provided as images. Analyse each "
            "image carefully — they correspond to the descriptions below:"
        )
    for i, desc in enumerate(graph_descriptions, 1):
        parts.append(f"Graph {i}: {desc}")
    parts.append("")
    return parts


# --- Step solver ------------------------------------------------------------


def step_reasoner_node(state: AnnotationState) -> Dict[str, Any]:
    """
    Reasons about the output of the current step's tool executions.
    """
    messages = list(state.get("messages", []))
    iteration = state.get("iteration", 0)
    plan_steps = state.get("plan_steps", [])
    current_step_index = state.get("current_step_index", 0)
    step = plan_steps[current_step_index]
    parent_main_labels = state.get("parent_main_labels", [])
    parent_start = state.get("parent_start", 0)
    parent_end = state.get("parent_end", 0)

    stats_text = state.get("current_step_statistics_text", "")
    graph_descriptions = state.get("current_step_graph_descriptions", [])
    graph_images = state.get("current_step_graph_images", [])

    prompt = (
        f"You are a telemetry graph describer.  Your ONLY job is to produce a "
        f"detailed, precise description of the data and graphs provided.  "
        f"Write in flowing prose paragraphs — do NOT use numbered lists, "
        f"bullet points, or step-by-step formatting.  Do NOT diagnose problems, "
        f"assign labels, or suggest what the observations mean.  Other nodes "
        f"in the pipeline will interpret your description — your job is to give "
        f"them accurate raw observations.\n\n"
        f"**Segment Context:**\n"
        f"- Index Range: {parent_start} to {parent_end}\n\n"
        f"**Analysis Goal: {step['description']}**\n\n"
        f"**Tool Outputs:**\n"
    )

    content_for_vlm = []

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
        "For each graph, write the paragraph description specified in the "
        "'what_to_describe' guidance above.  Cover every aspect listed, use "
        "the vocabulary terms provided, and avoid the listed description errors.  "
        "Your output should read as flowing prose — do NOT use numbered lists "
        "or bullet points.  Use exact index positions and numerical values "
        "wherever readable.\n\n"
        "IMPORTANT: Do NOT interpret, diagnose, or suggest labels.  Do NOT say "
        "'this indicates a mistake' or 'this suggests the driver did X wrong'.  "
        "Only describe WHAT you see — shapes, values, positions, colours, "
        "separations, sequences.  Downstream nodes will handle interpretation."
    )
    content_for_vlm.append({"type": "text", "text": prompt})

    for image_bytes in graph_images:
        import base64
        content_for_vlm.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64.b64encode(image_bytes).decode()}"}})

    vlm_response_text = _call_vlm(prompt, graph_images)
    messages.append({"role": "assistant", "content": vlm_response_text})

    # Increment index for the next step
    next_step_index = current_step_index + 1

    return {
        "current_step_index": next_step_index,
        "step_results": [{
            "iteration": iteration,
            "step_id": step["step_id"],
            "description": step["description"],
            "graph_observations": vlm_response_text,
            "graph_descriptions": graph_descriptions,
        }],
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
    """Filter candidate labels by embedding similarity to step reasoner evidence.

    Embeds the concatenated step-reasoner observations as a query, embeds
    each candidate label's name + description, then keeps only labels whose
    cosine similarity exceeds _SIMILARITY_THRESHOLD.  At least
    _MIN_FILTER_LABELS labels are always passed (highest scoring), and the
    result is capped at _MAX_FILTER_LABELS to keep the synthesizer prompt
    focused.

    Replaces the previous per-label VLM binary-check loop, eliminating N
    separate VLM calls and replacing them with a single batch embedding
    computation.
    """
    import numpy as np

    messages = list(state.get("messages", []))
    iteration = state.get("iteration", 0)
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

    if not shortlisted:
        LOGGER.info("Label similarity filter: no candidate labels.")
        messages.append({
            "role": "label_verifier",
            "iteration": iteration,
            "content": "No candidate labels available for parent categories.",
        })
        return {
            "verified_labels": [],
            "verified_label_reasoning": {},
            "messages": messages,
        }

    # Build query from this iteration's step reasoner outputs
    current_iteration_steps = [
        res for res in state.get("step_results", [])
        if res.get("iteration") == iteration
    ]
    evidence_parts: list[str] = [
        step["graph_observations"]
        for step in sorted(current_iteration_steps, key=lambda x: x["step_id"])
        if step.get("graph_observations")
    ]
    query_text = " ".join(evidence_parts).strip()

    if not query_text:
        LOGGER.warning("Label similarity filter: no evidence text; passing top candidates.")
        fallback = shortlisted[:_MAX_FILTER_LABELS]
        messages.append({
            "role": "label_verifier",
            "iteration": iteration,
            "content": f"No evidence text; passed top {len(fallback)} candidates unchanged.",
        })
        return {
            "verified_labels": fallback,
            "verified_label_reasoning": {lid: "No evidence available." for lid in fallback},
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
        "iteration": iteration,
        "content": (
            f"Embedding filter: {len(verified)}/{len(shortlisted)} labels passed "
            f"(threshold={_SIMILARITY_THRESHOLD}):\n{passed_log}"
            + (f"\n\nRejected:\n{rejected_log}" if rejected_log else "")
        ),
    })

    return {
        "verified_labels": verified,
        "verified_label_reasoning": reasoning_map,
        "messages": messages,
    }


def proposal_synthesizer_node(state: AnnotationState) -> Dict[str, Any]:
    """Assemble the final proposal from verified labels and step evidence.

    The heavy label selection work is done by label_shortlister and
    label_verifier.  This node only needs to determine precise
    sub-segment boundaries and emit well-formed JSON.
    """
    messages = list(state.get("messages", []))
    iteration = state.get("iteration", 0)
    parent_main_labels = state.get("parent_main_labels", [])
    existing_children = state.get("existing_children", [])
    parent_start = state.get("parent_start", 0)
    parent_end = state.get("parent_end", 0)
    segment_data = state.get("segment_data", {})
    verified_labels = state.get("verified_labels", [])
    verified_reasoning = state.get("verified_label_reasoning", {})

    # Collect step evidence for this iteration
    current_iteration_steps = [
        res for res in state.get("step_results", [])
        if res.get("iteration") == iteration
    ]
    evidence_summary = ""
    for step in sorted(current_iteration_steps, key=lambda x: x["step_id"]):
        evidence_summary += f"--- Graph {step['step_id']}: {step['description']} ---\n"
        if step.get("graph_descriptions"):
            evidence_summary += f"Graphs: {', '.join(step['graph_descriptions'])}\n"
        evidence_summary += f"Graph Observations: {step['graph_observations']}\n\n"

    # Build concise verified-label section (small, focused)
    verified_section_lines: list[str] = ["=== Verified Labels (use these) ==="]
    catalog = get_label_catalog()
    for lid in verified_labels:
        entry = catalog.get_label(lid)
        name = entry.name if entry else LABEL_MAPPING.get(lid, lid)
        justification = verified_reasoning.get(lid, "")
        verified_section_lines.append(f"- ID: {lid} | Name: {name} | {justification}")
    if not verified_labels:
        verified_section_lines.append("(No labels verified — propose the best-fit label IDs based on evidence.)")
    verified_section = "\n".join(verified_section_lines)

    # Existing children for duplicate avoidance
    children_section = ""
    if existing_children:
        lines = ["=== Already Discovered Sub-Segments (avoid overlap) ==="]
        for child in existing_children:
            child_labels = [LABEL_MAPPING.get(l, l) for l in child.get("labels", [])]
            lines.append(
                f"- [{child['start_index']}, {child['end_index']}]: "
                f"{', '.join(child_labels)}"
            )
        children_section = "\n".join(lines)

    prompt_parts = [
        "You are a racing telemetry analyst assembling a sub-segment proposal.",
        "",
        "=== Task ===",
        "Define exactly ONE sub-segment within the parent range "
        f"[{parent_start}, {parent_end}] "
        f"(length: {parent_end - parent_start} data points).",
        "Determine the precise start and end indices based on the evidence.",
        "",
        "=== Parent Segment ===",
        f"Main labels: {', '.join([LABEL_MAPPING.get(l, l) for l in parent_main_labels])}",
        "",
        "=== Graph Observations ===",
        evidence_summary,
        "",
    ]
    prompt_parts.extend([
        verified_section,
        "",
    ])
    if children_section:
        prompt_parts.extend([children_section, ""])
    prompt_parts.extend([
        "=== Instructions ===",
        "Use ONLY the verified labels listed above (you may drop some if "
        "the evidence does not support them).",
        "Each label may cover a different precise sub-range within "
        f"[{parent_start}, {parent_end}]. "
        "Determine separate start_index and end_index for every label based on where "
        "that specific behaviour begins and ends in the graphs. "
        "start_index must be strictly less than end_index for each label.",
        "For each label explain why those specific boundaries were chosen, "
        "referencing the graph observations from the step reasoners.",
        "",
        "Respond in this JSON format ONLY:",
        "```json",
        "{",
        '  "labels": [',
        '    {',
        '      "label_id": "LABEL_ID_1",',
        f'      "start_index": <integer within [{parent_start}, {parent_end}]>,',
        f'      "end_index": <integer within [{parent_start}, {parent_end}]>,',
        '      "reasoning": "..."',
        '    }',
        '  ]',
        "}",
        "```",
    ])

    prompt = "\n".join(prompt_parts)

    raw_response = _call_vlm(prompt, [])
    if not raw_response:
        raw_response = json.dumps({
            "start_index": parent_start,
            "end_index": min(parent_start + 10, parent_end),
            "proposed_labels": verified_labels,
            "reasoning": "Passthrough — VLM not available.",
        })

    # Parse
    sub_labels: list[str] = []
    label_proposals: list[dict] = []
    reasoning = raw_response
    parse_warnings: list[str] = list(state.get("parse_warnings", []))
    proposed_start = parent_start
    proposed_end = parent_end
    parsed = _parse_json_response(raw_response)
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
        reasoning = "; ".join(p["reasoning"] for p in label_proposals) if label_proposals else raw_response
        if starts:
            proposed_start = min(starts)
        if ends:
            proposed_end = max(ends)
    else:
        LOGGER.warning("Proposal synthesizer response was not valid JSON.")
        parse_warnings.append(
            "Proposal synthesizer output was not valid JSON. "
            "Raw response needs to be restructured."
        )

    # Validate hierarchy (auto-insert missing parents for sub-labels)
    fixed_labels, hierarchy_warnings = _validate_and_fix_hierarchy(sub_labels)
    for w in hierarchy_warnings:
        LOGGER.info("Hierarchy fix: %s", w)
    parse_warnings.extend(hierarchy_warnings)

    messages.append({"role": "assistant", "content": raw_response})

    return {
        "proposed_sub_start": proposed_start,
        "proposed_sub_end": proposed_end,
        "proposed_labels": fixed_labels,
        "proposed_label_annotations": label_proposals,
        "proposed_reasoning": reasoning,
        "parse_warnings": parse_warnings,
        "raw_step_solver_response": raw_response,
        "messages": messages,
    }


def _validate_solver_json_format(
    raw_subsegment: str,
    proposed_labels: List[str],
    parent_start: int,
    parent_end: int,
) -> List[str]:
    """Validate step solver JSON output and return detailed format issues.

    Checks the raw response for structural correctness: parsability,
    required top-level key "labels", and per-entry keys
    (label_id, start_index, end_index, reasoning), value types,
    range bounds, and valid label IDs.
    """
    issues: list[str] = []

    if not raw_subsegment:
        return issues

    parsed = _parse_json_response(raw_subsegment)
    if parsed is None:
        issues.append(
            "Step solver did not produce valid JSON. "
            'Expected a ```json``` code block containing an object with a "labels" '
            'array where each entry has: "label_id", "start_index", "end_index", "reasoning".'
        )
        return issues

    if "labels" not in parsed:
        issues.append(
            'Step solver JSON is missing required key "labels". '
            'The solver must include a "labels" array.'
        )
        return issues

    labels_arr = parsed.get("labels")
    if not isinstance(labels_arr, list):
        issues.append(
            f'"labels" must be a JSON array, got {type(labels_arr).__name__}.'
        )
        return issues

    if not labels_arr:
        issues.append('"labels" array is empty — at least one label entry is required.')

    for i, entry in enumerate(labels_arr):
        if not isinstance(entry, dict):
            issues.append(f'"labels[{i}]" must be an object, got {type(entry).__name__}.')
            continue

        for key in ("label_id", "start_index", "end_index", "reasoning"):
            if key not in entry:
                issues.append(f'"labels[{i}]" is missing required key "{key}".')

        lid = entry.get("label_id")
        if lid is not None:
            if not isinstance(lid, str):
                issues.append(
                    f'"labels[{i}].label_id" must be a string, got {type(lid).__name__}.'
                )
            elif lid not in LABEL_MAPPING:
                issues.append(f'"labels[{i}].label_id" contains unknown ID: "{lid}".')

        si = entry.get("start_index")
        ei = entry.get("end_index")
        if si is not None and not isinstance(si, (int, float)):
            issues.append(
                f'"labels[{i}].start_index" must be an integer, got {type(si).__name__}.'
            )
        if ei is not None and not isinstance(ei, (int, float)):
            issues.append(
                f'"labels[{i}].end_index" must be an integer, got {type(ei).__name__}.'
            )

        if isinstance(si, (int, float)) and isinstance(ei, (int, float)):
            si_int, ei_int = int(si), int(ei)
            if si_int >= ei_int:
                issues.append(
                    f'"labels[{i}]" start_index ({si_int}) must be strictly less than '
                    f'end_index ({ei_int}).'
                )
            if si_int < parent_start:
                issues.append(
                    f'"labels[{i}]" start_index ({si_int}) is outside parent range — '
                    f'must be >= {parent_start}.'
                )
            if ei_int > parent_end:
                issues.append(
                    f'"labels[{i}]" end_index ({ei_int}) exceeds parent end ({parent_end}) — '
                    f'must be <= {parent_end}.'
                )

        rs = entry.get("reasoning")
        if rs is not None and not isinstance(rs, str):
            issues.append(
                f'"labels[{i}].reasoning" must be a string, got {type(rs).__name__}.'
            )

    return issues


def evaluator_node(state: AnnotationState) -> dict:
    """Evaluate the step solver's proposed sub-segment.

    Validates the JSON format/range, then asks the VLM to review the
    proposed range and labels against the telemetry data. On accept the
    evaluator sets ``final_sub_start``, ``final_sub_end``, ``final_labels``,
    and ``final_reasoning``.
    """
    proposed_labels = state.get("proposed_labels", [])
    proposed_label_annotations = state.get("proposed_label_annotations", [])
    proposed_reasoning = state.get("proposed_reasoning", "")
    segment_data = state.get("segment_data", {})
    iteration = state.get("iteration", 0)
    all_graph_images = state.get("all_graph_images", [])
    all_graph_descriptions = state.get("all_graph_descriptions", [])
    parse_warnings = state.get("parse_warnings", [])
    raw_subsegment = state.get("raw_step_solver_response", "")
    parent_start = state.get("parent_start", 0)
    parent_end = state.get("parent_end", 0)
    proposed_sub_start = state.get("proposed_sub_start")
    proposed_sub_end = state.get("proposed_sub_end")
    parent_main_labels = state.get("parent_main_labels", [])
    messages = list(state.get("messages", []))

    # --- JSON format / range validation ---
    format_issues = _validate_solver_json_format(
        raw_subsegment, proposed_labels, parent_start, parent_end,
    )
    all_warnings = list(parse_warnings) + format_issues

    # --- Structural pre-check: auto-reject if JSON was broken ---
    if all_warnings:
        warning_text = "\n".join(f"- {w}" for w in all_warnings)
        feedback = (
            f"JSON format / structural issues detected in solver output:\n"
            f"{warning_text}\n\n"
            "=== Required JSON Format ===\n"
            "The step solver must output:\n"
            "```json\n"
            '{"labels": [{"label_id": "LABEL_ID", "start_index": <int>, '
            '"end_index": <int>, "reasoning": "..."}]}\n'
            "```\n\n"
            "Each label must have its own start_index and end_index "
            f"within parent bounds [{parent_start}, {parent_end}].\n"
            "Instruct the solver to fix these issues in the next iteration."
        )
        LOGGER.warning("Evaluator auto-reject (format): %s", warning_text)
        msg = {
            "role": "evaluator",
            "iteration": iteration,
            "verdict": "reject",
            "content": feedback,
        }
        messages.append(msg)
        return {
            "evaluation": "reject",
            "evaluation_feedback": feedback,
            "parse_warnings": [],
            "messages": messages,
        }

    # --- VLM evaluation ---

    # Filter step results for the current iteration
    current_iteration_steps = [res for res in state.get("step_results", []) if res.get("iteration") == iteration]

    # Format the evidence from all steps
    evidence_summary = ""
    for step in sorted(current_iteration_steps, key=lambda x: x['step_id']):
        evidence_summary += f"--- Graph {step['step_id']}: {step['description']} ---\n"
        if step.get('graph_descriptions'):
            evidence_summary += f"Graphs: {', '.join(step['graph_descriptions'])}\n"
        evidence_summary += f"Graph Observations: {step['graph_observations']}\n\n"

    prompt_parts = [
        "You are a meticulous quality assurance specialist. Your job is to critically evaluate a proposed sub-segment annotation based on all available evidence.",
        "",
        "=== Parent Segment Context ===",
        f"Main labels: {[LABEL_MAPPING.get(l, l) for l in parent_main_labels]}",
        f"Range: [{parent_start}, {parent_end}]",
        "",
        "=== Graph Observations ===",
        evidence_summary,
        "",
    ]
    prompt_parts.extend(
        _build_graph_prompt_section(all_graph_descriptions, all_graph_images)
    )
    # Build per-label annotation lines for the evaluator prompt
    if proposed_label_annotations:
        label_ann_lines = ["=== Proposed Sub-Segment (per-label boundaries) ==="]
        for ann in proposed_label_annotations:
            lid = ann["label_id"]
            name = LABEL_MAPPING.get(lid, lid)
            label_ann_lines.append(
                f"- {name} ({lid}): [{ann['start_index']}, {ann['end_index']}] — {ann['reasoning']}"
            )
        label_ann_lines.append(f"Overall envelope: [{proposed_sub_start}, {proposed_sub_end}]")
        proposed_segment_section = label_ann_lines
    else:
        proposed_segment_section = [
            "=== Proposed Sub-Segment ===",
            f"Proposed Labels: {[LABEL_MAPPING.get(l, l) for l in proposed_labels]}",
            f"Proposed Range: [{proposed_sub_start}, {proposed_sub_end}]",
            f"Proposer's Reasoning: {proposed_reasoning}",
        ]

    prompt_parts.extend(proposed_segment_section)
    prompt_parts.extend([
        "",
        "=== Your Task ===",
        "Critically evaluate the proposal. Does the reasoning logically follow from the graph observations? Are the proposed start/end indices tightly aligned with the evidence in the telemetry data and graphs? Are the labels appropriate?",
        "1. **Analyze:** Compare the proposal to the evidence. Look for inconsistencies, loose boundaries, or flawed logic.",
        "2. **Decide:** Output a single word: `pass` or `fail`.",
        "3. **Justify:** On a new line, provide a concise but detailed justification for your decision. If it fails, explain exactly what is wrong (e.g., 'The end_index is 50 points too late, as the graph shows the anomaly ends at index 1500,' or 'The reasoning ignores the braking data from step 2.').",
    ])

    prompt = "\n".join(prompt_parts)

    evaluation = _call_vlm(prompt, all_graph_images)
    if not evaluation:
        evaluation = "fail\nCould not evaluate."

    verdict_str, *feedback_lines = evaluation.strip().split("\n", 1)
    verdict = "fail"
    if "pass" in verdict_str.lower():
        verdict = "pass"
    feedback = feedback_lines[0].strip() if feedback_lines else "No feedback provided."

    msg = {
        "role": "evaluator",
        "iteration": iteration,
        "content": feedback,
        "verdict": verdict,
    }
    messages.append(msg)

    # On accept, promote proposed to final
    if verdict == "pass":
        return {
            "evaluation": verdict,
            "evaluation_feedback": feedback,
            "final_sub_start": proposed_sub_start,
            "final_sub_end": proposed_sub_end,
            "final_labels": proposed_labels,
            "final_label_annotations": proposed_label_annotations,
            "final_reasoning": proposed_reasoning,
            "messages": messages,
        }

    # On max-iteration exhaustion, promote the last proposal to final so the
    # pipeline always returns a useful result even without an explicit "pass".
    if state.get("iteration", 0) >= state.get("max_iterations", 3):
        LOGGER.info(
            "Max iterations reached without 'pass'; promoting last proposal to final."
        )
        return {
            "evaluation": verdict,
            "evaluation_feedback": feedback,
            "final_sub_start": proposed_sub_start,
            "final_sub_end": proposed_sub_end,
            "final_labels": proposed_labels,
            "final_label_annotations": proposed_label_annotations,
            "final_reasoning": proposed_reasoning,
            "messages": messages,
        }

    return {
        "evaluation": verdict,
        "evaluation_feedback": feedback,
        "messages": messages,
    }


def should_retry(state: AnnotationState) -> Literal["planner", "end"]:
    """Decide whether to loop back to the planner or finish."""
    if state.get("evaluation") == "pass":
        return "end"
    if state.get("iteration", 0) >= state.get("max_iterations", 3):
        # Max retries reached — accept the latest proposal as-is
        return "end"
    return "planner"


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


def build_annotation_graph(
    vlm_llm: Callable,
    text_llm: Callable,
) -> CompiledGraph:
    """Build the annotation agent graph."""
    with _llm_lock:
        _llm_holder["vlm"] = vlm_llm
        _llm_holder["llm"] = text_llm

    graph = StateGraph(AnnotationState)

    graph.add_node("planner", planner_node)
    graph.add_node("steps_data_fetcher", steps_data_fetcher_node)
    graph.add_node("step_reasoner", step_reasoner_node)
    graph.add_node("label_verifier", label_verifier_node)
    graph.add_node("proposal_synthesizer", proposal_synthesizer_node)
    graph.add_node("evaluator", evaluator_node)

    graph.set_entry_point("planner")
    graph.add_edge("planner", "steps_data_fetcher")
    graph.add_edge("steps_data_fetcher", "step_reasoner")
    graph.add_conditional_edges(
        "step_reasoner",
        step_router,
        {
            "steps_data_fetcher": "steps_data_fetcher",
            "label_verifier": "label_verifier",
        },
    )
    graph.add_edge("label_verifier", "proposal_synthesizer")
    graph.add_edge("proposal_synthesizer", "evaluator")
    graph.add_conditional_edges(
        "evaluator",
        should_retry,
        {"planner": "planner", "end": END},
    )

    return graph.compile()


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------


@dataclass
class AnnotationPipelineConfig:
    """Configuration for the annotation pipeline."""

    max_new_tokens: int = 512
    temperature: float = 0.7
    max_iterations: int = 3

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
) -> AnnotationResult:
    """Execute the planner → tool_executor → step_solver → evaluator pipeline.

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
        ``fn(step_name: str, iteration: int, detail: str)``
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
            vlm_prompt_callback(prompt)
        return vlm_service.generate(
            prompt,
            images=images,
            max_tokens=config.max_new_tokens,
            temperature=config.temperature,
            stream_callback=vlm_stream_callback,
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

    # Register VLM + LLM functions for this run
    with _llm_lock:
        _llm_holder["vlm"] = vlm_generate
        _llm_holder["llm"] = llm_generate

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
        "max_iterations": config.max_iterations,
    })

    # Stream through graph nodes for progress reporting
    final_state = dict(initial_state)
    for event in graph.stream(initial_state, config={"recursion_limit": 100}):
        for node_name, node_output in event.items():
            final_state.update(node_output)
            if progress_callback:
                iteration = final_state.get("iteration", 0)
                detail = ""
                if node_name == "planner":
                    req_s = final_state.get("requested_statistics", [])
                    req_g = final_state.get("requested_graphs", [])
                    detail = (
                        f"Requested stats: {req_s or 'all'}, "
                        f"graphs: {req_g or 'all'}"
                    )
                elif node_name == "steps_data_fetcher":
                    n_graphs = len(final_state.get("current_step_graph_descriptions", []))
                    detail = f"Gathered {n_graphs} graph(s)"
                elif node_name == "label_verifier":
                    verified = final_state.get("verified_labels", [])
                    detail = (
                        f"Verified {len(verified)} labels: "
                        f"{', '.join(LABEL_MAPPING.get(l, l) for l in verified)}"
                    )
                elif node_name == "proposal_synthesizer":
                    labels = final_state.get("proposed_labels", [])
                    ss = final_state.get("proposed_sub_start", "?")
                    se = final_state.get("proposed_sub_end", "?")
                    detail = (
                        f"Range [{ss}, {se}], "
                        f"labels: {', '.join(LABEL_MAPPING.get(l, l) for l in labels)}"
                    )
                elif node_name == "evaluator":
                    detail = (
                        f"{final_state.get('evaluation', '').upper()}: "
                        f"{final_state.get('evaluation_feedback', '')}"
                    )
                progress_callback(node_name, iteration, detail)

    # Clean up VLM/LLM holders
    with _llm_lock:
        _llm_holder["vlm"] = None
        _llm_holder["llm"] = None

    accepted = final_state.get("evaluation") == "pass"

    # Safety-net: if final_* were never written (e.g. an unexpected code path),
    # fall back to the last proposed values so the result is never empty.
    final_labels = final_state.get("final_labels") or final_state.get("proposed_labels", [])
    final_reasoning = final_state.get("final_reasoning") or final_state.get("proposed_reasoning", "")
    final_sub_start = final_state.get("final_sub_start") or final_state.get("proposed_sub_start")
    final_sub_end = final_state.get("final_sub_end") or final_state.get("proposed_sub_end")

    return AnnotationResult(
        sub_start=final_sub_start,
        sub_end=final_sub_end,
        final_labels=final_labels,
        final_reasoning=final_reasoning,
        accepted=accepted,
        iterations=final_state.get("iteration", 0),
        messages=final_state.get("messages", []),
        graph_images=final_state.get("all_graph_images", []),
    )
