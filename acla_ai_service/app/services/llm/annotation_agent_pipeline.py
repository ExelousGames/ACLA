"""
Multi-agent annotation pipeline using LangGraph.

Implements a Planner → Tool Executor → Solver → Evaluator cycle for
automated telemetry segment annotation.  The **Tool Executor** gathers
evidence via two tool categories:

* **get_telemetry_statistics** – numerical summaries (text)
* **generate_telemetry_graphs** – rendered graphs (PIL images)

The Vision Language Model (VLM) receives both the statistical text *and*
the graph images, replicating the same visual evidence a human annotator
would see.

Graph flow:
    planner  →  tool_executor  →  solver  →  evaluator  ─┐
       ↑                                                   │
       └───────────── (retry if rejected) ─────────────────┘
"""

from __future__ import annotations

import io
import json
import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Annotated, Callable, Dict, List, Literal, Optional, Sequence, TypedDict

from langgraph.graph import StateGraph, END
from PIL import Image

from app.models.segment_models import (
    LABEL_MAPPING,
    LABEL_CATEGORIES,
    MAIN_LABEL_GUIDELINES,
)
from app.models.label_catalog import get_label_catalog, LabelCatalog

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
    label_guidelines: dict
    # --- planner tool requests (selective execution) ---
    requested_statistics: list   # stat category IDs from STATISTIC_CATEGORIES
    requested_graphs: list       # graph IDs from AGENT_GRAPH_DEFINITIONS
    # --- tool outputs ---
    tool_statistics_text: str
    tool_graph_images: list  # list of bytes (PNG images)
    tool_graph_descriptions: list
    # --- agent state ---
    plan: str
    proposed_sub_start: int      # VLM-proposed sub-segment start index
    proposed_sub_end: int        # VLM-proposed sub-segment end index
    proposed_labels: list
    proposed_reasoning: str
    parse_warnings: list         # JSON parse / structural issues from solver
    raw_subsegment_solver_response: str  # raw VLM output from subsegment solver
    evaluation: str
    evaluation_feedback: str
    iteration: int
    max_iterations: int
    final_labels: list
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
        "label_guidelines": {},
        "requested_statistics": [],
        "requested_graphs": [],
        "tool_statistics_text": "",
        "tool_graph_images": [],
        "tool_graph_descriptions": [],
        "plan": "",
        "proposed_sub_start": 0,
        "proposed_sub_end": 0,
        "proposed_labels": [],
        "proposed_reasoning": "",
        "parse_warnings": [],
        "raw_subsegment_solver_response": "",
        "evaluation": "",
        "evaluation_feedback": "",
        "iteration": 0,
        "max_iterations": 3,
        "final_labels": [],
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
    """Build rich context for the **subsegment solver**.

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



    """Create a concise text summary of the telemetry slice.

    Delegates to the tools module when the pre-computed statistics text is
    not already available in ``segment_data``.
    """
    from .annotation_agent_tools import format_statistics_as_text
    return format_statistics_as_text(segment_data)


# ---------------------------------------------------------------------------
# Planner tool-request parser
# ---------------------------------------------------------------------------


def _parse_planner_tool_requests(
    plan_text: str,
) -> Dict[str, List[str]]:
    """Extract the structured tool-request JSON from the planner output.

    The planner is instructed to end its response with a JSON block like::

        ```json
        {"requested_statistics": ["speed", "throttle"], "requested_graphs": ["speed", "throttle"]}
        ```

    Returns
    -------
    dict with keys ``requested_statistics`` and ``requested_graphs``.
    If parsing fails, both lists are empty (caller treats empty as "all").
    """
    from .annotation_agent_tools import ALL_STATISTIC_CATEGORY_IDS, AGENT_GRAPH_DEFINITIONS

    valid_graph_ids = {g["id"] for g in AGENT_GRAPH_DEFINITIONS}
    valid_stat_ids = set(ALL_STATISTIC_CATEGORY_IDS)

    result: Dict[str, List[str]] = {
        "requested_statistics": [],
        "requested_graphs": [],
    }

    try:
        json_str = plan_text
        if "```json" in json_str:
            json_str = json_str.split("```json")[-1].split("```")[0]
        elif "```" in json_str:
            # Try last code block
            parts = json_str.split("```")
            for part in reversed(parts):
                stripped = part.strip()
                if stripped.startswith("{"):
                    json_str = stripped
                    break

        parsed = json.loads(json_str.strip())

        raw_stats = parsed.get("requested_statistics", [])
        if isinstance(raw_stats, list):
            result["requested_statistics"] = [
                s for s in raw_stats if s in valid_stat_ids
            ]

        raw_graphs = parsed.get("requested_graphs", [])
        if isinstance(raw_graphs, list):
            result["requested_graphs"] = [
                g for g in raw_graphs if g in valid_graph_ids
            ]
    except (json.JSONDecodeError, IndexError, KeyError, ValueError):
        LOGGER.debug("Planner did not produce parseable tool requests; using all tools.")

    return result


# ---------------------------------------------------------------------------
# Telemetry helpers
# ---------------------------------------------------------------------------


def _summarise_telemetry(segment_data: dict) -> str:
    """Convert a segment_data dict into human-readable text for LLM prompts."""
    from .annotation_agent_tools import format_statistics_as_text
    return format_statistics_as_text(segment_data)


# ---------------------------------------------------------------------------
# Agent node functions
# ---------------------------------------------------------------------------


def planner_node(state: AnnotationState) -> dict:
    """Analyse the segment and produce an analysis plan for sub-segment discovery.

    The planner inspects the telemetry summary, parent Main Labels, and
    existing children to decide *which statistics and graphs* are needed
    to find a new sub-segment within the parent range.
    """
    from .annotation_agent_tools import STATISTIC_CATEGORIES, AGENT_GRAPH_DEFINITIONS

    iteration = state.get("iteration", 0) + 1
    feedback = state.get("evaluation_feedback", "")
    parent_main_labels = state.get("parent_main_labels", [])
    existing_children = state.get("existing_children", [])
    parent_start = state.get("parent_start", 0)
    parent_end = state.get("parent_end", 0)
    segment_data = state.get("segment_data", {})

    telemetry_summary = _summarise_telemetry(segment_data)

    # Build tool catalogue for the planner
    stat_catalogue = ", ".join(
        f"`{cid}` ({cdef['description']})"
        for cid, cdef in STATISTIC_CATEGORIES.items()
    )
    graph_catalogue = ", ".join(
        f"`{gdef['id']}` ({gdef['title']})"
        for gdef in AGENT_GRAPH_DEFINITIONS
    )

    main_readable = [LABEL_MAPPING.get(l, l) for l in parent_main_labels]

    prompt_parts = [
        "You are a racing telemetry analyst planning a sub-segment discovery strategy.",
        "",
        "=== Parent Segment ===",
        f"Main labels: {', '.join(main_readable)} (IDs: {json.dumps(parent_main_labels)})",
        f"Range: [{parent_start}, {parent_end}] (length: {parent_end - parent_start} data points)",
        "",
    ]

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
        "1. **get_telemetry_statistics** — numerical statistics (mean, min, max, std) "
        "and player-vs-expert deltas.",
        f"   Available statistic categories: {stat_catalogue}",
        "2. **generate_telemetry_graphs** — visual telemetry graphs rendered as images.",
        f"   Available graph IDs: {graph_catalogue}",
        "   The solver has a Vision Language Model and CAN analyse the "
        "graph images directly.",
    ])

    prompt_parts.extend([
        "",
        "=== Telemetry Data (Statistics Tool Output) ===",
        telemetry_summary,
        "",
        "=== Task ===",
        "Plan which statistics and graphs to generate to help discover "
        "ONE notable sub-segment within the parent segment range. "
        "A sub-segment is a contiguous region where a specific event "
        "or behaviour occurs.",
        "",
        "At the end of your response, output a JSON block specifying "
        "which tools to run:",
        "```json",
        '{',
        '  "requested_statistics": ["category_id_1", "category_id_2"],',
        '  "requested_graphs": ["graph_id_1", "graph_id_2"]',
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
                "well-formed JSON inside ```json``` code blocks with the exact "
                "required keys (start_index, end_index, proposed_labels, reasoning).",
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
    tool_requests = _parse_planner_tool_requests(plan)

    msg = {"role": "planner", "iteration": iteration, "content": plan}
    messages = list(state.get("messages", []))
    messages.append(msg)

    return {
        "plan": plan,
        "iteration": iteration,
        "requested_statistics": tool_requests["requested_statistics"],
        "requested_graphs": tool_requests["requested_graphs"],
        "messages": messages,
    }


# ---------------------------------------------------------------------------
# Tool executor node
# ---------------------------------------------------------------------------


def tool_executor_node(state: AnnotationState) -> dict:
    """Execute the annotation tools to gather evidence for the solver.

    Only runs the statistics categories and graphs that the planner
    requested.  If the planner did not produce parseable requests
    (empty lists), falls back to running all tools.
    """
    from .annotation_agent_tools import (
        get_telemetry_statistics,
        format_statistics_as_text,
        generate_telemetry_graphs,
    )

    segment_data = state.get("segment_data", {})
    df = state.get("df_ref")
    start = segment_data.get("start_index", 0)
    end = segment_data.get("end_index", 0)
    session_id = segment_data.get("session_id", "unknown")

    req_stats = state.get("requested_statistics", [])
    req_graphs = state.get("requested_graphs", [])

    # --- Tool 1: statistics (selective) ---
    stats = get_telemetry_statistics(
        df, start, end, session_id,
        stat_categories=req_stats or None,  # None = all
    )
    stats_text = format_statistics_as_text(stats)

    # --- Tool 2: graphs (selective) ---
    graph_image_bytes: list = []
    graph_descriptions: list = []

    if df is not None:
        graph_results = generate_telemetry_graphs(
            df, start, end,
            graph_ids=req_graphs or None,  # None = all
        )
        for img, desc in graph_results:
            graph_descriptions.append(desc)
            # Serialize PIL Image → PNG bytes for state transport
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            graph_image_bytes.append(buf.getvalue())

    msg = {
        "role": "tool_executor",
        "iteration": state.get("iteration", 0),
        "content": (
            f"Statistics: {len(stats.get('telemetry_summary', {}))} columns "
            f"(categories: {req_stats or 'all'}). "
            f"Graphs: {len(graph_descriptions)} "
            f"(ids: {req_graphs or 'all'})"
            f"{' with images' if graph_image_bytes else ' (descriptions only)'}."
        ),
    }
    messages = list(state.get("messages", []))
    messages.append(msg)

    return {
        "segment_data": stats,  # overwrite with full stats
        "tool_statistics_text": stats_text,
        "tool_graph_images": graph_image_bytes,
        "tool_graph_descriptions": graph_descriptions,
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


# --- Subsegment solver -----------------------------------------------------


def subsegment_solver_node(state: AnnotationState) -> dict:
    """Propose ONE sub-segment (start, end, labels) within the parent range.

    Uses :func:`_build_subsegment_context` to provide parent Main Labels as
    hints, available sub-labels, segment types, circuit sections, and
    existing children for duplicate avoidance.  Calls
    :func:`_validate_and_fix_hierarchy` to auto-insert missing parents.
    """
    plan = state.get("plan", "")
    parent_main_labels = state.get("parent_main_labels", [])
    existing_children = state.get("existing_children", [])
    parent_start = state.get("parent_start", 0)
    parent_end = state.get("parent_end", 0)
    segment_data = state.get("segment_data", {})
    stats_text = state.get("tool_statistics_text", "")
    graph_descriptions = state.get("tool_graph_descriptions", [])
    graph_image_bytes = state.get("tool_graph_images", [])

    subsegment_context = _build_subsegment_context(
        parent_main_labels, existing_children, parent_start, parent_end,
    )
    telemetry_summary = stats_text or _summarise_telemetry(segment_data)

    main_readable = [LABEL_MAPPING.get(l, l) for l in parent_main_labels]

    prompt_parts = [
        "You are a racing telemetry analyst discovering sub-segments.",
        "",
        "=== Task ===",
        "Find exactly ONE notable sub-segment within the parent segment "
        f"range [{parent_start}, {parent_end}] "
        f"(total length: {parent_end - parent_start} data points).",
        "A sub-segment is a contiguous region where a specific event or "
        "behaviour occurs (e.g. a braking mistake, a corner apex, a "
        "recovery manoeuvre).",
        "",
        "=== Parent Segment Main Labels ===",
        f"{', '.join(main_readable)} (IDs: {json.dumps(parent_main_labels)})",
        "Use these as hints for what kind of events to look for.",
        "",
        "=== Analysis Plan ===",
        plan,
        "",
        "=== Telemetry Statistics ===",
        telemetry_summary,
        "",
    ]
    prompt_parts.extend(
        _build_graph_prompt_section(graph_descriptions, graph_image_bytes)
    )
    prompt_parts.extend([
        subsegment_context,
        "",
        "=== Instructions ===",
        "Propose ONE sub-segment with its index range and labels.",
        f"The start_index must be >= {parent_start} and end_index must be <= {parent_end}.",
        "start_index must be strictly less than end_index.",
        "In your reasoning, describe the chosen range as [start_index, end_index] with its "
        "length (end_index - start_index) and explain why those specific boundaries were chosen.",
        "You MUST respond in the following JSON format only:",
        "```json",
        '{',
        '  "start_index": <int>,',
        '  "end_index": <int>,',
        '  "proposed_labels": ["LABEL_ID_1", "LABEL_ID_2", ...],',
        '  "reasoning": "Your detailed reasoning for the chosen range and labels."',
        '}',
        "```",
    ])

    prompt = "\n".join(prompt_parts)

    raw_response = _call_vlm(prompt, graph_image_bytes)
    if not raw_response:
        raw_response = json.dumps({
            "start_index": parent_start,
            "end_index": min(parent_start + 10, parent_end),
            "proposed_labels": [],
            "reasoning": "Passthrough — VLM not available.",
        })

    # Parse
    sub_labels: list[str] = []
    reasoning = raw_response
    parse_warnings: list[str] = list(state.get("parse_warnings", []))
    proposed_start = parent_start
    proposed_end = parent_end
    parsed = _parse_json_response(raw_response)
    if parsed:
        sub_labels = [l for l in parsed.get("proposed_labels", []) if l in LABEL_MAPPING]
        reasoning = parsed.get("reasoning", raw_response)
        if not isinstance(reasoning, str):
            reasoning = json.dumps(reasoning) if isinstance(reasoning, dict) else str(reasoning)
        # Extract range
        raw_start = parsed.get("start_index")
        raw_end = parsed.get("end_index")
        if isinstance(raw_start, (int, float)):
            proposed_start = int(raw_start)
        if isinstance(raw_end, (int, float)):
            proposed_end = int(raw_end)
    else:
        LOGGER.warning("Subsegment solver response was not valid JSON.")
        parse_warnings.append(
            "Subsegment solver output was not valid JSON. "
            "Raw response needs to be restructured."
        )

    # Validate hierarchy (auto-insert missing parents for sub-labels)
    fixed_labels, hierarchy_warnings = _validate_and_fix_hierarchy(sub_labels)
    for w in hierarchy_warnings:
        LOGGER.info("Hierarchy fix: %s", w)

    msg = {
        "role": "subsegment_solver",
        "iteration": state.get("iteration", 0),
        "content": reasoning,
        "proposed_labels": fixed_labels,
        "proposed_range": [proposed_start, proposed_end],
        "hierarchy_warnings": hierarchy_warnings,
    }
    messages = list(state.get("messages", []))
    messages.append(msg)

    return {
        "proposed_sub_start": proposed_start,
        "proposed_sub_end": proposed_end,
        "proposed_labels": fixed_labels,
        "proposed_reasoning": reasoning,
        "parse_warnings": parse_warnings,
        "raw_subsegment_solver_response": raw_response,
        "messages": messages,
    }


def _validate_solver_json_format(
    raw_subsegment: str,
    proposed_labels: List[str],
    parent_start: int,
    parent_end: int,
) -> List[str]:
    """Validate subsegment solver JSON output and return detailed format issues.

    Checks the raw response for structural correctness: parsability,
    required keys (start_index, end_index, proposed_labels, reasoning),
    value types, range bounds, and valid label IDs.
    """
    issues: list[str] = []

    if not raw_subsegment:
        return issues

    parsed = _parse_json_response(raw_subsegment)
    if parsed is None:
        issues.append(
            "Subsegment solver did not produce valid JSON. "
            "Expected a ```json``` code block containing an object with keys: "
            '"start_index", "end_index", "proposed_labels", "reasoning".'
        )
        return issues

    # Required keys
    for key in ("start_index", "end_index", "proposed_labels", "reasoning"):
        if key not in parsed:
            issues.append(
                f'Subsegment solver JSON is missing required key "{key}". '
                "The solver must include all four keys."
            )

    # Type checks for indices
    si = parsed.get("start_index")
    ei = parsed.get("end_index")
    if si is not None and not isinstance(si, (int, float)):
        issues.append(
            f'"start_index" must be an integer, got {type(si).__name__}.'
        )
    if ei is not None and not isinstance(ei, (int, float)):
        issues.append(
            f'"end_index" must be an integer, got {type(ei).__name__}.'
        )

    # Range checks
    if isinstance(si, (int, float)) and isinstance(ei, (int, float)):
        si_int, ei_int = int(si), int(ei)
        if si_int >= ei_int:
            issues.append(
                f"start_index ({si_int}) must be strictly less than "
                f"end_index ({ei_int})."
            )
        if si_int < parent_start:
            issues.append(
                f"start_index ({si_int}) is outside parent range — "
                f"must be >= {parent_start}."
            )
        if ei_int > parent_end:
            issues.append(
                f"end_index ({ei_int}) exceeds parent end ({parent_end}) — "
                f"must be <= {parent_end}."
            )

    # Label checks
    pl = parsed.get("proposed_labels")
    if pl is not None and not isinstance(pl, list):
        issues.append(
            f'"proposed_labels" must be a JSON array, '
            f"got {type(pl).__name__}."
        )
    elif isinstance(pl, list):
        bad_ids = [l for l in pl if not isinstance(l, str)]
        if bad_ids:
            issues.append(
                f'"proposed_labels" contains non-string elements: {bad_ids}. '
                "All label IDs must be strings."
            )
        unknown = [l for l in pl if isinstance(l, str) and l not in LABEL_MAPPING]
        if unknown:
            issues.append(
                f'"proposed_labels" contains unknown IDs: {unknown}.'
            )

    # Reasoning check
    rs = parsed.get("reasoning")
    if rs is not None and not isinstance(rs, str):
        issues.append(
            f'"reasoning" must be a string, got {type(rs).__name__}.'
        )

    return issues


def evaluator_node(state: AnnotationState) -> dict:
    """Evaluate the subsegment solver's proposed sub-segment.

    Validates the JSON format/range, then asks the VLM to review the
    proposed range and labels against the telemetry data. On accept the
    evaluator sets ``final_sub_start``, ``final_sub_end``, ``final_labels``,
    and ``final_reasoning``.
    """
    proposed_labels = state.get("proposed_labels", [])
    proposed_reasoning = state.get("proposed_reasoning", "")
    segment_data = state.get("segment_data", {})
    iteration = state.get("iteration", 0)
    stats_text = state.get("tool_statistics_text", "")
    graph_descriptions = state.get("tool_graph_descriptions", [])
    graph_image_bytes = state.get("tool_graph_images", [])
    parse_warnings = state.get("parse_warnings", [])
    raw_subsegment = state.get("raw_subsegment_solver_response", "")
    parent_start = state.get("parent_start", 0)
    parent_end = state.get("parent_end", 0)
    proposed_sub_start = state.get("proposed_sub_start")
    proposed_sub_end = state.get("proposed_sub_end")

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
            "The subsegment solver must output:\n"
            "```json\n"
            '{"start_index": <int>, "end_index": <int>, '
            '"proposed_labels": ["LABEL_ID", ...], '
            '"reasoning": "..."}\n'
            "```\n\n"
            f"Range must be within parent bounds [{parent_start}, {parent_end}].\n"
            "Instruct the solver to fix these issues in the next iteration."
        )
        LOGGER.warning("Evaluator auto-reject (format): %s", warning_text)
        msg = {
            "role": "evaluator",
            "iteration": iteration,
            "verdict": "reject",
            "content": feedback,
        }
        messages = list(state.get("messages", []))
        messages.append(msg)
        return {
            "evaluation": "reject",
            "evaluation_feedback": feedback,
            "parse_warnings": [],
            "messages": messages,
        }

    if not proposed_labels:
        feedback = (
            "No labels were proposed by the solver. The JSON output was "
            "malformed or contained no valid label IDs. "
            "The solver must propose at least one label."
        )
        msg = {
            "role": "evaluator",
            "iteration": iteration,
            "verdict": "reject",
            "content": feedback,
        }
        messages = list(state.get("messages", []))
        messages.append(msg)
        return {
            "evaluation": "reject",
            "evaluation_feedback": feedback,
            "parse_warnings": [],
            "messages": messages,
        }

    telemetry_summary = stats_text or _summarise_telemetry(segment_data)
    proposed_readable = [LABEL_MAPPING.get(l, l) for l in proposed_labels]

    # Run hierarchy validation for exclusive-with detection
    _, hierarchy_warnings = _validate_and_fix_hierarchy(proposed_labels)
    conflict_warnings = [w for w in hierarchy_warnings if "exclusive" in w.lower()]

    prompt_parts = [
        "You are a quality evaluator for racing telemetry sub-segment proposals.",
        "",
        "=== Telemetry Statistics ===",
        telemetry_summary,
        "",
    ]
    prompt_parts.extend(
        _build_graph_prompt_section(graph_descriptions, graph_image_bytes)
    )
    prompt_parts.extend([
        "=== Proposed Sub-Segment ===",
        f"Range: [{proposed_sub_start}, {proposed_sub_end}] "
        f"(length: {(proposed_sub_end or 0) - (proposed_sub_start or 0)})  "
        f"parent range: [{parent_start}, {parent_end}] "
        f"(length: {parent_end - parent_start})",
        f"Labels: {', '.join(proposed_readable)} (IDs: {json.dumps(proposed_labels)})",
        "",
        "=== Solver's Reasoning ===",
        proposed_reasoning,
        "",
    ])

    if conflict_warnings:
        prompt_parts.append("=== Exclusive-With Conflicts ===")
        for w in conflict_warnings:
            prompt_parts.append(f"- {w}")
        prompt_parts.append("")

    prompt_parts.extend([
        "=== Evaluation Criteria ===",
        "1. Does the proposed range capture a distinct telemetry pattern?",
        "2. Do the proposed labels match the observed behaviour in that range?",
        "3. Are there any labels missing, contradicting the data, or exclusive-with conflicts?",
        "4. Does the range [start_index, end_index] tightly capture the identified event, "
        "or is it too wide or too narrow? Suggest tighter or wider boundaries if needed.",
        "",
        "Provide your evaluation. If the proposal is wrong or incomplete, "
        "explain what should change.",
    ])

    prompt = "\n".join(prompt_parts)

    raw_response = _call_vlm(prompt, graph_image_bytes)
    if not raw_response:
        raw_response = "No VLM available. VERDICT: ACCEPT"

    feedback = raw_response
    verdict = _extract_verdict(raw_response)

    msg = {
        "role": "evaluator",
        "iteration": iteration,
        "verdict": verdict,
        "content": feedback,
    }
    messages = list(state.get("messages", []))
    messages.append(msg)

    result: dict = {
        "evaluation": verdict,
        "evaluation_feedback": feedback,
        "messages": messages,
    }

    if verdict == "accept":
        result["final_labels"] = proposed_labels
        result["final_reasoning"] = proposed_reasoning
        result["final_sub_start"] = proposed_sub_start
        result["final_sub_end"] = proposed_sub_end

    return result


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


def should_retry(state: AnnotationState) -> Literal["planner", "end"]:
    """Decide whether to loop back to the planner or finish."""
    if state.get("evaluation") == "accept":
        return "end"
    if state.get("iteration", 0) >= state.get("max_iterations", 3):
        # Max retries reached — accept the latest proposal as-is
        return "end"
    return "planner"


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


def build_annotation_graph() -> StateGraph:
    """Construct and compile the annotation graph.

    Flow: planner → tool_executor → subsegment_solver → evaluator
    On rejection the evaluator loops back to the planner.
    """
    graph = StateGraph(AnnotationState)

    graph.add_node("planner", planner_node)
    graph.add_node("tool_executor", tool_executor_node)
    graph.add_node("subsegment_solver", subsegment_solver_node)
    graph.add_node("evaluator", evaluator_node)

    graph.set_entry_point("planner")
    graph.add_edge("planner", "tool_executor")
    graph.add_edge("tool_executor", "subsegment_solver")
    graph.add_edge("subsegment_solver", "evaluator")
    graph.add_conditional_edges(
        "evaluator",
        should_retry,
        {
            "planner": "planner",
            "end": END,
        },
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
    context_size: int = 16384
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
    df,
    start_index: int,
    end_index: int,
    session_id: str,
    current_labels: List[str],
) -> dict:
    """Extract a telemetry summary dict from a DataFrame slice.

    Delegates to :func:`annotation_agent_tools.get_telemetry_statistics`.
    """
    from .annotation_agent_tools import get_telemetry_statistics
    return get_telemetry_statistics(df, start_index, end_index, session_id)


def run_annotation_pipeline(
    df,
    start_index: int,
    end_index: int,
    session_id: str,
    parent_main_labels: List[str],
    existing_children: Optional[List[dict]] = None,
    config: Optional[AnnotationPipelineConfig] = None,
    progress_callback=None,
) -> AnnotationResult:
    """Execute the planner → tool_executor → subsegment_solver → evaluator pipeline.

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
    segment_data = _prepare_segment_data(
        df, start_index, end_index, session_id, parent_main_labels,
    )

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
        return vlm_service.generate(
            prompt,
            images=images,
            max_tokens=config.max_new_tokens,
            temperature=config.temperature,
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
    graph = build_annotation_graph()

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
        "label_guidelines": MAIN_LABEL_GUIDELINES,
        "max_iterations": config.max_iterations,
    })

    # Stream through graph nodes for progress reporting
    final_state = dict(initial_state)
    for event in graph.stream(initial_state):
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
                elif node_name == "tool_executor":
                    n_stats = len(final_state.get("segment_data", {}).get("telemetry_summary", {}))
                    n_graphs = len(final_state.get("tool_graph_descriptions", []))
                    detail = f"Gathered {n_stats} stat columns + {n_graphs} graph(s)"
                elif node_name == "subsegment_solver":
                    labels = final_state.get("proposed_labels", [])
                    ss = final_state.get("proposed_sub_start", "?")
                    se = final_state.get("proposed_sub_end", "?")
                    detail = (
                        f"Range [{ss}, {se}], "
                        f"labels: {', '.join(LABEL_MAPPING.get(l, l) for l in labels)}"
                    )
                elif node_name == "evaluator":
                    detail = f"{final_state.get('evaluation', '').upper()}: " \
                             f"{final_state.get('evaluation_feedback', '')[:200]}"
                progress_callback(node_name, iteration, detail)

    # Clean up VLM/LLM holders
    with _llm_lock:
        _llm_holder["vlm"] = None
        _llm_holder["llm"] = None

    accepted = final_state.get("evaluation") == "accept"
    return AnnotationResult(
        final_labels=final_state.get("final_labels") or final_state.get("proposed_labels", []),
        final_reasoning=final_state.get("final_reasoning") or final_state.get("proposed_reasoning", ""),
        accepted=accepted,
        iterations=final_state.get("iteration", 0),
        messages=final_state.get("messages", []),
        graph_images=final_state.get("tool_graph_images", []),
        sub_start=final_state.get("final_sub_start") or final_state.get("proposed_sub_start"),
        sub_end=final_state.get("final_sub_end") or final_state.get("proposed_sub_end"),
    )
