"""
describe_graphs meta-Agent — overview + question-driven zoom + prose synth.

Uniform topology (inherited from the framework):

    planner (overview + zoom questions) ──► step_solver ──► synthesizer ──► evaluator
                                              (delegates       (single VLM      (no-op)
                                               ONCE to zoom     analysis call
                                               with the full    using overview
                                               question list)   images + zoom
                                                                answer prose)

Phase 1 — Planner renders a low-resolution overview of the requested graphs
at the parent range and asks the VLM, using the per-graph skill, which
precise readings cannot be cleanly resolved at this scale. Every such
reading is collected into a single ``questions`` list (each entry carries
a natural-language ``question``, a ``context`` cue, a sub-range, and a
``requested_graphs`` subset) and dispatched as ONE zoom step. If every
reading is resolvable from the overview alone, a ``skip=True`` sentinel
step is emitted and the synthesizer runs against the overview only.

Phase 2 — Executor delegates the single zoom step to ``zoom``. The zoom
worker handles every question in one invocation: it renders all
sub-ranges, asks the VLM to pick queries from
``PIPELINE_QUERY_DEFINITIONS`` for every question at once, runs each
query deterministically, and emits ONE ``step_solver.1.answer``
attachment carrying the combined prose answer covering every question.

Phase 3 — Synthesizer is the ONLY VLM analysis step. It receives the
overview images (for trend/shape language) plus the zoom answer
attachment (which already cites exact ilocs from deterministic queries),
and writes one prose paragraph per graph. It must cite ilocs verbatim
from the zoom answer and explicitly flag any unresolved readings — no
estimating from pixels. The evaluator suite runs inline here.
"""

from __future__ import annotations

import io
import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from app.models.graph_analysis_skill import get_graph_skill
from app.services.llm.agent_framework import Agent, AgentState, delegate_step
from app.services.llm.step_evaluator_agents import (
    AttachmentPool,
    EvalPipelineResult,
    PipelineAttachment,
    _eval_llm_holder,
    render_inputs_for_prompt,
    run_evaluator_suite,
    set_active_attachments,
    set_active_iteration,
    set_active_stage,
)

LOGGER = logging.getLogger(__name__)

DESCRIBE_GRAPHS_AGENT_NAME = "describe_graphs"

# Bounded zoom — caps locked at design time. Zoom-step count is uncapped:
# the planner emits as many sub-ranges as the skill's how_to_analyze procedure
# demands.
MIN_ZOOM_SPAN = 3           # min indices in a zoom sub-range
MAX_RECURSIVE_ZOOM_DEPTH = 1  # beyond this depth, planner falls back to terminal


def _call_vlm(prompt: str, graph_image_bytes: List[bytes]) -> str:
    vlm_fn = _eval_llm_holder.get("vlm")
    if not vlm_fn:
        return ""
    if graph_image_bytes:
        return vlm_fn(prompt, graph_image_bytes)
    return vlm_fn(prompt)


# ---------------------------------------------------------------------------
# Plan parsing — VLM emits a JSON object with zoom decision
# ---------------------------------------------------------------------------


def _parse_zoom_decision(
    text: str,
    parent_start: int,
    parent_end: int,
    available_graph_ids: List[str],
) -> Tuple[bool, List[Dict[str, Any]]]:
    """Extract (zoom_required, [valid zoom requests]) from the planner's response.

    Accepts a JSON object of the form::

        {"zoom": true,
         "ranges": [
           {"start": int, "end": int,
            "question": str,
            "context": str,
            "requested_graphs": [str, ...]   # optional, subset of parent's
           }, ...]}

    or ``{"zoom": false}``. Invalid entries (bad types, outside parent range,
    span < MIN_ZOOM_SPAN, missing question) are silently dropped. Returns
    ``(False, [])`` when no parseable decision is found or no entry has a
    usable question.
    """
    def _try_loads(s: str) -> Optional[Any]:
        try:
            return json.loads(s)
        except (json.JSONDecodeError, ValueError):
            return None

    parsed = None
    json_match = re.search(r"```json\s*(\{[\s\S]*?\})\s*```", text, re.DOTALL)
    if json_match:
        parsed = _try_loads(json_match.group(1))
    if parsed is None:
        brace_match = re.search(r"\{[\s\S]*\}", text)
        if brace_match:
            parsed = _try_loads(brace_match.group())
    if not isinstance(parsed, dict):
        return False, []

    if not parsed.get("zoom"):
        return False, []

    raw_ranges = parsed.get("ranges", [])
    if not isinstance(raw_ranges, list):
        return False, []

    available_set = set(available_graph_ids)

    valid: List[Dict[str, Any]] = []
    for entry in raw_ranges:
        if not isinstance(entry, dict):
            continue
        s = entry.get("start")
        e = entry.get("end")
        question = entry.get("question", "")
        context = entry.get("context", "")
        if not isinstance(s, int) or not isinstance(e, int):
            continue
        if e - s < MIN_ZOOM_SPAN:
            continue
        if s < parent_start or e > parent_end:
            continue
        if not isinstance(question, str) or not question.strip():
            continue
        req_graphs_raw = entry.get("requested_graphs")
        if isinstance(req_graphs_raw, list):
            req_graphs = [g for g in req_graphs_raw if g in available_set]
        else:
            req_graphs = list(available_graph_ids)
        if not req_graphs:
            req_graphs = list(available_graph_ids)
        valid.append({
            "start": s, "end": e,
            "question": question.strip(),
            "context": context.strip() if isinstance(context, str) else "",
            "requested_graphs": req_graphs,
        })

    return bool(valid), valid


# ---------------------------------------------------------------------------
# Planner — two-phase: render overview, ask VLM whether to zoom
# ---------------------------------------------------------------------------


def _planner(state: AgentState) -> Dict[str, Any]:
    """Decide between a single-terminal plan or a multi-zoom plan.

    At depth > MAX_RECURSIVE_ZOOM_DEPTH the planner short-circuits to a
    single terminal step regardless of VLM input — this bounds zoom-of-zoom
    recursion. Otherwise it renders an overview, asks the VLM for a zoom
    decision, and emits plan_steps accordingly.
    """
    from app.services.llm.annotation_agent_tools import (
        build_graph,
        render_graph_builds,
    )

    parent_start = state.get("parent_start", 0)
    parent_end = state.get("parent_end", 0)
    df = state.get("df_ref")
    goal = state.get("goal") or "Analyse telemetry graphs."
    requested_graphs: List[str] = state.get("requested_graphs", []) or []
    requested_tools: List[str] = state.get("tools", []) or []
    depth = state.get("depth", 0)

    # Constrained per-graph tables (parent agent's responsibility). Each
    # entry is the full-df-length DataFrame of just the data that graph
    # draws / consumes. Forwarded into every zoom step so children can
    # only see (render + query) the data the picked graphs actually use.
    graph_builds: Dict[str, Any] = {}
    if df is not None:
        for gid in requested_graphs:
            table = build_graph(gid, df)
            if table is not None and not table.empty:
                graph_builds[gid] = table

    messages = list(state.get("messages", []))

    def _no_zoom_plan(reason: str) -> Dict[str, Any]:
        """Emit a single skip-sentinel step — describe_graphs._executor no-ops on it.

        The framework requires at least one plan step (it'll substitute a
        trivial fallback otherwise), so we emit a marker the executor
        recognises and skips, rather than waste a VLM call on a question-less
        zoom.
        """
        step = {
            "step_id": 1,
            "agent": "zoom",
            "skip": True,
            "description": f"No zoom needed — {reason}",
            "requested_graphs": list(requested_graphs),
            "tools": list(requested_tools),
        }
        plan_attachment = PipelineAttachment(
            name="planner.plan",
            kind="text",
            label="describe_graphs Planner Plan",
            content=f"No zoom — {reason}",
        )
        messages.append({
            "role": "planner",
            "content": f"describe_graphs plan: no-zoom ({reason})",
        })
        return {
            "plan": f"no-zoom: {reason}",
            "plan_steps": [step],
            "current_step_index": 0,
            "attachment_pool": {plan_attachment.name: plan_attachment},
            "messages": messages,
        }

    # No graphs requested — nothing to zoom over, go straight to single render.
    if not requested_graphs:
        return _no_zoom_plan("no graphs requested")

    # Too deep already — don't recurse further.
    if depth > MAX_RECURSIVE_ZOOM_DEPTH:
        return _no_zoom_plan(f"recursion depth {depth} exceeds zoom cap")

    # Render overview for the VLM's zoom decision — straight from the
    # constrained per-graph tables, not raw df.
    overview_images: List[bytes] = []
    overview_descriptions: List[str] = []
    for img, desc in render_graph_builds(
        graph_builds, parent_start, parent_end,
    ):
        overview_descriptions.append(desc)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        overview_images.append(buf.getvalue())

    if not overview_images:
        return _no_zoom_plan("no graph images could be rendered")

    # Built before the VLM call so the UI tracker (set_active_attachments)
    # shows what the planner is feeding to the VLM.
    overview_imgs_attachment = PipelineAttachment(
        name="describe_graphs.overview_graph_images",
        kind="image_set",
        label=f"Overview [{parent_start}, {parent_end}] — Graph Images",
        content=overview_images,
    )
    overview_descs_attachment = PipelineAttachment(
        name="describe_graphs.overview_graph_descriptions",
        kind="structured",
        content_schema="graph_descriptions",
        label=f"Overview [{parent_start}, {parent_end}] — Graph Descriptions",
        content=overview_descriptions,
    )

    skill_prompt = get_graph_skill().build_graph_prompt(requested_graphs)
    graphs_list = ", ".join(f"`{g}`" for g in requested_graphs)

    decision_prompt = (
        "You are planning a telemetry-graph description task. Your job here "
        "is NOT to describe the graphs — it is to decide which precise "
        "readings the downstream description needs that you cannot reliably "
        "make from this overview, and to dispatch each one to a zoom worker "
        "as a natural-language question over a sub-range.\n\n"
        f"**Goal:** {goal}\n"
        f"**Full range:** [{parent_start}, {parent_end}] "
        f"(length {parent_end - parent_start})\n"
        f"**Graphs rendered:** {', '.join(overview_descriptions)}\n\n"
        f"{skill_prompt}\n\n"
        "**Follow the per-graph `how_to_analyze` procedure above as written** "
        "— it is the authoritative spec for what this analysis must produce. "
        "Walk through each step against the overview images and **use zoom to "
        "actually nail down the evidence**: the overview is a sketch, the zoom "
        "is where you commit to a number. For every precise reading the "
        "procedure calls for (exact iloc, threshold crossing, extremum, slope "
        "window, etc.), dispatch ONE zoom step to pin it down — do not guess "
        "from the overview. Each zoom step carries:\n"
        "- a sub-range tight around the feature,\n"
        "- a `question` in plain English stating EXACTLY what to find (the "
        "zoom worker will pick a query + parameters; you do NOT need to name "
        "a query or column),\n"
        "- a `context` sentence summarising the overview cue that motivates "
        "the question (e.g. 'overview shows brake decaying from ~30 to ~120'),\n"
        "- an optional `requested_graphs` subset (defaults to all parent "
        f"graphs: {graphs_list}).\n\n"
        "If every reading the procedure needs is already cleanly resolvable "
        "from the overview, return `{\"zoom\": false}`.\n\n"
        "**Output (JSON only — no prose, no comments):**\n"
        "```json\n"
        "{\n"
        '  "zoom": true,\n'
        '  "ranges": [\n'
        '    {\n'
        f'      "start": <int in [{parent_start},{parent_end}]>,\n'
        f'      "end":   <int in [{parent_start},{parent_end}]>,\n'
        '      "question": "<what exact reading to find, in plain English>",\n'
        '      "context":  "<overview cue motivating the question>",\n'
        '      "requested_graphs": ["<subset of parent graphs>"]\n'
        '    }\n'
        "  ]\n"
        "}\n"
        "```\n"
        "Or, if no zoom is needed:\n"
        "```json\n"
        '{"zoom": false}\n'
        "```\n"
        f"Each zoom span must be >= {MIN_ZOOM_SPAN} indices and lie within "
        f"[{parent_start}, {parent_end}]."
    )

    set_active_stage(DESCRIBE_GRAPHS_AGENT_NAME, "planner", graphs=requested_graphs)
    set_active_attachments([overview_imgs_attachment, overview_descs_attachment])
    raw_decision = _call_vlm(decision_prompt, overview_images)
    if not raw_decision:
        return _no_zoom_plan("VLM unavailable for zoom decision")

    zoom_required, zoom_ranges = _parse_zoom_decision(
        raw_decision, parent_start, parent_end, requested_graphs,
    )

    if not zoom_required or not zoom_ranges:
        return _no_zoom_plan("VLM judged no zoom needed")

    # Collapse every VLM-emitted zoom range into ONE zoom step carrying the
    # full question list. The zoom worker handles all questions in a single
    # invocation — picks queries for every question with one VLM call,
    # runs them deterministically, and emits one combined answer.
    questions: List[Dict[str, Any]] = []
    for r in zoom_ranges:
        questions.append({
            "question": r["question"],
            "context": r["context"],
            "start": r["start"],
            "end": r["end"],
            "requested_graphs": r["requested_graphs"],
        })

    plan_steps: List[Dict[str, Any]] = [{
        "step_id": 1,
        "agent": "zoom",
        "description": f"Answer {len(questions)} zoom question(s)",
        "questions": questions,
        "parent_start": parent_start,
        "parent_end": parent_end,
        "requested_graphs": list(requested_graphs),
        "tools": list(requested_tools),
        "graph_builds": graph_builds,
    }]

    plan_attachment = PipelineAttachment(
        name="planner.plan",
        kind="text",
        label="describe_graphs Planner Plan",
        content=raw_decision,
    )
    messages.append({
        "role": "planner",
        "content": f"describe_graphs plan: 1 zoom step ({len(questions)} question(s))",
    })

    return {
        "plan": raw_decision,
        "plan_steps": plan_steps,
        "current_step_index": 0,
        "attachment_pool": {
            plan_attachment.name: plan_attachment,
            overview_imgs_attachment.name: overview_imgs_attachment,
            overview_descs_attachment.name: overview_descs_attachment,
        },
        "messages": messages,
    }


# ---------------------------------------------------------------------------
# Executor — delegate to zoom (rendering only); analysis happens in synthesizer
# ---------------------------------------------------------------------------


def _executor(state: AgentState, step: Dict[str, Any], registry) -> Dict[str, Any]:
    """Delegate the step to ``zoom`` — or no-op on a skip-sentinel.

    A ``skip=True`` step is the no-zoom sentinel emitted by the planner: the
    synthesizer will fall back to the overview alone, so we return empty.
    Otherwise we delegate to ``zoom``, which emits a single
    ``step_solver.{step_id}.answer`` attachment. The zoom worker already
    labels its answer with the parent's step_id, so no relabelling is needed.
    """
    if step.get("skip"):
        return {}
    return delegate_step(state, step, registry)


# ---------------------------------------------------------------------------
# Synthesizer — merges multiple zoom observations into one description
# ---------------------------------------------------------------------------


def _synthesizer(state: AgentState) -> Dict[str, Any]:
    """Write the unified prose description, citing zoom answers verbatim.

    Inputs:
      * the planner's overview images (for trend / shape language only),
      * every ``step_solver.*.answer`` attachment (the structured zoom
        results — already contain exact ilocs from deterministic queries),
      * any tool attachments emitted by zoom (currently none).

    The VLM does NOT re-read indices from pixels — when a precise iloc is
    needed it cites the answer table verbatim. This is the only VLM
    analysis call in the describe_graphs subgraph.
    """
    messages = list(state.get("messages", []))
    parent_start = state.get("parent_start", 0)
    parent_end = state.get("parent_end", 0)
    goal = state.get("goal") or "Describe telemetry graphs."
    requested_graphs: List[str] = state.get("requested_graphs", []) or []

    pool: AttachmentPool = state.get("attachment_pool", {})

    overview_images: List[bytes] = []
    aggregated_descriptions: List[str] = []
    image_attachments: List[PipelineAttachment] = []
    desc_attachments: List[PipelineAttachment] = []
    answer_attachments: List[PipelineAttachment] = []
    tool_attachments: List[PipelineAttachment] = []

    overview_imgs_att = pool.get("describe_graphs.overview_graph_images")
    overview_descs_att = pool.get("describe_graphs.overview_graph_descriptions")
    if overview_imgs_att and isinstance(overview_imgs_att.content, list) and overview_imgs_att.content:
        image_attachments.append(overview_imgs_att)
        overview_images.extend(overview_imgs_att.content)
    if overview_descs_att and isinstance(overview_descs_att.content, list):
        desc_attachments.append(overview_descs_att)
        for d in overview_descs_att.content:
            if d not in aggregated_descriptions:
                aggregated_descriptions.append(d)

    # Collect every zoom answer in step-id order.
    answer_names = sorted(
        n for n in pool.keys()
        if n.startswith("step_solver.") and n.endswith(".answer")
    )
    for name in answer_names:
        answer_attachments.append(pool[name])
        # Pull in any sibling attachments from the same step (tools, etc.)
        sid_prefix = name[: -len(".answer")] + "."
        for sib_name, sib_att in pool.items():
            if sib_name.startswith(sid_prefix) and sib_name != name:
                tool_attachments.append(sib_att)

    prompt_parts = [
        "You are a telemetry graph describer. Your ONLY job is to produce a "
        "detailed, precise description of the data and graphs provided. "
        "Write in flowing prose paragraphs — do NOT use numbered lists, "
        "bullet points, or step-by-step formatting. Do NOT diagnose problems, "
        "assign labels, or suggest what the observations mean. Downstream "
        "nodes will interpret your description.",
        "",
        f"**Analysis Goal:** {goal}",
        f"**Full range:** [{parent_start}, {parent_end}] "
        f"(length {parent_end - parent_start})",
        "",
    ]
    if aggregated_descriptions:
        prompt_parts.append(
            f"**Overview [{parent_start}, {parent_end}] (full range):** "
            + "; ".join(aggregated_descriptions)
        )
        prompt_parts.append("")
    if answer_attachments:
        prompt_parts.append("**Zoom answers (exact ilocs from deterministic queries):**")
        prompt_parts.append(render_inputs_for_prompt(answer_attachments))
        prompt_parts.append("")
    if tool_attachments:
        prompt_parts.append(render_inputs_for_prompt(tool_attachments))
        prompt_parts.append("")
    if requested_graphs:
        skill_prompt = get_graph_skill().build_graph_prompt(requested_graphs)
        if skill_prompt:
            prompt_parts.append(skill_prompt)
            prompt_parts.append("")
    prompt_parts.append(
        "**Your task:** Write one cohesive prose paragraph per graph, "
        "following the per-graph guidance above. Use the overview image(s) "
        "for trend, shape, and duration language. When a precise iloc / "
        "value is needed, cite it VERBATIM from the zoom-answer table above "
        "— do NOT estimate indices from the image. If a needed reading is "
        "missing from the answers (or marked unresolved), say so explicitly "
        "in the prose rather than guessing. Anchor observations to specific "
        f"indices in [{parent_start}, {parent_end}]."
    )
    prompt = "\n".join(prompt_parts)

    set_active_stage(DESCRIBE_GRAPHS_AGENT_NAME, "synthesizer", graphs=requested_graphs)
    set_active_iteration(1, 1)
    set_active_attachments(
        image_attachments + desc_attachments + answer_attachments + tool_attachments
    )

    if not overview_images:
        # Defensive fallback — overview rendering failed upstream.
        att = PipelineAttachment(
            name="describe_graphs.observations",
            kind="structured",
            content_schema="step_observation",
            label="describe_graphs — Observations",
            content={
                "requested_graphs": requested_graphs,
                "graph_descriptions": aggregated_descriptions,
                "graph_observations": "No graph images were rendered.",
            },
        )
        return {
            "evaluation": "fail",
            "attachment_pool": {att.name: att},
            "messages": messages,
        }

    raw_response = _call_vlm(prompt, overview_images)
    if not raw_response:
        raise RuntimeError(
            f"describe_graphs synthesizer: VLM returned empty response "
            f"(range=[{parent_start}, {parent_end}])"
        )

    suite_result: EvalPipelineResult = run_evaluator_suite(
        parent_prompt=prompt,
        parent_output_text=raw_response,
        parent_inputs=image_attachments + answer_attachments + tool_attachments,
        step_name="describe_graphs",
        parent_start=parent_start,
        parent_end=parent_end,
    )
    evaluated_response = suite_result.final_result

    obs_attachment = PipelineAttachment(
        name="describe_graphs.observations",
        kind="structured",
        content_schema="step_observation",
        label="describe_graphs — Observations",
        content={
            "requested_graphs": requested_graphs,
            "graph_descriptions": aggregated_descriptions,
            "graph_observations": evaluated_response,
        },
    )

    messages.append({"role": "assistant", "content": evaluated_response})

    return {
        "evaluation": suite_result.final_verdict,
        "attachment_pool": {obs_attachment.name: obs_attachment},
        "messages": messages,
    }


# ---------------------------------------------------------------------------
# Evaluator — no-op pass-through (synthesizer / terminal ran suite inline)
# ---------------------------------------------------------------------------


def _evaluator(state: AgentState) -> Dict[str, Any]:
    return {"evaluation": state.get("evaluation", "pass")}


class DescribeGraphs(Agent):
    """Meta-agent: planner emits zoom questions; synthesizer writes prose.

    Planner renders an overview and emits a zoom step (with question +
    sub-range) for each reading the overview can't resolve. The executor
    delegates each to ``zoom``, which picks a query + parameters and runs
    deterministic math for the exact iloc. The synthesizer makes the single
    VLM analysis call against the overview images + zoom-answer table and
    emits the unified prose description.
    """

    name = DESCRIBE_GRAPHS_AGENT_NAME
    consumes = ["init.parent_segment"]
    produces = ["observations"]
    delegates_to = ["zoom"]

    def planner(self, state: AgentState) -> Dict[str, Any]:
        return _planner(state)

    def executor(self, state: AgentState, step, registry) -> Dict[str, Any]:
        return _executor(state, step, registry)

    def synthesizer(self, state: AgentState) -> Dict[str, Any]:
        return _synthesizer(state)

    def evaluator(self, state: AgentState) -> Dict[str, Any]:
        return _evaluator(state)


DESCRIBE_GRAPHS_SPEC = DescribeGraphs.register()
