"""
describe_graphs meta-Agent — planner owns vision, synthesizer is text-only.

Uniform topology (inherited from the framework):

    planner (VLM: overview + zoom questions) ──► step_solver ──► synthesizer ──► evaluator
                                                    (delegates       (text-only       (no-op)
                                                     ONCE to zoom     LLM analysis
                                                     with the full    using zoom
                                                     question list)   answer prose)

Phase 1 — Planner renders a low-resolution overview of the requested
graphs at the parent range and is the ONLY stage with image access. The
VLM walks the per-graph ``how_to_analyze`` procedure against the overview
and emits zoom questions covering **every** observation the procedure
needs — trend shape, slopes, extrema, threshold crossings, transients,
exact ilocs. Each question carries a natural-language ``question``, a
``context`` cue, a sub-range, and a ``requested_graphs`` subset; all
questions are collected into a single zoom step. If no graphs are
requested or rendering fails, a ``skip=True`` sentinel is emitted instead.

Phase 2 — Executor delegates the single zoom step to ``zoom``. The zoom
worker handles every question in one invocation: it renders all
sub-ranges, asks the VLM to pick queries from
``PIPELINE_QUERY_DEFINITIONS`` for every question at once, runs each
query deterministically, and emits ONE ``step_solver.1.answer``
attachment carrying the combined prose answer covering every question.

Phase 3 — Synthesizer is a TEXT-ONLY LLM analysis step. It receives the
overview text descriptions (column/unit context) and the zoom answer
prose, and writes one prose paragraph per graph. It has no image access:
every observation must be backed by a zoom answer, with ilocs cited
verbatim, and unresolved readings flagged explicitly. The evaluator suite
runs inline here.
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


def _call_llm(prompt: str) -> str:
    llm_fn = _eval_llm_holder.get("llm")
    if not llm_fn:
        return ""
    return llm_fn(prompt)


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
        get_pipeline_tool,
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

    # Pre-compute tools — invoke each requested PIPELINE_TOOL over the parent
    # range and stash the resulting attachment in the pool under a
    # ``describe_graphs.pipeline_tool.<id>`` name. The synthesizer harvests
    # this prefix so the tool output reaches the final prompt.
    pipeline_tool_attachments: Dict[str, "PipelineAttachment"] = {}
    if df is not None and requested_tools:
        for tid in requested_tools:
            tool_def = get_pipeline_tool(tid)
            if tool_def is None:
                LOGGER.warning("planner requested unknown pipeline tool: %s", tid)
                continue
            try:
                att = tool_def["callable"](df, parent_start, parent_end)
            except Exception:
                LOGGER.exception("pipeline tool '%s' raised", tid)
                continue
            if att is not None:
                pool_name = f"describe_graphs.pipeline_tool.{tid}"
                pipeline_tool_attachments[pool_name] = PipelineAttachment(
                    name=pool_name,
                    kind=att.kind,
                    label=att.label,
                    content=att.content,
                )

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
        pool = {plan_attachment.name: plan_attachment}
        pool.update(pipeline_tool_attachments)
        return {
            "plan": f"no-zoom: {reason}",
            "plan_steps": [step],
            "current_step_index": 0,
            "attachment_pool": pool,
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
        "You are the **only** stage that sees the overview images. The "
        "downstream synthesizer is text-only — it will write the prose "
        "description from your zoom answers alone. You must dispatch zoom "
        "questions that cover **every** observation the per-graph "
        "`how_to_analyze` procedure needs. Anything you don't capture here "
        "is lost.\n\n"
        f"**Goal:** {goal}\n"
        f"**Full range:** [{parent_start}, {parent_end}] "
        f"(length {parent_end - parent_start})\n"
        f"**Graphs rendered:** {', '.join(overview_descriptions)}\n\n"
        f"{skill_prompt}\n\n"
        "**Follow the per-graph `how_to_analyze` procedure above as written** "
        "— it is the authoritative spec for what this analysis must produce. "
        "Walk through every step against the overview images, and for each "
        "observation the procedure calls for, emit a zoom question. The zoom "
        "worker runs deterministic queries (slopes, extrema, dip detection, "
        "threshold crossings, point reads) so trend shape, peak/valley "
        "locations, transients, and exact ilocs all come through zoom — not "
        "through pixel reading downstream. Each zoom step carries:\n"
        "- a sub-range tight around the feature,\n"
        "- a `question` in plain English stating EXACTLY what to find (the "
        "zoom worker picks a query + parameters; you do NOT name a query or "
        "column),\n"
        "- a `context` field formatted as `<glossary_term> — <overview cue>`, "
        "where `<glossary_term>` is exactly one key from THIS question's "
        "graph's `glossary:` block above — whatever shape that glossary "
        "uses (an event onset, a segment class, a phase marker, a shape "
        "descriptor, a modifier — whichever the question resolves). Do "
        "not copy terms from other graphs' glossaries. The cue summarises "
        "the overview signal motivating the question. Examples across "
        "different graph types: 'release onset — overview shows trace "
        "decaying around idx 95'; 'apex (phase 1) — overview shows "
        "trajectory pinching tight at idx 120'; 'time gain — overview "
        "shows delta dropping by ~0.3 between idx 40 and 60',\n"
        "- an optional `requested_graphs` subset (defaults to all parent "
        f"graphs: {graphs_list}).\n\n"
        "**You must return at least one zoom range.** The synthesizer has no "
        "image access; `zoom: false` leaves it with nothing to describe.\n\n"
        "**Output (JSON only — no prose, no comments):**\n"
        "```json\n"
        "{\n"
        '  "zoom": true,\n'
        '  "ranges": [\n'
        '    {\n'
        f'      "start": <int in [{parent_start},{parent_end}]>,\n'
        f'      "end":   <int in [{parent_start},{parent_end}]>,\n'
        '      "question": "<what observation to capture, in plain English>",\n'
        '      "context":  "<glossary_term> — <overview cue>",\n'
        '      "requested_graphs": ["<subset of parent graphs>"]\n'
        '    }\n'
        "  ]\n"
        "}\n"
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

    pool = {
        plan_attachment.name: plan_attachment,
        overview_imgs_attachment.name: overview_imgs_attachment,
        overview_descs_attachment.name: overview_descs_attachment,
    }
    pool.update(pipeline_tool_attachments)
    return {
        "plan": raw_decision,
        "plan_steps": plan_steps,
        "current_step_index": 0,
        "attachment_pool": pool,
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
    """Write the unified prose description from zoom-answer text alone.

    Inputs:
      * the planner's overview text descriptions (column / unit context),
      * every ``step_solver.*.answer`` attachment (zoom prose — exact ilocs
        from deterministic queries),
      * any sibling tool attachments from the same step.

    Text-only LLM call. The synthesizer has NO image access — the planner
    is the only stage that sees pixels. Every observation cited here must
    be backed by a zoom answer.
    """
    messages = list(state.get("messages", []))
    parent_start = state.get("parent_start", 0)
    parent_end = state.get("parent_end", 0)
    goal = state.get("goal") or "Describe telemetry graphs."
    requested_graphs: List[str] = state.get("requested_graphs", []) or []

    pool: AttachmentPool = state.get("attachment_pool", {})

    aggregated_descriptions: List[str] = []
    desc_attachments: List[PipelineAttachment] = []
    answer_attachments: List[PipelineAttachment] = []
    tool_attachments: List[PipelineAttachment] = []

    overview_descs_att = pool.get("describe_graphs.overview_graph_descriptions")
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

    # Pre-computed pipeline tools (planner-level) — invoked once over the
    # parent range and stored under ``describe_graphs.pipeline_tool.*``.
    for name in sorted(pool.keys()):
        if name.startswith("describe_graphs.pipeline_tool."):
            tool_attachments.append(pool[name])

    set_active_stage(DESCRIBE_GRAPHS_AGENT_NAME, "synthesizer", graphs=requested_graphs)
    set_active_iteration(1, 1)
    set_active_attachments(desc_attachments + answer_attachments + tool_attachments)

    if not answer_attachments:
        # No zoom data was gathered — the planner either skipped or failed.
        # Without images and without zoom answers there is nothing to describe.
        att = PipelineAttachment(
            name="describe_graphs.observations",
            kind="structured",
            content_schema="step_observation",
            label="describe_graphs — Observations",
            content={
                "requested_graphs": requested_graphs,
                "graph_descriptions": aggregated_descriptions,
                "graph_observations": "No zoom data was gathered; no observations available.",
            },
        )
        return {
            "evaluation": "fail",
            "attachment_pool": {att.name: att},
            "messages": messages,
        }

    prompt_parts = [
        "You are a telemetry graph describer. For each requested graph, "
        "follow its `how_to_analyze` procedure below — that is the "
        "authoritative spec for what to produce. Classify with the glossary; "
        "compose with the `sentence_format_guide`. Only use glossary terms "
        "defined in the skill below; do not invent terms not in the "
        "glossary. Write in flowing prose paragraphs — no numbered lists "
        "or bullets.",
        "",
        f"**Analysis Goal:** {goal}",
        f"**Full range:** [{parent_start}, {parent_end}] "
        f"(length {parent_end - parent_start})",
        "",
    ]
    if requested_graphs:
        skill_prompt = get_graph_skill().build_graph_prompt(requested_graphs)
        if skill_prompt:
            prompt_parts.append(skill_prompt)
            prompt_parts.append("")
    if aggregated_descriptions:
        prompt_parts.append(
            f"**Overview [{parent_start}, {parent_end}] (full range):** "
            + "; ".join(aggregated_descriptions)
        )
        prompt_parts.append("")
    prompt_parts.append(
        "**Zoom answers (exact ilocs from deterministic queries — your "
        "numeric ground truth):**"
    )
    prompt_parts.append(render_inputs_for_prompt(answer_attachments))
    prompt_parts.append("")
    if tool_attachments:
        prompt_parts.append(render_inputs_for_prompt(tool_attachments))
        prompt_parts.append("")
    prompt_parts.append(
        "**Your task:** Emit one paragraph per graph in the order "
        "requested, each following THAT graph's `sentence_format_guide`. "
        "You have NO image access — every numeric claim must be backed by "
        "a zoom answer, with ilocs and values cited VERBATIM. Each zoom "
        "answer's `context:` phrase names the glossary term it resolves — "
        "use that term in your paragraph. If the graph's glossary defines "
        "an overall classification (segment class, shape category, phase "
        "type — whatever that glossary uses), apply it as the graph's "
        "`sentence_format_guide` directs; if the glossary has no such "
        "classification, omit it. Only use terms from THIS graph's "
        "glossary in its paragraph — do not borrow vocabulary from other "
        "graphs. If a needed reading is unresolved, say so explicitly "
        "rather than guessing. Anchor every event to indices in "
        f"[{parent_start}, {parent_end}]. After the per-graph paragraphs, "
        "if any cross-graph guidelines above apply to the requested "
        "combination, add a final cross-graph paragraph using those "
        "guidelines."
    )
    prompt = "\n".join(prompt_parts)

    raw_response = _call_llm(prompt)
    if not raw_response:
        raise RuntimeError(
            f"describe_graphs synthesizer: LLM returned empty response "
            f"(range=[{parent_start}, {parent_end}])"
        )

    suite_result: EvalPipelineResult = run_evaluator_suite(
        parent_prompt=prompt,
        parent_output_text=raw_response,
        parent_inputs=answer_attachments + tool_attachments,
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
    """Meta-agent: planner owns vision, synthesizer is text-only.

    Planner renders an overview and (with VLM image access) emits a zoom
    step covering every observation the per-graph procedure requires. The
    executor delegates to ``zoom``, which picks queries + parameters and
    runs deterministic math for each question. The synthesizer makes a
    text-only LLM call against the zoom-answer prose + skill text and
    emits the unified description — it has no image access, so every
    observation it cites is backed by a deterministic zoom answer.
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
