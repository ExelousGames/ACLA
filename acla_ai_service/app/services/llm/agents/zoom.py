"""
zoom — Agent that answers ONE planner question over a sub-range.

Topology (standard planner → executor(loop) → synthesizer):

    planner     — renders the zoom graph(s), calls the VLM once to classify
                  the question and pick ONE OR MORE queries from
                  ``PIPELINE_QUERY_DEFINITIONS``. Emits N plan steps, each
                  carrying ``{query, params, why}``. If the VLM declines
                  (or there's no question / no graph), emits a single
                  ``query=None`` step so the synthesizer records the
                  unresolved answer.

    executor    — runs the ONE planned query per step on the DataFrame
                  (deterministic math) and emits ``partial.{step_id}`` — an
                  internal attachment, NOT in ``produces``. The framework
                  accumulates partials across executor iterations via the
                  attachment-pool reducer.

    synthesizer — collects every ``partial.*`` from the pool, makes ONE
                  text-LLM call to convert the structured query results into
                  a short prose paragraph that directly answers the question
                  (citing ilocs / values verbatim from the deterministic
                  queries), and emits ONE ``answer`` attachment (kind=text)
                  carrying that prose. ``delegate_step`` renamespaces it to
                  ``step_solver.{parent_step_id}.answer`` for the
                  describe_graphs synthesizer to consume directly.

Inputs from the parent (forwarded into sub_state by ``delegate_step`` —
the parent step's fields land directly in ``state``):

    state["question"]          required — natural-language question
    state["context"]           optional — overview cue motivating it
    state["requested_graphs"]  list of graph ids to render
    state["parent_start"/_end] zoom sub-range bounds
    state["graph_builds"]      required — {graph_id: DataFrame} built by
                               the parent agent. Each table holds only the
                               data that graph uses (drawn series +
                               renderer-internal columns). Zoom renders /
                               queries against these tables only — raw df
                               is not available.

Output (single attachment, leaf name ``answer``, kind=text):
    A prose paragraph answering the question, with ilocs / values cited
    verbatim from the structured query partials. The structured partials
    themselves remain available in the attachment pool as
    ``partial.{step_id}`` for UI inspection but are NOT propagated to the
    parent — the parent consumes only the prose.
"""

from __future__ import annotations

import io
import json
import logging
import re
from typing import Any, Dict, List, Optional

from app.services.llm.agent_framework import Agent, AgentState
from app.services.llm.step_evaluator_agents import (
    PipelineAttachment,
    _eval_llm_holder,
    emit_step_event,
    set_active_attachments,
    set_active_stage,
)

LOGGER = logging.getLogger(__name__)

ZOOM_AGENT_NAME = "zoom"

# Cap on the number of queries the planner may pick per zoom invocation —
# prevents the VLM from going wild with redundant readings.
MAX_QUERIES_PER_ZOOM = 4


def _union_columns(graph_builds: Dict[str, Any]) -> List[str]:
    """Stable de-duplicated union of every per-graph table's columns."""
    seen: Dict[str, None] = {}
    for table in graph_builds.values():
        for col in getattr(table, "columns", []):
            if col not in seen:
                seen[col] = None
    return list(seen.keys())


def _build_query_table(graph_builds: Dict[str, Any]):
    """Concat the per-graph tables column-wise, dropping duplicates, so the
    query layer sees one DataFrame containing every queryable column."""
    if not graph_builds:
        return None
    import pandas as pd
    tables = list(graph_builds.values())
    if len(tables) == 1:
        return tables[0]
    combined = pd.concat(tables, axis=1)
    return combined.loc[:, ~combined.columns.duplicated()]


def _call_vlm(prompt: str, image_bytes: List[bytes]) -> str:
    vlm_fn = _eval_llm_holder.get("vlm")
    if not vlm_fn:
        return ""
    if image_bytes:
        return vlm_fn(prompt, image_bytes)
    return vlm_fn(prompt)


def _call_llm(prompt: str) -> str:
    llm_fn = _eval_llm_holder.get("llm")
    if not llm_fn:
        return ""
    return llm_fn(prompt)


def _parse_query_picks(text: str) -> Optional[List[Dict[str, Any]]]:
    """Extract the list of query picks from the VLM's response.

    The VLM must respond with the multi-pick shape regardless of how many
    queries it picks::

        {"queries": [{"query": "...", "params": {...}, "why": "..."}, ...]}

    Returns the list (possibly empty) or ``None`` when no JSON object is
    found or the ``queries`` key is missing.
    """
    def _try_loads(s: str) -> Optional[Any]:
        try:
            return json.loads(s)
        except (json.JSONDecodeError, ValueError):
            return None

    parsed = None
    fenced = re.search(r"```json\s*(\{[\s\S]*?\})\s*```", text, re.DOTALL)
    if fenced:
        parsed = _try_loads(fenced.group(1))
    if parsed is None:
        brace = re.search(r"\{[\s\S]*\}", text)
        if brace:
            parsed = _try_loads(brace.group())
    if not isinstance(parsed, dict):
        return None
    if not isinstance(parsed.get("queries"), list):
        return None
    return [p for p in parsed["queries"] if isinstance(p, dict)]


def _unresolved_plan(
    question: str,
    context: str,
    zoom_start: int,
    zoom_end: int,
    why: str,
    error: Optional[str] = None,
) -> Dict[str, Any]:
    """Emit a single plan step recording an unresolved answer."""
    return {
        "plan": f"unresolved — {why}",
        "plan_steps": [{
            "step_id": 1,
            "description": question or "(no question)",
            "question": question,
            "context": context,
            "range": [int(zoom_start), int(zoom_end)],
            "query": None,
            "params": {},
            "why": why,
            "error": error,
        }],
        "current_step_index": 0,
    }


def _planner(state: AgentState) -> Dict[str, Any]:
    """Classify the question + pick one or more queries from the catalog.

    Reads the parent's forwarded fields (``question``, ``context``,
    ``requested_graphs``) from ``state`` — ``delegate_step`` puts them
    there, not in a step dict. Renders the zoom graph(s) for the VLM
    classification call (the images are NOT propagated downstream).
    """
    from app.services.llm.annotation_agent_tools import (
        render_graph_builds,
        render_query_catalog_for_prompt,
    )

    question: str = (state.get("question") or state.get("goal") or "").strip()
    context: str = str(state.get("context") or "").strip()
    requested_graphs: List[str] = list(state.get("requested_graphs") or [])
    zoom_start = state.get("parent_start", 0)
    zoom_end = state.get("parent_end", 0)
    graph_builds: Dict[str, Any] = state.get("graph_builds") or {}

    LOGGER.info(
        "zoom planner: range=[%s,%s] graphs=%s question=%r",
        zoom_start, zoom_end, requested_graphs, question,
    )

    if not question:
        return _unresolved_plan(
            question, context, zoom_start, zoom_end,
            why="parent did not supply a question",
        )

    if not graph_builds:
        return _unresolved_plan(
            question, context, zoom_start, zoom_end,
            why="parent did not supply graph_builds — no data available to query",
        )

    zoom_images: List[bytes] = []
    zoom_descriptions: List[str] = []
    # Zoom range is inclusive on both ends — render_graph_builds slices
    # tables with .iloc[start:end] (exclusive end), so add 1 to include
    # iloc=zoom_end in the rendered image.
    for img, desc in render_graph_builds(
        graph_builds, zoom_start, zoom_end + 1,
    ):
        zoom_descriptions.append(desc)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        zoom_images.append(buf.getvalue())

    if not zoom_images:
        return _unresolved_plan(
            question, context, zoom_start, zoom_end,
            why="graph rendering produced no images",
        )

    # The VLM may only pick from columns present in the parent-supplied
    # tables — its query menu reflects exactly what's queryable.
    query_columns = list(_union_columns(graph_builds))
    catalog_block = render_query_catalog_for_prompt(query_columns)
    prompt_lines = [
        "You are a telemetry-query selector. Your ONLY job is to pick the "
        "minimum set of queries that, together, answer the question from "
        "the actual telemetry data. Do NOT estimate any answer from the "
        "image — each chosen query runs deterministic math on the DataFrame "
        "and returns the exact iloc / value.",
        "",
        f"**Question:** {question}",
    ]
    if context:
        prompt_lines.append(f"**Context:** {context}")
    prompt_lines.extend([
        f"**Zoom range:** [{zoom_start}, {zoom_end}] "
        f"(inclusive — length {zoom_end - zoom_start + 1})",
        f"**Graphs shown:** {', '.join(zoom_descriptions) or '(none)'}",
        "",
        catalog_block,
        "",
        "Inspect the graph(s) to classify what feature(s) the question is "
        "asking about, then pick one or more queries — typically one is "
        "enough, but questions that combine readings (e.g. 'values at "
        "indices X and Y AND the slope between them') need multiple. "
        f"Use the minimum number of queries that fully answers (cap: "
        f"{MAX_QUERIES_PER_ZOOM}). Respond JSON ONLY — no prose, no comments:",
        "```json",
        "{",
        '  "queries": [',
        '    {',
        '      "query": "<query id from catalog>",',
        '      "params": { ... params required by that query ... },',
        '      "why":   "<one sentence justifying this pick>"',
        '    }',
        '  ]',
        "}",
        "```",
        'If no query in the catalog can answer the question, return '
        '`{"queries": []}` (empty array).',
    ])
    prompt = "\n".join(prompt_lines)

    set_active_stage(ZOOM_AGENT_NAME, "planner", graphs=requested_graphs)
    set_active_attachments([])
    raw = _call_vlm(prompt, zoom_images)
    if not raw:
        return _unresolved_plan(
            question, context, zoom_start, zoom_end,
            why="VLM unavailable",
        )

    picks = _parse_query_picks(raw)
    if picks is None:
        return _unresolved_plan(
            question, context, zoom_start, zoom_end,
            why="VLM response was not parseable JSON",
            error=raw[:200],
        )

    # Cap to MAX_QUERIES_PER_ZOOM, drop entries without a query_id.
    valid_picks: List[Dict[str, Any]] = []
    for p in picks[:MAX_QUERIES_PER_ZOOM]:
        query_id = p.get("query")
        if not query_id:
            continue
        params = p.get("params") if isinstance(p.get("params"), dict) else {}
        why = str(p.get("why", ""))
        valid_picks.append({
            "query": str(query_id),
            "params": params,
            "why": why,
        })

    if not valid_picks:
        return _unresolved_plan(
            question, context, zoom_start, zoom_end,
            why="VLM picked no usable queries",
        )

    plan_steps: List[Dict[str, Any]] = []
    for i, pick in enumerate(valid_picks, start=1):
        plan_steps.append({
            "step_id": i,
            "description": question,
            "question": question,
            "context": context,
            "range": [int(zoom_start), int(zoom_end)],
            "query": pick["query"],
            "params": pick["params"],
            "why": pick["why"],
            "error": None,
        })

    plan_summary = ", ".join(
        f"{p['query']}({', '.join(f'{k}={v!r}' for k, v in p['params'].items())})"
        for p in valid_picks
    )
    return {
        "plan": plan_summary,
        "plan_steps": plan_steps,
        "current_step_index": 0,
    }


def _executor(state: AgentState, step: Dict[str, Any], registry) -> Dict[str, Any]:
    """Run the ONE planned query for this step, emit a partial attachment.

    Also emits a step event so each query run surfaces in the UI between
    the planner and synthesizer phases — otherwise the executor work is
    invisible to the user.
    """
    from app.services.llm.annotation_agent_tools import run_pipeline_query

    graph_builds: Dict[str, Any] = state.get("graph_builds") or {}
    step_id = step.get("step_id", 1)
    rng = step.get("range") or [
        state.get("parent_start", 0), state.get("parent_end", 0),
    ]
    zoom_start, zoom_end = int(rng[0]), int(rng[1])
    query_id = step.get("query")
    params = step.get("params") or {}
    why = step.get("why") or ""
    plan_error = step.get("error")

    payload: Dict[str, Any] = {
        "iloc": None, "value": None, "samples": None, "extra": None,
    }
    error: Optional[str] = plan_error
    if query_id:
        query_table = _build_query_table(graph_builds)
        if query_table is None:
            error = "graph_builds missing in state — parent did not supply any queryable tables"
        else:
            payload, error = run_pipeline_query(
                query_table, zoom_start, zoom_end, query_id, params,
            )

    iloc = payload.get("iloc")
    value = payload.get("value")
    samples = payload.get("samples")
    extra = payload.get("extra")

    partial = {
        "query": query_id,
        "params": params,
        "iloc": iloc,
        "value": value,
        "samples": samples,
        "extra": extra,
        "why": why,
        "error": error,
    }
    att = PipelineAttachment(
        name=f"partial.{step_id}",   # internal — not in produces
        kind="structured",
        content_schema="zoom_query_partial",
        label=f"Zoom partial {step_id}",
        content=partial,
    )

    # Build a human-readable per-query summary for the UI.
    params_str = ", ".join(f"{k}={v!r}" for k, v in params.items())
    if query_id is None:
        result_line = f"unresolved — {why or error or 'no query supplied'}"
    elif error and iloc is None and not samples:
        result_line = f"FAILED — {error}"
    elif samples:
        sample_lines = "\n".join(
            f"    - iloc={s.get('iloc')}, value={s.get('value')}"
            + (f" — {s.get('note')}" if s.get("note") else "")
            for s in samples if isinstance(s, dict)
        )
        head = (
            f"{len(samples)} sample(s)"
            if iloc is None
            else f"iloc={iloc}, value={value}; {len(samples)} sample(s)"
        )
        result_line = f"{head}\n{sample_lines}"
    elif iloc is not None:
        result_line = f"iloc={iloc}, value={value}"
    elif extra:
        result_line = "no primary match — see diagnostics"
    else:
        result_line = f"no match — {error or 'unknown'}"
    if extra:
        result_line += f"\n    extra: {extra}"

    emit_step_event(
        node_name=ZOOM_AGENT_NAME,
        phase="executor",
        summary=(
            f"**Query {step_id}:** `{query_id}({params_str})`\n\n"
            f"**Why:** {why}\n\n"
            f"**Result:** {result_line}"
        ),
        attachments=[att],
    )

    return {
        "step_results": [{
            "step_id": step_id,
            "agent": ZOOM_AGENT_NAME,
            "partial": partial,
        }],
        "attachment_pool": {att.name: att},
    }


def _render_partials_for_prompt(results: List[Dict[str, Any]]) -> str:
    """Render the executor's structured partials as a compact table for the LLM.

    The LLM uses this as its ONLY source of truth — every iloc / value in the
    prose must come from here, never invented.
    """
    blocks: List[str] = []
    for i, r in enumerate(results, start=1):
        query = r.get("query")
        params = r.get("params") or {}
        iloc = r.get("iloc")
        value = r.get("value")
        samples = r.get("samples")
        extra = r.get("extra")
        why = r.get("why") or ""
        error = r.get("error")

        if query is None:
            blocks.append(
                f"Query {i}: unresolved — {why or error or 'no query fit'}"
            )
            continue

        params_str = ", ".join(f"{k}={v!r}" for k, v in params.items())
        lines = [f"Query {i}: {query}({params_str})"]
        if why:
            lines.append(f"  why: {why}")

        has_scalar = iloc is not None
        has_samples = isinstance(samples, list) and len(samples) > 0
        has_extras = isinstance(extra, dict) and bool(extra)
        if not has_scalar and not has_samples and not has_extras:
            lines.append(f"  result: no match — {error or '(unspecified)'}")
        else:
            if has_scalar:
                lines.append(f"  iloc={iloc}, value={value}")
            elif not has_samples:
                # Extras-only payload — diagnostic info instead of a primary match.
                lines.append("  no primary match — diagnostic only")
            if has_samples:
                for s in samples:
                    if not isinstance(s, dict):
                        continue
                    s_iloc = s.get("iloc")
                    s_val = s.get("value")
                    s_note = s.get("note")
                    tail = f" ({s_note})" if s_note else ""
                    lines.append(f"  - iloc={s_iloc}, value={s_val}{tail}")
            if has_extras:
                extras_str = ", ".join(f"{k}={v!r}" for k, v in extra.items())
                lines.append(f"  extra: {extras_str}")
        if error:
            lines.append(f"  error: {error}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def _synthesizer(state: AgentState) -> Dict[str, Any]:
    """Convert structured query partials into a prose answer via a text LLM call.

    The LLM receives the original question + the structured query results
    (deterministic ground truth) and emits ONE short paragraph that directly
    answers the question, citing ilocs / values VERBATIM. The prose replaces
    the structured results downstream — describe_graphs consumes the prose
    directly as the answer attachment's text content.
    """
    question = (state.get("question") or state.get("goal") or "").strip()
    context = str(state.get("context") or "").strip()
    zoom_start = state.get("parent_start", 0)
    zoom_end = state.get("parent_end", 0)
    pool = state.get("attachment_pool", {})

    def _step_id_of(name: str) -> int:
        # "partial.{step_id}" → int(step_id), fallback 0.
        suffix = name.split(".", 1)[1] if "." in name else ""
        try:
            return int(suffix)
        except ValueError:
            return 0

    partial_names = sorted(
        (n for n in pool.keys() if n.startswith("partial.")),
        key=_step_id_of,
    )
    partial_attachments: List[PipelineAttachment] = []
    results: List[Dict[str, Any]] = []
    for name in partial_names:
        att = pool[name]
        partial_attachments.append(att)
        if isinstance(att.content, dict):
            results.append(att.content)

    label = f"Zoom [{zoom_start}, {zoom_end}] — Answer"

    if not results:
        prose = f"unresolved — no queries ran for question: {question}"
        answer = PipelineAttachment(
            name="answer",
            kind="text",
            label=label,
            content=prose,
        )
        emit_step_event(
            node_name=ZOOM_AGENT_NAME,
            phase="synthesizer",
            summary=prose,
            attachments=[answer],
        )
        return {"attachment_pool": {answer.name: answer}}

    partials_block = _render_partials_for_prompt(results)
    prompt_lines = [
        "You convert deterministic telemetry-query results into ONE short prose "
        "paragraph that directly answers a sub-range question. The numbers below "
        "are the ONLY source of truth — cite ilocs and values VERBATIM from the "
        "table. Do NOT invent numbers, do NOT estimate, do NOT add commentary "
        "beyond what the data shows. If a query is unresolved or failed, state "
        "that explicitly in the prose. No bullets, no headers, no lists — one "
        "flowing paragraph.",
        "",
        f"**Question:** {question}",
    ]
    if context:
        prompt_lines.append(f"**Context:** {context}")
    prompt_lines.extend([
        f"**Zoom range:** [{zoom_start}, {zoom_end}] "
        f"(inclusive — length {zoom_end - zoom_start + 1})",
        "",
        "**Query results (ground truth):**",
        partials_block,
        "",
        "Write the answer paragraph now. Output prose only — no JSON, no code "
        "fences, no preamble.",
    ])
    prompt = "\n".join(prompt_lines)

    set_active_stage(ZOOM_AGENT_NAME, "synthesizer")
    set_active_attachments(partial_attachments)
    prose = _call_llm(prompt).strip()
    if not prose:
        raise RuntimeError(
            f"zoom synthesizer: LLM returned empty response "
            f"(range=[{zoom_start}, {zoom_end}], question={question!r})"
        )

    answer = PipelineAttachment(
        name="answer",
        kind="text",
        label=label,
        content=prose,
    )

    return {"attachment_pool": {answer.name: answer}}


class Zoom(Agent):
    """Worker agent: answer ONE planner question over a sub-range, possibly
    via multiple deterministic queries.

    Planner picks one or more queries (VLM classification over the zoomed
    graph); executor runs each query (deterministic math); synthesizer
    aggregates all partials into a single ``answer`` attachment carrying
    the full results array. No images or descriptions propagated.
    """

    name = ZOOM_AGENT_NAME
    consumes = ["init.parent_segment"]
    produces = ["answer"]
    delegates_to: list = []

    def planner(self, state: AgentState) -> Dict[str, Any]:
        return _planner(state)

    def executor(self, state: AgentState, step, registry) -> Dict[str, Any]:
        return _executor(state, step, registry)

    def synthesizer(self, state: AgentState) -> Dict[str, Any]:
        return _synthesizer(state)

    def evaluator(self, state: AgentState):
        return None     # no verdict — describe_graphs synthesizer arbitrates


ZOOM_SPEC = Zoom.register()
