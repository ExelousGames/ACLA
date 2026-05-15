"""
zoom — Agent that answers ALL planner questions over their sub-ranges in one go.

Topology (standard planner → executor(loop) → synthesizer):

    planner     — receives the parent's full ``questions`` list (each
                  carrying its own question, context, sub-range, and
                  requested_graphs subset). Renders the zoom graphs for
                  every question, then makes ONE VLM call that classifies
                  every question against its images and picks one or more
                  queries from ``PIPELINE_QUERY_DEFINITIONS`` for each.
                  Emits N plan steps — one per picked query — each tagged
                  with ``question_index`` plus ``{query, params, why}``.
                  Questions with no usable query become a single marker
                  step so the synthesizer can still record an unresolved
                  answer.

    executor    — runs the planned query for the step on the question's
                  sub-range against the question's graph-filtered
                  ``graph_builds`` (deterministic math). Emits
                  ``partial.{step_id}`` — an internal attachment, NOT in
                  ``produces`` — carrying the result plus its
                  ``question_index``.

    synthesizer — groups every ``partial.*`` by ``question_index``, makes
                  ONE text-LLM call that converts the structured results
                  for every question into a clearly delimited prose block
                  (one paragraph per question, citing ilocs verbatim), and
                  emits ONE ``answer`` attachment (kind=text) covering
                  every question. ``delegate_step`` renamespaces it to
                  ``step_solver.{parent_step_id}.answer`` for the
                  describe_graphs synthesizer to consume directly.

Inputs from the parent (forwarded into sub_state by ``delegate_step`` —
the parent step's fields land directly in ``state``):

    state["questions"]         required — list of question dicts, each:
                                 {question, context, start, end,
                                  requested_graphs: [graph_id, ...]}
    state["graph_builds"]      required — {graph_id: DataFrame} built by
                               the parent agent. Each table holds only the
                               data that graph uses (drawn series +
                               renderer-internal columns). Zoom renders /
                               queries against these tables only — raw df
                               is not available.
    state["parent_start"/_end] envelope range (full parent scope).

Output (single attachment, leaf name ``answer``, kind=text):
    Combined prose covering every question, with ilocs / values cited
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

# Cap on the number of queries the planner may pick per question — prevents
# the VLM from going wild with redundant readings on any single question.
MAX_QUERIES_PER_QUESTION = 4


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


def _parse_multi_question_picks(text: str) -> Optional[Dict[int, List[Dict[str, Any]]]]:
    """Extract ``{question_index: [query_pick, ...]}`` from the VLM response.

    The VLM must respond with::

        {"questions": [
           {"question_index": 1,
            "queries": [{"query": "...", "params": {...}, "why": "..."}, ...]},
           ...]}

    Returns the mapping (possibly with empty lists for unresolved questions),
    or ``None`` when no JSON object is found or the ``questions`` key is
    missing / malformed.
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
    raw_questions = parsed.get("questions")
    if not isinstance(raw_questions, list):
        return None

    out: Dict[int, List[Dict[str, Any]]] = {}
    for entry in raw_questions:
        if not isinstance(entry, dict):
            continue
        qi = entry.get("question_index")
        if not isinstance(qi, int):
            continue
        queries = entry.get("queries")
        if not isinstance(queries, list):
            continue
        out[qi] = [q for q in queries if isinstance(q, dict)]
    return out


def _normalize_questions(state: AgentState) -> List[Dict[str, Any]]:
    """Read questions from state, accepting either the multi-question or
    legacy single-question shape.

    Returns a list of normalized question dicts:
        {question, context, start, end, requested_graphs}
    """
    raw = state.get("questions")
    if isinstance(raw, list) and raw:
        normalized: List[Dict[str, Any]] = []
        for q in raw:
            if not isinstance(q, dict):
                continue
            normalized.append({
                "question": str(q.get("question", "")).strip(),
                "context": str(q.get("context", "") or "").strip(),
                "start": int(q.get("start", state.get("parent_start", 0))),
                "end": int(q.get("end", state.get("parent_end", 0))),
                "requested_graphs": list(q.get("requested_graphs") or []),
            })
        return [q for q in normalized if q["question"]]

    # Legacy single-question shape — kept so direct callers without the
    # describe_graphs collapsing still work.
    q_text = (state.get("question") or state.get("goal") or "").strip()
    if not q_text:
        return []
    return [{
        "question": q_text,
        "context": str(state.get("context") or "").strip(),
        "start": int(state.get("parent_start", 0)),
        "end": int(state.get("parent_end", 0)),
        "requested_graphs": list(state.get("requested_graphs") or []),
    }]


def _unresolved_step(
    sid: int,
    qi: int,
    question: Dict[str, Any],
    why: str,
    error: Optional[str] = None,
) -> Dict[str, Any]:
    """Single marker step recording an unresolved answer for one question."""
    return {
        "step_id": sid,
        "question_index": qi,
        "description": question["question"] or "(no question)",
        "question": question["question"],
        "context": question["context"],
        "question_range": [int(question["start"]), int(question["end"])],
        "requested_graphs": list(question["requested_graphs"]),
        "query": None,
        "params": {},
        "why": why,
        "error": error,
    }


def _planner(state: AgentState) -> Dict[str, Any]:
    """Classify every question + pick queries from the catalog in ONE VLM call.

    Reads the parent's forwarded ``questions`` list from ``state`` (or
    falls back to the legacy single-question shape). Renders every
    question's sub-range graphs into a single image batch, labels them
    in the prompt so the VLM can associate images with questions, and
    asks the VLM to pick queries per question_index.
    """
    from app.services.llm.annotation_agent_tools import (
        render_graph_builds,
        render_query_catalog_for_prompt,
    )

    questions = _normalize_questions(state)
    graph_builds: Dict[str, Any] = state.get("graph_builds") or {}

    LOGGER.info(
        "zoom planner: %d question(s), graphs=%s",
        len(questions), sorted(graph_builds.keys()),
    )

    if not questions:
        return {
            "plan": "unresolved — no questions supplied",
            "plan_steps": [_unresolved_step(
                1, 1,
                {"question": "(no question)", "context": "",
                 "start": state.get("parent_start", 0),
                 "end": state.get("parent_end", 0),
                 "requested_graphs": []},
                why="parent did not supply any questions",
            )],
            "current_step_index": 0,
        }

    if not graph_builds:
        plan_steps = [
            _unresolved_step(
                i, i, q,
                why="parent did not supply graph_builds — no data available to query",
            )
            for i, q in enumerate(questions, start=1)
        ]
        return {
            "plan": "unresolved — no graph_builds",
            "plan_steps": plan_steps,
            "current_step_index": 0,
        }

    # Render each question's sub-range graphs. Track image-index ranges so
    # the prompt can tell the VLM "images N..M correspond to question Q".
    per_q_render: List[Dict[str, Any]] = []
    all_images: List[bytes] = []
    for qi, q in enumerate(questions, start=1):
        sub_builds = {
            gid: graph_builds[gid]
            for gid in q["requested_graphs"]
            if gid in graph_builds
        } or graph_builds
        q_images: List[bytes] = []
        q_descs: List[str] = []
        # Zoom range is inclusive on both ends — render_graph_builds slices
        # with .iloc[start:end] (exclusive end), so add 1 to include the
        # last iloc in the rendered image.
        for img, desc in render_graph_builds(
            sub_builds, q["start"], q["end"] + 1,
        ):
            q_descs.append(desc)
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            q_images.append(buf.getvalue())
        per_q_render.append({
            "qi": qi,
            "question": q,
            "image_count": len(q_images),
            "descriptions": q_descs,
        })
        all_images.extend(q_images)

    if not all_images:
        plan_steps = [
            _unresolved_step(
                i, i, q,
                why="graph rendering produced no images for this question",
            )
            for i, q in enumerate(questions, start=1)
        ]
        return {
            "plan": "unresolved — no images rendered",
            "plan_steps": plan_steps,
            "current_step_index": 0,
        }

    # The VLM may only pick from columns present in the parent-supplied
    # tables — its query menu reflects exactly what's queryable.
    catalog_block = render_query_catalog_for_prompt(list(_union_columns(graph_builds)))

    prompt_lines = [
        "You are a telemetry-query selector. For EACH question below pick the "
        "minimum set of queries that, together, answer it from the actual "
        "telemetry data. Each chosen query runs deterministic math on the "
        "DataFrame and returns the exact iloc / value — do NOT estimate any "
        "answer from the image.",
        "",
    ]
    img_offset = 1
    for entry in per_q_render:
        q = entry["question"]
        ic = entry["image_count"]
        if ic == 0:
            img_label = "no images rendered"
        elif ic == 1:
            img_label = f"image {img_offset}"
        else:
            img_label = f"images {img_offset}-{img_offset + ic - 1}"
        prompt_lines.append(f"### Question {entry['qi']}")
        prompt_lines.append(f"**Question:** {q['question']}")
        if q["context"]:
            prompt_lines.append(f"**Context:** {q['context']}")
        prompt_lines.append(
            f"**Sub-range:** [{q['start']}, {q['end']}] "
            f"(inclusive — length {q['end'] - q['start'] + 1})"
        )
        prompt_lines.append(
            f"**Graphs ({img_label}):** "
            f"{', '.join(entry['descriptions']) or '(none)'}"
        )
        prompt_lines.append("")
        img_offset += ic

    prompt_lines.extend([
        catalog_block,
        "",
        "Inspect the relevant images for each question, classify the feature "
        "the question is asking about, then pick one or more queries — "
        "typically one is enough, but questions that combine readings (e.g. "
        "'values at ilocs X and Y AND the slope between them') need multiple. "
        f"Cap per question: {MAX_QUERIES_PER_QUESTION}. **Every query you "
        "pick MUST include `range: [start_iloc, end_iloc]` in its params.** "
        "Use a tight window around the feature you're trying to capture; "
        "stay within the question's sub-range. Respond JSON ONLY — "
        "no prose, no comments:",
        "```json",
        "{",
        '  "questions": [',
        '    {',
        '      "question_index": <int — matches the Question N heading>,',
        '      "queries": [',
        '        {',
        '          "query":  "<query id from catalog>",',
        '          "params": {',
        '            "range":  [<start_iloc>, <end_iloc>],',
        '            ...other params required by that query...',
        '          },',
        '          "why":    "<one sentence justifying this pick>"',
        '        }',
        '      ]',
        '    }',
        '  ]',
        '}',
        "```",
        'If no query in the catalog can answer a question, return '
        '`"queries": []` for that question_index. Include every question_index '
        'in the response.',
    ])
    prompt = "\n".join(prompt_lines)

    set_active_stage(
        ZOOM_AGENT_NAME, "planner",
        graphs=list(graph_builds.keys()),
    )
    set_active_attachments([])
    raw = _call_vlm(prompt, all_images)
    if not raw:
        plan_steps = [
            _unresolved_step(i, i, q, why="VLM unavailable")
            for i, q in enumerate(questions, start=1)
        ]
        return {
            "plan": "unresolved — VLM unavailable",
            "plan_steps": plan_steps,
            "current_step_index": 0,
        }

    picks_by_qi = _parse_multi_question_picks(raw)
    if picks_by_qi is None:
        plan_steps = [
            _unresolved_step(
                i, i, q,
                why="VLM response was not parseable JSON",
                error=raw[:200],
            )
            for i, q in enumerate(questions, start=1)
        ]
        return {
            "plan": "unresolved — VLM response not parseable",
            "plan_steps": plan_steps,
            "current_step_index": 0,
        }

    # Build plan_steps: one per (question, query). Questions with no usable
    # pick get a single unresolved marker step so the synthesizer can record
    # them in the prose.
    plan_steps: List[Dict[str, Any]] = []
    sid = 1
    plan_summary_parts: List[str] = []
    for entry in per_q_render:
        qi = entry["qi"]
        q = entry["question"]
        raw_picks = picks_by_qi.get(qi, [])
        valid_picks: List[Dict[str, Any]] = []
        for p in raw_picks[:MAX_QUERIES_PER_QUESTION]:
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

        # Drop picks that don't include a valid `range` in their params —
        # every query now requires a per-query range.
        ranged_picks: List[Dict[str, Any]] = []
        for pick in valid_picks:
            r = pick["params"].get("range")
            if (
                not isinstance(r, (list, tuple))
                or len(r) != 2
                or not all(isinstance(v, int) for v in r)
            ):
                LOGGER.info(
                    "zoom planner: dropping Q%d pick %s — missing/invalid range %r",
                    qi, pick["query"], r,
                )
                continue
            # Normalize to a plain list of ints so downstream rendering is stable.
            pick["params"]["range"] = [int(r[0]), int(r[1])]
            ranged_picks.append(pick)

        if not ranged_picks:
            plan_steps.append(_unresolved_step(
                sid, qi, q,
                why="VLM picked no queries with a valid `range` param for this question",
            ))
            plan_summary_parts.append(f"Q{qi}: unresolved")
            sid += 1
            continue

        for pick in ranged_picks:
            plan_steps.append({
                "step_id": sid,
                "question_index": qi,
                "description": q["question"],
                "question": q["question"],
                "context": q["context"],
                "question_range": [int(q["start"]), int(q["end"])],
                "requested_graphs": list(q["requested_graphs"]),
                "query": pick["query"],
                "params": pick["params"],
                "why": pick["why"],
                "error": None,
            })
            params_str = ", ".join(f"{k}={v!r}" for k, v in pick["params"].items())
            plan_summary_parts.append(f"Q{qi}: {pick['query']}({params_str})")
            sid += 1

    return {
        "plan": "; ".join(plan_summary_parts) or "no picks",
        "plan_steps": plan_steps,
        "current_step_index": 0,
    }


def _executor(state: AgentState, step: Dict[str, Any], registry) -> Dict[str, Any]:
    """Run the planned query for this step, emit a partial attachment.

    Filters ``graph_builds`` to the step's ``requested_graphs`` subset so a
    query for question A can't accidentally read columns supplied only for
    question B's graphs. Also emits a step event so each query run surfaces
    in the UI between the planner and synthesizer phases.
    """
    from app.services.llm.annotation_agent_tools import run_pipeline_query

    all_builds: Dict[str, Any] = state.get("graph_builds") or {}
    step_id = step.get("step_id", 1)
    qi = step.get("question_index", step_id)
    query_id = step.get("query")
    params = step.get("params") or {}
    why = step.get("why") or ""
    plan_error = step.get("error")
    requested = step.get("requested_graphs") or []
    step_builds = {gid: all_builds[gid] for gid in requested if gid in all_builds} or all_builds

    # The per-query range lives in params["range"] (set by the planner from
    # the VLM's pick). Fall back to the question's range for unresolved /
    # query-less steps so the partial still records a meaningful window.
    raw_range = params.get("range") if isinstance(params, dict) else None
    if isinstance(raw_range, (list, tuple)) and len(raw_range) == 2:
        zoom_start, zoom_end = int(raw_range[0]), int(raw_range[1])
    else:
        fallback = step.get("question_range") or [
            state.get("parent_start", 0), state.get("parent_end", 0),
        ]
        zoom_start, zoom_end = int(fallback[0]), int(fallback[1])

    payload: Dict[str, Any] = {
        "iloc": None, "value": None, "samples": None, "extra": None,
    }
    error: Optional[str] = plan_error
    if query_id:
        query_table = _build_query_table(step_builds)
        if query_table is None:
            error = "graph_builds missing in state — parent did not supply any queryable tables"
        else:
            payload, error = run_pipeline_query(
                query_table, query_id, params,
            )

    iloc = payload.get("iloc")
    value = payload.get("value")
    samples = payload.get("samples")
    extra = payload.get("extra")

    partial = {
        "question_index": qi,
        "question": step.get("question", ""),
        "context": step.get("context", ""),
        "range": [zoom_start, zoom_end],
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
        label=f"Zoom partial {step_id} (Q{qi})",
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
            f"**Q{qi} step {step_id}:** `{query_id}({params_str})`\n\n"
            f"**Why:** {why}\n\n"
            f"**Result:** {result_line}"
        ),
        attachments=[att],
    )

    return {
        "step_results": [{
            "step_id": step_id,
            "question_index": qi,
            "agent": ZOOM_AGENT_NAME,
            "partial": partial,
        }],
        "attachment_pool": {att.name: att},
    }


def _render_partials_block(
    qi: int,
    question: str,
    context: str,
    rng: List[int],
    results: List[Dict[str, Any]],
) -> str:
    """Render one question's structured partials as a compact prompt block."""
    lines = [f"### Question {qi}: {question}"]
    if context:
        lines.append(f"  context: {context}")
    if rng and len(rng) == 2:
        lines.append(f"  range: [{rng[0]}, {rng[1]}] (length {rng[1] - rng[0] + 1})")
    if not results:
        lines.append("  (no queries ran)")
        return "\n".join(lines)
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
            lines.append(
                f"  query {i}: unresolved — {why or error or 'no query fit'}"
            )
            continue
        params_str = ", ".join(f"{k}={v!r}" for k, v in params.items())
        lines.append(f"  query {i}: {query}({params_str})")
        if why:
            lines.append(f"    why: {why}")
        has_scalar = iloc is not None
        has_samples = isinstance(samples, list) and len(samples) > 0
        has_extras = isinstance(extra, dict) and bool(extra)
        if not has_scalar and not has_samples and not has_extras:
            lines.append(f"    result: no match — {error or '(unspecified)'}")
        else:
            if has_scalar:
                lines.append(f"    iloc={iloc}, value={value}")
            elif not has_samples:
                lines.append("    no primary match — diagnostic only")
            if has_samples:
                for s in samples:
                    if not isinstance(s, dict):
                        continue
                    s_iloc = s.get("iloc")
                    s_val = s.get("value")
                    s_note = s.get("note")
                    tail = f" ({s_note})" if s_note else ""
                    lines.append(f"    - iloc={s_iloc}, value={s_val}{tail}")
            if has_extras:
                extras_str = ", ".join(f"{k}={v!r}" for k, v in extra.items())
                lines.append(f"    extra: {extras_str}")
        if error:
            lines.append(f"    error: {error}")
    return "\n".join(lines)


def _synthesizer(state: AgentState) -> Dict[str, Any]:
    """Convert per-question query partials into combined prose via one LLM call.

    Groups every ``partial.*`` by ``question_index``, then emits one prompt
    that asks the LLM to write a clearly delimited paragraph for each
    question, citing ilocs / values verbatim. The combined prose is the
    single ``answer`` attachment describe_graphs consumes.
    """
    parent_start = state.get("parent_start", 0)
    parent_end = state.get("parent_end", 0)
    pool = state.get("attachment_pool", {})

    def _step_id_of(name: str) -> int:
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
    # Preserve question_index ordering as encountered in the partials.
    groups_order: List[int] = []
    groups: Dict[int, Dict[str, Any]] = {}
    for name in partial_names:
        att = pool[name]
        partial_attachments.append(att)
        content = att.content if isinstance(att.content, dict) else {}
        qi = content.get("question_index")
        if not isinstance(qi, int):
            qi = _step_id_of(name)
        if qi not in groups:
            groups_order.append(qi)
            groups[qi] = {
                "question": content.get("question", ""),
                "context": content.get("context", ""),
                "range": content.get("range") or [parent_start, parent_end],
                "results": [],
            }
        groups[qi]["results"].append(content)

    label = f"Zoom [{parent_start}, {parent_end}] — Answer"

    if not groups:
        prose = "unresolved — no queries ran for any question."
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

    blocks = [
        _render_partials_block(
            qi,
            groups[qi]["question"],
            groups[qi]["context"],
            groups[qi]["range"],
            groups[qi]["results"],
        )
        for qi in groups_order
    ]

    prompt_lines = [
        "You convert deterministic telemetry-query results into a prose answer "
        "that covers EVERY question below. Write one short paragraph per "
        "question, in the order given, each prefixed with `**Q{i} (range "
        "[a,b]):**` so downstream readers can locate each answer. Preserve "
        "each question's `context:` phrase verbatim in its paragraph — that "
        "phrase carries the glossary term the downstream describer needs. "
        "The numbers below are the ONLY source of truth — cite ilocs and "
        "values VERBATIM from the table. Do NOT invent numbers, do NOT "
        "estimate, do NOT add commentary beyond what the data shows. If a "
        "query is unresolved or failed, state that explicitly in that "
        "question's paragraph. No bullets, no headers beyond the question "
        "prefix, no lists.",
        "",
        f"**Parent range:** [{parent_start}, {parent_end}]",
        "",
        "**Query results (ground truth — grouped by question):**",
        "\n\n".join(blocks),
        "",
        "Write the answer paragraphs now. Output prose only — no JSON, no "
        "code fences, no preamble.",
    ]
    prompt = "\n".join(prompt_lines)

    set_active_stage(ZOOM_AGENT_NAME, "synthesizer")
    set_active_attachments(partial_attachments)
    prose = _call_llm(prompt).strip()
    if not prose:
        raise RuntimeError(
            f"zoom synthesizer: LLM returned empty response "
            f"(parent_range=[{parent_start}, {parent_end}], "
            f"questions={len(groups)})"
        )

    answer = PipelineAttachment(
        name="answer",
        kind="text",
        label=label,
        content=prose,
    )

    return {"attachment_pool": {answer.name: answer}}


class Zoom(Agent):
    """Worker agent: answer ALL parent questions in a single invocation.

    Planner picks queries for every question in one VLM call (one or more
    queries per question, capped per-question); executor runs each query
    deterministically; synthesizer aggregates every partial into one
    ``answer`` attachment carrying combined prose with per-question
    paragraphs. No images or descriptions propagated to the parent.
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
