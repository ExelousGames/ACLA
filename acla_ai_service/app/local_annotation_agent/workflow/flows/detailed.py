"""
Detailed (sub-segment discovery) flow.

Wraps the agent box for the "discover ONE notable sub-segment within a
parent segment" use case. Provides:

    build_request(provider_id, prompt_mode, df, range_, ...) -> AgentRequest
    parse(response, ...) -> AnnotationResult

The prompts and parsing here are racing-specific (parent_main_labels,
sub-label discovery, label_id JSON schema). The box never sees them
directly — it just executes the planner / synth / submit prompts the
caller supplies and returns raw text the parser decodes.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from app.internal_knowledge_base import skills
from app.shared.labels import LABEL_MAPPING
from app.shared.contracts import (
    AgentCallbacks,
    AgentRequest,
    AgentResponse,
    Attachment,
    NoopCallbacks,
    ProviderConfig,
)
from app.local_annotation_agent.workflow.results import (
    AnnotationResult,
    parse_json_response,
)
from app.local_annotation_agent.workflow.preflight_detailed import (
    build_preflight_context,
)
from app.local_annotation_agent.workflow.tools import shape_label_doc_for_llm


def _is_full_parent_range(
    start: int,
    end: int,
    parent_start: int,
    parent_end: int,
) -> bool:
    return int(start) == int(parent_start) and int(end) == int(parent_end)


# ---------------------------------------------------------------------------
# Tool-agent planner prompt — task description + submit schema
# ---------------------------------------------------------------------------


def _tool_agent_task_prompt(
    *,
    parent_start: int,
    parent_end: int,
    parent_main_labels: List[str],
    existing_children: List[dict],
) -> str:
    """User-message prompt the tool-agent runner sends as the session start."""
    range_fit_rule = str(
        skills.get("sub_label_annotation.range_fit_rule", "")
    ).strip()
    flow_rules = skills.get("sub_label_annotation.detailed_flow_rules", [])
    if not isinstance(flow_rules, list):
        raise RuntimeError(
            "sub_label_annotation.detailed_flow_rules must be a list"
        )
    annotation_rules = [
        rule
        for rule in [
            range_fit_rule,
            *(str(rule).strip() for rule in flow_rules),
        ]
        if rule
    ]
    annotation_rules_block = "\n".join(annotation_rules)
    annotation_rule_bullets = "\n".join(
        f"- {rule}" for rule in annotation_rules
    )

    parent_label_blocks = [
        f"  - `{pid}` ({LABEL_MAPPING.get(pid, pid)})"
        for pid in parent_main_labels
    ]

    existing_block = ""
    if existing_children:
        lines = []
        for c in existing_children:
            names = ", ".join(LABEL_MAPPING.get(l, l) for l in c.get("labels", []))
            lines.append(
                f"  - [{c['start_index']}, {c['end_index']}] — {names}"
            )
        existing_block = (
            "\n### Existing child proposals for duplicate checks\n"
            + "\n".join(lines) + "\n"
        )

    return (
        "Discover strongly supported candidate-label sub-segment(s) within "
        "the parent segment below and submit them via `submit_result`. A valid "
        "sub-segment is a strict child range: it may touch one parent "
        "boundary, but it must not be identical to the parent range.\n"
        "\n"
        "### Parent segment\n"
        f"- index range: [{parent_start}, {parent_end}] "
        f"(length {parent_end - parent_start})\n"
        "- parent main label(s):\n"
        + ("\n".join(parent_label_blocks) or "  (none)")
        + "\n"
        f"{existing_block}"
        "\n"
        "### How to work\n"
        "1. Use the Required Upfront Detailed Statistical Preflight block "
        "as the primary evidence package. It contains human-readable fact "
        "sentences from deterministic tools. Preflight does not identify "
        "labels; it only provides facts.\n"
        "2. Review the Upfront Detailed Embedding Label Candidates block. "
        "Those candidates come from hybrid embedding search over the "
        "annotation knowledge base using the preflight semantic search "
        "words.\n"
        "3. Call additional data tools only to resolve a concrete missing "
        "detail, not to rediscover the basic analysis path already covered "
        "by preflight.\n"
        "4. Audit the parent range according to the detailed annotation "
        "rules below.\n"
        "5. Submit via `submit_result(payload_json, summary)` when evidence "
        "is sufficient, then stop after it returns `ok: true`. If the "
        "evidence only supports the whole parent range, submit an empty "
        "`proposals` list and say no strict child sub-segment was found.\n"
        "\n"
        "### Detailed annotation rules\n"
        f"{annotation_rules_block}\n"
        "\n"
        "### Submit payload shape\n"
        "`payload_json` must be a JSON object of this shape:\n"
        "```json\n"
        "{\n"
        '  "proposals": [\n'
        '    {\n'
        '      "label_id": "<a label_id from the upfront embedding candidates>",\n'
        f'      "start_index": <int in [{parent_start}, {parent_end}]>,\n'
        f'      "end_index": <int in [{parent_start}, {parent_end}]>,\n'
        '      "reasoning": "<2-4 sentence human-readable evidence note citing ilocs, values, trends, tool verdicts, and any ambiguous option rejected>"\n'
        '    }\n'
        "  ]\n"
        "}\n"
        "```\n"
        "Use `{\"proposals\": []}` when no strict child range is supported.\n"
        "\n"
        "### Hard rules\n"
        f"- Every proposed range must satisfy {parent_start} <= start_index < end_index <= {parent_end}.\n"
        f"- A proposed range must not be identical to the parent range [{parent_start}, {parent_end}].\n"
        f"{annotation_rule_bullets}\n"
        "- Parent labels are inherited context only; they are not enough evidence for a child proposal.\n"
        "- Only propose label_ids from the Upfront Detailed Embedding Label "
        "Candidates block. The AI does not search for labels in this flow.\n"
        "- For O, OD, and MSR racing sub-labels, do not use expert/reference-lap "
        "comparisons as evidence. Use opponent-relative preflight facts such "
        "as who started ahead, who ended ahead, who drew alongside whom, "
        "which side the opponent was on, gap shrink/flip facts, relative "
        "speed, acceleration, and deceleration facts.\n"
        "- For O, OD, and MSR racing sub-labels, reuse phrases from the "
        "preflight fact sentences in the `reasoning` field when those facts "
        "support the selected label.\n"
        "- For time-delta and trajectory evidence, cite deterministic tool "
        "verdict fields (unit, label-significance, whole-section "
        "slope-shape trend); do not "
        "create strength judgments from raw numbers.\n"
        "- For uphill, level, and downhill altitude labels, use the "
        "deterministic slope-angle verdict from `measure_segment_shape`; "
        "do not decide the label directly from raw height difference.\n"
        "- Make each `reasoning` field a human annotation note: 2-4 "
        "concise sentences with the key ilocs/ranges, values/trends, "
        "tool verdicts, and why those facts support the proposed child "
        "range.\n"
        "- When evidence is ambiguous between two plausible labels or "
        "ranges, choose the best-supported option and state in `reasoning` "
        "which other option was not selected and why.\n"
        "- Do not propose ranges that exactly match an already-discovered sub-segment.\n"
        "- After `submit_result` returns `ok: true`, stop calling tools."
    )


def _preflight_semantic_search_text(preflight) -> str:
    for attachment in getattr(preflight, "attachments", []) or []:
        if getattr(attachment, "name", "") != "init.annotation_preflight_context":
            continue
        content = getattr(attachment, "content", None)
        if not isinstance(content, dict):
            continue
        evidence = str(
            content.get("semantic_search_text")
            or content.get("semantic_evidence_text")
            or ""
        ).strip()
        if evidence:
            return evidence
    return ""


def _embedding_label_candidates(
    *,
    evidence_text: str,
    parent_main_labels: List[str],
) -> List[Dict[str, Any]]:
    """Hybrid-search candidate labels for detailed annotation.

    Detailed preflight only prepares semantic evidence sentences; this flow-level
    step performs the embedding retrieval before the AI chooses ranges.
    """
    from app.internal_knowledge_base.label_reranker import rerank_label_docs
    from app.internal_knowledge_base.label_search import get_doc, search

    query = (evidence_text or "").strip()
    if not query:
        return []

    merged: Dict[str, Dict[str, Any]] = {}

    def add(docs: List[Dict[str, Any]]) -> None:
        for doc in docs:
            label_id = str(doc.get("id") or "")
            if not label_id:
                continue
            current = merged.get(label_id)
            score = float(doc.get("score", 0.0) or 0.0)
            current_score = float((current or {}).get("score", 0.0) or 0.0)
            if current is None or score > current_score:
                merged[label_id] = dict(doc)

    main_parents = [
        label_id
        for label_id in parent_main_labels
        if (get_doc(label_id) or {}).get("type") == "main"
    ]

    add(search(query, filters={"type": "segment_type"}, top_k=12))
    if main_parents:
        for parent_id in main_parents:
            add(search(query, filters={"parent": parent_id}, top_k=12))
    else:
        add(search(query, filters={"type": "main"}, top_k=12))

    return [
        shape_label_doc_for_llm(doc)
        for doc in rerank_label_docs(query, list(merged.values()))
    ]


def _embedding_candidates_prompt_block(candidates: List[Dict[str, Any]]) -> str:
    lines = [
        "#### Upfront Detailed Embedding Label Candidates",
        "The detailed flow already ran hybrid embedding search over "
        "annotation knowledge using the preflight semantic evidence sentences.",
        "These are candidate labels, not final labels; attach one only "
        "when its definition fits the whole proposed child range.",
        "Candidate labels:",
    ]
    if not candidates:
        lines.append(
            "- (none found; submit an empty proposals list because no "
            "label_id is authorized for this detailed pass)"
        )
        return "\n".join(lines)

    for entry in candidates:
        desc = str(entry.get("description") or "").strip()
        row = (
            f"- `{entry.get('id')}` ({entry.get('name', '')}) "
            f"| type={entry.get('type')} | score={entry.get('score')}"
        )
        if entry.get("parent"):
            row += f" | parent={entry.get('parent')}"
        if desc:
            row += f" — {desc}"
        lines.append(row)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public flow API
# ---------------------------------------------------------------------------


def build_request(
    *,
    provider_id: str,
    prompt_mode: str,
    df,
    parent_start: int,
    parent_end: int,
    parent_main_labels: List[str],
    existing_children: Optional[List[dict]] = None,
    config: Optional[ProviderConfig] = None,
    callbacks: Optional[AgentCallbacks] = None,
    session_id: str = "",
) -> AgentRequest:
    """Build the AgentRequest for one detailed-flow run.

    Detailed annotation is driven by Claude/OpenAI tool-agent providers.
    """
    if prompt_mode != "tool_agent":
        raise ValueError(
            f"detailed flow only supports prompt_mode='tool_agent'; got {prompt_mode!r}"
        )

    existing_children = list(existing_children or [])
    config = config or ProviderConfig(provider_id=provider_id)
    callbacks = callbacks or NoopCallbacks()

    parent_segment = Attachment(
        name="init.parent_segment",
        kind="structured",
        content_schema="parent_segment",
        label="Parent Segment",
        content={
            "parent_start": int(parent_start),
            "parent_end": int(parent_end),
            "main_labels": list(parent_main_labels),
            "main_label_names": [LABEL_MAPPING.get(l, l) for l in parent_main_labels],
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

    preflight = build_preflight_context(
        df=df,
        start=parent_start,
        end=parent_end,
        parent_main_labels=parent_main_labels,
        extra_query_terms=[
            LABEL_MAPPING.get(label_id, label_id)
            for label_id in parent_main_labels
        ],
    )
    embedding_candidates = _embedding_label_candidates(
        evidence_text=_preflight_semantic_search_text(preflight),
        parent_main_labels=parent_main_labels,
    )
    for attachment in preflight.attachments:
        if attachment.name == "init.annotation_preflight_context" and isinstance(
            attachment.content,
            dict,
        ):
            attachment.content["label_candidate_ids"] = [
                c["id"] for c in embedding_candidates if c.get("id")
            ]
            break

    planner_prompt = _tool_agent_task_prompt(
        parent_start=parent_start,
        parent_end=parent_end,
        parent_main_labels=parent_main_labels,
        existing_children=existing_children,
    )
    planner_prompt = "\n\n".join([
        preflight.prompt_block,
        _embedding_candidates_prompt_block(embedding_candidates),
        planner_prompt,
    ])
    synth_prompt = lambda _state: ("", "")

    extra_state: Dict[str, Any] = {}

    return AgentRequest(
        provider_id=provider_id,
        config=config,
        planner_prompt=planner_prompt,
        synth_prompt=synth_prompt,
        df_ref=df,
        parent_start=int(parent_start),
        parent_end=int(parent_end),
        initial_attachments=[
            parent_segment,
            *preflight.attachments,
            Attachment(
                name="init.preflight_label_candidates",
                kind="structured",
                label="Upfront Detailed Embedding Label Candidates",
                content={
                    "range": [int(parent_start), int(parent_end)],
                    "candidates": embedding_candidates,
                },
                content_schema="annotation_preflight_labels",
            ),
        ],
        callbacks=callbacks,
        session_id=session_id,
        extra_state=extra_state,
    )


def parse(
    response: AgentResponse,
    *,
    prompt_mode: str,
    parent_start: int,
    parent_end: int,
) -> AnnotationResult:
    """Decode the agent's raw_response into an AnnotationResult.

    ``prompt_mode="tool_agent"`` expects a submitted JSON object with a
    ``proposals`` key, or a direct ``label_ids`` payload from providers that
    captured plain assistant JSON.
    """
    if prompt_mode != "tool_agent":
        raise ValueError(
            f"detailed flow only supports prompt_mode='tool_agent'; got {prompt_mode!r}"
        )

    raw = response.raw_response or ""
    return _parse_tool_agent(response, raw, parent_start, parent_end)


def _parse_tool_agent(
    response: AgentResponse,
    raw: str,
    parent_start: int,
    parent_end: int,
) -> AnnotationResult:
    label_ids: List[str] = []
    label_proposals: List[dict] = []
    proposed_start = parent_start
    proposed_end = parent_end
    reasoning = raw

    parsed = parse_json_response(raw) if raw else None
    if parsed:
        proposals = parsed.get("proposals", [])
        if isinstance(proposals, list):
            starts: List[int] = []
            ends: List[int] = []
            for p in proposals:
                if not isinstance(p, dict):
                    continue
                lid = p.get("label_id")
                if lid not in LABEL_MAPPING:
                    continue
                try:
                    s = int(p.get("start_index"))
                    e = int(p.get("end_index"))
                except (TypeError, ValueError):
                    continue
                if not (parent_start <= s < e <= parent_end):
                    continue
                if _is_full_parent_range(s, e, parent_start, parent_end):
                    continue
                if lid not in label_ids:
                    label_ids.append(lid)
                label_proposals.append({
                    "label_id": lid,
                    "start_index": s,
                    "end_index": e,
                    "reasoning": str(p.get("reasoning", "")),
                })
                starts.append(s)
                ends.append(e)

            if starts:
                proposed_start = min(starts)
            if ends:
                proposed_end = max(ends)
        elif isinstance(parsed.get("label_ids"), list):
            try:
                s = int(parsed.get("start_index", parent_start))
                e = int(parsed.get("end_index", parent_end))
            except (TypeError, ValueError):
                s, e = parent_start, parent_end
            if parent_start <= s < e <= parent_end:
                proposed_start = s
                proposed_end = e
                note = str(parsed.get("reasoning") or parsed.get("summary") or "")
                for lid in parsed["label_ids"]:
                    if lid not in LABEL_MAPPING or lid in label_ids:
                        continue
                    label_ids.append(lid)
                    label_proposals.append({
                        "label_id": lid,
                        "start_index": proposed_start,
                        "end_index": proposed_end,
                        "reasoning": note,
                    })

    # Prefer the synthesizer.summary attachment as the high-level
    # reasoning; otherwise use the transcript / raw payload.
    summary_att = response.attachments.get("synthesizer.summary")
    if summary_att and isinstance(summary_att.content, str) and summary_att.content:
        reasoning = summary_att.content
    elif label_proposals:
        reasoning = "; ".join(p["reasoning"] for p in label_proposals)

    return AnnotationResult(
        sub_start=proposed_start,
        sub_end=proposed_end,
        final_labels=list(dict.fromkeys(label_ids)),
        final_reasoning=reasoning,
        accepted=response.verdict == "submitted" and len(label_proposals) > 0,
        iterations=1,
        messages=list(response.messages),
        graph_images=list(response.graph_images),
        label_annotations=label_proposals,
    )
