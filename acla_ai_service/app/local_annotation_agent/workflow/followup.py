"""
Follow-up Q&A chat against a finished annotation.

Sits OUTSIDE the agent box because the framing (prior proposals, candidate
labels, skill-debugging stance) is racing-specific intent. Uses the same
MCP tool surface the Claude runner exposes so the user can re-investigate
telemetry while debugging skill text.

    reply = run_claude_followup(
        df=df, start_index=..., end_index=...,
        parent_main_labels=..., existing_children=...,
        claude_model="claude-sonnet-4-6",
        use_thinking=False, max_turns=30,
        prior_result=annotation_result,
        chat_history=[{"role": "user", "content": "..."}],
        user_question="why didn't EA1 fit here?",
        on_text_chunk=on_text,
    )
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Callable, Dict, List, Optional

from app.annotation_providers.tool_surface import (
    AnnotationToolSurface,
    ToolAgentCapture,
    annotation_tool_registry,
)
from app.shared.contracts import AgentRequest, NoopCallbacks, ProviderConfig
from app.shared.labels import LABEL_MAPPING
from app.internal_knowledge_base.label_search import get_doc

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tool registration — uses the shared annotation tool surface, but exposes
# only the follow-up-safe subset.
# ---------------------------------------------------------------------------


_FOLLOWUP_TOOL_NAMES = [
    "query_telemetry",
    "compute_expert_phases",
    "measure_segment_shape",
    "locate_circuit_section",
    "find_nearest_opponent",
    "classify_opponent_interaction",
    "query_opponent_trajectory",
]
_FOLLOWUP_TOOL_NAME_SET = set(_FOLLOWUP_TOOL_NAMES)


def _build_tool_set(surface: AnnotationToolSurface):
    from claude_agent_sdk import tool, create_sdk_mcp_server

    def _make_tool(defn: Dict[str, Any]):
        name = str(defn["name"])

        @tool(name, str(defn["description"]), defn["params_schema"])
        async def _wrapped(args):
            result, text, _images = surface.call_tool(name, args or {})
            if isinstance(result, dict):
                return result
            return {"content": [{"type": "text", "text": text}]}

        return _wrapped

    tools_list = [
        _make_tool(defn)
        for defn in annotation_tool_registry()
        if str(defn["name"]) in _FOLLOWUP_TOOL_NAME_SET
    ]
    tool_names = [f"mcp__followup__{name}" for name in _FOLLOWUP_TOOL_NAMES]

    from app.local_annotation_agent.workflow.tools import SEARCH_LABELS_TOOL

    @tool(
        SEARCH_LABELS_TOOL["name"],
        SEARCH_LABELS_TOOL["description"],
        SEARCH_LABELS_TOOL["params_schema"],
    )
    async def search_labels(args):
        _result, text, _images = surface.call_tool("search_labels", args or {})
        return {"content": [{"type": "text", "text": text}]}

    tools_list.append(search_labels)
    tool_names.append("mcp__followup__search_labels")

    server = create_sdk_mcp_server(
        name="followup", version="1.0.0", tools=tools_list,
    )
    return server, tool_names


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------


def _format_prior_proposals(prior_result) -> str:
    proposals = list(getattr(prior_result, "label_annotations", None) or [])
    if not proposals:
        body = "  (no proposals were submitted)\n"
    else:
        lines = []
        for p in proposals:
            lid = p.get("label_id", "?")
            name = LABEL_MAPPING.get(lid, lid)
            lines.append(
                f"  - `{lid}` ({name}) over [{p.get('start_index')}, "
                f"{p.get('end_index')}]\n"
                f"    reasoning: {p.get('reasoning', '') or '(none)'}"
            )
        body = "\n".join(lines) + "\n"
    summary = getattr(prior_result, "final_reasoning", "") or "(none)"
    return f"Proposals submitted:\n{body}\nOverall summary: {summary}\n"


def _build_system_prompt(
    *,
    parent_start: int,
    parent_end: int,
    parent_main_labels: List[str],
    existing_children: List[Dict[str, Any]],
    prior_result,
    use_thinking: bool,
) -> str:
    parent_label_blocks: List[str] = []
    for pid in parent_main_labels:
        entry = get_doc(pid)
        if entry is None:
            parent_label_blocks.append(f"  - `{pid}` ({LABEL_MAPPING.get(pid, pid)})")
            continue
        desc = entry.get("description") or "(no description)"
        guideline_text = entry.get("annotation_guideline")
        guideline = f"\n      guideline: {guideline_text}" if guideline_text else ""
        parent_label_blocks.append(
            f"  - `{entry['id']}` ({entry['name']}): {desc}{guideline}"
        )

    existing_block = ""
    if existing_children:
        lines = []
        for c in existing_children:
            names = ", ".join(LABEL_MAPPING.get(l, l) for l in c.get("labels", []))
            lines.append(f"  - [{c['start_index']}, {c['end_index']}] — {names}")
        existing_block = (
            "\n### Already discovered sub-segments\n" + "\n".join(lines) + "\n"
        )

    thinking_clause = (
        "\nThink step-by-step before each tool call.\n" if use_thinking else ""
    )

    proposals_block = _format_prior_proposals(prior_result)

    return (
        "You are a racing telemetry analyst answering follow-up questions "
        "about a prior annotation pass. Your job is to help the user "
        "understand the prior proposals so they can edit the skill YAMLs "
        "(label catalog descriptions / annotation guidelines / per-graph "
        "`how_to_analyze` blocks). You are NOT producing new proposals — "
        "no submit tool is available.\n"
        "\n"
        "### Parent segment\n"
        f"- index range: [{parent_start}, {parent_end}] "
        f"(length {parent_end - parent_start})\n"
        "- parent main label(s):\n"
        + ("\n".join(parent_label_blocks) or "  (none)")
        + "\n"
        f"{existing_block}"
        "\n"
        f"### Prior session output\n{proposals_block}"
        "\n"
        "### How to answer\n"
        "- Ground every claim in telemetry evidence. Cite ilocs and values. "
        "Use `query_telemetry` / `compute_expert_phases` / "
        "`measure_segment_shape` / `classify_opponent_interaction` "
        "/ `find_nearest_opponent` / `query_opponent_trajectory` "
        "to re-inspect when the question demands fresh evidence.\n"
        "- Look labels up with `search_labels` (describe the behaviour, or "
        "pass the label's name/parent) to pull its description + guideline "
        "from the skill — don't rely on memory.\n"
        "- When asked 'why didn't label X fit?', `search_labels` for X, "
        "quote the relevant text from its description / guideline, then say "
        "which predicate failed against the data.\n"
        "- If the prior proposal was wrong, say so directly.\n"
        "- When the user is debugging the skill text, suggest concrete "
        "edits — the specific wording that was ambiguous or missing.\n"
        "- Keep replies tight. Bullets > paragraphs.\n"
        f"{thinking_clause}"
    )


def _build_initial_prompt(chat_history: List[Dict[str, str]], user_question: str) -> str:
    if not chat_history:
        history_block = "  (this is the first follow-up question)"
    else:
        lines = []
        for turn in chat_history:
            role = turn.get("role", "user")
            content = (turn.get("content", "") or "").strip()
            tag = "User" if role == "user" else "You"
            lines.append(f"- {tag}: {content}")
        history_block = "\n".join(lines)

    return (
        "Earlier conversation in this follow-up chat:\n"
        f"{history_block}\n\n"
        f"Latest user question:\n{user_question.strip()}\n\n"
        "Answer concisely, cite ilocs / values, and use the telemetry "
        "query tools if you need fresh evidence."
    )


# ---------------------------------------------------------------------------
# Session loop
# ---------------------------------------------------------------------------


def _import_sdk_types():
    from types import SimpleNamespace
    try:
        from claude_agent_sdk import (
            query, ClaudeAgentOptions,
            AssistantMessage, TextBlock, ToolUseBlock, ToolResultBlock,
        )
    except ImportError as exc:
        raise RuntimeError(
            "claude-agent-sdk is not installed. Install with "
            "`pip install claude-agent-sdk`."
        ) from exc
    try:
        from claude_agent_sdk.types import ThinkingBlock  # type: ignore
    except ImportError:
        ThinkingBlock = None  # type: ignore
    return SimpleNamespace(
        query=query, ClaudeAgentOptions=ClaudeAgentOptions,
        AssistantMessage=AssistantMessage, TextBlock=TextBlock,
        ToolUseBlock=ToolUseBlock, ThinkingBlock=ThinkingBlock,
    )


async def _run_async(
    *,
    df,
    parent_start: int,
    parent_end: int,
    parent_main_labels: List[str],
    existing_children: List[Dict[str, Any]],
    claude_model: str,
    use_thinking: bool,
    max_turns: int,
    prior_result,
    chat_history: List[Dict[str, str]],
    user_question: str,
    on_text_chunk: Optional[Callable[[str], None]],
) -> str:
    sdk = _import_sdk_types()

    from app.local_annotation_agent.workflow.tools import SEARCH_LABELS_TOOL

    capture = ToolAgentCapture(
        node_name="followup",
        cur_start=int(parent_start),
        cur_end=int(parent_end),
    )
    request = AgentRequest(
        provider_id="claude_cli",
        config=ProviderConfig(
            provider_id="claude_cli",
            model=claude_model,
            provider_options={"use_thinking": use_thinking},
        ),
        planner_prompt="",
        synth_prompt=lambda _state: ("", ""),
        df_ref=df,
        parent_start=int(parent_start),
        parent_end=int(parent_end),
        callbacks=NoopCallbacks(),
        extra_state={"tool_agent_extra_tools": [SEARCH_LABELS_TOOL]},
    )
    surface = AnnotationToolSurface(request, capture)
    server, tool_names = _build_tool_set(surface)

    options = sdk.ClaudeAgentOptions(
        model=claude_model,
        mcp_servers={"followup": server},
        allowed_tools=tool_names,
        system_prompt=_build_system_prompt(
            parent_start=parent_start,
            parent_end=parent_end,
            parent_main_labels=parent_main_labels,
            existing_children=existing_children,
            prior_result=prior_result,
            use_thinking=use_thinking,
        ),
        max_turns=max_turns,
    )

    prompt = _build_initial_prompt(chat_history, user_question)
    response_chunks: List[str] = []

    async for message in sdk.query(prompt=prompt, options=options):
        if not isinstance(message, sdk.AssistantMessage):
            continue
        for block in getattr(message, "content", None) or []:
            if isinstance(block, sdk.TextBlock):
                text = getattr(block, "text", "") or ""
                if text:
                    response_chunks.append(text)
                    if on_text_chunk is not None:
                        on_text_chunk(text)
            elif isinstance(block, sdk.ToolUseBlock):
                capture.tool_calls += 1

    return "".join(response_chunks).strip()


def run_claude_followup(
    *,
    df,
    start_index: int,
    end_index: int,
    parent_main_labels: List[str],
    existing_children: Optional[List[dict]],
    claude_model: str,
    use_thinking: bool,
    max_turns: int,
    prior_result,
    chat_history: List[Dict[str, str]],
    user_question: str,
    on_text_chunk: Optional[Callable[[str], None]] = None,
) -> str:
    """One follow-up Q&A turn against a finished annotation. Returns the reply text."""
    return asyncio.run(_run_async(
        df=df,
        parent_start=int(start_index),
        parent_end=int(end_index),
        parent_main_labels=list(parent_main_labels),
        existing_children=list(existing_children or []),
        claude_model=claude_model,
        use_thinking=bool(use_thinking),
        max_turns=int(max_turns),
        prior_result=prior_result,
        chat_history=list(chat_history),
        user_question=user_question,
        on_text_chunk=on_text_chunk,
    ))
