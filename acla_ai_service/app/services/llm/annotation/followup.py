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
        on_text_chunk=on_text, on_tool_event=on_tool,
    )
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from app.skills import skills
from app.models.segment_models import LABEL_MAPPING

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tool surface — mirrors the Claude runner's tools so the user can
# re-investigate during a follow-up.
# ---------------------------------------------------------------------------


@dataclass
class _FollowupCapture:
    rendered_images: List[bytes] = field(default_factory=list)
    tool_calls: int = 0


class _ToolSurface:
    def __init__(self, df, parent_start: int, parent_end: int, capture: _FollowupCapture) -> None:
        self.df = df
        self.parent_start = int(parent_start)
        self.parent_end = int(parent_end)
        self.capture = capture

    def _clamp(self, s: int, e: int) -> tuple[int, int]:
        s2 = max(self.parent_start, int(s))
        e2 = min(self.parent_end, int(e))
        if e2 <= s2:
            e2 = min(self.parent_end, s2 + 1)
        return s2, e2

    def list_graphs(self) -> str:
        from app.services.llm.agent.tools import AGENT_GRAPH_DEFINITIONS
        out = [
            {"id": g["id"], "title": g["title"], "description": g["description"]}
            for g in AGENT_GRAPH_DEFINITIONS
        ]
        return json.dumps({"graphs": out}, indent=2)

    def get_graph_guidance(self, graph_ids: List[str]) -> str:
        text = skills.render("graph_analysis", graph_ids=list(graph_ids))
        return text or "(no guidance available for the requested graph(s))"

    def render_graph(self, graph_id: str, start: int, end: int) -> Dict[str, Any]:
        from app.services.llm.agent.tools import build_graph, render_graph_builds
        s, e = self._clamp(start, end)
        table = build_graph(graph_id, self.df)
        if table is None or table.empty:
            return {
                "content": [{"type": "text", "text": f"Cannot render `{graph_id}` over [{s}, {e}]."}],
                "is_error": True,
            }
        rendered = render_graph_builds({graph_id: table}, s, e)
        if not rendered:
            return {
                "content": [{"type": "text", "text": f"`{graph_id}` produced no image for [{s}, {e}]."}],
                "is_error": True,
            }
        img, desc = rendered[0]
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        png = buf.getvalue()
        self.capture.rendered_images.append(png)
        encoded = base64.b64encode(png).decode("ascii")
        return {
            "content": [
                {"type": "image", "data": encoded, "mimeType": "image/png"},
                {"type": "text", "text": f"{desc} (rendered over [{s}, {e}])"},
            ],
        }

    def query_telemetry(self, query_id: str, params_json: str) -> str:
        from app.services.llm.agent.tools import run_pipeline_query
        try:
            params = json.loads(params_json) if params_json else {}
        except json.JSONDecodeError as exc:
            return json.dumps({"error": f"params_json was not valid JSON: {exc}"})
        if not isinstance(params, dict):
            return json.dumps({"error": "params_json must decode to a JSON object."})
        payload, err = run_pipeline_query(self.df, query_id, params)
        out = {"query": query_id, "params": params, "result": payload}
        if err:
            out["error"] = err
        return json.dumps(out, default=str)

    def compute_expert_phases(self, start: int, end: int) -> str:
        from app.services.llm.agent.tools import compute_expert_phases
        s, e = self._clamp(start, end)
        att = compute_expert_phases(self.df, s, e)
        return json.dumps({"phases_range": [s, e], "data": att.content}, default=str)

    def locate_circuit_section(self, start: int, end: int) -> str:
        from app.services.llm.agent.tools import locate_circuit_section
        s, e = self._clamp(start, end)
        att = locate_circuit_section(self.df, s, e)
        return json.dumps({"range": [s, e], "data": att.content}, default=str)


def _build_tool_set(surface: _ToolSurface):
    from claude_agent_sdk import tool, create_sdk_mcp_server

    @tool(
        "list_graphs",
        "List every available telemetry graph as a compact catalog of `id` + `title` + `description`.",
        {},
    )
    async def list_graphs(args):  # noqa: ARG001
        return {"content": [{"type": "text", "text": surface.list_graphs()}]}

    @tool(
        "get_graph_guidance",
        "Return the per-graph `how_to_analyze` + glossary for the given graph IDs.",
        {"graph_ids": list},
    )
    async def get_graph_guidance(args):
        ids = args.get("graph_ids") or []
        if not isinstance(ids, list):
            ids = [str(ids)]
        ids = [str(x) for x in ids]
        return {"content": [{"type": "text", "text": surface.get_graph_guidance(ids)}]}

    @tool(
        "render_graph",
        "Render ONE telemetry graph over an iloc window and return the PNG + descriptor.",
        {"graph_id": str, "start": int, "end": int},
    )
    async def render_graph(args):
        return surface.render_graph(
            str(args["graph_id"]), int(args["start"]), int(args["end"]),
        )

    @tool(
        "query_telemetry",
        "Run a deterministic numeric query on the raw telemetry DataFrame. "
        "`query_id` is one of: find_extremum, find_first_match, "
        "read_values_at_indices, compute_slope, find_dips_on_main_slope, "
        "find_threshold_crossing. `params_json` must include `range: [start, end]`.",
        {"query_id": str, "params_json": str},
    )
    async def query_telemetry(args):
        text = surface.query_telemetry(
            str(args["query_id"]), str(args.get("params_json", "")),
        )
        return {"content": [{"type": "text", "text": text}]}

    @tool(
        "compute_expert_phases",
        "Detect expert-anchored corner phases over an iloc window.",
        {"start": int, "end": int},
    )
    async def compute_expert_phases(args):
        text = surface.compute_expert_phases(int(args["start"]), int(args["end"]))
        return {"content": [{"type": "text", "text": text}]}

    @tool(
        "locate_circuit_section",
        "Identify which named circuit_section the iloc window overlaps.",
        {"start": int, "end": int},
    )
    async def locate_circuit_section(args):
        text = surface.locate_circuit_section(int(args["start"]), int(args["end"]))
        return {"content": [{"type": "text", "text": text}]}

    tools_list = [
        list_graphs, get_graph_guidance, render_graph, query_telemetry,
        compute_expert_phases, locate_circuit_section,
    ]
    tool_names = [f"mcp__followup__{t}" for t in [
        "list_graphs", "get_graph_guidance", "render_graph",
        "query_telemetry", "compute_expert_phases", "locate_circuit_section",
    ]]

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
        entry = skills.get(f"sub_label_catalog.labels.{pid}")
        if entry is None:
            parent_label_blocks.append(f"  - `{pid}` ({LABEL_MAPPING.get(pid, pid)})")
            continue
        desc = entry.get("description") or "(no description)"
        guideline_text = entry.get("annotation_guideline")
        guideline = f"\n      guideline: {guideline_text}" if guideline_text else ""
        parent_label_blocks.append(
            f"  - `{entry['id']}` ({entry['name']}): {desc}{guideline}"
        )

    sub_label_blocks: List[str] = []
    seen: set = set()
    for pid in parent_main_labels:
        for entry in skills.find("sub_label_catalog.labels", parent=pid):
            if entry["id"] in seen:
                continue
            seen.add(entry["id"])
            desc = entry.get("description") or "(no description)"
            guideline_text = entry.get("annotation_guideline")
            guideline = f"\n    guideline: {guideline_text}" if guideline_text else ""
            sub_label_blocks.append(
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
        "### Candidate labels referenced by the prior pass\n"
        + ("\n".join(sub_label_blocks) or "  (none)")
        + "\n\n"
        f"### Prior session output\n{proposals_block}"
        "\n"
        "### How to answer\n"
        "- Ground every claim in telemetry evidence. Cite ilocs and values. "
        "Use `render_graph` / `query_telemetry` / `compute_expert_phases` "
        "to re-inspect when the question demands fresh evidence.\n"
        "- When asked 'why didn't label X fit?', quote the relevant text "
        "from the label's description / guideline above, then say which "
        "predicate failed against the data.\n"
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
        "tools if you need fresh evidence."
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
    on_tool_event: Optional[Callable[[str, Dict[str, Any]], None]],
) -> str:
    sdk = _import_sdk_types()

    capture = _FollowupCapture()
    surface = _ToolSurface(df, parent_start, parent_end, capture)
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
                if on_tool_event is not None:
                    on_tool_event(
                        getattr(block, "name", "tool"),
                        getattr(block, "input", {}) or {},
                    )

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
    on_tool_event: Optional[Callable[[str, Dict[str, Any]], None]] = None,
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
        on_tool_event=on_tool_event,
    ))
