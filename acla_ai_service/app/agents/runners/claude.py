"""
Claude runner — one agentic Claude session per AgentRequest.

Different paradigm from the local runner: instead of a LangGraph
planner/executor/synth/eval cycle, this hands control to a single Claude
session that calls MCP tools to inspect telemetry and submit a result.
One subprocess start, multi-turn reasoning in one context.

The runner builds its OWN system prompt (Claude-specific framing of tool
surface + submission mechanics) and uses ``AgentRequest.planner_prompt``
as the initial user message — the caller's intent reaches Claude there.

Box stays flow-free by exposing generic capability tools:

    list_graphs                 catalog of telemetry graphs
    get_graph_guidance          per-graph how_to_analyze blocks
    render_graph                PNG + descriptor over [start, end]
    query_telemetry             deterministic math on the df
    compute_expert_phases       per-arc entry/apex/exit ilocs
    locate_circuit_section      named-section match for an iloc window
    get_circuit_id              canonical circuit id from Static_track
    revise_range                shrink/extend the working iloc range
    submit_result               capture the final structured answer + summary

Callers add domain-specific tools via
``AgentRequest.extra_state["claude_extra_tools"]``. Each entry is a
``{name, description, params_schema, handler}`` dict; ``handler`` is a
callable ``(surface, args_dict) -> str | dict`` whose return is wrapped
as an MCP text result.

Whether ``revise_range`` and ``submit_result`` semantics fit the flow is
decided by the caller's planner prompt — the runner just exposes the
capability and captures whatever Claude submits.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from app.agents.contracts import (
    AgentRequest,
    AgentResponse,
    Attachment,
    StepEvent,
)

LOGGER = logging.getLogger(__name__)

_CLAUDE_NODE = "claude_agent"


def _stage(phase: str, **extra) -> Dict[str, Any]:
    return {"node_name": _CLAUDE_NODE, "phase": phase, **extra}


# ---------------------------------------------------------------------------
# Per-session capture
# ---------------------------------------------------------------------------


@dataclass
class _Capture:
    """Mutable state collected during one Claude session."""

    # The agent's working range — starts at the request's parent_start/end
    # and is updated by ``revise_range`` calls. Returned in the AgentResponse
    # so flows that care (e.g. lap excerpter) can read the final range.
    cur_start: int = 0
    cur_end: int = 0
    revised: bool = False

    # ``submit_result`` deposits its raw JSON payload + summary here.
    submit_payload: str = ""
    submit_summary: str = ""
    submitted: bool = False

    rendered_images: List[bytes] = field(default_factory=list)
    text_chunks: List[str] = field(default_factory=list)
    step_events: List[StepEvent] = field(default_factory=list)
    tool_calls: int = 0


# ---------------------------------------------------------------------------
# System prompt — Claude-specific framing built INSIDE the runner
# ---------------------------------------------------------------------------


def _build_system_prompt(request: AgentRequest) -> str:
    """Compose the Claude session's system prompt.

    The caller's intent rides in ``request.planner_prompt`` (delivered as
    the user message); this prompt is just the tool-surface + workflow
    framing Claude needs to operate agentically. It deliberately contains
    NO racing-specific text — every domain rule comes through the user
    message.
    """
    thinking_clause = (
        "\nThink step-by-step before each tool call: state what you need "
        "to confirm, pick the most direct tool, read the result before "
        "deciding the next step.\n"
        if request.config.claude_use_thinking else ""
    )

    return (
        "You are an analyst with agentic access to a domain dataset via "
        "MCP tools. Your task is described in the user message — follow it "
        "precisely. Inspect the data, run queries, then submit a final "
        "structured result.\n"
        "\n"
        "### Available tools\n"
        "- `list_graphs` — compact catalog of every renderable graph (id, "
        "title, description). Call once early to learn what's available.\n"
        "- `get_graph_guidance(graph_ids)` — per-graph `how_to_analyze` + "
        "glossary blocks for the subset you actually want to inspect.\n"
        "- `render_graph(graph_id, start, end)` — PNG image + one-line "
        "descriptor over an iloc window. You can see the image natively. "
        "Range must lie inside the working section.\n"
        "- `peek_graph(graph_id, start, end)` — same as `render_graph` but "
        "the range may extend outside the working section, up to the full "
        "lap. Use to inspect adjacent telemetry for disambiguation (e.g. "
        "pit-limiter speed just before the section). Does NOT change the "
        "working range; you cannot label outside the section.\n"
        "- `query_telemetry(query_id, params_json)` — deterministic numeric "
        "queries on the raw DataFrame. Returns exact ilocs / values — "
        "prefer this over estimating from images. `params_json` must "
        "include `range: [start_iloc, end_iloc]`; the range may extend "
        "outside the working section for context (same envelope as "
        "`peek_graph`).\n"
        "- `compute_expert_phases(start, end)` — corner entry/apex/exit "
        "ilocs derived from expert telemetry.\n"
        "- `locate_circuit_section(start, end)` — named-section match "
        "ranked by normalised-position overlap.\n"
        "- `get_circuit_id()` — canonical circuit id from `Static_track`. "
        "Call once at session start to scope downstream tool calls.\n"
        "- `revise_range(new_start, new_end)` — change the iloc window "
        "you are working on before submitting. Only call this when the "
        "user message explicitly allows boundary revision.\n"
        "- `submit_result(payload_json, summary)` — capture the final "
        "structured answer. `payload_json` is a JSON-encoded object whose "
        "shape the user message defines. `summary` is a one-paragraph "
        "human-readable note. After `submit_result` returns `ok: true`, "
        "stop calling tools.\n"
        "\n"
        "The user message may describe additional task-specific tools the "
        "session also has access to — call them as it directs.\n"
        "\n"
        "### Working range\n"
        f"Initial range: [{request.parent_start}, {request.parent_end}].\n"
        "Iloc arguments to `render_graph`, `compute_expert_phases`, and "
        "`locate_circuit_section` must lie inside this range unless "
        "`revise_range` has moved it. `peek_graph` and `query_telemetry` "
        "may use any range inside the full lap for context, but the "
        "submission stays anchored to the working section.\n"
        "\n"
        "### Hard rules\n"
        "- Do not invent identifiers. Use only the IDs / labels / "
        "categories the user message authorises.\n"
        "- Budget tool calls: prefer the smallest set of renders / "
        "queries that proves the answer.\n"
        "- After `submit_result` returns `ok: true`, do not call more "
        "tools.\n"
        f"{thinking_clause}"
    )


# ---------------------------------------------------------------------------
# Tool implementations — call into the same agent/tools layer the
# local runner's sub-agents use.
# ---------------------------------------------------------------------------


class _ToolSurface:
    """Thin object holding df + capture; closures call its methods."""

    def __init__(self, request: AgentRequest, capture: _Capture) -> None:
        self.df = request.df_ref
        self.request = request
        self.capture = capture

    def _clamp_to_window(self, s: int, e: int) -> tuple[int, int]:
        lo = min(self.capture.cur_start, self.request.parent_start)
        hi = max(self.capture.cur_end, self.request.parent_end)
        s2 = max(lo, int(s))
        e2 = min(hi, int(e))
        if e2 <= s2:
            e2 = min(hi, s2 + 1)
        return s2, e2

    def _clamp_to_lap(self, s: int, e: int) -> tuple[int, int]:
        n = len(self.df)
        s2 = max(0, int(s))
        e2 = min(n, int(e))
        if e2 <= s2:
            e2 = min(n, s2 + 1)
        return s2, e2

    def _emit_tool_event(self, name: str, inp: Dict[str, Any], summary: str) -> None:
        inp_str = json.dumps(inp, default=str)
        if len(inp_str) > 400:
            inp_str = inp_str[:400] + "…"
        if len(summary) > 600:
            summary = summary[:600] + "…"
        msg = (
            f"**Tool:** `{name}`\n\n"
            f"**Input:** `{inp_str}`\n\n"
            f"**Result:** {summary}"
        )
        stage = _stage(f"tool:{name}")
        self.capture.step_events.append(StepEvent(
            stage=stage["node_name"], summary=msg, detail=stage,
        ))
        cb = self.request.callbacks
        if cb.step_event:
            cb.step_event(msg, stage)

    def list_graphs(self) -> str:
        from app.agents.tools import AGENT_GRAPH_DEFINITIONS
        out = [
            {"id": g["id"], "title": g["title"], "description": g["description"]}
            for g in AGENT_GRAPH_DEFINITIONS
        ]
        return json.dumps({"graphs": out}, indent=2)

    def get_circuit_id(self) -> str:
        from app.agents.tools import get_circuit_id
        att = get_circuit_id(self.df)
        return json.dumps(att.content, default=str)

    def get_graph_guidance(self, graph_ids: List[str]) -> str:
        from app.agents.tools import graph_analysis_prompt
        text = graph_analysis_prompt(graph_ids=list(graph_ids))
        return text or "(no guidance available for the requested graph(s))"

    def render_graph(self, graph_id: str, start: int, end: int) -> Dict[str, Any]:
        from app.agents.tools import build_graph, render_graph_builds
        s, e = self._clamp_to_window(start, end)
        table = build_graph(graph_id, self.df)
        if table is None or table.empty:
            return {
                "content": [{
                    "type": "text",
                    "text": (
                        f"Cannot render `{graph_id}` over [{s}, {e}]: the "
                        f"underlying telemetry columns are not present."
                    ),
                }],
                "is_error": True,
            }
        rendered = render_graph_builds({graph_id: table}, s, e)
        if not rendered:
            return {
                "content": [{
                    "type": "text",
                    "text": f"`{graph_id}` produced no image for [{s}, {e}].",
                }],
                "is_error": True,
            }
        img, desc = rendered[0]
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        png_bytes = buf.getvalue()
        self.capture.rendered_images.append(png_bytes)
        encoded = base64.b64encode(png_bytes).decode("ascii")
        return {
            "content": [
                {"type": "image", "data": encoded, "mimeType": "image/png"},
                {"type": "text", "text": f"{desc} (rendered over [{s}, {e}])"},
            ],
        }

    def peek_graph(self, graph_id: str, start: int, end: int) -> Dict[str, Any]:
        """Like ``render_graph`` but clamped to the LAP envelope, not the
        working section. Use to inspect telemetry just before / after the
        section for context (e.g. pit-limiter speed in the prior ilocs).
        Does NOT change the working range; cannot be used by
        ``submit_result`` to justify a label outside the section.
        """
        from app.agents.tools import build_graph, render_graph_builds
        s, e = self._clamp_to_lap(start, end)
        table = build_graph(graph_id, self.df)
        if table is None or table.empty:
            return {
                "content": [{
                    "type": "text",
                    "text": (
                        f"Cannot peek `{graph_id}` over [{s}, {e}]: the "
                        f"underlying telemetry columns are not present."
                    ),
                }],
                "is_error": True,
            }
        rendered = render_graph_builds({graph_id: table}, s, e)
        if not rendered:
            return {
                "content": [{
                    "type": "text",
                    "text": f"`{graph_id}` produced no image for [{s}, {e}].",
                }],
                "is_error": True,
            }
        img, desc = rendered[0]
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        png_bytes = buf.getvalue()
        self.capture.rendered_images.append(png_bytes)
        encoded = base64.b64encode(png_bytes).decode("ascii")
        return {
            "content": [
                {"type": "image", "data": encoded, "mimeType": "image/png"},
                {"type": "text", "text": (
                    f"{desc} (peek — context only, working range unchanged, "
                    f"rendered over [{s}, {e}])"
                )},
            ],
        }

    def query_telemetry(self, query_id: str, params_json: str) -> str:
        from app.agents.tools import run_pipeline_query
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
        from app.agents.tools import compute_expert_phases
        s, e = self._clamp_to_window(start, end)
        att = compute_expert_phases(self.df, s, e)
        return json.dumps({"phases_range": [s, e], "data": att.content}, default=str)

    def locate_circuit_section(self, start: int, end: int) -> str:
        from app.agents.tools import locate_circuit_section
        s, e = self._clamp_to_window(start, end)
        att = locate_circuit_section(self.df, s, e)
        return json.dumps({"range": [s, e], "data": att.content}, default=str)

    def revise_range(self, new_start: int, new_end: int) -> str:
        s, e = int(new_start), int(new_end)
        # Caller defines the legal envelope through the planner prompt;
        # the runner only enforces a basic well-formedness check so a
        # nonsense revise doesn't poison subsequent tool calls.
        if e <= s:
            return json.dumps({
                "ok": False,
                "error": f"new range [{s}, {e}] requires start < end",
            })
        if (e - s) < 5:
            return json.dumps({
                "ok": False,
                "error": f"new range too short ({e - s} ilocs) — minimum 5 required",
            })
        self.capture.cur_start = s
        self.capture.cur_end = e
        self.capture.revised = True
        return json.dumps({
            "ok": True,
            "new_range": [s, e],
            "note": "Working range updated. Tool calls now operate against this range.",
        })

    def submit_result(self, payload_json: str, summary: str) -> str:
        # Validate JSON parseability so Claude can fix obvious mistakes;
        # schema-level validation belongs to the caller's parse() step.
        try:
            json.loads(payload_json)
        except json.JSONDecodeError as exc:
            return json.dumps({
                "ok": False,
                "error": f"payload_json was not valid JSON: {exc}. Re-emit with valid JSON.",
            })
        self.capture.submit_payload = payload_json
        self.capture.submit_summary = str(summary or "")
        self.capture.submitted = True
        return json.dumps({
            "ok": True,
            "note": "Result captured. Session can end now.",
        })


# ---------------------------------------------------------------------------
# MCP tool registration
# ---------------------------------------------------------------------------


def _build_tool_set(surface: _ToolSurface):
    """Return (mcp_server, allowed_tool_names) for the session."""
    from claude_agent_sdk import tool, create_sdk_mcp_server

    @tool(
        "list_graphs",
        "List every available telemetry graph as a compact catalog of `id` + `title` + `description`.",
        {},
    )
    async def list_graphs(args):  # noqa: ARG001
        text = surface.list_graphs()
        surface._emit_tool_event("list_graphs", {}, f"{text.count(chr(10))}-line catalog")
        return {"content": [{"type": "text", "text": text}]}

    @tool(
        "get_circuit_id",
        "Read the `Static_track` column from the telemetry and return the "
        "canonical circuit id (e.g. `brands_hatch`, `silverstone`). Call "
        "once at the start of a session so subsequent calls (eligible "
        "labels, circuit-section match) can be scoped to the right circuit.",
        {},
    )
    async def get_circuit_id(args):  # noqa: ARG001
        text = surface.get_circuit_id()
        surface._emit_tool_event("get_circuit_id", {}, text)
        return {"content": [{"type": "text", "text": text}]}

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
        text = surface.get_graph_guidance(ids)
        surface._emit_tool_event(
            "get_graph_guidance", {"graph_ids": ids},
            f"{text.count(chr(10))} line(s) of guidance",
        )
        return {"content": [{"type": "text", "text": text}]}

    @tool(
        "render_graph",
        "Render ONE telemetry graph over an iloc window and return the PNG + descriptor.",
        {"graph_id": str, "start": int, "end": int},
    )
    async def render_graph(args):
        result = surface.render_graph(
            str(args["graph_id"]), int(args["start"]), int(args["end"]),
        )
        surface._emit_tool_event(
            "render_graph",
            {"graph_id": args.get("graph_id"), "start": args.get("start"), "end": args.get("end")},
            "image returned" if not result.get("is_error") else "render failed",
        )
        return result

    @tool(
        "peek_graph",
        "Render ONE telemetry graph over an iloc window that may extend "
        "OUTSIDE the working section, up to the full lap envelope. Same "
        "PNG + descriptor shape as `render_graph`. Use to inspect telemetry "
        "just before / after the section for disambiguation (e.g. "
        "pit-limiter speed in the prior ilocs). Does NOT change the working "
        "range — `submit_result` is still constrained to the section. "
        "`query_telemetry` already accepts arbitrary ranges, so this tool "
        "exists specifically for image-based context.",
        {"graph_id": str, "start": int, "end": int},
    )
    async def peek_graph(args):
        result = surface.peek_graph(
            str(args["graph_id"]), int(args["start"]), int(args["end"]),
        )
        surface._emit_tool_event(
            "peek_graph",
            {"graph_id": args.get("graph_id"), "start": args.get("start"), "end": args.get("end")},
            "image returned (peek)" if not result.get("is_error") else "peek failed",
        )
        return result

    @tool(
        "query_telemetry",
        "Run a deterministic numeric query on the raw telemetry DataFrame. "
        "`query_id` is one of: find_extremum, find_first_match, "
        "read_values_at_indices, compute_slope, find_dips_on_main_slope, "
        "find_threshold_crossing. `params_json` is a JSON-encoded object that "
        "must include `range: [start_iloc, end_iloc]` plus per-query fields. "
        "The `range` may extend outside the working section for context "
        "(same envelope as `peek_graph`).",
        {"query_id": str, "params_json": str},
    )
    async def query_telemetry(args):
        params_json = str(args.get("params_json", ""))
        text = surface.query_telemetry(str(args["query_id"]), params_json)
        surface._emit_tool_event(
            "query_telemetry",
            {"query_id": args.get("query_id"), "params_json": params_json},
            text,
        )
        return {"content": [{"type": "text", "text": text}]}

    @tool(
        "compute_expert_phases",
        "Detect expert-anchored corner phases (per-arc entry / apex / exit ilocs) over an iloc window.",
        {"start": int, "end": int},
    )
    async def compute_expert_phases(args):
        text = surface.compute_expert_phases(int(args["start"]), int(args["end"]))
        surface._emit_tool_event(
            "compute_expert_phases",
            {"start": args.get("start"), "end": args.get("end")}, text,
        )
        return {"content": [{"type": "text", "text": text}]}

    @tool(
        "locate_circuit_section",
        "Identify which named circuit_section the iloc window overlaps.",
        {"start": int, "end": int},
    )
    async def locate_circuit_section(args):
        text = surface.locate_circuit_section(int(args["start"]), int(args["end"]))
        surface._emit_tool_event(
            "locate_circuit_section",
            {"start": args.get("start"), "end": args.get("end")}, text,
        )
        return {"content": [{"type": "text", "text": text}]}

    @tool(
        "revise_range",
        "Shrink or extend the working iloc range before submitting. Only call "
        "when the user message authorises boundary revision. After calling, "
        "subsequent tool calls operate against the new range.",
        {"new_start": int, "new_end": int},
    )
    async def revise_range(args):
        text = surface.revise_range(int(args["new_start"]), int(args["new_end"]))
        surface._emit_tool_event(
            "revise_range",
            {"new_start": args.get("new_start"), "new_end": args.get("new_end")},
            text,
        )
        return {"content": [{"type": "text", "text": text}]}

    @tool(
        "submit_result",
        "Submit the final structured answer and end the session. "
        "`payload_json` is a JSON-encoded object whose shape the user "
        "message defines. `summary` is a one-paragraph human-readable "
        "note. Returns `ok: true` on success.",
        {"payload_json": str, "summary": str},
    )
    async def submit_result(args):
        text = surface.submit_result(
            str(args.get("payload_json", "")),
            str(args.get("summary", "")),
        )
        parsed = json.loads(text)
        summary_line = (
            "captured" if parsed.get("ok") else f"REJECTED: {parsed.get('error', '?')}"
        )
        surface._emit_tool_event(
            "submit_result", {"summary": args.get("summary")}, summary_line,
        )
        return {"content": [{"type": "text", "text": text}]}

    tools_list = [
        list_graphs, get_graph_guidance, render_graph, peek_graph, query_telemetry,
        compute_expert_phases, locate_circuit_section,
        get_circuit_id,
        revise_range, submit_result,
    ]
    tool_names = [
        "mcp__agent__list_graphs",
        "mcp__agent__get_graph_guidance",
        "mcp__agent__render_graph",
        "mcp__agent__peek_graph",
        "mcp__agent__query_telemetry",
        "mcp__agent__compute_expert_phases",
        "mcp__agent__locate_circuit_section",
        "mcp__agent__get_circuit_id",
        "mcp__agent__revise_range",
        "mcp__agent__submit_result",
    ]

    # Caller-supplied extras — each spec becomes an MCP tool whose
    # implementation is the caller's ``handler(surface, args) -> str|dict``.
    for spec in surface.request.extra_state.get("claude_extra_tools") or []:
        wrapped, qualified_name = _make_extra_tool(spec, surface, tool)
        tools_list.append(wrapped)
        tool_names.append(qualified_name)

    server = create_sdk_mcp_server(
        name="agent",
        version="1.0.0",
        tools=tools_list,
    )
    return server, tool_names


def _make_extra_tool(spec: Dict[str, Any], surface: "_ToolSurface", tool_decorator):
    """Build an MCP tool from a caller-supplied spec.

    Spec shape::

        {
            "name": str,
            "description": str,
            "params_schema": {param_name: type, ...},
            "handler": Callable[[surface, args_dict], str | dict],
        }

    The handler may be sync or async; its return is wrapped as a single
    MCP text block.
    """
    name = str(spec["name"])
    description = str(spec["description"])
    params_schema = spec.get("params_schema") or {}
    handler = spec["handler"]

    @tool_decorator(name, description, params_schema)
    async def _wrapped(args):
        try:
            if asyncio.iscoroutinefunction(handler):
                result = await handler(surface, args)
            else:
                result = handler(surface, args)
        except Exception as exc:
            result = json.dumps({"error": str(exc)})
        if not isinstance(result, str):
            result = json.dumps(result, default=str)
        short_summary = result if len(result) <= 200 else result[:200] + "…"
        surface._emit_tool_event(name, args, short_summary)
        return {"content": [{"type": "text", "text": result}]}

    return _wrapped, f"mcp__agent__{name}"


# ---------------------------------------------------------------------------
# Session runner
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
            "`pip install claude-agent-sdk` and ensure the `claude` CLI is "
            "logged in."
        ) from exc
    try:
        from claude_agent_sdk.types import ThinkingBlock  # type: ignore
    except ImportError:
        ThinkingBlock = None  # type: ignore
    return SimpleNamespace(
        query=query, ClaudeAgentOptions=ClaudeAgentOptions,
        AssistantMessage=AssistantMessage, TextBlock=TextBlock,
        ToolUseBlock=ToolUseBlock, ToolResultBlock=ToolResultBlock,
        ThinkingBlock=ThinkingBlock,
    )


async def _run_session_async(
    request: AgentRequest,
    capture: _Capture,
) -> None:
    sdk = _import_sdk_types()
    surface = _ToolSurface(request, capture)
    server, tool_names = _build_tool_set(surface)

    system_prompt = _build_system_prompt(request)
    user_message = request.planner_prompt

    options = sdk.ClaudeAgentOptions(
        model=request.config.claude_model,
        mcp_servers={"agent": server},
        allowed_tools=tool_names,
        system_prompt=system_prompt,
        # Bound the session — generous enough for multi-step exploration,
        # tight enough to stop runaway. Caller can override via extra_state.
        max_turns=int(request.extra_state.get("claude_max_turns", 30)),
    )

    cb = request.callbacks
    if cb.vlm_prompt:
        cb.vlm_prompt(user_message, _stage("main"))
    if cb.progress:
        cb.progress(_CLAUDE_NODE, "session starting")

    async for message in sdk.query(prompt=user_message, options=options):
        _handle_message(
            message, capture, cb,
            sdk.AssistantMessage, sdk.TextBlock, sdk.ToolUseBlock,
            sdk.ThinkingBlock,
        )

    if cb.progress:
        cb.progress(
            _CLAUDE_NODE,
            f"done — {capture.tool_calls} tool call(s), "
            f"submitted={capture.submitted}, revised={capture.revised}",
        )


def _handle_message(
    message, capture, callbacks,
    AssistantMessage, TextBlock, ToolUseBlock, ThinkingBlock,
) -> None:
    if not isinstance(message, AssistantMessage):
        return
    for block in getattr(message, "content", None) or []:
        if isinstance(block, TextBlock):
            text = getattr(block, "text", "") or ""
            if text:
                capture.text_chunks.append(text)
                if callbacks.vlm_stream:
                    callbacks.vlm_stream(text)
        elif ThinkingBlock is not None and isinstance(block, ThinkingBlock):
            thinking = getattr(block, "thinking", "") or ""
            if thinking and callbacks.vlm_reasoning:
                callbacks.vlm_reasoning(thinking)
        elif isinstance(block, ToolUseBlock):
            capture.tool_calls += 1
            if callbacks.progress:
                callbacks.progress(
                    _CLAUDE_NODE,
                    f"tool {capture.tool_calls}: {block.name}",
                )


# ---------------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------------


def run_claude(request: AgentRequest) -> AgentResponse:
    """Execute one run on the agentic Claude backend."""
    capture = _Capture(
        cur_start=int(request.parent_start),
        cur_end=int(request.parent_end),
    )

    asyncio.run(_run_session_async(request, capture))

    # The synth-equivalent output is the JSON Claude submitted. If the
    # session ended without submit_result, raw_response is empty and the
    # caller's parser can detect the no-submission case via verdict.
    raw_response = capture.submit_payload

    transcript = "".join(capture.text_chunks).strip()

    # Surface revise_range outcome + transcript + image bytes via attachments
    # so the caller's flow.parse() can reach them without depending on the
    # runner's internal _Capture type.
    attachments: Dict[str, Attachment] = {}
    if capture.submitted and capture.submit_summary:
        attachments["synthesizer.summary"] = Attachment(
            name="synthesizer.summary", kind="text",
            label="Claude Submission Summary", content=capture.submit_summary,
        )
    if transcript:
        attachments["claude.transcript"] = Attachment(
            name="claude.transcript", kind="text",
            label="Claude Transcript", content=transcript,
        )
    if capture.revised:
        attachments["claude.revised_range"] = Attachment(
            name="claude.revised_range", kind="structured",
            label="Revised Range",
            content={
                "start_index": capture.cur_start,
                "end_index": capture.cur_end,
                "revised_from": [request.parent_start, request.parent_end],
            },
        )

    messages = [{
        "role": _CLAUDE_NODE,
        "content": transcript or "(no text output)",
        "verdict": "submitted" if capture.submitted else "no_submission",
    }]

    return AgentResponse(
        raw_response=raw_response,
        verdict="submitted" if capture.submitted else "no_submission",
        attachments=attachments,
        step_events=capture.step_events,
        graph_images=list(capture.rendered_images),
        plan_steps=[],
        messages=messages,
    )
