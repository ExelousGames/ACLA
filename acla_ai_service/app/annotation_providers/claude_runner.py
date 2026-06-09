"""
Claude runner — one agentic Claude session per AgentRequest.

Different paradigm from the local runner: instead of a LangGraph
planner/executor/synth/eval cycle, this hands control to a single Claude
session that calls MCP tools to inspect telemetry and submit a result.
One subprocess start, multi-turn reasoning in one context.

The runner wraps the shared annotation tool surface in Claude MCP tools and
uses ``AgentRequest.planner_prompt`` as the initial user message — the
caller's intent reaches Claude there.

Box stays flow-free by exposing generic capability tools:

    list_graphs                 catalog of telemetry graphs
    get_graph_guidance          per-graph how_to_analyze blocks
    render_graph                PNG + descriptor over [start, end]
    query_telemetry             deterministic math on the df
    compute_expert_phases       per-arc entry/apex/exit ilocs
    measure_segment_shape       deterministic ST shape + altitude summary
    locate_circuit_section      named-section match for an iloc window
    find_nearest_opponent       multi-car positional context (top opponents)
    classify_opponent_interaction deterministic O / OD / MSR outcome gate
    query_opponent_trajectory   per-iloc relative trajectory for one slot
    get_circuit_id              canonical circuit id from Static_track
    revise_range                shrink/extend the working iloc range
    submit_result               capture the final structured answer + summary

Callers add domain-specific tools via
``AgentRequest.extra_state["tool_agent_extra_tools"]``. Each entry is a
``{name, description, params_schema, handler}`` dict; ``handler`` is a
callable ``(surface, args_dict) -> str | dict`` whose return is wrapped
as an MCP text result.

Whether ``revise_range`` and ``submit_result`` semantics fit the flow is
decided by the caller's planner prompt — the runner just exposes the
capability and captures whatever Claude submits.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Dict, List

from app.annotation_providers.tool_surface import (
    ANNOTATION_TOOL_DEFINITIONS,
    AnnotationToolSurface,
    ToolAgentCapture,
    annotation_tool_names,
    build_tool_agent_system_prompt,
    tool_agent_response,
    tool_agent_stage,
    tool_agent_extra_tools,
)
from app.shared.contracts import AgentRequest, AgentResponse, Attachment

LOGGER = logging.getLogger(__name__)

_CLAUDE_NODE = "claude_agent"


class ClaudeUsageExhausted(RuntimeError):
    """The Claude CLI / Agent SDK reported the account is out of usage —
    Max-plan quota hit, 5-hour window, or API credit balance depleted.
    Batch callers should halt and let the user retry later."""


_USAGE_EXHAUSTED_PATTERNS = (
    "usage limit reached",
    "usage limit",
    "5-hour limit",
    "credit balance",
    "out of credits",
    "rate limit",
    "rate_limit",
    "quota exceeded",
    "quota_exceeded",
)


def _is_usage_exhausted_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return any(p in msg for p in _USAGE_EXHAUSTED_PATTERNS)


# ---------------------------------------------------------------------------
# MCP tool registration
# ---------------------------------------------------------------------------


def _build_tool_set(surface: AnnotationToolSurface):
    """Return (mcp_server, allowed_tool_names) for the session."""
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

    tools_list = [_make_tool(defn) for defn in ANNOTATION_TOOL_DEFINITIONS]
    tool_names = [f"mcp__agent__{name}" for name in annotation_tool_names()]

    for spec in tool_agent_extra_tools(surface.request):
        wrapped = _make_extra_tool(spec, surface, tool)
        tools_list.append(wrapped)
        tool_names.append(f"mcp__agent__{spec['name']}")

    server = create_sdk_mcp_server(
        name="agent",
        version="1.0.0",
        tools=tools_list,
    )
    return server, tool_names


def _make_extra_tool(spec: Dict[str, Any], surface: AnnotationToolSurface, tool_decorator):
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
                if not isinstance(result, str):
                    result = json.dumps(result, default=str)
                text = result
                surface._emit_tool_event(name, args, text)
            else:
                _raw, text, _images = surface.call_tool(name, args or {})
        except Exception as exc:
            text = json.dumps({"error": str(exc)})
            surface._emit_tool_event(name, args, text)
        return {"content": [{"type": "text", "text": text}]}

    return _wrapped


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
    capture: ToolAgentCapture,
) -> None:
    sdk = _import_sdk_types()
    surface = AnnotationToolSurface(request, capture)
    server, tool_names = _build_tool_set(surface)

    system_prompt = build_tool_agent_system_prompt(request)
    user_message = request.planner_prompt

    options = sdk.ClaudeAgentOptions(
        model=request.config.model,
        mcp_servers={"agent": server},
        allowed_tools=tool_names,
        system_prompt=system_prompt,
        # Bound the session — generous enough for multi-step exploration,
        # tight enough to stop runaway. Caller can override via extra_state.
        max_turns=int(request.config.provider_options.get("max_turns") or 30),
    )

    cb = request.callbacks
    if cb.vlm_prompt:
        cb.vlm_prompt(user_message, tool_agent_stage(_CLAUDE_NODE, "main"))
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
    capture = ToolAgentCapture(
        node_name=_CLAUDE_NODE,
        cur_start=int(request.parent_start),
        cur_end=int(request.parent_end),
    )

    try:
        asyncio.run(_run_session_async(request, capture))
    except ClaudeUsageExhausted:
        raise
    except Exception as exc:
        if _is_usage_exhausted_error(exc):
            raise ClaudeUsageExhausted(str(exc)) from exc
        raise

    response = tool_agent_response(capture, request)
    if capture.submit_summary:
        response.attachments["synthesizer.summary"] = Attachment(
            name="synthesizer.summary",
            kind="text",
            label="Claude Submission Summary",
            content=capture.submit_summary,
        )
    transcript = "".join(capture.text_chunks).strip()
    if transcript:
        response.attachments["claude.transcript"] = Attachment(
            name="claude.transcript",
            kind="text",
            label="Claude Transcript",
            content=transcript,
        )
    if capture.revised:
        response.attachments["claude.revised_range"] = Attachment(
            name="claude.revised_range",
            kind="structured",
            label="Revised Range",
            content={
                "start_index": capture.cur_start,
                "end_index": capture.cur_end,
                "revised_from": [request.parent_start, request.parent_end],
            },
        )
    return response
