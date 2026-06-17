"""
Claude runner — one agentic Claude session per AgentRequest.

Different paradigm from the local runner: instead of a LangGraph
planner/executor/synth/eval cycle, this hands control to a single Claude
session that calls MCP tools to inspect telemetry and submit a result.
One subprocess start, multi-turn reasoning in one context.

The runner wraps the shared annotation tool surface in Claude MCP tools and
uses ``AgentRequest.planner_prompt`` as the initial user message — the
caller's intent reaches Claude there.

Box stays flow-free by exposing a compact provider-neutral surface:
recommend capabilities, execute a recommended capability by ID, retrieve
guidance / label definitions, and submit the result.

Callers add domain-specific tools via
``AgentRequest.extra_state["tool_agent_extra_tools"]``. Each entry is a
``{name, description, params_schema, handler}`` dict; ``handler`` is a
callable ``(surface, args_dict) -> str | dict`` whose return is wrapped
as an MCP text result.

The runner exposes the shared tools and captures whatever Claude submits.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Dict

from app.annotation_providers.tool_surface import (
    AnnotationToolSurface,
    ToolAgentCapture,
    annotation_tool_definitions,
    build_tool_agent_system_prompt,
    tool_agent_response,
    tool_agent_stage,
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

    tool_defs = annotation_tool_definitions(surface.request)
    tools_list = [_make_tool(defn) for defn in tool_defs]
    tool_names = [f"mcp__agent__{defn['name']}" for defn in tool_defs]

    server = create_sdk_mcp_server(
        name="agent",
        version="1.0.0",
        tools=tools_list,
    )
    return server, tool_names


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
            f"submitted={capture.submitted}",
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
    return response
