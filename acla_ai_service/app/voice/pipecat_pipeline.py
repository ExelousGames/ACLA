"""Pipecat voice-conversation pipeline factory (Phase 3).

Builds a per-WebSocket-session pipeline:

    FastAPIWebsocketTransport.input()
        → SileroVADAnalyzer (endpoint detection)
        → faster-whisper STT
        → OpenAILLMService                              -- selected OpenAI or
                                                           hosted OpenAI-
                                                           compatible model
        → KokoroTTSProcessor                            -- our Phase 2 engine
        → FastAPIWebsocketTransport.output()

The factory returns a `PipelineTask` that the WS endpoint runs via
`PipelineRunner`. Each connection gets a fresh pipeline instance, with
conversation history restored from the process-local chat session registry.

All Pipecat imports are deferred so the AI service still boots when
pipecat-ai isn't installed in the active container (e.g. a partial dev
setup). Voice WS connections fail with a clear error in that case;
HTTP endpoints continue to work.

Phase 3b additions:
    - Tool calling wired through Pipecat's `register_function` API,
      using an isolated application-tool selector before the existing browser
      and AIService._execute_function dispatch paths.

Known limitations (deferred):
    - Chat history is process-local and does not survive service restarts.
"""

from __future__ import annotations

import logging
import json
import re
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional

from app.chat_llm import resolve_chat_llm_config
from app.infra.config import settings
from app.voice.session_modes import (
    DEFAULT_CHATBOT_SESSION_MODE,
    SESSION_MODE_AGENT_BEHAVIORS,
    normalize_chatbot_session_mode,
)
from app.voice.application_tool_search_side_chat import (
    APPLICATION_TOOL_SEARCH_NAME,
    ApplicationToolSearchSideChat,
)
from app.voice.session_ai_tool_service import (
    SessionAIToolService,
    SessionToolCatalogError,
)
from app.voice.tool_relay import normalize_voice_session_context

LOGGER = logging.getLogger(__name__)

# Matches a leading [emotion_name] tag (e.g. "[vibing] ") at the start of LLM output.
_EMOTION_TAG_RE = re.compile(r'^\[([a-z]+)\]\s*')
_VALID_EMOTIONS = frozenset(["sad", "vibing", "scared", "waiting", "hearing"])
_FUNCTION_TAG_OPEN = "<function="
_FUNCTION_TAG_CLOSE = "</function>"
_FUNCTION_TAG_RE = re.compile(
    r"^<function=([A-Za-z_]\w*)>\s*(.*?)\s*</function>$",
    re.DOTALL,
)
_FRONTEND_TOOL_RESULT_TYPE = "tool_result"
_FRONTEND_TOOL_STATUS_PREFIX = "Tool status update: "
_SHARED_STARTUP_BEHAVIORS = (
    "tool_use",
    "procedure_plan",
    "emotion",
    "transcript_resilience",
)
_VALID_CHILD_AGENT_BEHAVIORS = frozenset([
    "track_guide",
    "overtake",
    "live_performance_analyst",
])


# ----------------------------------------------------------------------
# System prompt for the voice coach
# ----------------------------------------------------------------------

_VOICE_COACH_PROMPT_TEMPLATE = """You are a race engineer speaking to your driver over the radio. Stay in character.

Voice: short radio sentences, 1-3 per turn unless asked to elaborate.
No markdown, no bullets, no headings. Racing terms freely (apex,
trail-brake, kerb, slip, weight transfer, etc.).
"""

_TOOL_RESULT_HANDLING_PROMPT = """Tool result handling:
- Tools may return a status field such as running, complete, failed, blocked, or skipped.
- Treat complete or ok=true as a successful result and use the returned result/data payload.
- Treat running as not ready yet; wait for the final result instead of answering from partial data.
- Treat failed, blocked, or skipped as unavailable and explain the issue or choose another available tool.
- If no status is present, treat an error field as failed; otherwise treat the payload as a completed result.
"""

_APPLICATION_TOOL_SEARCH_PROMPT = (
    "Application tool use:\n"
    "- Your only application-tool entry point is search_application_tool.\n"
    "- Call it explicitly when application data or an application action is "
    "needed.\n"
    "- Call it without arguments. Its selector receives this complete parent "
    "conversation and the current session context.\n"
    "- Individual tool names in earlier coaching guidance describe "
    "capabilities, not parent-callable tools. Translate those instructions "
    "into a search_application_tool call; do not call or guess hidden tools "
    "directly.\n"
)


def _format_session_context_for_prompt(session_context: Optional[Dict[str, Any]]) -> str:
    normalized_context = normalize_voice_session_context(session_context)
    if not normalized_context:
        return ""
    try:
        encoded = json.dumps(normalized_context, ensure_ascii=True, sort_keys=True, default=str)
    except Exception:
        LOGGER.exception("Failed to serialize voice session context")
        return ""
    return (
        "Session context: "
        f"{encoded}\n"
        "Use this context to decide which tools are appropriate. "
    )


def _startup_agent_behavior_name(session_context: Optional[Dict[str, Any]]) -> str:
    context = normalize_voice_session_context(session_context)
    raw_mode = context.get("agent_mode")
    raw_session_mode = context.get("session_mode")

    session_mode = normalize_chatbot_session_mode(raw_session_mode)
    if session_mode is None:
        LOGGER.warning(
            "Unknown voice session_mode %r; falling back to %s",
            raw_session_mode,
            DEFAULT_CHATBOT_SESSION_MODE,
        )
        session_mode = DEFAULT_CHATBOT_SESSION_MODE

    if raw_mode is None or str(raw_mode).strip() == "":
        return SESSION_MODE_AGENT_BEHAVIORS[session_mode]

    agent_mode = str(raw_mode).strip()
    if agent_mode in _VALID_CHILD_AGENT_BEHAVIORS:
        return agent_mode

    LOGGER.warning(
        "Unknown voice agent_mode %r; falling back to session_mode %s",
        agent_mode,
        session_mode,
    )
    return SESSION_MODE_AGENT_BEHAVIORS[session_mode]


def _raw_knowledge_doc(doc: Any) -> str:
    if not isinstance(doc, dict):
        return ""
    return str(doc.get("_raw_body") or "").strip()


def _build_startup_knowledge_prompt(
    session_context: Optional[Dict[str, Any]],
) -> str:
    """Return shared + one agent-specific startup knowledge bundle."""
    from app.external_knowledge_base import (
        agent_behavior as _agent_behavior,
        behavior as _behavior,
    )

    sections: List[str] = []
    for behavior_name in _SHARED_STARTUP_BEHAVIORS:
        section = _raw_knowledge_doc(_behavior(behavior_name))
        if section:
            sections.append(section)

    agent_name = _startup_agent_behavior_name(session_context)
    agent_section = _raw_knowledge_doc(_agent_behavior(agent_name))
    if agent_section:
        sections.append(agent_section)
    else:
        LOGGER.warning("Missing startup agent behavior doc: %s", agent_name)

    return "\n\n".join(sections)


def _build_system_prompt(
    session_context: Optional[Dict[str, Any]],
) -> str:
    system_prompt = _VOICE_COACH_PROMPT_TEMPLATE
    session_context_prompt = _format_session_context_for_prompt(session_context)
    if session_context_prompt:
        system_prompt = f"{system_prompt.rstrip()}\n\n{session_context_prompt}"

    system_prompt = (
        f"{system_prompt.rstrip()}\n\n{_TOOL_RESULT_HANDLING_PROMPT}"
    )

    startup_knowledge = _build_startup_knowledge_prompt(session_context)
    if startup_knowledge:
        system_prompt = f"{system_prompt.rstrip()}\n\n{startup_knowledge}"
    return (
        f"{system_prompt.rstrip()}\n\n"
        f"{_APPLICATION_TOOL_SEARCH_PROMPT}"
    )


def _compact_json(value: Any, *, max_chars: int = 3000) -> str:
    try:
        encoded = json.dumps(value, ensure_ascii=True, sort_keys=True, default=str)
    except Exception:
        return str(value)[:max_chars]
    if len(encoded) <= max_chars:
        return encoded
    return f"{encoded[:max_chars]}...<truncated>"


def _llm_context_messages_from_user_text(text: str) -> List[Dict[str, Any]]:
    """Return LLMContext message(s) for typed user text."""
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return [{"role": "user", "content": text}]

    if not isinstance(payload, dict):
        return [{"role": "user", "content": text}]

    messages = payload.get("messages")
    if isinstance(messages, list):
        native_messages = [m for m in messages if isinstance(m, dict)]
        if native_messages and _native_message_batch_is_valid(native_messages):
            return native_messages

    role = payload.get("role")
    if isinstance(role, str) and role != "tool":
        return [payload]

    return [{"role": "user", "content": text}]


def _llm_context_messages_from_tool_result(text: str) -> List[Dict[str, Any]]:
    """Return an unmatched final result as a valid native tool-call pair."""
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return []

    if not isinstance(payload, dict):
        return []
    if payload.get("type") != _FRONTEND_TOOL_RESULT_TYPE:
        return []
    if payload.get("final") is False:
        return []

    messages = payload.get("messages")
    if isinstance(messages, list):
        native_messages = [m for m in messages if isinstance(m, dict)]
        if native_messages and _native_message_batch_is_valid(native_messages):
            return native_messages

    tool_call_id = payload.get("id")
    if not isinstance(tool_call_id, str) or not tool_call_id:
        LOGGER.warning("Dropped final frontend tool result without a call id")
        return []

    tool_status_message = _format_frontend_tool_status_for_prompt(payload)
    if not tool_status_message:
        return []

    tool_name = payload.get("name")
    if not isinstance(tool_name, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", tool_name):
        tool_name = "frontend_tool_result"

    return [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": tool_call_id,
                "type": "function",
                "function": {
                    "name": tool_name,
                    "arguments": "{}",
                },
            }],
        },
        {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": tool_status_message,
        },
    ]


def _format_frontend_tool_status_for_prompt(payload: Dict[str, Any]) -> str:
    fields = _frontend_tool_status_fields(payload)
    if not fields:
        return ""

    return f"{_FRONTEND_TOOL_STATUS_PREFIX}{_compact_json(fields)}"


def _frontend_tool_status_fields(payload: Dict[str, Any]) -> Dict[str, Any]:
    payload_type = payload.get("type")
    if payload_type != _FRONTEND_TOOL_RESULT_TYPE:
        return {}

    fields: Dict[str, Any] = {"type": payload_type}
    name = payload.get("name")
    if isinstance(name, str) and name:
        fields["name"] = name

    result = payload.get("result")
    prompt_source = result if isinstance(result, dict) else payload
    for field in ("status", "message"):
        value = prompt_source.get(field)
        if isinstance(value, str) and value:
            fields[field] = value
    if "message" not in fields:
        text = prompt_source.get("text")
        if isinstance(text, str) and text:
            fields["message"] = text
    if result is not None:
        fields["result"] = result

    return fields


def _native_message_batch_is_valid(messages: List[Dict[str, Any]]) -> bool:
    """Validate frontend-supplied native messages before adding them to context."""
    pending_tool_call_ids: set[str] = set()
    for message in messages:
        role = message.get("role")
        if role == "assistant":
            tool_calls = message.get("tool_calls")
            if isinstance(tool_calls, list):
                for tool_call in tool_calls:
                    if isinstance(tool_call, dict) and isinstance(tool_call.get("id"), str):
                        pending_tool_call_ids.add(tool_call["id"])
            continue
        if role != "tool":
            continue

        tool_call_id = message.get("tool_call_id")
        if not isinstance(tool_call_id, str) or tool_call_id not in pending_tool_call_ids:
            return False
        pending_tool_call_ids.remove(tool_call_id)
    return not pending_tool_call_ids


def _build_openai_llm_service(
    OpenAILLMService: Any,
    model: Optional[str] = None,
) -> Any:
    llm_config = resolve_chat_llm_config(model)
    return OpenAILLMService(
        base_url=llm_config.base_url,
        api_key=llm_config.api_key,
        settings=OpenAILLMService.Settings(
            model=llm_config.model,
            # Newer OpenAI chat models reject max_tokens on chat completions.
            max_completion_tokens=1000,
        ),
    )


def _build_voice_tool_surfaces(
    session_tools: Optional[List[Dict[str, Any]]],
) -> tuple[
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    frozenset[str],
]:
    """Build parent-visible and selector-visible tool catalogs."""
    tool_service = SessionAIToolService()
    session_tool_descriptors = deepcopy(session_tools or [])
    session_tool_names = frozenset(
        tool["name"] for tool in session_tool_descriptors
        if isinstance(tool.get("name"), str) and tool["name"]
    )
    application_tool_descriptors = [
        *session_tool_descriptors,
        *tool_service.get_ai_tools(),
    ]
    descriptor_names = [
        tool.get("name") for tool in application_tool_descriptors
    ]
    if (
        not all(isinstance(name, str) and name for name in descriptor_names)
        or len(set(descriptor_names)) != len(descriptor_names)
    ):
        raise SessionToolCatalogError("Voice pipeline tool names must be unique")
    return (
        tool_service.get_side_chat_tools(),
        application_tool_descriptors,
        session_tool_names,
    )


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------


@dataclass
class VoiceSessionConfig:
    """Per-connection configuration for a reconnectable chat session.

    ``session_id`` remains the optional telemetry session identifier and is
    deliberately separate from ``chat_session_id``.
    """

    chat_session_id: str
    committed_history: List[Dict[str, Any]] = field(default_factory=list)
    session_id: Optional[str] = None
    session_context: Optional[Dict[str, Any]] = None
    user_id: Optional[str] = None
    voice: Optional[str] = None  # Kokoro voice override
    chat_llm_model: Optional[str] = None


@dataclass(frozen=True)
class _SessionToolDispatch:
    """Browser dispatch awaiting a final result for its parent tool call."""

    call_id: Optional[str]


def _make_tool_handler(
    tool_executor,
    session_config: "VoiceSessionConfig",
    chat_session_id: str,
    *,
    session_tool_names: frozenset[str],
    allowed_tools: Optional[List[Dict[str, Any]]] = None,
    application_tool_search: Optional[ApplicationToolSearchSideChat] = None,
    parent_message_source: Optional[Callable[[], Iterable[Any]]] = None,
    pending_session_tool_callbacks: Optional[Dict[str, Callable[[Any], Any]]] = None,
):
    """Build a per-session selector handler with two-bucket dispatch.

    The parent-visible selector resolves one allowed call, then:

    * Names in ``session_tool_names`` (retrieved from the backend)
      → forwarded to the browser through the active transport bound to
      ``chat_session_id``.
    * AI-owned names → forwarded to ``tool_executor`` (server-side path,
      typically ``AIService._execute_function``).

    Both paths pass tool returns through unchanged so the LLM sees exactly
    what the browser or server-side executor returned.

    Browser-owned calls contain only relay routing data.
    """
    from app.voice.tool_relay import get_relay

    relay = get_relay()
    application_tools = deepcopy(allowed_tools or [])
    application_tool_names = frozenset(
        descriptor.get("name")
        for descriptor in application_tools
        if isinstance(descriptor, dict)
        and isinstance(descriptor.get("name"), str)
        and descriptor["name"]
    )

    async def send_session_tool(function_name: str, arguments: Dict[str, Any]) -> Optional[str]:
        """Send one browser-owned session tool call and return without waiting."""
        arguments = arguments or {}

        LOGGER.info("[SESSION-TOOL-CALL] name=%s args=%r", function_name, arguments)
        call_id = await relay.send_tool_call(
            chat_session_id,
            function_name,
            arguments,
        )
        LOGGER.info(
            "[SESSION-TOOL-DISPATCHED] name=%s ok=%s call_id=%r",
            function_name, bool(call_id), call_id,
        )
        return call_id

    async def dispatch_server_tool(function_name: str, arguments: Dict[str, Any]) -> Any:
        """Execute one tool by name and return the LLM-visible payload.

        Shared by native Pipecat ``register_function`` calls. Routes session
        vs. server tools, leaves the tool return unchanged, and logs the
        result. Never raises —
        failures come back as ``{"error": ...}`` so Pipecat can hand the
        result back cleanly.
        """
        arguments = arguments or {}

        LOGGER.info("[TOOL-CALL] name=%s args=%r", function_name, arguments)

        ok = True
        error_msg: Optional[str] = None
        try:
            if function_name == APPLICATION_TOOL_SEARCH_NAME:
                if application_tool_search is None:
                    raise RuntimeError("Application tool search is unavailable")
                if parent_message_source is None:
                    raise RuntimeError("Parent session content is unavailable")

                selected = await application_tool_search.run({
                    "parent_messages": deepcopy(list(parent_message_source())),
                    "session_context": deepcopy(normalize_voice_session_context(
                        session_config.session_context,
                    )),
                    "allowed_tools": deepcopy(application_tools),
                })
                selected_name = selected.get("name")
                selected_arguments = selected.get("arguments")
                if selected_name not in application_tool_names:
                    raise RuntimeError(
                        f"Application tool selector returned unknown tool: {selected_name}",
                    )
                if not isinstance(selected_arguments, dict):
                    raise RuntimeError(
                        "Application tool selector arguments must be a JSON object",
                    )
                if selected_name in session_tool_names:
                    call_id = await send_session_tool(selected_name, selected_arguments)
                    return _SessionToolDispatch(call_id)
                return await dispatch_server_tool(
                    selected_name,
                    selected_arguments,
                )

            if function_name in session_tool_names:
                # dispatch() never raises — failures come back as {"error": ...}.
                raise RuntimeError("session tool reached server dispatcher")
            else:
                # Server-side path. Context carries the connect-time IDs;
                # track/car are intentionally absent (LLM fetches via tool).
                context = {
                    "session_id": session_config.session_id,
                    "session_context": session_config.session_context,
                    "user_id": session_config.user_id,
                    "_chat_session_id": chat_session_id,
                }
                result = await tool_executor(function_name, arguments, context)
        except Exception as exc:
            LOGGER.exception("Voice tool %s failed", function_name)
            error_msg = str(exc)
            return {"error": error_msg}

        payload = result
        if isinstance(payload, dict) and "error" in payload:
            ok = False
            error_msg = str(payload.get("error"))

        # Truncate large payloads for log readability.
        _payload_log = payload
        if isinstance(_payload_log, str) and len(_payload_log) > 400:
            _payload_log = _payload_log[:400] + f"... [+{len(_payload_log)-400} chars]"
        LOGGER.info(
            "[TOOL-RESULT] name=%s ok=%s error=%r payload=%r",
            function_name, ok, error_msg, _payload_log,
        )
        return payload

    async def handle_tool_call(params):
        if params.function_name in session_tool_names:
            call_id = await send_session_tool(params.function_name, params.arguments or {})
            if call_id and pending_session_tool_callbacks is not None:
                pending_session_tool_callbacks[call_id] = params.result_callback
            return
        payload = await dispatch_server_tool(params.function_name, params.arguments or {})
        if isinstance(payload, _SessionToolDispatch):
            if payload.call_id and pending_session_tool_callbacks is not None:
                pending_session_tool_callbacks[payload.call_id] = params.result_callback
            return
        await params.result_callback(payload)

    return handle_tool_call, send_session_tool, dispatch_server_tool


def _split_function_tag_prefix(text: str) -> tuple[str, str]:
    """Keep a trailing partial '<function=' prefix buffered across chunks."""
    max_len = min(len(text), len(_FUNCTION_TAG_OPEN) - 1)
    for size in range(max_len, 0, -1):
        suffix = text[-size:]
        if _FUNCTION_TAG_OPEN.startswith(suffix):
            return text[:-size], suffix
    return text, ""


def _build_function_tag_recovery():
    """Strip and dispatch Llama-style text-channel function tags.

    Some OpenAI-compatible local models occasionally emit
    ``<function=name>{...}</function>`` as text instead of using native
    ``tool_calls``. This processor sits before transcript/TTS so those tags
    are never shown or spoken, and sends them through the same dispatch path
    as a native tool call. Empty argument bodies are treated as ``{}``.
    """
    import asyncio
    import json as _json
    import uuid as _uuid
    from pipecat.frames.frames import (
        Frame,
        LLMFullResponseEndFrame,
        LLMFullResponseStartFrame,
        TextFrame,
    )
    from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

    class FunctionTagRecovery(FrameProcessor):
        def __init__(
            self,
            send_session_tool,
            dispatch_server_tool,
            session_tool_names: frozenset[str],
            parent_tool_names: frozenset[str],
            context: Any,
            get_task,
        ) -> None:
            super().__init__()
            self._send_session_tool = send_session_tool
            self._dispatch_server_tool = dispatch_server_tool
            self._session_tool_names = session_tool_names
            self._parent_tool_names = parent_tool_names
            self._context = context
            self._get_task = get_task
            self._buf = ""

        async def process_frame(self, frame: "Frame", direction: "FrameDirection") -> None:
            await super().process_frame(frame, direction)

            if isinstance(frame, LLMFullResponseStartFrame):
                self._buf = ""
                await self.push_frame(frame, direction)
                return

            if isinstance(frame, TextFrame):
                self._buf += getattr(frame, "text", "") or ""
                await self._drain(direction, final=False)
                return

            if isinstance(frame, LLMFullResponseEndFrame):
                await self._drain(direction, final=True)
                await self.push_frame(frame, direction)
                return

            await self.push_frame(frame, direction)

        async def _drain(self, direction: "FrameDirection", *, final: bool) -> None:
            while self._buf:
                idx = self._buf.find(_FUNCTION_TAG_OPEN)
                if idx < 0:
                    if final:
                        await self._push_text(self._buf, direction)
                        self._buf = ""
                        return
                    emit, hold = _split_function_tag_prefix(self._buf)
                    if emit:
                        await self._push_text(emit, direction)
                    self._buf = hold
                    return

                if idx > 0:
                    await self._push_text(self._buf[:idx], direction)
                    self._buf = self._buf[idx:]

                close = self._buf.find(_FUNCTION_TAG_CLOSE)
                if close < 0:
                    if final:
                        LOGGER.warning(
                            "FunctionTagRecovery: incomplete tag dropped: %r",
                            self._buf,
                        )
                        self._buf = ""
                    return

                tag_end = close + len(_FUNCTION_TAG_CLOSE)
                full_tag = self._buf[:tag_end]
                self._buf = self._buf[tag_end:]
                match = _FUNCTION_TAG_RE.match(full_tag)
                if not match:
                    LOGGER.warning(
                        "FunctionTagRecovery: malformed tag, passing through as text: %r",
                        full_tag,
                    )
                    await self._push_text(full_tag, direction)
                    continue

                name = match.group(1)
                raw_args = (match.group(2) or "").strip()
                args: Dict[str, Any] = {}
                if raw_args:
                    try:
                        parsed = _json.loads(raw_args)
                    except _json.JSONDecodeError:
                        LOGGER.warning(
                            "FunctionTagRecovery: bad JSON args for <function=%s>: %r",
                            name, raw_args,
                        )
                        continue
                    if not isinstance(parsed, dict):
                        LOGGER.warning(
                            "FunctionTagRecovery: non-object args for <function=%s>: %r",
                            name, parsed,
                        )
                        continue
                    args = parsed

                LOGGER.info(
                    "[TOOL-CALL-RECOVERED] name=%s args=%r "
                    "(from text-channel <function=...> tag)",
                    name, args,
                )
                asyncio.create_task(self._recover(name, args))

        async def _push_text(self, text: str, direction: "FrameDirection") -> None:
            if text:
                await self.push_frame(TextFrame(text=text), direction)

        async def _recover(self, name: str, args: Dict[str, Any]) -> None:
            if name not in self._parent_tool_names:
                LOGGER.warning(
                    "FunctionTagRecovery: ignored unavailable parent tool %s",
                    name,
                )
                return
            if name in self._session_tool_names:
                await self._send_session_tool(name, args)
                return

            result = await self._dispatch_server_tool(name, args)
            if isinstance(result, _SessionToolDispatch):
                return
            try:
                call_id = f"recovered_{_uuid.uuid4().hex}"
                self._context.add_message({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": name,
                            "arguments": _json.dumps(args),
                        },
                    }],
                })
                self._context.add_message({
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": _json.dumps(result),
                })
            except Exception:
                LOGGER.exception("FunctionTagRecovery: context injection failed")
                return

            task = self._get_task()
            if task is None:
                LOGGER.warning(
                    "FunctionTagRecovery: no PipelineTask bound; tool result "
                    "is in context but model won't speak until next user turn"
                )
                return

            try:
                from pipecat.frames.frames import LLMRunFrame
                await task.queue_frame(LLMRunFrame())
            except Exception:
                LOGGER.exception("FunctionTagRecovery: failed to queue LLMRunFrame")

    return FunctionTagRecovery


def _build_transcript_observer():
    """Construct a pass-through FrameProcessor that emits transcript text
    frames over the bound WebSocket.

    Two roles:
      * ``role="user"`` — emits ``user_transcript`` on each final
        :class:`TranscriptionFrame` from STT.
      * ``role="assistant"`` — buffers :class:`TextFrame` chunks between
        ``LLMFullResponseStartFrame`` and ``LLMFullResponseEndFrame`` and
        emits one ``assistant_transcript`` per turn.

    All frames pass through unchanged — this processor only mirrors transcripts.
    """
    import json as _json
    from pipecat.frames.frames import (
        Frame,
        LLMFullResponseEndFrame,
        LLMFullResponseStartFrame,
        TextFrame,
        TranscriptionFrame,
    )
    from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

    class TranscriptObserver(FrameProcessor):
        def __init__(self, send_text, role: str) -> None:
            super().__init__()
            self._send_text = send_text
            self._role = role
            self._assistant_buf: list[str] = []

        async def process_frame(self, frame: "Frame", direction: "FrameDirection") -> None:
            await super().process_frame(frame, direction)

            try:
                if self._role == "user" and isinstance(frame, TranscriptionFrame):
                    text = (getattr(frame, "text", "") or "").strip()
                    if text:
                        await self._emit("user_transcript", text)
                elif self._role == "assistant":
                    if isinstance(frame, LLMFullResponseStartFrame):
                        self._assistant_buf = []
                    elif isinstance(frame, TextFrame):
                        chunk = getattr(frame, "text", "") or ""
                        if chunk:
                            self._assistant_buf.append(chunk)
                    elif isinstance(frame, LLMFullResponseEndFrame):
                        raw = "".join(self._assistant_buf).strip()
                        self._assistant_buf = []
                        if raw:
                            m = _EMOTION_TAG_RE.match(raw)
                            emotion = m.group(1) if m and m.group(1) in _VALID_EMOTIONS else None
                            text = raw[m.end():] if emotion else raw
                            await self._emit("assistant_transcript", text, emotion=emotion)
            except Exception:
                LOGGER.exception("TranscriptObserver: emit failed (role=%s)", self._role)

            await self.push_frame(frame, direction)

        async def _emit(self, kind: str, text: str, *, emotion: Optional[str] = None) -> None:
            payload: Dict[str, Any] = {"type": kind, "text": text}
            if emotion:
                payload["emotion"] = emotion
            try:
                await self._send_text(_json.dumps(payload))
            except Exception:
                LOGGER.debug("%s emit failed (WS likely closed)", kind, exc_info=True)

    return TranscriptObserver


def _build_emotion_tag_stripper():
    """Strips leading [emotion] tags from LLM TextFrame output before TTS.

    Sits between the LLM and Kokoro in the pipeline. Buffers the very first
    chunk(s) until it can determine whether the response starts with a valid
    emotion tag, then either strips it or flushes the buffer unchanged.

    This keeps Kokoro from ever speaking "[vibing]".
    """
    from pipecat.frames.frames import (
        Frame,
        LLMFullResponseStartFrame,
        TextFrame,
    )
    from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

    class EmotionTagStripper(FrameProcessor):
        def __init__(self) -> None:
            super().__init__()
            self._buf: str = ""
            self._checking: bool = True

        async def process_frame(self, frame: "Frame", direction: "FrameDirection") -> None:
            await super().process_frame(frame, direction)

            if isinstance(frame, LLMFullResponseStartFrame):
                self._buf = ""
                self._checking = True
                await self.push_frame(frame, direction)
                return

            # Once the tag determination is made, pass everything through.
            if not isinstance(frame, TextFrame) or not self._checking:
                await self.push_frame(frame, direction)
                return

            chunk = getattr(frame, "text", "") or ""
            if not chunk:
                return

            self._buf += chunk
            m = _EMOTION_TAG_RE.match(self._buf)
            if m and m.group(1) in _VALID_EMOTIONS:
                self._checking = False
                remainder = self._buf[m.end():]
                self._buf = ""
                if remainder:
                    await self.push_frame(TextFrame(text=remainder), direction)
                return

            # Clearly no tag (first char not '[', or buffer too long) — flush and stop.
            if not self._buf.startswith("[") or len(self._buf) > 25:
                self._checking = False
                await self.push_frame(TextFrame(text=self._buf), direction)
                self._buf = ""

    return EmotionTagStripper


def _build_context_logger():
    """Diagnostic FrameProcessor: dumps LLMContext messages on each LLM turn.

    Logs at INFO level under `[CTX-DUMP]` whenever the LLM starts producing
    a response — at that moment ``context.messages`` is exactly what was
    sent to llama-server, so we can see prior assistant turns, tool calls,
    and tool results in the order the model saw them.
    """
    from pipecat.frames.frames import Frame, LLMFullResponseStartFrame
    from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

    class ContextLogger(FrameProcessor):
        def __init__(self, context: Any) -> None:
            super().__init__()
            self._context = context

        async def process_frame(self, frame: "Frame", direction: "FrameDirection") -> None:
            await super().process_frame(frame, direction)

            if isinstance(frame, LLMFullResponseStartFrame):
                try:
                    msgs = list(getattr(self._context, "messages", []) or [])
                    LOGGER.info("[CTX-DUMP] LLM responding — %d messages in context", len(msgs))
                    for i, m in enumerate(msgs):
                        if not isinstance(m, dict):
                            LOGGER.info("[CTX-DUMP]   [%d] %r", i, m)
                            continue
                        role = m.get("role")
                        content = m.get("content")
                        if isinstance(content, str) and len(content) > 300:
                            content = content[:300] + f"... [+{len(content) - 300} chars]"
                        tool_calls = m.get("tool_calls")
                        tool_call_id = m.get("tool_call_id")
                        LOGGER.info(
                            "[CTX-DUMP]   [%d] role=%s content=%r tool_calls=%s tool_call_id=%s",
                            i, role, content, bool(tool_calls), tool_call_id,
                        )
                except Exception:
                    LOGGER.exception("[CTX-DUMP] dump failed")

            await self.push_frame(frame, direction)

    return ContextLogger


def _build_initial_context_messages(
    session_config: VoiceSessionConfig,
) -> tuple[List[Dict[str, Any]], int]:
    """Build a fresh connection root followed by stored conversation history."""
    session_config.session_context = normalize_voice_session_context(
        session_config.session_context,
    )
    system_prompt = _build_system_prompt(session_config.session_context)
    history = [
        deepcopy(message)
        for message in session_config.committed_history
        if isinstance(message, dict)
    ]
    return ([{"role": "system", "content": system_prompt}, *history], len(history))


def _committed_history_from_messages(
    messages: Iterable[Any],
    initial_history_length: int,
) -> List[Dict[str, Any]]:
    """Remove the root prompt and any new turn lacking a final assistant reply."""
    conversation = [
        deepcopy(message)
        for message in list(messages)[1:]
        if isinstance(message, dict)
    ]
    prior_count = min(max(initial_history_length, 0), len(conversation))
    prior_history = conversation[:prior_count]
    current_messages = conversation[prior_count:]

    last_complete_assistant = -1
    for index, message in enumerate(current_messages):
        if (
            message.get("role") == "assistant"
            and message.get("content")
            and not message.get("tool_calls")
        ):
            last_complete_assistant = index

    if last_complete_assistant < 0:
        return prior_history
    return prior_history + current_messages[:last_complete_assistant + 1]


async def build_voice_pipeline_task(
    websocket: Any,
    session_config: VoiceSessionConfig,
    tool_executor: Any,
    *,
    session_tools: Optional[List[Dict[str, Any]]] = None,
):
    """Build a Pipecat PipelineTask bound to the given WebSocket.

    Returns the task; caller is responsible for running it via
    `PipelineRunner.run(task)`.

    Side effect: registers the WebSocket with :mod:`app.voice.tool_relay`
    so session tool calls and tool payloads routed via text frames
    reach this session's LLM context. The caller (api/voice.py) is
    responsible for unbinding on session end.

    Raises ImportError if pipecat-ai or faster-whisper aren't available
    — the caller should map this to an explicit error frame to the WS.
    """
    # Deferred imports — see module docstring.
    import asyncio
    from pipecat.adapters.schemas.function_schema import FunctionSchema
    from pipecat.adapters.schemas.tools_schema import ToolsSchema
    from pipecat.audio.vad.silero import SileroVADAnalyzer
    from pipecat.frames.frames import LLMRunFrame
    from pipecat.pipeline.pipeline import Pipeline
    from pipecat.pipeline.task import PipelineParams, PipelineTask
    from pipecat.processors.aggregators.llm_context import LLMContext
    from pipecat.processors.aggregators.llm_response_universal import (
        LLMContextAggregatorPair,
    )
    from pipecat.processors.audio.vad_processor import VADProcessor
    from pipecat.services.openai.llm import OpenAILLMService
    from pipecat.services.whisper.stt import WhisperSTTService
    from pipecat.transports.websocket.fastapi import (
        FastAPIWebsocketParams,
        FastAPIWebsocketTransport,
    )

    from app.voice.chat_sessions import get_chat_session_registry
    from app.voice.pipecat_kokoro import build_kokoro_processor
    from app.voice.raw_pcm_serializer import RawPCMSerializer
    from app.voice.tool_relay import get_relay

    TranscriptObserver = _build_transcript_observer()
    EmotionTagStripper = _build_emotion_tag_stripper()
    FunctionTagRecovery = _build_function_tag_recovery()
    ContextLogger = _build_context_logger()

    LOGGER.info(
        "Building voice pipeline (chat_session=%s telemetry_session=%s user=%s)",
        session_config.chat_session_id,
        session_config.session_id,
        session_config.user_id,
    )

    # --- Transport ---
    # The FastAPI websocket transport runs the WS lifecycle (accept, recv,
    # send, close) inside our existing endpoint.
    #
    # WIRE FORMAT (must match the frontend in use-voice-conversation.ts):
    #   - serializer=RawPCMSerializer(): client sends/receives raw PCM16 mono
    #     bytes. Pipecat 1.2.1's input transport silently drops every inbound
    #     frame when serializer is None (see RawPCMSerializer docstring), so
    #     we need a trivial pass-through serializer to actually feed mic bytes
    #     into the pipeline as InputAudioRawFrame.
    #   - audio_in_sample_rate=16000: frontend captures at 16kHz to match
    #     Whisper's preferred rate. AudioContext({ sampleRate: 16000 }).
    #   - audio_out_sample_rate=24000: Kokoro's native rate. The frontend's
    #     playback AudioContext uses 24kHz so no client-side resampling needed.
    #
    # NOTE: VAD is NOT configured on the transport in Pipecat 1.2.1 — the
    # `vad_enabled / vad_analyzer / vad_audio_passthrough` fields were removed
    # from TransportParams (Pydantic silently drops unknown kwargs, which
    # masked this for a while). VAD now lives as a dedicated VADProcessor
    # inserted into the pipeline below.
    transport = FastAPIWebsocketTransport(
        websocket=websocket,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            audio_in_sample_rate=16000,
            audio_out_sample_rate=settings.kokoro_sample_rate,
            add_wav_header=False,
            serializer=RawPCMSerializer(),
        ),
    )

    # --- VAD (Silero, in-pipeline) ---
    # Emits VADUserStartedSpeakingFrame / VADUserStoppedSpeakingFrame which
    # the downstream STT uses to gate Whisper inference.
    vad_processor = VADProcessor(vad_analyzer=SileroVADAnalyzer())

    # --- STT (faster-whisper) ---
    # Use the multilingual large-v3-turbo model on whichever device is
    # available. Pipecat's WhisperSTTService wraps faster-whisper directly.
    stt = WhisperSTTService(
        model="large-v3-turbo",
        # device="cuda" if available; Pipecat auto-detects via faster-whisper.
    )

    # --- LLM (remote OpenAI-compatible client) ---
    llm = _build_openai_llm_service(
        OpenAILLMService,
        session_config.chat_llm_model,
    )

    # --- Tool calling (Phase 3b) ---
    # The parent LLM sees only the application-tool selector. Its isolated
    # side chat receives the union of backend-retrieved session tools (browser
    # relay) and AI-owned knowledge tools (server executor).
    (
        parent_tool_descriptors,
        application_tool_descriptors,
        session_tool_names,
    ) = _build_voice_tool_surfaces(session_tools)
    tool_schemas = [
        FunctionSchema(**descriptor) for descriptor in parent_tool_descriptors
    ]
    tools = ToolsSchema(standard_tools=tool_schemas)

    # Build the live parent context before the selector handler so each search
    # can snapshot every message accumulated up to that tool call.
    initial_messages, initial_history_length = _build_initial_context_messages(
        session_config,
    )
    context = LLMContext(
        messages=initial_messages,
        tools=tools,
    )
    context_aggregator = LLMContextAggregatorPair(context)

    application_tool_search = ApplicationToolSearchSideChat.from_chat_llm_model(
        session_config.chat_llm_model,
    )

    pending_session_tool_callbacks: Dict[str, Callable[[Any], Any]] = {}
    tool_handler, send_session_tool, dispatch_server_tool = _make_tool_handler(
        tool_executor,
        session_config,
        chat_session_id=session_config.chat_session_id,
        session_tool_names=session_tool_names,
        allowed_tools=application_tool_descriptors,
        application_tool_search=application_tool_search,
        parent_message_source=lambda: getattr(context, "messages", []) or [],
        pending_session_tool_callbacks=pending_session_tool_callbacks,
    )
    for schema in tool_schemas:
        llm.register_function(schema.name, tool_handler)

    # System message + context object (Pipecat's history aggregator). The
    # engineer prompt has no {track}/{car} placeholders — the LLM doesn't
    # carry that state; it responds to what the driver says.
    #
    # Startup behavior docs live in editable .md files. Each new socket gets
    # shared chatbot rules plus exactly one agent-specific role document.
    # --- TTS (Kokoro via custom Pipecat processor) ---
    KokoroProcessor = build_kokoro_processor()
    tts = KokoroProcessor(sample_rate=settings.kokoro_sample_rate)

    # --- Transcript observers ---
    # Two pass-through processors that emit `user_transcript` /
    # `assistant_transcript` text frames on the same WS so the chat UI
    # can display the conversation as text alongside the audio.
    async def _send_text(payload: str) -> None:
        await websocket.send_text(payload)

    user_transcript_observer = TranscriptObserver(send_text=_send_text, role="user")
    assistant_transcript_observer = TranscriptObserver(send_text=_send_text, role="assistant")
    emotion_tag_stripper = EmotionTagStripper()
    context_logger = ContextLogger(context)
    task_ref: Dict[str, Any] = {"task": None}
    function_tag_recovery = FunctionTagRecovery(
        send_session_tool,
        dispatch_server_tool,
        session_tool_names,
        frozenset(schema.name for schema in tool_schemas),
        context,
        lambda: task_ref["task"],
    )

    # --- Pipeline composition ---
    # VAD sits between the transport input and STT so Whisper only runs
    # on actual speech windows (gated by VADUserStartedSpeakingFrame /
    # VADUserStoppedSpeakingFrame from the VADProcessor).
    #
    # user_transcript_observer sits AFTER stt so it sees the final
    # TranscriptionFrame before context_aggregator.user() consumes it.
    # function_tag_recovery catches local-model text-channel function tags
    # before they can reach transcript/TTS. Native tool calls still use
    # Pipecat's registered function channel.
    # emotion_tag_stripper sits AFTER the observer and BEFORE tts so Kokoro
    # never receives the [emotion] tag.
    # context_aggregator.assistant() is the LAST processor (canonical
    # Pipecat placement). It consumes TextFrame/LLMFullResponse{Start,End}Frame
    # to commit spoken assistant turns to LLMContext. Requires every upstream
    # processor (including KokoroTTSProcessor) to FORWARD TextFrames after
    # consuming them for their own purposes — otherwise the aggregator sees
    # empty turns and the model can't see what it just said.
    pipeline = Pipeline([
        transport.input(),
        vad_processor,
        stt,
        user_transcript_observer,
        context_aggregator.user(),
        llm,
        context_logger,
        function_tag_recovery,
        assistant_transcript_observer,
        emotion_tag_stripper,
        tts,
        transport.output(),
        context_aggregator.assistant(),
    ])

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            allow_interruptions=True,
            enable_metrics=False,
        ),
    )
    task._acla_llm_context = context
    task._acla_initial_history_length = initial_history_length
    task_ref["task"] = task
    # --- Text control sinks -------------------------------------------------
    # Tool results/errors are serialized by the relay and sent through a
    # dedicated sink so typed chat and session tool payloads stay separate.
    loop = asyncio.get_running_loop()

    def _trigger_llm_run(source: str) -> None:
        try:
            loop.create_task(task.queue_frame(LLMRunFrame()))
        except Exception:
            LOGGER.exception("%s: could not trigger LLM run", source)

    def _remember_session_context(session_context: Dict[str, Any]) -> None:
        normalized_context = normalize_voice_session_context(session_context)
        session_config.session_context = normalized_context
        get_chat_session_registry().update_session_context(
            session_config.chat_session_id,
            normalized_context,
        )

    def user_text_sink(text: str) -> None:
        """Inject typed chat text."""
        import time as _time
        LOGGER.info("[LAT-DIAG] user_text_in t=%.3f chars=%d", _time.monotonic(), len(text))
        for message in _llm_context_messages_from_user_text(text):
            context.add_message(message)
        _trigger_llm_run("user_text_sink")

    def tool_result_sink(text: str) -> None:
        """Inject a session tool response.

        Tool-result frames can include browser-supplied native messages for
        the LLM context.
        """
        import time as _time
        LOGGER.info("[LAT-DIAG] tool_result_in t=%.3f chars=%d", _time.monotonic(), len(text))
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            LOGGER.warning("Dropped malformed frontend tool result")
            return
        if not isinstance(payload, dict) or payload.get("type") != _FRONTEND_TOOL_RESULT_TYPE:
            LOGGER.warning("Dropped invalid frontend tool result payload")
            return
        if payload.get("final") is False:
            return

        call_id = payload.get("id")
        result_callback = (
            pending_session_tool_callbacks.pop(call_id, None)
            if isinstance(call_id, str)
            else None
        )
        if result_callback is not None:
            async def deliver_result() -> None:
                try:
                    await result_callback(payload.get("result"))
                except Exception:
                    LOGGER.exception(
                        "tool_result_sink: could not complete tool call %s",
                        call_id,
                    )

            loop.create_task(deliver_result())
            return

        messages = _llm_context_messages_from_tool_result(text)
        if not messages:
            return
        for message in messages:
            context.add_message(message)
        _trigger_llm_run("tool_result_sink")

    get_relay().bind(
        session_config.chat_session_id,
        send_text=_send_text,
        user_text_sink=user_text_sink,
        tool_result_sink=tool_result_sink,
        session_context_sink=_remember_session_context,
    )
    start_control_pump = getattr(websocket, "start_text_control_pump", None)
    if callable(start_control_pump):
        start_control_pump()

    return task


async def run_voice_session(
    websocket: Any,
    session_config: VoiceSessionConfig,
    tool_executor: Any,
    *,
    session_tools: Optional[List[Dict[str, Any]]] = None,
) -> None:
    """Bind a Pipecat pipeline to `websocket` and run it to completion.

    Returns when the WS closes or the pipeline exits. Caller is responsible
    for any auth/lifecycle concerns around `websocket`, supplying a
    ``tool_executor`` (typically AIService._execute_function), and passing
    backend-retrieved ``session_tools`` (see :mod:`app.api.voice`).

    On exit, committed context is copied back to the chat session registry,
    the active transport is unbound, and the session becomes resumable.
    """
    # Deferred imports.
    from pipecat.pipeline.runner import PipelineRunner
    from app.voice.chat_sessions import get_chat_session_registry
    from app.voice.tool_relay import get_relay

    task = None
    try:
        task = await build_voice_pipeline_task(
            websocket, session_config, tool_executor,
            session_tools=session_tools,
        )
        runner = PipelineRunner()
        await runner.run(task)
    finally:
        stop_control_pump = getattr(websocket, "stop_text_control_pump", None)
        if callable(stop_control_pump):
            try:
                await stop_control_pump()
            except Exception:
                LOGGER.exception(
                    "Could not stop voice control pump (chat_session=%s)",
                    session_config.chat_session_id,
                )
        get_relay().unbind(session_config.chat_session_id)

        committed_history = None
        context = getattr(task, "_acla_llm_context", None)
        if context is not None:
            try:
                committed_history = _committed_history_from_messages(
                    getattr(context, "messages", []) or [],
                    getattr(task, "_acla_initial_history_length", 0),
                )
            except Exception:
                LOGGER.exception(
                    "Could not snapshot voice chat history (chat_session=%s)",
                    session_config.chat_session_id,
                )

        get_chat_session_registry().detach(
            session_config.chat_session_id,
            committed_history,
            session_config.session_context,
        )
        LOGGER.info(
            "Voice session ended (chat_session=%s user=%s)",
            session_config.chat_session_id,
            session_config.user_id,
        )
