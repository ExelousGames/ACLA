"""Pipecat voice-conversation pipeline factory (Phase 3).

Builds a per-WebSocket-session pipeline:

    FastAPIWebsocketTransport.input()
        → SileroVADAnalyzer (endpoint detection)
        → faster-whisper STT
        → OpenAILLMService                              -- local llama-server,
                                                           or hosted OpenAI-
                                                           compatible endpoint
                                                           if HOSTED_LLM_BASE_URL
                                                           is set
        → KokoroTTSProcessor                            -- our Phase 2 engine
        → FastAPIWebsocketTransport.output()

The factory returns a `PipelineTask` that the WS endpoint runs via
`PipelineRunner`. Each connection gets its own pipeline instance, so
conversation history is isolated.

All Pipecat imports are deferred so the AI service still boots when
pipecat-ai isn't installed in the active container (e.g. a partial dev
setup). Voice WS connections fail with a clear error in that case;
HTTP endpoints continue to work.

Phase 3b additions:
    - Tool calling wired through Pipecat's `register_function` API,
      delegating to AIService._execute_function so voice and text share
      the same tool implementations.

Known limitations (deferred):
    - Side products from tools (e.g. _guidance_enabled, _track_corner_data)
      are LOGGED but not surfaced over the WS — the voice path has no UI
      side-channel. Voice users hear spoken guidance but won't trigger
      the in-chat track-guide UI overlay.
    - No per-user conversation history persistence — each WS = fresh context.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

from app.infra.config import settings

LOGGER = logging.getLogger(__name__)

# Matches a leading [emotion_name] tag (e.g. "[vibing] ") at the start of LLM output.
_EMOTION_TAG_RE = re.compile(r'^\[([a-z]+)\]\s*')
_VALID_EMOTIONS = frozenset(["sad", "vibing", "scared", "waiting", "hearing"])


# ----------------------------------------------------------------------
# System prompt for the voice coach
# ----------------------------------------------------------------------

_VOICE_COACH_PROMPT_TEMPLATE = """You are a race engineer speaking to your driver over the radio. Stay in character.

Voice: short radio sentences, 1-3 per turn unless asked to elaborate.
No markdown, no bullets, no headings. Racing terms freely (apex,
trail-brake, kerb, slip, weight transfer, etc.).

Tool use:
- Only call a tool when the question needs data you don't have.
- General concept questions ("what is trail braking?") — answer in 2-3
  sentences, no tool.
- Don't offer to do things — either call the tool now, or say you can't
  and stop. No "would you like…", no "shall I…", no pivoting to a
  different track or topic the driver didn't ask about.
- When analyze_telemetry returns labels with definitions and remedies,
  pick the 1-2 that matter most and weave them into a natural comment.
  Don't read the whole catalog aloud.

Output rules:
- If a tool errors or telemetry is down, say so plainly ("can't see your
  telemetry right now"). Never fabricate numbers or label names.
- Translate label codes to natural English before speaking.
"""


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------


@dataclass
class VoiceSessionConfig:
    """Per-WS-session configuration.

    The WS connection takes only ``session_id`` and ``user_id`` as query
    params — anything else the LLM wants (current track/car, lap data,
    recent telemetry, etc.) is fetched on demand via tool calls. See the
    plan's "everything is pulled on demand" principle.
    """

    session_id: Optional[str] = None
    user_id: Optional[str] = None
    voice: Optional[str] = None  # Kokoro voice override


# Human-readable titles shown in the chat UI for each tool. The driver
# sees these in a "tool box" while the LLM is calling the function — they
# should read like a brief status line, not the raw function name.
#
# Frontend-tool titles arrive over the WS handshake; this map is the
# server-side fallback for the server-implemented tools only.
_SERVER_TOOL_TITLES: Dict[str, str] = {
    "analyze_telemetry": "Analyzing telemetry",
    "explain_label": "Looking up the term",
    "get_track_knowledge": "Pulling track notes",
    "search_racing_knowledge": "Searching racing knowledge",
}


def _prettify(name: str) -> str:
    return name.replace("_", " ").strip().capitalize()


def _build_server_tool_schemas(query_scope_schema: Optional[Dict[str, Any]]):
    """Pipecat FunctionSchemas for the server-implemented tools only.

    Frontend tools come in over the WS handshake (see :mod:`app.api.voice`)
    and are built in :func:`_build_frontend_tool_schemas`. Together they
    form the LLM's tool surface. Deferred import — only loaded when a voice
    session is actually built.

    ``query_scope_schema`` is the frontend-owned JSON Schema for QueryScope;
    server-side tools whose params reference a scope (``analyze_telemetry``)
    consume it from the handshake — the AI service must NOT declare its own
    copy. Missing → hard error (no silent fallback to a Python-defined shape).
    """
    if not query_scope_schema:
        raise ValueError(
            "_build_server_tool_schemas: query_scope_schema is required "
            "(must come from the WS frontend_info handshake)"
        )

    from pipecat.adapters.schemas.function_schema import FunctionSchema

    return [
        FunctionSchema(
            name="analyze_telemetry",
            description=(
                "Classify driving actions over a scope; returns engineer "
                "labels with definitions and remedies. Use for 'what just "
                "happened', 'why X', 'how was lap N'."
            ),
            properties={"scope": query_scope_schema},
            required=["scope"],
        ),
        FunctionSchema(
            name="explain_label",
            description="Definition, interpretation, and remedies for one action label.",
            properties={
                "label_id": {
                    "type": "string",
                    "description": "Label id ('MS44') or natural name ('Oversteering at entry').",
                },
            },
            required=["label_id"],
        ),
        FunctionSchema(
            name="get_track_knowledge",
            description=(
                "Per-track curated notes (overview + corner-by-corner). "
                "Omit `corner` to get the overview plus the list of corner "
                "names; pass a corner name to get just that section."
            ),
            properties={
                "track": {
                    "type": "string",
                    "description": "Track id (e.g. 'spa', 'silverstone').",
                },
                "corner": {
                    "type": "string",
                    "description": "Optional corner name (e.g. 'Eau Rouge').",
                },
            },
            required=["track"],
        ),
        FunctionSchema(
            name="search_racing_knowledge",
            description=(
                "Semantic search over driver transcripts, race reports, and "
                "theory notes. Use for cross-cutting questions where the right "
                "doc isn't obvious ('what do drivers say about wet setup', "
                "'where do most cars lose time under braking'). Returns top-k "
                "matching snippets."
            ),
            properties={
                "query": {"type": "string", "description": "Free-text question or topic."},
                "top_k": {"type": "integer", "description": "Snippets to return (default 5)."},
            },
            required=["query"],
        ),
    ]


def _build_frontend_tool_schemas(frontend_tools: Iterable[Dict[str, Any]]) -> List[Any]:
    """Convert the frontend's tool descriptors into Pipecat FunctionSchemas.

    Each ``frontend_tools`` entry is a plain dict with ``name``, ``description``,
    ``properties`` and ``required`` (mirrors FunctionSchema's constructor).
    Entries missing ``name`` are skipped with a warning — defensive against
    a misbehaving frontend, since this is an untrusted boundary.
    """
    from pipecat.adapters.schemas.function_schema import FunctionSchema

    schemas: List[Any] = []
    for tool in frontend_tools:
        name = tool.get("name")
        if not isinstance(name, str) or not name:
            LOGGER.warning("frontend_info: tool entry missing 'name': %r", tool)
            continue
        schemas.append(FunctionSchema(
            name=name,
            description=str(tool.get("description") or ""),
            properties=dict(tool.get("properties") or {}),
            required=list(tool.get("required") or []),
        ))
    return schemas


def _build_title_map(frontend_tools: Iterable[Dict[str, Any]]) -> Dict[str, str]:
    """Merge server-tool titles with frontend-supplied titles."""
    titles = dict(_SERVER_TOOL_TITLES)
    for tool in frontend_tools:
        name = tool.get("name")
        title = tool.get("title")
        if isinstance(name, str) and name and isinstance(title, str) and title:
            titles[name] = title
    return titles


def _make_tool_handler(
    tool_executor,
    session_config: "VoiceSessionConfig",
    conn: Any,
    *,
    frontend_tool_names: frozenset[str],
    tool_titles: Dict[str, str],
):
    """Build a per-session async handler with two-bucket dispatch.

    * Tool names in ``frontend_tool_names`` (derived from the WS handshake)
      → forwarded to the frontend over the WS via
      :func:`app.voice.tool_relay.get_relay().dispatch`. The ``conn`` arg
      identifies which WS connection to send the call on.
    * Everything else → forwarded to ``tool_executor`` (server-side path,
      typically ``AIService._execute_function``).

    Both paths share the side-product filter (underscore-prefixed keys are
    logged but not sent back to the LLM) so server-side and frontend-side
    tools behave consistently from the LLM's perspective.

    Each call also emits ``tool_event`` text frames (started + completed)
    on the same WS so the chat UI can render a "tool box" with the
    human-readable title from ``tool_titles`` (server-side fallback +
    frontend-supplied titles from the handshake).
    """
    import json as _json
    from app.voice.tool_relay import get_relay

    relay = get_relay()

    def _tool_title(name: str) -> str:
        return tool_titles.get(name) or _prettify(name)

    async def _emit_tool_event(payload: Dict[str, Any]) -> None:
        try:
            await conn.send_text(_json.dumps({"type": "tool_event", **payload}))
        except Exception:
            LOGGER.debug("tool_event emit failed (WS likely closed)", exc_info=True)

    async def dispatch_tool(function_name: str, arguments: Dict[str, Any]) -> Any:
        """Execute one tool by name and return the LLM-visible payload.

        Shared by the Pipecat ``register_function`` handler and the
        FunctionTagRecovery fallback. Emits tool_event start/complete frames,
        routes frontend vs. server tools, filters underscore-prefixed side
        products, and logs the result. Never raises — failures come back as
        ``{"error": ...}`` so both call sites can inject cleanly.
        """
        title = _tool_title(function_name)
        arguments = arguments or {}

        LOGGER.info("[TOOL-CALL] name=%s args=%r", function_name, arguments)

        await _emit_tool_event({
            "name": function_name,
            "title": title,
            "status": "started",
            "arguments": arguments,
        })

        ok = True
        error_msg: Optional[str] = None
        try:
            if function_name in frontend_tool_names:
                # Relayed to the Electron app over the same WS as audio.
                # dispatch() never raises — failures come back as {"error": ...}.
                result = await relay.dispatch(conn, function_name, arguments)
            else:
                # Server-side path. Context carries the connect-time IDs;
                # track/car are intentionally absent (LLM fetches via tool).
                # ``_conn`` is an opaque handle that server-side composite
                # tools (e.g. analyze_telemetry) use to relay back to the
                # frontend via the same WS — underscore-prefixed because
                # it's a server-internal channel, not part of the OpenAI
                # context schema.
                context = {
                    "session_id": session_config.session_id,
                    "user_id": session_config.user_id,
                    "_conn": conn,
                }
                result = await tool_executor(function_name, arguments, context)
        except Exception as exc:
            LOGGER.exception("Voice tool %s failed", function_name)
            error_msg = str(exc)
            await _emit_tool_event({
                "name": function_name,
                "title": title,
                "status": "completed",
                "ok": False,
                "error": error_msg,
            })
            return {"error": error_msg}

        # Side-product filter — underscore-prefixed keys never reach the LLM.
        if isinstance(result, dict):
            public = {k: v for k, v in result.items() if not k.startswith("_")}
            side_products = {k: v for k, v in result.items() if k.startswith("_")}
            if side_products:
                LOGGER.info(
                    "Voice tool %s produced side products (not forwarded to LLM): %s",
                    function_name, list(side_products.keys()),
                )
            payload = public if public else result
            if isinstance(payload, dict) and "error" in payload:
                ok = False
                error_msg = str(payload.get("error"))
        else:
            payload = result

        await _emit_tool_event({
            "name": function_name,
            "title": title,
            "status": "completed",
            "ok": ok,
            "error": error_msg,
        })
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
        payload = await dispatch_tool(params.function_name, params.arguments or {})
        await params.result_callback(payload)

    return handle_tool_call, dispatch_tool


# Streamed-text salvage for hosted models that drop out of the OpenAI
# tool_calls channel and emit the call inline as text.
#
# Matches the Llama-3 "Python tag" form
#   <function=NAME>{json…}</function>
# Tolerates whitespace around the JSON body and uses a non-greedy match
# so multiple tags in one response don't collapse into one.
_FUNCTION_TAG_RE = re.compile(
    r"<function=([A-Za-z_][\w]*)>\s*(\{.*?\})\s*</function>",
    re.DOTALL,
)
_FUNCTION_TAG_OPEN = "<function="
_FUNCTION_TAG_CLOSE = "</function>"


def _build_function_tag_recovery(*, dispatch_tool, context, get_task):
    """Salvage Llama-style ``<function=name>{json}</function>`` text emissions.

    Some hosted models (notably llama-3.3-70b-versatile on Groq at
    temperature ≥ 0.5) drop out of the OpenAI ``tool_calls`` channel under
    load and emit the call inline as text using their training-data Python-
    tag pattern. ``OpenAILLMService`` doesn't recognise that shape, so
    without this processor the raw tag leaks to the chat UI and TTS.

    Sits between the LLM and the assistant transcript observer. Watches
    every TextFrame, holding back any tail that *could* be the start of a
    function tag so a chunk boundary mid-tag doesn't leak the prefix
    (``"…<fun"`` arrives, recovery keeps the ``<fun``, then ``"ction=…"``
    completes the match). On a full match it:

    * suppresses the tag text (so neither transcript observer nor TTS sees
      it — chat UI stays clean, Kokoro doesn't try to read XML aloud);
    * dispatches the tool via the shared ``dispatch_tool`` so tool_event
      frames, frontend/server routing, and side-product filtering all
      happen exactly like a native tool_calls round-trip;
    * injects a synthetic ``assistant(tool_calls)`` + ``tool(result)`` pair
      into ``LLMContext`` so the next turn looks like a normal tool exchange
      to the model;
    * queues an ``LLMRunFrame`` so the model immediately produces a proper
      spoken reply using the tool result.

    Any prose *before* the tag (e.g. ``"Let me check…"``) is forwarded
    normally — the driver still hears the preamble; the tool runs; then
    the model speaks the result. Prose *after* the tag is also forwarded
    (rare but possible).

    ``get_task`` is a zero-arg callable returning the PipelineTask so the
    processor can queue ``LLMRunFrame`` without a circular import at build
    time (the task is constructed *after* this processor).
    """
    import json as _json
    import uuid as _uuid

    from pipecat.frames.frames import (
        Frame,
        LLMFullResponseEndFrame,
        LLMFullResponseStartFrame,
        LLMRunFrame,
        TextFrame,
    )
    from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

    class FunctionTagRecovery(FrameProcessor):
        def __init__(self) -> None:
            super().__init__()
            self._held: str = ""  # tail that might still be (or be growing into) a tag

        async def process_frame(self, frame: "Frame", direction: "FrameDirection") -> None:
            await super().process_frame(frame, direction)

            if isinstance(frame, LLMFullResponseStartFrame):
                self._held = ""
                await self.push_frame(frame, direction)
                return

            if isinstance(frame, LLMFullResponseEndFrame):
                # Response over — whatever we still hold can't complete a tag.
                if self._held:
                    await self.push_frame(TextFrame(text=self._held), direction)
                    self._held = ""
                await self.push_frame(frame, direction)
                return

            if not isinstance(frame, TextFrame):
                await self.push_frame(frame, direction)
                return

            text = getattr(frame, "text", "") or ""
            if not text:
                return

            buf = self._held + text
            self._held = ""
            await self._drain(buf, direction)

        async def _drain(self, buf: str, direction: "FrameDirection") -> None:
            """Walk ``buf`` left-to-right: emit prose, hold tag prefixes, dispatch full tags."""
            while buf:
                idx = buf.find("<")
                if idx == -1:
                    await self.push_frame(TextFrame(text=buf), direction)
                    return

                if idx > 0:
                    await self.push_frame(TextFrame(text=buf[:idx]), direction)
                    buf = buf[idx:]

                # buf now starts with '<'. Three cases:
                #   1. shorter than the open marker but still a prefix of it  → hold
                #   2. starts with the open marker but no close yet           → hold
                #   3. starts with the open marker AND has a close            → match-or-flush
                #   4. starts with '<' but can't be the open marker           → emit '<' and continue
                if len(buf) < len(_FUNCTION_TAG_OPEN):
                    if _FUNCTION_TAG_OPEN.startswith(buf):
                        self._held = buf
                        return
                    await self.push_frame(TextFrame(text=buf[:1]), direction)
                    buf = buf[1:]
                    continue

                if not buf.startswith(_FUNCTION_TAG_OPEN):
                    await self.push_frame(TextFrame(text=buf[:1]), direction)
                    buf = buf[1:]
                    continue

                close = buf.find(_FUNCTION_TAG_CLOSE)
                if close == -1:
                    self._held = buf
                    return

                tag_end = close + len(_FUNCTION_TAG_CLOSE)
                full_tag = buf[:tag_end]
                buf = buf[tag_end:]

                m = _FUNCTION_TAG_RE.match(full_tag)
                if not m:
                    # Looked like a tag but the body isn't well-formed JSON-wrapped.
                    # Treat as plain text so we don't silently swallow content.
                    LOGGER.warning(
                        "FunctionTagRecovery: malformed tag, passing through as text: %r",
                        full_tag,
                    )
                    await self.push_frame(TextFrame(text=full_tag), direction)
                    continue

                function_name = m.group(1)
                raw_args = m.group(2)
                try:
                    arguments = _json.loads(raw_args)
                except _json.JSONDecodeError:
                    LOGGER.warning(
                        "FunctionTagRecovery: bad JSON args for <function=%s>: %r",
                        function_name, raw_args,
                    )
                    await self.push_frame(TextFrame(text=full_tag), direction)
                    continue

                LOGGER.info(
                    "[TOOL-CALL-RECOVERED] name=%s args=%r (from text-channel <function=…> tag)",
                    function_name, arguments,
                )
                # Fire-and-forget — dispatch_tool already swallows exceptions
                # and emits tool_event frames. We don't await here so we
                # don't block the streaming pipeline.
                import asyncio as _asyncio
                _asyncio.create_task(self._recover(function_name, arguments))

        async def _recover(self, function_name: str, arguments: Dict[str, Any]) -> None:
            payload = await dispatch_tool(function_name, arguments)

            tool_call_id = f"recover_{_uuid.uuid4().hex[:8]}"
            try:
                context.add_message({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": tool_call_id,
                        "type": "function",
                        "function": {
                            "name": function_name,
                            "arguments": _json.dumps(arguments),
                        },
                    }],
                })
                context.add_message({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": _json.dumps(payload) if not isinstance(payload, str) else payload,
                })
            except Exception:
                LOGGER.exception("FunctionTagRecovery: context injection failed")
                return

            task = get_task()
            if task is None:
                LOGGER.warning(
                    "FunctionTagRecovery: no PipelineTask bound, can't queue LLMRunFrame "
                    "— tool result is in context but model won't speak until next user turn"
                )
                return
            try:
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

    All frames pass through unchanged — this processor is observation-only.
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

    This keeps Kokoro from ever speaking "[vibing]" — parallel to how
    tool_event frames are UI-only signals that also never reach TTS.
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


async def build_voice_pipeline_task(
    websocket: Any,
    session_config: VoiceSessionConfig,
    tool_executor: Any,
    *,
    frontend_tools: Optional[List[Dict[str, Any]]] = None,
    query_scope_schema: Optional[Dict[str, Any]] = None,
):
    """Build a Pipecat PipelineTask bound to the given WebSocket.

    Returns the task; caller is responsible for running it via
    `PipelineRunner.run(task)`.

    Side effect: registers the WebSocket with :mod:`app.voice.tool_relay`
    so frontend tool calls and observation pushes routed via text frames
    reach this session's LLM context. The caller (api/voice.py) is
    responsible for unbinding on session end.

    Raises ImportError if pipecat-ai or faster-whisper aren't available
    — the caller should map this to an explicit error frame to the WS.
    """
    # Deferred imports — see module docstring.
    import asyncio
    import json as _json
    from pipecat.adapters.schemas.tools_schema import ToolsSchema
    from pipecat.audio.vad.silero import SileroVADAnalyzer
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

    from app.voice.pipecat_kokoro import build_kokoro_processor
    from app.voice.raw_pcm_serializer import RawPCMSerializer
    from app.voice.tool_relay import get_relay

    TranscriptObserver = _build_transcript_observer()
    EmotionTagStripper = _build_emotion_tag_stripper()
    ContextLogger = _build_context_logger()

    LOGGER.info(
        "Building voice pipeline (session=%s user=%s)",
        session_config.session_id, session_config.user_id,
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
    # Phase 3 starting point: small English model on whichever device is
    # available. Pipecat's WhisperSTTService wraps faster-whisper directly.
    stt = WhisperSTTService(
        model="small.en",
        # device="cuda" if available; Pipecat auto-detects via faster-whisper.
    )

    # --- LLM (OpenAI-compatible client) ---
    # Backend = hosted endpoint if HOSTED_LLM_BASE_URL is set, else local
    # llama-server sidecar.
    if settings.hosted_llm_base_url:
        missing = [
            name for name, val in (
                ("HOSTED_LLM_API_KEY", settings.hosted_llm_api_key),
                ("HOSTED_LLM_MODEL", settings.hosted_llm_model),
            ) if not val
        ]
        if missing:
            raise RuntimeError(
                f"HOSTED_LLM_BASE_URL is set; also requires {', '.join(missing)}"
            )
        llm_base_url = settings.hosted_llm_base_url
        llm_api_key = settings.hosted_llm_api_key
        llm_model = settings.hosted_llm_model
    else:
        llm_base_url = settings.llama_server_url
        llm_api_key = "not-needed"  # llama-server ignores auth; OpenAI client requires non-empty.
        llm_model = settings.llama_model_name

    llm = OpenAILLMService(
        base_url=llm_base_url,
        api_key=llm_api_key,
        settings=OpenAILLMService.Settings(
            model=llm_model,
            temperature=0.7,
            max_tokens=600,  # Engineer-voice answers can run a few sentences; Pipecat still stops at end-of-turn.
        ),
    )

    # --- Tool calling (Phase 3b) ---
    # The LLM's tool surface is the union of:
    #   * server-side tools (analyze_telemetry, explain_label) — schemas
    #     live in Python next to their executor.
    #   * frontend-side tools — schemas arrive over the WS handshake from
    #     api/voice.py (see :func:`_await_frontend_info`). Frontend owns
    #     the schema definitions so there's a single source of truth.
    # The handler dispatches by name: frontend names go through the WS
    # tool relay; everything else goes through ``tool_executor`` (typically
    # AIService._execute_function, bound by api/voice.py).
    fe_tools = frontend_tools or []
    frontend_tool_names = frozenset(
        t["name"] for t in fe_tools if isinstance(t.get("name"), str)
    )
    tool_schemas = _build_server_tool_schemas(query_scope_schema) + _build_frontend_tool_schemas(fe_tools)
    tool_titles = _build_title_map(fe_tools)
    tools = ToolsSchema(standard_tools=tool_schemas)

    tool_handler, dispatch_tool = _make_tool_handler(
        tool_executor, session_config, conn=websocket,
        frontend_tool_names=frontend_tool_names,
        tool_titles=tool_titles,
    )
    for schema in tool_schemas:
        llm.register_function(schema.name, tool_handler)

    # System message + context object (Pipecat's history aggregator). The
    # engineer prompt has no {track}/{car} placeholders — the LLM doesn't
    # carry that state; it responds to what the driver says.
    #
    # Append behavior specs loaded from the skills corpus so the instructions
    # live in editable .md files, not in Python code.
    from app.skills.racing_engineer import behavior as _skill_behavior
    system_prompt = _VOICE_COACH_PROMPT_TEMPLATE
    for _behavior_name in ("emotion", "transcript_resilience"):
        _skill = _skill_behavior(_behavior_name)
        _section = _skill.get("_raw_body", "") if _skill else ""
        if _section:
            system_prompt = f"{system_prompt.rstrip()}\n\n{_section}"

    context = LLMContext(
        messages=[{"role": "system", "content": system_prompt}],
        tools=tools,
    )
    context_aggregator = LLMContextAggregatorPair(context)

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

    # --- Function-tag recovery (Llama-text-channel salvage) ---
    # See _build_function_tag_recovery for the why. ``get_task`` closes
    # over a single-element list so the processor can resolve the task
    # lazily — PipelineTask is built *after* the pipeline, but the
    # processor needs to live *inside* the pipeline.
    _task_holder: List[Any] = []
    FunctionTagRecovery = _build_function_tag_recovery(
        dispatch_tool=dispatch_tool,
        context=context,
        get_task=lambda: _task_holder[0] if _task_holder else None,
    )
    function_tag_recovery = FunctionTagRecovery()

    # --- Pipeline composition ---
    # VAD sits between the transport input and STT so Whisper only runs
    # on actual speech windows (gated by VADUserStartedSpeakingFrame /
    # VADUserStoppedSpeakingFrame from the VADProcessor).
    #
    # user_transcript_observer sits AFTER stt so it sees the final
    # TranscriptionFrame before context_aggregator.user() consumes it.
    # function_tag_recovery sits immediately AFTER llm so it can suppress
    # text-channel <function=…> emissions before they reach transcript,
    # TTS, or the context aggregator — see its docstring for why this
    # placement matters.
    # assistant_transcript_observer sits AFTER recovery so it only sees
    # legitimate spoken text (chat UI stays clean of leaked tool tags).
    # emotion_tag_stripper sits AFTER the observer and BEFORE tts so Kokoro
    # never receives the [emotion] tag — same principle as tool_event frames
    # being UI-only signals that never reach speech synthesis.
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
        function_tag_recovery,
        context_logger,
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
    _task_holder.append(task)

    # --- Observation sink (frontend monitoring-agent pushes) ----------------
    # When the frontend WS sends {"type":"observation","data":{...}}, the
    # relay calls this sink. We format the observation as a synthetic user
    # turn, append it to the live LLMContext, and trigger the LLM to
    # respond. Same conversation context as voice turns — the LLM doesn't
    # know an observation came from a different path than STT.
    loop = asyncio.get_running_loop()

    def observation_sink(data: dict) -> None:
        text = _format_observation_for_llm(data)
        context.add_message({"role": "user", "content": f"[OBSERVATION] {text}"})
        # Trigger the LLM to generate a response now (don't wait for the
        # next spoken user turn). Best-effort across Pipecat versions:
        # LLMRunFrame is the canonical trigger; fall back to a transcription
        # frame which the user-aggregator forwards.
        try:
            from pipecat.frames.frames import LLMRunFrame
            loop.create_task(task.queue_frame(LLMRunFrame()))
        except ImportError:
            try:
                from pipecat.frames.frames import TranscriptionFrame
                loop.create_task(task.queue_frame(
                    TranscriptionFrame(text="", user_id="observation", timestamp="")
                ))
            except Exception:
                LOGGER.exception(
                    "observation_sink: could not push trigger frame; "
                    "message appended to context but LLM won't fire until "
                    "the next spoken user turn"
                )

    def user_text_sink(text: str) -> None:
        """Inject a typed chat message as a synthetic user turn.

        Same path as observation_sink minus the [OBSERVATION] framing —
        the LLM treats it as if the driver had spoken it. We also echo a
        ``user_transcript`` text frame back so the UI shows the typed
        message immediately, even before the LLM responds.
        """
        import time as _time
        LOGGER.info("[LAT-DIAG] user_text_in t=%.3f chars=%d", _time.monotonic(), len(text))
        context.add_message({"role": "user", "content": text})
        try:
            loop.create_task(_send_text(_json.dumps({
                "type": "user_transcript", "text": text, "source": "typed",
            })))
        except Exception:
            LOGGER.exception("user_text_sink: failed to echo user_transcript")
        try:
            from pipecat.frames.frames import LLMRunFrame
            loop.create_task(task.queue_frame(LLMRunFrame()))
        except ImportError:
            try:
                from pipecat.frames.frames import TranscriptionFrame
                loop.create_task(task.queue_frame(
                    TranscriptionFrame(text="", user_id="user_text", timestamp="")
                ))
            except Exception:
                LOGGER.exception("user_text_sink: could not trigger LLM run")

    get_relay().bind(
        websocket,
        send_text=_send_text,
        observation_sink=observation_sink,
        user_text_sink=user_text_sink,
    )

    return task


def _format_observation_for_llm(data: dict) -> str:
    """Turn an observation payload into a one-line prompt for the LLM.

    Observations come in raw from the frontend monitoring agent — they
    carry an ``event`` name plus arbitrary context (telemetry rows, lap
    number, etc.). We compress them to a short prompt so the LLM has
    something concrete to respond to without re-classifying every channel.
    Classification is still on the LLM to invoke (via analyze_telemetry)
    if it decides the observation warrants it.
    """
    event = data.get("event", "event")
    bits = [event]
    for k in ("section", "lap", "lap_number"):
        v = data.get(k)
        if v is not None:
            bits.append(f"{k}={v}")
    n_rows = len(data.get("telemetry_rows") or [])
    if n_rows:
        bits.append(f"telemetry_rows={n_rows}")
    return " ".join(bits) + ". Respond with one short engineer suggestion."


async def run_voice_session(
    websocket: Any,
    session_config: VoiceSessionConfig,
    tool_executor: Any,
    *,
    frontend_tools: Optional[List[Dict[str, Any]]] = None,
    query_scope_schema: Optional[Dict[str, Any]] = None,
) -> None:
    """Bind a Pipecat pipeline to `websocket` and run it to completion.

    Returns when the WS closes or the pipeline exits. Caller is responsible
    for any auth/lifecycle concerns around `websocket`, supplying a
    ``tool_executor`` (typically AIService._execute_function), and passing
    ``frontend_tools`` plus ``query_scope_schema`` from the WS handshake
    (see :mod:`app.api.voice`).

    Also unbinds the WebSocket from :mod:`app.voice.tool_relay` on exit so
    in-flight tool-call futures are cancelled cleanly.
    """
    # Deferred imports.
    from pipecat.pipeline.runner import PipelineRunner
    from app.voice.tool_relay import get_relay

    task = await build_voice_pipeline_task(
        websocket, session_config, tool_executor,
        frontend_tools=frontend_tools,
        query_scope_schema=query_scope_schema,
    )
    runner = PipelineRunner()
    try:
        await runner.run(task)
    finally:
        get_relay().unbind(websocket)
        LOGGER.info("Voice session ended (user=%s)", session_config.user_id)
