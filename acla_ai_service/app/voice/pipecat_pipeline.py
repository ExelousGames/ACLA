"""Pipecat voice-conversation pipeline factory (Phase 3).

Builds a per-WebSocket-session pipeline:

    FastAPIWebsocketTransport.input()
        → SileroVADAnalyzer (endpoint detection)
        → faster-whisper STT
        → OpenAILLMService(base_url=llama_server_url)  -- our local Qwen
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
from dataclasses import dataclass
from typing import Any, Dict, Optional

from app.infra.config import settings

LOGGER = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# System prompt for the voice coach
# ----------------------------------------------------------------------

_VOICE_COACH_PROMPT_TEMPLATE = """You are a race engineer speaking to your driver over the radio. Stay in character.

Speak like a real engineer: short radio-style sentences, 1–3 max per turn
unless asked to elaborate. Natural spoken English — no markdown, no bullet
points, no headings. Use racing vocabulary freely: apex, trail-brake,
understeer at entry, throttle pickup, traction-limited exit, slip balance,
kerb, kerb-strike, brake bias, weight transfer.

Tools (call only when the question requires data you don't have):
- analyze_recent_segment: canonical "what just happened" tool. One call
  bundles recent telemetry + classified actions + their engineer-voice
  descriptions. Use for "what just happened", "why did I go wide", "did I
  brake too late", "what was wrong with that corner".
- analyze_lap: same one-shot bundle but for a specific completed lap.
  Use for "how was lap 12", "any mistakes on lap 3", "walk me through my
  best lap". Defaults to most recent completed lap if no number given.
- get_session_info: returns current track and car. Call ONCE per session
  if you need to mention them; cache the answer mentally.
- get_recent_telemetry / get_lap_telemetry / classify_segment: primitives
  for unusual asks where a composite doesn't fit.
- explain_label: definition + remedies for a specific labelled action,
  by id or natural name.
- start_per_turn_coaching / stop_per_turn_coaching: activate / deactivate
  the per-corner monitoring agent. While active, the system pushes
  observations as "[OBSERVATION]" user turns describing each completed
  segment — respond to each with one short suggestion.

When a composite returns labels with definitions and remedies, do NOT
read every field aloud. Pick the one or two that matter most for THIS
corner and weave them into a natural 1–3 sentence engineer comment.
The driver wants advice, not a catalog.

Rules:
- If a tool returns an error or the link is down, say so plainly ("can't
  see your telemetry right now"). Never fabricate numbers or label names.
- Translate any label codes in tool output to natural English before
  speaking. The driver should never hear "MS44" — say "oversteer at entry".
- For general racing questions ("what is trail braking?"), answer directly
  in 2-3 sentences without calling any tool.
- Don't repeat the same advice across turns. If the driver makes the same
  mistake twice, escalate ("again — this is the third time on this corner").
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


# Tool names whose execution goes through the frontend WS relay (see
# app/voice/tool_relay.py). Everything else routes to the server-side
# AIService._execute_function dispatcher.
FRONTEND_TOOLS: frozenset[str] = frozenset({
    "get_session_info",
    "get_recent_telemetry",
    "get_lap_telemetry",
    "start_per_turn_coaching",
    "stop_per_turn_coaching",
})


# Human-readable titles shown in the chat UI for each tool. The driver
# sees these in a "tool box" while the LLM is calling the function — they
# should read like a brief status line, not the raw function name.
_TOOL_TITLES: Dict[str, str] = {
    "analyze_recent_segment": "Analyzing the last few seconds",
    "analyze_lap": "Analyzing the lap",
    "explain_label": "Looking up the term",
    "classify_segment": "Classifying the segment",
    "get_session_info": "Checking session info",
    "get_recent_telemetry": "Reading recent telemetry",
    "get_lap_telemetry": "Reading lap telemetry",
    "start_per_turn_coaching": "Starting per-turn coaching",
    "stop_per_turn_coaching": "Stopping per-turn coaching",
}


def _tool_title(name: str) -> str:
    """Return the chat-visible title for a tool name, falling back to a
    prettified version of the raw name."""
    return _TOOL_TITLES.get(name) or name.replace("_", " ").strip().capitalize()


def _build_tool_schemas():
    """Return Pipecat FunctionSchemas for the racing-engineer tools.

    Mirrors AIService.get_available_functions() so the voice and HTTP-style
    paths advertise the same capabilities to the LLM. Deferred import — only
    loaded when a voice session is actually built.
    """
    from pipecat.adapters.schemas.function_schema import FunctionSchema

    schemas = [
        # ── Server-side composite (canonical "what just happened" flow) ─────
        FunctionSchema(
            name="analyze_recent_segment",
            description=(
                "ONE-SHOT analysis of the most recent driving segment. Fetches "
                "recent telemetry from the frontend, runs the classifier, looks "
                "up the racing-engineer concept for each detected action, and "
                "returns a bundled payload. Prefer this over chaining "
                "get_recent_telemetry + classify_segment + explain_label."
            ),
            properties={
                "seconds": {
                    "type": "integer",
                    "description": "How many seconds of recent telemetry to analyze. Default 8.",
                },
            },
            required=[],
        ),
        FunctionSchema(
            name="analyze_lap",
            description=(
                "ONE-SHOT analysis of a specific completed lap. Fetches the "
                "lap's telemetry from the frontend, runs the classifier, looks "
                "up the racing-engineer concept for each detected action, and "
                "returns a bundled payload. Use for 'how was lap 12?', "
                "'any mistakes on lap 3?', 'walk me through my best lap'."
            ),
            properties={
                "lap": {
                    "type": "integer",
                    "description": "Lap number to analyze. Defaults to the most recently completed lap.",
                },
            },
            required=[],
        ),
        # ── Server-side primitives (corpus + ML) ────────────────────────────
        FunctionSchema(
            name="explain_label",
            description=(
                "Return the racing-engineer concept for a single action label "
                "(definition, engineer interpretation, common remedies). Use "
                "when the driver asks 'what does <action> mean?' or after a "
                "classifier call to deepen one specific finding."
            ),
            properties={
                "label_id": {
                    "type": "string",
                    "description": "Label id like 'MS44' or its natural name like 'Oversteering at entry'.",
                },
            },
            required=["label_id"],
        ),
        FunctionSchema(
            name="classify_segment",
            description=(
                "Run the segment classifier over a window of telemetry rows and "
                "return the action labels present (translated to natural names). "
                "Use for unusual asks where analyze_recent_segment doesn't fit."
            ),
            properties={
                "telemetry_rows": {
                    "type": "array",
                    "description": "List of telemetry row dicts (as returned by get_recent_telemetry).",
                    "items": {"type": "object"},
                },
            },
            required=["telemetry_rows"],
        ),
        # ── Frontend data accessors (relayed via WS) ────────────────────────
        FunctionSchema(
            name="get_session_info",
            description=(
                "Return current track / car / user identity from the live "
                "session. Call once if you need to reference the track or car "
                "by name; the answer doesn't change during a session."
            ),
            properties={},
            required=[],
        ),
        FunctionSchema(
            name="get_recent_telemetry",
            description=(
                "Return the last N seconds of raw telemetry rows from the live "
                "in-memory buffer. Optional channel filter narrows to specific "
                "columns. No classification — returns raw data only."
            ),
            properties={
                "seconds": {
                    "type": "integer",
                    "description": "How many seconds back from now. Default 8.",
                },
                "channels": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional list of channel names to include.",
                },
            },
            required=[],
        ),
        FunctionSchema(
            name="get_lap_telemetry",
            description=(
                "Return raw telemetry rows for one completed lap. Optional "
                "channel filter narrows to specific columns. No classification "
                "— returns raw data only. Defaults to the most recently "
                "completed lap if no number is given."
            ),
            properties={
                "lap": {
                    "type": "integer",
                    "description": "Lap number. Defaults to most recent completed lap.",
                },
                "channels": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional list of channel names to include.",
                },
            },
            required=[],
        ),
        # ── Frontend monitoring activators ──────────────────────────────────
        FunctionSchema(
            name="start_per_turn_coaching",
            description=(
                "Activate the background monitoring agent that watches for "
                "corner-end events and pushes an observation after each one. "
                "Returns immediately; subsequent observations arrive as "
                "'[OBSERVATION]' user turns. Use when the driver asks to be "
                "coached every corner / through the rest of the session."
            ),
            properties={},
            required=[],
        ),
        FunctionSchema(
            name="stop_per_turn_coaching",
            description=(
                "Stop the per-corner monitoring agent. Use when the driver "
                "says 'stop coaching', 'quiet for a bit', or asks to be left "
                "alone."
            ),
            properties={},
            required=[],
        ),
    ]
    return schemas


def _make_tool_handler(tool_executor, session_config: "VoiceSessionConfig", conn: Any):
    """Build a per-session async handler with two-bucket dispatch.

    * Tool names in :data:`FRONTEND_TOOLS` → forwarded to the frontend over
      the WS via :func:`app.voice.tool_relay.get_relay().dispatch`. The
      ``conn`` arg identifies which WS connection to send the call on.
    * Everything else → forwarded to ``tool_executor`` (server-side path,
      typically ``AIService._execute_function``).

    Both paths share the side-product filter (underscore-prefixed keys are
    logged but not sent back to the LLM) so server-side and frontend-side
    tools behave consistently from the LLM's perspective.

    Each call also emits ``tool_event`` text frames (started + completed)
    on the same WS so the chat UI can render a "tool box" with the
    human-readable title from :data:`_TOOL_TITLES`.
    """
    import json as _json
    from app.voice.tool_relay import get_relay

    relay = get_relay()

    async def _emit_tool_event(payload: Dict[str, Any]) -> None:
        try:
            await conn.send_text(_json.dumps({"type": "tool_event", **payload}))
        except Exception:
            LOGGER.debug("tool_event emit failed (WS likely closed)", exc_info=True)

    async def handle_tool_call(params):
        function_name = params.function_name
        arguments = params.arguments or {}
        title = _tool_title(function_name)

        await _emit_tool_event({
            "name": function_name,
            "title": title,
            "status": "started",
            "arguments": arguments,
        })

        ok = True
        error_msg: Optional[str] = None
        try:
            if function_name in FRONTEND_TOOLS:
                # Relayed to the Electron app over the same WS as audio.
                # dispatch() never raises — failures come back as {"error": ...}.
                result = await relay.dispatch(conn, function_name, arguments)
            else:
                # Server-side path. Context carries the connect-time IDs;
                # track/car are intentionally absent (LLM fetches via tool).
                # ``_conn`` is an opaque handle that server-side composite
                # tools (e.g. analyze_recent_segment) use to relay back to
                # the frontend via the same WS — underscore-prefixed because
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
            ok = False
            error_msg = str(exc)
            await _emit_tool_event({
                "name": function_name,
                "title": title,
                "status": "completed",
                "ok": False,
                "error": error_msg,
            })
            await params.result_callback({"error": error_msg})
            return

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
        await params.result_callback(payload)

    return handle_tool_call


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
                        text = "".join(self._assistant_buf).strip()
                        self._assistant_buf = []
                        if text:
                            await self._emit("assistant_transcript", text)
            except Exception:
                LOGGER.exception("TranscriptObserver: emit failed (role=%s)", self._role)

            await self.push_frame(frame, direction)

        async def _emit(self, kind: str, text: str) -> None:
            try:
                await self._send_text(_json.dumps({"type": kind, "text": text}))
            except Exception:
                LOGGER.debug("%s emit failed (WS likely closed)", kind, exc_info=True)

    return TranscriptObserver


async def build_voice_pipeline_task(
    websocket: Any,
    session_config: VoiceSessionConfig,
    tool_executor: Any,
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

    # --- LLM (local llama-server via OpenAI-compatible client) ---
    llm = OpenAILLMService(
        base_url=settings.llama_server_url,
        api_key="not-needed",
        settings=OpenAILLMService.Settings(
            model=settings.llama_model_name,
            temperature=0.7,
            max_tokens=600,  # Engineer-voice answers can run a few sentences; Pipecat still stops at end-of-turn.
        ),
    )

    # --- Tool calling (Phase 3b) ---
    # Build schemas + register handlers that delegate to the caller-provided
    # `tool_executor` (typically AIService._execute_function, bound by
    # api/voice.py). Voice and text paths share the same tool implementations,
    # so behaviour stays consistent.
    tool_schemas = _build_tool_schemas()
    tools = ToolsSchema(standard_tools=tool_schemas)

    tool_handler = _make_tool_handler(tool_executor, session_config, conn=websocket)
    for schema in tool_schemas:
        llm.register_function(schema.name, tool_handler)

    # System message + context object (Pipecat's history aggregator). The
    # engineer prompt has no {track}/{car} placeholders — the LLM fetches
    # those on demand via get_session_info if it actually needs to mention
    # them.
    context = LLMContext(
        messages=[{"role": "system", "content": _VOICE_COACH_PROMPT_TEMPLATE}],
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

    # --- Pipeline composition ---
    # VAD sits between the transport input and STT so Whisper only runs
    # on actual speech windows (gated by VADUserStartedSpeakingFrame /
    # VADUserStoppedSpeakingFrame from the VADProcessor).
    #
    # user_transcript_observer sits AFTER stt so it sees the final
    # TranscriptionFrame before context_aggregator.user() consumes it.
    # assistant_transcript_observer sits AFTER llm so it sees the LLM
    # text chunks before tts converts them to audio.
    pipeline = Pipeline([
        transport.input(),
        vad_processor,
        stt,
        user_transcript_observer,
        context_aggregator.user(),
        llm,
        assistant_transcript_observer,
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
    Classification is still on the LLM to invoke (via classify_segment or
    analyze_recent_segment) if it decides the observation warrants it.
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
) -> None:
    """Bind a Pipecat pipeline to `websocket` and run it to completion.

    Returns when the WS closes or the pipeline exits. Caller is responsible
    for any auth/lifecycle concerns around `websocket` and for supplying a
    ``tool_executor`` (typically AIService._execute_function).

    Also unbinds the WebSocket from :mod:`app.voice.tool_relay` on exit so
    in-flight tool-call futures are cancelled cleanly.
    """
    # Deferred imports.
    from pipecat.pipeline.runner import PipelineRunner
    from app.voice.tool_relay import get_relay

    task = await build_voice_pipeline_task(websocket, session_config, tool_executor)
    runner = PipelineRunner()
    try:
        await runner.run(task)
    finally:
        get_relay().unbind(websocket)
        LOGGER.info("Voice session ended (user=%s)", session_config.user_id)
