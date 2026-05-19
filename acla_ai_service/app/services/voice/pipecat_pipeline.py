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
from typing import Any, Optional

from app.core import settings

LOGGER = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# System prompt for the voice coach
# ----------------------------------------------------------------------

_VOICE_COACH_PROMPT_TEMPLATE = """You are a racing telemetry coach for sim racing. Stay in character.
Current session: track={track_name}, car={car_name}

You are speaking to the driver over voice. Keep responses CONCISE — usually
1-2 sentences. Use natural spoken English: short, direct, no markdown,
no bullet points, no headings. Pronounce numbers as words where it sounds
natural (say "one-thirty-two five" not "1.32.5") but normal numbers are fine.

Available tools (call only when the driver explicitly asks for them):
- check_car_limit: when the driver asks "am I at the limit", "is this fast
  enough", "how is my braking", etc.
- track_detail_for_guide: when the driver asks to be guided on the track
  or for a driving guide. After calling, summarize the guidance in 1-2
  spoken sentences — do not list everything verbatim.

For general racing questions, answer briefly and confidently without
calling tools.
"""


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------


@dataclass
class VoiceSessionConfig:
    """Per-WS-session configuration."""

    track_name: str = "Unknown"
    car_name: str = "Unknown"
    user_id: Optional[str] = None
    voice: Optional[str] = None  # Kokoro voice override


def _build_tool_schemas():
    """Return Pipecat FunctionSchemas for the racing-coach tools.

    Mirrors the schemas in AIService.get_available_functions() so the voice
    and text paths advertise the same capabilities to the LLM.

    Deferred import — only loaded when a voice session is actually built.
    """
    from pipecat.adapters.schemas.function_schema import FunctionSchema

    check_car_limit_schema = FunctionSchema(
        name="check_car_limit",
        description=(
            "Check whether the driver is pushing the car within its optimal limits "
            "based on telemetry. Call when the driver asks 'am I at the limit', "
            "'is this fast enough', 'how is my braking', etc."
        ),
        properties={
            "session_id": {
                "type": "string",
                "description": "Session ID for telemetry analysis.",
            },
            "data_types": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Telemetry channels to analyze (speed, acceleration, braking, steering).",
            },
        },
        required=["session_id"],
    )

    track_detail_schema = FunctionSchema(
        name="track_detail_for_guide",
        description=(
            "Start the on-track guidance flow. Call when the driver asks to be "
            "guided on the track, 'guide me', 'help me drive this track', etc."
        ),
        properties={
            "track_name": {
                "type": "string",
                "description": "Name of the track to retrieve data for.",
            },
        },
        required=["track_name"],
    )

    return [check_car_limit_schema, track_detail_schema]


def _make_tool_handler(ai_service, session_config: "VoiceSessionConfig"):
    """Build a per-session async handler that delegates to AIService._execute_function.

    Closes over `session_config` so per-session context (track/car) is
    forwarded to the existing tool implementations exactly as the HTTP
    path does.
    """
    async def handle_tool_call(params):
        function_name = params.function_name
        arguments = params.arguments or {}
        # Build the same `context` dict the HTTP path passes.
        context = {
            "track_name": session_config.track_name,
            "car_name": session_config.car_name,
            "user_id": session_config.user_id,
        }

        try:
            result = await ai_service._execute_function(function_name, arguments, context)
        except Exception as exc:
            LOGGER.exception("Voice tool %s failed", function_name)
            await params.result_callback({"error": str(exc)})
            return

        # Split public vs side-product keys (underscore-prefixed). Send only
        # the public ones back to the LLM. Side products would normally drive
        # UI features in the text-chat path; voice has no UI side-channel today.
        if isinstance(result, dict):
            public = {k: v for k, v in result.items() if not k.startswith("_")}
            side_products = {k: v for k, v in result.items() if k.startswith("_")}
            if side_products:
                LOGGER.info(
                    "Voice tool %s produced side products (not forwarded to client): %s",
                    function_name, list(side_products.keys()),
                )
            payload = public if public else result
        else:
            payload = result

        await params.result_callback(payload)

    return handle_tool_call


async def build_voice_pipeline_task(
    websocket: Any,
    session_config: VoiceSessionConfig,
):
    """Build a Pipecat PipelineTask bound to the given WebSocket.

    Returns the task; caller is responsible for running it via
    `PipelineRunner.run(task)`.

    Raises ImportError if pipecat-ai or faster-whisper aren't available
    — the caller should map this to an explicit error frame to the WS.
    """
    # Deferred imports — see module docstring.
    from openai import AsyncOpenAI
    from pipecat.adapters.schemas.tools_schema import ToolsSchema
    from pipecat.audio.vad.silero import SileroVADAnalyzer
    from pipecat.pipeline.pipeline import Pipeline
    from pipecat.pipeline.task import PipelineParams, PipelineTask
    from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
    from pipecat.services.openai.llm import OpenAILLMService
    from pipecat.services.whisper.stt import WhisperSTTService
    from pipecat.transports.network.fastapi_websocket import (
        FastAPIWebsocketParams,
        FastAPIWebsocketTransport,
    )

    from app.services.ai_service import AIService
    from app.services.voice.pipecat_kokoro import build_kokoro_processor

    LOGGER.info(
        "Building voice pipeline (track=%s car=%s user=%s)",
        session_config.track_name, session_config.car_name, session_config.user_id,
    )

    # --- Transport ---
    # The FastAPI websocket transport runs the WS lifecycle (accept, recv,
    # send, close) inside our existing endpoint.
    #
    # WIRE FORMAT (must match the frontend in use-voice-conversation.ts):
    #   - serializer=None: client sends/receives raw PCM16 mono bytes.
    #     The browser AudioWorklet posts Int16Array.buffer over the WS,
    #     and we play received PCM16 via a 24kHz AudioContext.
    #   - audio_in_sample_rate=16000: frontend captures at 16kHz to match
    #     Whisper's preferred rate. AudioContext({ sampleRate: 16000 }).
    #   - audio_out_sample_rate=24000: Kokoro's native rate. The frontend's
    #     playback AudioContext uses 24kHz so no client-side resampling needed.
    transport = FastAPIWebsocketTransport(
        websocket=websocket,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            audio_in_sample_rate=16000,
            audio_out_sample_rate=settings.kokoro_sample_rate,
            add_wav_header=False,
            vad_enabled=True,
            vad_analyzer=SileroVADAnalyzer(),
            vad_audio_passthrough=True,
            serializer=None,
        ),
    )

    # --- STT (faster-whisper) ---
    # Phase 3 starting point: small English model on whichever device is
    # available. Pipecat's WhisperSTTService wraps faster-whisper directly.
    stt = WhisperSTTService(
        model="small.en",
        # device="cuda" if available; Pipecat auto-detects via faster-whisper.
    )

    # --- LLM (local llama-server via OpenAI-compatible client) ---
    llama_client = AsyncOpenAI(
        base_url=settings.llama_server_url,
        api_key="not-needed",
    )
    llm = OpenAILLMService(
        client=llama_client,
        model=settings.llama_model_name,
        params=OpenAILLMService.InputParams(
            temperature=0.7,
            max_tokens=300,  # Voice responses are short — discourage rambling.
        ),
    )

    # --- Tool calling (Phase 3b) ---
    # Build schemas + register handlers that delegate to the existing
    # AIService._execute_function. Voice and text paths share the same tool
    # implementations, so behavior stays consistent.
    ai_service = AIService()
    tool_schemas = _build_tool_schemas()
    tools = ToolsSchema(standard_tools=tool_schemas)

    tool_handler = _make_tool_handler(ai_service, session_config)
    for schema in tool_schemas:
        llm.register_function(schema.name, tool_handler)

    # System message + context object (Pipecat's history aggregator).
    system_message = _VOICE_COACH_PROMPT_TEMPLATE.format(
        track_name=session_config.track_name,
        car_name=session_config.car_name,
    )
    context = OpenAILLMContext(
        messages=[{"role": "system", "content": system_message}],
        tools=tools,
    )
    context_aggregator = llm.create_context_aggregator(context)

    # --- TTS (Kokoro via custom Pipecat processor) ---
    KokoroProcessor = build_kokoro_processor()
    tts = KokoroProcessor(sample_rate=settings.kokoro_sample_rate)

    # --- Pipeline composition ---
    pipeline = Pipeline([
        transport.input(),
        stt,
        context_aggregator.user(),
        llm,
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

    return task


async def run_voice_session(websocket: Any, session_config: VoiceSessionConfig) -> None:
    """Bind a Pipecat pipeline to `websocket` and run it to completion.

    Returns when the WS closes or the pipeline exits. Caller is responsible
    for any auth/lifecycle concerns around `websocket`.
    """
    # Deferred import.
    from pipecat.pipeline.runner import PipelineRunner

    task = await build_voice_pipeline_task(websocket, session_config)
    runner = PipelineRunner()
    try:
        await runner.run(task)
    finally:
        LOGGER.info("Voice session ended (user=%s)", session_config.user_id)
