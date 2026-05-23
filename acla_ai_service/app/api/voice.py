"""Voice synthesis endpoints.

Phase 2: `POST /voice/synthesize` (text → WAV), `GET /voice/voices`,
`GET /voice/health`.

Phase 3: `WS /voice/stream` — full bidirectional voice conversation via a
Pipecat pipeline (Silero VAD → Whisper STT → llama-server LLM → Kokoro TTS).
Each connection spawns its own pipeline; interruption is built-in.
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import Response
from pydantic import BaseModel, Field

from app.voice import get_kokoro_service

LOGGER = logging.getLogger(__name__)

router = APIRouter(prefix="/voice", tags=["voice"])


# ----------------------------------------------------------------------
# Request models
# ----------------------------------------------------------------------


class SynthesizeRequest(BaseModel):
    """Body for POST /voice/synthesize."""

    text: str = Field(..., min_length=1, max_length=4000)
    voice: Optional[str] = Field(
        None,
        description="Voice ID (e.g. 'af_bella'). Defaults to settings.kokoro_default_voice.",
    )
    speed: float = Field(
        1.0,
        ge=0.5,
        le=2.0,
        description="Speech rate multiplier. 1.0 = normal, 0.5 = half-speed, 2.0 = double.",
    )
    language: str = Field(
        "en-us",
        description="Language code passed to Kokoro (e.g. 'en-us', 'en-gb').",
    )


# ----------------------------------------------------------------------
# Endpoints
# ----------------------------------------------------------------------


@router.post(
    "/synthesize",
    responses={
        200: {
            "content": {"audio/wav": {}},
            "description": "Synthesized speech as a WAV file.",
        }
    },
)
async def synthesize(req: SynthesizeRequest) -> Response:
    """Synthesize the given text to WAV audio.

    Non-streaming: the full WAV is buffered server-side and returned in one
    response. Client plays it via `HTMLAudioElement` (Electron renderer).

    Latency: ~300ms on CPU for a short sentence, ~80ms on GPU.
    """
    try:
        service = await get_kokoro_service()
        wav_bytes = await service.synthesize(
            req.text,
            voice=req.voice,
            speed=req.speed,
            language=req.language,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        LOGGER.exception("Kokoro synthesize failed")
        raise HTTPException(
            status_code=500,
            detail=f"TTS synthesis failed: {type(exc).__name__}: {exc}",
        ) from exc

    return Response(
        content=wav_bytes,
        media_type="audio/wav",
        headers={
            "Content-Disposition": 'inline; filename="speech.wav"',
            "Cache-Control": "no-store",
        },
    )


@router.get("/voices")
async def list_voices() -> dict:
    """List available Kokoro voice IDs."""
    try:
        service = await get_kokoro_service()
        voices = await service.list_voices()
    except Exception as exc:
        LOGGER.exception("Failed to list Kokoro voices")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list voices: {type(exc).__name__}: {exc}",
        ) from exc

    return {"voices": voices, "count": len(voices)}


@router.get("/health")
async def voice_health() -> dict:
    """Reports whether the Kokoro engine has been loaded yet.

    Does NOT trigger a load — that happens on the first synthesize() call.
    Use this to distinguish "engine cold" (first request will be slow) from
    "engine warm" (sub-second).
    """
    service = await get_kokoro_service()
    return {
        "loaded": await service.is_ready(),
        "engine": "kokoro-onnx",
    }


# ----------------------------------------------------------------------
# Phase 3 — Bidirectional voice conversation over WebSocket
# ----------------------------------------------------------------------


@router.websocket("/stream")
async def voice_stream(
    websocket: WebSocket,
    session_id: Optional[str] = Query(None),
    user_id: Optional[str] = Query(None),
):
    """WebSocket endpoint for full bidirectional voice conversation.

    Single chat surface for the racing engineer. The connection carries
    BOTH:

    * **Binary frames** — raw PCM16 mono audio (mic in / Kokoro TTS out).
      Consumed by Pipecat's transport unchanged.
    * **Text frames** — JSON tool-relay messages (``tool_call`` /
      ``tool_result`` / ``tool_error`` / ``observation``) — see
      :mod:`app.voice.tool_relay`. Routed off the audio path before
      Pipecat sees them.

    Pipeline (binary frames only):
        VAD → Whisper STT → llama-server LLM → Kokoro TTS

    Query params kept minimal — only what the relay needs at connect
    time. ``track_name`` / ``car_name`` are no longer query params; the
    LLM fetches them on demand via the ``get_session_info`` frontend
    tool. See the plan's "everything is pulled on demand" principle.
    """
    await websocket.accept()

    # Deferred imports — keeps the rest of the API importable even when
    # pipecat isn't installed in the running container.
    try:
        from app.voice.pipecat_pipeline import (
            VoiceSessionConfig,
            run_voice_session,
        )
    except ImportError as exc:
        LOGGER.error("Pipecat / faster-whisper not installed: %s", exc)
        await websocket.send_json({
            "type": "error",
            "message": (
                "Voice conversation is not available in this environment "
                "(pipecat-ai or faster-whisper not installed)."
            ),
            "error_type": "DependencyMissing",
        })
        await websocket.close(code=1011, reason="voice dependency missing")
        return

    config = VoiceSessionConfig(
        session_id=session_id,
        user_id=user_id,
    )

    # Construct the tool executor here, in the inbound-adapter band, so
    # app/voice/ never imports from app/pipelines/ (see .importlinter
    # contract voice-no-pipeline-or-api).
    from app.pipelines.chat import AIService
    ai_service = AIService()
    tool_executor = ai_service._execute_function

    # Wrap the WS so inbound text frames go to the tool relay and only
    # binary frames reach Pipecat. The relay singleton is bound to the
    # underlying ``websocket`` (identity is keyed by id(websocket)) inside
    # ``build_voice_pipeline_task``.
    filtered_ws = _TextFilteringWebSocket(websocket)

    LOGGER.info(
        "Voice WS connected (session=%s user=%s)", session_id, user_id,
    )

    try:
        await run_voice_session(filtered_ws, config, tool_executor)
    except WebSocketDisconnect:
        LOGGER.info("Voice WS client disconnected (user=%s)", user_id)
    except Exception:
        LOGGER.exception("Voice session crashed (user=%s)", user_id)
        try:
            await websocket.close(code=1011, reason="voice session error")
        except Exception:
            pass


class _TextFilteringWebSocket:
    """Proxy around a Starlette WebSocket that re-routes inbound text frames
    to :mod:`app.voice.tool_relay` while letting binary frames pass through
    to Pipecat unchanged.

    Pipecat's :class:`FastAPIWebsocketTransport` runs its own ``receive``
    loop over the WS. By interposing this proxy we keep Pipecat's audio
    contract intact (it only ever sees binary frames) and turn the same
    connection into a JSON RPC channel for tool relay traffic.

    Identity is preserved via :py:meth:`__hash__` / :py:meth:`__eq__` so
    callers can use either the proxy or the underlying WS as a dict key
    interchangeably (the relay binds against the proxy; downstream code
    that compares identity still works).
    """

    def __init__(self, ws: WebSocket) -> None:
        self._ws = ws

    # Delegate everything we don't override (send_bytes, send_text, accept,
    # close, headers, query_params, state, etc.).
    def __getattr__(self, name: str):
        return getattr(self._ws, name)

    def __hash__(self) -> int:  # so id(proxy) is stable and unique per WS
        return id(self)

    def __eq__(self, other: object) -> bool:
        return other is self

    # ---- receive path: route text frames into the relay --------------------

    async def receive(self) -> dict:
        """Return the next inbound frame, swallowing any text frames the
        upstream is already routed elsewhere."""
        import json as _json
        from app.voice.tool_relay import get_relay
        relay = get_relay()
        while True:
            msg = await self._ws.receive()
            text = msg.get("text")
            if text is not None:
                try:
                    payload = _json.loads(text)
                except Exception:
                    LOGGER.exception("voice WS: bad JSON text frame")
                    continue
                # Pass ``self`` (the proxy) — the relay binds against this
                # same identity inside build_voice_pipeline_task, so the
                # call_id-to-future map and observation_sink find it.
                relay.handle_text_frame(self, payload)
                continue
            return msg

    async def receive_bytes(self) -> bytes:
        msg = await self.receive()
        if "bytes" in msg:
            return msg["bytes"]
        # Disconnect or unexpected — re-raise via the underlying WS so
        # Pipecat sees the normal Starlette failure path.
        return await self._ws.receive_bytes()

    async def receive_text(self) -> str:
        # Defensive — Pipecat is configured for binary frames, so this
        # should rarely fire. If it does, route the same way as receive().
        msg = await self.receive()
        if "text" in msg:
            return msg["text"]
        return await self._ws.receive_text()

    async def iter_bytes(self):
        try:
            while True:
                yield await self.receive_bytes()
        except Exception:
            return

    async def iter_text(self):
        try:
            while True:
                yield await self.receive_text()
        except Exception:
            return
