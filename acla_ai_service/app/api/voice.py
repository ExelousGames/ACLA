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

from app.services.voice import get_kokoro_service

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
    track_name: str = Query("Unknown"),
    car_name: str = Query("Unknown"),
    user_id: Optional[str] = Query(None),
):
    """WebSocket endpoint for full bidirectional voice conversation.

    The client sends mic audio frames (PCM16 mono, raw — Pipecat handles
    the wire framing via its Protobuf serializer). The server runs:
        VAD → Whisper STT → llama-server LLM → Kokoro TTS
    and streams audio back over the same connection.

    Query parameters provide per-session context (track/car/user) without
    requiring an initial message protocol.
    """
    await websocket.accept()

    # Deferred imports — keeps the rest of the API importable even when
    # pipecat isn't installed in the running container.
    try:
        from app.services.voice.pipecat_pipeline import (
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
        track_name=track_name,
        car_name=car_name,
        user_id=user_id,
    )

    LOGGER.info(
        "Voice WS connected (track=%s car=%s user=%s)",
        track_name, car_name, user_id,
    )

    try:
        await run_voice_session(websocket, config)
    except WebSocketDisconnect:
        LOGGER.info("Voice WS client disconnected (user=%s)", user_id)
    except Exception:
        LOGGER.exception("Voice session crashed (user=%s)", user_id)
        try:
            await websocket.close(code=1011, reason="voice session error")
        except Exception:
            pass
