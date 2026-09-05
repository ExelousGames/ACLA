"""Voice synthesis endpoints.

Phase 2: `POST /voice/synthesize` (text → WAV), `GET /voice/voices`,
`GET /voice/health`.

Phase 3: `WS /voice/stream` — full bidirectional voice conversation via a
Pipecat pipeline (Silero VAD → Whisper STT → selected chat LLM → Kokoro TTS).
Each connection spawns its own pipeline; interruption is built-in.
"""

from __future__ import annotations

import asyncio
from contextlib import suppress
import json
import logging
from typing import Any, Dict, Optional, Tuple

from fastapi import APIRouter, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import Response
from pydantic import BaseModel, Field

from app.chat_llm import (
    normalize_chat_llm_model,
    parse_chat_llm_model_selector,
)
from app.voice import get_speech_core
from app.voice.session_modes import (
    VALID_CHATBOT_SESSION_MODES,
)
from app.voice.session_ai_tool_service import (
    SessionAIToolService,
    SessionToolCatalogError,
)
from app.voice.tool_relay import normalize_voice_session_context

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
        service = get_speech_core()
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
        service = get_speech_core()
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
    """Report loaded model counts and availability without warming either pool."""
    service = get_speech_core()
    tts = service.tts.stats
    return {
        "loaded": tts["total"] >= tts["minimum"],
        "engine": "kokoro-onnx",
        "pools": {"tts": tts, "stt": service.stt.stats},
    }


# ----------------------------------------------------------------------
# Phase 3 — Bidirectional voice conversation over WebSocket
# ----------------------------------------------------------------------


@router.websocket("/stream")
async def voice_stream(
    websocket: WebSocket,
    session_id: Optional[str] = Query(None),
    user_id: Optional[str] = Query(None),
    chat_llm_model: Optional[str] = Query(None),
    chat_session_action: Optional[str] = Query(None),
    chat_session_id: Optional[str] = Query(None),
):
    """WebSocket endpoint for full bidirectional voice conversation.

    Single chat surface for the racing engineer. The connection carries
    BOTH:

    * **Binary frames** — raw PCM16 mono audio (mic in / Kokoro TTS out).
      Consumed by Pipecat's transport unchanged.
    * **Text frames** — JSON tool-relay messages (``tool_call`` /
      ``tool_result`` / ``user_text`` /
      ``session_context``) — see
      :mod:`app.voice.tool_relay`. Routed off the audio path before
      Pipecat sees them.

    Pipeline (binary frames only):
        VAD → Whisper STT → selected chat LLM → Kokoro TTS

    ``session_id`` identifies optional telemetry context. The separately
    issued ``chat_session_id`` owns reconnectable LLM conversation history.
    """
    await websocket.accept()

    from app.voice.chat_sessions import (
        ChatSessionError,
        get_chat_session_registry,
    )

    try:
        action, owner_user_id, requested_chat_session_id = (
            _validate_chat_session_request(
                chat_session_action,
                chat_session_id,
                user_id,
            )
        )
    except ChatSessionError as exc:
        await _reject_chat_session_request(websocket, exc)
        return

    registry = get_chat_session_registry()
    chat_session = None
    resumed = action == "resume"
    if resumed:
        try:
            chat_session = registry.resume_attached(
                requested_chat_session_id,
                owner_user_id,
            )
        except ChatSessionError as exc:
            await _reject_chat_session_request(websocket, exc)
            return

    try:
        selected_chat_llm_model = normalize_chat_llm_model(chat_llm_model)
        if selected_chat_llm_model is not None:
            try:
                parse_chat_llm_model_selector(selected_chat_llm_model)
            except RuntimeError as exc:
                await websocket.send_json({
                    "type": "error",
                    "message": str(exc),
                    "error_type": "InvalidChatLLMModel",
                })
                await websocket.close(code=1008, reason="invalid chat_llm_model")
                return

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

        # The first text frame on every connection declares current context.
        # Audio frames before it are dropped.
        try:
            session_context = await _await_session_info(
                websocket,
                timeout=5.0,
            )
        except _HandshakeError as exc:
            LOGGER.warning(
                "Voice WS handshake failed (user=%s): %s", owner_user_id, exc,
            )
            try:
                await websocket.send_json({
                    "type": "error",
                    "message": str(exc),
                    "error_type": "HandshakeError",
                })
            except Exception:
                pass
            try:
                await websocket.close(code=1002, reason="session_info handshake failed")
            except Exception:
                pass
            return

        try:
            session_tools = await SessionAIToolService().get_session_tools(
                session_context,
            )
        except SessionToolCatalogError as exc:
            LOGGER.error(
                "Voice session-tool lookup failed (user=%s): %s",
                owner_user_id,
                exc,
            )
            try:
                await websocket.send_json({
                    "type": "error",
                    "message": str(exc),
                    "error_type": "SessionToolCatalogError",
                })
            except Exception:
                pass
            try:
                await websocket.close(code=1011, reason="session tool catalog error")
            except Exception:
                pass
            return

        if not resumed:
            chat_session = registry.create_attached(
                owner_user_id,
                session_context,
            )
        else:
            registry.update_session_context(
                chat_session.chat_session_id,
                session_context,
            )

        await websocket.send_json({
            "type": "chat_session_ready",
            "chat_session_id": chat_session.chat_session_id,
            "resumed": resumed,
        })

        config = VoiceSessionConfig(
            chat_session_id=chat_session.chat_session_id,
            committed_history=chat_session.committed_history,
            session_id=session_id,
            session_context=session_context,
            user_id=owner_user_id,
            chat_llm_model=selected_chat_llm_model,
        )

        # Construct the server-side tool executor in the inbound adapter band.
        from app.racing_engineer import AIService
        ai_service = AIService(
            chat_llm_model=selected_chat_llm_model,
        )
        tool_executor = ai_service._execute_function

        filtered_ws = _TextFilteringWebSocket(
            websocket,
            chat_session.chat_session_id,
        )

        LOGGER.info(
            "Voice WS connected (chat_session=%s telemetry_session=%s user=%s "
            "chat_llm_model=%s session_tools=%d resumed=%s)",
            chat_session.chat_session_id,
            session_id,
            owner_user_id,
            selected_chat_llm_model or "default",
            len(session_tools),
            resumed,
        )

        await run_voice_session(
            filtered_ws, config, tool_executor,
            session_tools=session_tools,
        )
    except WebSocketDisconnect:
        LOGGER.info("Voice WS client disconnected (user=%s)", owner_user_id)
    except Exception:
        LOGGER.exception("Voice session crashed (user=%s)", owner_user_id)
        try:
            await websocket.close(code=1011, reason="voice session error")
        except Exception:
            pass
    finally:
        if chat_session is not None:
            registry.detach(chat_session.chat_session_id)


def _validate_chat_session_request(
    chat_session_action: Optional[str],
    chat_session_id: Optional[str],
    user_id: Optional[str],
) -> Tuple[str, str, Optional[str]]:
    """Validate the strict create/resume query contract before the handshake."""
    from app.voice.chat_sessions import ChatSessionError

    if not isinstance(user_id, str) or not user_id.strip():
        raise ChatSessionError(
            "UserIdRequired",
            "user_id is required for voice chat sessions.",
        )
    owner_user_id = user_id.strip()

    if chat_session_action is None:
        raise ChatSessionError(
            "ChatSessionActionRequired",
            "chat_session_action is required and must be 'create' or 'resume'.",
        )
    if chat_session_action not in {"create", "resume"}:
        raise ChatSessionError(
            "InvalidChatSessionAction",
            "chat_session_action must be 'create' or 'resume'.",
        )

    if chat_session_action == "create":
        if chat_session_id is not None:
            raise ChatSessionError(
                "ChatSessionIdNotAllowed",
                "chat_session_id must be absent when creating a chat session.",
            )
        return chat_session_action, owner_user_id, None

    if not isinstance(chat_session_id, str) or not chat_session_id.strip():
        raise ChatSessionError(
            "ChatSessionIdRequired",
            "chat_session_id is required when resuming a chat session.",
        )
    return chat_session_action, owner_user_id, chat_session_id.strip()


async def _reject_chat_session_request(websocket: WebSocket, error: Any) -> None:
    """Send one explicit policy error and close a rejected connection."""
    try:
        await websocket.send_json({
            "type": "error",
            "message": str(error),
            "error_type": error.error_type,
        })
    except Exception:
        pass
    try:
        await websocket.close(code=1008, reason="chat session policy violation")
    except Exception:
        pass


class _HandshakeError(Exception):
    """Raised when the session_info handshake fails (timeout, bad frame, etc.)."""


async def _await_session_info(
    websocket: WebSocket, *, timeout: float,
) -> Dict[str, Any]:
    """Receive and parse the first text frame as ``session_info``.

    Returns compact frontend view/session state. Raises
    :class:`_HandshakeError` on timeout, non-text first frame, malformed
    JSON, a wrong ``type``, or invalid session context.

    Per-session — does not block the event loop or other sessions. Any
    binary frames that arrive before the handshake are dropped.
    """
    deadline = asyncio.get_running_loop().time() + timeout
    while True:
        remaining = deadline - asyncio.get_running_loop().time()
        if remaining <= 0:
            raise _HandshakeError("Timed out waiting for session_info handshake")
        try:
            msg = await asyncio.wait_for(websocket.receive(), timeout=remaining)
        except asyncio.TimeoutError as exc:
            raise _HandshakeError(
                "Timed out waiting for session_info handshake",
            ) from exc

        if msg.get("type") == "websocket.disconnect":
            raise _HandshakeError("Client disconnected before sending session_info")

        text = msg.get("text")
        if text is None:
            # Stray binary frame before handshake — drop and keep waiting.
            continue

        try:
            payload = json.loads(text)
        except Exception as exc:
            raise _HandshakeError(f"session_info: bad JSON ({exc})") from exc

        if not isinstance(payload, dict) or payload.get("type") != "session_info":
            raise _HandshakeError(
                f"First text frame must have type='session_info' "
                f"(got {payload.get('type') if isinstance(payload, dict) else type(payload).__name__})"
            )

        session_context = payload.get("session_context")
        if not isinstance(session_context, dict):
            raise _HandshakeError("session_info: 'session_context' must be an object")
        context_session_mode = session_context.get("session_mode")
        if (
            not isinstance(context_session_mode, str)
            or context_session_mode not in VALID_CHATBOT_SESSION_MODES
        ):
            raise _HandshakeError(
                "session_info: 'session_context.session_mode' must be "
                f"{', '.join(sorted(VALID_CHATBOT_SESSION_MODES))}"
            )
        agent_mode = session_context.get("agent_mode")
        if agent_mode is not None and (
            not isinstance(agent_mode, str)
            or agent_mode not in {
                "track_guide",
                "overtake",
                "live_performance_analyst",
            }
        ):
            raise _HandshakeError(
                "session_info: 'session_context.agent_mode' is invalid"
            )
        session_context = normalize_voice_session_context(session_context)
        return session_context


class _TextFilteringWebSocket:
    """Proxy around a Starlette WebSocket that re-routes inbound text frames
    to :mod:`app.voice.tool_relay` while letting binary frames pass through
    to Pipecat unchanged.

    A dedicated pump owns the underlying ``receive`` loop. Text frames are
    routed immediately, even when Pipecat is not currently pulling microphone
    audio, while binary/disconnect frames are queued for Pipecat.

    Text routing uses the server-issued chat session ID. The WebSocket proxy
    itself is never used as process-wide relay identity.
    """

    def __init__(self, ws: WebSocket, chat_session_id: str) -> None:
        self._ws = ws
        self._chat_session_id = chat_session_id
        self._pipecat_frames: asyncio.Queue[dict] = asyncio.Queue()
        self._receive_task: Optional[asyncio.Task] = None

    # Delegate everything we don't override (send_bytes, send_text, accept,
    # close, headers, query_params, state, etc.).
    def __getattr__(self, name: str):
        return getattr(self._ws, name)

    # ---- receive path: route text frames into the relay --------------------

    def start_text_control_pump(self) -> None:
        """Start routing text control frames independently of audio reads."""
        if self._receive_task is not None and not self._receive_task.done():
            return
        self._receive_task = asyncio.create_task(self._receive_loop())

    async def stop_text_control_pump(self) -> None:
        """Stop the control pump when the voice session ends."""
        task = self._receive_task
        if task is None or task.done():
            return
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task

    async def _receive_loop(self) -> None:
        import json as _json
        from app.voice.tool_relay import get_relay

        relay = get_relay()
        try:
            while True:
                msg = await self._ws.receive()
                text = msg.get("text")
                if text is not None:
                    try:
                        payload = _json.loads(text)
                    except Exception:
                        LOGGER.exception("voice WS: bad JSON text frame")
                        continue
                    relay.handle_text_frame(self._chat_session_id, payload)
                    continue

                await self._pipecat_frames.put(msg)
                if msg.get("type") == "websocket.disconnect":
                    return
        except asyncio.CancelledError:
            raise
        except Exception:
            LOGGER.exception("voice WS receive pump failed")
            await self._pipecat_frames.put({"type": "websocket.disconnect"})

    async def receive(self) -> dict:
        self.start_text_control_pump()
        return await self._pipecat_frames.get()

    async def receive_bytes(self) -> bytes:
        msg = await self.receive()
        if msg.get("bytes") is not None:
            return msg["bytes"]
        raise WebSocketDisconnect(code=msg.get("code", 1000))

    async def receive_text(self) -> str:
        msg = await self.receive()
        if msg.get("text") is not None:
            return msg["text"]
        raise WebSocketDisconnect(code=msg.get("code", 1000))

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
