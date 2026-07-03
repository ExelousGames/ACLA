"""Per-connection WS tool relay for the voice pipeline.

Backend and frontend share the same ``/voice/stream`` WebSocket. Binary
frames carry PCM audio for Pipecat. Text frames carry JSON control messages
for frontend tool calls/results, typed user text, and session context.

Frontend tool calls are fire-and-forget from the AI service perspective. The
relay sends a ``tool_call`` frame to the frontend and does not wait for a
matching result. Later AI-visible data should come back through
``tool_result`` / ``user_text`` / ``session_context`` frames.
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any, Awaitable, Callable, Dict, Optional

LOGGER = logging.getLogger(__name__)
MAX_AI_VISIBLE_TOOL_PAYLOAD_CHARS = 64 * 1024

SendText = Callable[[str], Awaitable[None]]
UserTextSink = Callable[[str], Any]
SessionContextSink = Callable[[Dict[str, Any]], Any]


class _ConnectionState:
    """Per-connection state held by the relay."""

    __slots__ = (
        "send_text",
        "user_text_sink",
        "session_context_sink",
    )

    def __init__(
        self,
        send_text: SendText,
        user_text_sink: UserTextSink,
        session_context_sink: Optional[SessionContextSink] = None,
    ) -> None:
        self.send_text = send_text
        self.user_text_sink = user_text_sink
        self.session_context_sink = session_context_sink


class ToolRelay:
    """Process-wide registry for active voice connections."""

    def __init__(self, max_ai_visible_tool_payload_chars: int = MAX_AI_VISIBLE_TOOL_PAYLOAD_CHARS) -> None:
        self._by_conn: Dict[int, _ConnectionState] = {}
        self._max_ai_visible_tool_payload_chars = max_ai_visible_tool_payload_chars

    def bind(
        self,
        conn: Any,
        send_text: SendText,
        user_text_sink: UserTextSink,
        session_context_sink: Optional[SessionContextSink] = None,
    ) -> None:
        """Register a connection and its inbound text-frame sinks."""
        self._by_conn[id(conn)] = _ConnectionState(
            send_text, user_text_sink, session_context_sink,
        )

    def unbind(self, conn: Any) -> None:
        """Drop the registration for a connection."""
        self._by_conn.pop(id(conn), None)

    async def send_tool_call(
        self,
        conn: Any,
        name: str,
        arguments: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """Send a frontend ``tool_call`` frame without awaiting a result.

        Returns the generated call id on successful send, or ``None`` if the
        connection is unavailable or the send fails. The return value is only
        for backend diagnostics and UI metadata; it is not LLM-visible
        frontend tool data.
        """
        state = self._by_conn.get(id(conn))
        if state is None:
            LOGGER.warning("tool_relay: no bound connection for %s", name)
            return None

        call_id = uuid.uuid4().hex
        frame = json.dumps({
            "type": "tool_call",
            "id": call_id,
            "name": name,
            "arguments": arguments or {},
        })

        try:
            await state.send_text(frame)
        except Exception as exc:
            LOGGER.warning("tool_relay: send_text failed for %s: %s", name, exc)
            return None
        return call_id

    def handle_text_frame(self, conn: Any, payload: Dict[str, Any]) -> None:
        """Route one inbound text frame.

        Tool payload frames are serialized to text and sent through the same
        AI-visible path as typed user text.
        """
        state = self._by_conn.get(id(conn))
        if state is None:
            return

        frame_type = payload.get("type")

        if frame_type == "tool_result":
            try:
                state.user_text_sink(self._serialize_ai_visible_tool_payload(payload))
            except Exception:
                LOGGER.exception("tool_relay: user_text_sink raised for %s", frame_type)
            return

        if frame_type == "user_text":
            text = str(payload.get("text") or "").strip()
            if not text:
                return
            try:
                state.user_text_sink(text)
            except Exception:
                LOGGER.exception("tool_relay: user_text_sink raised")
            return

        if frame_type == "session_context":
            if state.session_context_sink is None:
                LOGGER.warning("tool_relay: session_context frame received but no sink bound")
                return
            session_context = payload.get("session_context") or {}
            if not isinstance(session_context, dict):
                LOGGER.warning("tool_relay: dropped non-object session_context frame")
                return
            try:
                state.session_context_sink(session_context)
            except Exception:
                LOGGER.exception("tool_relay: session_context_sink raised")
            return

        LOGGER.warning("tool_relay: unknown frame type %r", frame_type)

    def _serialize_ai_visible_tool_payload(self, payload: Dict[str, Any]) -> str:
        serialized = json.dumps(
            payload,
            ensure_ascii=True,
            sort_keys=True,
            default=str,
        )
        if len(serialized) <= self._max_ai_visible_tool_payload_chars:
            return serialized

        frame_type = payload.get("type")
        name = payload.get("name")
        compact_payload = {
            "type": frame_type,
            "id": payload.get("id"),
            "name": name,
            "ai_visible_payload_truncated": True,
            "original_payload_chars": len(serialized),
            "max_payload_chars": self._max_ai_visible_tool_payload_chars,
            "message": (
                "Tool payload omitted because it exceeded the AI-visible size cap. "
                "Use a compact tool result or a server-side classifier path."
            ),
        }
        compact_payload["result"] = {
            "status": "omitted",
            "message": compact_payload["message"],
        }

        LOGGER.warning(
            "tool_relay: truncated oversized %s payload name=%r chars=%d cap=%d",
            frame_type,
            name,
            len(serialized),
            self._max_ai_visible_tool_payload_chars,
        )
        return json.dumps(
            compact_payload,
            ensure_ascii=True,
            sort_keys=True,
            default=str,
        )


_RELAY = ToolRelay()


def get_relay() -> ToolRelay:
    """Return the process-wide ``ToolRelay`` instance."""
    return _RELAY
