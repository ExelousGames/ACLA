"""SSE event protocol for /naturallanguagequery/stream (Phase 2.5).

Single source of truth shared with the frontend (mirrored in
acla_front/src/views/lap-analysis/ai-chat/streaming-chat.ts).

Each event is one Server-Sent Event whose `data:` field is a JSON object
of the shapes below. The connection terminates after a `done` or `error`
event.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Literal, Optional

EventType = Literal[
    "token",       # one or more characters of the user-visible answer
    "audio",       # base64-encoded WAV chunk for one synthesized sentence
    "tool_start",  # a tool/function call is about to execute
    "tool_end",    # tool/function call finished (success or failure)
    "done",        # generation finished; includes final side_products + messages
    "error",       # fatal error; the stream is closed
]


def sse_event(event_type: EventType, payload: Dict[str, Any]) -> str:
    """Format a single SSE event line for `StreamingResponse`.

    Output ends with a blank line per the SSE spec, so concatenating
    these strings produces a valid event stream.
    """
    body = {"type": event_type, **payload}
    return f"data: {json.dumps(body, ensure_ascii=False)}\n\n"


# ---------------------------------------------------------------------------
# Helpers for individual event types — keep the call sites tidy.
# ---------------------------------------------------------------------------


def event_token(text: str) -> str:
    """Token delta — append `text` to the current assistant message bubble."""
    return sse_event("token", {"text": text})


def event_audio(
    sentence: str,
    wav_b64: str,
    voice: Optional[str] = None,
) -> str:
    """Synthesized audio for one complete sentence.

    `wav_b64` is a standard WAV file encoded as base64 — the same shape
    `/voice/synthesize` returns, just over SSE. The frontend decodes and
    queues it in the audio playback queue.
    """
    payload: Dict[str, Any] = {"sentence": sentence, "wav_b64": wav_b64}
    if voice is not None:
        payload["voice"] = voice
    return sse_event("audio", payload)


def event_tool_start(name: str, arguments: Dict[str, Any]) -> str:
    """Tool invocation is about to run. Frontend can show a 'thinking' state."""
    return sse_event("tool_start", {"name": name, "arguments": arguments})


def event_tool_end(name: str, ok: bool, error: Optional[str] = None) -> str:
    """Tool invocation completed. `ok=False` means it errored."""
    payload: Dict[str, Any] = {"name": name, "ok": ok}
    if error is not None:
        payload["error"] = error
    return sse_event("tool_end", payload)


def event_done(
    answer: str,
    side_products: Optional[Dict[str, Any]] = None,
    context: Optional[Dict[str, Any]] = None,
    messages: Optional[list] = None,
) -> str:
    """Stream terminator carrying the same shape the non-streaming endpoint returns.

    The frontend uses this for post-message side effects (e.g. visualization
    commands, conversation history updates) that the streaming path can't
    do incrementally.
    """
    payload: Dict[str, Any] = {"answer": answer}
    if side_products is not None:
        payload["side_products"] = side_products
    if context is not None:
        payload["context"] = context
    if messages is not None:
        payload["messages"] = messages
    return sse_event("done", payload)


def event_error(message: str, error_type: str = "RuntimeError") -> str:
    """Fatal error — stream closes after this event."""
    return sse_event("error", {"message": message, "error_type": error_type})
