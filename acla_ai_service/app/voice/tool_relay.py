"""Per-connection WS tool-relay for the voice pipeline.

Backend ↔ frontend tool RPC over the same ``/voice/stream`` WebSocket the
Pipecat audio path uses. The two channels are multiplexed by WebSocket
frame type — binary frames carry PCM audio (consumed by Pipecat), text
frames carry JSON tool-relay messages (consumed by this module).

The relay does not own the WebSocket. The voice endpoint binds a
connection by handing the relay a ``send_text`` callback plus an
``observation_sink`` callback, and forwards every inbound text frame to
:py:meth:`ToolRelay.handle_text_frame`. The relay resolves in-flight
tool-call futures by uuid and pushes observation events into the sink.

Frame shapes (over the WS as JSON text):

* Backend → Frontend::

      {"type": "tool_call", "id": "<uuid>",
       "name": "<fn>", "arguments": {...}}

* Frontend → Backend (one of, per tool_call id)::

      {"type": "tool_result", "id": "<uuid>", "result": {...}}
      {"type": "tool_error",  "id": "<uuid>", "error": "<msg>"}

* Frontend → Backend (independent of any specific tool call)::

      {"type": "observation", "data": {...}}

Public surface:

* :func:`get_relay` — process-singleton accessor.
* :py:meth:`ToolRelay.bind(conn, send_text, observation_sink)` — register
  a connection so dispatch / observation routing work for it.
* :py:meth:`ToolRelay.unbind(conn)` — drop registration and cancel any
  in-flight calls.
* ``await``:py:meth:`ToolRelay.dispatch(conn, name, args, timeout)` —
  send a ``tool_call`` and await the matching ``tool_result`` /
  ``tool_error``. Returns ``{"error": "..."}`` on dispatch failure /
  timeout so the LLM can verbalize the failure cleanly.
* :py:meth:`ToolRelay.handle_text_frame(conn, payload)` — feed each
  inbound text frame in (the voice endpoint calls this from its WS
  receive loop).
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from typing import Any, Awaitable, Callable, Dict, Optional

LOGGER = logging.getLogger(__name__)

SendText = Callable[[str], Awaitable[None]]
ObservationSink = Callable[[Dict[str, Any]], Any]


class _ConnectionState:
    """Per-connection state held by the relay."""

    __slots__ = ("send_text", "observation_sink", "in_flight")

    def __init__(self, send_text: SendText, observation_sink: ObservationSink) -> None:
        self.send_text = send_text
        self.observation_sink = observation_sink
        self.in_flight: Dict[str, asyncio.Future] = {}


class ToolRelay:
    """Process-wide registry for active voice connections + in-flight calls."""

    def __init__(self) -> None:
        self._by_conn: Dict[int, _ConnectionState] = {}

    # ------------------------------------------------------------------ binding

    def bind(
        self,
        conn: Any,
        send_text: SendText,
        observation_sink: ObservationSink,
    ) -> None:
        """Register a connection. ``conn`` is any hashable identifier (we use
        ``id(websocket)``). ``send_text`` writes one text frame; the
        ``observation_sink`` receives the ``data`` payload of each inbound
        ``observation`` frame."""
        self._by_conn[id(conn)] = _ConnectionState(send_text, observation_sink)

    def unbind(self, conn: Any) -> None:
        """Drop the registration and cancel every in-flight call so awaiters
        unblock with ``CancelledError`` (which ``dispatch`` maps to a clean
        ``{"error": "cancelled"}`` payload)."""
        state = self._by_conn.pop(id(conn), None)
        if state is None:
            return
        for fut in state.in_flight.values():
            if not fut.done():
                fut.cancel()
        state.in_flight.clear()

    # ----------------------------------------------------------------- dispatch

    async def dispatch(
        self,
        conn: Any,
        name: str,
        arguments: Optional[Dict[str, Any]] = None,
        timeout: float = 10.0,
    ) -> Dict[str, Any]:
        """Send a ``tool_call`` and await the response.

        Returns the ``result`` payload on success. On send-failure / timeout /
        cancellation, returns ``{"error": "<reason>"}`` — never raises — so
        the LLM tool handler can hand the dict back to Pipecat unchanged.
        """
        state = self._by_conn.get(id(conn))
        if state is None:
            return {"error": "telemetry_link_down"}

        call_id = uuid.uuid4().hex
        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()
        state.in_flight[call_id] = future

        frame = json.dumps({
            "type": "tool_call",
            "id": call_id,
            "name": name,
            "arguments": arguments or {},
        })

        try:
            await state.send_text(frame)
        except Exception as exc:
            state.in_flight.pop(call_id, None)
            LOGGER.warning("tool_relay: send_text failed for %s: %s", name, exc)
            return {"error": "telemetry_link_down"}

        try:
            return await asyncio.wait_for(future, timeout=timeout)
        except asyncio.TimeoutError:
            return {"error": "timeout"}
        except asyncio.CancelledError:
            return {"error": "cancelled"}
        finally:
            state.in_flight.pop(call_id, None)

    # ------------------------------------------------------------------ ingress

    def handle_text_frame(self, conn: Any, payload: Dict[str, Any]) -> None:
        """Route one inbound text frame.

        - ``tool_result`` / ``tool_error`` → resolve the matching in-flight
          future. Unknown ids are dropped (likely a late response after the
          caller already timed out / cancelled).
        - ``observation`` → forward ``payload["data"]`` to the connection's
          ``observation_sink``. Sink exceptions are caught and logged so a
          buggy sink never breaks the relay.
        """
        state = self._by_conn.get(id(conn))
        if state is None:
            return

        frame_type = payload.get("type")

        if frame_type in ("tool_result", "tool_error"):
            call_id = payload.get("id")
            future = state.in_flight.get(call_id) if isinstance(call_id, str) else None
            if future is None or future.done():
                return
            if frame_type == "tool_result":
                result = payload.get("result")
                future.set_result(result if isinstance(result, dict) else {"result": result})
            else:
                future.set_result({"error": str(payload.get("error", "unknown"))})
            return

        if frame_type == "observation":
            data = payload.get("data") or {}
            try:
                state.observation_sink(data)
            except Exception:
                LOGGER.exception("tool_relay: observation_sink raised")
            return

        LOGGER.warning("tool_relay: unknown frame type %r", frame_type)


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_RELAY = ToolRelay()


def get_relay() -> ToolRelay:
    """Return the process-wide ``ToolRelay`` instance."""
    return _RELAY
