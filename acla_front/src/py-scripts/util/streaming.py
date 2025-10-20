"""Reusable helpers for building streaming-style Python scripts.

These helpers provide a simple event loop that emits JSON messages to stdout
and listens for JSON commands from stdin. They are designed to integrate with
Electron's Python process bridge, allowing multiple concurrent streaming
scripts to run simultaneously while supporting `ping` and `shutdown` control
messages.
"""
from __future__ import annotations

import json
import queue
import sys
import threading
import time
import traceback
from dataclasses import dataclass
from typing import Any, Dict, Optional

JsonDict = Dict[str, Any]


def _emit(payload: JsonDict) -> None:
    sys.stdout.write(json.dumps(payload, separators=(",", ":")) + "\n")
    sys.stdout.flush()


@dataclass
class StreamingEvent:
    action: str
    payload: JsonDict
    request_id: Optional[str] = None


class StreamingServer:
    """Convenience base class for streaming-style Python scripts."""

    def __init__(
        self,
        *,
        name: str,
        poll_interval: float = 1.0,
        sleep_chunk: float = 0.1,
    ) -> None:
        self.name = name
        self.poll_interval = max(poll_interval, 0.01)
        self.sleep_chunk = max(min(sleep_chunk, self.poll_interval), 0.01)
        self._stop_event = threading.Event()
        self._command_queue: "queue.Queue[Optional[str]]" = queue.Queue()
        self._stdin_thread = threading.Thread(
            target=self._stdin_reader, name=f"{name}-stdin", daemon=True
        )
        self._is_running = False

    # ---------------------------------------------------------------------
    # Lifecycle management helpers
    # ---------------------------------------------------------------------
    def run(self) -> None:
        """Start the streaming loop."""
        if self._is_running:
            raise RuntimeError("StreamingServer is already running")

        self._is_running = True
        self._stdin_thread.start()

        try:
            self.emit_ready()
            self.on_start()
            while not self._stop_event.is_set():
                self._drain_commands()
                try:
                    self.poll()
                except Exception:  # pragma: no cover - defensive guard
                    self.emit_error("poll", "Unhandled exception during poll", sys.exc_info())
                    self.shutdown()
                    break
                self._sleep_until_next_poll()
        finally:
            try:
                self.on_stop()
            finally:
                self.emit_shutdown()
                self._is_running = False

    def shutdown(self) -> None:
        """Signal the streaming loop to stop."""
        self._stop_event.set()

    # ---------------------------------------------------------------------
    # Methods for subclasses to override
    # ---------------------------------------------------------------------
    def on_start(self) -> None:
        """Hook called once before streaming begins."""

    def on_stop(self) -> None:
        """Hook called once after the loop exits."""

    def poll(self) -> None:
        """Hook called repeatedly; subclasses should emit updates here."""
        raise NotImplementedError

    def handle_event(self, event: StreamingEvent) -> None:
        """Hook for handling custom commands from the renderer."""

    # ---------------------------------------------------------------------
    # Emission helpers
    # ---------------------------------------------------------------------
    def emit(self, payload: JsonDict) -> None:
        if "source" not in payload:
            payload["source"] = self.name
        _emit(payload)

    def emit_ready(self, extra: Optional[JsonDict] = None) -> None:
        payload: JsonDict = {"status": "ready", "source": self.name}
        if extra:
            payload.update(extra)
        _emit(payload)

    def emit_log(self, message: str, *, stage: Optional[str] = None) -> None:
        if not message:
            return
        payload: JsonDict = {"status": "log", "message": message, "source": self.name}
        if stage:
            payload["stage"] = stage
        _emit(payload)

    def emit_update(self, data: Any, *, request_id: Optional[str] = None) -> None:
        payload: JsonDict = {"status": "update", "data": data, "source": self.name}
        if request_id is not None:
            payload["request_id"] = request_id
        _emit(payload)

    def emit_error(
        self,
        stage: str,
        message: str,
        exc_info: Optional[Any] = None,
        *,
        request_id: Optional[str] = None,
    ) -> None:
        payload: JsonDict = {
            "status": "error",
            "stage": stage,
            "message": message,
            "source": self.name,
        }
        if exc_info:
            payload["traceback"] = "".join(traceback.format_exception(*exc_info))
        if request_id is not None:
            payload["request_id"] = request_id
        _emit(payload)

    def emit_shutdown(self) -> None:
        payload: JsonDict = {"status": "shutdown", "source": self.name}
        _emit(payload)

    # ---------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------
    def _stdin_reader(self) -> None:
        try:
            for raw_line in sys.stdin:
                if not raw_line:
                    continue
                cleaned = raw_line.strip()
                if cleaned:
                    self._command_queue.put(cleaned)
        finally:
            self._command_queue.put(None)

    def _drain_commands(self) -> None:
        while not self._command_queue.empty():
            try:
                raw = self._command_queue.get_nowait()
            except queue.Empty:  # pragma: no cover - race guard
                break

            if raw is None:
                self.shutdown()
                break

            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                self.emit_error("command", f"Received non-JSON command: {raw!r}")
                continue

            action = payload.get("action")
            if not action:
                self.emit_error("command", f"Command missing 'action': {payload!r}")
                continue

            request_id = payload.get("request_id")

            if action == "shutdown":
                self.shutdown()
                continue
            if action == "ping":
                response: JsonDict = {
                    "status": "pong",
                    "source": self.name,
                    "request_id": request_id,
                    "timestamp": time.time(),
                }
                _emit(response)
                continue

            event_payload = payload.get("payload")
            if event_payload is None:
                event_payload = {}

            try:
                event = StreamingEvent(action=action, payload=event_payload, request_id=request_id)
                self.handle_event(event)
            except Exception:  # pragma: no cover - defensive guard
                self.emit_error(
                    action,
                    f"Unhandled exception while handling action '{action}'",
                    sys.exc_info(),
                    request_id=request_id,
                )

    def _sleep_until_next_poll(self) -> None:
        target = time.monotonic() + self.poll_interval
        while not self._stop_event.is_set():
            remaining = target - time.monotonic()
            if remaining <= 0:
                break
            self._drain_commands()
            time.sleep(min(self.sleep_chunk, remaining))