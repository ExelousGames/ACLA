from __future__ import annotations

import json
import sys
import time
from typing import Any, Optional

from pyaccsharedmemory import accSharedMemory

from util.json_utils import DataclassJSONUtility
from util.streaming import StreamingEvent, StreamingServer


POLL_INTERVAL_SECONDS = 1.0
IDLE_EMIT_INTERVAL_SECONDS = 5.0


class ACCSessionChecker(StreamingServer):
    """Continuously polls ACC shared memory and emits session updates."""

    def __init__(self, poll_interval: float = POLL_INTERVAL_SECONDS) -> None:
        super().__init__(name="acc_session_checker", poll_interval=poll_interval)
        self.asm = accSharedMemory()
        self._last_payload_json: Optional[str] = None
        self._last_emit_time: float = 0.0
        self._connected = False

    def on_stop(self) -> None:
        try:
            self.asm.close()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Command handling
    # ------------------------------------------------------------------
    def handle_event(self, event: StreamingEvent) -> None:
        if event.action == "set_interval":
            self._handle_set_interval(event)
            return

        if event.action == "request_update":
            self._emit_snapshot(force=True, request_id=event.request_id)
            return

        self.emit_log(f"Unknown action '{event.action}'", stage="command")

    def _handle_set_interval(self, event: StreamingEvent) -> None:
        seconds = event.payload.get("seconds")
        try:
            value = float(seconds)
        except (TypeError, ValueError):
            self.emit_error(
                "set_interval",
                f"Invalid seconds value: {seconds!r}",
                exc_info=None,
                request_id=event.request_id,
            )
            return

        if value <= 0:
            self.emit_error(
                "set_interval",
                f"Interval must be positive: {seconds!r}",
                exc_info=None,
                request_id=event.request_id,
            )
            return

        self.poll_interval = max(value, 0.1)
        self.emit_log(
            f"Poll interval updated to {self.poll_interval:.2f}s",
            stage="config",
        )

    # ------------------------------------------------------------------
    # Polling loop
    # ------------------------------------------------------------------
    def poll(self) -> None:
        snapshot = self._read_shared_memory()
        now = time.monotonic()

        if snapshot is None:
            if self._connected and now - self._last_emit_time >= IDLE_EMIT_INTERVAL_SECONDS:
                self._connected = False
                self.emit_update({"available": False, "message": "No live session detected"})
                self._last_emit_time = now
            return

        self._connected = True
        payload_json = self._serialize_snapshot(snapshot)

        should_emit = False
        if payload_json != self._last_payload_json:
            should_emit = True
        elif now - self._last_emit_time >= IDLE_EMIT_INTERVAL_SECONDS:
            should_emit = True

        if should_emit:
            self._last_payload_json = payload_json
            self._last_emit_time = now
            payload = json.loads(payload_json)
            payload.setdefault("available", True)
            self.emit_update(payload)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _read_shared_memory(self) -> Optional[Any]:
        try:
            return self.asm.read_shared_memory()
        except Exception:
            self.emit_error(
                "shared_memory",
                "Failed to read ACC shared memory",
                sys.exc_info(),
            )
            self._reset_shared_memory()
            return None

    def _reset_shared_memory(self) -> None:
        try:
            self.asm.close()
        except Exception:
            pass

        try:
            self.asm = accSharedMemory()
        except Exception:
            self.emit_error(
                "shared_memory",
                "Failed to reinitialize ACC shared memory",
                sys.exc_info(),
            )

    def _serialize_snapshot(self, snapshot: Any) -> str:
        return DataclassJSONUtility.to_json(snapshot, compact=True)

    def _emit_snapshot(self, *, force: bool, request_id: Optional[str]) -> None:
        snapshot = self._read_shared_memory()
        if snapshot is None:
            return

        payload_json = self._serialize_snapshot(snapshot)
        if not force and payload_json == self._last_payload_json:
            return

        self._last_payload_json = payload_json
        self._last_emit_time = time.monotonic()
        payload = json.loads(payload_json)
        payload.setdefault("available", True)
        self.emit_update(payload, request_id=request_id)


def main() -> None:
    ACCSessionChecker().run()


if __name__ == "__main__":
    main()