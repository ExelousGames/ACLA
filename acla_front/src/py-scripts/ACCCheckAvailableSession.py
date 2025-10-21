from __future__ import annotations

import json
import sys
import time
from typing import Any, Optional

try:
    from pyaccsharedmemory import accSharedMemory as _ACCSharedMemoryClass
except Exception as _import_error:  # pragma: no cover - environment dependent
    _ACCSharedMemoryClass = None
    _ACC_SHARED_MEMORY_IMPORT_ERROR = _import_error
else:
    _ACC_SHARED_MEMORY_IMPORT_ERROR = None

from util.json_utils import DataclassJSONUtility
from util.streaming import StreamingEvent, StreamingServer


POLL_INTERVAL_SECONDS = 1.0
IDLE_EMIT_INTERVAL_SECONDS = 5.0


class ACCSessionChecker(StreamingServer):
    """Continuously polls ACC shared memory and emits session updates."""

    def __init__(self, poll_interval: float = POLL_INTERVAL_SECONDS) -> None:
        super().__init__(name="acc_session_checker", poll_interval=poll_interval)
        self._shared_memory_class = _ACCSharedMemoryClass
        self._import_error_message = (
            f"pyaccsharedmemory unavailable: {_ACC_SHARED_MEMORY_IMPORT_ERROR}"
            if _ACC_SHARED_MEMORY_IMPORT_ERROR
            else None
        )
        self.asm = None
        self._last_payload_json = None
        self._last_emit_time = -IDLE_EMIT_INTERVAL_SECONDS
        self._connected = False
        self._last_connect_error_time = -IDLE_EMIT_INTERVAL_SECONDS

    def on_start(self) -> None:
        # Emit an initial checking status so the UI reflects the polling state immediately.
        self._emit_checking(time.monotonic())

    def on_stop(self) -> None:
        try:
            if self.asm is not None:
                self.asm.close()
        except Exception:
            pass
        finally:
            self.asm = None

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
        now = time.monotonic()

        try:
            snapshot = self._read_shared_memory()
        except Exception:
            # Defensive catch-all: never allow poll to raise and terminate the loop.
            self.emit_error(
                "poll",
                "Unexpected failure while checking ACC shared memory",
                sys.exc_info(),
            )
            self._connected = False
            self._emit_checking(now)
            return

        if snapshot is None:
            if now - self._last_emit_time >= IDLE_EMIT_INTERVAL_SECONDS:
                self._connected = False
                self._emit_checking(now)
            return

        self._connected = True

        try:
            payload_json = self._serialize_snapshot(snapshot)
        except Exception:
            self.emit_error(
                "serialization",
                "Failed to serialize ACC shared memory snapshot",
                sys.exc_info(),
            )
            self._connected = False
            self._emit_checking(now)
            return

        should_emit = False
        if payload_json != self._last_payload_json:
            should_emit = True
        elif now - self._last_emit_time >= IDLE_EMIT_INTERVAL_SECONDS:
            should_emit = True

        if should_emit:
            self._last_payload_json = payload_json
            self._last_emit_time = now
            try:
                payload = json.loads(payload_json)
            except json.JSONDecodeError:
                self.emit_error(
                    "serialization",
                    "Received non-JSON payload from serializer",
                    exc_info=None,
                )
                self._connected = False
                self._emit_checking(now)
                return

            payload.setdefault("available", True)
            self.emit_update(payload)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _read_shared_memory(self) -> Optional[Any]:
        if not self._ensure_shared_memory():
            return None

        try:
            return self.asm.read_shared_memory()  # type: ignore[union-attr]
        except Exception:
            self.emit_error(
                "shared_memory",
                "Failed to read ACC shared memory",
                sys.exc_info(),
            )
            self._reset_shared_memory()
            return None

    def _ensure_shared_memory(self) -> bool:
        if self.asm is not None:
            return True

        now = time.monotonic()

        if self._shared_memory_class is None:
            if self._import_error_message and now - self._last_connect_error_time >= IDLE_EMIT_INTERVAL_SECONDS:
                self._last_connect_error_time = now
                self.emit_error(
                    "shared_memory",
                    self._import_error_message,
                    exc_info=None,
                )
            return False

        try:
            self.asm = self._shared_memory_class()
            self._connected = True
            self._last_connect_error_time = now
            return True
        except Exception:
            self.asm = None
            if now - self._last_connect_error_time >= IDLE_EMIT_INTERVAL_SECONDS:
                self._last_connect_error_time = now
                self.emit_log("ACC shared memory not available yet", stage="connection")
            return False

    def _reset_shared_memory(self) -> None:
        if self._shared_memory_class is None:
            return

        try:
            if self.asm is not None:
                self.asm.close()
        except Exception:
            pass

        try:
            self.asm = self._shared_memory_class()
        except Exception:
            self.emit_error(
                "shared_memory",
                "Failed to reinitialize ACC shared memory",
                sys.exc_info(),
            )
            self.asm = None

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

    def _emit_checking(self, now: float) -> None:
        payload = {
            "available": False,
            "checking": True,
            "message": "Checking for live ACC session",
        }
        payload_json = json.dumps(payload, separators=(",", ":"))
        if payload_json != self._last_payload_json:
            self._last_payload_json = payload_json
        self.emit_update(payload)
        self._last_emit_time = now


def main() -> None:
    ACCSessionChecker().run()


if __name__ == "__main__":
    main()