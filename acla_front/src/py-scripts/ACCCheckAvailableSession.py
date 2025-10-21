from __future__ import annotations

import json
import sys
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
        self._initializing_error_reported = False

    def on_start(self) -> None:
        self._initialize_shared_memory()
        self._emit_checking()

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
        if event.action == "request_update":
            self._emit_current_state(request_id=event.request_id)
            return

        self.emit_log(f"Unknown action '{event.action}'", stage="command")

    # ------------------------------------------------------------------
    # Polling loop
    # ------------------------------------------------------------------
    def poll(self) -> None:
        if self.asm is None and not self._initialize_shared_memory():
            self._emit_checking()
            return

        snapshot = self._read_snapshot()
        if not snapshot:
            self._emit_checking()
            return

        payload_json = self._serialize_snapshot(snapshot)
        if payload_json is None:
            self._emit_checking()
            return

        if payload_json == self._last_payload_json:
            return

        payload = json.loads(payload_json)
        if not payload:
            self._emit_checking()
            return

        self._last_payload_json = payload_json
        payload.setdefault("available", True)
        self.emit_update(payload)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _initialize_shared_memory(self) -> bool:
        if self.asm is not None:
            return True

        if self._shared_memory_class is None:
            if self._import_error_message and not self._initializing_error_reported:
                self.emit_error(
                    "shared_memory",
                    self._import_error_message,
                    exc_info=None,
                )
                self._initializing_error_reported = True
            return False

        try:
            self.asm = self._shared_memory_class()
            self._initializing_error_reported = False
            return True
        except Exception:
            self.asm = None
            if not self._initializing_error_reported:
                self.emit_log("ACC shared memory not available yet", stage="connection")
                self._initializing_error_reported = True
            return False

    def _read_snapshot(self) -> Optional[Any]:
        if self.asm is None:
            return None

        try:
            return self.asm.read_shared_memory()  # type: ignore[union-attr]
        except Exception:
            self.emit_error(
                "shared_memory",
                "Failed to read ACC shared memory",
                sys.exc_info(),
            )
            self.asm = None
            return None

    def _serialize_snapshot(self, snapshot: Any) -> Optional[str]:
        try:
            return DataclassJSONUtility.to_json(snapshot, compact=True)
        except Exception:
            self.emit_error(
                "serialization",
                "Failed to serialize ACC shared memory snapshot",
                sys.exc_info(),
            )
            return None

    def _emit_current_state(self, *, request_id: Optional[str]) -> None:
        if self.asm is None and not self._initialize_shared_memory():
            self._emit_checking(request_id=request_id)
            return

        snapshot = self._read_snapshot()
        if not snapshot:
            self._emit_checking(request_id=request_id)
            return

        payload_json = self._serialize_snapshot(snapshot)
        if payload_json is None:
            self._emit_checking(request_id=request_id)
            return

        payload = json.loads(payload_json)
        if not payload:
            self._emit_checking(request_id=request_id)
            return

        payload.setdefault("available", True)
        self._last_payload_json = payload_json
        self.emit_update(payload, request_id=request_id)

    def _emit_checking(self, *, request_id: Optional[str] = None) -> None:
        payload = {
            "available": False,
            "checking": True,
            "message": "Checking for live ACC session",
        }
        payload_json = json.dumps(payload, separators=(",", ":"))
        if payload_json != self._last_payload_json:
            self._last_payload_json = payload_json
            self.emit_update(payload, request_id=request_id)


def main() -> None:
    ACCSessionChecker().run()


if __name__ == "__main__":
    main()