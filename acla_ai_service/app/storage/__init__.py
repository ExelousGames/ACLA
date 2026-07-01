"""Telemetry-store factory.

The codebase used to ship two backends — the original Zarr chunk store
and the Lance migration target — selected via
``TELEMETRY_STORE_BACKEND``. The Zarr backend was removed once the
Phase-2 typed Lance migration completed and the parity tests landed; the
factory is kept (a) as the canonical entry point for consumers, and (b)
to leave a single line to extend if another backend ever shows up.

Consumers should always go through :func:`get_shared_telemetry_store`.
"""

from __future__ import annotations

import os
from typing import Any


def _resolve_backend() -> str:
    raw = os.environ.get("TELEMETRY_STORE_BACKEND", "lance").strip().lower()
    if raw not in {"lance"}:
        raise ValueError(
            f"TELEMETRY_STORE_BACKEND must be 'lance' (Zarr backend removed); "
            f"got {raw!r}"
        )
    return raw


def get_shared_telemetry_store() -> Any:
    """Return the process-wide telemetry store.

    Currently always returns a :class:`app.storage.lance.LanceTelemetryStore`;
    the function signature is preserved so consumers don't need to change
    if another backend gets added later.
    """
    backend = _resolve_backend()
    if backend == "lance":
        from app.storage.lance import get_shared_lance_store
        return get_shared_lance_store()
    raise ValueError(f"Unknown telemetry store backend: {backend!r}")


__all__ = ["get_shared_telemetry_store"]
