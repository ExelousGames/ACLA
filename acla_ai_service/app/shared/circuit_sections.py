"""Resolve section identities and geometry from backend circuit maps."""

from __future__ import annotations

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

import pandas as pd


logger = logging.getLogger(__name__)


def normalize_circuit_id(value: str) -> str:
    return value.strip().lower().replace(" ", "_").replace("-", "_")


async def _fetch_sections(track: str, game: str) -> list[dict]:
    from app.integrations.backend.client import BackendService

    # Authentication locks must belong to the worker's event loop.
    backend = BackendService()
    response = await backend.call_backend_function(
        f"circuit-map/list?game={game}", timeout_seconds=10,
    )
    if "error" in response:
        raise ConnectionError(response["error"])
    maps = response["list"]
    # Maps are newest first; prefer the telemetry key over a display name.
    matching = [
        item for item in maps
        if normalize_circuit_id(item.get("source_track_key") or "") == track
    ] or [
        item for item in maps
        if normalize_circuit_id(item["circuit_name"]) == track
    ]
    if not matching:
        return []
    circuit_map = await backend.call_backend_function(
        f"circuit-map/{matching[0]['id']}", timeout_seconds=10,
    )
    if "error" in circuit_map:
        raise ConnectionError(circuit_map["error"])
    return [
        {
            "id": section["id"],
            "name": section.get("name") or ", ".join(section["tags"]),
            "normalized_position_range": (
                section["start_position"], section["end_position"],
            ),
        }
        for section in circuit_map.get("centerline_segments", [])
    ]


@lru_cache(maxsize=32)
def _cached_sections(track: str, game: str, cache_minute: int) -> list[dict]:
    # Synchronous inference also runs inside async API handlers. Isolate the
    # backend client's event loop so both callers can use the same lookup.
    with ThreadPoolExecutor(max_workers=1) as executor:
        try:
            return executor.submit(lambda: asyncio.run(_fetch_sections(track, game))).result()
        except Exception as exc:
            logger.warning("Could not load backend sections for %s: %s", track, exc)
            return []


def get_circuit_sections(df: pd.DataFrame, circuit_id: str | None = None) -> list[dict]:
    """Load this telemetry's map sections without a static label fallback."""
    track = circuit_id
    if not track and "Static_track" in df.columns:
        values = df["Static_track"].dropna()
        if not values.empty:
            track = str(values.iloc[0])
    if not track or not normalize_circuit_id(track):
        return []
    game = df.attrs.get("game", "acc")
    if "game" in df.columns:
        values = df["game"].dropna()
        if not values.empty:
            game = str(values.iloc[0])
    if game not in {"acc", "iracing"}:
        return []
    return _cached_sections(normalize_circuit_id(track), game, int(time.monotonic() // 60))
