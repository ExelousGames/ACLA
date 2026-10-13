"""Backend circuit-map sections used by training tools and visualizations."""

from __future__ import annotations

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

import pandas as pd


LOGGER = logging.getLogger(__name__)


def _track_key(value: str) -> str:
    return value.strip().lower().replace(" ", "_").replace("-", "_")


async def _fetch_sections(track: str, game: str) -> list[dict]:
    from app.integrations.backend.client import BackendService

    # Authentication locks belong to this worker’s event loop.
    backend_service = BackendService()
    response = await backend_service.call_backend_function(
        f"circuit-map/list?game={game}", timeout_seconds=10,
    )
    if "error" in response:
        raise ConnectionError(response["error"])
    maps = response["list"]
    # The backend returns newest maps first. Prefer the telemetry track key.
    matching = [
        item for item in maps
        if _track_key(item.get("source_track_key") or "") == track
    ] or [
        item for item in maps
        if _track_key(item["circuit_name"]) == track
    ]
    if not matching:
        return []
    circuit_map = await backend_service.call_backend_function(
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
    # Annotation tools can run inside an async provider as well as Streamlit.
    # Keep the backend client's event loop separate from either caller.
    with ThreadPoolExecutor(max_workers=1) as executor:
        try:
            return executor.submit(lambda: asyncio.run(_fetch_sections(track, game))).result()
        except Exception as exc:
            LOGGER.warning("Could not load backend sections for %s: %s", track, exc)
            return []


def get_circuit_sections(df: pd.DataFrame, circuit_id: str | None = None) -> list[dict]:
    """Resolve the telemetry's map, with no static name or range fallback."""
    track = circuit_id
    if not track and "Static_track" in df.columns:
        values = df["Static_track"].dropna()
        if not values.empty:
            track = str(values.iloc[0])
    if not track or not _track_key(track):
        return []
    game = df.attrs.get("game", "acc")
    if "game" in df.columns:
        values = df["game"].dropna()
        if not values.empty:
            game = str(values.iloc[0])
    if game not in {"acc", "iracing"}:
        return []
    return _cached_sections(_track_key(track), game, int(time.monotonic() // 60))
