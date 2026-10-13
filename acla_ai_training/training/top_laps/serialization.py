"""Serialize training-built top-lap references for backend publication."""

from __future__ import annotations

import base64
import io
import pickle
from typing import Any, Dict

from app.top_laps.model import TopLapStore


def encode_components(data: Dict[str, Any]) -> str:
    """Encode one top-lap component dictionary for backend storage."""

    buffer = io.BytesIO()
    pickle.dump(data, buffer, protocol=pickle.HIGHEST_PROTOCOL)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def serialize_top_lap_store(store: TopLapStore) -> Dict[str, Any]:
    """Serialize a store using the ``top_lap_store`` payload."""

    if not store.entries:
        raise ValueError("No stored top laps to serialize. Record laps first.")

    serialized_entries: Dict[str, str] = {}
    for (track, car, grip), entry in store.entries.items():
        key_str = f"{track}|{car}|grip{grip}"
        serialized_entries[key_str] = encode_components(entry.to_components())
    return {"top_lap_store": serialized_entries}


__all__ = ["encode_components", "serialize_top_lap_store"]
