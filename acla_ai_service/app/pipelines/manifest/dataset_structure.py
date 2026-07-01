"""Dataset-shape checks shared by pipeline UI and tests."""

from __future__ import annotations

from typing import Any, Optional

from app.pipelines.manifest.node_kinds import InputDatasetShape


def payload_records(payload: Any) -> tuple[str, list[Any]]:
    if isinstance(payload, dict) and isinstance(payload.get("data"), list):
        return "dict.data", payload["data"]
    if isinstance(payload, list):
        return "list", payload
    if isinstance(payload, dict):
        return "dict", [payload]
    return type(payload).__name__, [payload]


def is_segment_record(record: Any) -> bool:
    return (
        isinstance(record, dict)
        and "start_index" in record
        and "end_index" in record
        and ("labels" in record or "telemetry_data" in record)
    )


def is_session_record(record: Any) -> bool:
    if not isinstance(record, dict):
        return False
    if not is_segment_record(record):
        return True
    telemetry = record.get("telemetry_data")
    return isinstance(telemetry, list) and bool(telemetry)


def has_cached_data(store: Any, key: Optional[str]) -> bool:
    if not key:
        return False
    try:
        return store.has_cached_data(key)
    except Exception:
        return False


def cache_key_matches_dataset_structure(
    store: Any,
    key: Optional[str],
    *,
    required_structure: InputDatasetShape,
) -> bool:
    if not has_cached_data(store, key):
        return False
    try:
        chunk_ids = list(store.list_chunk_ids(key))
    except Exception:
        return False
    for chunk_id in chunk_ids:
        try:
            payload = store.get_chunk(key, chunk_id)
        except Exception:
            continue
        _, records = payload_records(payload)
        if required_structure == "segments":
            if any(is_segment_record(record) for record in records):
                return True
            continue
        if any(is_session_record(record) for record in records):
            return True
    return False


__all__ = [
    "cache_key_matches_dataset_structure",
    "has_cached_data",
    "is_segment_record",
    "is_session_record",
    "payload_records",
]
