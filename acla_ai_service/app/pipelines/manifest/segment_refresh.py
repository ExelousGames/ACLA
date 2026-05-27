"""Re-slice ``telemetry_data`` on a node's saved segments from its input.

"Update from source" pulls new columns into ``node.input_key``, but the
segments already saved in ``node.output_key`` keep the per-row dicts
they were created with. :func:`refresh_node_segments` re-slices every
segment's ``telemetry_data`` from the current
``input_key.iloc[start:end]`` so the new columns land on the saved
segments.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd

from app.pipelines.manifest.models import AnnotationNode


def _load_input_session(store: Any, input_key: str, session_id: str) -> pd.DataFrame:
    chunk = store.get_chunk(input_key, session_id)
    if chunk is None:
        return pd.DataFrame()
    if isinstance(chunk, list):
        return pd.DataFrame(chunk)
    if isinstance(chunk, dict):
        if "data" in chunk and isinstance(chunk["data"], list):
            return pd.DataFrame(chunk["data"])
        return pd.DataFrame([chunk])
    return pd.DataFrame()


def _slice_segment(df: pd.DataFrame, seg: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
    start = seg.get("start_index")
    end = seg.get("end_index")
    if start is None or end is None:
        return None
    s = max(0, int(start))
    e = min(len(df), int(end))
    if s >= e:
        return None
    return df.iloc[s:e].to_dict(orient="records")


@dataclass
class RefreshSummary:
    chunks_total: int = 0
    chunks_written: int = 0
    segments_total: int = 0
    segments_refreshed: int = 0
    segments_skipped: int = 0
    missing_input_sessions: List[str] = field(default_factory=list)


def refresh_node_segments(store: Any, node: AnnotationNode) -> RefreshSummary:
    """Refresh ``telemetry_data`` on every segment in ``node.output_key``
    from the current rows in ``node.input_key``. Raises ``ValueError``
    if the node isn't wired for a refresh (missing keys / empty input)."""
    input_key = node.input_key
    output_key = node.output_key
    if not input_key:
        raise ValueError(f"Node {node.id!r} has no input_key.")
    if not output_key:
        raise ValueError(f"Node {node.id!r} has no output_key.")
    if not store.has_cached_data(input_key):
        raise ValueError(f"Input {input_key!r} not in store.")
    if not store.has_cached_data(output_key):
        return RefreshSummary()

    session_ids = store.list_chunk_ids(output_key)
    summary = RefreshSummary(chunks_total=len(session_ids))

    for sid in session_ids:
        segments = store.get_chunk(output_key, sid)
        if not isinstance(segments, list) or not segments:
            continue
        df = _load_input_session(store, input_key, sid)
        if df.empty:
            summary.missing_input_sessions.append(sid)
            continue

        updated: List[Dict[str, Any]] = []
        refreshed_here = 0
        for seg in segments:
            if not isinstance(seg, dict):
                updated.append(seg)
                continue
            sliced = _slice_segment(df, seg)
            if sliced is None:
                summary.segments_skipped += 1
                updated.append(seg)
                continue
            new_seg = dict(seg)
            new_seg["telemetry_data"] = sliced
            new_seg["segment_length"] = len(sliced)
            updated.append(new_seg)
            refreshed_here += 1

        summary.segments_total += len(segments)
        summary.segments_refreshed += refreshed_here
        if refreshed_here > 0:
            store.save_chunk(output_key, sid, updated)
            summary.chunks_written += 1

    return summary


__all__ = ["RefreshSummary", "refresh_node_segments"]
