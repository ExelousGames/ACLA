"""Source-copy helpers for source-mode annotation nodes.

Source-mode annotations should not read the live upstream dataset
directly. If data preparation refreshes that source, an annotation
session that was already opened can otherwise point at different rows.
Instead, each source-mode output gets a sibling input-copy dataset.
Sessions are copied once and never overwritten; sessions that already
have saved annotation output are also skipped.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Set

from app.pipelines.manifest.protection import (
    normalise_session_ids,
    session_id_from_chunk_id,
)


SOURCE_COPY_SUFFIX = "__source_copy"


def source_copy_key(output_key: str) -> str:
    """Return the cache key used for a source-mode node's copied input."""
    return f"{output_key}{SOURCE_COPY_SUFFIX}"


@dataclass
class SourceCopySummary:
    source_key: str
    copy_key: str
    source_chunks_total: int = 0
    chunks_copied: int = 0
    chunks_skipped_existing_copy: int = 0
    chunks_skipped_touched: int = 0
    read_failures: list[str] = field(default_factory=list)
    write_failures: list[str] = field(default_factory=list)
    touched_without_copy: list[str] = field(default_factory=list)


def _existing_copy_sessions(store: Any, copy_key: str) -> Set[str]:
    try:
        if not store.has_cached_data(copy_key):
            return set()
        return normalise_session_ids(store.list_chunk_ids(copy_key))
    except Exception:
        return set()


def sync_source_copy(
    store: Any,
    *,
    source_key: str,
    copy_key: str,
    touched_session_ids: Iterable[Any] | None = None,
) -> SourceCopySummary:
    """Copy missing source chunks into ``copy_key``.

    Existing copied sessions are left untouched, so source refreshes do
    not change sessions the annotation node has already seen. Sessions
    with saved user/AI annotation chunks are also left untouched.
    """
    summary = SourceCopySummary(source_key=source_key, copy_key=copy_key)
    touched = normalise_session_ids(touched_session_ids)
    copied_sessions = _existing_copy_sessions(store, copy_key)

    try:
        source_chunk_ids = list(store.list_chunk_ids(source_key))
    except Exception as exc:
        summary.read_failures.append(f"{source_key}: {exc}")
        return summary

    summary.source_chunks_total = len(source_chunk_ids)
    for chunk_id in source_chunk_ids:
        session_id = session_id_from_chunk_id(chunk_id)
        if session_id in copied_sessions:
            summary.chunks_skipped_existing_copy += 1
            continue
        if session_id in touched:
            summary.chunks_skipped_touched += 1
            summary.touched_without_copy.append(session_id)
            continue

        try:
            payload = store.get_chunk(source_key, chunk_id)
        except Exception as exc:
            summary.read_failures.append(f"{chunk_id}: {exc}")
            continue
        if payload is None:
            summary.read_failures.append(f"{chunk_id}: empty")
            continue
        try:
            saved = store.save_chunk(copy_key, chunk_id, payload)
        except Exception as exc:
            summary.write_failures.append(f"{chunk_id}: {exc}")
            continue
        if saved:
            summary.chunks_copied += 1
            copied_sessions.add(session_id)
        else:
            summary.write_failures.append(f"{chunk_id}: save failed")

    return summary


__all__ = [
    "SOURCE_COPY_SUFFIX",
    "SourceCopySummary",
    "source_copy_key",
    "sync_source_copy",
]
