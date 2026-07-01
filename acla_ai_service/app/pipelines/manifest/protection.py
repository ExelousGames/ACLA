"""Protected-session helpers for annotation-aware data refreshes.

The data-prep pipeline runs inside Docker, so this module reads pipeline
manifests through the normal registry and uses only paths stored in those
manifests. A session is protected once any annotation output has a saved
chunk for it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Set

from app.pipelines.manifest import registry


RAW_CHUNK_MARKER = ":chunk_"


def session_id_from_chunk_id(chunk_id: Any) -> str:
    """Return the session id for plain or raw-stream chunk ids."""
    text = str(chunk_id)
    if RAW_CHUNK_MARKER in text:
        return text.split(RAW_CHUNK_MARKER, 1)[0]
    return text


def normalise_session_ids(session_ids: Iterable[Any] | None) -> Set[str]:
    if not session_ids:
        return set()
    return {session_id_from_chunk_id(session_id) for session_id in session_ids}


def is_protected_chunk_id(
    chunk_id: Any,
    protected_session_ids: Iterable[Any] | None,
) -> bool:
    return session_id_from_chunk_id(chunk_id) in normalise_session_ids(
        protected_session_ids
    )


def collect_protected_session_ids(store: Any) -> Set[str]:
    """Collect session ids with saved annotation chunks across all manifests."""
    pipelines = []
    for pipeline_id in registry.list_pipelines():
        pipeline = registry.load(pipeline_id)
        if pipeline is not None:
            pipelines.append(pipeline)

    for pipeline in pipelines:
        for node in pipeline.annotations:
            if node.output_key and node.output_dir and hasattr(store, "register_directory"):
                try:
                    store.register_directory(node.output_key, node.output_dir)
                except Exception as exc:
                    print(
                        f"[WARNING] Failed to register annotation output directory "
                        f"{node.output_dir!r} for {node.output_key!r}: {exc}"
                    )

    protected: Set[str] = set()
    seen_output_keys: Set[str] = set()
    for pipeline in pipelines:
        for node in pipeline.annotations:
            output_key = pipeline.effective_output_key(node)
            if not output_key or output_key in seen_output_keys:
                continue
            seen_output_keys.add(output_key)
            try:
                if not store.has_cached_data(output_key):
                    continue
                for chunk_id in store.list_chunk_ids(output_key):
                    protected.add(session_id_from_chunk_id(chunk_id))
            except Exception as exc:
                print(
                    f"[WARNING] Failed to inspect annotation output "
                    f"{output_key!r} for protected sessions: {exc}"
                )
    return protected


@dataclass
class ProtectedCleanupSummary:
    cache_key: str
    chunks_total: int = 0
    chunks_deleted: int = 0
    chunks_kept: int = 0


def clear_unprotected_chunks(
    store: Any,
    cache_key: str,
    protected_session_ids: Iterable[Any] | None,
) -> ProtectedCleanupSummary:
    """Clear a cache while preserving chunks for protected sessions."""
    summary = ProtectedCleanupSummary(cache_key=cache_key)
    protected = normalise_session_ids(protected_session_ids)
    if not protected:
        store.clear_cache(cache_key)
        return summary

    if not store.has_cached_data(cache_key):
        return summary

    chunk_ids = list(store.list_chunk_ids(cache_key))
    summary.chunks_total = len(chunk_ids)
    for chunk_id in chunk_ids:
        if session_id_from_chunk_id(chunk_id) in protected:
            summary.chunks_kept += 1
            continue
        if store.delete_chunk(cache_key, chunk_id):
            summary.chunks_deleted += 1
    return summary


def has_unprotected_chunks(
    store: Any,
    cache_key: str,
    protected_session_ids: Iterable[Any] | None,
) -> bool:
    """Return True when a cache has at least one chunk safe to refresh."""
    protected = normalise_session_ids(protected_session_ids)
    if not store.has_cached_data(cache_key):
        return False
    for chunk_id in store.list_chunk_ids(cache_key):
        if session_id_from_chunk_id(chunk_id) not in protected:
            return True
    return False


__all__ = [
    "ProtectedCleanupSummary",
    "clear_unprotected_chunks",
    "collect_protected_session_ids",
    "has_unprotected_chunks",
    "is_protected_chunk_id",
    "normalise_session_ids",
    "session_id_from_chunk_id",
]
