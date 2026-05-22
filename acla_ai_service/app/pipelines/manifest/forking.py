"""Copy a Lance dataset into a per-annotation cache_key.

Each annotation in a pipeline owns a forked copy of whatever source the
user picked (an external cache_key or another annotation's output). We
copy via the store's public chunk API so the copy goes through the same
strategy plumbing as the originals.

:func:`fork_dataset` does the copy; :func:`compare_against_source` is
the git-style "N records behind" check.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional


def derive_annotation_input_key(
    pipeline_id: str, node_id: str, version: int
) -> str:
    """Stable per-annotation key for an input fork.

    Version bumps when the fork is re-pulled from its source, so older
    forks remain addressable if anything still references them.
    """
    return f"pipeline_input__{pipeline_id}__{node_id}__v{version}"


def fork_dataset(
    source_key: str,
    target_key: str,
    store: Any,
    progress: Optional[Callable[[int, int], None]] = None,
) -> str:
    """Copy every chunk from ``source_key`` into ``target_key``.

    Idempotent: if ``target_key`` already has data, returns it as-is.
    """
    if store.has_cached_data(target_key):
        return target_key

    chunk_ids = store.list_chunk_ids(source_key)
    total = len(chunk_ids)
    for i, chunk_id in enumerate(chunk_ids):
        payload = store.get_chunk(source_key, chunk_id)
        if payload is None:
            continue
        store.save_chunk(target_key, chunk_id, payload)
        if progress is not None:
            progress(i + 1, total)

    return target_key


@dataclass
class SourceComparison:
    source_updated_at: Optional[str]
    copy_updated_at: Optional[str]
    source_total_records: int
    copy_total_records: int
    source_chunk_count: int
    copy_chunk_count: int

    @property
    def is_behind(self) -> bool:
        """True when the source has changed (or grown) since the copy was made."""
        if self.source_chunk_count != self.copy_chunk_count:
            return True
        if self.source_total_records != self.copy_total_records:
            return True
        if self.source_updated_at and self.copy_updated_at:
            return self.source_updated_at > self.copy_updated_at
        return False


def compare_against_source(store: Any, source_key: str, copy_key: str) -> SourceComparison:
    src_meta = store.get_cache_metadata(source_key)
    cpy_meta = store.get_cache_metadata(copy_key)
    return SourceComparison(
        source_updated_at=src_meta.updated_at if src_meta else None,
        copy_updated_at=cpy_meta.updated_at if cpy_meta else None,
        source_total_records=src_meta.total_records if src_meta else 0,
        copy_total_records=cpy_meta.total_records if cpy_meta else 0,
        source_chunk_count=src_meta.chunk_count if src_meta else 0,
        copy_chunk_count=cpy_meta.chunk_count if cpy_meta else 0,
    )


__all__ = [
    "fork_dataset",
    "derive_annotation_input_key",
    "compare_against_source",
    "SourceComparison",
]
