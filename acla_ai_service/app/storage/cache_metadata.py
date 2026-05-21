"""Storage-band metadata dataclasses.

Pulled out of ``app.storage.zarr`` so the Lance backend (and any future
backends) can reuse them without depending on the legacy Zarr module.
The dataclasses describe **bookkeeping**, not data: ``CacheMetadata``
tracks per-cache_key chunk counts/sizes/timestamps, ``ChunkPayload``
normalises the ``(payload, chunk_id)`` shape stream consumers can yield
into ``cache_chunks_streaming``.

Backend modules import ``CacheMetadata`` to persist it next to their
datasets (sidecar JSON in the Lance backend; was an in-dataset row in
the now-removed Zarr backend).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class CacheMetadata:
    cache_key: str
    chunk_sizes: List[int] = field(default_factory=list)
    total_records: int = 0
    total_bytes: int = 0
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())

    @property
    def chunk_count(self) -> int:
        return len(self.chunk_sizes)

    @property
    def next_chunk_index(self) -> int:
        # Data chunks start after the (legacy) metadata chunk at index 0.
        return self.chunk_count + 1

    def register_chunk(self, chunk_bytes: int) -> None:
        self.chunk_sizes.append(chunk_bytes)
        self.total_bytes += chunk_bytes
        self.updated_at = datetime.now().isoformat()

    def update_chunk(self, chunk_index: int, chunk_bytes: int) -> None:
        # chunk_index is 1-based (legacy convention from the Zarr backend
        # where the metadata chunk sat at index 0).
        list_index = chunk_index - 1
        while len(self.chunk_sizes) <= list_index:
            self.chunk_sizes.append(0)

        old_size = self.chunk_sizes[list_index]
        self.chunk_sizes[list_index] = chunk_bytes
        self.total_bytes = self.total_bytes - old_size + chunk_bytes
        self.updated_at = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cache_key": self.cache_key,
            "chunk_sizes": list(self.chunk_sizes),
            "total_records": self.total_records,
            "total_bytes": self.total_bytes,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "CacheMetadata":
        cache_key = payload.get("cache_key", "")
        chunk_sizes = payload.get("chunk_sizes", [])
        normalised_sizes: List[int] = []
        for size in chunk_sizes:
            try:
                normalised_sizes.append(int(size))
            except (TypeError, ValueError):
                continue
        total_records = int(payload.get("total_records", 0))
        total_bytes = int(payload.get("total_bytes", 0))
        created_at = payload.get("created_at") or datetime.now().isoformat()
        updated_at = payload.get("updated_at") or datetime.now().isoformat()

        return cls(
            cache_key=cache_key,
            chunk_sizes=normalised_sizes,
            total_records=total_records,
            total_bytes=total_bytes,
            created_at=created_at,
            updated_at=updated_at,
        )


@dataclass
class ChunkPayload:
    payload: Any
    chunk_id: Optional[str] = None


__all__ = ["CacheMetadata", "ChunkPayload"]
