"""Zarr-backed telemetry storage for large training datasets.

This module replaces the previous Parquet-based cache with a chunked Zarr
store that is optimised for streaming large telemetry payloads without
exhausting memory. Chunks are persisted as compressed JSON payloads so the
existing processing pipeline can continue to operate on any type without redesigning every consumer.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncIterator, Dict, Iterable, Iterator, List, Optional

import numpy as np
import zarr
from numcodecs import Blosc


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
        # Data chunks start after the metadata chunk at index 0.
        return self.chunk_count + 1

    def register_chunk(self, chunk_bytes: int) -> None:
        self.chunk_sizes.append(chunk_bytes)
        self.total_bytes += chunk_bytes
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

        metadata = cls(
            cache_key=cache_key,
            chunk_sizes=normalised_sizes,
            total_records=total_records,
            total_bytes=total_bytes,
            created_at=created_at,
            updated_at=updated_at,
        )
        return metadata


@dataclass
class ChunkPayload:
    payload: Any


class ZarrTelemetryStore:
    """Minimal telemetry data store built on top of Zarr."""

    def __init__(
        self,
        store_directory: str = "telemetry_zarr_store",
        *,
        compressor: Optional[Blosc] = None,
        chunk_max_mb: float = 10,
    ) -> None:
        self.store_dir = Path(store_directory)
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self.compressor = compressor or Blosc(cname="zstd", clevel=5, shuffle=Blosc.BITSHUFFLE)
        # Allow callers to specify in megabytes while storing the byte ceiling internally.
        self.chunk_max_bytes = max(1024, int(chunk_max_mb * 1024 * 1024))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _group_path(self, cache_key: str) -> Path:
        safe_key = cache_key.replace("/", "_")
        return self.store_dir / f"{safe_key}.zarr"

    def _open_group(self, cache_key: str, mode: str = "a") -> zarr.Group:
        group_path = self._group_path(cache_key)
        return zarr.open_group(str(group_path), mode=mode)

    @staticmethod
    def _metadata_dataset_name() -> str:
        return "chunk_000000"

    @staticmethod
    def _estimate_record_count(payload: Any) -> int:
        if isinstance(payload, dict):
            data_block = payload.get("data")
            if isinstance(data_block, Iterable):
                try:
                    return len(data_block)  # type: ignore[arg-type]
                except TypeError:
                    pass
        return 0

    @staticmethod
    def _ensure_async_iterator(iterator: Any) -> AsyncIterator[Any]:
        if hasattr(iterator, "__aiter__"):
            return iterator  # type: ignore[return-value]

        async def _adapter() -> AsyncIterator[Any]:
            for item in iterator:
                yield item

        return _adapter()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    async def cache_chunks_streaming(
        self,
        cache_key: str,
        chunks_iterator: Any,
    ) -> bool:
        """Persist chunks provided by ``chunks_iterator`` into the Zarr store."""

        group = self._open_group(cache_key, mode="a")
        metadata = self._load_or_initialise_metadata(group, cache_key)

        processed = 0
        async for chunk_candidate in self._ensure_async_iterator(chunks_iterator):
            chunk_entry = self._normalise_chunk_payload(chunk_candidate)
            if chunk_entry.payload is None:
                continue

            try:
                payload_bytes = json.dumps(chunk_entry.payload, ensure_ascii=False).encode("utf-8")
            except (TypeError, ValueError) as serialization_error:
                print(f"[WARNING] Failed to serialize chunk for {cache_key}: {serialization_error}")
                continue

            data_array = np.frombuffer(payload_bytes, dtype=np.uint8)
            chunk_len = min(len(data_array), self.chunk_max_bytes)
            if chunk_len == 0:
                print(f"[WARNING] Encountered empty chunk payload for {cache_key} (index {metadata.next_chunk_index})")
                continue

            dataset_name = self._format_chunk_name(metadata.next_chunk_index)
            group.create_dataset(
                dataset_name,
                data=data_array,
                compressor=self.compressor,
                overwrite=True,
                chunks=(chunk_len,),
            )
            group[dataset_name].attrs.update({
                "created_at": datetime.now().isoformat(),
                "kind": "data",
            })

            metadata.register_chunk(chunk_bytes=len(payload_bytes))
            processed += 1

        self._persist_metadata(group, metadata)

        return processed > 0

    def get_cached_data_chunks(self, cache_key: str) -> Iterator[Any]:
        """Yield cached chunks exactly as they were stored."""

        group_path = self._group_path(cache_key)
        if not group_path.exists():
            return iter(())

        group = self._open_group(cache_key, mode="r")
        chunk_names = sorted(group.array_keys())

        def _generator() -> Iterator[Any]:
            for chunk_name in chunk_names:
                if chunk_name == self._metadata_dataset_name():
                    continue
                try:
                    raw_bytes = bytes(group[chunk_name][:])
                    payload = json.loads(raw_bytes.decode("utf-8"))
                    yield payload
                except Exception as read_error:  # pragma: no cover - safety logging
                    print(f"[WARNING] Failed to read chunk '{chunk_name}' for {cache_key}: {read_error}")
                    continue

        return _generator()

    def get_cache_metadata(self, cache_key: str) -> Optional[CacheMetadata]:
        group_path = self._group_path(cache_key)
        if not group_path.exists():
            return None

        group = self._open_group(cache_key, mode="r")
        metadata = self._load_metadata_if_present(group, cache_key)
        return metadata

    def has_cached_data(self, cache_key: str) -> bool:
        group_path = self._group_path(cache_key)
        if not group_path.exists():
            return False

        try:
            group = self._open_group(cache_key, mode="r")
            metadata = self._load_metadata_if_present(group, cache_key)
            if metadata:
                return bool(metadata.chunk_count)

            dataset_name = self._metadata_dataset_name()
            return any(name != dataset_name for name in group.array_keys())
        except Exception:
            return False

    def list_cache_keys(self) -> List[str]:
        """List all available cache keys in the store."""
        cache_keys = []
        if not self.store_dir.exists():
            return cache_keys
        
        for zarr_dir in self.store_dir.glob("*.zarr"):
            cache_key = zarr_dir.stem
            if self.has_cached_data(cache_key):
                cache_keys.append(cache_key)
        
        return sorted(cache_keys)

    def clear_cache(self, cache_key: Optional[str] = None) -> None:
        if cache_key is None:
            if self.store_dir.exists():
                shutil.rmtree(self.store_dir)
            self.store_dir.mkdir(parents=True, exist_ok=True)
            return

        group_path = self._group_path(cache_key)
        if group_path.exists():
            shutil.rmtree(group_path, ignore_errors=True)

    def get_cache_info(self) -> Dict[str, Any]:
        entries = []
        total_size_bytes = 0

        for zarr_dir in self.store_dir.glob("*.zarr"):
            size_bytes = sum(file.stat().st_size for file in zarr_dir.rglob("*") if file.is_file())
            total_size_bytes += size_bytes

            try:
                group = zarr.open_group(str(zarr_dir), mode="r")
                metadata = self._load_metadata_if_present(group, zarr_dir.stem)
            except Exception:
                metadata = None

            if metadata:
                cache_key = metadata.cache_key or zarr_dir.stem
                entry = {
                    "cache_key": cache_key,
                    "chunk_count": metadata.chunk_count,
                    "total_records": metadata.total_records,
                    "total_bytes": metadata.total_bytes,
                    "chunk_sizes": list(metadata.chunk_sizes),
                    "updated_at": metadata.updated_at,
                    "size_mb": round(size_bytes / (1024 * 1024), 2),
                }
            else:
                entry = {
                    "cache_key": zarr_dir.stem,
                    "chunk_count": 0,
                    "total_records": 0,
                    "total_bytes": 0,
                    "chunk_sizes": [],
                    "updated_at": None,
                    "size_mb": round(size_bytes / (1024 * 1024), 2),
                }

            entries.append(entry)

        return {
            "store_directory": str(self.store_dir),
            "entry_count": len(entries),
            "total_size_mb": round(total_size_bytes / (1024 * 1024), 2),
            "entries": entries,
        }

    # ------------------------------------------------------------------
    # Internal metadata helpers
    # ------------------------------------------------------------------
    def _load_metadata_if_present(self, group: zarr.Group, cache_key: str) -> Optional[CacheMetadata]:
        dataset_name = self._metadata_dataset_name()
        if dataset_name not in group:
            return None

        try:
            raw_bytes = bytes(group[dataset_name][:])
            payload = json.loads(raw_bytes.decode("utf-8"))
            metadata = CacheMetadata.from_dict(payload)
            if not metadata.cache_key:
                metadata.cache_key = cache_key
            return metadata
        except Exception as metadata_error:
            print(f"[WARNING] Failed to read metadata chunk for {cache_key}: {metadata_error}")
            return None

    def _load_or_initialise_metadata(
        self,
        group: zarr.Group,
        cache_key: str,
    ) -> CacheMetadata:
        existing_metadata = self._load_metadata_if_present(group, cache_key)

        if existing_metadata:
            return existing_metadata

        metadata = CacheMetadata(cache_key=cache_key)
        self._persist_metadata(group, metadata)
        return metadata

    def _persist_metadata(self, group: zarr.Group, metadata: CacheMetadata) -> None:
        dataset_name = self._metadata_dataset_name()
        payload_bytes = json.dumps(metadata.to_dict(), ensure_ascii=False).encode("utf-8")
        data_array = np.frombuffer(payload_bytes, dtype=np.uint8)
        chunk_len = min(len(data_array), self.chunk_max_bytes)
        if chunk_len == 0:
            return

        group.create_dataset(
            dataset_name,
            data=data_array,
            compressor=self.compressor,
            overwrite=True,
            chunks=(chunk_len,),
        )
        group[dataset_name].attrs.update({
            "kind": "metadata",
            "created_at": metadata.created_at,
            "updated_at": metadata.updated_at,
        })

        group.attrs.update({
            "cache_key": metadata.cache_key,
            "chunk_count": metadata.chunk_count,
            "total_records": metadata.total_records,
            "total_bytes": metadata.total_bytes,
            "updated_at": metadata.updated_at,
            "next_index": metadata.next_chunk_index,
        })

    @staticmethod
    def _format_chunk_name(index: int) -> str:
        return f"chunk_{index:06d}"

    @staticmethod
    def _normalise_chunk_payload(candidate: Any) -> ChunkPayload:
        if isinstance(candidate, ChunkPayload):
            return candidate

        if isinstance(candidate, tuple) and candidate:
            # Allow callers to yield (payload, ...) without forcing a custom dataclass.
            return ChunkPayload(payload=candidate[0])

        return ChunkPayload(payload=candidate)


_shared_zarr_store = ZarrTelemetryStore()


def get_shared_zarr_store() -> ZarrTelemetryStore:
    return _shared_zarr_store
