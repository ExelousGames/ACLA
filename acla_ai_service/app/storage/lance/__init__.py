"""Lance-backed telemetry store.

This module is the migration target for the previous Zarr chunk store. It
exposes the same public surface as ``app.storage.zarr.ZarrTelemetryStore``
so the ten existing consumers (training pipelines, annotation UI, label
studio sync, ML services, inference) can be cut over by swapping the
factory return value without changing call sites.

Internally each ``cache_key`` is owned by a :class:`SchemaStrategy` picked
from :mod:`app.storage.lance.strategies`. Telemetry-shaped keys
(``racing_sessions_*``, ``top_laps_``, ``training_segments_``) use typed
columnar strategies that store one telemetry record per Lance row, while
annotation keys stay on the original blob strategy. The store's dict-list
API is preserved end-to-end — typed strategies reconstruct chunks via
``__chunk_id__`` / ``__order__`` columns at read time.
"""

from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncIterator, Iterable, Iterator, List, Optional, Union

from app.storage.cache_metadata import CacheMetadata, ChunkPayload
from app.storage.lance.strategies import (
    SchemaStrategy,
    BlobStrategy,
    load_sidecar_metadata,
    write_sidecar_metadata,
    delete_sidecar_metadata,
    strategy_for,
    is_typed,
)


def _format_chunk_name(index: Union[int, str]) -> str:
    """Mirror :func:`app.storage.zarr._format_chunk_name`.

    Integer chunk indices are zero-padded so sorted iteration matches
    insertion order; string chunk_ids (e.g. session UUIDs) pass through.
    """
    if isinstance(index, int):
        return f"chunk_{index:06d}"
    return str(index)


def _parse_chunk_index(chunk_id: str) -> Optional[int]:
    if chunk_id.startswith("chunk_"):
        try:
            return int(chunk_id.split("_", 1)[1])
        except ValueError:
            return None
    return None


class LanceTelemetryStore:
    """Telemetry document store backed by Lance datasets."""

    def __init__(
        self,
        store_directory: Optional[str] = None,
        *,
        chunk_max_mb: float = 10,
    ) -> None:
        if store_directory:
            self.store_dir = Path(store_directory)
        else:
            # Default to acla_ai_service/app/storage/telemetry_lance_store/,
            # mirroring the location convention used by the Zarr store
            # (one directory over from this code module). __file__ is
            # app/storage/lance/__init__.py — parents[1] is app/storage/.
            self.store_dir = Path(__file__).resolve().parents[1] / "telemetry_lance_store"

        self.store_dir.mkdir(parents=True, exist_ok=True)
        self.chunk_max_bytes = max(1024, int(chunk_max_mb * 1024 * 1024))

    # ------------------------------------------------------------------
    # Strategy plumbing
    # ------------------------------------------------------------------
    def _strategy(self, cache_key: str) -> SchemaStrategy:
        return strategy_for(cache_key)

    def _load_metadata(self, cache_key: str) -> Optional[CacheMetadata]:
        """Read CacheMetadata from the sidecar, falling back to the legacy
        in-dataset metadata row used by Phase-1 blob datasets."""
        sidecar = load_sidecar_metadata(self.store_dir, cache_key)
        if sidecar is not None:
            return sidecar

        # Legacy fallback: peek inside the blob dataset for a __metadata__
        # row. Only meaningful for cache_keys whose strategy is BlobStrategy
        # AND that pre-date the sidecar migration.
        strategy = self._strategy(cache_key)
        if not isinstance(strategy, BlobStrategy):
            return None

        try:
            import json

            import lance

            path = self.store_dir / f"{cache_key}.lance"
            if not path.exists():
                return None
            dataset = lance.dataset(str(path))
            table = dataset.scanner(
                filter="chunk_id = '__metadata__'",
                columns=["payload"],
            ).to_table()
            if table.num_rows == 0:
                return None
            raw = bytes(table.column("payload")[0].as_py())
            data = json.loads(raw.decode("utf-8"))
            metadata = CacheMetadata.from_dict(data)
            if not metadata.cache_key:
                metadata.cache_key = cache_key
            return metadata
        except Exception:
            return None

    def _load_or_init_metadata(self, cache_key: str) -> CacheMetadata:
        existing = self._load_metadata(cache_key)
        if existing is not None:
            return existing
        return CacheMetadata(cache_key=cache_key)

    def _persist_metadata(self, metadata: CacheMetadata) -> None:
        write_sidecar_metadata(self.store_dir, metadata)

    @staticmethod
    def _ensure_async_iterator(iterator: Any) -> AsyncIterator[Any]:
        if hasattr(iterator, "__aiter__"):
            return iterator  # type: ignore[return-value]

        async def _adapter() -> AsyncIterator[Any]:
            for item in iterator:
                yield item

        return _adapter()

    @staticmethod
    def _normalise_chunk_payload(candidate: Any) -> ChunkPayload:
        if isinstance(candidate, ChunkPayload):
            return candidate
        if isinstance(candidate, tuple) and candidate:
            payload = candidate[0]
            chunk_id = candidate[1] if len(candidate) > 1 else None
            return ChunkPayload(payload=payload, chunk_id=chunk_id)
        return ChunkPayload(payload=candidate)

    # ------------------------------------------------------------------
    # Public API — mirrors ZarrTelemetryStore
    # ------------------------------------------------------------------
    async def cache_chunks_streaming(
        self,
        cache_key: str,
        chunks_iterator: Any,
    ) -> bool:
        strategy = self._strategy(cache_key)
        metadata = self._load_or_init_metadata(cache_key)

        processed = 0
        async for candidate in self._ensure_async_iterator(chunks_iterator):
            entry = self._normalise_chunk_payload(candidate)
            if entry.payload is None:
                continue

            if entry.chunk_id is not None:
                chunk_id = _format_chunk_name(entry.chunk_id)
            else:
                chunk_id = _format_chunk_name(metadata.next_chunk_index)

            try:
                size_estimate = strategy.write_chunk(
                    self.store_dir, cache_key, chunk_id, entry.payload
                )
            except Exception as exc:
                print(f"[WARNING] Failed to write chunk '{chunk_id}' for {cache_key}: {exc}")
                continue

            metadata.register_chunk(chunk_bytes=size_estimate)
            processed += 1

        self._persist_metadata(metadata)
        return processed > 0

    def save_chunk(self, cache_key: str, chunk_index: Union[int, str], payload: Any) -> bool:
        strategy = self._strategy(cache_key)
        metadata = self._load_or_init_metadata(cache_key)

        chunk_id = _format_chunk_name(chunk_index)
        try:
            size_estimate = strategy.write_chunk(self.store_dir, cache_key, chunk_id, payload)
        except Exception as exc:
            print(f"[WARNING] Failed to write chunk '{chunk_id}' for {cache_key}: {exc}")
            return False

        if isinstance(chunk_index, int):
            metadata.update_chunk(chunk_index, size_estimate)
        else:
            metadata.updated_at = datetime.now().isoformat()
        self._persist_metadata(metadata)
        return True

    def delete_chunk(self, cache_key: str, chunk_index: Union[int, str]) -> bool:
        strategy = self._strategy(cache_key)
        chunk_id = _format_chunk_name(chunk_index)
        deleted = strategy.delete_chunk(self.store_dir, cache_key, chunk_id)

        metadata = self._load_or_init_metadata(cache_key)
        if isinstance(chunk_index, int):
            metadata.update_chunk(chunk_index, 0)
        else:
            metadata.updated_at = datetime.now().isoformat()
        self._persist_metadata(metadata)
        return deleted

    def get_cached_data_chunks(self, cache_key: str, include_ids: bool = False) -> Iterator[Any]:
        strategy = self._strategy(cache_key)
        return strategy.iter_chunks(self.store_dir, cache_key, include_ids)

    def get_chunk(self, cache_key: str, chunk_index: Union[int, str]) -> Optional[Any]:
        strategy = self._strategy(cache_key)
        chunk_id = _format_chunk_name(chunk_index)
        return strategy.read_chunk(self.store_dir, cache_key, chunk_id)

    def get_cache_metadata(self, cache_key: str) -> Optional[CacheMetadata]:
        return self._load_metadata(cache_key)

    def has_cached_data(self, cache_key: str) -> bool:
        return self._strategy(cache_key).has_data(self.store_dir, cache_key)

    def list_cache_keys(self) -> List[str]:
        """Return cache_keys that have actual chunk data on disk.

        Typed strategies emit a single ``<key>.lance`` directory (with an
        optional ``<key>.telemetry.lance`` sidekick for segments). We scan
        for the primary one and ask the strategy whether the dataset is
        populated.
        """
        if not self.store_dir.exists():
            return []
        keys: List[str] = []
        for entry in self.store_dir.glob("*.lance"):
            cache_key = entry.stem
            # Strategies that emit auxiliary datasets name them with a known
            # suffix (e.g. ``<key>.telemetry.lance``). Filter those out so
            # they don't surface as separate cache_keys.
            if "." in cache_key:
                continue
            if self.has_cached_data(cache_key):
                keys.append(cache_key)
        return sorted(keys)

    def list_chunk_ids(self, cache_key: str) -> List[str]:
        return self._strategy(cache_key).list_chunk_ids(self.store_dir, cache_key)

    def clear_cache(self, cache_key: Optional[str] = None) -> None:
        if cache_key is None:
            if self.store_dir.exists():
                shutil.rmtree(self.store_dir)
            self.store_dir.mkdir(parents=True, exist_ok=True)
            return
        self._strategy(cache_key).clear(self.store_dir, cache_key)
        delete_sidecar_metadata(self.store_dir, cache_key)

    def get_cache_info(self) -> dict:
        entries = []
        total_size_bytes = 0

        if self.store_dir.exists():
            for entry_path in self.store_dir.glob("*.lance"):
                if "." in entry_path.stem:
                    continue  # skip auxiliary datasets
                cache_key = entry_path.stem
                size_bytes = sum(
                    f.stat().st_size
                    for path in [entry_path] + self._strategy(cache_key).dataset_paths(self.store_dir, cache_key)[1:]
                    for f in path.rglob("*")
                    if f.is_file()
                )
                total_size_bytes += size_bytes

                metadata = self._load_metadata(cache_key)
                if metadata:
                    entry = {
                        "cache_key": metadata.cache_key or cache_key,
                        "chunk_count": metadata.chunk_count,
                        "total_records": metadata.total_records,
                        "total_bytes": metadata.total_bytes,
                        "chunk_sizes": list(metadata.chunk_sizes),
                        "updated_at": metadata.updated_at,
                        "size_mb": round(size_bytes / (1024 * 1024), 2),
                        "strategy": self._strategy(cache_key).name,
                    }
                else:
                    entry = {
                        "cache_key": cache_key,
                        "chunk_count": 0,
                        "total_records": 0,
                        "total_bytes": 0,
                        "chunk_sizes": [],
                        "updated_at": None,
                        "size_mb": round(size_bytes / (1024 * 1024), 2),
                        "strategy": self._strategy(cache_key).name,
                    }
                entries.append(entry)

        return {
            "store_directory": str(self.store_dir),
            "entry_count": len(entries),
            "total_size_mb": round(total_size_bytes / (1024 * 1024), 2),
            "entries": entries,
        }


_shared_lance_store: Optional[LanceTelemetryStore] = None


def get_shared_lance_store() -> LanceTelemetryStore:
    global _shared_lance_store
    if _shared_lance_store is None:
        _shared_lance_store = LanceTelemetryStore()
    return _shared_lance_store
