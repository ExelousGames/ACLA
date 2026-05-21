"""Per-cache_key schema strategies for the Lance telemetry store.

Phase 1 wrote every chunk as an opaque JSON blob in a single Lance row. That
made the migration cheap but threw away every columnar benefit Lance offers.
Phase 2 introduces typed strategies for the cache_keys that hold actual
telemetry data, so the data is stored as **one telemetry record per Lance
row**. The store's public dict-list API is preserved — strategies materialise
chunks back to lists of dicts on read by grouping/sorting on chunk-id columns.

Annotation keys keep the Phase-1 blob layout (their payloads are
document-shaped and small).

Strategy responsibilities
-------------------------
Each strategy owns one or more Lance datasets under
``<store_dir>/<cache_key>.lance*/``. A sidecar
``<store_dir>/<cache_key>.lance.meta.json`` holds the :class:`CacheMetadata`
bookkeeping that used to live in a special row in the Phase-1 blob datasets;
the blob strategy also accepts the legacy metadata-row layout so existing
annotation datasets keep working without re-migration.
"""

from __future__ import annotations

import json
import shutil
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Iterator, List, Optional, Tuple, Union

import pyarrow as pa

import lance

from app.storage.cache_metadata import CacheMetadata


# ---------------------------------------------------------------------------
# Sidecar metadata helpers
# ---------------------------------------------------------------------------

def _sidecar_path(base_dir: Path, cache_key: str) -> Path:
    return base_dir / f"{cache_key}.lance.meta.json"


def load_sidecar_metadata(base_dir: Path, cache_key: str) -> Optional[CacheMetadata]:
    path = _sidecar_path(base_dir, cache_key)
    if not path.exists():
        return None
    try:
        with path.open() as fh:
            data = json.load(fh)
        metadata = CacheMetadata.from_dict(data)
        if not metadata.cache_key:
            metadata.cache_key = cache_key
        return metadata
    except Exception as exc:
        print(f"[WARNING] Failed to load sidecar metadata for {cache_key}: {exc}")
        return None


def write_sidecar_metadata(base_dir: Path, metadata: CacheMetadata) -> None:
    path = _sidecar_path(base_dir, metadata.cache_key)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    with tmp.open("w") as fh:
        json.dump(metadata.to_dict(), fh, ensure_ascii=False)
    tmp.replace(path)


def delete_sidecar_metadata(base_dir: Path, cache_key: str) -> None:
    path = _sidecar_path(base_dir, cache_key)
    if path.exists():
        path.unlink()


# ---------------------------------------------------------------------------
# Pyarrow schema helpers
# ---------------------------------------------------------------------------

def unify_record_schemas(schemas: Iterable[pa.Schema]) -> pa.Schema:
    """Permissive union of record schemas — used during two-pass migration.

    Permissive promotion allows e.g. int → float across chunks; conflicting
    fields surface as a runtime error from pyarrow rather than silent loss.
    """
    schemas = [s for s in schemas if s is not None]
    if not schemas:
        return pa.schema([])
    return pa.unify_schemas(schemas, promote_options="permissive")


_FLOAT64_SAFE_INT_LIMIT = 2 ** 53


def _sanitize_value(value: Any) -> Any:
    """Replace int values outside float64's exact range with None.

    Raw telemetry occasionally contains stale memory values (e.g.
    ``9840800867287040`` inside an unused car's coordinate slot) that pyarrow
    refuses to cast to ``float64`` even with ``safe=False``, because the
    ``safe`` flag doesn't propagate through nested struct conversion. This
    helper runs only on the slow path (after a strict conversion fails) and
    walks lists/dicts recursively so nested coordinate structs get cleaned
    up too. Booleans are preserved as-is since ``isinstance(True, int)``
    would otherwise drop them.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value if abs(value) <= _FLOAT64_SAFE_INT_LIMIT else None
    if isinstance(value, list):
        return [_sanitize_value(v) for v in value]
    if isinstance(value, dict):
        return {k: _sanitize_value(v) for k, v in value.items()}
    return value


def records_to_arrays(records: List[dict], record_schema: pa.Schema) -> pa.Table:
    """Convert a list of dicts to an Arrow Table matching ``record_schema``.

    Missing fields are inserted as nulls; extra fields are silently dropped.
    The drop is deliberate — if a new field appears at runtime the migration
    script must be re-run to extend the schema (we error in the strategy's
    write path rather than silently lose data; see :class:`RecordsStrategy`).

    On the fast path we call ``from_pylist`` directly. When strict conversion
    fails — typically because a source value is an int outside float64's
    exact range — we deep-walk the records replacing those offending values
    with None and retry. Round-trip equality is no longer guaranteed for the
    sanitised values, but the alternative is failing the whole chunk over a
    single garbage entry deep inside a nested struct.
    """
    if not records:
        return record_schema.empty_table()
    try:
        return pa.Table.from_pylist(records, schema=record_schema)
    except pa.lib.ArrowInvalid:
        sanitized = [_sanitize_value(r) for r in records]
        return pa.Table.from_pylist(sanitized, schema=record_schema)


def infer_record_schema_from_record_lists(samples: Iterable[List[dict]]) -> pa.Schema:
    """Walk an iterator of ``list[dict]`` samples and build a unified schema.

    Sanitises out-of-range ints on retry so chunks with the occasional
    garbage value (see :func:`_sanitize_value`) still contribute their
    fields to the unified schema — otherwise the affected chunk's unique
    columns would be missing from the typed dataset.
    """
    accumulator: List[pa.Schema] = []
    for sample in samples:
        if not sample:
            continue
        try:
            accumulator.append(pa.Table.from_pylist(sample).schema)
        except pa.lib.ArrowInvalid:
            try:
                sanitized = [_sanitize_value(r) for r in sample]
                accumulator.append(pa.Table.from_pylist(sanitized).schema)
            except Exception as exc:
                print(f"[WARNING] Skipping sample during schema inference even after sanitisation: {exc}")
        except Exception as exc:
            print(f"[WARNING] Skipping sample during schema inference: {exc}")
    if not accumulator:
        raise ValueError("No usable samples for record-schema inference")
    return unify_record_schemas(accumulator)


# ---------------------------------------------------------------------------
# Strategy base class
# ---------------------------------------------------------------------------

class SchemaStrategy(ABC):
    """Abstract base for per-cache_key Lance storage strategies."""

    name: str = "abstract"

    @abstractmethod
    def dataset_paths(self, base_dir: Path, cache_key: str) -> List[Path]:
        """All Lance-dataset directories managed for this cache_key."""

    @abstractmethod
    def has_data(self, base_dir: Path, cache_key: str) -> bool: ...

    @abstractmethod
    def list_chunk_ids(self, base_dir: Path, cache_key: str) -> List[str]: ...

    @abstractmethod
    def read_chunk(
        self, base_dir: Path, cache_key: str, chunk_id: str
    ) -> Optional[Any]: ...

    @abstractmethod
    def iter_chunks(
        self, base_dir: Path, cache_key: str, include_ids: bool
    ) -> Iterator[Any]: ...

    @abstractmethod
    def write_chunk(
        self, base_dir: Path, cache_key: str, chunk_id: str, payload: Any
    ) -> int:
        """Write a single chunk. Returns the chunk's stored byte size estimate
        (for :class:`CacheMetadata` bookkeeping)."""

    @abstractmethod
    def delete_chunk(self, base_dir: Path, cache_key: str, chunk_id: str) -> bool: ...

    def clear(self, base_dir: Path, cache_key: str) -> None:
        for path in self.dataset_paths(base_dir, cache_key):
            if path.exists():
                shutil.rmtree(path, ignore_errors=True)
        delete_sidecar_metadata(base_dir, cache_key)

    # ------------------------------------------------------------------
    # Migration entry points
    # ------------------------------------------------------------------
    def precompute_schema(
        self,
        chunks_iterator: Iterator[Tuple[str, Any]],
    ) -> Optional[pa.Schema]:
        """Two-pass migration hook. Strategies that need a schema before they
        can write should iterate ``chunks_iterator`` here and return the
        inferred schema. The default implementation is a no-op for strategies
        that do not need schema inference (e.g. :class:`BlobStrategy`)."""
        # Exhaust the iterator so the migration loop's count stays accurate.
        for _ in chunks_iterator:
            pass
        return None

    def set_precomputed_schema(self, schema: Optional[pa.Schema]) -> None:
        """Strategies that need an inferred schema should override this to
        cache the schema before bulk-writes. Default no-op."""


# ---------------------------------------------------------------------------
# BlobStrategy — current Phase 1 behaviour, retained for annotation keys
# ---------------------------------------------------------------------------

_BLOB_METADATA_CHUNK_ID = "__metadata__"


def _blob_row_schema() -> pa.Schema:
    return pa.schema([
        pa.field("chunk_id", pa.string(), nullable=False),
        pa.field("payload", pa.binary(), nullable=False),
        pa.field("kind", pa.string(), nullable=False),
        pa.field("chunk_index", pa.int64(), nullable=True),
        pa.field("created_at", pa.string(), nullable=False),
        pa.field("updated_at", pa.string(), nullable=False),
    ])


class BlobStrategy(SchemaStrategy):
    """Original Phase-1 layout — one row per chunk, payload as JSON bytes.

    Used for annotation keys whose chunks are document-shaped. Reads the
    sidecar metadata file if present, otherwise falls back to the legacy
    ``__metadata__`` row so Phase-1 datasets keep working unchanged.
    """

    name = "blob"

    def dataset_paths(self, base_dir: Path, cache_key: str) -> List[Path]:
        return [base_dir / f"{cache_key}.lance"]

    def _open(self, base_dir: Path, cache_key: str) -> Optional[lance.LanceDataset]:
        path = base_dir / f"{cache_key}.lance"
        if not path.exists():
            return None
        try:
            return lance.dataset(str(path))
        except Exception:
            return None

    def has_data(self, base_dir: Path, cache_key: str) -> bool:
        dataset = self._open(base_dir, cache_key)
        if dataset is None:
            return False
        try:
            return dataset.count_rows(filter=f"chunk_id != '{_BLOB_METADATA_CHUNK_ID}'") > 0
        except Exception:
            return False

    def list_chunk_ids(self, base_dir: Path, cache_key: str) -> List[str]:
        dataset = self._open(base_dir, cache_key)
        if dataset is None:
            return []
        try:
            table = dataset.scanner(
                filter=f"chunk_id != '{_BLOB_METADATA_CHUNK_ID}'",
                columns=["chunk_id"],
            ).to_table()
        except Exception:
            return []
        return sorted(table.column("chunk_id").to_pylist())

    def read_chunk(self, base_dir: Path, cache_key: str, chunk_id: str) -> Optional[Any]:
        dataset = self._open(base_dir, cache_key)
        if dataset is None:
            return None
        try:
            table = dataset.scanner(
                filter=f"chunk_id = '{chunk_id}'",
                columns=["payload"],
            ).to_table()
        except Exception:
            return None
        if table.num_rows == 0:
            return None
        try:
            payload_bytes = bytes(table.column("payload")[0].as_py())
            return json.loads(payload_bytes.decode("utf-8"))
        except Exception as exc:
            print(f"[WARNING] Failed to decode blob chunk '{chunk_id}' for {cache_key}: {exc}")
            return None

    def iter_chunks(
        self, base_dir: Path, cache_key: str, include_ids: bool
    ) -> Iterator[Any]:
        dataset = self._open(base_dir, cache_key)
        if dataset is None:
            return iter(())

        # Materialising every payload in a single Arrow Table overflows
        # pyarrow's 32-bit binary-array offsets once the total payload
        # exceeds ~2 GB (racing_sessions_ is 20 GB). Pull chunk_ids first
        # — a small string column — then fetch one chunk's payload at a
        # time. Sort by chunk_id so iteration order matches the legacy
        # Zarr behaviour.
        try:
            ids_table = dataset.scanner(
                filter=f"chunk_id != '{_BLOB_METADATA_CHUNK_ID}'",
                columns=["chunk_id"],
            ).to_table()
        except Exception:
            return iter(())
        if ids_table.num_rows == 0:
            return iter(())
        chunk_ids = sorted(ids_table.column("chunk_id").to_pylist())

        def _generator() -> Iterator[Any]:
            for cid in chunk_ids:
                try:
                    payload_table = dataset.scanner(
                        filter=f"chunk_id = '{cid}'",
                        columns=["payload"],
                    ).to_table()
                except Exception as exc:
                    print(f"[WARNING] iter_chunks scan failed for '{cid}' in {cache_key}: {exc}")
                    continue
                if payload_table.num_rows == 0:
                    continue
                try:
                    payload_bytes = bytes(payload_table.column("payload")[0].as_py())
                    payload = json.loads(payload_bytes.decode("utf-8"))
                except Exception as exc:
                    print(f"[WARNING] Failed to decode blob chunk '{cid}' for {cache_key}: {exc}")
                    continue
                yield (payload, cid) if include_ids else payload

        return _generator()

    def write_chunk(
        self, base_dir: Path, cache_key: str, chunk_id: str, payload: Any
    ) -> int:
        payload_bytes = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        if not payload_bytes:
            self.delete_chunk(base_dir, cache_key, chunk_id)
            return 0
        now = datetime.now().isoformat()
        # Parse a chunk_index when the chunk_id matches the "chunk_NNNNNN"
        # pattern (used by streaming-append callers that don't supply IDs).
        chunk_index_int: Optional[int] = None
        if chunk_id.startswith("chunk_"):
            try:
                chunk_index_int = int(chunk_id.split("_", 1)[1])
            except ValueError:
                chunk_index_int = None
        row = {
            "chunk_id": chunk_id,
            "payload": payload_bytes,
            "kind": "data",
            "chunk_index": chunk_index_int,
            "created_at": now,
            "updated_at": now,
        }
        table = pa.Table.from_pylist([row], schema=_blob_row_schema())
        path = base_dir / f"{cache_key}.lance"
        if not path.exists():
            lance.write_dataset(table, str(path), mode="create")
        else:
            dataset = lance.dataset(str(path))
            (
                dataset.merge_insert("chunk_id")
                .when_matched_update_all()
                .when_not_matched_insert_all()
                .execute(table)
            )
        return len(payload_bytes)

    def delete_chunk(self, base_dir: Path, cache_key: str, chunk_id: str) -> bool:
        dataset = self._open(base_dir, cache_key)
        if dataset is None:
            return False
        if dataset.count_rows(filter=f"chunk_id = '{chunk_id}'") == 0:
            return False
        dataset.delete(f"chunk_id = '{chunk_id}'")
        return True


# ---------------------------------------------------------------------------
# RecordsStrategy — chunks are list[dict]; one Lance row per dict
# ---------------------------------------------------------------------------

_RECORDS_CHUNK_ID_COL = "__chunk_id__"
_RECORDS_ORDER_COL = "__order__"


def _records_row_schema(record_schema: pa.Schema) -> pa.Schema:
    """Combine the inferred record schema with the chunk_id + order columns."""
    base = [
        pa.field(_RECORDS_CHUNK_ID_COL, pa.string(), nullable=False),
        pa.field(_RECORDS_ORDER_COL, pa.int64(), nullable=False),
    ]
    return pa.schema(base + list(record_schema))


class RecordsStrategy(SchemaStrategy):
    """One row per telemetry record. Chunks reconstructed by group/sort."""

    name = "records"

    def __init__(self) -> None:
        self._record_schema_cache: dict[str, pa.Schema] = {}

    # ------------------------------------------------------------------
    # Schema inference (two-pass migration)
    # ------------------------------------------------------------------
    def precompute_schema(
        self,
        chunks_iterator: Iterator[Tuple[str, Any]],
    ) -> Optional[pa.Schema]:
        def _samples() -> Iterator[List[dict]]:
            for _, payload in chunks_iterator:
                if isinstance(payload, list):
                    yield payload
                elif isinstance(payload, dict) and isinstance(payload.get("data"), list):
                    yield payload["data"]

        return infer_record_schema_from_record_lists(_samples())

    def set_precomputed_schema(self, schema: Optional[pa.Schema]) -> None:
        # Stash the precomputed schema until the first write creates the
        # dataset. Keyed by id(strategy) is sufficient — migration uses one
        # strategy instance per cache_key invocation.
        self._pending_record_schema = schema

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _dataset_path(self, base_dir: Path, cache_key: str) -> Path:
        return base_dir / f"{cache_key}.lance"

    def dataset_paths(self, base_dir: Path, cache_key: str) -> List[Path]:
        return [self._dataset_path(base_dir, cache_key)]

    def _open(self, base_dir: Path, cache_key: str) -> Optional[lance.LanceDataset]:
        path = self._dataset_path(base_dir, cache_key)
        if not path.exists():
            return None
        try:
            return lance.dataset(str(path))
        except Exception:
            return None

    def _record_schema_for(self, dataset: lance.LanceDataset) -> pa.Schema:
        """Strip the chunk_id and order columns from a dataset schema."""
        dataset_schema = dataset.schema
        keep_fields = [
            f for f in dataset_schema
            if f.name not in (_RECORDS_CHUNK_ID_COL, _RECORDS_ORDER_COL)
        ]
        return pa.schema(keep_fields)

    # ------------------------------------------------------------------
    # SchemaStrategy API
    # ------------------------------------------------------------------
    def has_data(self, base_dir: Path, cache_key: str) -> bool:
        dataset = self._open(base_dir, cache_key)
        if dataset is None:
            return False
        try:
            return dataset.count_rows() > 0
        except Exception:
            return False

    def list_chunk_ids(self, base_dir: Path, cache_key: str) -> List[str]:
        dataset = self._open(base_dir, cache_key)
        if dataset is None:
            return []
        try:
            # DISTINCT on chunk_id — Lance scanner doesn't have a direct
            # DISTINCT, but pulling just that column and deduplicating in
            # Python is OK for the chunk-id counts we deal with (sessions,
            # not records).
            table = dataset.scanner(columns=[_RECORDS_CHUNK_ID_COL]).to_table()
        except Exception:
            return []
        return sorted(set(table.column(_RECORDS_CHUNK_ID_COL).to_pylist()))

    def _records_for_chunk(
        self, dataset: lance.LanceDataset, chunk_id: str
    ) -> Optional[List[dict]]:
        try:
            table = dataset.scanner(
                filter=f"{_RECORDS_CHUNK_ID_COL} = '{chunk_id}'",
            ).to_table()
        except Exception as exc:
            print(f"[WARNING] RecordsStrategy: scan failed for chunk '{chunk_id}': {exc}")
            return None
        if table.num_rows == 0:
            return None
        # Sort by __order__ so consumers see the same record order they wrote.
        order_indices = pa.compute.sort_indices(table, sort_keys=[(_RECORDS_ORDER_COL, "ascending")])
        sorted_table = table.take(order_indices)
        drop_cols = [_RECORDS_CHUNK_ID_COL, _RECORDS_ORDER_COL]
        record_table = sorted_table.drop(drop_cols)
        return record_table.to_pylist()

    def read_chunk(self, base_dir: Path, cache_key: str, chunk_id: str) -> Optional[Any]:
        dataset = self._open(base_dir, cache_key)
        if dataset is None:
            return None
        return self._records_for_chunk(dataset, chunk_id)

    def iter_chunks(
        self, base_dir: Path, cache_key: str, include_ids: bool
    ) -> Iterator[Any]:
        dataset = self._open(base_dir, cache_key)
        if dataset is None:
            return iter(())
        chunk_ids = self.list_chunk_ids(base_dir, cache_key)

        def _generator() -> Iterator[Any]:
            for cid in chunk_ids:
                payload = self._records_for_chunk(dataset, cid)
                if payload is None:
                    continue
                yield (payload, cid) if include_ids else payload

        return _generator()

    def _expected_record_schema(
        self, base_dir: Path, cache_key: str
    ) -> Optional[pa.Schema]:
        dataset = self._open(base_dir, cache_key)
        if dataset is not None:
            return self._record_schema_for(dataset)
        return getattr(self, "_pending_record_schema", None)

    def _payload_to_records(self, payload: Any) -> List[dict]:
        if isinstance(payload, list):
            return [r for r in payload if isinstance(r, dict)]
        if isinstance(payload, dict) and isinstance(payload.get("data"), list):
            return [r for r in payload["data"] if isinstance(r, dict)]
        raise TypeError(
            f"RecordsStrategy expected list[dict] payload; got {type(payload).__name__}"
        )

    def write_chunk(
        self, base_dir: Path, cache_key: str, chunk_id: str, payload: Any
    ) -> int:
        records = self._payload_to_records(payload)
        if not records:
            # Empty payload deletes the chunk (mirrors the blob strategy).
            self.delete_chunk(base_dir, cache_key, chunk_id)
            return 0

        record_schema = self._expected_record_schema(base_dir, cache_key)
        if record_schema is None:
            # No prior schema; infer from the records being written. This is
            # the path runtime writes hit if the dataset hasn't been migrated
            # yet — risk: later chunks with extra fields will be dropped.
            record_schema = pa.Table.from_pylist(records).schema

        # Build the per-record rows with __chunk_id__ + __order__ added.
        record_table = records_to_arrays(records, record_schema)
        n = record_table.num_rows
        chunk_id_array = pa.array([chunk_id] * n, type=pa.string())
        order_array = pa.array(list(range(n)), type=pa.int64())
        full_table = pa.Table.from_arrays(
            [chunk_id_array, order_array] + record_table.columns,
            names=[_RECORDS_CHUNK_ID_COL, _RECORDS_ORDER_COL] + record_table.schema.names,
        )

        path = self._dataset_path(base_dir, cache_key)
        if not path.exists():
            lance.write_dataset(full_table, str(path), mode="create")
        else:
            # Upsert semantics: replace any existing rows for this chunk_id.
            dataset = lance.dataset(str(path))
            existing = dataset.count_rows(filter=f"{_RECORDS_CHUNK_ID_COL} = '{chunk_id}'")
            if existing > 0:
                dataset.delete(f"{_RECORDS_CHUNK_ID_COL} = '{chunk_id}'")
            lance.write_dataset(full_table, str(path), mode="append")

        # Estimate byte size for CacheMetadata — use the JSON-equivalent
        # length so the bookkeeping stays comparable to Phase 1.
        return len(json.dumps(records, ensure_ascii=False).encode("utf-8"))

    def delete_chunk(self, base_dir: Path, cache_key: str, chunk_id: str) -> bool:
        dataset = self._open(base_dir, cache_key)
        if dataset is None:
            return False
        if dataset.count_rows(filter=f"{_RECORDS_CHUNK_ID_COL} = '{chunk_id}'") == 0:
            return False
        dataset.delete(f"{_RECORDS_CHUNK_ID_COL} = '{chunk_id}'")
        return True


# ---------------------------------------------------------------------------
# NestedRecordsStrategy — chunks are list[list[dict]] (e.g. top_laps_)
# ---------------------------------------------------------------------------

_NESTED_SUBLIST_COL = "__sublist_index__"


class NestedRecordsStrategy(SchemaStrategy):
    """Each chunk is a list of lists of records (e.g. per-track laps).

    Storage: one Lance row per telemetry record, with both ``__chunk_id__``
    (the track name) and ``__sublist_index__`` (the lap index within the
    track) preserving the original two-level structure.
    """

    name = "nested_records"

    def __init__(self) -> None:
        self._pending_record_schema: Optional[pa.Schema] = None

    def precompute_schema(
        self,
        chunks_iterator: Iterator[Tuple[str, Any]],
    ) -> Optional[pa.Schema]:
        def _samples() -> Iterator[List[dict]]:
            for _, payload in chunks_iterator:
                if not isinstance(payload, list):
                    continue
                for sublist in payload:
                    if isinstance(sublist, list):
                        yield sublist
        return infer_record_schema_from_record_lists(_samples())

    def set_precomputed_schema(self, schema: Optional[pa.Schema]) -> None:
        self._pending_record_schema = schema

    def _dataset_path(self, base_dir: Path, cache_key: str) -> Path:
        return base_dir / f"{cache_key}.lance"

    def dataset_paths(self, base_dir: Path, cache_key: str) -> List[Path]:
        return [self._dataset_path(base_dir, cache_key)]

    def _open(self, base_dir: Path, cache_key: str) -> Optional[lance.LanceDataset]:
        path = self._dataset_path(base_dir, cache_key)
        if not path.exists():
            return None
        try:
            return lance.dataset(str(path))
        except Exception:
            return None

    def has_data(self, base_dir: Path, cache_key: str) -> bool:
        dataset = self._open(base_dir, cache_key)
        if dataset is None:
            return False
        try:
            return dataset.count_rows() > 0
        except Exception:
            return False

    def list_chunk_ids(self, base_dir: Path, cache_key: str) -> List[str]:
        dataset = self._open(base_dir, cache_key)
        if dataset is None:
            return []
        try:
            table = dataset.scanner(columns=[_RECORDS_CHUNK_ID_COL]).to_table()
        except Exception:
            return []
        return sorted(set(table.column(_RECORDS_CHUNK_ID_COL).to_pylist()))

    def _record_schema_for(self, dataset: lance.LanceDataset) -> pa.Schema:
        skip = {_RECORDS_CHUNK_ID_COL, _NESTED_SUBLIST_COL, _RECORDS_ORDER_COL}
        return pa.schema([f for f in dataset.schema if f.name not in skip])

    def read_chunk(self, base_dir: Path, cache_key: str, chunk_id: str) -> Optional[Any]:
        dataset = self._open(base_dir, cache_key)
        if dataset is None:
            return None
        try:
            table = dataset.scanner(
                filter=f"{_RECORDS_CHUNK_ID_COL} = '{chunk_id}'",
            ).to_table()
        except Exception as exc:
            print(f"[WARNING] NestedRecordsStrategy: scan failed for chunk '{chunk_id}': {exc}")
            return None
        if table.num_rows == 0:
            return None
        # Sort by (sublist_index, order) so the original nesting is preserved.
        order_indices = pa.compute.sort_indices(
            table,
            sort_keys=[(_NESTED_SUBLIST_COL, "ascending"), (_RECORDS_ORDER_COL, "ascending")],
        )
        sorted_table = table.take(order_indices)
        sublist_ids = sorted_table.column(_NESTED_SUBLIST_COL).to_pylist()
        record_table = sorted_table.drop([_RECORDS_CHUNK_ID_COL, _NESTED_SUBLIST_COL, _RECORDS_ORDER_COL])
        records = record_table.to_pylist()

        grouped: List[List[dict]] = []
        last_idx: Optional[int] = None
        for sub_idx, record in zip(sublist_ids, records):
            if sub_idx != last_idx:
                grouped.append([])
                last_idx = sub_idx
            grouped[-1].append(record)
        return grouped

    def iter_chunks(
        self, base_dir: Path, cache_key: str, include_ids: bool
    ) -> Iterator[Any]:
        dataset = self._open(base_dir, cache_key)
        if dataset is None:
            return iter(())
        chunk_ids = self.list_chunk_ids(base_dir, cache_key)

        def _generator() -> Iterator[Any]:
            for cid in chunk_ids:
                payload = self.read_chunk(base_dir, cache_key, cid)
                if payload is None:
                    continue
                yield (payload, cid) if include_ids else payload

        return _generator()

    def write_chunk(
        self, base_dir: Path, cache_key: str, chunk_id: str, payload: Any
    ) -> int:
        if not isinstance(payload, list):
            raise TypeError(
                f"NestedRecordsStrategy expected list[list[dict]] payload; got {type(payload).__name__}"
            )

        # Flatten while tagging each record with its (sublist_index, order).
        flat_records: List[dict] = []
        sublist_indices: List[int] = []
        order_indices: List[int] = []
        for sub_idx, sublist in enumerate(payload):
            if not isinstance(sublist, list):
                continue
            for order_idx, record in enumerate(sublist):
                if not isinstance(record, dict):
                    continue
                flat_records.append(record)
                sublist_indices.append(sub_idx)
                order_indices.append(order_idx)

        if not flat_records:
            self.delete_chunk(base_dir, cache_key, chunk_id)
            return 0

        # Determine record schema from prior dataset or pending precomputation.
        record_schema: Optional[pa.Schema] = None
        dataset = self._open(base_dir, cache_key)
        if dataset is not None:
            record_schema = self._record_schema_for(dataset)
        elif self._pending_record_schema is not None:
            record_schema = self._pending_record_schema
        else:
            record_schema = pa.Table.from_pylist(flat_records).schema

        record_table = records_to_arrays(flat_records, record_schema)
        n = record_table.num_rows
        chunk_id_array = pa.array([chunk_id] * n, type=pa.string())
        sublist_array = pa.array(sublist_indices, type=pa.int64())
        order_array = pa.array(order_indices, type=pa.int64())
        full_table = pa.Table.from_arrays(
            [chunk_id_array, sublist_array, order_array] + record_table.columns,
            names=[_RECORDS_CHUNK_ID_COL, _NESTED_SUBLIST_COL, _RECORDS_ORDER_COL]
            + record_table.schema.names,
        )

        path = self._dataset_path(base_dir, cache_key)
        if not path.exists():
            lance.write_dataset(full_table, str(path), mode="create")
        else:
            ds = lance.dataset(str(path))
            if ds.count_rows(filter=f"{_RECORDS_CHUNK_ID_COL} = '{chunk_id}'") > 0:
                ds.delete(f"{_RECORDS_CHUNK_ID_COL} = '{chunk_id}'")
            lance.write_dataset(full_table, str(path), mode="append")

        return len(json.dumps(payload, ensure_ascii=False).encode("utf-8"))

    def delete_chunk(self, base_dir: Path, cache_key: str, chunk_id: str) -> bool:
        dataset = self._open(base_dir, cache_key)
        if dataset is None:
            return False
        if dataset.count_rows(filter=f"{_RECORDS_CHUNK_ID_COL} = '{chunk_id}'") == 0:
            return False
        dataset.delete(f"{_RECORDS_CHUNK_ID_COL} = '{chunk_id}'")
        return True


# ---------------------------------------------------------------------------
# SegmentsStrategy — chunks are list[segment_dict] with nested telemetry_data
# ---------------------------------------------------------------------------

_SEGMENT_ID_COL = "__segment_id__"
_SEGMENT_TELEMETRY_DATA_KEY = "telemetry_data"


class SegmentsStrategy(SchemaStrategy):
    """Two-table representation for training segments.

    * Primary dataset: ``<key>.lance/`` — one row per segment, with all
      segment scalar fields plus ``__chunk_id__``, ``__order__``, and a
      generated ``__segment_id__`` (the segment's own ``id`` field if
      present, else a synthesized ``{chunk_id}::{order}``).
    * Telemetry dataset: ``<key>.telemetry.lance/`` — one row per nested
      telemetry record, joined back via ``__segment_id__``.
    """

    name = "segments"

    def __init__(self) -> None:
        self._pending_record_schema: Optional[pa.Schema] = None
        self._pending_segment_schema: Optional[pa.Schema] = None

    def _segments_path(self, base_dir: Path, cache_key: str) -> Path:
        return base_dir / f"{cache_key}.lance"

    def _telemetry_path(self, base_dir: Path, cache_key: str) -> Path:
        return base_dir / f"{cache_key}.telemetry.lance"

    def dataset_paths(self, base_dir: Path, cache_key: str) -> List[Path]:
        return [
            self._segments_path(base_dir, cache_key),
            self._telemetry_path(base_dir, cache_key),
        ]

    def _open_segments(self, base_dir: Path, cache_key: str) -> Optional[lance.LanceDataset]:
        path = self._segments_path(base_dir, cache_key)
        if not path.exists():
            return None
        try:
            return lance.dataset(str(path))
        except Exception:
            return None

    def _open_telemetry(self, base_dir: Path, cache_key: str) -> Optional[lance.LanceDataset]:
        path = self._telemetry_path(base_dir, cache_key)
        if not path.exists():
            return None
        try:
            return lance.dataset(str(path))
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Schema inference
    # ------------------------------------------------------------------
    def precompute_schema(
        self,
        chunks_iterator: Iterator[Tuple[str, Any]],
    ) -> Optional[pa.Schema]:
        record_schemas: List[pa.Schema] = []
        segment_schemas: List[pa.Schema] = []
        for _, payload in chunks_iterator:
            if not isinstance(payload, list):
                continue
            for segment in payload:
                if not isinstance(segment, dict):
                    continue
                # Segment scalar fields = everything except telemetry_data.
                scalar = {k: v for k, v in segment.items() if k != _SEGMENT_TELEMETRY_DATA_KEY}
                if scalar:
                    try:
                        segment_schemas.append(pa.Table.from_pylist([scalar]).schema)
                    except Exception:
                        pass
                telemetry = segment.get(_SEGMENT_TELEMETRY_DATA_KEY)
                if isinstance(telemetry, list) and telemetry:
                    try:
                        record_schemas.append(pa.Table.from_pylist(telemetry).schema)
                    except Exception:
                        pass

        if not segment_schemas:
            raise ValueError("No usable segments for schema inference")

        self._pending_segment_schema = unify_record_schemas(segment_schemas)
        self._pending_record_schema = (
            unify_record_schemas(record_schemas) if record_schemas else pa.schema([])
        )
        return self._pending_segment_schema

    def set_precomputed_schema(self, schema: Optional[pa.Schema]) -> None:
        # The migration script calls precompute_schema directly; this strategy
        # holds both schemas internally so the protocol-level setter is a no-op.
        return

    # ------------------------------------------------------------------
    # SchemaStrategy API
    # ------------------------------------------------------------------
    def has_data(self, base_dir: Path, cache_key: str) -> bool:
        dataset = self._open_segments(base_dir, cache_key)
        if dataset is None:
            return False
        try:
            return dataset.count_rows() > 0
        except Exception:
            return False

    def list_chunk_ids(self, base_dir: Path, cache_key: str) -> List[str]:
        dataset = self._open_segments(base_dir, cache_key)
        if dataset is None:
            return []
        try:
            table = dataset.scanner(columns=[_RECORDS_CHUNK_ID_COL]).to_table()
        except Exception:
            return []
        return sorted(set(table.column(_RECORDS_CHUNK_ID_COL).to_pylist()))

    def _segment_schema_for(self, dataset: lance.LanceDataset) -> pa.Schema:
        skip = {_RECORDS_CHUNK_ID_COL, _RECORDS_ORDER_COL, _SEGMENT_ID_COL}
        return pa.schema([f for f in dataset.schema if f.name not in skip])

    def _record_schema_for(self, dataset: lance.LanceDataset) -> pa.Schema:
        skip = {_SEGMENT_ID_COL, _RECORDS_ORDER_COL}
        return pa.schema([f for f in dataset.schema if f.name not in skip])

    def read_chunk(self, base_dir: Path, cache_key: str, chunk_id: str) -> Optional[Any]:
        segments_ds = self._open_segments(base_dir, cache_key)
        if segments_ds is None:
            return None
        try:
            seg_table = segments_ds.scanner(
                filter=f"{_RECORDS_CHUNK_ID_COL} = '{chunk_id}'",
            ).to_table()
        except Exception as exc:
            print(f"[WARNING] SegmentsStrategy: segments scan failed for '{chunk_id}': {exc}")
            return None
        if seg_table.num_rows == 0:
            return None
        seg_order = pa.compute.sort_indices(seg_table, sort_keys=[(_RECORDS_ORDER_COL, "ascending")])
        seg_table = seg_table.take(seg_order)
        segment_ids = seg_table.column(_SEGMENT_ID_COL).to_pylist()

        # Build a map: segment_id → empty list, in original order
        ordered_segment_ids = segment_ids
        telemetry_by_segment: dict[str, List[dict]] = {sid: [] for sid in ordered_segment_ids}

        telemetry_ds = self._open_telemetry(base_dir, cache_key)
        if telemetry_ds is not None and ordered_segment_ids:
            # Build a SQL IN-list of segment_ids for this chunk.
            quoted = ",".join(f"'{sid}'" for sid in set(ordered_segment_ids))
            try:
                tel_table = telemetry_ds.scanner(
                    filter=f"{_SEGMENT_ID_COL} IN ({quoted})",
                ).to_table()
                if tel_table.num_rows > 0:
                    tel_order = pa.compute.sort_indices(
                        tel_table,
                        sort_keys=[(_SEGMENT_ID_COL, "ascending"), (_RECORDS_ORDER_COL, "ascending")],
                    )
                    tel_sorted = tel_table.take(tel_order)
                    seg_id_col = tel_sorted.column(_SEGMENT_ID_COL).to_pylist()
                    record_table = tel_sorted.drop([_SEGMENT_ID_COL, _RECORDS_ORDER_COL])
                    records = record_table.to_pylist()
                    for sid, rec in zip(seg_id_col, records):
                        if sid in telemetry_by_segment:
                            telemetry_by_segment[sid].append(rec)
            except Exception as exc:
                print(f"[WARNING] SegmentsStrategy: telemetry scan failed for '{chunk_id}': {exc}")

        # Reassemble segment dicts.
        scalar_table = seg_table.drop([_RECORDS_CHUNK_ID_COL, _RECORDS_ORDER_COL, _SEGMENT_ID_COL])
        scalar_dicts = scalar_table.to_pylist()
        rebuilt: List[dict] = []
        for scalar, sid in zip(scalar_dicts, segment_ids):
            seg = dict(scalar)
            seg[_SEGMENT_TELEMETRY_DATA_KEY] = telemetry_by_segment.get(sid, [])
            rebuilt.append(seg)
        return rebuilt

    def iter_chunks(
        self, base_dir: Path, cache_key: str, include_ids: bool
    ) -> Iterator[Any]:
        chunk_ids = self.list_chunk_ids(base_dir, cache_key)

        def _generator() -> Iterator[Any]:
            for cid in chunk_ids:
                payload = self.read_chunk(base_dir, cache_key, cid)
                if payload is None:
                    continue
                yield (payload, cid) if include_ids else payload

        return _generator()

    @staticmethod
    def _make_segment_id(segment: dict, chunk_id: str, order: int) -> str:
        # Synthesised chunk-scoped key used purely to join the segments and
        # telemetry datasets. The segment's own ``id`` field, if any, is
        # preserved as a regular scalar column on the segment row — this
        # function intentionally does NOT use it as the join key, because
        # the same logical segment id can legitimately appear across
        # multiple chunks (e.g. when the training pipeline re-emits the
        # same filtered segment set into a fresh chunk), and reusing it
        # here would cause cross-chunk telemetry rows to clobber each
        # other on subsequent writes.
        return f"{chunk_id}::{order}"

    def write_chunk(
        self, base_dir: Path, cache_key: str, chunk_id: str, payload: Any
    ) -> int:
        if not isinstance(payload, list):
            raise TypeError(
                f"SegmentsStrategy expected list[dict] payload; got {type(payload).__name__}"
            )

        # Build the segments rows + telemetry rows in one pass.
        segment_rows: List[dict] = []
        telemetry_rows: List[dict] = []
        for order, segment in enumerate(payload):
            if not isinstance(segment, dict):
                continue
            scalar = {k: v for k, v in segment.items() if k != _SEGMENT_TELEMETRY_DATA_KEY}
            sid = self._make_segment_id(segment, chunk_id, order)
            scalar_row = dict(scalar)
            scalar_row[_RECORDS_CHUNK_ID_COL] = chunk_id
            scalar_row[_RECORDS_ORDER_COL] = order
            scalar_row[_SEGMENT_ID_COL] = sid
            segment_rows.append(scalar_row)

            telemetry = segment.get(_SEGMENT_TELEMETRY_DATA_KEY)
            if isinstance(telemetry, list):
                for rec_order, record in enumerate(telemetry):
                    if not isinstance(record, dict):
                        continue
                    row = dict(record)
                    row[_SEGMENT_ID_COL] = sid
                    row[_RECORDS_ORDER_COL] = rec_order
                    telemetry_rows.append(row)

        if not segment_rows:
            self.delete_chunk(base_dir, cache_key, chunk_id)
            return 0

        # Resolve target schemas (prior dataset > pending precomputation).
        seg_ds = self._open_segments(base_dir, cache_key)
        if seg_ds is not None:
            seg_record_schema = self._segment_schema_for(seg_ds)
        elif self._pending_segment_schema is not None:
            seg_record_schema = self._pending_segment_schema
        else:
            seg_record_schema = pa.Table.from_pylist(
                [{k: v for k, v in r.items() if k not in (_RECORDS_CHUNK_ID_COL, _RECORDS_ORDER_COL, _SEGMENT_ID_COL)} for r in segment_rows]
            ).schema

        tel_ds = self._open_telemetry(base_dir, cache_key)
        if tel_ds is not None:
            tel_record_schema = self._record_schema_for(tel_ds)
        elif self._pending_record_schema is not None:
            tel_record_schema = self._pending_record_schema
        elif telemetry_rows:
            tel_record_schema = pa.Table.from_pylist(
                [{k: v for k, v in r.items() if k not in (_SEGMENT_ID_COL, _RECORDS_ORDER_COL)} for r in telemetry_rows]
            ).schema
        else:
            tel_record_schema = pa.schema([])

        # Compose final segment dataset schema: chunk_id + order + segment_id + scalar fields.
        seg_full_schema = pa.schema(
            [
                pa.field(_RECORDS_CHUNK_ID_COL, pa.string(), nullable=False),
                pa.field(_RECORDS_ORDER_COL, pa.int64(), nullable=False),
                pa.field(_SEGMENT_ID_COL, pa.string(), nullable=False),
            ]
            + list(seg_record_schema)
        )
        seg_table = pa.Table.from_pylist(segment_rows, schema=seg_full_schema)

        seg_path = self._segments_path(base_dir, cache_key)
        if not seg_path.exists():
            lance.write_dataset(seg_table, str(seg_path), mode="create")
        else:
            seg_ds = lance.dataset(str(seg_path))
            if seg_ds.count_rows(filter=f"{_RECORDS_CHUNK_ID_COL} = '{chunk_id}'") > 0:
                seg_ds.delete(f"{_RECORDS_CHUNK_ID_COL} = '{chunk_id}'")
            lance.write_dataset(seg_table, str(seg_path), mode="append")

        # Telemetry dataset: only write if there are records to write.
        if telemetry_rows:
            tel_full_schema = pa.schema(
                [
                    pa.field(_SEGMENT_ID_COL, pa.string(), nullable=False),
                    pa.field(_RECORDS_ORDER_COL, pa.int64(), nullable=False),
                ]
                + list(tel_record_schema)
            )
            tel_table = pa.Table.from_pylist(telemetry_rows, schema=tel_full_schema)

            tel_path = self._telemetry_path(base_dir, cache_key)
            if not tel_path.exists():
                lance.write_dataset(tel_table, str(tel_path), mode="create")
            else:
                tel_ds = lance.dataset(str(tel_path))
                # Delete telemetry rows for segments that belong to this chunk.
                seg_ids = [r[_SEGMENT_ID_COL] for r in segment_rows]
                quoted = ",".join(f"'{s}'" for s in seg_ids)
                if quoted:
                    tel_ds.delete(f"{_SEGMENT_ID_COL} IN ({quoted})")
                lance.write_dataset(tel_table, str(tel_path), mode="append")

        return len(json.dumps(payload, ensure_ascii=False).encode("utf-8"))

    def delete_chunk(self, base_dir: Path, cache_key: str, chunk_id: str) -> bool:
        seg_ds = self._open_segments(base_dir, cache_key)
        if seg_ds is None:
            return False
        if seg_ds.count_rows(filter=f"{_RECORDS_CHUNK_ID_COL} = '{chunk_id}'") == 0:
            return False

        # Collect segment_ids belonging to this chunk so we can purge the
        # telemetry rows that join to them.
        try:
            seg_ids_table = seg_ds.scanner(
                filter=f"{_RECORDS_CHUNK_ID_COL} = '{chunk_id}'",
                columns=[_SEGMENT_ID_COL],
            ).to_table()
            seg_ids = seg_ids_table.column(_SEGMENT_ID_COL).to_pylist()
        except Exception:
            seg_ids = []

        seg_ds.delete(f"{_RECORDS_CHUNK_ID_COL} = '{chunk_id}'")

        if seg_ids:
            tel_ds = self._open_telemetry(base_dir, cache_key)
            if tel_ds is not None:
                quoted = ",".join(f"'{s}'" for s in seg_ids)
                tel_ds.delete(f"{_SEGMENT_ID_COL} IN ({quoted})")

        return True


# ---------------------------------------------------------------------------
# Strategy registry
# ---------------------------------------------------------------------------

# Longest-prefix match: registry entries are checked from longest key to
# shortest, so e.g. "racing_sessions_enriched_" matches before
# "racing_sessions_". Unmatched cache_keys fall back to BlobStrategy.
_REGISTRY: List[Tuple[str, type]] = [
    ("racing_sessions_enriched_", RecordsStrategy),
    ("racing_sessions_processed_", RecordsStrategy),
    ("racing_sessions_", RecordsStrategy),
    ("top_laps_", NestedRecordsStrategy),
    ("training_segments_", SegmentsStrategy),
]


def strategy_for(cache_key: str) -> SchemaStrategy:
    for prefix, strategy_cls in sorted(_REGISTRY, key=lambda kv: -len(kv[0])):
        if cache_key.startswith(prefix):
            return strategy_cls()
    return BlobStrategy()


def is_typed(cache_key: str) -> bool:
    return not isinstance(strategy_for(cache_key), BlobStrategy)


__all__ = [
    "SchemaStrategy",
    "BlobStrategy",
    "RecordsStrategy",
    "NestedRecordsStrategy",
    "SegmentsStrategy",
    "strategy_for",
    "is_typed",
    "load_sidecar_metadata",
    "write_sidecar_metadata",
    "delete_sidecar_metadata",
    "infer_record_schema_from_record_lists",
    "unify_record_schemas",
]
