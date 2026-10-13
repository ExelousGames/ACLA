"""Source-copy helpers for source-mode annotation nodes.

Source-mode annotations read from a private copy of their configured input.
Refreshing that copy preserves copied rows whose selected input identity field
is referenced by the selected output field.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any, Mapping, Sequence


SOURCE_COPY_SUFFIX = "__source_copy"
CHUNK_ID_PATH = "__chunk_id__"
DICTIONARY_KEY_PATH = "__dictionary_key__"


def source_copy_key(output_key: str) -> str:
    """Return the cache key used for a source-mode node's copied input."""
    return f"{output_key}{SOURCE_COPY_SUFFIX}"


@dataclass(frozen=True)
class ProtectionReference:
    output_path: str
    input_path: str
    label: str = ""


@dataclass
class SourceCopySummary:
    source_key: str
    copy_key: str
    source_chunks_total: int = 0
    chunks_copied: int = 0
    chunks_updated: int = 0
    chunks_skipped_unchanged: int = 0
    rows_preserved: int = 0
    read_failures: list[str] = field(default_factory=list)
    write_failures: list[str] = field(default_factory=list)


def _coerce_reference(reference: ProtectionReference | Mapping[str, Any] | None) -> ProtectionReference:
    if isinstance(reference, ProtectionReference):
        return reference
    if not isinstance(reference, Mapping):
        raise ValueError("Source copy protection is not configured.")
    output_path = str(reference.get("output_path") or "").strip()
    input_path = str(reference.get("input_path") or "").strip()
    if not output_path or not input_path:
        raise ValueError("Source copy protection is missing output_path or input_path.")
    return ProtectionReference(
        output_path=output_path,
        input_path=input_path,
        label=str(reference.get("label") or output_path),
    )


def _rows_from_payload(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict) and isinstance(payload.get("data"), list):
        return [row for row in payload["data"] if isinstance(row, dict)]
    return []


def _output_rows_from_payload(payload: Any) -> list[dict[str, Any]]:
    rows = _rows_from_payload(payload)
    if rows:
        return rows
    if isinstance(payload, dict):
        return [value for value in payload.values() if isinstance(value, dict)]
    return []


def _replace_payload_rows(payload: Any, rows: Sequence[Mapping[str, Any]]) -> Any:
    new_rows = [dict(row) for row in rows]
    if isinstance(payload, list):
        return new_rows
    if isinstance(payload, dict) and isinstance(payload.get("data"), list):
        updated = dict(payload)
        updated["data"] = new_rows
        return updated
    return payload


def _path_parts(path: str) -> list[tuple[str, bool]]:
    parts: list[tuple[str, bool]] = []
    for raw_part in path.split("."):
        part = raw_part.strip()
        if not part:
            continue
        if part.endswith("[]"):
            parts.append((part[:-2], True))
        else:
            parts.append((part, False))
    return parts


def _extract_values(value: Any, path: str) -> list[Any]:
    values = [value]
    for key, is_many in _path_parts(path):
        next_values: list[Any] = []
        for item in values:
            if not isinstance(item, Mapping) or key not in item:
                continue
            child = item[key]
            if is_many:
                if isinstance(child, list):
                    next_values.extend(child)
            else:
                next_values.append(child)
        values = next_values
        if not values:
            break
    return values


def _identity(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        return text or None
    if isinstance(value, (int, float, bool)):
        return str(value)
    try:
        return json.dumps(value, sort_keys=True, default=str)
    except TypeError:
        return str(value)


def _collect_protected_identities(
    store: Any,
    output_key: str | None,
    reference: ProtectionReference,
) -> set[str]:
    protected: set[str] = set()
    if not output_key:
        return protected
    try:
        if not store.has_cached_data(output_key):
            return protected
        output_chunk_ids = list(store.list_chunk_ids(output_key))
    except Exception:
        return protected

    for chunk_id in output_chunk_ids:
        try:
            payload = store.get_chunk(output_key, chunk_id)
        except Exception:
            continue
        if reference.output_path == CHUNK_ID_PATH:
            identity = _identity(chunk_id)
            if identity is not None:
                protected.add(identity)
            continue
        if isinstance(payload, dict) and not isinstance(payload.get("data"), list):
            if reference.output_path == DICTIONARY_KEY_PATH:
                for key in payload.keys():
                    identity = _identity(key)
                    if identity is not None:
                        protected.add(identity)
                continue
        for row in _output_rows_from_payload(payload):
            for value in _extract_values(row, reference.output_path):
                identity = _identity(value)
                if identity is not None:
                    protected.add(identity)
    return protected


def _row_identity(
    row: Mapping[str, Any],
    input_path: str,
) -> str | None:
    for value in _extract_values(row, input_path):
        identity = _identity(value)
        if identity is not None:
            return identity
    return None


def _merge_preserving_rows(
    source_payload: Any,
    existing_payload: Any,
    protected_identities: set[str],
    input_path: str,
    chunk_id: Any,
) -> tuple[Any, int]:
    if not protected_identities:
        return source_payload, 0

    if input_path == CHUNK_ID_PATH:
        identity = _identity(chunk_id)
        if identity in protected_identities:
            return existing_payload, _payload_item_count(existing_payload)
        return source_payload, 0

    if (
        isinstance(source_payload, dict)
        and isinstance(existing_payload, dict)
        and not isinstance(source_payload.get("data"), list)
        and not isinstance(existing_payload.get("data"), list)
    ):
        return _merge_preserving_dictionary_entries(
            source_payload,
            existing_payload,
            protected_identities,
            input_path,
        )

    source_rows = _rows_from_payload(source_payload)
    existing_rows = _rows_from_payload(existing_payload)
    if not source_rows or not existing_rows:
        return source_payload, 0

    existing_by_identity: dict[str, list[Mapping[str, Any]]] = {}
    for row in existing_rows:
        identity = _row_identity(row, input_path)
        if identity is not None:
            existing_by_identity.setdefault(identity, []).append(row)

    merged = []
    preserved = 0
    for row in source_rows:
        identity = _row_identity(row, input_path)
        candidates = existing_by_identity.get(identity or "")
        if identity in protected_identities and candidates:
            merged.append(candidates.pop(0))
            preserved += 1
        else:
            merged.append(row)

    if not preserved:
        return source_payload, 0
    return _replace_payload_rows(source_payload, merged), preserved


def _payload_item_count(payload: Any) -> int:
    rows = _rows_from_payload(payload)
    if rows:
        return len(rows)
    if isinstance(payload, dict) and not isinstance(payload.get("data"), list):
        return len(payload)
    return 1 if payload is not None else 0


def _merge_preserving_dictionary_entries(
    source_payload: Any,
    existing_payload: Any,
    protected_identities: set[str],
    input_path: str,
) -> tuple[Any, int]:
    if input_path != DICTIONARY_KEY_PATH:
        raise ValueError("Dictionary input datasets can only be protected by dictionary key.")
    if (
        not isinstance(source_payload, dict)
        or not isinstance(existing_payload, dict)
        or isinstance(source_payload.get("data"), list)
        or isinstance(existing_payload.get("data"), list)
    ):
        return source_payload, 0

    existing_by_identity: dict[str, Any] = {}
    for key, value in existing_payload.items():
        identity = _identity(key)
        if identity is not None:
            existing_by_identity[identity] = value

    merged = dict(source_payload)
    preserved = 0
    for key in source_payload.keys():
        identity = _identity(key)
        if identity in protected_identities and identity in existing_by_identity:
            merged[key] = existing_by_identity[identity]
            preserved += 1

    if not preserved:
        return source_payload, 0
    return merged, preserved


def sync_source_copy(
    store: Any,
    *,
    source_key: str,
    copy_key: str,
    output_key: str | None = None,
    protection_reference: ProtectionReference | Mapping[str, Any] | None = None,
) -> SourceCopySummary:
    """Copy/update source chunks into ``copy_key``.

    Rows whose input identity is referenced by annotation output are preserved
    from the existing copy. All unreferenced rows refresh from ``source_key``.
    """
    reference = _coerce_reference(protection_reference)
    summary = SourceCopySummary(source_key=source_key, copy_key=copy_key)
    protected_identities = _collect_protected_identities(store, output_key, reference)

    try:
        copied_chunk_ids = (
            set(store.list_chunk_ids(copy_key))
            if store.has_cached_data(copy_key)
            else set()
        )
    except Exception:
        copied_chunk_ids = set()

    try:
        source_chunk_ids = list(store.list_chunk_ids(source_key))
    except Exception as exc:
        summary.read_failures.append(f"{source_key}: {exc}")
        return summary

    summary.source_chunks_total = len(source_chunk_ids)
    for chunk_id in source_chunk_ids:
        try:
            payload = store.get_chunk(source_key, chunk_id)
        except Exception as exc:
            summary.read_failures.append(f"{chunk_id}: {exc}")
            continue
        if payload is None:
            summary.read_failures.append(f"{chunk_id}: empty")
            continue

        is_existing_chunk = chunk_id in copied_chunk_ids
        payload_to_save = payload
        if is_existing_chunk:
            try:
                existing_payload = store.get_chunk(copy_key, chunk_id)
            except Exception as exc:
                summary.read_failures.append(f"{copy_key}/{chunk_id}: {exc}")
                existing_payload = None
            if existing_payload is not None:
                payload_to_save, preserved = _merge_preserving_rows(
                    payload,
                    existing_payload,
                    protected_identities,
                    reference.input_path,
                    chunk_id,
                )
                summary.rows_preserved += preserved
                if payload_to_save == existing_payload:
                    summary.chunks_skipped_unchanged += 1
                    continue

        try:
            saved = store.save_chunk(copy_key, chunk_id, payload_to_save)
        except Exception as exc:
            summary.write_failures.append(f"{chunk_id}: {exc}")
            continue
        if saved:
            if is_existing_chunk:
                summary.chunks_updated += 1
            else:
                summary.chunks_copied += 1
            copied_chunk_ids.add(chunk_id)
        else:
            summary.write_failures.append(f"{chunk_id}: save failed")

    return summary


__all__ = [
    "CHUNK_ID_PATH",
    "DICTIONARY_KEY_PATH",
    "SOURCE_COPY_SUFFIX",
    "ProtectionReference",
    "SourceCopySummary",
    "source_copy_key",
    "sync_source_copy",
]
