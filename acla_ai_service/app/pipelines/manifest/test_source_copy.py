from __future__ import annotations

import pytest

from app.pipelines.manifest.models import AnnotationNode
from app.pipelines.manifest.source_copy import (
    CHUNK_ID_PATH,
    DICTIONARY_KEY_PATH,
    sync_source_copy,
)


PROTECTION_REF = {
    "label": "source_row_id -> row_id",
    "output_path": "source_row_id",
    "input_path": "row_id",
}

DICTIONARY_KEY_PROTECTION_REF = {
    "label": "Dictionary key -> Dictionary key",
    "output_path": DICTIONARY_KEY_PATH,
    "input_path": DICTIONARY_KEY_PATH,
}


class MemoryStore:
    def __init__(self) -> None:
        self.data: dict[str, dict[str, object]] = {}

    def has_cached_data(self, cache_key: str) -> bool:
        return bool(self.data.get(cache_key))

    def list_chunk_ids(self, cache_key: str) -> list[str]:
        return sorted(self.data.get(cache_key, {}).keys())

    def get_chunk(self, cache_key: str, chunk_id: str):
        return self.data.get(cache_key, {}).get(chunk_id)

    def save_chunk(self, cache_key: str, chunk_id: str, payload) -> bool:
        self.data.setdefault(cache_key, {})[chunk_id] = payload
        return True


def test_source_copy_requires_configured_protection_reference() -> None:
    store = MemoryStore()
    store.save_chunk("source", "chunk-1", [{"row_id": 1, "v": "new"}])

    with pytest.raises(ValueError, match="protection is not configured"):
        sync_source_copy(
            store,
            source_key="source",
            copy_key="copy",
            output_key="output",
        )


def test_source_copy_copies_missing_chunks_with_configured_protection() -> None:
    store = MemoryStore()
    store.save_chunk("source", "chunk-1", [{"row_id": 1, "v": "new"}])

    summary = sync_source_copy(
        store,
        source_key="source",
        copy_key="copy",
        output_key="output",
        protection_reference=PROTECTION_REF,
    )

    assert summary.chunks_copied == 1
    assert store.get_chunk("copy", "chunk-1") == [{"row_id": 1, "v": "new"}]


def test_source_copy_updates_unreferenced_rows_from_source() -> None:
    store = MemoryStore()
    store.save_chunk("source", "chunk-1", [{"row_id": 1, "v": "new"}])
    store.save_chunk("copy", "chunk-1", [{"row_id": 1, "v": "old"}])

    summary = sync_source_copy(
        store,
        source_key="source",
        copy_key="copy",
        output_key="output",
        protection_reference=PROTECTION_REF,
    )

    assert summary.chunks_updated == 1
    assert summary.rows_preserved == 0
    assert store.get_chunk("copy", "chunk-1") == [{"row_id": 1, "v": "new"}]


def test_source_copy_preserves_referenced_input_rows() -> None:
    store = MemoryStore()
    store.save_chunk(
        "source",
        "chunk-1",
        [{"row_id": 1, "v": "new"}, {"row_id": 2, "v": "new"}],
    )
    store.save_chunk(
        "copy",
        "chunk-1",
        [{"row_id": 1, "v": "old"}, {"row_id": 2, "v": "old"}],
    )
    store.save_chunk(
        "output",
        "output-chunk",
        [
            {
                "source_row_id": 1,
            }
        ],
    )

    summary = sync_source_copy(
        store,
        source_key="source",
        copy_key="copy",
        output_key="output",
        protection_reference=PROTECTION_REF,
    )

    assert summary.chunks_updated == 1
    assert summary.rows_preserved == 1
    assert store.get_chunk("copy", "chunk-1") == [
        {"row_id": 1, "v": "old"},
        {"row_id": 2, "v": "new"},
    ]


def test_many_output_rows_can_reference_one_protected_input_row() -> None:
    store = MemoryStore()
    store.save_chunk(
        "source",
        "chunk-1",
        [{"row_id": 1, "v": "new"}, {"row_id": 2, "v": "new"}],
    )
    store.save_chunk(
        "copy",
        "chunk-1",
        [{"row_id": 1, "v": "old"}, {"row_id": 2, "v": "old"}],
    )
    store.save_chunk(
        "output",
        "output-chunk",
        [
            {
                "source_row_id": 1,
            },
            {
                "source_row_id": 1,
            },
        ],
    )

    summary = sync_source_copy(
        store,
        source_key="source",
        copy_key="copy",
        output_key="output",
        protection_reference=PROTECTION_REF,
    )

    assert summary.rows_preserved == 1
    assert store.get_chunk("copy", "chunk-1") == [
        {"row_id": 1, "v": "old"},
        {"row_id": 2, "v": "new"},
    ]


def test_source_copy_preserves_referenced_input_chunk() -> None:
    store = MemoryStore()
    store.save_chunk("source", "session-1", [{"row_id": 1, "v": "new"}])
    store.save_chunk("copy", "session-1", [{"row_id": 1, "v": "old"}])
    store.save_chunk("output", "session-1", [{"chunk_index": "session-1"}])

    summary = sync_source_copy(
        store,
        source_key="source",
        copy_key="copy",
        output_key="output",
        protection_reference={
            "output_path": "chunk_index",
            "input_path": CHUNK_ID_PATH,
        },
    )

    assert summary.chunks_skipped_unchanged == 1
    assert summary.rows_preserved == 1
    assert store.get_chunk("copy", "session-1") == [{"row_id": 1, "v": "old"}]


def test_source_copy_can_match_output_chunk_to_input_chunk() -> None:
    store = MemoryStore()
    store.save_chunk("source", "session-1", [{"row_id": 1, "v": "new"}])
    store.save_chunk("copy", "session-1", [{"row_id": 1, "v": "old"}])
    store.save_chunk("output", "session-1", [{"id": "annotation"}])

    summary = sync_source_copy(
        store,
        source_key="source",
        copy_key="copy",
        output_key="output",
        protection_reference={
            "output_path": CHUNK_ID_PATH,
            "input_path": CHUNK_ID_PATH,
        },
    )

    assert summary.rows_preserved == 1
    assert store.get_chunk("copy", "session-1") == [{"row_id": 1, "v": "old"}]


def test_source_copy_preserves_dictionary_entry_by_key() -> None:
    store = MemoryStore()
    store.save_chunk(
        "source",
        "chunk-1",
        {
            "row-1": {"v": "new"},
            "row-2": {"v": "new"},
        },
    )
    store.save_chunk(
        "copy",
        "chunk-1",
        {
            "row-1": {"v": "old"},
            "row-2": {"v": "old"},
        },
    )
    store.save_chunk(
        "output",
        "output-chunk",
        {"row-1": {"label": "protected"}},
    )

    summary = sync_source_copy(
        store,
        source_key="source",
        copy_key="copy",
        output_key="output",
        protection_reference=DICTIONARY_KEY_PROTECTION_REF,
    )

    assert summary.chunks_updated == 1
    assert summary.rows_preserved == 1
    assert store.get_chunk("copy", "chunk-1") == {
        "row-1": {"v": "old"},
        "row-2": {"v": "new"},
    }


def test_dictionary_input_rejects_value_field_identity() -> None:
    store = MemoryStore()
    store.save_chunk("source", "chunk-1", {"row-1": {"row_id": 1, "v": "new"}})
    store.save_chunk("copy", "chunk-1", {"row-1": {"row_id": 1, "v": "old"}})
    store.save_chunk("output", "output-chunk", [{"source_row_id": 1}])

    with pytest.raises(ValueError, match="dictionary key"):
        sync_source_copy(
            store,
            source_key="source",
            copy_key="copy",
            output_key="output",
            protection_reference=PROTECTION_REF,
        )


def test_dictionary_output_value_field_can_reference_input_key() -> None:
    store = MemoryStore()
    store.save_chunk(
        "source",
        "chunk-1",
        {
            "row-1": {"v": "new"},
            "row-2": {"v": "new"},
        },
    )
    store.save_chunk(
        "copy",
        "chunk-1",
        {
            "row-1": {"v": "old"},
            "row-2": {"v": "old"},
        },
    )
    store.save_chunk("output", "output-chunk", {"row-1": {"source_row_id": "row-1"}})

    summary = sync_source_copy(
        store,
        source_key="source",
        copy_key="copy",
        output_key="output",
        protection_reference={
            "output_path": "source_row_id",
            "input_path": DICTIONARY_KEY_PATH,
        },
    )

    assert summary.rows_preserved == 1
    assert store.get_chunk("copy", "chunk-1") == {
        "row-1": {"v": "old"},
        "row-2": {"v": "new"},
    }


def test_annotation_node_serializes_protection_reference() -> None:
    node = AnnotationNode(
        id="ann",
        kind="lap",
        protection_reference=PROTECTION_REF,
    )

    restored = AnnotationNode.from_dict(node.to_dict())

    assert restored.protection_reference == PROTECTION_REF
