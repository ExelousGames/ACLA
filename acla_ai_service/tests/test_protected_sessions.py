from __future__ import annotations

from app.integrations.backend.client import _filter_protected_session_metadata
from app.pipelines.manifest.models import MODE_SOURCE, AnnotationNode, Pipeline
from app.pipelines.manifest.protection import (
    clear_unprotected_chunks,
    collect_protected_session_ids,
    has_unprotected_chunks,
    is_protected_chunk_id,
    session_id_from_chunk_id,
)
from app.pipelines.manifest.segment_refresh import refresh_node_segments


class FakeStore:
    def __init__(self, chunks=None):
        self.chunks = chunks or {}
        self.registered_dirs = []
        self.saved = []
        self.cleared = []

    def register_directory(self, cache_key, directory):
        self.registered_dirs.append((cache_key, directory))

    def has_cached_data(self, cache_key):
        return bool(self.chunks.get(cache_key))

    def list_chunk_ids(self, cache_key):
        return sorted(self.chunks.get(cache_key, {}).keys())

    def get_chunk(self, cache_key, chunk_id):
        return self.chunks.get(cache_key, {}).get(chunk_id)

    def save_chunk(self, cache_key, chunk_id, payload):
        self.chunks.setdefault(cache_key, {})[chunk_id] = payload
        self.saved.append((cache_key, chunk_id, payload))
        return True

    def delete_chunk(self, cache_key, chunk_id):
        if chunk_id not in self.chunks.get(cache_key, {}):
            return False
        del self.chunks[cache_key][chunk_id]
        return True

    def clear_cache(self, cache_key):
        self.cleared.append(cache_key)
        self.chunks.pop(cache_key, None)


def test_chunk_id_matching_handles_raw_download_chunks() -> None:
    assert session_id_from_chunk_id("session-1:chunk_000000") == "session-1"
    assert session_id_from_chunk_id("session-1") == "session-1"
    assert is_protected_chunk_id("session-1:chunk_000001", {"session-1"})


def test_collect_protected_session_ids_scans_pipeline_outputs(monkeypatch) -> None:
    pipeline = Pipeline(
        id="pipe",
        annotations=[
            AnnotationNode(
                id="ann",
                kind="lap",
                output_key="ann_out",
                output_dir="/app/custom-output",
            )
        ],
    )
    store = FakeStore(
        {
            "ann_out": {
                "session-a": [{"labels": ["EA"]}],
                "session-b:chunk_000000": [{"labels": ["MSP"]}],
            }
        }
    )

    monkeypatch.setattr("app.pipelines.manifest.registry.list_pipelines", lambda: ["pipe"])
    monkeypatch.setattr("app.pipelines.manifest.registry.load", lambda pipeline_id: pipeline)

    assert collect_protected_session_ids(store) == {"session-a", "session-b"}
    assert store.registered_dirs == [("ann_out", "/app/custom-output")]


def test_clear_unprotected_chunks_keeps_protected_session_chunks() -> None:
    store = FakeStore(
        {
            "raw": {
                "session-a:chunk_000000": [{"row": 1}],
                "session-b:chunk_000000": [{"row": 2}],
            }
        }
    )

    summary = clear_unprotected_chunks(store, "raw", {"session-a"})

    assert summary.chunks_deleted == 1
    assert summary.chunks_kept == 1
    assert store.list_chunk_ids("raw") == ["session-a:chunk_000000"]
    assert has_unprotected_chunks(store, "raw", {"session-a"}) is False


def test_backend_metadata_filter_skips_protected_sessions() -> None:
    filtered, skipped = _filter_protected_session_metadata(
        [{"sessionId": "session-a"}, {"sessionId": "session-b"}],
        {"session-a"},
    )

    assert skipped == 1
    assert filtered == [{"sessionId": "session-b"}]


def test_default_annotation_reads_existing_source() -> None:
    pipeline = Pipeline(
        id="pipe",
        annotations=[
            AnnotationNode(
                id="ann",
                kind="lap",
                source_ref="source",
            )
        ],
    )

    assert pipeline.effective_input_key(pipeline.annotation("ann")) == "source"


def test_legacy_copy_mode_loads_as_source_mode() -> None:
    node = AnnotationNode.from_dict(
        {
            "id": "ann",
            "kind": "lap",
            "source_ref": "source",
            "input_key": "old_input_dataset",
            "mode": "copy",
        }
    )

    assert node.mode == MODE_SOURCE
    assert not hasattr(node, "input_key")


def test_refresh_node_segments_skips_protected_sessions() -> None:
    store = FakeStore(
        {
            "input": {
                "session-a": [{"new": "protected"}],
                "session-b": [{"new": "updated"}],
            },
            "output": {
                "session-a": [
                    {"start_index": 0, "end_index": 1, "telemetry_data": [{"old": 1}]}
                ],
                "session-b": [
                    {"start_index": 0, "end_index": 1, "telemetry_data": [{"old": 2}]}
                ],
            },
        }
    )
    node = AnnotationNode(
        id="ann",
        kind="lap",
        output_key="output",
    )

    summary = refresh_node_segments(
        store,
        node,
        input_key="input",
        protected_session_ids={"session-a"},
    )

    assert summary.chunks_skipped_protected == 1
    assert store.get_chunk("output", "session-a")[0]["telemetry_data"] == [{"old": 1}]
    assert store.get_chunk("output", "session-b")[0]["telemetry_data"] == [
        {"new": "updated"}
    ]
