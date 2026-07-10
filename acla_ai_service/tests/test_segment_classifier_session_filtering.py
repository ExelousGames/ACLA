import pytest

from app.ml.segment_classifier.service import SegmentClassifierService


class _Store:
    def __init__(self, source_chunks):
        self.source_chunks = source_chunks
        self.saved = {}

    def get_cached_data_chunks(self, cache_key, include_ids=False):
        chunks = self.saved.get(cache_key, self.source_chunks.get(cache_key, {}))
        for chunk_id, payload in chunks.items():
            yield (payload, chunk_id) if include_ids else payload

    def clear_cache(self, cache_key):
        self.saved[cache_key] = {}

    def save_chunk(self, cache_key, chunk_index, payload):
        self.saved.setdefault(cache_key, {})[str(chunk_index)] = payload
        return True


def _segment(session_id, label="MSP"):
    return {
        "labels": [label],
        "chunk_index": session_id,
        "start_index": 0,
        "end_index": 1,
        "telemetry_data": [
            {
                "Static_track": "test-track",
                "speed": 42,
                "brake": 0,
            }
        ],
    }


def _service_with_store(store):
    service = SegmentClassifierService.__new__(SegmentClassifierService)
    service.store = store
    service.max_length = 100
    return service


def _saved_session_ids(store):
    saved_segments = []
    for cache_key in ("train", "val"):
        for chunk in store.saved.get(cache_key, {}).values():
            saved_segments.extend(chunk)
    return {segment["chunk_index"] for segment in saved_segments}


@pytest.mark.asyncio
async def test_prepare_training_data_uses_all_sessions_by_default():
    store = _Store({
        "source": {
            "session-a": [_segment("session-a")],
            "session-b": [_segment("session-b")],
            "session-c": [_segment("session-c")],
        }
    })
    service = _service_with_store(store)

    await service.prepare_training_data("source", "train", "val", val_split=0.2)

    assert _saved_session_ids(store) == {"session-a", "session-b", "session-c"}


@pytest.mark.asyncio
async def test_prepare_training_data_filters_selected_sessions():
    store = _Store({
        "source": {
            "session-a": [_segment("session-a")],
            "session-b": [_segment("session-b")],
            "session-c": [_segment("session-c")],
        }
    })
    service = _service_with_store(store)

    await service.prepare_training_data(
        "source",
        "train",
        "val",
        val_split=0.2,
        session_ids=["session-a", "session-c"],
    )

    assert _saved_session_ids(store) == {"session-a", "session-c"}


@pytest.mark.asyncio
@pytest.mark.parametrize("session_ids", [[], ["missing-session"]])
async def test_selected_sessions_with_no_data_fail_during_preprocessor_fit(session_ids):
    store = _Store({
        "source": {
            "session-a": [_segment("session-a")],
        }
    })
    service = _service_with_store(store)

    await service.prepare_training_data(
        "source",
        "train",
        "val",
        val_split=0.2,
        session_ids=session_ids,
    )

    with pytest.raises(ValueError, match="No valid training data found in cache"):
        await service.fit_preprocessors("train")
