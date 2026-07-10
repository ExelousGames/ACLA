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


def _segment(segment_id, label="MSP"):
    return {
        "labels": [label],
        "chunk_index": segment_id,
        "start_index": segment_id,
        "end_index": segment_id + 1,
        "telemetry_data": [
            {
                "Static_track": "test-track",
                "speed": 42 + segment_id,
                "brake": 0,
            }
        ],
    }


def _service_with_store(store):
    service = SegmentClassifierService.__new__(SegmentClassifierService)
    service.store = store
    service.max_length = 100
    return service


def _saved_segments(store, cache_key):
    saved_segments = []
    for chunk in store.saved.get(cache_key, {}).values():
        saved_segments.extend(chunk)
    return saved_segments


def _saved_segment_ids(store, *cache_keys):
    saved_segments = []
    for cache_key in cache_keys:
        saved_segments.extend(_saved_segments(store, cache_key))
    return {segment["chunk_index"] for segment in saved_segments}


@pytest.mark.asyncio
async def test_prepare_training_data_uses_all_segments_from_all_chunks():
    store = _Store({
        "source": {
            "chunk-a": [_segment(1), _segment(2)],
            "chunk-b": [_segment(3)],
            "chunk-c": [_segment(4)],
        }
    })
    service = _service_with_store(store)

    await service.prepare_training_data("source", "train", "val", val_split=0.2)

    assert _saved_segment_ids(store, "train", "val") == {1, 2, 3, 4}


@pytest.mark.asyncio
async def test_prepare_training_data_uses_only_selected_sessions():
    store = _Store({
        "source": {
            "session-a": [_segment(1), _segment(2)],
            "session-b": [_segment(3)],
            "session-c": [_segment(4), _segment(5)],
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

    assert _saved_segment_ids(store, "train", "val") == {1, 2, 4, 5}


@pytest.mark.asyncio
async def test_prepare_training_data_rejects_empty_session_selection():
    store = _Store({"source": {"session-a": [_segment(1)]}})
    service = _service_with_store(store)

    with pytest.raises(ValueError, match="At least one session must be selected"):
        await service.prepare_training_data(
            "source", "train", "val", session_ids=[]
        )


@pytest.mark.asyncio
async def test_prepare_training_data_rejects_selected_sessions_without_valid_segments():
    store = _Store({
        "source": {
            "session-a": [_segment(1)],
            "session-empty": [{"labels": ["MSP"], "telemetry_data": []}],
        }
    })
    service = _service_with_store(store)

    with pytest.raises(ValueError, match="No valid labeled segments.*session-empty"):
        await service.prepare_training_data(
            "source",
            "train",
            "val",
            session_ids=["session-empty"],
        )


@pytest.mark.asyncio
async def test_single_source_chunk_can_still_produce_training_data():
    store = _Store({
        "source": {
            "single-chunk": [_segment(1), _segment(2), _segment(3)],
        }
    })
    service = _service_with_store(store)

    await service.prepare_training_data("source", "train", "val", val_split=0.5)

    assert _saved_segments(store, "train")
    assert _saved_segment_ids(store, "train", "val") == {1, 2, 3}


@pytest.mark.asyncio
async def test_empty_source_data_still_fails_during_preprocessor_fit():
    store = _Store({"source": {}})
    service = _service_with_store(store)

    await service.prepare_training_data("source", "train", "val", val_split=0.2)

    with pytest.raises(ValueError, match="No valid training data found in cache"):
        await service.fit_preprocessors("train")


@pytest.mark.asyncio
async def test_val_split_zero_keeps_all_segments_in_train():
    store = _Store({
        "source": {
            "chunk-a": [_segment(1), _segment(2)],
            "chunk-b": [_segment(3)],
        }
    })
    service = _service_with_store(store)

    await service.prepare_training_data("source", "train", "val", val_split=0)

    assert _saved_segment_ids(store, "train") == {1, 2, 3}
    assert _saved_segments(store, "val") == []


@pytest.mark.asyncio
async def test_positive_val_split_creates_train_and_val_when_possible():
    store = _Store({
        "source": {
            "chunk-a": [_segment(1), _segment(2)],
        }
    })
    service = _service_with_store(store)

    await service.prepare_training_data("source", "train", "val", val_split=0.2)

    assert _saved_segments(store, "train")
    assert _saved_segments(store, "val")
    assert _saved_segment_ids(store, "train", "val") == {1, 2}
