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

    def save_chunk(self, cache_key, chunk_id, payload):
        self.saved.setdefault(cache_key, {})[str(chunk_id)] = payload
        return True


def _segment(segment_id, label="MSP", parent_id=None):
    segment = {
        "id": f"segment-{segment_id}",
        "labels": [label],
        "start_index": segment_id,
        "end_index": segment_id + 2,
        "telemetry_data": [{"speed": 40}, {"speed": 41}],
    }
    if parent_id is not None:
        segment["parent_id"] = parent_id
    return segment


def _service(store):
    service = SegmentClassifierService.__new__(SegmentClassifierService)
    service.store = store
    return service


def _saved_session_ids(store, *keys):
    return {
        session_id
        for key in keys
        for session_id in store.saved.get(key, {})
    }


@pytest.mark.asyncio
async def test_prepare_training_data_splits_samples_and_keeps_children_with_parent():
    store = _Store({
        "source": {
            "session-a": [
                _segment(1),
                _segment(101, label="MSP1", parent_id="segment-1"),
                _segment(2),
            ],
            "session-b": [_segment(3)],
            "session-c": [_segment(4)],
        }
    })

    await _service(store).prepare_training_data(
        "source",
        "train",
        "val",
        val_split=0.25,
    )

    train_records = [
        record for records in store.saved["train"].values() for record in records
    ]
    val_records = [
        record for records in store.saved["val"].values() for record in records
    ]
    train_parents = {record["id"] for record in train_records if not record.get("parent_id")}
    val_parents = {record["id"] for record in val_records if not record.get("parent_id")}

    assert len(train_parents) == 3
    assert len(val_parents) == 1
    assert train_parents.isdisjoint(val_parents)
    assert train_parents | val_parents == {
        "segment-1",
        "segment-2",
        "segment-3",
        "segment-4",
    }
    child_records = [
        record
        for record in train_records + val_records
        if record.get("parent_id")
    ]
    assert len(child_records) == 1
    child_partition = train_parents if child_records[0] in train_records else val_parents
    assert child_records[0]["parent_id"] in child_partition


@pytest.mark.asyncio
async def test_prepare_training_data_uses_only_selected_sessions():
    store = _Store({
        "source": {
            "session-a": [_segment(1)],
            "session-b": [_segment(2)],
            "session-c": [_segment(3)],
        }
    })

    await _service(store).prepare_training_data(
        "source",
        "train",
        "val",
        session_ids=["session-a", "session-c"],
    )

    assert _saved_session_ids(store, "train", "val") == {"session-a", "session-c"}


@pytest.mark.asyncio
async def test_prepare_training_data_rejects_empty_session_selection():
    store = _Store({"source": {"session-a": [_segment(1)]}})

    with pytest.raises(ValueError, match="At least one session"):
        await _service(store).prepare_training_data(
            "source",
            "train",
            "val",
            session_ids=[],
        )


@pytest.mark.asyncio
async def test_prepare_training_data_rejects_missing_selected_sessions():
    store = _Store({"source": {"session-a": [_segment(1)]}})

    with pytest.raises(ValueError, match="No annotation sessions.*session-missing"):
        await _service(store).prepare_training_data(
            "source",
            "train",
            "val",
            session_ids=["session-missing"],
        )


@pytest.mark.asyncio
async def test_zero_validation_split_keeps_every_session_in_training():
    store = _Store({
        "source": {
            "session-a": [_segment(1)],
            "session-b": [_segment(2)],
        }
    })

    await _service(store).prepare_training_data(
        "source",
        "train",
        "val",
        val_split=0,
    )

    assert _saved_session_ids(store, "train") == {"session-a", "session-b"}
    assert store.saved["val"] == {}


@pytest.mark.asyncio
async def test_positive_validation_split_requires_two_samples():
    store = _Store({"source": {"session-a": [_segment(1)]}})

    with pytest.raises(ValueError, match="at least two annotated behavior samples"):
        await _service(store).prepare_training_data(
            "source",
            "train",
            "val",
            val_split=0.1,
        )
