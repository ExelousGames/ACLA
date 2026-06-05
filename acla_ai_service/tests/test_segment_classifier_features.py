from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from app.ml.segment_classifier.service import SegmentClassifierService


class FakeScaler:
    n_features_in_ = 486


class FakeSmallScaler:
    n_features_in_ = 4


class FakeMlb:
    classes_ = np.array(["EA", "MSP1"])


class FakeSingleLabelMlb:
    classes_ = np.array(["EA"])


class FakeLegacyMlb:
    classes_ = np.array(["MS1", "MSP1"])


class FakeModel:
    def __call__(self, x):
        return torch.tensor([[[2.0, 1.0]]]), None


class FakeSequenceModel:
    def __init__(self, probability: float):
        self.probability = probability

    def eval(self):
        return None

    def __call__(self, x):
        probabilities = torch.full((1, x.shape[1], 1), self.probability)
        return torch.logit(probabilities), None


class FakeIdentityScaler:
    def transform(self, values):
        return values


class FakeStore:
    def __init__(self, chunks):
        self.chunks = chunks
        self.cleared = []
        self.saved = {}

    def clear_cache(self, key):
        self.cleared.append(key)
        self.saved[key] = []

    def get_cached_data_chunks(self, cache_key):
        return self.chunks

    def save_chunk(self, cache_key, chunk_idx, payload):
        self.saved.setdefault(cache_key, []).append((chunk_idx, payload))


def test_legacy_scaler_uses_pre_gap_feature_layout():
    service = SegmentClassifierService.__new__(SegmentClassifierService)
    service.scaler = FakeScaler()
    service.feature_names = None

    features = service._feature_names_for_model()

    assert len(features) * 2 == FakeScaler.n_features_in_
    assert "Graphics_current_tyre_set" in features
    assert "Graphics_gap_ahead" not in features
    assert "Graphics_gap_behind" not in features

    row = {
        "Graphics_gap_ahead": 1.0,
        "Graphics_gap_behind": 2.0,
        "Graphics_current_tyre_set": 3.0,
    }
    numeric_df = service._prepare_numeric_features(pd.DataFrame([row]))

    assert numeric_df.shape[1] == FakeScaler.n_features_in_
    assert numeric_df["Graphics_current_tyre_set"].iloc[0] == 3.0


def test_prepare_numeric_features_warns_when_expected_columns_are_missing(caplog):
    service = SegmentClassifierService.__new__(SegmentClassifierService)
    service.scaler = FakeSmallScaler()
    service.feature_names = ["speed", "brake"]

    with caplog.at_level(logging.WARNING):
        numeric_df = service._prepare_numeric_features(pd.DataFrame([{"speed": 120.0}]))

    assert "segment_classifier input missing 1/2 expected feature columns" in caplog.text
    assert numeric_df["speed"].iloc[0] == 120.0
    assert numeric_df["brake"].iloc[0] == 0.0


def test_thresholds_are_loaded_per_label(tmp_path: Path):
    service = SegmentClassifierService.__new__(SegmentClassifierService)
    service.thresholds_path = tmp_path / "segment_thresholds.json"
    service.thresholds_path.write_text(
        '{"thresholds": {"EA": 0.7, "MSP1": 0.35, "bad": 2.0}}'
    )
    service.mlb = FakeMlb()

    service._load_label_thresholds()

    assert service._threshold_for_label("EA") == 0.7
    assert service._threshold_for_label("MSP1") == 0.7
    assert service._threshold_for_label("unknown") == 0.7
    assert service._thresholds_for_classes().tolist() == [0.7, 0.7]


def test_scan_telemetry_data_raises_when_model_is_missing():
    service = SegmentClassifierService.__new__(SegmentClassifierService)
    service.model = None
    service.load_model = lambda: False

    with pytest.raises(ValueError, match="Segment classifier model not trained or found"):
        service.scan_telemetry_data(pd.DataFrame([{"speed": 120.0}]))


def test_scan_telemetry_data_requires_seventy_percent_confidence():
    service = SegmentClassifierService.__new__(SegmentClassifierService)
    service.model = FakeSequenceModel(0.69)
    service.scaler = FakeIdentityScaler()
    service.mlb = FakeSingleLabelMlb()
    service.feature_names = ["speed"]
    service.label_thresholds = {}
    service.device = torch.device("cpu")

    data = pd.DataFrame([{"speed": 100.0}, {"speed": 101.0}, {"speed": 102.0}])

    assert service.scan_telemetry_data(data) == []

    service.model = FakeSequenceModel(0.7)
    segments = service.scan_telemetry_data(data)

    assert len(segments) == 1
    assert segments[0].labels == ["EA"]


def test_predict_segment_probabilities_normalizes_legacy_artifact_labels():
    service = SegmentClassifierService.__new__(SegmentClassifierService)
    service.model = FakeModel()
    service.mlb = FakeLegacyMlb()
    service.scaler = FakeIdentityScaler()
    service.max_length = 1
    service.device = torch.device("cpu")
    service._prepare_numeric_features = lambda dataframe: pd.DataFrame([{"speed": 1.0}])

    result = service.predict_segment_probabilities(pd.DataFrame([{"speed": 120.0}]))

    assert "MS1" not in result
    assert result["MSP1"] == pytest.approx(float(torch.sigmoid(torch.tensor(2.0))))


@pytest.mark.asyncio
async def test_prepare_training_data_groups_split_by_track_name():
    chunks = [
        [
            {"id": "brands-1", "chunk_index": "brands-session-a", "labels": ["EA"], "telemetry_data": [{"Static_track": "brands_hatch", "speed": 1}]},
            {"id": "brands-2", "chunk_index": "brands-session-b", "labels": ["MSP1"], "telemetry_data": [{"Static_track": "brands_hatch", "speed": 2}]},
        ],
        [
            {"id": "silverstone-1", "chunk_index": "silverstone-session-a", "labels": ["MSP1"], "telemetry_data": [{"Static_track": "silverstone", "speed": 3}]},
            {"id": "silverstone-2", "chunk_index": "silverstone-session-b", "labels": ["EA"], "telemetry_data": [{"Static_track": "silverstone", "speed": 4}]},
        ],
    ]
    service = SegmentClassifierService.__new__(SegmentClassifierService)
    service.store = FakeStore(chunks)
    service._assign_split = lambda segment_hash, val_split: "train"

    await service.prepare_training_data("source", "train", "val", val_split=0.2, chunk_size=10)

    train_segments = [
        segment
        for _, payload in service.store.saved["train"]
        for segment in payload
    ]
    val_segments = [
        segment
        for _, payload in service.store.saved["val"]
        for segment in payload
    ]

    train_tracks = {segment["telemetry_data"][0]["Static_track"] for segment in train_segments}
    val_tracks = {segment["telemetry_data"][0]["Static_track"] for segment in val_segments}

    assert train_tracks == {"brands_hatch", "silverstone"}
    assert val_tracks == {"brands_hatch", "silverstone"}
    assert len(train_segments) == 2
    assert len(val_segments) == 2


@pytest.mark.asyncio
async def test_prepare_training_data_splits_repeated_labels_within_track():
    chunks = [
        [
            {"id": "s1", "chunk_index": "session-a", "labels": ["EA"], "telemetry_data": [{"Static_track": "brands_hatch", "speed": 1}]},
            {"id": "s2", "chunk_index": "session-b", "labels": ["EA"], "telemetry_data": [{"Static_track": "brands_hatch", "speed": 2}]},
            {"id": "s3", "chunk_index": "session-c", "labels": ["MSP1"], "telemetry_data": [{"Static_track": "brands_hatch", "speed": 3}]},
            {"id": "s4", "chunk_index": "session-d", "labels": ["MSP1"], "telemetry_data": [{"Static_track": "brands_hatch", "speed": 4}]},
        ],
    ]
    service = SegmentClassifierService.__new__(SegmentClassifierService)
    service.store = FakeStore(chunks)
    service._assign_split = lambda segment_hash, val_split: "train"

    await service.prepare_training_data("source", "train", "val", val_split=0.2, chunk_size=10)

    train_segments = [
        segment
        for _, payload in service.store.saved["train"]
        for segment in payload
    ]
    val_segments = [
        segment
        for _, payload in service.store.saved["val"]
        for segment in payload
    ]

    train_labels = {label for segment in train_segments for label in segment["labels"]}
    val_labels = {label for segment in val_segments for label in segment["labels"]}
    train_sessions = {segment["chunk_index"] for segment in train_segments}
    val_sessions = {segment["chunk_index"] for segment in val_segments}

    assert train_labels == {"EA", "MSP1"}
    assert val_labels == {"EA", "MSP1"}
    assert train_sessions.isdisjoint(val_sessions)


@pytest.mark.asyncio
async def test_prepare_training_data_normalizes_legacy_labels_before_saving():
    chunks = [
        [
            {
                "chunk_index": "session-a",
                "labels": ["MS", "MS1", "MS53", "O2", 3],
                "telemetry_data": [{"speed": 1}],
            },
        ],
    ]
    service = SegmentClassifierService.__new__(SegmentClassifierService)
    service.store = FakeStore(chunks)
    service._assign_split = lambda segment_hash, val_split: "train"

    await service.prepare_training_data("source", "train", "val", val_split=0.2, chunk_size=10)

    saved_segments = [
        segment
        for split in ("train", "val")
        for _, payload in service.store.saved[split]
        for segment in payload
    ]

    assert saved_segments[0]["labels"] == ["MSP", "MSP1", "MSR1", "OD1", "EA"]
