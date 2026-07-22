import json
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock

import numpy as np
import pandas as pd
import pytest
import torch

import app.ml.segment_classifier.service as service_module
from app.ml.segment_classifier.model import TemporalDetectionModel
from app.ml.segment_classifier.service import MODEL_FORMAT, SegmentClassifierService
from app.storage.datasets.segment_dataset import build_temporal_sequences


def _rows(start, end):
    return [
        {"speed": float(index), "brake": float(index % 2)}
        for index in range(start, end)
    ]


def test_temporal_targets_follow_parent_and_child_ranges():
    chunk = [
        {
            "id": "parent-msp",
            "labels": ["MSP", "ST1", "silverstone"],
            "start_index": 10,
            "end_index": 14,
            "telemetry_data": _rows(10, 14),
        },
        {
            "id": "parent-ea",
            "labels": ["EA", "ST2"],
            "start_index": 14,
            "end_index": 18,
            "telemetry_data": _rows(14, 18),
        },
        {
            "labels": ["MSP", "MSP1", "ST1"],
            "parent_id": "parent-msp",
            "start_index": 11,
            "end_index": 13,
            "telemetry_data": _rows(11, 13),
        },
    ]

    sequences = build_temporal_sequences(
        chunk,
        expected_features=["speed", "brake"],
        label_ids=["MSP", "EA", "MSP1"],
        child_parent={"MSP1": "MSP"},
    )

    assert len(sequences) == 1
    sequence = sequences[0]
    assert sequence.start_index == 10
    assert sequence.features.shape == (8, 4)
    np.testing.assert_array_equal(sequence.targets[:, 0], [1, 1, 1, 1, 0, 0, 0, 0])
    np.testing.assert_array_equal(sequence.targets[:, 1], [0, 0, 0, 0, 1, 1, 1, 1])
    np.testing.assert_array_equal(sequence.targets[:, 2], [0, 1, 1, 0, 0, 0, 0, 0])
    np.testing.assert_array_equal(sequence.loss_mask, np.ones_like(sequence.targets))


def test_temporal_sequence_builder_splits_uncovered_gaps():
    chunk = [
        {
            "labels": ["MSP"],
            "start_index": 0,
            "end_index": 2,
            "telemetry_data": _rows(0, 2),
        },
        {
            "labels": ["EA"],
            "start_index": 5,
            "end_index": 7,
            "telemetry_data": _rows(5, 7),
        },
    ]

    sequences = build_temporal_sequences(
        chunk,
        expected_features=["speed", "brake"],
        label_ids=["MSP", "EA"],
        child_parent={},
    )

    assert [(sequence.start_index, len(sequence.features)) for sequence in sequences] == [
        (0, 2),
        (5, 2),
    ]


def test_temporal_model_preserves_the_entire_sequence_length():
    model = TemporalDetectionModel(input_dim=6, output_dim=4, hidden_dim=8)

    assert model(torch.zeros(2, 7, 6)).shape == (2, 7, 4)
    assert model(torch.zeros(1, 257, 6)).shape == (1, 257, 4)


def test_rocm_training_enables_miopen_immediate_mode(monkeypatch, capsys):
    service = SegmentClassifierService.__new__(SegmentClassifierService)
    service.device = torch.device("cuda")
    monkeypatch.setattr(torch.version, "hip", "7.2")
    monkeypatch.setattr(torch.backends.miopen, "immediate", False)

    service._configure_training_backend()

    assert torch.backends.miopen.immediate is True
    assert "Enabled MIOpen Immediate Mode" in capsys.readouterr().out


def test_non_rocm_training_does_not_change_miopen_mode(monkeypatch, capsys):
    service = SegmentClassifierService.__new__(SegmentClassifierService)
    monkeypatch.setattr(torch.backends.miopen, "immediate", False)

    service.device = torch.device("cuda")
    monkeypatch.setattr(torch.version, "hip", None)
    service._configure_training_backend()

    service.device = torch.device("cpu")
    monkeypatch.setattr(torch.version, "hip", "7.2")
    service._configure_training_backend()

    assert torch.backends.miopen.immediate is False
    assert capsys.readouterr().out == ""


def _preprocessor_service(targets):
    service = SegmentClassifierService.__new__(SegmentClassifierService)
    service.device = torch.device("cpu")
    service.label_ids = ["MSP", "MSP1"]
    service.behavior_label_ids = ["MSP"]
    sequence = SimpleNamespace(
        features=np.array([[40.0, 0.0], [41.0, 1.0]], dtype=np.float32),
        targets=np.array(targets, dtype=np.float32),
        loss_mask=np.ones((2, 2), dtype=np.float32),
    )
    service._iter_sequences = lambda cache_key: iter([sequence])
    return service


@pytest.mark.asyncio
async def test_parent_only_training_warns_and_fits_preprocessors(caplog):
    service = _preprocessor_service([[1.0, 0.0], [0.0, 0.0]])

    with caplog.at_level("WARNING", logger=service_module.LOGGER.name):
        await service.fit_preprocessors("train")

    assert service.scaler is not None
    assert service.pos_weight is not None
    assert service.pos_weight.shape == (2,)
    assert "training will continue with parent behavior labels only" in caplog.text
    assert "child-label predictions from this model will not be reliable" in caplog.text


@pytest.mark.asyncio
async def test_child_annotated_training_does_not_warn(caplog):
    service = _preprocessor_service([[1.0, 1.0], [0.0, 0.0]])

    with caplog.at_level("WARNING", logger=service_module.LOGGER.name):
        await service.fit_preprocessors("train")

    assert "No behavior sub-label annotations" not in caplog.text


@pytest.mark.asyncio
async def test_training_runs_all_epochs_and_restores_best_state(monkeypatch):
    class CountingModel(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(0.0))

        def forward(self, features):
            return self.weight.expand(features.shape[0], features.shape[1], 1)

    optimizers = []

    class CountingOptimizer:
        def __init__(self, parameters, **kwargs):
            self.parameter = next(iter(parameters))
            self.step_count = 0
            optimizers.append(self)

        def zero_grad(self):
            self.parameter.grad = None

        def step(self):
            with torch.no_grad():
                self.parameter.add_(1.0)
            self.step_count += 1

    class NoOpScheduler:
        def __init__(self, *args, **kwargs):
            pass

        def step(self, loss):
            pass

    batch = (
        torch.zeros(1, 2, 1),
        torch.zeros(1, 2, 1),
        torch.ones(1, 2, 1),
    )
    validation_losses = [1.0, 2.0, 3.0, 4.0, 5.0]

    service = SegmentClassifierService.__new__(SegmentClassifierService)
    service.device = torch.device("cpu")
    service.scaler = SimpleNamespace(mean_=np.zeros(1))
    service.label_ids = ["MSP"]
    service.hidden_dim = 1
    service.dilations = (1,)
    service.dropout = 0.0
    service.pos_weight = None
    service.threshold = 0.5
    service.model = None
    service._configure_training_backend = lambda: None
    service.prepare_training_data = AsyncMock()
    service.fit_preprocessors = AsyncMock()
    service._dataset = lambda cache_key: cache_key
    service._save_artifacts = lambda: None
    service.serialize_artifacts = lambda: {}

    def controlled_loss(logits, targets, mask, pos_weight):
        if service.model.training:
            return logits.mean() * 0 + 1.0
        return torch.tensor(validation_losses.pop(0))

    service._masked_loss = controlled_loss
    monkeypatch.setattr(service_module, "DataLoader", lambda *args, **kwargs: [batch])
    monkeypatch.setattr(service_module, "TemporalDetectionModel", CountingModel)
    monkeypatch.setattr(torch.optim, "Adam", CountingOptimizer)
    monkeypatch.setattr(torch.optim.lr_scheduler, "ReduceLROnPlateau", NoOpScheduler)

    backend_client = ModuleType("app.integrations.backend.client")
    backend_client.backend_service = SimpleNamespace(save_ai_model=AsyncMock())
    monkeypatch.setitem(sys.modules, "app.integrations.backend.client", backend_client)

    await service.train_model(epochs=5, annotation_cache_key="annotations")

    assert optimizers[0].step_count == 5
    assert validation_losses == []
    assert service.model.weight.item() == 1.0


def test_validation_metrics_only_count_unmasked_labels():
    logits = torch.tensor([[[10.0, 10.0], [-10.0, 10.0]]])
    targets = torch.tensor([[[1.0, 0.0], [1.0, 1.0]]])
    mask = torch.tensor([[[1.0, 0.0], [1.0, 1.0]]])

    counts = SegmentClassifierService._metric_counts(logits, targets, mask, 0.5)
    metrics = SegmentClassifierService._validation_metrics(*counts[:3])

    assert counts == (2, 0, 1, 3)
    assert metrics == (1.0, 2 / 3, 0.8)


def test_threshold_merge_uses_exact_adjacent_rows():
    runs = list(SegmentClassifierService._merge_score_runs(
        [0.2, 0.5, 0.8, 0.49, 0.7],
        0.5,
    ))

    assert runs == [(1, 3, 0.65), (4, 5, 0.7)]


def test_detection_allows_overlap_and_reruns_behavior_crop_for_children():
    service = SegmentClassifierService.__new__(SegmentClassifierService)
    service.threshold = 0.5
    service.behavior_label_ids = ["MSP", "EA"]
    service.child_parent = {"MSP1": "MSP"}
    call_lengths = []

    def score_sequence(dataframe):
        call_lengths.append(len(dataframe))
        if len(call_lengths) == 1:
            return pd.DataFrame({
                "MSP": [0.1, 0.8, 0.9, 0.9, 0.7, 0.1, 0.1, 0.1],
                "EA": [0.1, 0.1, 0.1, 0.8, 0.9, 0.8, 0.7, 0.1],
                "MSP1": [0.0] * 8,
            })
        return pd.DataFrame({
            "MSP": [0.9] * len(dataframe),
            "EA": [0.1] * len(dataframe),
            "MSP1": [0.1, 0.7, 0.8, 0.1],
        })

    service.score_sequence = score_sequence
    detections = service.detect_segments(pd.DataFrame(_rows(0, 8)))

    assert call_lengths == [8, 4]
    assert [(item.label, item.start_index, item.end_index) for item in detections] == [
        ("MSP", 1, 5),
        ("EA", 3, 7),
    ]
    child = detections[0].subsegments[0]
    assert (child.label, child.start_index, child.end_index) == ("MSP1", 2, 4)
    assert detections[0].to_dict()["subsegments"][0]["label"] == "MSP1"


def test_old_artifact_format_is_not_ready(tmp_path):
    service = SegmentClassifierService.__new__(SegmentClassifierService)
    service.models_directory = tmp_path
    service.model_path = tmp_path / "segment_classifier.pth"
    service.scaler_path = tmp_path / "segment_scaler.joblib"
    service.config_path = tmp_path / "segment_config.json"
    service.model_path.touch()
    service.scaler_path.touch()
    service.config_path.write_text(json.dumps({"format": "segment_classifier/v2"}))

    assert service.has_local_artifacts() is False


def test_temporal_artifacts_round_trip_without_legacy_files(tmp_path):
    source_dir = tmp_path / "source"
    target_dir = tmp_path / "target"
    source_dir.mkdir()
    target_dir.mkdir()

    source = SegmentClassifierService.__new__(SegmentClassifierService)
    source.models_directory = source_dir
    source.model_path = source_dir / "segment_classifier.pth"
    source.scaler_path = source_dir / "segment_scaler.joblib"
    source.config_path = source_dir / "segment_config.json"
    source.model_path.write_bytes(b"weights")
    source.scaler_path.write_bytes(b"scaler")
    source.config_path.write_text(json.dumps({"format": MODEL_FORMAT}))

    payload = source.serialize_artifacts()
    assert set(payload["files"]) == set(source._ARTIFACT_FILES)

    target = SegmentClassifierService.__new__(SegmentClassifierService)
    target.models_directory = target_dir
    target.model_path = target_dir / "segment_classifier.pth"
    target.scaler_path = target_dir / "segment_scaler.joblib"
    target.config_path = target_dir / "segment_config.json"
    target.deserialize_artifacts(payload)

    assert target.model_path.read_bytes() == b"weights"
    assert target.scaler_path.read_bytes() == b"scaler"
    assert json.loads(target.config_path.read_text())["format"] == MODEL_FORMAT
