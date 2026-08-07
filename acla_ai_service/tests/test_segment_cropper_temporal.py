from __future__ import annotations

import base64
import json

import numpy as np
import pytest
import torch
from sklearn.preprocessing import StandardScaler

from app.ml.segment_cropper.data import (
    CropperSequence,
    build_boundary_targets,
    pad_cropper_batch,
    parse_session_chunk,
)
from app.ml.segment_cropper.decoding import (
    CropCandidate,
    CropperThresholds,
    ValidationProbabilities,
    calibrate_thresholds,
    decode_probabilities,
    evaluate_thresholds,
    form_candidates,
    select_non_overlapping,
)
from app.ml.segment_cropper.model import BoundaryTCN
from app.ml.segment_cropper.service import MODEL_FORMAT, SegmentCropperService
from app.ml.segment_cropper.trainer import SegmentCropperTrainer
from app.ml import model_hub
from app.pipelines.manifest import node_kinds


def _candidate(start, end, confidence):
    return CropCandidate(start, end, confidence, confidence, confidence, confidence)


def test_complete_session_contract_builds_exclusive_union_targets_and_background():
    session = parse_session_chunk(
        {
            "telemetry_data": [{"speed": index} for index in range(7)],
            "annotations": [
                {"start_index": 1, "end_index": 4},
                {"start_index": 2, "end_index": 6},
                {"start_index": 0, "end_index": 1, "parent_id": "parent"},
                {"start_index": -1, "end_index": 2},
                {"start_index": 6, "end_index": 6},
                {"start_index": 6, "end_index": 8},
            ],
        },
        "session-1",
    )

    assert session.annotations == [(1, 4), (2, 6)]
    targets = build_boundary_targets(7, session.annotations)
    assert targets[:, 0].tolist() == [0, 1, 1, 0, 0, 0, 0]
    assert targets[:, 1].tolist() == [0, 0, 0, 1, 0, 1, 0]
    assert targets[:, 2].tolist() == [0, 1, 1, 1, 1, 1, 0]


@pytest.mark.parametrize(
    "payload",
    [
        [],
        {"telemetry_data": [], "annotations": []},
        {"telemetry_data": [{}]},
        {"telemetry_data": "rows", "annotations": []},
    ],
)
def test_complete_session_contract_rejects_non_session_chunks(payload):
    with pytest.raises(ValueError):
        parse_session_chunk(payload, "session")


def test_trainer_reads_only_configured_dataset_and_splits_whole_sessions():
    class Store:
        def __init__(self):
            self.keys = []

        def get_cached_data_chunks(self, cache_key, include_ids=False):
            self.keys.append((cache_key, include_ids))
            for index in range(10):
                yield {
                    "telemetry_data": [{"speed": index}],
                    "annotations": [{"start_index": 0, "end_index": 1}],
                }, f"session-{index}"

    store = Store()
    trainer = SegmentCropperTrainer(
        cropper_service=SegmentCropperService("/tmp/unused-segment-cropper-test"),
        store=store,
    )
    sessions = trainer.load_sessions("complete_sessions")
    training, validation = trainer.split_sessions(sessions)

    assert store.keys == [("complete_sessions", True)]
    assert len(training) == 9
    assert len(validation) == 1
    assert {item.session_id for item in training}.isdisjoint(
        item.session_id for item in validation
    )
    assert trainer.split_sessions(sessions) == (training, validation)


def test_preprocessing_and_class_weights_fit_training_sequences_only():
    trainer = SegmentCropperTrainer.__new__(SegmentCropperTrainer)
    trainer.device = torch.device("cpu")
    trainer.scaler = None
    trainer.class_weights = None
    training = [CropperSequence(
        session_id="train",
        features=np.asarray([[0.0], [2.0]], dtype=np.float32),
        targets=np.asarray([[1, 0, 1], [0, 1, 1]], dtype=np.float32),
    )]
    trainer.fit_preprocessors(training)
    assert trainer.scaler.mean_.tolist() == [1.0]


def test_tcn_output_and_padding_masks_preserve_row_alignment():
    model = BoundaryTCN(input_dim=4)
    outputs = model(torch.randn(2, 11, 4))
    assert [tuple(head.shape) for head in outputs] == [(2, 11), (2, 11), (2, 11)]

    features, targets, mask = pad_cropper_batch([
        (torch.ones(2, 4), torch.ones(2, 3)),
        (torch.ones(4, 4), torch.ones(4, 3)),
    ])
    assert tuple(features.shape) == (2, 4, 4)
    assert tuple(targets.shape) == (2, 4, 3)
    assert mask.tolist() == [[1, 1, 0, 0], [1, 1, 1, 1]]


def test_masked_losses_ignore_padded_rows():
    logits = torch.tensor([[0.0, 100.0]])
    targets = torch.tensor([[1.0, 0.0]])
    mask = torch.tensor([[1.0, 0.0]])
    weight = torch.tensor(2.0)
    focal = SegmentCropperTrainer.masked_focal_binary_loss(
        logits, targets, mask, weight,
    )
    weighted = SegmentCropperTrainer.masked_weighted_binary_loss(
        logits, targets, mask, weight,
    )
    assert torch.isfinite(focal)
    assert torch.isfinite(weighted)
    assert weighted.item() == pytest.approx(2 * np.log(2))


def test_candidate_score_uses_start_end_and_mean_inside_confidence():
    candidates = form_candidates(
        [0.9, 0.1, 0.2],
        [0.1, 0.2, 0.8],
        [0.6, 0.9, 0.3],
    )
    candidate = next(item for item in candidates if (item.start_index, item.end_index) == (0, 3))
    assert candidate.inside_probability == pytest.approx(0.6)
    assert candidate.confidence == pytest.approx((0.9 + 0.8 + 0.6) / 3)


def test_weighted_scheduling_allows_adjacency_and_forbids_positive_overlap():
    selected = select_non_overlapping([
        _candidate(0, 3, 0.6),
        _candidate(3, 5, 0.6),
        _candidate(1, 5, 1.0),
    ])
    assert [(item.start_index, item.end_index) for item in selected] == [(0, 3), (3, 5)]
    assert all(left.end_index <= right.start_index for left, right in zip(selected, selected[1:]))


def test_equal_schedule_scores_choose_lexicographically_earlier_ranges():
    selected = select_non_overlapping([
        _candidate(0, 2, 0.5),
        _candidate(2, 4, 0.5),
        _candidate(1, 4, 1.0),
    ])
    assert [(item.start_index, item.end_index) for item in selected] == [(0, 2), (2, 4)]


def test_validation_metrics_use_post_scheduling_one_to_one_matches():
    validation = [ValidationProbabilities(
        start=np.asarray([0.9, 0.8, 0.1]),
        end=np.asarray([0.1, 0.9, 0.8]),
        inside=np.asarray([0.9, 0.9, 0.9]),
        annotations=((0, 2),),
    )]
    metrics = evaluate_thresholds(validation, CropperThresholds(0.5, 0.5, 0.5))
    assert metrics["proposal_count"] == 1
    assert metrics["true_positives"] == 1
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0


def test_calibration_prioritizes_precision_at_target_recall_and_falls_back(monkeypatch):
    validation = [ValidationProbabilities(
        start=np.asarray([1.0]),
        end=np.asarray([1.0]),
        inside=np.asarray([1.0]),
        annotations=((0, 1),),
    )]

    def target_metrics(_validation, thresholds, minimum_iou=0.5):
        if thresholds.boundary == 0.5:
            return {"precision": 0.9, "recall": 0.95}
        return {"precision": 0.7, "recall": 0.99}

    monkeypatch.setattr(
        "app.ml.segment_cropper.decoding.evaluate_thresholds",
        target_metrics,
    )
    thresholds, metrics = calibrate_thresholds(validation, [0.1, 0.5])
    assert thresholds == CropperThresholds(0.5, 0.5, 0.5)
    assert metrics["target_recall_attained"] is True

    def fallback_metrics(_validation, thresholds, minimum_iou=0.5):
        if thresholds.boundary == 0.1:
            return {"precision": 0.4, "recall": 0.9}
        return {"precision": 0.99, "recall": 0.8}

    monkeypatch.setattr(
        "app.ml.segment_cropper.decoding.evaluate_thresholds",
        fallback_metrics,
    )
    thresholds, metrics = calibrate_thresholds(validation, [0.1, 0.5])
    assert thresholds.boundary == 0.1
    assert metrics["recall"] == 0.9
    assert metrics["target_recall_attained"] is False


def test_artifact_round_trip_and_feature_contract_validation(tmp_path):
    service = SegmentCropperService(tmp_path / "source")
    feature_count = len(service.derived_feature_names)
    service.model = BoundaryTCN(feature_count)
    service.scaler = StandardScaler().fit(np.zeros((2, feature_count)))
    service.class_weights = {"start": 2.0, "end": 2.0, "inside": 1.5}
    service.validation_metrics = {"precision": 1.0, "recall": 0.95}
    service.save_artifacts()
    payload = service.serialize_artifacts()
    assert payload["format"] == MODEL_FORMAT

    restored = SegmentCropperService(tmp_path / "restored")
    restored.deserialize_artifacts(payload)
    assert restored.load_model()
    assert restored.is_ready()
    assert restored.validation_metrics["recall"] == 0.95

    contract_name = "segment_cropper_contract.json"
    contract = json.loads(base64.b64decode(payload["files"][contract_name]))
    contract["raw_features"] = list(reversed(contract["raw_features"]))
    payload["files"][contract_name] = base64.b64encode(
        json.dumps(contract).encode("utf-8")
    ).decode("ascii")
    with pytest.raises(ValueError, match="feature contract"):
        SegmentCropperService(tmp_path / "invalid").deserialize_artifacts(payload)


def test_model_hub_registers_and_hydrates_independent_backend_type(tmp_path, monkeypatch):
    source = SegmentCropperService(tmp_path / "source")
    feature_count = len(source.derived_feature_names)
    source.model = BoundaryTCN(feature_count)
    source.scaler = StandardScaler().fit(np.zeros((2, feature_count)))
    source.save_artifacts()
    restored = SegmentCropperService(tmp_path / "restored")
    monkeypatch.setattr(model_hub, "get_segment_cropper", lambda: restored)

    spec = next(item for item in model_hub._MODEL_SPECS if item.name == "segment_cropper")
    assert spec.backend_model_type == "segment_cropper"
    assert spec.hydrate(source.serialize_artifacts())
    assert spec.is_ready()


def test_pipeline_registry_exposes_dedicated_cropper_training_component():
    spec = node_kinds.get("segment_cropper")
    assert spec.category == "training"
    assert spec.ui_route == "segment_cropper"
