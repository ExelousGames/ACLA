import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.preprocessing import StandardScaler

from app.ml.segment_classifier.model import TemporalDetectionModel
from app.ml.segment_classifier.service import SegmentClassifierService
from app.shared.segment_classifier_features import SEGMENT_CLASSIFIER_FEATURES


def _rows(start, end):
    return [
        {"speed": float(index), "brake": float(index % 2)}
        for index in range(start, end)
    ]


def _configured_service(models_directory):
    service = SegmentClassifierService(str(models_directory))
    service.configure_labels(["MSP", "MSP1", "ST1", "silverstone"])
    service.scaler = StandardScaler().fit(np.array([[0.0, 1.0], [2.0, 3.0]]))
    service.model = TemporalDetectionModel(
        input_dim=2,
        output_dim=len(service.label_ids),
        hidden_dim=service.hidden_dim,
        dilations=service.dilations,
        dropout=service.dropout,
    ).to(service.device)
    service.label_weights = {
        label_id: float(index + 1)
        for index, label_id in enumerate(service.label_ids)
    }
    return service


class _FixedLogitModel(torch.nn.Module):
    def __init__(self, logits):
        super().__init__()
        self.register_buffer("logits", torch.tensor(logits, dtype=torch.float32))

    def forward(self, features):
        return self.logits.view(1, 1, -1).expand(
            features.shape[0],
            features.shape[1],
            -1,
        )


class _IdentityScaler:
    @staticmethod
    def transform(values):
        return values


def _fixed_logit_service(label_ids, label_weights, logits):
    service = SegmentClassifierService.__new__(SegmentClassifierService)
    service.device = torch.device("cpu")
    service.model = _FixedLogitModel(logits)
    service.scaler = _IdentityScaler()
    service.label_ids = list(label_ids)
    service.label_weights = dict(label_weights)
    service._prepare_numeric_features = lambda dataframe: dataframe[["speed"]]
    return service


@pytest.mark.parametrize("missing_feature", SEGMENT_CLASSIFIER_FEATURES)
def test_classifier_zero_fills_missing_raw_or_expert_features(tmp_path, missing_feature):
    service = SegmentClassifierService(str(tmp_path))
    dataframe = pd.DataFrame([{
        feature: 1.0
        for feature in SEGMENT_CLASSIFIER_FEATURES
        if feature != missing_feature
    }])

    numeric = service._prepare_numeric_features(dataframe)

    assert numeric[missing_feature].tolist() == [0.0]
    assert numeric[f"{missing_feature}_diff"].tolist() == [0.0]
    assert (numeric[dataframe.columns] == 1.0).all().all()


def test_classifier_selects_feature_order_and_derives_changes(tmp_path):
    service = SegmentClassifierService(str(tmp_path))
    dataframe = pd.DataFrame([
        {**{feature: 1.0 for feature in reversed(SEGMENT_CLASSIFIER_FEATURES)}, "extra": 99},
        {**{feature: 3.0 for feature in reversed(SEGMENT_CLASSIFIER_FEATURES)}, "extra": 100},
    ])

    numeric = service._prepare_numeric_features(dataframe)

    assert list(numeric.columns) == [
        *SEGMENT_CLASSIFIER_FEATURES,
        *(f"{feature}_diff" for feature in SEGMENT_CLASSIFIER_FEATURES),
    ]
    assert numeric["Physics_speed_kmh"].tolist() == [1.0, 3.0]
    assert numeric["Physics_speed_kmh_diff"].tolist() == [0.0, 2.0]


def test_temporal_model_preserves_the_entire_sequence_length():
    model = TemporalDetectionModel(input_dim=6, output_dim=4, hidden_dim=8)

    assert model(torch.zeros(2, 7, 6)).shape == (2, 7, 4)
    assert model(torch.zeros(1, 257, 6)).shape == (1, 257, 4)


def test_score_sequence_corrects_weighted_logits_in_label_id_order():
    raw_logits = [0.0, float(np.log(20.0)), 0.0]
    service = _fixed_logit_service(
        label_ids=["neutral", "rare", "common"],
        label_weights={"common": 0.25, "neutral": 1.0, "rare": 20.0},
        logits=raw_logits,
    )

    scores = service.score_sequence(pd.DataFrame({"speed": [1.0, 2.0]}))
    raw_scores = torch.sigmoid(torch.tensor(raw_logits)).numpy()
    expected = torch.sigmoid(
        torch.tensor(raw_logits) - torch.log(torch.tensor([1.0, 20.0, 0.25]))
    ).numpy()

    assert list(scores.columns) == ["neutral", "rare", "common"]
    np.testing.assert_allclose(scores.iloc[0].to_numpy(), expected, rtol=1e-6)
    assert scores.loc[0, "neutral"] == pytest.approx(raw_scores[0])
    assert scores.loc[0, "rare"] < raw_scores[1]
    assert scores.loc[0, "common"] > raw_scores[2]


def test_threshold_merge_expands_boundaries_and_merges_only_overlapping_runs():
    runs = list(SegmentClassifierService._merge_score_runs(
        [0.2, 0.5, 0.8, 0.49, 0.7, 0.2, 0.1, 0.6, 0.8, 0.2],
        0.5,
    ))

    assert [(start, end) for start, end, _ in runs] == [(0, 6), (6, 10)]
    assert [score for _, _, score in runs] == pytest.approx([
        (0.5 + 0.8 + 0.7) / 3,
        0.7,
    ])


def test_threshold_merge_bounds_expansion_at_sequence_ends():
    runs = list(SegmentClassifierService._merge_score_runs(
        [0.7, 0.2, 0.1, 0.8],
        0.5,
    ))

    assert [(start, end) for start, end, _ in runs] == [(0, 2), (2, 4)]
    assert [score for _, _, score in runs] == pytest.approx([0.7, 0.8])


def test_threshold_merge_uses_the_active_custom_threshold_and_requires_a_core():
    runs = list(SegmentClassifierService._merge_score_runs(
        [0.7, 0.8, 0.79],
        0.8,
    ))
    assert [(start, end) for start, end, _ in runs] == [(0, 3)]
    assert [score for _, _, score in runs] == pytest.approx([0.8])
    assert list(SegmentClassifierService._merge_score_runs(
        [0.49, 0.1, 0.2],
        0.5,
    )) == []


def test_detection_uses_default_threshold_and_allows_override():
    service = SegmentClassifierService.__new__(SegmentClassifierService)
    service.threshold = 0.5
    service.label_ids = ["MSP"]
    service.behavior_label_ids = ["MSP"]
    service.child_parent = {}
    service.score_sequence = lambda dataframe: pd.DataFrame({"MSP": [0.1, 0.7, 0.1]})
    dataframe = pd.DataFrame(_rows(0, 3))

    assert len(service.detect_segments(dataframe)) == 1
    assert service.detect_segments(dataframe, threshold=0.8) == []


def test_detection_includes_every_trained_label_category():
    service = SegmentClassifierService.__new__(SegmentClassifierService)
    service.threshold = 0.5
    service.configure_labels(["MSP1", "ST1", "silverstone", "custom-label"])
    service.score_sequence = lambda dataframe: pd.DataFrame({
        "MSP1": [0.9, 0.1, 0.1],
        "ST1": [0.1, 0.7, 0.1],
        "silverstone": [0.8, 0.8, 0.1],
        "custom-label": [0.1, 0.1, 0.9],
    })

    detections = service.detect_segments(pd.DataFrame(_rows(0, 3)))

    assert [item.label for item in detections] == [
        "MSP1",
        "ST1",
        "silverstone",
        "custom-label",
    ]


def test_detection_allows_overlap_and_reruns_behavior_crop_for_children():
    service = SegmentClassifierService.__new__(SegmentClassifierService)
    service.threshold = 0.5
    service.label_ids = ["MSP", "EA", "MSP1"]
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
            "MSP1": [0.1, 0.1, 0.7, 0.8, 0.1, 0.1],
        })

    service.score_sequence = score_sequence
    detections = service.detect_segments(pd.DataFrame(_rows(0, 8)))

    assert call_lengths == [8, 6]
    assert [(item.label, item.start_index, item.end_index) for item in detections] == [
        ("MSP", 0, 6),
        ("EA", 2, 8),
    ]
    assert detections[0].score == pytest.approx(0.825)
    assert detections[0].telemetry_data == _rows(0, 6)
    assert detections[1].score == pytest.approx(0.8)
    assert detections[1].telemetry_data == _rows(2, 8)
    assert detections[1].subsegments == []

    child = detections[0].subsegments[0]
    assert (child.label, child.start_index, child.end_index) == ("MSP1", 1, 5)
    assert child.score == pytest.approx(0.75)
    assert child.telemetry_data == _rows(1, 5)
    assert detections[0].start_index <= child.start_index
    assert child.end_index <= detections[0].end_index
    assert detections[0].to_dict()["subsegments"][0]["label"] == "MSP1"


def test_detection_thresholds_corrected_parent_and_child_probabilities():
    service = _fixed_logit_service(
        label_ids=["MSP", "EA", "MSP1"],
        label_weights={"MSP1": 20.0, "EA": 20.0, "MSP": 20.0},
        logits=[float(np.log(20.0) + 1.0), 1.0, 1.0],
    )
    service.threshold = 0.5
    service.behavior_label_ids = ["MSP", "EA"]
    service.child_parent = {"MSP1": "MSP"}

    detections = service.detect_segments(pd.DataFrame({"speed": [1.0, 2.0, 3.0]}))

    assert [item.label for item in detections] == ["MSP"]
    assert detections[0].score == pytest.approx(float(torch.sigmoid(torch.tensor(1.0))))
    assert detections[0].subsegments == []


def test_model_round_trip_preserves_trained_label_configuration(tmp_path):
    source = _configured_service(tmp_path)
    source.model.eval()
    inputs = torch.randn(1, 5, 2, device=source.device)
    expected_output = source.model(inputs)
    dataframe = pd.DataFrame(_rows(0, 3))
    source._prepare_numeric_features = lambda frame: frame[["speed", "brake"]]
    expected_scores = source.score_sequence(dataframe)
    source._save_artifacts()
    checkpoint = torch.load(source.model_path, map_location="cpu")

    target = SegmentClassifierService(str(tmp_path))

    assert target.load_model() is True
    assert set(checkpoint) == {"model_state_dict", "label_weights"}
    assert checkpoint["label_weights"] == source.label_weights
    assert set(checkpoint["label_weights"]) == set(source.label_ids)
    assert target.feature_names == source.feature_names
    assert target.label_ids == source.label_ids
    assert target.label_weights == source.label_weights
    assert target.hidden_dim == source.hidden_dim
    assert target.dilations == source.dilations
    assert target.dropout == source.dropout
    assert target.threshold == source.threshold
    for name, parameter in source.model.state_dict().items():
        torch.testing.assert_close(target.model.state_dict()[name], parameter)
    torch.testing.assert_close(target.model(inputs), expected_output)
    target._prepare_numeric_features = lambda frame: frame[["speed", "brake"]]
    np.testing.assert_allclose(target.score_sequence(dataframe), expected_scores)
    assert not (tmp_path / "segment_config.json").exists()


def test_local_artifacts_do_not_depend_on_legacy_config(tmp_path):
    service = SegmentClassifierService.__new__(SegmentClassifierService)
    service.models_directory = tmp_path
    service.model_path = tmp_path / "segment_classifier.pth"
    service.scaler_path = tmp_path / "segment_scaler.joblib"
    service.model_path.touch()
    service.scaler_path.touch()
    (tmp_path / "segment_config.json").write_text(
        '{"format": "segment_classifier/v2"}'
    )

    assert service.has_local_artifacts() is True


def test_backend_artifact_round_trip_preserves_model_and_label_weights(tmp_path):
    source_dir = tmp_path / "source"
    target_dir = tmp_path / "target"
    source = _configured_service(source_dir)
    source.model.eval()
    dataframe = pd.DataFrame(_rows(0, 3))
    source._prepare_numeric_features = lambda frame: frame[["speed", "brake"]]
    expected_scores = source.score_sequence(dataframe)
    source._save_artifacts()

    payload = source.serialize_artifacts()
    assert payload["format"] == "segment_classifier/temporal-v2"
    assert set(payload["files"]) == set(source._ARTIFACT_FILES)

    target = SegmentClassifierService(str(target_dir))
    target.deserialize_artifacts(payload)

    assert target.load_model() is True
    assert target.label_weights == source.label_weights
    np.testing.assert_array_equal(target.scaler.mean_, source.scaler.mean_)
    for name, parameter in source.model.state_dict().items():
        torch.testing.assert_close(target.model.state_dict()[name], parameter)
    target._prepare_numeric_features = lambda frame: frame[["speed", "brake"]]
    np.testing.assert_allclose(target.score_sequence(dataframe), expected_scores)
    assert not (target_dir / "segment_config.json").exists()
