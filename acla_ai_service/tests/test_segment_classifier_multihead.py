import pandas as pd
import torch
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler

from app.ml.segment_classifier.label_heads import build_label_head_specs
from app.ml.segment_classifier.model import MultiHeadCNN1DModel
from app.storage.datasets.segment_dataset import MultiHeadStreamingSegmentDataset, compute_derived_features
from app.shared.labels import BEHAVIOR_LABELS, LABEL_CATEGORIES, TRACK_LABELS


class _Store:
    def __init__(self, chunks):
        self._chunks = chunks

    def get_cached_data_chunks(self, _cache_key):
        return self._chunks


def test_label_head_specs_split_label_types():
    specs = {spec.name: spec for spec in build_label_head_specs()}

    assert specs["behavior_main"].label_ids == tuple(BEHAVIOR_LABELS)
    assert specs["track_main"].label_ids == tuple(TRACK_LABELS)
    assert specs["segment_type"].label_ids == tuple(LABEL_CATEGORIES["Segment Type"])
    assert specs["sub:MSP"].label_ids == tuple(LABEL_CATEGORIES["MSP"])
    assert specs["sub:silverstone"].label_ids == tuple(LABEL_CATEGORIES["silverstone"])


def test_multihead_dataset_targets_and_subhead_masks():
    specs = [
        spec for spec in build_label_head_specs()
        if spec.name in {"behavior_main", "track_main", "segment_type", "sub:MSP", "sub:MSR"}
    ]
    head_mlbs = {}
    for spec in specs:
        mlb = MultiLabelBinarizer()
        mlb.fit([list(spec.label_ids)])
        head_mlbs[spec.name] = mlb

    expected_features = ["speed", "brake"]
    telemetry_data = [
        {"speed": 10, "brake": 0.1},
        {"speed": 12, "brake": 0.2},
    ]
    df_values = compute_derived_features(pd.DataFrame(telemetry_data).reindex(columns=expected_features)).values
    scaler = StandardScaler().fit(df_values)
    store = _Store([[
        {
            "labels": ["MSP", "MSP1", "silverstone", "ST1"],
            "telemetry_data": telemetry_data,
        }
    ]])
    dataset = MultiHeadStreamingSegmentDataset(
        store,
        "train",
        head_mlbs,
        specs,
        scaler,
        max_length=4,
        expected_features=expected_features,
    )

    _, targets, masks = next(iter(dataset))

    behavior_idx = list(head_mlbs["behavior_main"].classes_).index("MSP")
    sub_idx = list(head_mlbs["sub:MSP"].classes_).index("MSP1")
    track_idx = list(head_mlbs["track_main"].classes_).index("silverstone")
    segment_type_idx = list(head_mlbs["segment_type"].classes_).index("ST1")

    assert targets["behavior_main"][0, behavior_idx].item() == 1
    assert targets["sub:MSP"][0, sub_idx].item() == 1
    assert targets["track_main"][0, track_idx].item() == 1
    assert targets["segment_type"][0, segment_type_idx].item() == 1
    assert masks["sub:MSP"][:2].sum().item() == 2
    assert masks["sub:MSR"].sum().item() == 0


def test_multihead_model_returns_one_tensor_per_head():
    model = MultiHeadCNN1DModel(
        input_dim=4,
        hidden_dim=8,
        head_output_dims={"behavior_main": 3, "sub:MSP": 5},
        num_layers=1,
    )

    outputs, _ = model(torch.zeros(2, 7, 4))

    assert set(outputs.keys()) == {"behavior_main", "sub:MSP"}
    assert outputs["behavior_main"].shape == (2, 7, 3)
    assert outputs["sub:MSP"].shape == (2, 7, 5)
