import numpy as np

from training.storage.datasets.segment_dataset import build_temporal_sequences


def _rows(start, end):
    return [
        {"speed": float(index), "brake": float(index % 2)}
        for index in range(start, end)
    ]


def test_temporal_targets_include_all_labels_on_parent_and_child_ranges():
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
        label_ids=["MSP", "ST1", "silverstone", "EA", "ST2", "MSP1"],
    )

    assert len(sequences) == 1
    sequence = sequences[0]
    assert sequence.start_index == 10
    assert sequence.features.shape == (8, 4)
    np.testing.assert_array_equal(sequence.targets[:, 0], [1, 1, 1, 1, 0, 0, 0, 0])
    np.testing.assert_array_equal(sequence.targets[:, 1], [1, 1, 1, 1, 0, 0, 0, 0])
    np.testing.assert_array_equal(sequence.targets[:, 2], [1, 1, 1, 1, 0, 0, 0, 0])
    np.testing.assert_array_equal(sequence.targets[:, 3], [0, 0, 0, 0, 1, 1, 1, 1])
    np.testing.assert_array_equal(sequence.targets[:, 4], [0, 0, 0, 0, 1, 1, 1, 1])
    np.testing.assert_array_equal(sequence.targets[:, 5], [0, 1, 1, 0, 0, 0, 0, 0])
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
    )

    assert [(sequence.start_index, len(sequence.features)) for sequence in sequences] == [
        (0, 2),
        (5, 2),
    ]


