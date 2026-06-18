import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ui"))

from segment_tabs.batch_lap import _rightmost_annotation_end


def test_rightmost_annotation_end_returns_largest_valid_end():
    annotations = [
        {"start_index": 10, "end_index": 20, "labels": ["ST1"]},
        {"start_index": 25, "end_index": 35, "labels": ["ST2"]},
        {"start_index": 40, "end_index": 40, "labels": ["ST3"]},
    ]

    assert _rightmost_annotation_end(annotations) == 35


def test_rightmost_annotation_end_returns_none_without_valid_annotations():
    assert _rightmost_annotation_end([]) is None
    assert _rightmost_annotation_end([
        {"start_index": 10, "end_index": 10, "labels": ["ST1"]},
    ]) is None
