import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ui"))

from segment_tabs.batch_lap import _find_overlapping_annotation


def test_find_overlapping_annotation_returns_none_for_adjacent_ranges():
    annotations = [{"start_index": 10, "end_index": 20, "labels": ["ST1"]}]

    assert _find_overlapping_annotation(0, 10, annotations) is None
    assert _find_overlapping_annotation(20, 30, annotations) is None


def test_find_overlapping_annotation_returns_existing_overlap():
    annotation = {"start_index": 10, "end_index": 20, "labels": ["ST1"]}

    assert _find_overlapping_annotation(19, 30, [annotation]) is annotation
