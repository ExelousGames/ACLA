from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest
import yaml
from PIL import Image

from training.image_segmentation import read_labels
from training.image_segmentation.__main__ import main
from training.image_segmentation.dataset import prepare_dataset
from training.image_segmentation.trainer import train_model


def _annotations(tmp_path, shapes):
    for split in ("train", "val"):
        folder = tmp_path / split
        folder.mkdir()
        Image.new("RGB", (100, 100)).save(folder / "frame.png")
        (folder / "frame.json").write_text(json.dumps({
            "imagePath": "frame.png", "imageWidth": 100, "imageHeight": 100,
            "shapes": shapes,
        }))


def _mask(row):
    coordinates = np.array([float(value) for value in row.split()[1:]]).reshape(-1, 2)
    assert len(coordinates) >= 3
    assert np.isfinite(coordinates).all()
    assert ((coordinates >= 0) & (coordinates <= 1)).all()
    mask = np.zeros((100, 100), dtype=np.uint8)
    cv2.fillPoly(mask, [np.rint(coordinates * 100).astype(np.int32)], color=1)
    return mask


@pytest.fixture
def boundary_labels(tmp_path):
    path = tmp_path / "labels.txt"
    path.write_text("track\ncurb\ngrass\ncar\nother\nfence\ncar pack\nsand\nleft_boundary\nright_boundary\n")
    return path


def test_default_labels_contain_only_regions():
    assert read_labels() == [
        "track", "curb", "grass", "car", "other", "fence", "car pack", "sand",
        "Outfield asphalt road",
    ]


def test_prepare_keeps_open_boundaries_separate_from_region_polygons(tmp_path, boundary_labels):
    _annotations(tmp_path, [
        {"label": "track", "shape_type": "polygon", "points": [[0, 0], [100, 0], [50, 100]]},
        {"label": "left_boundary", "shape_type": "linestrip", "points": [[10, 90], [10, 10], [60, 10]]},
        {"label": "right_boundary", "shape_type": "line", "points": [[100, 0], [100, 100]]},
    ])
    original = (tmp_path / "train/frame.json").read_bytes()

    assert main([
        "prepare", "--train", str(tmp_path / "train"), "--val", str(tmp_path / "val"),
        "--output", str(tmp_path / "out"), "--polyline-width", "6",
        "--labels", str(boundary_labels),
    ]) == 0

    rows = (tmp_path / "out/labels/train/frame.txt").read_text().splitlines()
    assert [int(row.split()[0]) for row in rows] == [0, 8, 9]
    assert [float(value) for value in rows[0].split()] == [0, 0, 0, 1, 0, 0.5, 1]
    left, right = map(_mask, rows[1:])
    assert left[50, 10] == left[10, 40] == 1
    assert left[40, 20] == left[50, 35] == 0  # No filled interior or closing edge.
    assert left[50, 12] == 1 and left[50, 16] == 0
    assert right[50, 99] == 1 and right[50, 90] == 0  # Stroke clips at image edge.
    assert (tmp_path / "train/frame.json").read_bytes() == original
    assert yaml.safe_load((tmp_path / "out/data.yaml").read_text())["names"] == read_labels(boundary_labels)


@pytest.mark.parametrize("points", [
    [[10, 10], [90, 10]],  # Horizontal two-point polyline.
    [[10, 10], [10, 10], [50, 50], [90, 90]],  # Duplicate point and collinear segments.
    [[90.4, 90.2], [50.5, 40.1], [10.2, 10.8]],  # Fractional points in reverse order.
])
def test_valid_polylines_export_nonempty_masks(tmp_path, points, boundary_labels):
    _annotations(tmp_path, [{"label": "left_boundary", "shape_type": "linestrip", "points": points}])
    data = prepare_dataset(
        tmp_path / "train", tmp_path / "val", tmp_path / "out", labels_file=boundary_labels,
    )
    mask = _mask((data.parent / "labels/train/frame.txt").read_text())
    assert mask.sum() > 0


@pytest.mark.parametrize(("points", "message"), [
    ([[10, 10]], "at least two"),
    ([[10, 10], [10, 10]], "distinct"),
    ([[10, 10], [20, 20, 30]], "\\(x, y\\)"),
    ([[10, 10], [float("inf"), 20]], "finite"),
    ([[10, 10], [101, 20]], "inside the image"),
])
def test_invalid_polylines_fail_before_writing_output(tmp_path, points, message):
    _annotations(tmp_path, [{"label": "track", "shape_type": "linestrip", "points": points}])
    with pytest.raises(ValueError, match=message):
        prepare_dataset(tmp_path / "train", tmp_path / "val", tmp_path / "out")
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("width", [0, 1, -1, 2.5, float("nan")])
def test_invalid_polyline_width_fails_before_writing_output(tmp_path, width):
    _annotations(tmp_path, [{"label": "track", "shape_type": "linestrip", "points": [[10, 10], [90, 90]]}])
    with pytest.raises(ValueError, match="Polyline width"):
        prepare_dataset(tmp_path / "train", tmp_path / "val", tmp_path / "out", polyline_width=width)
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("points", [
    [[10, 10], [90, 90], [10, 90], [90, 10]],  # Self-intersection.
    [[10, 10], [90, 10], [90, 90], [10, 90], [10, 10]],  # Closed loop.
    [[10, 10], [20, 10], [20, 20], [10, 20], [10, 17]],  # One-pixel gap at width 8.
    [[10, 20], [40, 20], [40, 80], [10, 80], [10, 20], [60, 20],
     [90, 20], [90, 80], [60, 80], [60, 20]],  # Side-by-side holes: bridges can meet a joined hole.
    [[50, 50], [10, 10], [90, 10], [50, 50], [10, 90], [90, 90], [50, 50], [10, 10]],  # Retraced edge.
    [[0, 0], [100, 0], [100, 100], [0, 100], [0, 0]],  # Loop clipped at image edges.
])
@pytest.mark.parametrize("stroke_width", [2, 8, 14])
@pytest.mark.parametrize("reverse", [False, True])
def test_self_touching_boundaries_preserve_stroke_masks(tmp_path, points, stroke_width, reverse, boundary_labels):
    if reverse:
        points = points[::-1]
    _annotations(tmp_path, [{"label": "left_boundary", "shape_type": "linestrip", "points": points}])
    original = (tmp_path / "train/frame.json").read_bytes()
    expected = np.zeros((100, 100), dtype=np.uint8)
    cv2.polylines(expected, [np.array(points, dtype=np.int32)], isClosed=False, color=1, thickness=stroke_width)

    data = prepare_dataset(
        tmp_path / "train", tmp_path / "val", tmp_path / "out", polyline_width=stroke_width,
        labels_file=boundary_labels,
    )

    for split in ("train", "val"):
        rows = (data.parent / f"labels/{split}/frame.txt").read_text().splitlines()
        assert len(rows) == 1
        assert rows[0].split()[0] == "8"
        np.testing.assert_array_equal(_mask(rows[0]), expected)
    assert (tmp_path / "train/frame.json").read_bytes() == original


def test_line_with_more_than_two_points_is_rejected(tmp_path, boundary_labels):
    _annotations(tmp_path, [{
        "label": "left_boundary", "shape_type": "line", "points": [[10, 10], [50, 50], [90, 90]],
    }])
    with pytest.raises(ValueError, match="exactly two"):
        prepare_dataset(
            tmp_path / "train", tmp_path / "val", tmp_path / "out", labels_file=boundary_labels,
        )
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("names", [["track", "left_boundary"], {0: "right_boundary", 1: "track"}])
def test_boundary_training_preserves_side_labels(tmp_path, monkeypatch, names):
    import sys

    data = tmp_path / "data.yaml"
    data.write_text(yaml.safe_dump({"names": names}))
    model = MagicMock(task="segment")
    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=lambda *a, **kw: model))

    train_model(data, upload=False)

    options = model.train.call_args.kwargs
    assert options["fliplr"] == 0.0
    assert options["flipud"] == 0.0
    assert options["copy_paste"] == 0.0
    assert options["mask_ratio"] == 1
    assert options["overlap_mask"] is False
