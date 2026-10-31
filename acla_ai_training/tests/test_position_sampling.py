"""Exact density thinning, CLI options, and preservation of temporal history."""

import json
import math
from pathlib import Path
import random

import numpy as np
from PIL import Image
import pytest
import torch
from torch.utils.data import Subset

from training.image_segmentation.__main__ import main
from training.image_segmentation import position_training
from training.image_segmentation.position_model import PositionModel
from training.image_segmentation.position_sampling import select_position_samples


@pytest.fixture(autouse=True)
def offline_resnet(monkeypatch):
    from training.image_segmentation import position_model_v2

    encoder = position_model_v2.resnet18
    monkeypatch.setattr(position_model_v2, "resnet18", lambda **kwargs: encoder(weights=None))


def record(index, offset=(0, 0, 0), orientation=(0, 0, 0), game="acc", map_name="Monza", normalized_position=0):
    return {
        "image": Path(f"frame_{index:06d}.png"), "game": game, "map_name": map_name,
        "target": np.asarray([offset], dtype=np.float32), "orientation": np.asarray(orientation),
        "normalized_position": normalized_position,
    }


def close(a, b, offset_limit, angle_limit, position_limit):
    if (a["game"], a["map_name"]) != (b["game"], b["map_name"]):
        return False
    if offset_limit is not None and math.dist(a["target"][0], b["target"][0]) > offset_limit:
        return False
    if position_limit is not None and abs(math.remainder(a["normalized_position"] - b["normalized_position"], 1)) > position_limit:
        return False
    return angle_limit is None or all(
        abs(math.remainder(float(x) - float(y), 2 * math.pi)) <= math.radians(angle_limit)
        for x, y in zip(a["orientation"], b["orientation"])
    )


@pytest.mark.parametrize("offset_limit,angle_limit,position_limit", [
    (0.4, None, None), (None, 20, None), (0.4, 20, None),
    (None, None, 0.1), (0.4, None, 0.1), (None, 20, 0.1), (0.4, 20, 0.1),
])
def test_indexed_selection_matches_brute_force(offset_limit, angle_limit, position_limit):
    rng = np.random.default_rng(7)
    rows = [record(i, rng.normal(size=3) / 3, rng.normal(size=3) / 3,
                   map_name="Monza" if i % 2 else "Spa", normalized_position=rng.random()) for i in range(300)]
    expected = []
    order_rng = random.Random(19)
    for map_name in ("Monza", "Spa"):
        order = [i for i, row in enumerate(rows) if row["map_name"] == map_name]
        order_rng.shuffle(order)
        for index in order:
            if not any(close(rows[index], rows[kept], offset_limit, angle_limit, position_limit) for kept in expected):
                expected.append(index)
    actual = select_position_samples(rows, min_offset_distance=offset_limit,
                                     min_orientation_distance_deg=angle_limit,
                                     min_normalized_position_distance=position_limit, seed=19)
    assert actual == sorted(expected)
    assert 1 < len(actual) < len(rows)
    # A rejected sample must have a retained neighbor; retained pairs are separated.
    assert all(any(close(row, rows[index], offset_limit, angle_limit, position_limit) for index in actual) for row in rows)
    assert all(not close(rows[a], rows[b], offset_limit, angle_limit, position_limit)
               for i, a in enumerate(actual) for b in actual[i + 1:])


@pytest.mark.parametrize("positions,limit,count", [
    ([0.999, 0.001, 0.5], 0.01, 2),
    ([0, 1], 0.01, 1),
    ([0, 0.125], 0.125, 1),
    ([0, 0.126], 0.125, 2),
    ([0, 0.5, 1], 0.5, 1),
])
def test_normalized_position_wraps_and_includes_threshold_boundary(positions, limit, count):
    rows = [record(index, normalized_position=position) for index, position in enumerate(positions)]
    assert len(select_position_samples(rows, min_normalized_position_distance=limit)) == count


def test_lap_position_preserves_distant_examples_with_identical_offsets_and_orientation():
    rows = [record(index, normalized_position=position) for index, position in enumerate((0.1, 0.105, 0.4))]
    options = {"min_offset_distance": 0.1, "min_orientation_distance_deg": 5}
    assert len(select_position_samples(rows, **options)) == 1
    selected = select_position_samples(rows, min_normalized_position_distance=0.01, **options)
    assert len(set(selected) & {0, 1}) == 1 and 2 in selected


@pytest.mark.parametrize("axis", range(3))
def test_wrapped_angles_and_offset_axes_jointly_determine_duplicates(axis):
    a, b = np.zeros(3), np.zeros(3)
    a[axis], b[axis] = math.radians(179), math.radians(-179)
    far_offset = np.zeros(3)
    far_offset[axis] = 0.3
    rows = [record(0, orientation=a), record(1, (0.01, 0, 0), b),
            record(2, far_offset, a), record(3)]
    selected = select_position_samples(rows, min_offset_distance=0.1, min_orientation_distance_deg=3)
    assert len(set(selected) & {0, 1}) == 1
    assert {2, 3}.issubset(selected)


def test_euclidean_offsets_and_exact_angles_reject_tree_bounding_box_false_positives():
    # Each XYZ coordinate is within 1, but the 3D distance is greater than 1.
    rows = [record(0), record(1, (0.8, 0.8, 0))]
    assert select_position_samples(rows, min_offset_distance=1) == [0, 1]
    # Sine/cosine coordinates fit the query box although the angle exceeds 30 deg.
    rows = [record(0, orientation=(math.radians(25), 0, 0)),
            record(1, orientation=(math.radians(65), 0, 0))]
    assert select_position_samples(rows, min_orientation_distance_deg=30) == [0, 1]


def test_groups_seed_disabled_mode_and_empty_input():
    rows = [record(i) for i in range(20)] + [record(20, game="other"), record(21, map_name="Spa")]
    assert select_position_samples(rows) == list(range(len(rows)))
    assert select_position_samples([], min_offset_distance=1) == []
    selected = select_position_samples(rows, min_offset_distance=1, seed=4)
    assert len(selected) == 3 and {20, 21}.issubset(selected)
    assert selected == select_position_samples(rows, min_offset_distance=1, seed=4)
    assert selected != select_position_samples(rows, min_offset_distance=1, seed=5)
    reversed_rows = rows[::-1]
    reverse_indices = select_position_samples(reversed_rows, min_offset_distance=1, seed=4)
    assert {rows[i]["image"] for i in selected} == {reversed_rows[i]["image"] for i in reverse_indices}


@pytest.mark.parametrize("name,value", [
    (name, value) for name in ("min_offset_distance", "min_orientation_distance_deg", "min_normalized_position_distance")
    for value in (0, -1, float("nan"), float("inf"))
] + [("min_orientation_distance_deg", 181), ("min_normalized_position_distance", 0.501)])
def test_invalid_thresholds_are_rejected(name, value):
    with pytest.raises(ValueError, match=name):
        select_position_samples([], **{name: value})


@pytest.mark.parametrize("arguments,offset,angle,position", [
    ([], None, None, None), (["--min-offset-distance", "0.1"], 0.1, None, None),
    (["--min-orientation-distance-deg", "5"], None, 5, None),
    (["--min-offset-distance", "0.1", "--min-orientation-distance-deg", "5"], 0.1, 5, None),
    (["--min-normalized-position-distance", "0.01"], None, None, 0.01),
    (["--min-offset-distance", "0.1", "--min-orientation-distance-deg", "5",
      "--min-normalized-position-distance", "0.01"], 0.1, 5, 0.01),
])
@pytest.mark.parametrize("command", ["train-position", "train-position-v2"])
def test_cli_forwards_density_options(arguments, offset, angle, position, monkeypatch, command):
    captured = {}
    monkeypatch.setattr(position_training, "train_position_model", lambda *args, **kwargs: captured.update(kwargs))
    assert main([command, "--manifest", "data", "--output", "run", "--no-upload", *arguments]) == 0
    assert captured["min_offset_distance"] == offset
    assert captured["min_orientation_distance_deg"] == angle
    assert captured["min_normalized_position_distance"] == position


@pytest.mark.parametrize("sequence_length", [1, 4])
@pytest.mark.parametrize("options,count", [
    ({"min_offset_distance": 0.1, "min_orientation_distance_deg": 5}, 1),
    ({"min_normalized_position_distance": 0.11}, 2),
    ({"min_offset_distance": 0.1, "min_orientation_distance_deg": 5, "min_normalized_position_distance": 0.11}, 2),
])
def test_training_retains_history_validation_and_correct_metric_denominators(tmp_path, monkeypatch, sequence_length, options, count):
    sessions = {}
    for session in ("a", "b"):
        folder = tmp_path / session
        folder.mkdir()
        rows = []
        for index in range(5):
            row = record(index)
            row.update(session_id=session, image=folder / row["image"], normalized_position=index / 10,
                       references=np.asarray([[index, 0, 0]], dtype=np.float32))
            Image.new("RGB", (32, 32)).save(row["image"])
            rows.append(row)
        sessions[session] = rows
    monkeypatch.setattr(position_training, "load_position_manifest", lambda path: sessions)
    monkeypatch.setattr(position_training, "export_position_checkpoint", lambda path: path.with_suffix(".onnx"))
    loaders, totals = [], []
    original_loader = position_training.DataLoader
    original_loss = position_training.position_regression_loss

    def loader(*args, **kwargs):
        result = original_loader(*args, **kwargs)
        loaders.append(result)
        return result

    def loss(prediction, target):
        result = original_loss(prediction, target)
        totals.append((result.item() * len(prediction), (prediction.detach().double() - target.double()).abs().sum().item()))
        return result

    monkeypatch.setattr(position_training, "DataLoader", loader)
    monkeypatch.setattr(position_training, "position_regression_loss", loss)
    output = tmp_path / "run"
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        checkpoint = position_training.train_position_model(
            tmp_path, output=output, epochs=1, imgsz=32, batch=2, sequence_length=sequence_length,
            device="cpu", upload=False, **options,
        )
    finally:
        torch.set_num_threads(previous_threads)
    train, val = [loader.dataset for loader in loaders]
    assert isinstance(train, Subset) and len(train) == count
    assert len(train.dataset) == len(val) == 5
    assert {row["session_id"] for row in train.dataset.records}.isdisjoint(row["session_id"] for row in val.records)
    selected = [train.dataset.records[index] for index in train.indices]
    assert any(int(row["image"].stem.split("_")[-1]) > 0 for row in selected)
    for index, current in enumerate(selected):
        number = int(current["image"].stem.split("_")[-1])
        if sequence_length > 1:
            history = list(range(max(0, number - 3), number + 1))
            assert train[index]["references"][:, 0, 0].tolist() == [history[0]] * (4 - len(history)) + history
    split = json.loads((output / "split.json").read_text())
    assert split["train"] == [str(row["image"]) for row in selected]
    assert set(split["val"]) == {str(row["image"]) for row in val.records}
    assert split["density_sampling"] == {
        "min_offset_distance": None, "min_orientation_distance_deg": None, "min_normalized_position_distance": None,
        **options, "candidate_count": 5, "selected_count": count,
    }
    if sequence_length > 1:
        assert set(split["train_history"]) == {str(row["image"]) for row in train.dataset.records}
    else:
        assert "train_history" not in split
    metrics = json.loads((output / "results.json").read_text())[0]
    assert metrics["train_loss"] == pytest.approx(sum(loss for loss, _ in totals) / len(train))
    assert metrics["train_mae"] == pytest.approx(sum(error for _, error in totals) / (len(train) * 3))
    assert metrics["val_by_map"][0]["count"] == 5
    torch.testing.assert_close(PositionModel.load(checkpoint).position_mean, torch.tensor([2., 0., 0.]))
