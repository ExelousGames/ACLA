"""Driver-pose coverage, periodic angles, and density CLI integration."""

import json
from pathlib import Path
import random

import numpy as np
from PIL import Image
import pytest
import torch

from training.image_segmentation.__main__ import main
from training.image_segmentation import position_training
from training.image_segmentation.position_data import TARGET_DEFINITION, PositionDataset, load_position_manifest
from training.image_segmentation.position_density import select_position_endpoints


def record(index=0, *, xyz=(0, 0, 0), degrees=(0, 0, 0), game="acc", map_name="Monza"):
    return {
        "image": Path(f"frame_{index}.png"), "session_id": str(index),
        "game": game, "map_name": map_name, "driver_position": list(xyz),
        "orientation": np.deg2rad(degrees),
    }


def test_only_driver_pose_and_game_map_define_redundancy():
    rows = [record(0), record(1), record(2, game="other"), record(3, map_name="Spa"), record(4, xyz=(10, 0, 0))]
    for index, row in enumerate(rows):
        row.update(target=[[index, 0, 0]], references=[[index, 0, 0]], normalized_position=index / len(rows))
    kept = select_position_endpoints(rows, position_radius=1)
    assert len(set(kept) & {0, 1}) == 1  # Redundancy spans sessions before splitting.
    assert {2, 3, 4} <= set(kept)
    assert kept == sorted(kept)
    assert kept == select_position_endpoints(rows, position_radius=1)


@pytest.mark.parametrize("axis", range(3))
def test_each_world_position_axis_retains_sparse_coverage(axis):
    xyz = np.zeros(3)
    xyz[axis] = 1.01
    assert select_position_endpoints([record(), record(1, xyz=xyz)], position_radius=1) == [0, 1]


def test_position_radius_uses_euclidean_distance_and_includes_boundary():
    assert len(select_position_endpoints([record(), record(1, xyz=(1, 0, 0))], position_radius=1)) == 1
    assert select_position_endpoints([record(), record(1, xyz=(0.8, 0.8, 0))], position_radius=1) == [0, 1]


@pytest.mark.parametrize("axis", range(3))
def test_each_orientation_axis_wraps_and_preserves_distinct_poses(axis):
    left, right = np.zeros(3), np.zeros(3)
    left[axis], right[axis] = 179, -179
    rows = [record(degrees=left), record(1, degrees=right)]
    assert len(select_position_endpoints(rows, position_radius=1, orientation_degrees=3)) == 1
    assert select_position_endpoints(rows, position_radius=1, orientation_degrees=1) == [0, 1]


def test_indexed_selection_matches_brute_force_retained_neighbor_coverage():
    rng = np.random.default_rng(7)
    positions = rng.normal(size=(300, 3))
    angles = rng.normal(0, 4, size=(300, 3)) + rng.choice([-180, 180], size=(300, 3))
    rows = [record(index, xyz=xyz, degrees=angle) for index, (xyz, angle) in enumerate(zip(positions, angles))]
    # Independent reference: scan retained representatives, never suppress via a discarded frame.
    radians = np.deg2rad(angles)
    radius, limit = 0.7, np.deg2rad(5)
    order = list(range(len(rows)))
    random.Random(42).shuffle(order)
    expected = []
    for current in order:
        redundant = any(
            np.linalg.norm(positions[current] - positions[previous]) <= radius
            and np.all(np.abs((radians[current] - radians[previous] + np.pi) % (2 * np.pi) - np.pi) <= limit)
            for previous in expected
        )
        if not redundant:
            expected.append(current)
    actual = select_position_endpoints(rows, position_radius=radius)
    assert actual == sorted(expected)
    assert 1 < len(actual) < len(rows)


@pytest.mark.parametrize("radius", [0, -1, float("nan"), float("inf")])
def test_invalid_radius_is_rejected(radius):
    with pytest.raises(ValueError, match="position radius"):
        select_position_endpoints([], position_radius=radius)


@pytest.mark.parametrize("angle", [0, -1, 181, float("nan"), float("inf")])
def test_invalid_angle_is_rejected(angle):
    with pytest.raises(ValueError, match="orientation degrees"):
        select_position_endpoints([], position_radius=1, orientation_degrees=angle)


@pytest.mark.parametrize("value", [None, [1, 2], [0, 0, float("nan")], [0, 0, float("inf")], [True] * 3, ["0"] * 3])
def test_sampling_requires_real_finite_driver_xyz(value):
    row = record()
    row["driver_position"] = value
    with pytest.raises(ValueError, match="driver_position"):
        select_position_endpoints([row], position_radius=1)


def test_disabled_sampling_accepts_legacy_frames_without_driver_xyz():
    assert select_position_endpoints([{}, {}]) == [0, 1]
    row = record()
    del row["driver_position"]
    with pytest.raises(ValueError, match="driver_position"):
        select_position_endpoints([row], position_radius=1)


def test_density_sampling_requires_two_retained_sessions_per_map(tmp_path, monkeypatch):
    sessions = {str(index): [record(index)] for index in range(2)}
    monkeypatch.setattr(position_training, "load_position_manifest", lambda path: sessions)
    output = tmp_path / "run"
    with pytest.raises(ValueError, match="at least two sessions per map"):
        position_training.train_position_model(
            tmp_path, output=output, density_position_radius=1, device="cpu", upload=False,
        )
    assert not output.exists()


@pytest.mark.parametrize("indexed", [False, True])
def test_loader_retains_driver_xyz_only_as_selection_metadata(tmp_path, indexed):
    path = tmp_path / "frame_0.position.json"
    frame = {
        **record(), "image": "frame_0.png", "orientation": [0, 0, 0],
        "normalized_position": 0, "references": [[0, 0, 0]], "target": [[0, 0, 0]],
        "coordinate_frame": "car_local", "target_definition": TARGET_DEFINITION,
    }
    Image.new("RGB", (8, 8)).save(tmp_path / frame["image"])
    path.write_text(json.dumps(frame))
    manifest = tmp_path / "positions.json"
    manifest.write_text(json.dumps({"sessions": [{"id": "0", "frames": [{"position_file": path.name}]}]}))
    source = manifest if indexed else tmp_path
    rows = load_position_manifest(source)["0"]
    assert rows[0]["driver_position"] == [0, 0, 0]
    categories = {"game": ["acc"], "map_name": ["Monza"]}
    before = PositionDataset(rows, 8, categories)[0]
    frame["driver_position"] = [100, 200, 300]
    path.write_text(json.dumps(frame))
    after = PositionDataset(load_position_manifest(source)["0"], 8, categories)[0]
    assert "driver_position" not in before
    for field in ("image", "references", "context", "orientation", "target"):
        torch.testing.assert_close(before[field], after[field])


@pytest.mark.parametrize("command", ["train-position", "train-position-v2"])
@pytest.mark.parametrize("arguments,radius,angle", [([], None, 5), (["--density-position-radius", "0.5", "--density-orientation-deg", "3"], 0.5, 3)])
def test_both_commands_forward_density_options(monkeypatch, command, arguments, radius, angle):
    captured = {}

    def train(manifest, **kwargs):
        captured.update(kwargs)
        return Path("run/best.pt")

    monkeypatch.setattr(position_training, "train_position_model", train)
    assert main([command, "--manifest", "data", "--output", "run", "--no-upload", *arguments]) == 0
    assert captured["density_position_radius"] == radius
    assert captured["density_orientation_deg"] == angle
