from __future__ import annotations

import asyncio
import inspect
import json
from email import policy
from email.parser import BytesParser
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import numpy as np
import pytest
import torch
from PIL import Image

from training.image_segmentation.__main__ import main
from training.image_segmentation.position_data import (
    PositionCurve, PositionDataset, encode_position_context, load_position_manifest,
)
from training.image_segmentation.position_model import PositionModel
from training.image_segmentation.position_preparation import signed_targets
from training.image_segmentation.position_training import position_regression_loss, train_position_model


CATEGORIES = {"game": ["acc", "iracing"], "map_name": ["Display Name", "Other Track"]}


@pytest.fixture
def aligned_manifest(tmp_path):
    sessions = []
    for session_id, color in (("session-a", "red"), ("session-b", "blue")):
        Image.new("RGB", (80, 40), color).save(tmp_path / f"{session_id}.png")
        position = tmp_path / f"{session_id}.position.json"
        position.write_text(json.dumps({
            "session_id": session_id,
            "image": f"{session_id}.png",
            "game": "acc" if session_id == "session-a" else "iracing",
            "map_name": "Display Name" if session_id == "session-a" else "Other Track",
            "normalized_position": 0.25,
            "references": [[0.2, -0.1, 0.1], [0.25, -0.1, 0.1]],
            "target": [[0.1, -0.2, 0], [0.15, -0.2, 0]],
        }))
        sessions.append({
            "id": session_id,
            "frames": [{"position_file": position.name}],
        })
    path = tmp_path / "positions.json"
    path.write_text(json.dumps({"sessions": sessions}))
    return path


@pytest.fixture
def training_manifest(aligned_manifest):
    manifest = json.loads(aligned_manifest.read_text())
    for session in list(manifest["sessions"]):
        position = aligned_manifest.parent / session["frames"][0]["position_file"]
        frame = json.loads(position.read_text())
        image = aligned_manifest.parent / frame["image"]
        frame["image"] = f"{session['id']}-extra.png"
        frame["session_id"] = f"{session['id']}-extra"
        (aligned_manifest.parent / frame["image"]).write_bytes(image.read_bytes())
        extra = position.with_name(f"{session['id']}-extra.position.json")
        extra.write_text(json.dumps(frame))
        manifest["sessions"].append({
            "id": frame["session_id"], "frames": [{"position_file": extra.name}],
        })
    aligned_manifest.write_text(json.dumps(manifest))
    return aligned_manifest


@pytest.fixture
def position_folder(aligned_manifest):
    folder = aligned_manifest.parent / "pairs"
    folder.mkdir()
    manifest = json.loads(aligned_manifest.read_text())
    for session in manifest["sessions"]:
        for entry in session["frames"]:
            position = aligned_manifest.parent / entry["position_file"]
            frame = json.loads(position.read_text())
            image = aligned_manifest.parent / frame["image"]
            (folder / image.name).write_bytes(image.read_bytes())
            (folder / position.name).write_bytes(position.read_bytes())
    # Labelme annotations must not be interpreted as position files.
    (folder / "session-a.json").write_text(json.dumps({"shapes": []}))
    return folder


def test_position_folder_scans_root_and_nested_folders(position_folder):
    nested = position_folder / "nested" / "frames"
    nested.mkdir(parents=True)
    for name in ("session-b.png", "session-b.position.json"):
        (position_folder / name).rename(nested / name)
    (nested / "metadata.json").write_text("{}")

    sessions = load_position_manifest(position_folder)
    assert all(row["session_id"] == session_id for session_id, frames in sessions.items() for row in frames)
    assert {row["image"] for frames in sessions.values() for row in frames} == {
        position_folder / "session-a.png", nested / "session-b.png",
    }


@pytest.mark.parametrize("fault", ["missing_image", "references", "target", "session", "duplicate_image", "empty"])
def test_position_folder_rejects_invalid_pairs(position_folder, fault):
    path = position_folder / "session-a.position.json"
    position = json.loads(path.read_text())
    if fault == "missing_image":
        (position_folder / position["image"]).unlink()
    elif fault in ("references", "target"):
        position[fault] = [[0, 1, 2], [3, 4, float("nan")]]
    elif fault == "session":
        position["session_id"] = ""
    elif fault == "duplicate_image":
        (position_folder / "duplicate.position.json").write_text(json.dumps(position))
    else:
        for file in position_folder.glob("*.position.json"):
            file.unlink()
    if fault != "empty":
        path.write_text(json.dumps(position))
    with pytest.raises((ValueError, FileNotFoundError)):
        load_position_manifest(position_folder)


def test_signed_xyz_targets_use_raw_subtraction_without_units_or_norm():
    references = np.array([[0.003, -0.002, 0.005], [0.001, 0.004, -0.007]])
    np.testing.assert_allclose(
        signed_targets(references, [0.002, 0.001, 0.003]),
        [[0.001, -0.003, 0.002], [-0.001, 0.003, -0.010]], rtol=1e-6,
    )


def test_lookup_interpolates_all_axes_and_wraps_start_finish():
    curve = PositionCurve([0.9, 0.1], [[3, 4, 5], [1, 2, 3]])
    np.testing.assert_allclose(curve.at(0.95), [2.5, 3.5, 4.5])
    np.testing.assert_allclose(curve.at(0), [2, 3, 4])
    np.testing.assert_allclose(curve.at(1), curve.at(0))
    explicit = PositionCurve([0, 1], [[1, 2, 3], [4, 5, 6]])
    np.testing.assert_array_equal(explicit.at(0), [1, 2, 3])
    np.testing.assert_array_equal(explicit.at(1), [4, 5, 6])


@pytest.mark.parametrize("position", [float("nan"), float("inf"), -0.01, 1.01])
def test_invalid_normalized_position_is_rejected(position):
    with pytest.raises(ValueError, match="Normalized position"):
        PositionCurve([0, 0.5], [[0, 0, 0], [1, 1, 1]]).at(position)


@pytest.mark.parametrize("positions, xyz", [
    ([0, 0], [[1, 2, 3], [4, 5, 6]]),
    ([0, 0.5], [[1, 2, 3], [4, float("nan"), 6]]),
    ([0], [[1, 2, 3]]),
    ([0, 0.5], [[1, 2], [3, 4]]),
])
def test_missing_or_invalid_curve_is_not_replaced_with_zeros(positions, xyz):
    with pytest.raises(ValueError):
        PositionCurve(positions, xyz)


def test_prepared_targets_can_only_change_labels(aligned_manifest):
    before = load_position_manifest(aligned_manifest)
    path = aligned_manifest.parent / "session-a.position.json"
    frame = json.loads(path.read_text())
    frame["target"] = [[1.2, -2.1, 3.1], [1.25, -2.1, 3.1]]
    path.write_text(json.dumps(frame))
    after = load_position_manifest(aligned_manifest)
    a, b = (PositionDataset(records["session-a"], 64, CATEGORIES)[0] for records in (before, after))
    assert set(a) == {"image", "references", "context", "target", "game", "map_name"}
    assert (a["game"], a["map_name"]) == (b["game"], b["map_name"]) == ("acc", "Display Name")
    torch.testing.assert_close(a["image"], b["image"])
    torch.testing.assert_close(a["references"], b["references"])
    torch.testing.assert_close(a["context"], b["context"])
    torch.testing.assert_close(a["context"], torch.tensor([0.25, 1, 0, 1, 0]))
    assert not torch.equal(a["target"], b["target"])
    np.testing.assert_allclose(before["session-a"][0]["target"], [[0.1, -0.2, 0], [0.15, -0.2, 0]], atol=1e-7)
    assert list(inspect.signature(PositionModel.forward).parameters) == ["self", "images", "references", "context"]
    assert "driver_position" not in inspect.signature(PositionModel.predict_position).parameters
    assert "centerline" not in inspect.signature(PositionModel.predict_position).parameters


@pytest.mark.parametrize("remove_centerline", [False, True])
def test_centerline_data_is_not_used_as_a_model_input_or_target(aligned_manifest, remove_centerline):
    before = load_position_manifest(aligned_manifest)
    path = aligned_manifest.parent / "session-a.position.json"
    frame = json.loads(path.read_text())
    for field in ("references", "target"):
        if remove_centerline:
            frame[field] = frame[field][:1]
        else:
            frame[field][1] = [1e6, -1e6, 1e6]
    path.write_text(json.dumps(frame))
    after = load_position_manifest(aligned_manifest)
    a, b = (PositionDataset(records["session-a"], 64, CATEGORIES)[0] for records in (before, after))
    assert (a["game"], a["map_name"]) == (b["game"], b["map_name"])
    for field in ("image", "references", "context", "target"):
        torch.testing.assert_close(a[field], b[field])
    torch.testing.assert_close(a["references"], torch.tensor([[0.2, -0.1, 0.1]]))
    torch.testing.assert_close(a["target"], torch.tensor([[0.1, -0.2, 0]]))


def test_model_rejects_centerline_input():
    model = PositionModel(imgsz=64, categories=CATEGORIES)
    with pytest.raises(ValueError, match="reference-lap XYZ"):
        model(torch.zeros(1, 3, 64, 64), torch.zeros(1, 2, 3), torch.zeros(1, model.context_size))


@pytest.mark.parametrize("field, value", [
    ("game", None), ("game", " "), ("game", 1),
    ("map_name", None), ("map_name", " "), ("map_name", 1),
    ("normalized_position", None), ("normalized_position", True), ("normalized_position", "0.25"),
    ("normalized_position", -0.01), ("normalized_position", 1.01),
    ("normalized_position", float("nan")), ("normalized_position", float("inf")),
])
def test_prepared_context_must_be_valid(aligned_manifest, field, value):
    path = aligned_manifest.parent / "session-a.position.json"
    frame = json.loads(path.read_text())
    frame[field] = value
    path.write_text(json.dumps(frame))
    with pytest.raises(ValueError, match=field):
        load_position_manifest(aligned_manifest)


@pytest.mark.parametrize("field", ["game", "map_name", "normalized_position"])
def test_old_position_files_require_regeneration(aligned_manifest, field):
    path = aligned_manifest.parent / "session-a.position.json"
    frame = json.loads(path.read_text())
    del frame[field]
    path.write_text(json.dumps(frame))
    with pytest.raises(ValueError, match="regenerate the dataset"):
        load_position_manifest(aligned_manifest)


def test_context_encoding_uses_saved_category_order_and_preserves_lap_endpoints():
    categories = {"game": ["iracing", "acc"], "map_name": ["Other Track", "Display Name"]}
    np.testing.assert_array_equal(encode_position_context(0, "acc", "Display Name", categories), [0, 0, 1, 0, 1])
    np.testing.assert_array_equal(encode_position_context(1, "iracing", "Other Track", categories), [1, 1, 0, 1, 0])


@pytest.mark.parametrize("game, map_name, field", [
    ("unknown", "Display Name", "game"), ("acc", "unknown", "map_name"),
])
def test_context_encoding_rejects_unknown_categories(game, map_name, field):
    with pytest.raises(ValueError, match=f"Unknown {field}"):
        encode_position_context(0.25, game, map_name, CATEGORIES)


@pytest.mark.parametrize("field", ["references", "target"])
@pytest.mark.parametrize("value", [None, [1, 2, 3], [[1, 2], [3, 4]], [[1, 2, 3], [4, float("nan"), 6]],
                                  [[1, 2, 3], [4, 5, float("inf")]]])
def test_prepared_xyz_arrays_must_be_complete_and_finite(aligned_manifest, field, value):
    path = aligned_manifest.parent / "session-a.position.json"
    frame = json.loads(path.read_text())
    frame[field] = value
    path.write_text(json.dumps(frame))
    with pytest.raises(ValueError, match=field):
        load_position_manifest(aligned_manifest)


@pytest.mark.parametrize("frame", [
    {"image": "session-a.png", "normalized_position": 0.25, "driver_position": [0.1, 0.1, 0.1]},
    {"image": "session-a.png", "references": [[1, 2, 3], [4, 5, 6]], "target": [[0, 0, 0], [1, 1, 1]]},
])
def test_legacy_inline_manifest_requires_preparation(aligned_manifest, frame):
    manifest = json.loads(aligned_manifest.read_text())
    manifest["sessions"][0]["frames"][0] = frame
    aligned_manifest.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="rerun prepare_position_dataset.py"):
        load_position_manifest(aligned_manifest)


def test_legacy_preset_split_is_rejected(aligned_manifest):
    manifest = json.loads(aligned_manifest.read_text())
    manifest["sessions"][0]["split"] = "train"
    aligned_manifest.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="split"):
        load_position_manifest(aligned_manifest)


@pytest.mark.parametrize("leak", ["session", "image", "empty_session", "empty_manifest"])
def test_duplicate_or_empty_prepared_data_is_rejected(aligned_manifest, leak):
    manifest = json.loads(aligned_manifest.read_text())
    train, val = manifest["sessions"]
    if leak == "session":
        val["id"] = train["id"]
    elif leak == "image":
        path = aligned_manifest.parent / val["frames"][0]["position_file"]
        frame = json.loads(path.read_text())
        frame["image"] = "session-a.png"
        path.write_text(json.dumps(frame))
    elif leak == "empty_session":
        val["frames"] = []
    else:
        manifest["sessions"] = []
    aligned_manifest.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        load_position_manifest(aligned_manifest)


def test_existing_top_lap_entries_supply_reference_xyz():
    from app.shared.expert_features import ExpertFeatureCatalog
    from app.top_laps.model import TopLapEntry

    features = ExpertFeatureCatalog.ExpertOptimalFeature
    names = [getattr(features, f"EXPERT_OPTIMAL_PLAYER_POS_{axis}").value for axis in "XYZ"]
    entry = TopLapEntry(
        track="test", car="car", avg_grip_int=2,
        x=np.array([0, 0.5]), y=np.array([[1, 2, 3], [3, 4, 5]]), target_features=names,
    )
    np.testing.assert_allclose(PositionCurve.from_top_lap(entry).at(0.25), [2, 3, 4])
    entry.target_features = names[:2]
    with pytest.raises(ValueError, match="three player XYZ"):
        PositionCurve.from_top_lap(entry)


@pytest.fixture(autouse=True)
def single_threaded_torch():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("prediction, penalty", [
    ([1, 0, 0], 0),
    ([2, 0, 0], 0),
    ([1, 1, 0], 1 - 1 / np.sqrt(2)),
    ([0, 2, 0], 1),
    ([0, 0, 2], 1),
    ([-2, 0, 0], 2),
])
@pytest.mark.parametrize("prediction_scale, target_scale", [(1, 1), (0.001, 10), (4, 0.1)])
def test_position_loss_keeps_full_direction_weight_at_or_above_point_one(prediction, penalty, prediction_scale, target_scale):
    prediction = torch.tensor([[prediction]], dtype=torch.float32) * prediction_scale
    target = torch.tensor([[[target_scale, 0, 0]]], dtype=torch.float32)
    component_loss = torch.nn.functional.smooth_l1_loss(prediction, target)
    assert position_regression_loss(prediction, target).item() == pytest.approx(component_loss.item() + penalty)


@pytest.mark.parametrize("target_length, direction_penalty", [
    (0, 0), (0.001, 0.0001), (0.01, 0.01), (0.05, 0.25), (0.1, 1),
])
@pytest.mark.parametrize("prediction_length", [0.001, 1])
def test_position_loss_fades_direction_for_small_targets(target_length, direction_penalty, prediction_length):
    prediction = torch.tensor([[[0., prediction_length, 0]]])
    target = torch.tensor([[[target_length, 0., 0]]])
    component_loss = torch.nn.functional.smooth_l1_loss(prediction, target)
    assert position_regression_loss(prediction, target).item() == pytest.approx(
        component_loss.item() + direction_penalty,
    )


def test_position_loss_does_not_renormalize_small_target_weights_in_mixed_batches():
    prediction = torch.tensor([[[0., 1, 0]], [[-0.008, 0.006, 0]], [[1., 0, 0]]])
    target = torch.tensor([[[1., 0, 0]], [[0.006, 0.008, 0]], [[0., 0, 0]]])
    component_loss = torch.nn.functional.smooth_l1_loss(prediction, target)
    # Perpendicular targets have lengths 1 and 0.01; the zero target is excluded.
    assert position_regression_loss(prediction, target).item() == pytest.approx(
        component_loss.item() + (1 + 0.01) / 2,
    )


@pytest.mark.parametrize("length", [1e-3, 1e-6, 1e-12])
def test_direction_gradient_fades_when_prediction_and_target_approach_zero(length):
    prediction = torch.tensor([[[0., length, 0]]], requires_grad=True)
    target = torch.tensor([[[length, 0., 0]]])
    direction_loss = position_regression_loss(prediction, target) - torch.nn.functional.smooth_l1_loss(prediction, target)
    direction_loss.backward()
    assert torch.isfinite(prediction.grad).all()
    assert prediction.grad[0, 0, 0] < 0
    assert prediction.grad.norm().item() <= 101 * length


@pytest.mark.parametrize("zero_targets", [False, True])
def test_position_loss_excludes_zero_targets_from_direction_average(zero_targets):
    prediction = torch.tensor([[[2., 0, 0]], [[0, 1, 0]], [[-1, 0, 0]]], requires_grad=True)
    target = torch.tensor([[[0., 0, 0]], [[1, 0, 0]], [[1, 0, 0]]])
    if zero_targets:
        target.zero_()
    loss = position_regression_loss(prediction, target)
    component_loss = torch.nn.functional.smooth_l1_loss(prediction, target)
    assert loss.item() == pytest.approx(component_loss.item() + (0 if zero_targets else 1.5))
    loss.backward()
    assert torch.isfinite(prediction.grad).all()


@pytest.mark.parametrize("prediction_length", [0, 1e-12])
@pytest.mark.parametrize("zero_target", [False, True])
def test_position_loss_has_finite_gradients_near_zero(prediction_length, zero_target):
    prediction = torch.tensor([[[prediction_length, 0, 0]]], dtype=torch.float32, requires_grad=True)
    target = torch.tensor([[[0., 0, 0] if zero_target else [0., 1, 0]]])
    loss = position_regression_loss(prediction, target)
    component_loss = torch.nn.functional.smooth_l1_loss(prediction, target)
    assert loss.item() == pytest.approx(component_loss.item() + (0 if zero_target else 1))
    loss.backward()
    assert torch.isfinite(prediction.grad).all()
    if not zero_target:
        assert prediction.grad[0, 0, 1] < 0


def test_direction_penalty_gradient_rotates_prediction_towards_target():
    prediction = torch.tensor([[[0., 1, 0]]], requires_grad=True)
    target = torch.tensor([[[1., 0, 0]]])
    direction_loss = position_regression_loss(prediction, target) - torch.nn.functional.smooth_l1_loss(prediction, target)
    direction_loss.backward()
    assert torch.isfinite(prediction.grad).all()
    updated = prediction.detach() - 0.1 * prediction.grad
    assert torch.nn.functional.cosine_similarity(updated, target, dim=-1).item() > 0


def test_position_training_optimizes_component_and_direction_errors(training_manifest, tmp_path, monkeypatch):
    from training.image_segmentation import position_training

    network = PositionModel(imgsz=64, categories=CATEGORIES)
    with torch.no_grad():
        # Perpendicular to the training target [0.1, -0.2, 0].
        network.head[2].bias.copy_(torch.tensor([0.2, 0.1, 0]))
    network.register_forward_hook(
        lambda module, inputs, output: module.head[2].bias.reshape(1, 1, 3).expand_as(output),
    )
    monkeypatch.setattr(position_training, "PositionModel", lambda *args, **kwargs: network)
    monkeypatch.setattr(position_training, "export_position_checkpoint", lambda checkpoint: checkpoint.with_suffix(".onnx"))
    output = tmp_path / "run"
    train_position_model(training_manifest, output=output, epochs=1, imgsz=64, batch=2, device="cpu", upload=False)

    metrics = json.loads((output / "results.json").read_text())[0]
    assert metrics["train_loss"] == pytest.approx(1 / 60 + 1)
    assert metrics["train_mae"] == pytest.approx(0.4 / 3)


@pytest.mark.parametrize("batch", [3, 4])
def test_position_training_mae_weights_frames_and_resets_each_epoch(
    aligned_manifest, tmp_path, monkeypatch, capsys, batch,
):
    from training.image_segmentation import position_training

    # Absolute XYZ errors sum to 6, 5, 6, and 11 for these four frames.
    predictions = [[2, -1, 0], [-3, 0, 2], [0, 0, 0], [1, 5, -4]]
    targets = [[1, 2, -2], [0, -1, 1], [1, -2, 3], [-2, 1, 0]]

    def split(records, **kwargs):
        return {
            "train": [
                dict(records[0], references=np.array([prediction], dtype=np.float32),
                     target=np.array([target], dtype=np.float32))
                for prediction, target in zip(predictions, targets)
            ],
            "val": records[:1],
        }

    network = PositionModel(imgsz=64, categories=CATEGORIES)
    # Supply fixed predictions while retaining a gradient path for training.
    network.register_forward_hook(lambda module, inputs, output: inputs[1] + output * 0)
    monkeypatch.setattr(position_training, "PositionModel", lambda *args, **kwargs: network)
    monkeypatch.setattr(position_training, "split_position_frames", split)
    monkeypatch.setattr(position_training, "export_position_checkpoint", lambda checkpoint: checkpoint.with_suffix(".onnx"))
    output = tmp_path / "run"
    train_position_model(aligned_manifest, output=output, epochs=2, imgsz=64, batch=batch, device="cpu", upload=False)

    history = json.loads((output / "results.json").read_text())
    assert [row["train_mae"] for row in history] == pytest.approx([7 / 3, 7 / 3])
    for name in ("best.pt", "last.pt"):
        metrics = torch.load(output / name, weights_only=True)["metrics"]
        assert metrics["train_mae"] == pytest.approx(7 / 3)
    assert capsys.readouterr().out.count("train MAE=2.33333, val MAE=") == 2


def test_regression_trains_image_encoder_and_head_from_scratch_and_roundtrips(aligned_manifest, tmp_path):
    model = PositionModel(imgsz=64, categories=CATEGORIES)
    before = {key: value.clone() for key, value in model.image_encoder.state_dict().items()}
    records = load_position_manifest(aligned_manifest)
    sample = PositionDataset(records["session-a"], 64, CATEGORIES)[0]
    model.train()
    assert all(parameter.requires_grad for parameter in model.parameters())
    encoder_before = {key: value.clone() for key, value in model.position_encoder.state_dict().items()}
    optimizer = torch.optim.Adam((parameter for parameter in model.parameters() if parameter.requires_grad), lr=0.01)
    inputs = tuple(sample[field].unsqueeze(0) for field in ("image", "references", "context"))
    prediction = model(*inputs)
    assert prediction.shape == (1, 1, 3)
    loss = position_regression_loss(prediction, sample["target"].unsqueeze(0))
    loss.backward()
    assert model.head[0].weight.grad.abs().sum() > 0
    assert model.position_encoder[0].weight.grad.abs().sum() > 0
    assert all(parameter.grad is not None and parameter.grad.abs().sum() > 0
               for parameter in model.image_encoder.parameters())
    optimizer.step()
    assert any(not torch.equal(value, encoder_before[key]) for key, value in model.position_encoder.state_dict().items())
    assert all(not torch.equal(value, before[key]) for key, value in model.image_encoder.state_dict().items())
    model.eval()
    with torch.no_grad():
        expected = model(*inputs)
    checkpoint = tmp_path / "position.pt"
    model.save(checkpoint)
    restored = PositionModel.load(checkpoint)
    assert restored.categories == CATEGORIES
    assert all(parameter.requires_grad for parameter in restored.parameters())
    saved = torch.load(checkpoint, weights_only=True)
    assert "yolo_yaml" not in saved and "names" not in saved
    assert not any(key.startswith("yolo.") for key in saved["state_dict"])
    with torch.no_grad():
        actual = restored(*inputs)
    torch.testing.assert_close(actual, expected)
    curve = PositionCurve([0, 0.5], [[0, 0, 0], [1, 2, 3]])
    prediction = restored.predict_position(
        records["session-b"][0]["image"], 0.25, curve, game="iracing", map_name="Other Track",
    )
    assert set(prediction) == {
        "reference_delta",
    }
    assert prediction == model.predict_position(
        records["session-b"][0]["image"], 0.25, curve, game="iracing", map_name="Other Track",
    )
    with pytest.raises(ValueError, match="Unknown map_name"):
        restored.predict_position(tmp_path / "missing.png", 0.25, curve, game="acc", map_name="Unknown")


@pytest.mark.parametrize("image_size", [64, 96])
def test_position_predictions_distinguish_layouts_with_identical_global_averages(image_size):
    model = PositionModel(imgsz=image_size, categories=CATEGORIES).eval()

    def spatial_features(module, inputs, output):
        # Identical per-channel means, but opposite left/right layouts in the batch.
        features = torch.zeros_like(output)
        features[0, :, :, 0] = 1
        features[1] = features[0].flip(-1)
        torch.testing.assert_close(features[0].mean((-2, -1)), features[1].mean((-2, -1)))
        return features

    handle = model.image_encoder.register_forward_hook(spatial_features)
    try:
        with torch.no_grad():
            # Route one spatial cell through the encoder and regression head.
            for parameter in (*model.position_encoder.parameters(), *model.head.parameters()):
                parameter.zero_()
            model.position_encoder[0].weight[0, 0] = 1
            model.head[0].weight[0, 0] = 1
            model.head[2].weight[0, 0] = 1
            prediction = model(
                torch.zeros(2, 3, image_size, image_size), torch.zeros(2, 1, 3),
                torch.tensor([[0.25, 1, 0, 1, 0]]).repeat(2, 1),
            )
        assert prediction.shape == (2, 1, 3)
        assert prediction[0, 0, 0] > prediction[1, 0, 0]
    finally:
        handle.remove()


def test_all_context_inputs_affect_predictions_with_fixed_image_and_references():
    model = PositionModel(imgsz=64, categories=CATEGORIES).eval()
    with torch.no_grad():
        model.head[0].weight.zero_()
        model.head[0].bias.zero_()
        model.head[0].weight[0, -model.context_size:] = torch.tensor([1, 2, 4, 8, 16])
        model.head[2].weight.zero_()
        model.head[2].bias.zero_()
        model.head[2].weight[0, 0] = 1
        for position, game, map_name, expected in (
            (0.25, "acc", "Display Name", 10.25),
            (0.75, "acc", "Display Name", 10.75),
            (0.25, "iracing", "Display Name", 12.25),
            (0.25, "acc", "Other Track", 18.25),
        ):
            context = torch.from_numpy(encode_position_context(position, game, map_name, CATEGORIES)).unsqueeze(0)
            prediction = model(torch.zeros(1, 3, 64, 64), torch.zeros(1, 1, 3), context)
            assert prediction[0, 0, 0].item() == expected


@pytest.mark.parametrize("label", ["acla-yolo-position-v1", "acla-yolo-position-v2", None])
def test_checkpoint_without_context_requires_retraining(position_checkpoint, label):
    saved = torch.load(position_checkpoint, weights_only=True)
    saved["format"] = label
    saved.pop("categories")
    torch.save(saved, position_checkpoint)
    with pytest.raises(ValueError, match="retrain older models"):
        PositionModel.load(position_checkpoint)


def test_checkpoint_without_spatial_encoder_requires_retraining(position_checkpoint):
    saved = torch.load(position_checkpoint, weights_only=True)
    saved["state_dict"] = {
        key: value for key, value in saved["state_dict"].items() if not key.startswith("position_encoder.")
    }
    torch.save(saved, position_checkpoint)
    with pytest.raises(ValueError, match="spatial encoder; retrain from scratch"):
        PositionModel.load(position_checkpoint)


def test_checkpoint_with_centerline_input_requires_retraining(position_checkpoint):
    saved = torch.load(position_checkpoint, weights_only=True)
    state = saved["state_dict"]
    state["head.0.weight"] = torch.zeros(128, state["head.0.weight"].shape[1] + 3)
    state["head.2.weight"] = torch.zeros(6, 128)
    state["head.2.bias"] = torch.zeros(6)
    state["position_mean"], state["position_std"] = torch.zeros(6), torch.ones(6)
    torch.save(saved, position_checkpoint)
    with pytest.raises(ValueError, match="centerline inputs; retrain from scratch"):
        PositionModel.load(position_checkpoint)


def test_segmentation_based_checkpoint_requires_training_from_scratch(position_checkpoint):
    saved = torch.load(position_checkpoint, weights_only=True)
    saved["state_dict"] = {
        key.replace("image_encoder.", "yolo."): value for key, value in saved["state_dict"].items()
    }
    torch.save(saved, position_checkpoint)
    with pytest.raises(ValueError, match="standalone image encoder; retrain from scratch"):
        PositionModel.load(position_checkpoint)


@pytest.mark.parametrize("use_position_folder", [False, True])
def test_session_training_saves_raw_axis_metrics_and_uses_only_train_reference_statistics(
    aligned_manifest, tmp_path, position_folder, use_position_folder, capsys,
):
    path = aligned_manifest.parent / "session-b.position.json"
    frame = json.loads(path.read_text())
    frame.update(game="acc", map_name="Display Name")
    frame["references"] = [[0.3, -0.4, 0.5], [0.35, -0.4, 0.5]]
    frame["target"] = [[0.2, -0.5, 0.4], [0.25, -0.5, 0.4]]
    path.write_text(json.dumps(frame))
    if use_position_folder:
        (position_folder / path.name).write_bytes(path.read_bytes())
    training_input = position_folder if use_position_folder else aligned_manifest
    assert len(load_position_manifest(training_input)) == 2
    prepared = aligned_manifest.read_bytes()
    output = tmp_path / "run"
    checkpoint = train_position_model(
        training_input, output=output,
        epochs=2, imgsz=64, batch=1, device="cpu", upload=False,
    )
    assert checkpoint == output / "best.pt"
    assert (output / "last.pt").is_file()
    assert (output / "best.onnx").is_file()
    assert (output / "best.onnx.json").is_file()
    split = json.loads((output / "split.json").read_text())
    image_folder = position_folder if use_position_folder else aligned_manifest.parent
    assert split == {
        "seed": 42, "val_fraction": 0.2,
        "maps": {
            "train": [{"game": "acc", "map_name": "Display Name"}],
            "val": [{"game": "acc", "map_name": "Display Name"}],
        },
        "train": [str(image_folder / "session-a.png")], "val": [str(image_folder / "session-b.png")],
    }
    assert aligned_manifest.read_bytes() == prepared
    history = json.loads((output / "results.json").read_text())
    assert len(history) == 2
    assert len(history[-1]["val_mae_per_axis"]) == 3
    assert all(np.isfinite(row["val_mae"]) for row in history)
    for row in history:
        buckets = row["val_errors_by_offset_length"]
        assert list(buckets) == ["<0.1", "0.1-0.25", "0.25-0.5", "0.5-1", "1-2", "2-5", ">=5"]
        assert buckets["0.5-1"]["mean_distance_error"] >= 0
        assert 0 <= buckets["0.5-1"]["mean_direction_error_deg"] <= 180
        assert buckets["0.5-1"]["count"] == buckets["0.5-1"]["direction_valid_count"] == 1
        assert all(bucket == {
            "count": 0, "mean_distance_error": None,
            "mean_direction_error_deg": None, "direction_valid_count": 0,
        } for name, bucket in buckets.items() if name != "0.5-1")
        assert row["val_by_map"] == [{
            "game": "acc", "map_name": "Display Name", "count": 1,
            "val_mae": row["val_mae"], "val_mae_per_axis": row["val_mae_per_axis"],
            "val_errors_by_offset_length": buckets,
        }]
        assert not any(key.endswith("_per_reference") for key in row)
    assert torch.load(output / "last.pt", weights_only=True)["metrics"] == history[-1]
    best_metrics = torch.load(checkpoint, weights_only=True)["metrics"]
    assert json.loads((output / "best.onnx.json").read_text())["metadata"]["metrics"] == best_metrics
    log = capsys.readouterr().out
    assert "Training maps: acc/Display Name" in log
    assert "Validation maps: acc/Display Name" in log
    assert "reference: mean distance error=" not in log
    assert log.count("actual offset length 0.5-1:") == 2
    assert "centerline" not in log
    assert log.count("mean direction error=") == 14
    assert log.count("mean distance error=n/a") == 12
    restored = PositionModel.load(checkpoint)
    assert restored.categories == {"game": ["acc"], "map_name": ["Display Name"]}
    training_references = load_position_manifest(training_input)["session-a"][0]["references"]
    np.testing.assert_allclose(restored.position_mean.numpy(), training_references[0])
    with pytest.raises(FileExistsError):
        train_position_model(aligned_manifest, output=output)


@pytest.mark.parametrize("batch", [3, 4])
@pytest.mark.parametrize("scale", [1.0, 1e-25])
@pytest.mark.parametrize("zero_targets", [False, True])
def test_position_validation_measures_distance_and_direction_with_zero_vectors_and_uneven_batches(
    aligned_manifest, tmp_path, monkeypatch, capsys, batch, scale, zero_targets,
):
    from training.image_segmentation import position_training

    predictions = torch.tensor([
        [[2, 0, 0], [1, 2, 3]],
        [[0, 1, 0], [0, 2, 0]],
        [[-1, 0, 0], [0, 2, 0]],
        [[0, 0, 0], [0, 0, 0]],
    ], dtype=torch.float32) * scale
    targets = torch.tensor([
        [[1, 0, 0], [1, 2, 3]],
        [[0, 0, 1], [0, 0, 3]],
        [[1, 0, 0], [0, 0, 0]],
        [[3, 4, 0], [0, 0, 0]],
    ], dtype=torch.float32) * scale
    if zero_targets:
        targets.zero_()

    # Supply known validation predictions through the reference inputs.
    def split(records, **kwargs):
        return {
            "train": records[:1],
            "val": [dict(records[0], references=prediction.numpy(), target=target.numpy())
                    for prediction, target in zip(predictions, targets)],
        }

    network = PositionModel(imgsz=64, categories=CATEGORIES)
    network.register_forward_hook(lambda module, inputs, output: output if module.training else inputs[1])
    monkeypatch.setattr(position_training, "PositionModel", lambda *args, **kwargs: network)
    monkeypatch.setattr(position_training, "split_position_frames", split)
    monkeypatch.setattr(position_training, "export_position_checkpoint", lambda checkpoint: checkpoint.with_suffix(".onnx"))
    output = tmp_path / "run"
    train_position_model(aligned_manifest, output=output, epochs=1, imgsz=64, batch=batch, device="cpu", upload=False)
    metrics = json.loads((output / "results.json").read_text())[0]
    buckets = metrics["val_errors_by_offset_length"]
    expected = {name: {
        "count": 0, "mean_distance_error": None,
        "mean_direction_error_deg": None, "direction_valid_count": 0,
    } for name in ("<0.1", "0.1-0.25", "0.25-0.5", "0.5-1", "1-2", "2-5", ">=5")}
    if zero_targets or scale < 0.1:
        expected["<0.1"] = {
            "count": 4,
            "mean_distance_error": scale if zero_targets else (8 + np.sqrt(2)) / 4 * scale,
            "mean_direction_error_deg": None if zero_targets else 90,
            "direction_valid_count": 0 if zero_targets else 3,
        }
    else:
        expected["1-2"] = {
            "count": 3, "mean_distance_error": (3 + np.sqrt(2)) / 3,
            "mean_direction_error_deg": 90, "direction_valid_count": 3,
        }
        expected[">=5"] = {
            "count": 1, "mean_distance_error": 5,
            "mean_direction_error_deg": None, "direction_valid_count": 0,
        }
    assert buckets.keys() == expected.keys()
    for name, bucket in expected.items():
        assert buckets[name] == pytest.approx(bucket, rel=1e-6, abs=0)
    json.dumps(metrics, allow_nan=False)
    assert torch.load(output / "best.pt", weights_only=True)["metrics"] == metrics
    log = capsys.readouterr().out
    if zero_targets:
        assert log.count("mean direction error=n/a (0/4 valid)") == 1
    elif scale < 0.1:
        assert "mean direction error=90 deg (3/4 valid)" in log
    else:
        assert "mean direction error=90 deg (3/3 valid)" in log
        assert "mean direction error=n/a (0/1 valid)" in log


@pytest.mark.parametrize("batch", [3, 14])
def test_position_validation_groups_by_actual_offset_length_at_every_boundary(
    aligned_manifest, tmp_path, monkeypatch, batch,
):
    from training.image_segmentation import position_training

    below = lambda value: float(np.nextafter(np.float32(value), np.float32(0)))
    # Each pair belongs to one range, including its lower boundary and excluding its upper boundary.
    ranges = {
        "<0.1": (0, below(0.1)),
        "0.1-0.25": (0.1, below(0.25)),
        "0.25-0.5": (0.25, below(0.5)),
        "0.5-1": (0.5, below(1)),
        "1-2": (1, below(2)),
        "2-5": (2, below(5)),
        ">=5": (5, 6),
    }
    targets = [np.array([[0, 0, length]], dtype=np.float32) for pair in ranges.values() for length in pair]

    def split(records, **kwargs):
        return {
            "train": records[:1],
            "val": [dict(records[0], references=-2 * target, target=target) for target in targets],
        }

    network = PositionModel(imgsz=64, categories=CATEGORIES)
    network.register_forward_hook(lambda module, inputs, output: output if module.training else inputs[1])
    monkeypatch.setattr(position_training, "PositionModel", lambda *args, **kwargs: network)
    monkeypatch.setattr(position_training, "split_position_frames", split)
    monkeypatch.setattr(position_training, "export_position_checkpoint", lambda checkpoint: checkpoint.with_suffix(".onnx"))
    output = tmp_path / "run"
    train_position_model(aligned_manifest, output=output, epochs=1, imgsz=64, batch=batch, device="cpu", upload=False)
    metrics = json.loads((output / "results.json").read_text())[0]
    assert list(metrics["val_errors_by_offset_length"]) == list(ranges)
    for name, lengths in ranges.items():
        assert metrics["val_errors_by_offset_length"][name] == pytest.approx({
            "count": 2, "mean_distance_error": 3 * sum(lengths) / 2,
            "mean_direction_error_deg": 180, "direction_valid_count": 1 if name == "<0.1" else 2,
        })


@pytest.mark.parametrize("batch", [2, 4, 6])
def test_position_validation_reports_each_game_and_map_separately(
    aligned_manifest, tmp_path, monkeypatch, capsys, batch,
):
    from training.image_segmentation import position_training

    # Interleave maps with unequal frame counts, including the same map name in two games.
    frames = [
        ("acc", "Display Name", [2, 0, 0], [1, 0, 0]),
        ("iracing", "Display Name", [1, 0, 0], [0, 0, 0]),
        ("acc", "Other Track", [-1, 0, 0], [1, 0, 0]),
        ("acc", "Display Name", [0, 0, 0], [1, 0, 0]),
        ("acc", "Display Name", [0, 1, 0], [1, 0, 0]),
        ("acc", "Other Track", [3, 0, 0], [1, 0, 0]),
    ]

    def split(records, **kwargs):
        return {
            "train": records[:1],
            "val": [
                dict(records[0], game=game, map_name=map_name,
                     references=np.array([prediction], dtype=np.float32),
                     target=np.array([target], dtype=np.float32))
                for game, map_name, prediction, target in frames
            ],
        }

    network = PositionModel(imgsz=64, categories=CATEGORIES)
    network.register_forward_hook(lambda module, inputs, output: output if module.training else inputs[1])
    monkeypatch.setattr(position_training, "PositionModel", lambda *args, **kwargs: network)
    monkeypatch.setattr(position_training, "split_position_frames", split)
    monkeypatch.setattr(position_training, "export_position_checkpoint", lambda checkpoint: checkpoint.with_suffix(".onnx"))
    output = tmp_path / "run"
    train_position_model(aligned_manifest, output=output, epochs=1, imgsz=64, batch=batch, device="cpu", upload=False)
    metrics = json.loads((output / "results.json").read_text())[0]
    assert metrics["val_mae"] == pytest.approx(0.5)
    assert metrics["val_errors_by_offset_length"]["1-2"] == pytest.approx({
        "count": 5, "mean_distance_error": (6 + np.sqrt(2)) / 5,
        "mean_direction_error_deg": 67.5, "direction_valid_count": 4,
    })
    expected = [
        ("acc", "Display Name", 3, [1, 1 / 3, 0], "1-2", (2 + np.sqrt(2)) / 3, 45, 2),
        ("acc", "Other Track", 2, [2, 0, 0], "1-2", 2, 90, 2),
        ("iracing", "Display Name", 1, [1, 0, 0], "<0.1", 1, None, 0),
    ]
    assert len(metrics["val_by_map"]) == len(expected)
    log = capsys.readouterr().out
    for actual, (game, map_name, count, axes, length_range, distance, direction, valid_count) in zip(
        metrics["val_by_map"], expected,
    ):
        assert (actual["game"], actual["map_name"], actual["count"]) == (game, map_name, count)
        assert actual["val_mae"] == pytest.approx(sum(axes) / 3)
        assert actual["val_mae_per_axis"] == pytest.approx(dict(zip(
            ("reference_delta_x", "reference_delta_y", "reference_delta_z"), axes,
        )))
        assert actual["val_errors_by_offset_length"][length_range] == pytest.approx({
            "count": count, "mean_distance_error": distance,
            "mean_direction_error_deg": direction, "direction_valid_count": valid_count,
        })
        assert all(bucket == {
            "count": 0, "mean_distance_error": None,
            "mean_direction_error_deg": None, "direction_valid_count": 0,
        } for name, bucket in actual["val_errors_by_offset_length"].items() if name != length_range)
        heading = f"Validation map {game}/{map_name}: {count} frames, val MAE="
        assert heading in log
        section = log.split(heading, 1)[1].split("Validation map ", 1)[0]
        angle = "n/a" if direction is None else f"{direction:.6g} deg"
        assert f"mean direction error={angle} ({valid_count}/{count} valid)" in section
        assert section.count("actual offset length ") == 7
    assert torch.load(output / "best.pt", weights_only=True)["metrics"] == metrics


@pytest.mark.parametrize("offline", [False, True])
@pytest.mark.parametrize("device_args, expected_device", [([], "0"), (["--device", "cpu"], "cpu"), (["--device", "1"], "1")])
def test_position_cli_dispatches_without_running_segmentation_training(tmp_path, monkeypatch, offline, device_args, expected_device):
    from training.image_segmentation import position_training

    calls = []
    monkeypatch.setattr(position_training, "train_position_model", lambda *args, **kwargs: calls.append((args, kwargs)))
    assert main([
        "train-position", "--manifest", str(tmp_path / "positions.json"),
        "--output", str(tmp_path / "run"), *device_args,
        "--val-fraction", "0.3", "--seed", "7",
        *(["--no-upload"] if offline else []),
    ]) == 0
    assert "model" not in calls[0][1]
    assert calls[0][1]["device"] == expected_device
    assert calls[0][1]["val_fraction"] == 0.3
    assert calls[0][1]["seed"] == 7
    assert calls[0][1]["upload"] is not offline


@pytest.fixture
def position_checkpoint(tmp_path):
    checkpoint = tmp_path / "best.pt"
    PositionModel(imgsz=64, categories=CATEGORIES).save(checkpoint, metrics={"val_mae": 0.25})
    return checkpoint


@pytest.mark.parametrize("already_exported", [False, True])
def test_position_upload_streams_onnx_and_regression_metadata(position_checkpoint, monkeypatch, already_exported):
    from training import model_publication
    from training.image_segmentation import position_publication
    from training.image_segmentation.position_export import export_position_checkpoint
    from training.image_segmentation.position_publication import upload_position_checkpoint
    from training.image_segmentation.position_data import OUTPUT_NAMES

    backend = SimpleNamespace(
        base_url="http://backend", base_port="7001",
        ensure_connection=AsyncMock(return_value=True),
        get_auth_headers=lambda: {"Authorization": "Bearer test-token"},
    )
    requests = []

    def handle(request):
        requests.append(request)
        return httpx.Response(201, json={"_id": "position-model"})

    client_type = httpx.AsyncClient
    monkeypatch.setattr(
        model_publication.httpx, "AsyncClient",
        lambda **kwargs: client_type(transport=httpx.MockTransport(handle), **kwargs),
    )
    before = position_checkpoint.read_bytes()
    artifact = position_checkpoint.with_suffix(".onnx")
    if already_exported:
        export_position_checkpoint(position_checkpoint)
        monkeypatch.setattr(position_publication, "export_position_checkpoint", lambda _: pytest.fail("ONNX must not be exported again"))
    record = asyncio.run(upload_position_checkpoint(artifact if already_exported else position_checkpoint, backend_service=backend))
    assert record == {"_id": "position-model"}
    assert len(requests) == 1
    request = requests[0]
    assert str(request.url) == "http://backend:7001/ai-model/ultralytics"
    assert request.headers["Authorization"] == "Bearer test-token"
    message = BytesParser(policy=policy.default).parsebytes(
        f"Content-Type: {request.headers['content-type']}\r\n\r\n".encode() + request.content
    )
    parts = {part.get_param("name", header="content-disposition"): part for part in message.iter_parts()}
    assert parts["file"].get_filename() == "best.onnx"
    assert parts["file"].get_payload(decode=True) == artifact.read_bytes()
    assert before == position_checkpoint.read_bytes()
    metadata = json.loads(parts["metadata"].get_payload(decode=True))
    manifest = json.loads(artifact.with_suffix(".onnx.json").read_text())
    assert metadata == {"name": "track-position", **manifest}
    inference = metadata["metadata"].pop("inference")
    assert list(inference["inputs"]) == ["images", "references", "context"]
    assert list(inference["outputs"]) == ["deltas"]
    assert metadata == {
        "name": "track-position", "task": "position",
        "classNames": list(OUTPUT_NAMES),
        "metadata": {
            "format": "acla-position-cnn-v2", "artifactFormat": "onnx", "trainingRunId": position_checkpoint.parent.name,
            "categories": CATEGORIES,
            "imgsz": 64, "outputNames": list(OUTPUT_NAMES),
            "targetDefinition": "reference_position - current_position",
            "coordinateUnits": "source_coordinates", "metrics": {"val_mae": 0.25},
        },
    }


@pytest.mark.parametrize("problem", ["missing", "empty", "extension", "outputs", "weights", "metrics", "name", "target", "units", "std"])
def test_invalid_position_upload_never_connects_to_backend(position_checkpoint, problem):
    from training.image_segmentation.position_publication import upload_position_checkpoint

    backend = SimpleNamespace(ensure_connection=AsyncMock())
    checkpoint = position_checkpoint
    name = "track-position"
    if problem == "missing":
        checkpoint = checkpoint.with_name("missing.pt")
    elif problem == "empty":
        checkpoint.write_bytes(b"")
    elif problem == "extension":
        checkpoint = checkpoint.with_suffix(".json")
    elif problem == "name":
        name = " "
    else:
        saved = torch.load(checkpoint, weights_only=True)
        key, value = {
            "outputs": ("output_names", []),
            "weights": ("state_dict", {}), "metrics": ("metrics", {"val_mae": float("nan")}),
            "target": ("target_definition", "wrong sign"), "units": ("coordinate_units", "meters"),
            "std": ("state_dict", {**saved["state_dict"], "position_std": torch.zeros(3)}),
        }[problem]
        saved[key] = value
        torch.save(saved, checkpoint)
    with pytest.raises(ValueError):
        asyncio.run(upload_position_checkpoint(checkpoint, name=name, backend_service=backend))
    backend.ensure_connection.assert_not_awaited()


@pytest.mark.parametrize("failed", [False, True])
def test_position_training_uploads_best_and_preserves_weights_on_failure(
    training_manifest, tmp_path, monkeypatch, capsys, failed,
):
    from training.image_segmentation import position_publication, position_training

    network = PositionModel(imgsz=64, categories=CATEGORIES)
    encoder_before = {key: value.clone() for key, value in network.position_encoder.state_dict().items()}
    monkeypatch.setattr(
        position_training, "PositionModel",
        lambda imgsz, categories: network,
    )
    upload = AsyncMock(return_value={"_id": "position-model"})
    if failed:
        upload.side_effect = httpx.ReadTimeout("upload timed out")
    monkeypatch.setattr(position_publication, "upload_position_checkpoint", upload)
    output = tmp_path / "run"
    args = dict(output=output, epochs=1, imgsz=64, batch=1, device="cpu")
    if failed:
        with pytest.raises(RuntimeError, match="Training completed, but backend upload failed") as error:
            train_position_model(training_manifest, **args)
        assert str(output / "best.pt") in str(error.value)
        assert "upload-position" in str(error.value)
    else:
        assert train_position_model(training_manifest, **args) == output / "best.pt"
        assert "Model uploaded to backend: position-model" in capsys.readouterr().out
    upload.assert_awaited_once_with((output / "best.onnx").resolve())
    assert any(not torch.equal(value, encoder_before[key]) for key, value in network.position_encoder.state_dict().items())
    PositionModel.load(output / "best.pt")
    assert (output / "last.pt").is_file()
    assert (output / "results.json").is_file()
    assert (output / "best.onnx").is_file()
    assert (output / "best.onnx.json").is_file()
    split = json.loads((output / "split.json").read_text())
    assert split["maps"]["train"] == split["maps"]["val"] == [
        {"game": "acc", "map_name": "Display Name"},
        {"game": "iracing", "map_name": "Other Track"},
    ]


def test_position_training_preserves_checkpoint_and_skips_upload_when_export_fails(
    training_manifest, tmp_path, monkeypatch,
):
    from training.image_segmentation import position_publication, position_training

    network = PositionModel(imgsz=64, categories=CATEGORIES)
    monkeypatch.setattr(position_training, "PositionModel", lambda *args, **kwargs: network)

    def fail_export(checkpoint):
        raise RuntimeError("unsupported export operator")

    monkeypatch.setattr(position_training, "export_position_checkpoint", fail_export)
    upload = AsyncMock()
    monkeypatch.setattr(position_publication, "upload_position_checkpoint", upload)
    output = tmp_path / "run"
    with pytest.raises(RuntimeError, match="Training completed, but ONNX export failed") as error:
        train_position_model(training_manifest, output=output,
                             epochs=1, imgsz=64, batch=1, device="cpu")
    assert "upload-position" in str(error.value)
    PositionModel.load(output / "best.pt")
    assert (output / "last.pt").is_file()
    assert not (output / "best.onnx").exists()
    upload.assert_not_awaited()


@pytest.mark.parametrize("failed", [False, True])
@pytest.mark.parametrize("command, suffix, name_flag", [
    (["upload-position"], ".pt", "--name"),
    (["upload-position"], ".onnx", "--name"),
    (["train-position", "--upload-model"], ".onnx", None),
])
def test_position_upload_command_can_retry_without_training(tmp_path, monkeypatch, capsys, failed, command, suffix, name_flag):
    from training.image_segmentation import position_publication, position_training

    def no_training(*args, **kwargs):
        pytest.fail("Uploading a saved checkpoint must not train again")

    monkeypatch.setattr(position_training, "train_position_model", no_training)
    upload = AsyncMock(return_value={"_id": "position-model"})
    if failed:
        upload.side_effect = ConnectionError("offline")
    monkeypatch.setattr(position_publication, "upload_position_checkpoint", upload)
    checkpoint = tmp_path / f"best{suffix}"
    args = [*command, str(checkpoint), *([name_flag, "race-position"] if name_flag else [])]
    if failed:
        with pytest.raises(SystemExit) as error:
            main(args)
        assert error.value.code == 2
        assert "offline" in capsys.readouterr().err
    else:
        assert main(args) == 0
        assert "position-model" in capsys.readouterr().out
    upload.assert_awaited_once_with(checkpoint, name="race-position" if name_flag else "track-position")


@pytest.mark.parametrize("args, message", [
    ([], "--manifest, --output"),
    (["--manifest", "positions.json"], "--output"),
    (["--output", "run"], "--manifest"),
    (["--upload-model", "best.onnx", "--no-upload"], "not allowed with argument"),
])
def test_position_cli_requires_training_inputs_or_upload_model(args, message, capsys):
    with pytest.raises(SystemExit) as error:
        main(["train-position", *args])
    assert error.value.code == 2
    assert message in capsys.readouterr().err


@pytest.mark.parametrize("fraction", [0.2, 0.3, 0.8])
def test_training_split_keeps_sessions_together_and_balances_frames_per_map(fraction):
    from training.image_segmentation.position_training import split_position_frames

    records = [
        {
            "image": Path(f"map-{track}/session-{session}/frame-{frame}.png"),
            "session_id": f"map-{track}-session-{session}",
            "game": "acc", "map_name": f"map-{track}",
        }
        for track in range(10) for session in range(10) for frame in range((track + 1) * (session + 1))
    ]
    original = list(records)
    split = split_position_frames(records, val_fraction=fraction)
    train_maps = {(row["game"], row["map_name"]) for row in split["train"]}
    val_maps = {(row["game"], row["map_name"]) for row in split["val"]}
    assert train_maps == val_maps == {("acc", f"map-{track}") for track in range(10)}
    for track in range(10):
        total = 55 * (track + 1)
        val_count = sum(row["map_name"] == f"map-{track}" for row in split["val"])
        assert abs(val_count - total * fraction) <= (track + 1) / 2
        assert sum(row["map_name"] == f"map-{track}" for row in split["train"]) == total - val_count
    train_images = {row["image"] for row in split["train"]}
    val_images = {row["image"] for row in split["val"]}
    assert train_images.isdisjoint(val_images)
    assert train_images | val_images == {row["image"] for row in records}
    assert len(split["train"]) + len(split["val"]) == len(records)
    assert {row["session_id"] for row in split["train"]}.isdisjoint(row["session_id"] for row in split["val"])
    assert split["train"] != sorted(split["train"], key=lambda row: str(row["image"]))
    assert split_position_frames(list(reversed(records)), val_fraction=fraction) == split
    assert {row["image"] for row in split_position_frames(records, val_fraction=fraction, seed=7)["val"]} != val_images
    assert records == original


@pytest.mark.parametrize("fraction", [0.01, 0.2, 0.99])
@pytest.mark.parametrize("maps", [
    [("acc", "map-a")],
    [("acc", "map-a"), ("iracing", "map-a")],
    [("acc", "map-a"), ("acc", "map-b")],
])
def test_two_sessions_per_map_always_put_each_map_in_both_sets(fraction, maps):
    from training.image_segmentation.position_training import split_position_frames

    split = split_position_frames([
        {
            "image": Path(f"{game}/{map_name}/{index}.png"), "game": game, "map_name": map_name,
            "session_id": f"{game}/{map_name}/{index}",
        }
        for game, map_name in maps for index in range(2)
    ], val_fraction=fraction)
    assert len(split["train"]) == len(split["val"]) == len(maps)
    assert {(row["game"], row["map_name"]) for row in split["train"]} == set(maps)
    assert {(row["game"], row["map_name"]) for row in split["val"]} == set(maps)
    assert {row["image"] for row in split["train"]}.isdisjoint(row["image"] for row in split["val"])


@pytest.mark.parametrize("sizes", [(80, 5, 5, 5, 5), (65, 15, 10, 10), (60, 40), (9, 8, 7, 6)])
@pytest.mark.parametrize("fraction", [0.01, 0.2, 0.49, 0.8, 0.99])
def test_session_split_chooses_closest_possible_frame_count(sizes, fraction):
    from itertools import combinations
    from training.image_segmentation.position_training import split_position_frames

    records = [
        {
            "image": Path(f"{session}/{frame}.png"), "session_id": str(session),
            "game": "acc", "map_name": "map-a",
        }
        for session, size in enumerate(sizes) for frame in range(size)
    ]
    target = sum(sizes) * fraction
    best_error = min(
        abs(sum(subset) - target)
        for count in range(1, len(sizes)) for subset in combinations(sizes, count)
    )
    for seed in (0, 7, 42):
        split = split_position_frames(records, val_fraction=fraction, seed=seed)
        assert abs(len(split["val"]) - target) == pytest.approx(best_error)
        assert {row["session_id"] for row in split["train"]}.isdisjoint(row["session_id"] for row in split["val"])
        assert len(split["train"]) + len(split["val"]) == len(records)


def test_session_split_randomizes_equally_sized_session_choices():
    from training.image_segmentation.position_training import split_position_frames

    records = [
        {
            "image": Path(f"{session}/{frame}.png"), "session_id": str(session),
            "game": "acc", "map_name": "map-a",
        }
        for session in range(5) for frame in range(10)
    ]
    selections = {
        frozenset(row["session_id"] for row in split_position_frames(records, seed=seed)["val"])
        for seed in range(10)
    }
    assert len(selections) > 1
    assert all(len(selection) == 1 for selection in selections)


def test_session_spanning_maps_is_rejected_instead_of_leaking():
    from training.image_segmentation.position_training import split_position_frames

    records = [
        {
            "image": Path(f"{map_name}/{session}.png"), "session_id": str(session),
            "game": "acc", "map_name": map_name,
        }
        for map_name in ("a", "b") for session in range(2)
    ]
    with pytest.raises(ValueError, match="multiple game/map"):
        split_position_frames(records)


@pytest.mark.parametrize("fraction", [-0.1, 0, 1, 1.1, float("nan"), float("inf")])
def test_invalid_training_split_fraction_is_rejected(aligned_manifest, tmp_path, fraction):
    with pytest.raises(ValueError, match="Validation fraction"):
        train_position_model(
            aligned_manifest, output=tmp_path / "run", val_fraction=fraction,
        )
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize("single_map", [False, True])
def test_map_with_only_one_session_cannot_be_split_for_training(aligned_manifest, tmp_path, single_map):
    manifest = json.loads(aligned_manifest.read_text())
    if single_map:
        position = aligned_manifest.parent / "session-b.position.json"
        frame = json.loads(position.read_text())
        frame.update(session_id="session-a", game="acc", map_name="Display Name")
        position.write_text(json.dumps(frame))
        manifest["sessions"][0]["frames"].extend(manifest["sessions"].pop()["frames"])
        aligned_manifest.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="at least two sessions per map"):
        train_position_model(
            aligned_manifest, output=tmp_path / "run",
            epochs=1, imgsz=64, device="cpu", upload=False,
        )
    assert not (tmp_path / "run").exists()
