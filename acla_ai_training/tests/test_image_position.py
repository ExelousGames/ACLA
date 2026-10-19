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
from training.image_segmentation.position_training import train_position_model


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
    assert set(a) == {"image", "references", "context", "target"}
    torch.testing.assert_close(a["image"], b["image"])
    torch.testing.assert_close(a["references"], b["references"])
    torch.testing.assert_close(a["context"], b["context"])
    torch.testing.assert_close(a["context"], torch.tensor([0.25, 1, 0, 1, 0]))
    assert not torch.equal(a["target"], b["target"])
    np.testing.assert_allclose(before["session-a"][0]["target"], [[0.1, -0.2, 0], [0.15, -0.2, 0]], atol=1e-7)
    assert list(inspect.signature(PositionModel.forward).parameters) == ["self", "images", "references", "context"]
    assert "driver_position" not in inspect.signature(PositionModel.predict_position).parameters


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
    assert prediction.shape == (1, 2, 3)
    loss = (prediction - sample["target"]).square().mean()
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
        records["session-b"][0]["image"], 0.25, curve, curve, game="iracing", map_name="Other Track",
    )
    assert set(prediction) == {
        "reference_delta", "centerline_delta",
    }
    assert prediction == model.predict_position(
        records["session-b"][0]["image"], 0.25, curve, curve, game="iracing", map_name="Other Track",
    )
    with pytest.raises(ValueError, match="Unknown map_name"):
        restored.predict_position(tmp_path / "missing.png", 0.25, curve, curve, game="acc", map_name="Unknown")


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
                torch.zeros(2, 3, image_size, image_size), torch.zeros(2, 2, 3),
                torch.tensor([[0.25, 1, 0, 1, 0]]).repeat(2, 1),
            )
        assert prediction.shape == (2, 2, 3)
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
            prediction = model(torch.zeros(1, 3, 64, 64), torch.zeros(1, 2, 3), context)
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


def test_segmentation_based_checkpoint_requires_training_from_scratch(position_checkpoint):
    saved = torch.load(position_checkpoint, weights_only=True)
    saved["state_dict"] = {
        key.replace("image_encoder.", "yolo."): value for key, value in saved["state_dict"].items()
    }
    torch.save(saved, position_checkpoint)
    with pytest.raises(ValueError, match="standalone image encoder; retrain from scratch"):
        PositionModel.load(position_checkpoint)


@pytest.mark.parametrize("use_position_folder", [False, True])
def test_single_session_training_saves_raw_axis_metrics_and_uses_only_train_reference_statistics(
    aligned_manifest, tmp_path, position_folder, use_position_folder,
):
    path = aligned_manifest.parent / "session-b.position.json"
    frame = json.loads(path.read_text())
    frame["session_id"] = "session-a"
    frame["references"] = [[0.3, -0.4, 0.5], [0.35, -0.4, 0.5]]
    frame["target"] = [[0.2, -0.5, 0.4], [0.25, -0.5, 0.4]]
    path.write_text(json.dumps(frame))
    if use_position_folder:
        (position_folder / path.name).write_bytes(path.read_bytes())
    manifest = json.loads(aligned_manifest.read_text())
    manifest["sessions"][0]["frames"].extend(manifest["sessions"].pop()["frames"])
    aligned_manifest.write_text(json.dumps(manifest))
    training_input = position_folder if use_position_folder else aligned_manifest
    assert len(load_position_manifest(training_input)) == 1
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
        "train": [str(image_folder / "session-a.png")], "val": [str(image_folder / "session-b.png")],
    }
    assert aligned_manifest.read_bytes() == prepared
    history = json.loads((output / "results.json").read_text())
    assert len(history) == 2
    assert len(history[-1]["val_mae_per_axis"]) == 6
    assert all(np.isfinite(row["val_mae"]) for row in history)
    restored = PositionModel.load(checkpoint)
    assert restored.categories == CATEGORIES
    training_references = load_position_manifest(training_input)["session-a"][0]["references"]
    np.testing.assert_allclose(restored.position_mean.numpy(), training_references.reshape(6))
    with pytest.raises(FileExistsError):
        train_position_model(aligned_manifest, output=output)


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
        "--upload-name", "race-position", *(["--no-upload"] if offline else []),
    ]) == 0
    assert "model" not in calls[0][1]
    assert calls[0][1]["device"] == expected_device
    assert calls[0][1]["val_fraction"] == 0.3
    assert calls[0][1]["seed"] == 7
    assert calls[0][1]["upload"] is not offline
    assert calls[0][1]["upload_name"] == "race-position"


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
            "format": "acla-position-cnn-v1", "artifactFormat": "onnx", "trainingRunId": position_checkpoint.parent.name,
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
            "std": ("state_dict", {**saved["state_dict"], "position_std": torch.zeros(6)}),
        }[problem]
        saved[key] = value
        torch.save(saved, checkpoint)
    with pytest.raises(ValueError):
        asyncio.run(upload_position_checkpoint(checkpoint, name=name, backend_service=backend))
    backend.ensure_connection.assert_not_awaited()


@pytest.mark.parametrize("failed", [False, True])
def test_position_training_uploads_best_and_preserves_weights_on_failure(
    aligned_manifest, tmp_path, monkeypatch, capsys, failed,
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
            train_position_model(aligned_manifest, **args)
        assert str(output / "best.pt") in str(error.value)
        assert "upload-position" in str(error.value)
    else:
        assert train_position_model(aligned_manifest, **args) == output / "best.pt"
        assert "Model uploaded to backend: position-model" in capsys.readouterr().out
    upload.assert_awaited_once_with((output / "best.onnx").resolve(), name="track-position")
    assert any(not torch.equal(value, encoder_before[key]) for key, value in network.position_encoder.state_dict().items())
    PositionModel.load(output / "best.pt")
    assert (output / "last.pt").is_file()
    assert (output / "results.json").is_file()
    assert (output / "best.onnx").is_file()
    assert (output / "best.onnx.json").is_file()


def test_position_training_preserves_checkpoint_and_skips_upload_when_export_fails(
    aligned_manifest, tmp_path, monkeypatch,
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
        train_position_model(aligned_manifest, output=output,
                             epochs=1, imgsz=64, batch=1, device="cpu")
    assert "upload-position" in str(error.value)
    PositionModel.load(output / "best.pt")
    assert (output / "last.pt").is_file()
    assert not (output / "best.onnx").exists()
    upload.assert_not_awaited()


@pytest.mark.parametrize("failed", [False, True])
def test_position_upload_command_can_retry_without_training(tmp_path, monkeypatch, capsys, failed):
    from training.image_segmentation import position_publication, position_training

    def no_training(*args, **kwargs):
        pytest.fail("Uploading a saved checkpoint must not train again")

    monkeypatch.setattr(position_training, "train_position_model", no_training)
    upload = AsyncMock(return_value={"_id": "position-model"})
    if failed:
        upload.side_effect = ConnectionError("offline")
    monkeypatch.setattr(position_publication, "upload_position_checkpoint", upload)
    checkpoint = tmp_path / "best.pt"
    args = ["upload-position", str(checkpoint), "--name", "race-position"]
    if failed:
        with pytest.raises(SystemExit) as error:
            main(args)
        assert error.value.code == 2
        assert "offline" in capsys.readouterr().err
    else:
        assert main(args) == 0
        assert "position-model" in capsys.readouterr().out
    upload.assert_awaited_once_with(checkpoint, name="race-position")


def test_training_split_randomizes_frames_reproducibly_across_sessions():
    from training.image_segmentation.position_training import split_position_frames

    records = [
        {"image": Path(f"session-{session}/frame-{frame}.png"), "session_id": session}
        for session in range(10) for frame in range(session + 1)
    ]
    original = list(records)
    split = split_position_frames(records)
    assert len(split["train"]) == 44
    assert len(split["val"]) == 11
    train_images = {row["image"] for row in split["train"]}
    val_images = {row["image"] for row in split["val"]}
    assert train_images.isdisjoint(val_images)
    assert train_images | val_images == {row["image"] for row in records}
    assert {row["session_id"] for row in split["train"]} & {row["session_id"] for row in split["val"]}
    assert split_position_frames(list(reversed(records))) == split
    assert split_position_frames(records, seed=7) != split
    assert len(split_position_frames(records, val_fraction=0.3)["val"]) == round(len(records) * 0.3)
    assert records == original


@pytest.mark.parametrize("fraction", [0.01, 0.2, 0.99])
def test_two_frames_always_produce_nonempty_training_and_validation(fraction):
    from training.image_segmentation.position_training import split_position_frames

    split = split_position_frames([{"image": Path("a.png")}, {"image": Path("b.png")}], val_fraction=fraction)
    assert len(split["train"]) == len(split["val"]) == 1
    assert split["train"][0]["image"] != split["val"][0]["image"]


@pytest.mark.parametrize("fraction", [-0.1, 0, 1, 1.1, float("nan"), float("inf")])
def test_invalid_training_split_fraction_is_rejected(aligned_manifest, tmp_path, fraction):
    with pytest.raises(ValueError, match="Validation fraction"):
        train_position_model(
            aligned_manifest, output=tmp_path / "run", val_fraction=fraction,
        )
    assert not (tmp_path / "run").exists()


def test_single_frame_cannot_be_split_for_training(aligned_manifest, tmp_path):
    manifest = json.loads(aligned_manifest.read_text())
    manifest["sessions"] = manifest["sessions"][:1]
    aligned_manifest.write_text(json.dumps(manifest))
    assert len(load_position_manifest(aligned_manifest)) == 1
    with pytest.raises(ValueError, match="at least two frames"):
        train_position_model(aligned_manifest, output=tmp_path / "run")
    assert not (tmp_path / "run").exists()
