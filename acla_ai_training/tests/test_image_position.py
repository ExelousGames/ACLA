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
    PositionCurve, PositionDataset, load_position_manifest,
)
from training.image_segmentation.position_model import YOLOPositionModel
from training.image_segmentation.position_preparation import signed_targets
from training.image_segmentation.position_training import train_position_model


@pytest.fixture
def aligned_manifest(tmp_path):
    sessions = []
    for session_id, color in (("session-a", "red"), ("session-b", "blue")):
        Image.new("RGB", (80, 40), color).save(tmp_path / f"{session_id}.png")
        position = tmp_path / f"{session_id}.position.json"
        position.write_text(json.dumps({
            "session_id": session_id,
            "image": f"{session_id}.png",
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
    a, b = (PositionDataset(records["session-a"], 64)[0] for records in (before, after))
    assert set(a) == {"image", "references", "target"}
    torch.testing.assert_close(a["image"], b["image"])
    torch.testing.assert_close(a["references"], b["references"])
    assert not torch.equal(a["target"], b["target"])
    np.testing.assert_allclose(before["session-a"][0]["target"], [[0.1, -0.2, 0], [0.15, -0.2, 0]], atol=1e-7)
    assert list(inspect.signature(YOLOPositionModel.forward).parameters) == ["self", "images", "references"]
    assert "driver_position" not in inspect.signature(YOLOPositionModel.predict_position).parameters


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


@pytest.fixture
def tiny_yolo():
    from ultralytics import YOLO

    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield YOLO("yolo26n-seg.yaml").model
    torch.set_num_threads(previous)


def test_regression_trains_without_mutating_yolo_and_roundtrips(tiny_yolo, aligned_manifest, tmp_path):
    model = YOLOPositionModel(tiny_yolo, imgsz=64)
    before = {key: value.clone() for key, value in model.yolo.state_dict().items()}
    records = load_position_manifest(aligned_manifest)
    sample = PositionDataset(records["session-a"], 64)[0]
    model.train()
    optimizer = torch.optim.Adam(model.head.parameters(), lr=0.01)
    prediction = model(sample["image"].unsqueeze(0), sample["references"].unsqueeze(0))
    assert prediction.shape == (1, 2, 3)
    loss = (prediction - sample["target"]).square().mean()
    loss.backward()
    assert model.head[0].weight.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in model.yolo.parameters())
    optimizer.step()
    for key, value in model.yolo.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)
    model.eval()
    with torch.no_grad():
        expected = model(sample["image"].unsqueeze(0), sample["references"].unsqueeze(0))
    checkpoint = tmp_path / "position.pt"
    model.save(checkpoint)
    restored = YOLOPositionModel.load(checkpoint)
    with torch.no_grad():
        actual = restored(sample["image"].unsqueeze(0), sample["references"].unsqueeze(0))
    torch.testing.assert_close(actual, expected)
    curve = PositionCurve([0, 0.5], [[0, 0, 0], [1, 2, 3]])
    assert set(restored.predict_position(records["session-b"][0]["image"], 0.25, curve, curve)) == {
        "reference_delta", "centerline_delta",
    }


@pytest.mark.parametrize("use_position_folder", [False, True])
def test_single_session_training_saves_raw_axis_metrics_and_uses_only_train_reference_statistics(
    tiny_yolo, aligned_manifest, tmp_path, position_folder, use_position_folder,
):
    from ultralytics import YOLO

    # Exercise the real saved YOLO -> regression training path without downloads.
    segmentation = YOLO("yolo26n-seg.yaml")
    segmentation.model = tiny_yolo
    source = tmp_path / "segment.pt"
    segmentation.save(source)
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
        training_input, model=source, output=output,
        epochs=2, imgsz=64, batch=1, upload=False,
    )
    assert checkpoint == output / "best.pt"
    assert (output / "last.pt").is_file()
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
    restored = YOLOPositionModel.load(checkpoint)
    training_references = load_position_manifest(training_input)["session-a"][0]["references"]
    np.testing.assert_allclose(restored.position_mean.numpy(), training_references.reshape(6))
    with pytest.raises(FileExistsError):
        train_position_model(aligned_manifest, model=tmp_path / "segment.pt", output=output)


@pytest.mark.parametrize("offline", [False, True])
def test_position_cli_dispatches_without_running_segmentation_training(tmp_path, monkeypatch, offline):
    from training.image_segmentation import position_training

    calls = []
    monkeypatch.setattr(position_training, "train_position_model", lambda *args, **kwargs: calls.append((args, kwargs)))
    assert main([
        "train-position", "--manifest", str(tmp_path / "positions.json"),
        "--model", str(tmp_path / "best.pt"), "--output", str(tmp_path / "run"), "--device", "0",
        "--val-fraction", "0.3", "--seed", "7",
        "--upload-name", "race-position", *(["--no-upload"] if offline else []),
    ]) == 0
    assert calls[0][1]["device"] == "0"
    assert calls[0][1]["val_fraction"] == 0.3
    assert calls[0][1]["seed"] == 7
    assert calls[0][1]["upload"] is not offline
    assert calls[0][1]["upload_name"] == "race-position"


@pytest.fixture
def position_checkpoint(tiny_yolo, tmp_path):
    checkpoint = tmp_path / "best.pt"
    YOLOPositionModel(tiny_yolo, imgsz=64).save(checkpoint, metrics={"val_mae": 0.25})
    return checkpoint


def test_position_upload_streams_original_weights_and_regression_metadata(position_checkpoint, monkeypatch):
    from training import model_publication
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
    record = asyncio.run(upload_position_checkpoint(position_checkpoint, backend_service=backend))
    assert record == {"_id": "position-model"}
    assert len(requests) == 1
    request = requests[0]
    assert str(request.url) == "http://backend:7001/ai-model/ultralytics"
    assert request.headers["Authorization"] == "Bearer test-token"
    message = BytesParser(policy=policy.default).parsebytes(
        f"Content-Type: {request.headers['content-type']}\r\n\r\n".encode() + request.content
    )
    parts = {part.get_param("name", header="content-disposition"): part for part in message.iter_parts()}
    assert parts["file"].get_filename() == "best.pt"
    assert parts["file"].get_payload(decode=True) == before == position_checkpoint.read_bytes()
    metadata = json.loads(parts["metadata"].get_payload(decode=True))
    assert metadata == {
        "name": "track-position", "task": "position",
        "classNames": list(YOLOPositionModel.load(position_checkpoint).yolo.names.values()),
        "metadata": {
            "format": "acla-yolo-position-v1", "trainingRunId": position_checkpoint.parent.name,
            "imgsz": 64, "outputNames": list(OUTPUT_NAMES),
            "targetDefinition": "reference_position - current_position",
            "coordinateUnits": "source_coordinates", "metrics": {"val_mae": 0.25},
        },
    }


@pytest.mark.parametrize("problem", ["missing", "empty", "extension", "format", "outputs", "weights", "metrics", "name"])
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
            "format": ("format", "segmentation"), "outputs": ("output_names", []),
            "weights": ("state_dict", {}), "metrics": ("metrics", {"val_mae": float("nan")}),
        }[problem]
        saved[key] = value
        torch.save(saved, checkpoint)
    with pytest.raises(ValueError):
        asyncio.run(upload_position_checkpoint(checkpoint, name=name, backend_service=backend))
    backend.ensure_connection.assert_not_awaited()


@pytest.mark.parametrize("failed", [False, True])
def test_position_training_uploads_best_and_preserves_weights_on_failure(
    tiny_yolo, aligned_manifest, tmp_path, monkeypatch, capsys, failed,
):
    from training.image_segmentation import position_publication

    monkeypatch.setattr(
        YOLOPositionModel, "from_segmentation_checkpoint",
        lambda checkpoint, imgsz: YOLOPositionModel(tiny_yolo, imgsz),
    )
    upload = AsyncMock(return_value={"_id": "position-model"})
    if failed:
        upload.side_effect = httpx.ReadTimeout("upload timed out")
    monkeypatch.setattr(position_publication, "upload_position_checkpoint", upload)
    output = tmp_path / "run"
    args = dict(model=tmp_path / "segment.pt", output=output, epochs=1, imgsz=64, batch=1)
    if failed:
        with pytest.raises(RuntimeError, match="Training completed, but backend upload failed") as error:
            train_position_model(aligned_manifest, **args)
        assert str(output / "best.pt") in str(error.value)
        assert "upload-position" in str(error.value)
    else:
        assert train_position_model(aligned_manifest, **args) == output / "best.pt"
        assert "Model uploaded to backend: position-model" in capsys.readouterr().out
    upload.assert_awaited_once_with(output / "best.pt", name="track-position")
    YOLOPositionModel.load(output / "best.pt")
    assert (output / "last.pt").is_file()
    assert (output / "results.json").is_file()


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
            aligned_manifest, model=tmp_path / "missing.pt", output=tmp_path / "run", val_fraction=fraction,
        )
    assert not (tmp_path / "run").exists()


def test_single_frame_cannot_be_split_for_training(aligned_manifest, tmp_path):
    manifest = json.loads(aligned_manifest.read_text())
    manifest["sessions"] = manifest["sessions"][:1]
    aligned_manifest.write_text(json.dumps(manifest))
    assert len(load_position_manifest(aligned_manifest)) == 1
    with pytest.raises(ValueError, match="at least two frames"):
        train_position_model(aligned_manifest, model=tmp_path / "missing.pt", output=tmp_path / "run")
    assert not (tmp_path / "run").exists()
