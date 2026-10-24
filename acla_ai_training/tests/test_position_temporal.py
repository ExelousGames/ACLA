"""Causal position history, training isolation, and the exported temporal contract."""

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest
import torch
from PIL import Image

from training.image_segmentation.__main__ import main
from training.image_segmentation import position_data, position_training
from training.image_segmentation.position_data import PositionCurve, PositionDataset, TARGET_DEFINITION
from training.image_segmentation.position_export import export_position_checkpoint
from training.image_segmentation.position_model import PositionModel


CATEGORIES = {"game": ["acc"], "map_name": ["Monza"]}


@pytest.fixture(autouse=True)
def single_threaded_torch():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(9)
        yield
    torch.set_num_threads(previous)


def make_record(folder, number, session="session-a"):
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"frame_{number}.png"
    Image.new("RGB", (32, 32), (number * 10, 30, 100)).save(path)
    return {
        "session_id": session, "image": path, "game": "acc", "map_name": "Monza",
        "normalized_position": number / 20,
        "references": np.array([[number, number + 1, number + 2]], dtype=np.float32),
        "target": np.array([[number / 10, -0.2, 0.1]], dtype=np.float32),
    }


def model_inputs(batch, length, imgsz=32):
    return (
        torch.rand(batch, length, 3, imgsz, imgsz),
        torch.rand(batch, length, 1, 3),
        torch.cat((torch.rand(batch, length, 1), torch.ones(batch, length, 2)), dim=-1),
    )


def test_windows_use_numeric_frame_order_and_only_past_inputs(tmp_path):
    records = [make_record(tmp_path, number) for number in (10, 2, 0, 1)]
    dataset = PositionDataset(records, 32, CATEGORIES, sequence_length=4)
    for index, expected in enumerate(([0, 1, 2, 10], [0, 0, 1, 2], [0, 0, 0, 0], [0, 0, 0, 1])):
        sample = dataset[index]
        assert sample["references"][:, 0, 0].tolist() == expected
        assert sample["image"].shape == (4, 3, 32, 32)
        assert sample["context"].shape == (4, 3)
        torch.testing.assert_close(sample["target"], torch.from_numpy(records[index]["target"]))
        assert sample["normalized_position"] == records[index]["normalized_position"]

    before = dataset[1]
    # No future frame or past target can become an input to the current prediction.
    records[0]["references"][:] = 1e6
    for row in records:
        row["target"][:] = 1e6
        row["driver_position"] = [1e6, 1e6, 1e6]
    after = dataset[1]
    for field in ("image", "references", "context"):
        torch.testing.assert_close(before[field], after[field])


def test_windows_reset_at_session_and_recording_boundaries_and_never_cross_splits(tmp_path):
    records = [
        make_record(tmp_path / session / video, number, session)
        for session in ("a", "b") for video in ("video_0", "video_1") for number in (0, 1, 2)
    ]
    splits = position_training.split_position_frames(records)
    seen = {}
    for split, rows in splits.items():
        dataset = PositionDataset(rows, 32, CATEGORIES, sequence_length=4)
        seen[split] = {row["session_id"] for row in rows}
        for index, window in dataset.windows.items():
            current = rows[index]
            assert all(rows[i]["session_id"] == current["session_id"] for i in window)
            assert all(rows[i]["image"].parent == current["image"].parent for i in window)
            if current["image"].stem == "frame_0":
                assert window == [index] * 4
    assert seen["train"].isdisjoint(seen["val"])
    # Even when sessions share a directory, their histories remain separate.
    shared = [make_record(tmp_path, 0, "a"), make_record(tmp_path, 1, "b")]
    dataset = PositionDataset(shared, 32, CATEGORIES, sequence_length=4)
    assert dataset.windows == {0: [0] * 4, 1: [1] * 4}


def test_padding_reuses_the_same_augmented_frame(tmp_path, monkeypatch):
    row = make_record(tmp_path, 0)
    calls = []

    def augment(image):
        calls.append(image)
        return Image.new("RGB", image.size, "red")

    monkeypatch.setattr(position_data, "hud_coarse_dropout", augment)
    sample = PositionDataset([row], 32, CATEGORIES, sequence_length=4, augment=True)[0]
    assert len(calls) == 1
    for image in sample["image"]:
        torch.testing.assert_close(image, sample["image"][0])
    PositionDataset([row], 32, CATEGORIES, sequence_length=4)[0]
    assert len(calls) == 1


@pytest.mark.parametrize("names", [("a", "b"), ("frame_1", "frame_01"), ("one_1", "two_2")])
def test_ambiguous_frame_order_is_rejected(tmp_path, names):
    records = [make_record(tmp_path, number) for number in (0, 1)]
    for row, name in zip(records, names):
        row["image"] = tmp_path / f"{name}.png"
    with pytest.raises(ValueError, match="Temporal training requires"):
        PositionDataset(records, 32, CATEGORIES, sequence_length=4)


@pytest.mark.parametrize("length", [0, -1, True, 1.5])
def test_invalid_sequence_length_is_rejected(length):
    for construct in (
        lambda: PositionModel(32, categories=CATEGORIES, sequence_length=length),
        lambda: PositionDataset([], 32, CATEGORIES, sequence_length=length),
    ):
        with pytest.raises(ValueError, match="Sequence length"):
            construct()


def test_history_affects_predictions_and_receives_gradients_without_retained_state(tmp_path):
    model = PositionModel(32, categories=CATEGORIES, sequence_length=4)
    inputs = model_inputs(2, 4)
    inputs[0].requires_grad_()
    expected = model(*inputs)
    assert expected.shape == (2, 1, 3)
    for field in range(3):
        changed = [value.detach().clone() for value in inputs]
        changed[field][:, :-1] = 0
        assert not torch.allclose(expected, model(*changed), rtol=1e-6, atol=1e-7)
    reordered = [torch.cat((value[:, :3].flip(1), value[:, 3:]), dim=1) for value in inputs]
    assert not torch.allclose(expected, model(*reordered), rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(model(*inputs), expected)
    expected.square().mean().backward()
    assert inputs[0].grad[:, 0].abs().sum() > 0
    for parameter in model.temporal_encoder.parameters():
        assert torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum() > 0
    checkpoint = tmp_path / "best.pt"
    model.save(checkpoint)
    loaded = PositionModel.load(checkpoint)
    assert loaded.sequence_length == 4
    torch.testing.assert_close(loaded(*inputs), expected)


@pytest.mark.parametrize("field", [0, 1, 2])
def test_temporal_model_requires_aligned_history_shapes(field):
    model = PositionModel(32, categories=CATEGORIES, sequence_length=4)
    inputs = list(model_inputs(1, 4))
    inputs[field] = inputs[field][:, :-1]
    with pytest.raises(ValueError, match="Temporal"):
        model(*inputs)


def test_legacy_checkpoint_without_sequence_length_still_loads(tmp_path):
    checkpoint = tmp_path / "old.pt"
    model = PositionModel(32, categories=CATEGORIES)
    model.save(checkpoint)
    saved = torch.load(checkpoint, weights_only=True)
    del saved["sequence_length"]
    torch.save(saved, checkpoint)
    loaded = PositionModel.load(checkpoint)
    inputs = tuple(value[:, 0] for value in model_inputs(2, 1))
    torch.testing.assert_close(model(*inputs), loaded(*inputs))


def test_prediction_helper_uses_previous_images_and_lap_positions(tmp_path):
    records = [make_record(tmp_path, number) for number in range(6)]
    model = PositionModel(32, categories=CATEGORIES, sequence_length=4)
    curve = PositionCurve([0, 1], [[0, 1, 2], [20, 21, 22]])
    dataset = PositionDataset(records, 32, CATEGORIES, sequence_length=4)
    for index in range(len(records)):
        sample = dataset[index]
        expected = model(*(sample[field].unsqueeze(0) for field in ("image", "references", "context")))
        result = model.predict_position(
            records[index]["image"], records[index]["normalized_position"], curve,
            game="acc", map_name="Monza",
            history=[(row["image"], row["normalized_position"]) for row in records[:index]],
        )
        np.testing.assert_allclose(result["reference_delta"], expected.detach().numpy()[0, 0], atol=1e-6)


@pytest.mark.parametrize("length,imgsz", [(2, 32), (4, 96)])
def test_temporal_onnx_matches_pytorch_for_different_histories(tmp_path, length, imgsz):
    model = PositionModel(imgsz, categories=CATEGORIES, sequence_length=length).eval()
    checkpoint = tmp_path / "best.pt"
    model.save(checkpoint)
    artifact = export_position_checkpoint(checkpoint)
    contract = json.loads(artifact.with_suffix(".onnx.json").read_text())["metadata"]["inference"]
    assert contract["schemaVersion"] == 5
    assert contract["image"]["layout"] == "NTCHW"
    assert contract["temporal"]["sequenceLength"] == length
    assert contract["temporal"]["padding"] == "repeat-first"
    assert contract["temporal"]["state"] == "reset-per-window"
    assert contract["inputs"] == {
        "images": {"dtype": "float32", "shape": [1, length, 3, imgsz, imgsz]},
        "references": {"dtype": "float32", "shape": [1, length, 1, 3]},
        "context": {"dtype": "float32", "shape": [1, length, 3]},
    }
    assert contract["outputs"] == {"deltas": {"dtype": "float32", "shape": [1, 1, 3]}}
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(artifact), sess_options=options, providers=["CPUExecutionProvider"])
    inputs = model_inputs(1, length, imgsz)
    for field in range(3):
        changed = [value.clone() for value in inputs]
        changed[field][:, :-1] = 0
        actual = session.run(None, dict(zip(("images", "references", "context"), [value.numpy() for value in changed])))
        np.testing.assert_allclose(actual[0], model(*changed).detach().numpy(), rtol=1e-4, atol=1e-5)


def test_default_training_updates_temporal_weights_and_counts_each_final_frame_once(tmp_path, monkeypatch):
    records = [make_record(tmp_path / session, number, session) for session in ("a", "b") for number in range(5)]
    for row in records:
        row["image"].with_suffix(".position.json").write_text(json.dumps({
            **{key: value for key, value in row.items() if key not in ("image", "references", "target")},
            "image": row["image"].name, "references": row["references"].tolist(), "target": row["target"].tolist(),
            "coordinate_frame": "car_local", "target_definition": TARGET_DEFINITION,
        }))
    captured = {}

    def create_model(*args, **kwargs):
        model = PositionModel(*args, **kwargs)
        captured["before"] = model.temporal_encoder.weight_hh.detach().clone()
        return model

    monkeypatch.setattr(position_training, "PositionModel", create_model)
    output = tmp_path / "run"
    checkpoint = position_training.train_position_model(tmp_path, output=output, epochs=1, imgsz=32, batch=3, device="cpu", upload=False)
    model = PositionModel.load(checkpoint)
    assert model.sequence_length == 4
    assert not torch.equal(captured["before"], model.temporal_encoder.weight_hh)
    split = json.loads((output / "split.json").read_text())
    assert split["sequence_length"] == 4
    assert len(split["train"]) == len(split["val"]) == 5
    metrics = json.loads((output / "results.json").read_text())[0]
    assert np.isfinite(metrics["train_loss"]) and np.isfinite(metrics["train_mae"])
    assert metrics["val_by_map"][0]["count"] == 5
    assert sum(section["count"] for section in metrics["val_by_map"][0]["val_errors_by_normalized_position"]) == 5
    assert (output / "best.onnx").is_file()


@pytest.mark.parametrize("arguments,length", [([], 4), (["--sequence-length", "2"], 2), (["--sequence-length", "1"], 1)])
def test_cli_forwards_temporal_window(arguments, length, monkeypatch):
    captured = {}

    def train(manifest, **kwargs):
        captured.update(kwargs)
        return Path("run/best.pt")

    monkeypatch.setattr(position_training, "train_position_model", train)
    assert main(["train-position", "--manifest", "data", "--output", "run", "--no-upload", *arguments]) == 0
    assert captured["sequence_length"] == length
