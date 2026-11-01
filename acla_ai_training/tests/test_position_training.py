"""Full-frame and density-filtered training, temporal history, and metric denominators."""

import json

import numpy as np
from PIL import Image
import pytest
import torch

from training.image_segmentation import position_training
from training.image_segmentation.position_model import PositionModel
from training.image_segmentation.position_model_v2 import PositionModelV2


@pytest.fixture(autouse=True)
def offline_resnet(monkeypatch):
    from training.image_segmentation import position_model_v2

    encoder = position_model_v2.resnet18
    monkeypatch.setattr(position_model_v2, "resnet18", lambda **kwargs: encoder(weights=None))


@pytest.mark.parametrize("sequence_length", [1, 4])
@pytest.mark.parametrize("model_class", [PositionModel, PositionModelV2])
@pytest.mark.parametrize("density_radius", [None, 0.5])
def test_training_retains_all_frames_history_and_correct_metric_denominators(tmp_path, monkeypatch, sequence_length, model_class, density_radius):
    sessions = {}
    for session in ("a", "b"):
        folder = tmp_path / session
        folder.mkdir()
        rows = []
        for index in range(5):
            row = {
                "image": folder / f"frame_{index:06d}.png", "session_id": session,
                "game": "acc", "map_name": "Monza", "normalized_position": 0,
                "target": np.zeros((1, 3), dtype=np.float32), "orientation": np.zeros(3),
                "references": np.asarray([[index, 0, 0]], dtype=np.float32),
                "driver_position": [0 if index < 3 else 10, 0, 0],
            }
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
            device="cpu", upload=False, model_class=model_class,
            density_position_radius=density_radius,
        )
    finally:
        torch.set_num_threads(previous_threads)
    train, val = [loader.dataset for loader in loaders]
    assert len(train) == (5 if density_radius is None else 2)
    assert len(val) == 5
    original = train if density_radius is None else train.dataset
    selected = list(range(len(train))) if density_radius is None else train.indices
    assert {row["session_id"] for row in original.records}.isdisjoint(row["session_id"] for row in val.records)
    assert {row["image"] for row in original.records + val.records} == {
        row["image"] for rows in sessions.values() for row in rows
    }
    for index, endpoint in enumerate(selected):
        current = original.records[endpoint]
        number = int(current["image"].stem.split("_")[-1])
        if sequence_length > 1:
            history = list(range(max(0, number - 3), number + 1))
            assert train[index]["references"][:, 0, 0].tolist() == [history[0]] * (4 - len(history)) + history
    split = json.loads((output / "split.json").read_text())
    assert split["train"] == [str(original.records[index]["image"]) for index in selected]
    assert set(split["val"]) == {str(row["image"]) for row in val.records}
    if density_radius is None:
        assert "density_sampling" not in split
        assert "train_history" not in split
    else:
        assert split["density_sampling"] == {
            "position_radius": density_radius, "orientation_degrees": 5.0,
            "candidate_count": 5, "selected_count": 2,
        }
        if sequence_length > 1:
            assert split["train_history"] == [str(row["image"]) for row in original.records]
            assert any(set(original.windows[index]) - set(selected) for index in selected)
        else:
            assert "train_history" not in split
    metrics = json.loads((output / "results.json").read_text())[0]
    assert metrics["train_loss"] == pytest.approx(sum(loss for loss, _ in totals) / len(train))
    assert metrics["train_mae"] == pytest.approx(sum(error for _, error in totals) / (len(train) * 3))
    assert metrics["val_by_map"][0]["count"] == 5
    torch.testing.assert_close(PositionModel.load(checkpoint).position_mean, torch.tensor([2., 0., 0.]))
