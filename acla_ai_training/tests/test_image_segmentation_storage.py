from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from training.image_segmentation import WORKSPACE_DIR, splits, trainer
from training.image_segmentation.__main__ import main


@pytest.mark.parametrize("command", ["train", "train-labelme"])
def test_default_models_and_runs_stay_in_storage_from_any_cwd(tmp_path, monkeypatch, command):
    monkeypatch.chdir(tmp_path)
    data = tmp_path / "data.yaml"
    data.write_text("names: [track]\n")
    network = MagicMock(task="segment")
    factory = MagicMock(return_value=network)
    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=factory))
    monkeypatch.setattr(splits, "prepare_training_dataset", lambda *args, **kwargs: data)
    args = [command, "--no-upload"]
    if command == "train":
        args += ["--data", str(data)]

    assert main(args) == 0

    storage = WORKSPACE_DIR / "storage/image_segmentation"
    factory.assert_called_once_with(str(storage / "pretrained/yolo11n-seg.pt"), task="segment")
    assert network.train.call_args.kwargs["project"] == str(storage / "runs")
    assert not list(tmp_path.glob("*.pt"))


@pytest.mark.parametrize("model", ["yolo11s-seg.pt", "custom.pt", "checkpoints/custom.pt", "yolo11n-seg.yaml"])
def test_downloads_use_storage_and_explicit_local_models_are_preserved(tmp_path, monkeypatch, model):
    monkeypatch.chdir(tmp_path)
    data = tmp_path / "data.yaml"
    data.write_text("names: [track]\n")
    if model == "custom.pt":
        (tmp_path / model).write_bytes(b"custom checkpoint")
    network = MagicMock(task="segment")
    factory = MagicMock(return_value=network)
    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=factory))

    trainer.train_model(data, model=model, upload=False)

    expected = (
        str(WORKSPACE_DIR / "storage/image_segmentation/pretrained" / model)
        if model == "yolo11s-seg.pt" else model
    )
    factory.assert_called_once_with(expected, task="segment")
    assert network.train.call_args.kwargs["project"] == str(
        WORKSPACE_DIR / "storage/image_segmentation/runs"
    )
