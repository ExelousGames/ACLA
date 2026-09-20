from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import yaml
from PIL import Image

from training.image_segmentation import DEFAULT_LABELS, PACKAGE_DIR, WORKSPACE_DIR, read_labels
from training.image_segmentation.__main__ import main
from training.image_segmentation.dataset import prepare_dataset
from training.image_segmentation.trainer import train_model


def _annotation(folder: Path, *, labels=("track",), name="frame") -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (100, 50)).save(folder / f"{name}.png")
    path = folder / f"{name}.json"
    path.write_text(json.dumps({
        "imagePath": f"{name}.png", "imageWidth": 100, "imageHeight": 50,
        "shapes": [
            {"label": label, "shape_type": "polygon", "points": [[0, 0], [100, 0], [50, 50]]}
            for label in labels
        ],
    }))
    return path


def test_preparation_preserves_classes_polygons_and_nested_names(tmp_path):
    labels = read_labels()
    _annotation(tmp_path / "train/session_a", labels=labels)
    _annotation(tmp_path / "train/session_b", labels=("car", "car"))
    _annotation(tmp_path / "val/session_c", labels=())
    Image.new("RGB", (100, 50)).save(tmp_path / "train/unreviewed.png")
    original = (tmp_path / "train/session_a/frame.json").read_bytes()

    data = prepare_dataset(tmp_path / "train", tmp_path / "val", tmp_path / "output")

    assert yaml.safe_load(data.read_text()) == {
        "train": "images/train", "val": "images/val", "names": labels,
    }
    rows = (data.parent / "labels/train/session_a/frame.txt").read_text().splitlines()
    assert len(rows) == len(labels)
    for index, row in enumerate(rows):
        assert [float(value) for value in row.split()] == [index, 0, 0, 1, 0, 0.5, 1]
    assert len((data.parent / "labels/train/session_b/frame.txt").read_text().splitlines()) == 2
    assert (data.parent / "labels/val/session_c/frame.txt").read_text() == ""
    assert not (data.parent / "images/train/unreviewed.png").exists()
    assert (data.parent / "images/train/session_a/frame.png").read_bytes() == (
        tmp_path / "train/session_a/frame.png"
    ).read_bytes()
    assert (tmp_path / "train/session_a/frame.json").read_bytes() == original


@pytest.mark.parametrize(("change", "message"), [
    ({"label": "gravel"}, "Unknown label"),
    ({"shape_type": "rectangle"}, "Only polygon"),
    ({"points": [[0, 0], [10, 10]]}, "at least three"),
    ({"points": [[0, 0], [10, 10], [20, 20]]}, "nonzero area"),
    ({"points": [[0, 0], [101, 0], [50, 50]]}, "inside the image"),
    ({"points": [[0, 0], [float("nan"), 0], [50, 50]]}, "finite numbers"),
])
def test_bad_polygons_fail_before_writing_output(tmp_path, change, message):
    _annotation(tmp_path / "train")
    path = _annotation(tmp_path / "val")
    annotation = json.loads(path.read_text())
    annotation["shapes"][0].update(change)
    path.write_text(json.dumps(annotation))

    with pytest.raises(ValueError, match=message):
        prepare_dataset(tmp_path / "train", tmp_path / "val", tmp_path / "output")
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("problem", ["missing_image", "wrong_size", "duplicate_image"])
def test_invalid_image_references_are_rejected(tmp_path, problem):
    _annotation(tmp_path / "train")
    path = _annotation(tmp_path / "val")
    annotation = json.loads(path.read_text())
    if problem == "missing_image":
        annotation["imagePath"] = "missing.png"
    elif problem == "wrong_size":
        annotation["imageWidth"] = 200
    else:
        annotation["imagePath"] = "../train/frame.png"
    path.write_text(json.dumps(annotation))

    with pytest.raises(ValueError, match=str(path)):
        prepare_dataset(tmp_path / "train", tmp_path / "val", tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_custom_labels_and_external_relative_image_path(tmp_path):
    path = _annotation(tmp_path / "train", labels=("barrier",))
    _annotation(tmp_path / "val", labels=("barrier",))
    images = tmp_path / "images"
    images.mkdir()
    (path.parent / "frame.png").rename(images / "frame.png")
    annotation = json.loads(path.read_text())
    annotation["imagePath"] = "../images/frame.png"
    path.write_text(json.dumps(annotation))
    labels = tmp_path / "labels.txt"
    labels.write_text("track\nbarrier\n")

    data = prepare_dataset(tmp_path / "train", tmp_path / "val", tmp_path / "out", labels_file=labels)

    assert (data.parent / "labels/train/frame.txt").read_text().startswith("1 ")
    assert yaml.safe_load(data.read_text())["names"] == ["track", "barrier"]


def test_refuses_missing_split_and_existing_output(tmp_path):
    _annotation(tmp_path / "train")
    with pytest.raises(ValueError, match="No Labelme JSON annotations in val"):
        prepare_dataset(tmp_path / "train", tmp_path / "val", tmp_path / "out")
    _annotation(tmp_path / "val")
    data = prepare_dataset(tmp_path / "train", tmp_path / "val", tmp_path / "out")
    before = data.read_bytes()
    with pytest.raises(FileExistsError):
        prepare_dataset(tmp_path / "train", tmp_path / "val", tmp_path / "out")
    assert data.read_bytes() == before


def test_desktop_launcher_uses_same_python_and_shared_labels(tmp_path, monkeypatch):
    from training.image_segmentation import __main__ as cli

    launch = MagicMock(return_value=0)
    monkeypatch.setenv("DISPLAY", ":0")
    monkeypatch.setattr(cli.importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(cli.subprocess, "call", launch)

    assert main(["annotate", str(tmp_path)]) == 0
    command = launch.call_args.args[0]
    assert command[:3] == [sys.executable, str(PACKAGE_DIR / "labelme_editor.py"), str(tmp_path)]
    assert command[command.index("--labels") + 1] == str(DEFAULT_LABELS)
    config = yaml.safe_load(Path(command[command.index("--config") + 1]).read_text())
    assert config["validate_label"] == "exact"
    assert config["with_image_data"] is False


@pytest.mark.parametrize("with_images", [False, True])
def test_headless_launcher_opens_browser_with_optional_images(tmp_path, monkeypatch, with_images):
    from training.image_segmentation import __main__ as cli
    from training.image_segmentation import browser

    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    monkeypatch.setattr(cli.importlib.util, "find_spec", lambda name: object())
    launch = MagicMock(return_value=7)
    monkeypatch.setattr(browser, "run_browser", launch)
    images = [str(tmp_path)] if with_images else []

    assert main(["annotate", *images]) == 7
    command = launch.call_args.args[0]
    assert command[:2] == [sys.executable, str(PACKAGE_DIR / "labelme_editor.py")]
    assert command[2:2 + len(images)] == images
    assert command[2 + len(images)] == "--labels"


def test_browser_flag_overrides_desktop_display(monkeypatch):
    from training.image_segmentation import __main__ as cli
    from training.image_segmentation import browser

    monkeypatch.setenv("DISPLAY", ":0")
    monkeypatch.setattr(cli.importlib.util, "find_spec", lambda name: object())
    launch = MagicMock(return_value=0)
    monkeypatch.setattr(browser, "run_browser", launch)

    assert main(["annotate", "--browser"]) == 0
    launch.assert_called_once()


@pytest.mark.parametrize("plugin_dir", ["/fake/cv2/qt/plugins", "/custom/qt/plugins"])
def test_labelme_child_removes_only_opencv_qt_paths(monkeypatch, plugin_dir):
    from training.image_segmentation import labelme_editor

    monkeypatch.setitem(sys.modules, "cv2", SimpleNamespace(__file__="/fake/cv2/__init__.py"))
    monkeypatch.setattr(labelme_editor.importlib.util, "find_spec", lambda name: object())
    monkeypatch.setenv("QT_QPA_PLATFORM_PLUGIN_PATH", plugin_dir)
    monkeypatch.setenv("QT_QPA_FONTDIR", "/fake/cv2/qt/fonts")

    def start_labelme(*args, **kwargs):
        assert "QT_QPA_FONTDIR" not in labelme_editor.os.environ
        expected = plugin_dir if plugin_dir.startswith("/custom") else None
        assert labelme_editor.os.environ.get("QT_QPA_PLATFORM_PLUGIN_PATH") == expected

    launch = MagicMock(side_effect=start_labelme)
    labelme_cli = SimpleNamespace(MainWindow=object, main=launch)
    monkeypatch.setitem(sys.modules, "labelme", SimpleNamespace(__main__=labelme_cli))
    labelme_editor.main()
    launch.assert_called_once_with()


def test_train_cli_passes_dataset_and_device_to_segmentation_model(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    data = tmp_path / "data.yaml"
    data.write_text("names: [track]\n")
    model = MagicMock(task="segment")
    factory = MagicMock(return_value=model)
    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=factory))

    assert main([
        "train", "--no-upload", "--data", str(data), "--epochs", "2", "--imgsz", "128",
        "--batch", "2", "--device", "0", "--project", str(tmp_path / "runs"),
    ]) == 0

    factory.assert_called_once_with(
        str(WORKSPACE_DIR / "storage/image_segmentation/pretrained/yolo11n-seg.pt"), task="segment",
    )
    model.train.assert_called_once_with(
        data=str(data), epochs=2, imgsz=128, batch=2, device="0", workers=0,
        project=str(tmp_path / "runs"), name="train", overlap_mask=False,
    )


def test_detection_checkpoint_cannot_silently_train_boxes(tmp_path, monkeypatch):
    data = tmp_path / "data.yaml"
    data.touch()
    model = MagicMock(task="detect")
    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=lambda *a, **kw: model))

    with pytest.raises(ValueError, match="segmentation checkpoint"):
        train_model(data, model="yolo11n.pt")
    model.train.assert_not_called()
