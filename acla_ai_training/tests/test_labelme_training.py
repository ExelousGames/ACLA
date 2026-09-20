from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import yaml
from PIL import Image

from training.image_segmentation import WORKSPACE_DIR
from training.image_segmentation.__main__ import main
from training.image_segmentation.splits import prepare_training_dataset


def _sample(source: Path, relative: str) -> Path:
    path = source / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 64)).save(path.with_suffix(".png"))
    path.write_text(json.dumps({
        "imagePath": path.with_suffix(".png").name, "imageWidth": 64, "imageHeight": 64,
        "shapes": [{
            "label": "track", "shape_type": "polygon", "points": [[0, 0], [64, 0], [32, 64]],
        }],
    }))
    return path


@pytest.fixture
def samples(tmp_path):
    source = tmp_path / "storage/annotation_images"
    for folder in ("recording_a/nested", "recording_b"):
        for index in range(5):
            _sample(source, f"{folder}/frame_{index}.json")
    return source, tmp_path / "storage/image_segmentation/split.json"


def _assert_export(data, source, manifest):
    config = yaml.safe_load(data.read_text())
    for split in ("train", "val"):
        folder = data.parent / config[split]
        exported = {p.relative_to(folder).with_suffix(".json").as_posix() for p in folder.rglob("*.png")}
        assert exported == set(manifest[split])
        for sample in manifest[split]:
            relative = Path(sample)
            image = relative.with_suffix(".png")
            assert (folder / image).read_bytes() == (source / image).read_bytes()
            assert (data.parent / "labels" / split / relative.with_suffix(".txt")).is_file()


def test_recursive_split_mixes_images_and_exports_only_saved_samples(samples):
    source, split_file = samples
    (source / "metadata.json").write_text('{"fps": 30}')
    Image.new("RGB", (64, 64)).save(source / "unannotated.png")

    data = prepare_training_dataset(source, split_file=split_file)

    manifest = json.loads(split_file.read_text())
    assert manifest["source"] == "../annotation_images"
    assert len(manifest["train"]) == 8
    assert len(manifest["val"]) == 2
    assert not set(manifest["train"]) & set(manifest["val"])
    assert set(manifest["train"] + manifest["val"]) == {
        path.relative_to(source).as_posix() for path in source.rglob("frame_*.json")
    }
    # Individual frames from a recording can appear in both sets.
    assert {Path(p).parent for p in manifest["val"]} <= {Path(p).parent for p in manifest["train"]}
    _assert_export(data, source, manifest)


def test_saved_split_is_reused_but_annotation_edits_are_exported(samples):
    source, split_file = samples
    first = prepare_training_dataset(source, split_file=split_file)
    saved = split_file.read_bytes()
    manifest = json.loads(saved)
    edited = source / manifest["train"][0]
    annotation = json.loads(edited.read_text())
    annotation["shapes"] = []
    edited.write_text(json.dumps(annotation))
    _sample(source, "new_recording/frame.json")

    second = prepare_training_dataset(split_file=split_file, seed=100, val_fraction=0.5)

    assert split_file.read_bytes() == saved
    assert first != second
    label = Path("labels/train") / Path(manifest["train"][0]).with_suffix(".txt")
    assert (first.parent / label).read_text()
    assert (second.parent / label).read_text() == ""
    _assert_export(second, source, manifest)


def test_rebuild_uses_saved_source_and_includes_new_annotations(samples):
    source, split_file = samples
    prepare_training_dataset(source, split_file=split_file)
    _sample(source, "new/frame.json")

    data = prepare_training_dataset(split_file=split_file, rebuild_split=True, val_fraction=0.3)

    manifest = json.loads(split_file.read_text())
    assert len(manifest["train"]) == 8
    assert len(manifest["val"]) == 3
    assert "new/frame.json" in manifest["train"] + manifest["val"]
    _assert_export(data, source, manifest)


def test_new_splits_are_repeatable_and_paths_survive_workspace_move(samples, tmp_path):
    source, split_file = samples
    prepare_training_dataset(source, split_file=split_file, seed=7, val_fraction=0.4)
    another = split_file.with_name("another.json")
    prepare_training_dataset(source, split_file=another, seed=7, val_fraction=0.4)
    assert another.read_bytes() == split_file.read_bytes()
    moved = tmp_path / "moved_storage"
    shutil.move(str(source.parent), moved)
    relocated_split = moved / "image_segmentation/split.json"

    data = prepare_training_dataset(split_file=relocated_split)

    _assert_export(data, moved / "annotation_images", json.loads(relocated_split.read_text()))


@pytest.mark.parametrize("fraction", [0, 1, -0.1, 1.1, float("nan")])
def test_invalid_fraction_does_not_write_a_split(samples, fraction):
    source, split_file = samples
    with pytest.raises(ValueError, match="Validation fraction"):
        prepare_training_dataset(source, split_file=split_file, val_fraction=fraction)
    assert not split_file.parent.exists()


@pytest.mark.parametrize("count", [0, 1, 2])
def test_small_datasets_keep_both_splits_nonempty(tmp_path, count):
    source = tmp_path / "annotations"
    source.mkdir()
    for index in range(count):
        _sample(source, f"frame_{index}.json")
    split_file = tmp_path / "storage/split.json"
    if count < 2:
        with pytest.raises(ValueError, match="at least two annotated images"):
            prepare_training_dataset(source, split_file=split_file)
        assert not split_file.exists()
    else:
        prepare_training_dataset(source, split_file=split_file)
        manifest = json.loads(split_file.read_text())
        assert len(manifest["train"]) == len(manifest["val"]) == 1


def test_missing_saved_sample_fails_without_changing_split(samples):
    source, split_file = samples
    prepare_training_dataset(source, split_file=split_file)
    saved = split_file.read_bytes()
    manifest = json.loads(saved)
    (source / manifest["val"][0]).with_suffix(".png").unlink()
    exports = list(split_file.parent.glob("dataset_*"))

    with pytest.raises(ValueError, match="No such file"):
        prepare_training_dataset(split_file=split_file)

    assert split_file.read_bytes() == saved
    assert list(split_file.parent.glob("dataset_*")) == exports


def test_split_cannot_be_silently_reused_for_another_source(samples, tmp_path):
    source, split_file = samples
    prepare_training_dataset(source, split_file=split_file)
    with pytest.raises(ValueError, match="Saved split belongs to"):
        prepare_training_dataset(tmp_path / "other", split_file=split_file)


@pytest.mark.parametrize("problem", ["empty", "overlap", "absolute", "traversal"])
def test_invalid_saved_lists_fail_before_exporting(samples, problem):
    source, split_file = samples
    prepare_training_dataset(source, split_file=split_file)
    manifest = json.loads(split_file.read_text())
    manifest["val"] = {
        "empty": [], "overlap": [manifest["train"][0]],
        "absolute": [str(source / manifest["val"][0])], "traversal": ["../frame.json"],
    }[problem]
    split_file.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        prepare_training_dataset(split_file=split_file)


def test_duplicate_image_references_do_not_save_a_split(samples):
    source, split_file = samples
    first = next(source.rglob("frame_*.json"))
    first.with_name("duplicate.json").write_bytes(first.read_bytes())
    with pytest.raises(ValueError, match="Image occurs more than once"):
        prepare_training_dataset(source, split_file=split_file)
    assert not split_file.exists()


def test_training_uses_dataset_from_saved_assignments(samples, monkeypatch):
    source, split_file = samples
    prepare_training_dataset(source, split_file=split_file)
    manifest = json.loads(split_file.read_text())
    # A reviewed list is authoritative even if images exist outside that list.
    manifest["train"] = manifest["train"][:2]
    split_file.write_text(json.dumps(manifest))
    model = MagicMock(task="segment")
    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=lambda *a, **kw: model))

    assert main([
        "train-labelme", "--no-upload", "--split-file", str(split_file), "--epochs", "1", "--device", "0",
    ]) == 0

    options = model.train.call_args.kwargs
    assert options["epochs"] == 1 and options["device"] == "0"
    _assert_export(Path(options["data"]), source, manifest)


def test_launcher_prepares_from_any_working_directory(samples, tmp_path):
    source, split_file = samples
    result = subprocess.run([
        sys.executable, str(WORKSPACE_DIR / "scripts/train_labelme.py"),
        str(source), "--split-file", str(split_file), "--prepare-only",
    ], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "8 training, 2 validation" in result.stdout
    assert split_file.is_file()
