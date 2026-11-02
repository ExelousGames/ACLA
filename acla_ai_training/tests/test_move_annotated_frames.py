from __future__ import annotations

import json
import sys

import pytest

from scripts import prepare_position_dataset as script


@pytest.fixture
def downloads(tmp_path, monkeypatch):
    root = tmp_path / "session_downloads" / "session_downloaded_by_prepare_position_dataset"
    monkeypatch.setattr(script, "SESSION_DOWNLOAD_DIR", root)
    return root


def annotated_frame(folder, *, position=True, name="frame_000001.jpg"):
    folder.mkdir(parents=True, exist_ok=True)
    image = folder / name
    image.write_bytes(b"frame image")
    annotation = image.with_suffix(".json")
    annotation.write_text(json.dumps({"imagePath": image.name, "shapes": []}))
    files = [image, annotation]
    if position:
        sidecar = image.with_suffix(".position.json")
        sidecar.write_text(json.dumps({"image": image.name, "target": [[1, 2, 3], [4, 5, 6]]}))
        files.append(sidecar)
    return files


def test_copies_only_annotated_frames_preserving_sessions_videos_and_position_files(downloads):
    originals = {}
    for session, video, position in [("a", 0, True), ("a", 1, False), ("b", 0, True)]:
        folder = downloads / (session * 24) / "frames" / f"video_{video:03d}"
        originals.update({path: path.read_bytes() for path in annotated_frame(folder, position=position)})
    remaining = {
        folder / "frame_000002.jpg": b"unannotated image",
        folder / "frame_000002.position.json": b'{"image": "frame_000002.jpg"}',
        folder.parent / "manifest.json": b'{"session": {}}',
        folder.parent.parent / "telemetry.json": b"[]",
        folder.parent.parent / "video_000.mp4": b"video",
    }
    for path, data in remaining.items():
        path.write_bytes(data)

    result = script.move_annotated_frames()

    destination = downloads.parent / "session image segmentation samples"
    assert result == {
        "frames": 3, "sessions": [{"id": "a" * 24, "frames": 2}, {"id": "b" * 24, "frames": 1}],
        "directory": str(destination), "duplicated_frames": 0,
        "duplicated_directories": [],
    }
    for source, data in originals.items():
        target = destination / source.relative_to(downloads)
        assert target.read_bytes() == data
        assert source.read_bytes() == data
    assert {path: path.read_bytes() for path in remaining} == remaining
    for number in (1, 2):
        result = script.move_annotated_frames()
        assert result["frames"] == result["duplicated_frames"] == 3
        duplicated_directories = set()
        for source, data in originals.items():
            target = destination / source.relative_to(downloads)
            duplicated = target.parent.with_name(f"{target.parent.name}_{number}")
            duplicated_directories.add(str(duplicated))
            assert (duplicated / source.name).read_bytes() == data
            assert target.read_bytes() == source.read_bytes() == data
        assert set(result["duplicated_directories"]) == duplicated_directories
    assert len(list(destination.rglob("*.jpg"))) == 9
    assert not (downloads.parent / "duplicated").exists()

    new_files = annotated_frame(downloads / ("c" * 24) / "frames/video_000")
    result = script.move_annotated_frames()
    assert result["frames"] == 4
    assert result["duplicated_frames"] == 3
    for source in new_files:
        assert (destination / source.relative_to(downloads)).read_bytes() == source.read_bytes()


@pytest.mark.parametrize("suffix", [".jpg", ".json", ".position.json"])
def test_destination_conflicts_copy_complete_frame_to_numbered_folder(downloads, suffix):
    originals = {}
    for session in ("a", "b"):
        folder = downloads / (session * 24) / "frames/video_000"
        originals.update({path: path.read_bytes() for path in annotated_frame(folder)})
    destination = downloads.parent / "session image segmentation samples"
    conflict = destination / ("b" * 24) / "frames/video_000" / f"frame_000001{suffix}"
    conflict.parent.mkdir(parents=True)
    conflict.write_bytes(b"existing sample")

    result = script.move_annotated_frames()

    assert result["frames"] == 2
    assert result["duplicated_frames"] == 1
    assert {path: path.read_bytes() for path in originals} == originals
    assert conflict.read_bytes() == b"existing sample"
    assert list(conflict.parent.iterdir()) == [conflict]
    for source, data in originals.items():
        target = destination / source.relative_to(downloads)
        if target.parent == conflict.parent:
            target = target.parent.with_name("video_000_1") / target.name
        assert target.read_bytes() == data


def test_numbered_folder_keeps_frames_together_and_skips_existing_names(downloads):
    folder = downloads / ("a" * 24) / "frames/video_000"
    files = annotated_frame(folder)
    script.move_annotated_frames()
    files += annotated_frame(folder, name="frame_000002.jpg")
    destination = downloads.parent / "session image segmentation samples"
    video = destination / folder.relative_to(downloads)
    video.with_name("video_000_1").write_bytes(b"existing file")
    video.with_name("video_000_2").mkdir()
    video.with_name("video_000_3").symlink_to("missing-folder", target_is_directory=True)
    originals = {path: path.read_bytes() for path in downloads.parent.rglob("*") if path.is_file()}

    result = script.move_annotated_frames()

    duplicated = video.with_name("video_000_4")
    assert result["frames"] == result["duplicated_frames"] == 2
    assert result["duplicated_directories"] == [str(duplicated)]
    for source in files:
        assert (duplicated / source.name).read_bytes() == source.read_bytes()
    assert not (video / "frame_000002.jpg").exists()
    assert {path: path.read_bytes() for path in originals} == originals
    assert video.with_name("video_000_3").is_symlink()


@pytest.mark.parametrize("invalid", ["missing", "outside", "absolute", "mismatch", "shapes", "json"])
def test_invalid_annotations_leave_all_source_files_in_place(downloads, invalid):
    annotated_frame(downloads / ("a" * 24) / "frames/video_000")
    image, annotation, _ = annotated_frame(downloads / ("b" * 24) / "frames/video_000")
    data = json.loads(annotation.read_text())
    if invalid == "missing":
        image.unlink()
    elif invalid == "outside":
        data["imagePath"] = "../frame_000001.jpg"
    elif invalid == "absolute":
        data["imagePath"] = str(image)
    elif invalid == "mismatch":
        data["imagePath"] = "frame_000002.jpg"
    elif invalid == "shapes":
        data["shapes"] = None
    annotation.write_text("{" if invalid == "json" else json.dumps(data))
    originals = {path: path.read_bytes() for path in downloads.rglob("*") if path.is_file()}

    with pytest.raises((ValueError, FileNotFoundError)):
        script.move_annotated_frames()

    assert {path: path.read_bytes() for path in downloads.rglob("*") if path.is_file()} == originals
    assert not (downloads.parent / "session image segmentation samples").exists()


@pytest.mark.parametrize("duplicate", [False, True])
def test_failed_copy_removes_partial_output_and_keeps_originals(downloads, monkeypatch, duplicate):
    files = annotated_frame(downloads / ("a" * 24) / "frames/video_000")
    destination = downloads.parent / "session image segmentation samples"
    if duplicate:
        script.move_annotated_frames()
        destination = (destination / files[0].parent.relative_to(downloads)).with_name("video_000_1")
    existing = {path: path.read_bytes() for path in downloads.parent.rglob("*") if path.is_file()}
    originals = {path: path.read_bytes() for path in files}
    copy = script.shutil.copy2

    def fail_position_copy(source, target):
        if source == files[2]:
            target.write_bytes(b"partial copy")
            raise OSError("Cannot copy position file")
        return copy(source, target)

    monkeypatch.setattr(script.shutil, "copy2", fail_position_copy)
    with pytest.raises(OSError, match="Cannot copy position file"):
        script.move_annotated_frames()

    assert {path: path.read_bytes() for path in originals} == originals
    assert {path: path.read_bytes() for path in downloads.parent.rglob("*") if path.is_file()} == existing
    assert not any(path.is_file() for path in destination.rglob("*"))


@pytest.mark.parametrize("choice", ["2", "Copy annotated images"])
def test_menu_copies_without_backend_or_download_prompts(downloads, monkeypatch, capsys, choice):
    files = annotated_frame(downloads / ("a" * 24) / "frames/video_000")
    monkeypatch.setitem(sys.modules, "training.integrations.backend", None)
    answers = iter([choice])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))

    assert script.main([]) == 0

    output = capsys.readouterr().out
    assert '"frames": 1' in output
    assert "session image segmentation samples" in output
    assert all(path.exists() for path in files)
    assert "Download sessions" in output
    assert "Copy annotated images" in output


def test_empty_downloads_need_no_backend_and_create_no_output(downloads, monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "training.integrations.backend", None)
    monkeypatch.setattr("builtins.input", lambda _: "2")
    assert script.main([]) == 0
    assert '"frames": 0' in capsys.readouterr().out
    assert not downloads.exists()


def test_main_menu_exit_needs_no_backend(downloads, monkeypatch):
    monkeypatch.setitem(sys.modules, "training.integrations.backend", None)
    monkeypatch.setattr("builtins.input", lambda _: "3")
    assert script.main([]) == 0
    assert not downloads.exists()


@pytest.mark.parametrize("options", [
    ["--sessions", "a" * 24], ["--fps", "2"], ["--max-delta-ms", "100"],
    ["--video-offset", f"{'a' * 24}=0"], ["--move-annotated"], ["--help"],
])
def test_cli_options_are_rejected_before_touching_files(downloads, options, capsys):
    assert script.main(options) == 2
    assert "Run without command-line options" in capsys.readouterr().err
    assert not downloads.exists()
