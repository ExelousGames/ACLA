import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from scripts import prepare_position_dataset as preparation


SESSION_ID = "0123456789abcdef01234567"


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def bundle(tmp_path):
    folder = tmp_path / SESSION_ID
    folder.mkdir()
    path = folder / "video_000.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 4.0, (32, 32))
    assert writer.isOpened()
    try:
        for frame_index in range(4):
            writer.write(np.full((32, 32, 3), frame_index * 50, dtype=np.uint8))
    finally:
        writer.release()
    write_json(folder / "manifest.json", {
        "session": {"sessionId": SESSION_ID, "chunkCount": 1, "dataPoints": 8},
        "telemetry_file": "telemetry.json", "videos": [{
            "index": 0, "name": "recording.mp4", "file": path.name,
            "size": path.stat().st_size, "mimeType": "video/mp4", "startedAt": 1700000000000,
        }],
    })
    write_json(folder / "telemetry.json", [
        {
            "Graphics_analyst_session_time": time,
            "Graphics_current_time": (index % 4) * 250,
            "Physics_speed_kmh": index * 10,
            "Graphics_car_coordinates": {"0": [index, 1, 2]},
        }
        for index, time in enumerate([125, 125.24, 125.5, 125.75, 126, 126.25, 126.5, 126.75])
    ])
    return folder


@pytest.mark.parametrize("include_simulator_clock", [False, True])
def test_pairs_session_video_frames_in_memory_without_saving_telemetry(bundle, include_simulator_clock):
    if include_simulator_clock:
        rows = json.loads((bundle / "telemetry.json").read_text())
        for index, row in enumerate(rows):
            row["Graphics_session_time"] = 500 + index
        write_json(bundle / "telemetry.json", rows)
    result = preparation.extract_session_frames(bundle)
    output = Path(result["directory"])
    rows = json.loads((bundle / "telemetry.json").read_text())
    assert result["videos"][0]["matched_frames"] == 4
    assert result["videos"][0]["offset_seconds"] == 0
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["session"]["sessionId"] == SESSION_ID
    assert manifest["telemetry_time_field"] == "Graphics_analyst_session_time"
    assert manifest["first_telemetry_time_seconds"] == 125
    assert "frames" not in manifest
    assert {path.suffix for path in (output / "video_000").iterdir()} == {".jpg"}
    records = result["frames"]
    assert len(records) == 4
    for index, pair in enumerate(records):
        image = cv2.imread(str(output / "video_000" / pair["image"]))
        assert image.shape == (32, 32, 3)
        assert abs(image.mean() - index * 50) < 5
        assert pair["telemetry"] == rows[index]
        assert pair["telemetry_row_index"] == index
        assert pair["frame_index"] == index
        assert pair["video_time_ms"] == pytest.approx(index * 250)
        assert pair["telemetry_time_seconds"] == rows[index]["Graphics_analyst_session_time"]
    assert records[1]["alignment_delta_ms"] == pytest.approx(-10)


def test_requires_analyst_clock_even_when_simulator_clock_is_present(bundle):
    rows = json.loads((bundle / "telemetry.json").read_text())
    for row in rows:
        row["Graphics_session_time"] = row.pop("Graphics_analyst_session_time")
    write_json(bundle / "telemetry.json", rows)
    with pytest.raises(ValueError, match="Graphics_analyst_session_time"):
        preparation.extract_session_frames(bundle)
    assert not (bundle / "frames").exists()


@pytest.mark.parametrize("fps,indices", [(2, [0, 2]), (0.5, [0]), (100, [0, 1, 2, 3])])
def test_sampling_does_not_duplicate_frames(bundle, fps, indices):
    result = preparation.extract_session_frames(bundle, fps=fps)
    assert [pair["frame_index"] for pair in result["frames"]] == indices


@pytest.mark.parametrize("offset,frames,rows,skipped", [
    (0.5, [0, 1, 2, 3], [2, 3, 4, 5], 0),
    (-0.25, [1, 2, 3], [0, 1, 2], 1),
])
def test_session_video_offset_aligns_with_telemetry(bundle, offset, frames, rows, skipped):
    result = preparation.extract_session_frames(bundle, video_offset=offset)
    assert [pair["frame_index"] for pair in result["frames"]] == frames
    assert [pair["telemetry_row_index"] for pair in result["frames"]] == rows
    assert result["videos"][0]["unmatched_frames"] == skipped
    assert result["videos"][0]["offset_source"] == "explicit"


def test_skips_gaps_and_frames_beyond_telemetry_coverage(bundle):
    write_json(bundle / "telemetry.json", [{"Graphics_analyst_session_time": 125}, {"Graphics_analyst_session_time": 125.75}])
    result = preparation.extract_session_frames(bundle)
    assert [pair["frame_index"] for pair in result["frames"]] == [0, 3]
    assert result["videos"][0]["unmatched_frames"] == 2
    assert result["videos"][0]["matched_frames"] == 2


@pytest.mark.parametrize("start_ms", [0, 500])
def test_uses_presentation_times_for_variable_frame_rate(bundle, monkeypatch, start_ms):
    class Capture:
        position = -1

        def isOpened(self):
            return True

        def read(self):
            self.position += 1
            return self.position < 4, np.zeros((32, 32, 3), dtype=np.uint8)

        def get(self, prop):
            assert prop == cv2.CAP_PROP_POS_MSEC
            return start_ms + [0, 100, 550, 900][self.position]

        def release(self):
            pass

    monkeypatch.setattr(cv2, "VideoCapture", lambda _: Capture())
    result = preparation.extract_session_frames(bundle, fps=2)
    records = result["frames"]
    assert [pair["frame_index"] for pair in records] == [0, 2]
    assert [pair["video_time_ms"] for pair in records] == [0, 550]
    assert [pair["telemetry_row_index"] for pair in records] == [0, 2]
    assert records[1]["alignment_delta_ms"] == pytest.approx(-50)
    assert result["videos"][0]["first_frame_time_ms"] == start_ms


@pytest.mark.parametrize("times", [[125, 124], [125, None], [125, "125.25"], [125, float("nan")]])
def test_rejects_missing_invalid_or_reset_analyst_clock(bundle, times):
    write_json(bundle / "telemetry.json", [{"Graphics_analyst_session_time": value} for value in times])
    with pytest.raises(ValueError, match="Graphics_analyst_session_time"):
        preparation.extract_session_frames(bundle)
    assert not (bundle / "frames").exists()


def test_repeated_telemetry_times_and_equal_distance_choose_earlier_row(bundle):
    write_json(bundle / "telemetry.json", [
        {"Graphics_analyst_session_time": value} for value in [125, 125, 125.5, 126.75]
    ])
    result = preparation.extract_session_frames(bundle, max_delta_ms=250)
    records = result["frames"]
    assert records[0]["telemetry_row_index"] == 0
    assert records[1]["telemetry_time_seconds"] == 125


@pytest.mark.parametrize("failure", ["video", "write", "no_matches"])
def test_failure_removes_partial_output_and_preserves_download(bundle, monkeypatch, failure):
    if failure == "video":
        (bundle / "video_000.mp4").write_bytes(b"broken")
    elif failure == "write":
        monkeypatch.setattr(cv2, "imwrite", lambda *_: False)
    before = {path.name: path.read_bytes() for path in bundle.iterdir()}
    offset = 100 if failure == "no_matches" else None
    with pytest.raises((OSError, ValueError)):
        preparation.extract_session_frames(bundle, video_offset=offset)
    assert {path.name: path.read_bytes() for path in bundle.iterdir()} == before


def test_existing_extraction_is_preserved(bundle):
    output = bundle.parent / "chosen_output"
    preparation.extract_session_frames(bundle, output, fps=2, video_offset=0.5)
    before = {path.relative_to(output): path.read_bytes() for path in output.rglob("*") if path.is_file()}
    with pytest.raises(FileExistsError, match="Output already exists"):
        preparation.extract_session_frames(bundle, output, fps=2, video_offset=0.5)
    assert {path.relative_to(output): path.read_bytes() for path in output.rglob("*") if path.is_file()} == before


@pytest.mark.parametrize("options", [
    {"fps": 0}, {"fps": float("nan")}, {"max_delta_ms": -1}, {"max_delta_ms": float("inf")},
])
def test_invalid_extraction_options_leave_no_output(bundle, options):
    with pytest.raises(ValueError):
        preparation.extract_session_frames(bundle, **options)
    assert not (bundle / "frames").exists()
