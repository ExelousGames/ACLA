from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
import json
from pathlib import Path
import shutil
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import cv2
import numpy as np
import pytest

from app.shared.expert_features import ExpertFeatureCatalog
from app.top_laps.model import TopLapEntry, TopLapStore
from scripts import prepare_position_dataset as script
from scripts.prepare_position_dataset import extract_session_frames
from training.image_segmentation.position_data import load_position_manifest
from training.image_segmentation.position_preparation import _prepare_session, driver_position, prepare_position_dataset
from training.top_laps.serialization import serialize_top_lap_store


def write_json(path, data):
    path.write_text(json.dumps(data), encoding="utf-8")


@pytest.fixture
def dataset(tmp_path):
    folders = []
    extractions = []
    for session_id in ("a" * 24, "b" * 24):
        folder = tmp_path / session_id
        folder.mkdir()
        video = folder / "video_000.mp4"
        writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 4, (32, 32))
        assert writer.isOpened()
        for index in range(3):
            writer.write(np.full((32, 32, 3), index * 60, dtype=np.uint8))
        writer.release()
        write_json(folder / "manifest.json", {
            "session": {"sessionId": session_id, "game_recorded_from": "acc"},
            "telemetry_file": "telemetry.json",
            "videos": [{"index": 0, "file": video.name, "startedAt": 1000}],
        })
        write_json(folder / "telemetry.json", [{
            "Graphics_analyst_session_time": 100 + index / 4,
            "Static_track": "test_track", "Static_car_model": "car",
            "Graphics_track_grip_status": 2, "Graphics_normalized_car_position": index / 4,
            "Graphics_active_cars": 2, "Graphics_player_car_id": 7, "Graphics_car_id": [44, 7],
            "Graphics_car_coordinates": [{"x": 99, "y": 99, "z": 99}, {"x": 0.1, "y": 0.2, "z": 0.3}],
        } for index in range(3)])
        extractions.append(extract_session_frames(folder))
        folders.append(folder)

    features = ExpertFeatureCatalog.ExpertOptimalFeature
    names = [getattr(features, f"EXPERT_OPTIMAL_PLAYER_POS_{axis}").value for axis in "XYZ"]
    store = TopLapStore()
    store.entries[("acc", "test_track", "car", 2)] = TopLapEntry(
        track="test_track", car="car", avg_grip_int=2, x=np.array([0, 0.5, 1]),
        y=np.array([[1, 2, 3], [3, 4, 5], [1, 2, 3]]), target_features=names,
    )
    model = SimpleNamespace(modelData=serialize_top_lap_store(store), modelType="top_lap_reference", metadata={"laps": 1})
    circuit_map = {
        "id": "map-id", "game": "acc", "circuit_name": "Display Name", "source_track_key": "Test Track",
        "samples": {"middle_line": [
            {"normalized_position": p, "x": x, "y": y, "z": z}
            for p, x, y, z in [(0, -1, 2, -3), (0.5, -3, 4, -5), (1, -1, 2, -3)]
        ]},
    }
    listing = {key: value for key, value in circuit_map.items() if key != "samples"}
    responses = {"circuit-map/list?game=acc": {"list": [listing]}, "circuit-map/map-id": circuit_map}
    backend = SimpleNamespace(
        getCompleteActiveModelData=AsyncMock(return_value=model),
        call_backend_function=AsyncMock(side_effect=lambda endpoint, **_: responses[endpoint]),
    )
    return SimpleNamespace(folders=folders, extractions=extractions, backend=backend, model=model, store=store,
                           responses=responses, downloads=tmp_path,
                           output=tmp_path / "image_segmentation/position-data")


def prepare(data):
    return asyncio.run(prepare_position_dataset(data.backend, data.extractions))


def change_pair(data, update):
    update(data.extractions[0]["frames"][0])


def test_preparation_accepts_one_session_without_assigning_a_split(dataset):
    result = asyncio.run(prepare_position_dataset(dataset.backend, dataset.extractions[:1]))
    assert result["frames"] == 3
    assert len(result["sessions"]) == 1
    assert "split" not in result["sessions"][0]
    assert list(load_position_manifest(dataset.downloads)) == [dataset.folders[0].name]
    assert not dataset.output.exists()


@pytest.mark.parametrize("game, expected", [
    ("acc", [0.1, 0.3, 0.2]),
    ("iracing", [0.1, 0.2, 0.3]),
    ("iracing_live", [0.1, 0.2, 0.3]),
    ("iracing_recorded", [0.1, 0.2, 0.3]),
])
@pytest.mark.parametrize("encoded", [False, True])
def test_session_game_controls_driver_coordinate_order(dataset, game, expected, encoded):
    extraction = dataset.extractions[0]
    extraction["session"]["game_recorded_from"] = game
    for pair in extraction["frames"]:
        if encoded:
            pair["telemetry"]["Graphics_car_coordinates"] = json.dumps(pair["telemetry"]["Graphics_car_coordinates"])
    original_frames = json.dumps(extraction["frames"])

    session = _prepare_session(extraction)

    assert all(frame["driver_position"] == expected for frame in session["frames"])
    assert json.dumps(extraction["frames"]) == original_frames


def test_extracted_frames_become_offline_trainable_dataset_with_signed_targets(dataset):
    result = prepare(dataset)
    assert result["frames"] == 6
    assert not dataset.output.exists()
    assert result["sessions"][0]["reference_bucket"] == ["acc", "test_track", "car", 2]
    assert dataset.backend.getCompleteActiveModelData.await_args.args == ("top_lap_reference",)
    assert dataset.backend.call_backend_function.await_count == 2  # One shared map lookup.
    assert not list(dataset.downloads.rglob("positions.json"))
    assert "directory" not in result
    assert "manifest" not in result
    for session in result["sessions"]:
        assert "split" not in session
        assert "reference_lap" not in session
        assert "circuit_map" not in session
    # The trainer must work without the downloaded source JSON files.
    for folder in dataset.folders:
        for path in folder.rglob("*.json"):
            if not path.name.endswith(".position.json"):
                path.unlink()
    records = load_position_manifest(dataset.downloads)
    for frames in records.values():
        assert [row["normalized_position"] for row in frames] == [0, 0.25, 0.5]
        assert all(row["game"] == "acc" and row["map_name"] == "Display Name" for row in frames)
    np.testing.assert_allclose(records["a" * 24][1]["references"], [[2, 3, 4], [-2, 3, -4]])
    np.testing.assert_allclose(records["a" * 24][1]["target"], [[1.9, 2.7, 3.8], [-2.1, 2.7, -4.2]])


@pytest.mark.parametrize("source", ["iracing", "iracing_live", "iracing_recorded"])
def test_context_comes_from_each_selected_map_detail(dataset, source):
    dataset.extractions[1]["session"]["game_recorded_from"] = source
    reference = dataset.store.entries[("acc", "test_track", "car", 2)]
    dataset.store.entries[(source, "test_track", "car", 2)] = TopLapEntry(
        game=source, track=reference.track, car=reference.car, avg_grip_int=2,
        x=reference.x, y=reference.y, target_features=reference.target_features,
    )
    dataset.model.modelData = serialize_top_lap_store(dataset.store)
    dataset.responses["circuit-map/list?game=iracing"] = {"list": [{
        "id": "iracing-map", "source_track_key": "Test Track", "circuit_name": "Old Display Name",
    }]}
    dataset.responses["circuit-map/iracing-map"] = {
        **dataset.responses["circuit-map/map-id"], "id": "iracing-map",
        "game": "iracing", "circuit_name": "iRacing Display Name",
    }

    assert prepare(dataset)["frames"] == 6
    records = load_position_manifest(dataset.downloads)
    for session_id, game, map_name in (
        ("a" * 24, "acc", "Display Name"), ("b" * 24, "iracing", "iRacing Display Name"),
    ):
        assert all(row["game"] == game and row["map_name"] == map_name for row in records[session_id])


@pytest.mark.parametrize("name", [None, " ", 7])
def test_preparation_rejects_missing_or_invalid_map_names_before_writing(dataset, name):
    dataset.responses["circuit-map/map-id"]["circuit_name"] = name
    with pytest.raises(ValueError, match="map_name"):
        prepare(dataset)
    assert not list(dataset.downloads.rglob("*.position.json"))


def test_driver_xyz_and_other_telemetry_never_become_training_inputs(dataset):
    prepare(dataset)
    before = load_position_manifest(dataset.downloads)["a" * 24][0]
    change_pair(dataset, lambda pair: pair["telemetry"].update({
        "Graphics_car_coordinates": [{"x": 99, "y": 99, "z": 99}, {"x": -1, "y": 2, "z": -3}],
        "Physics_speed_kmh": 300, "Physics_brake": 1, "Physics_gas": 0,
    }))
    for folder in dataset.folders:
        for path in folder.rglob("*.position.json"):
            path.unlink()
    prepare(dataset)
    after = load_position_manifest(dataset.downloads)["a" * 24][0]
    assert set(before) == set(after) == {
        "session_id", "image", "references", "target", "game", "map_name", "normalized_position",
    }
    assert before["image"] == after["image"]
    for field in ("session_id", "game", "map_name", "normalized_position"):
        assert before[field] == after[field]
    np.testing.assert_array_equal(before["references"], after["references"])
    np.testing.assert_allclose(before["target"], [[0.9, 1.7, 2.8], [-1.1, 1.7, -3.2]])
    np.testing.assert_allclose(after["target"], [[2, 5, 1], [0, 5, -5]])


def test_cli_skips_existing_ids_and_custom_frame_folders(dataset, monkeypatch, capsys):
    from app.integrations.backend import client

    monkeypatch.setattr(client, "backend_service", dataset.backend)
    monkeypatch.setattr(script, "SESSION_DOWNLOAD_DIR", dataset.folders[0].parent)
    assert script.main([
        "--sessions", dataset.folders[0].name.upper(), str(dataset.folders[1] / "frames"),
    ]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["frames"] == 0
    assert result["skipped_sessions"] == [dataset.folders[0].name, str(dataset.folders[1] / "frames")]
    dataset.backend.call_backend_function.assert_not_awaited()
    dataset.backend.getCompleteActiveModelData.assert_not_awaited()
    assert not dataset.output.exists()


@pytest.mark.parametrize("encoded", [False, True])
def test_player_slot_lookup_matches_car_id_instead_of_using_it_as_array_index(encoded):
    ids = [42, 7, 0, 7]
    positions = [{"x": 99, "y": 99, "z": 99}, {"x": -1, "y": 2, "z": -3}, None, None]
    row = {"Graphics_active_cars": 2, "Graphics_player_car_id": 7,
           "Graphics_car_id": json.dumps(ids) if encoded else ids,
           "Graphics_car_coordinates": json.dumps(positions) if encoded else positions}
    assert driver_position(row) == [-1, 2, -3]
    # A player found only in the inactive tail must not be selected.
    row["Graphics_player_car_id"] = 0
    assert driver_position(row) is None


@pytest.mark.parametrize("encoded", [False, True])
@pytest.mark.parametrize("active_cars", [1, 2])
def test_preparation_finds_player_zero_after_trimming_inactive_slots(dataset, encoded, active_cars):
    ids = [42] * (active_cars - 1) + [0] * (61 - active_cars)
    coordinates = ([{"x": 99, "y": 99, "z": 99}] * (active_cars - 1)
                   + [{"x": -1, "y": 2, "z": -3}]
                   + [{"x": 0, "y": 0, "z": 0}] * (60 - active_cars))
    for pair in dataset.extractions[0]["frames"]:
        pair["telemetry"].update({
            "Graphics_active_cars": active_cars, "Graphics_player_car_id": 0,
            "Graphics_car_id": json.dumps(ids) if encoded else ids,
            "Graphics_car_coordinates": json.dumps(coordinates) if encoded else coordinates,
        })
    result = prepare(dataset)
    assert result["frames"] == 6
    assert result["skipped_sessions"] == []
    records = load_position_manifest(dataset.downloads)
    np.testing.assert_allclose(records["a" * 24][0]["target"], [[2, 5, 1], [0, 5, -5]])


def test_duplicate_player_ids_within_active_slots_remain_ambiguous():
    row = {
        "Graphics_active_cars": 2, "Graphics_player_car_id": 0,
        "Graphics_car_id": [0] * 60,
        "Graphics_car_coordinates": [{"x": -1, "y": 2, "z": -3}] * 60,
    }
    assert driver_position(row) is None


def test_no_active_cars_skips_player_lookup():
    assert driver_position({"Graphics_active_cars": 0}) is None


@pytest.mark.parametrize("active_cars", [None, True, -1, 1.5, float("inf"), "2"])
def test_active_car_count_must_be_a_nonnegative_integer(active_cars):
    with pytest.raises(ValueError, match="Graphics_active_cars"):
        driver_position({"Graphics_active_cars": active_cars})


@pytest.mark.parametrize("ids", [None, [7]])
def test_single_active_car_still_requires_a_matching_player_id(ids):
    assert driver_position({
        "Graphics_active_cars": 1, "Graphics_player_car_id": 0, "Graphics_car_id": ids,
        "Graphics_car_coordinates": [{"x": 1, "y": 2, "z": 3}],
    }) is None


def test_zero_coordinates_are_valid_for_an_active_player():
    assert driver_position({
        "Graphics_active_cars": 2, "Graphics_player_car_id": 0, "Graphics_car_id": [7, 0, 0],
        "Graphics_car_coordinates": [{"x": 1, "y": 2, "z": 3}] + [{"x": 0, "y": 0, "z": 0}] * 2,
    }) == [0, 0, 0]


@pytest.mark.parametrize("coordinates,message", [
    ([], "missing the player's XYZ coordinates"),
    ([{"x": 1, "y": 2}], "Player z"),
    ([{"x": 1, "y": 2, "z": float("nan")}], "Player z"),
])
def test_single_active_car_requires_valid_first_coordinates(coordinates, message):
    with pytest.raises(ValueError, match=message):
        driver_position({
            "Graphics_active_cars": 1, "Graphics_player_car_id": 0, "Graphics_car_id": [0],
            "Graphics_car_coordinates": coordinates,
        })


@pytest.mark.parametrize("encoded", [False, True])
@pytest.mark.parametrize("player,ids", [(0, [0] * 60), (99, [44, 7]), (-1, [-1] * 60)])
def test_preparation_skips_frames_without_an_identifiable_player(dataset, player, ids, encoded):
    change_pair(dataset, lambda pair: pair["telemetry"].update({
        "Graphics_player_car_id": player,
        "Graphics_car_id": json.dumps(ids) if encoded else ids,
        "Graphics_track_grip_status": 6,
    }))
    result = prepare(dataset)
    assert result["frames"] == 5
    assert not (dataset.folders[0] / "frames/video_000/frame_000000.position.json").exists()
    records = load_position_manifest(dataset.downloads)
    assert len(records["a" * 24]) == 2
    assert len(records["b" * 24]) == 3
    np.testing.assert_allclose(records["a" * 24][0]["target"], [[1.9, 2.7, 3.8], [-2.1, 2.7, -4.2]])


@pytest.mark.parametrize("legacy", [
    {"Graphics_player_pos_x": -1, "Graphics_player_pos_y": 2, "Graphics_player_pos_z": -3},
    {"Graphics_player_pos_x": None},
])
def test_legacy_player_coordinates_are_ignored(legacy):
    row = {
        **legacy, "Graphics_active_cars": 1, "Graphics_player_car_id": 7, "Graphics_car_id": [7],
        "Graphics_car_coordinates": [{"x": 1, "y": 2, "z": 3}],
    }
    assert driver_position(row) == [1, 2, 3]
    row["Graphics_car_coordinates"] = []
    with pytest.raises(ValueError, match="Graphics_car_coordinates"):
        driver_position(row)
    row["Graphics_car_coordinates"] = [{"x": 1, "y": 2, "z": 3}]
    row["Graphics_car_id"] = [42]
    assert driver_position(row) is None


@pytest.mark.parametrize("field,value", [
    ("Graphics_car_coordinates", []),
    ("Graphics_normalized_car_position", 1.1), ("Graphics_normalized_car_position", None),
    ("Graphics_track_grip_status", None), ("Graphics_track_grip_status", 7),
    ("Static_track", "another_track"), ("Static_car_model", ""),
])
def test_invalid_telemetry_fails_before_download_without_partial_output(dataset, field, value):
    change_pair(dataset, lambda pair: pair["telemetry"].update({field: value}))
    with pytest.raises(ValueError):
        prepare(dataset)
    dataset.backend.getCompleteActiveModelData.assert_not_awaited()
    assert not dataset.output.exists()


@pytest.fixture(params=["bucket", "map"])
def missing_dependency(dataset, request):
    if request.param == "bucket":
        update = {"Graphics_track_grip_status": 3}
        reason = "No exact reference lap"
    else:
        update = {"Static_track": "unmapped_track"}
        entry = dataset.store.entries[("acc", "test_track", "car", 2)]
        dataset.store.entries[("acc", "unmapped_track", "car", 2)] = TopLapEntry(
            track="unmapped_track", car=entry.car, avg_grip_int=entry.avg_grip_int,
            x=entry.x, y=entry.y, target_features=entry.target_features,
        )
        dataset.model.modelData = serialize_top_lap_store(dataset.store)
        reason = "No circuit map"
    folder = dataset.folders[0]
    rows = json.loads((folder / "telemetry.json").read_text())
    for row in rows:
        row.update(update)
    write_json(folder / "telemetry.json", rows)
    for pair in dataset.extractions[0]["frames"]:
        pair["telemetry"].update(update)
    return reason


def test_missing_dependency_skips_session_and_prepares_later_sessions(dataset, missing_dependency, caplog):
    result = prepare(dataset)
    assert result["frames"] == 3
    assert result["skipped_sessions"] == [dataset.folders[0].name]
    assert [session["id"] for session in result["sessions"]] == [dataset.folders[1].name]
    records = load_position_manifest(dataset.downloads)
    assert list(records) == [dataset.folders[1].name]
    np.testing.assert_allclose(records[dataset.folders[1].name][1]["target"], [[1.9, 2.7, 3.8], [-2.1, 2.7, -4.2]])
    assert dataset.folders[0].name in caplog.text
    assert missing_dependency in caplog.text


@pytest.mark.parametrize("fault", ["wrong-game", "wrong-track", "empty-centerline", "backend-error", "image"])
def test_invalid_reference_or_incomplete_dataset_never_publishes(dataset, fault):
    if fault == "wrong-game":
        dataset.responses["circuit-map/map-id"]["game"] = "iracing"
    elif fault == "wrong-track":
        dataset.responses["circuit-map/map-id"]["source_track_key"] = "elsewhere"
    elif fault == "empty-centerline":
        dataset.responses["circuit-map/map-id"]["samples"]["middle_line"] = []
    elif fault == "backend-error":
        dataset.responses["circuit-map/map-id"] = {"error": "Denied"}
    else:
        (dataset.folders[0] / "frames/video_000/frame_000000.jpg").unlink()
    with pytest.raises((ValueError, FileNotFoundError, RuntimeError)):
        prepare(dataset)
    assert not dataset.output.exists()
    assert all(not list(folder.rglob("*.position.json")) for folder in dataset.folders)


@pytest.mark.parametrize("failure", ["existing_position", "write", "validation"])
@pytest.mark.parametrize("output_exists", [False, True])
def test_failed_preparation_removes_new_position_files_and_preserves_existing_data(
    dataset, monkeypatch, failure, output_exists,
):
    from training.image_segmentation import position_preparation

    if output_exists:
        dataset.output.mkdir(parents=True)
        write_json(dataset.output / "positions.json", {"sessions": []})
    output_before = {path: path.read_bytes() for path in dataset.output.rglob("*") if path.is_file()}
    image_folder = dataset.folders[0] / "frames/video_000"
    annotation = image_folder / "frame_000000.json"
    write_json(annotation, {"imagePath": "frame_000000.jpg", "shapes": []})
    existing = image_folder / "frame_000001.position.json"
    if failure == "existing_position":
        existing.write_text("existing position data")
    elif failure == "write":
        original_dump = position_preparation.json.dump

        def fail_write(data, target, **kwargs):
            if Path(target.name) == existing:
                target.write("partial")
                raise OSError("Write failed")
            return original_dump(data, target, **kwargs)

        monkeypatch.setattr(position_preparation.json, "dump", fail_write)
    else:
        def fail_validation(*_):
            raise ValueError("Validation failed")

        monkeypatch.setattr(position_preparation, "load_position_manifest", fail_validation)
    before = {path: path.read_bytes() for folder in dataset.folders for path in folder.rglob("*") if path.is_file()}

    with pytest.raises((OSError, ValueError)):
        prepare(dataset)

    assert {path: path.read_bytes() for folder in dataset.folders for path in folder.rglob("*") if path.is_file()} == before
    assert dataset.output.exists() == output_exists
    assert {path: path.read_bytes() for path in dataset.output.rglob("*") if path.is_file()} == output_before


@pytest.mark.parametrize("source_key", ["different_track", None, ""])
def test_map_name_cannot_replace_source_track(dataset, caplog, source_key):
    listing = dataset.responses["circuit-map/list?game=acc"]["list"][0]
    listing.update(source_track_key=source_key, circuit_name="test_track")
    result = prepare(dataset)
    assert result["frames"] == 0
    assert result["sessions"] == []
    assert result["skipped_sessions"] == [folder.name for folder in dataset.folders]
    assert "No circuit map" in caplog.text
    assert not dataset.output.exists()


def test_map_details_require_source_track_even_when_name_matches(dataset):
    detail = dataset.responses["circuit-map/map-id"]
    del detail["source_track_key"]
    detail["circuit_name"] = "test_track"
    with pytest.raises(ValueError, match="does not match"):
        prepare(dataset)
    assert not dataset.output.exists()
    assert all(not list(folder.rglob("*.position.json")) for folder in dataset.folders)


def test_duplicate_session_is_rejected_before_backend_access(dataset):
    dataset.extractions[1] = dataset.extractions[0]
    with pytest.raises(ValueError, match="must appear once"):
        prepare(dataset)
    dataset.backend.getCompleteActiveModelData.assert_not_awaited()


@pytest.mark.parametrize("fault", ["session", "count", "image-escape"])
def test_mismatched_pairs_are_rejected(dataset, fault):
    if fault == "session":
        change_pair(dataset, lambda pair: pair.update(session_id="another-session"))
    elif fault == "count":
        dataset.extractions[0]["frames"].pop(0)
    else:
        change_pair(dataset, lambda pair: pair.update(image="../../outside.jpg"))
    with pytest.raises(ValueError):
        prepare(dataset)
    assert not dataset.output.exists()


@pytest.mark.parametrize("include_valid", [False, True])
def test_empty_extraction_skips_session_with_a_clear_reason(dataset, caplog, include_valid):
    dataset.extractions[0]["frames"] = []
    dataset.extractions[0]["videos"][0]["matched_frames"] = 0
    if not include_valid:
        dataset.extractions = dataset.extractions[:1]
    result = prepare(dataset)
    assert result["skipped_sessions"] == [dataset.folders[0].name]
    assert result["frames"] == (3 if include_valid else 0)
    assert [session["id"] for session in result["sessions"]] == ([dataset.folders[1].name] if include_valid else [])
    assert f"Skipping session {dataset.folders[0].name}" in caplog.text
    assert "No frame/telemetry pairs" in caplog.text
    assert "Run frame extraction" in caplog.text
    if not include_valid:
        dataset.backend.getCompleteActiveModelData.assert_not_awaited()
    assert not dataset.output.exists()


@pytest.fixture
def downloadable_dataset(dataset, monkeypatch, tmp_path):
    """Serve the real fixture MP4s and telemetry through the download interface."""
    from app.integrations.backend import client

    source = {folder.name: folder for folder in dataset.folders}
    requests = []

    async def request(endpoint, method="GET", data=None, **kwargs):
        requests.append(endpoint)
        if endpoint == "racing-session/with-videos":
            now = datetime.now().astimezone()
            return {"list": [
                {"sessionId": session_id, "created_date": (now - timedelta(days=index)).isoformat()}
                for index, session_id in enumerate(source)
            ]}
        if endpoint in dataset.responses:
            return dataset.responses[endpoint]
        if endpoint.endswith("/videos"):
            folder = source[endpoint.split("/")[1]]
            return [{"index": 0, "name": "recording.mp4", "size": (folder / "video_000.mp4").stat().st_size,
                     "mimeType": "video/mp4", "startedAt": 1000}]
        if endpoint == "racing-session/download/init":
            session_id = data["sessionId"]
            return {"downloadId": session_id, "sessionMetadata": [
                {"sessionId": session_id, "game_recorded_from": "acc", "chunkCount": 1, "dataPoints": 3},
            ]}
        if endpoint == "racing-session/download/chunk":
            return json.loads((source[data["sessionId"]] / "telemetry.json").read_text())
        assert endpoint == "racing-session/download/cleanup"
        return {}

    async def video(session_id, index):
        assert index == 0
        yield (source[session_id] / "video_000.mp4").read_bytes()

    dataset.backend.call_backend_function.side_effect = request
    dataset.backend.iter_session_video = video
    dataset.requests = requests
    dataset.downloads = tmp_path / "downloads"
    dataset.args = ["--sessions", dataset.folders[0].name, dataset.folders[1].name]
    monkeypatch.setattr(client, "backend_service", dataset.backend)
    monkeypatch.setattr(script, "SESSION_DOWNLOAD_DIR", dataset.downloads)
    return dataset


@pytest.mark.parametrize("player,ids", [
    (0, [0] * 60), (99, [44, 7]), (-1, [-1] * 60), (0, None),
])
@pytest.mark.parametrize("include_valid", [False, True])
def test_cli_skips_sessions_with_unidentifiable_players_and_explains_why(
    downloadable_dataset, capsys, caplog, player, ids, include_valid,
):
    data = downloadable_dataset
    rejected = data.folders[:1] if include_valid else data.folders
    for folder in rejected:
        path = folder / "telemetry.json"
        rows = json.loads(path.read_text())
        for row in rows:
            row.update(Graphics_player_car_id=player, Graphics_car_id=ids)
        write_json(path, rows)

    assert script.main(data.args) == 0

    captured = capsys.readouterr()
    result = json.loads(captured.out)
    assert "Preparation failed" not in captured.err
    assert result["skipped_sessions"] == [folder.name for folder in rejected]
    assert result["frames"] == (3 if include_valid else 0)
    assert [session["id"] for session in result["sessions"]] == ([data.folders[1].name] if include_valid else [])
    assert result["annotation_directories"] == (
        [str(data.downloads / data.folders[1].name / "frames")] if include_valid else []
    )
    message = caplog.text
    for folder in rejected:
        assert f"Skipping session {folder.name}" in message
    assert "No usable position-training frames" in message
    assert "all 3 extracted frames were skipped" in message
    assert f"Graphics_player_car_id={player}" in message
    assert "Graphics_active_cars=2" in message
    assert "first Graphics_active_cars slots of Graphics_car_id" in message
    assert "telemetry row 0" in message
    assert "exactly once" in message
    assert "Graphics_player_pos" not in message
    assert str(data.downloads / data.folders[0].name / "frames/video_000") in message
    assert len(list(data.downloads.rglob("*.jpg"))) == 6
    for folder in rejected:
        assert not list((data.downloads / folder.name).rglob("*.position.json"))
    assert not data.output.exists()
    if include_valid:
        records = load_position_manifest(data.downloads)
        assert list(records) == [data.folders[1].name]
        assert len(records[data.folders[1].name]) == 3
    else:
        data.backend.getCompleteActiveModelData.assert_not_awaited()
        assert not any(endpoint.startswith("circuit-map/") for endpoint in data.requests)


def test_one_command_downloads_extracts_and_prepares_before_annotation(downloadable_dataset, capsys):
    data = downloadable_dataset
    assert script.main([*data.args, "--fps", "2", "--video-offset", f"{data.folders[0].name}=0.25"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["frames"] == 3
    assert not data.output.exists()
    assert not list(data.downloads.rglob("positions.json"))
    assert "manifest" not in result
    assert data.requests.count("racing-session/download/init") == 2
    assert data.requests.count("racing-session/download/cleanup") == 2
    records = load_position_manifest(data.downloads)
    np.testing.assert_allclose(records["a" * 24][0]["references"], [[2, 3, 4], [-2, 3, -4]])
    np.testing.assert_allclose(records["a" * 24][0]["target"], [[1.9, 2.7, 3.8], [-2.1, 2.7, -4.2]])
    for session_id, directory in zip(records, result["annotation_directories"]):
        folder = Path(directory)
        assert folder.is_dir()
        assert not list(folder.rglob("*.telemetry.json"))
        assert len(list(folder.rglob("*.position.json"))) == len(records[session_id])
        # Human-created Labelme JSON can be saved beside each image afterwards.
        for row in records[session_id]:
            assert row["image"].is_relative_to(folder)
            assert not row["image"].with_suffix(".json").exists()
            position = json.loads(row["image"].with_suffix(".position.json").read_text())
            assert set(position) == {"session_id", "image", "references", "target", "game", "map_name", "normalized_position"}
            assert position["game"] == row["game"] == "acc"
            assert position["map_name"] == row["map_name"] == "Display Name"
            assert position["normalized_position"] == row["normalized_position"]
            assert position["session_id"] == session_id
            assert position["image"] == row["image"].name
            np.testing.assert_allclose(position["references"], row["references"])
            np.testing.assert_allclose(position["target"], row["target"])

    # Moving only images and their position files must leave a trainable dataset.
    moved = data.downloads.parent / "moved"
    for session_id, frames in records.items():
        destination = moved / session_id
        destination.mkdir(parents=True)
        for row in frames:
            for path in (row["image"], row["image"].with_suffix(".position.json")):
                shutil.copy2(path, destination / path.name)
    moved_records = load_position_manifest(moved)
    assert moved_records.keys() == records.keys()
    for session_id, frames in moved_records.items():
        for row, original in zip(frames, records[session_id]):
            assert row["image"].parent == moved / session_id
            for field in ("game", "map_name", "normalized_position"):
                assert row[field] == original[field]
            np.testing.assert_array_equal(row["references"], original["references"])
            np.testing.assert_array_equal(row["target"], original["target"])


@pytest.mark.parametrize("include_valid", [False, True])
def test_cli_reports_dependency_skips_with_existing_sessions(
    downloadable_dataset, missing_dependency, include_valid, capsys,
):
    data = downloadable_dataset
    existing = data.downloads / ("c" * 24)
    existing.mkdir(parents=True)
    selected = [existing.name, data.folders[0].name]
    if include_valid:
        selected.append(data.folders[1].name)

    assert script.main(["--sessions", *selected]) == 0

    result = json.loads(capsys.readouterr().out)
    assert result["skipped_sessions"] == [existing.name, data.folders[0].name]
    assert result["frames"] == (3 if include_valid else 0)
    assert [session["id"] for session in result["sessions"]] == ([data.folders[1].name] if include_valid else [])
    assert result["annotation_directories"] == (
        [str(data.downloads / data.folders[1].name / "frames")] if include_valid else []
    )
    assert not data.output.exists()


@pytest.mark.parametrize("use_path", [False, True])
@pytest.mark.parametrize("output_exists", [False, True])
def test_existing_session_folder_skips_all_work_without_a_manifest(tmp_path, monkeypatch, use_path, output_exists):
    from training.image_segmentation import position_preparation

    session_id = "a" * 24
    folder = tmp_path / session_id
    folder.mkdir()
    output = tmp_path / "prepared"
    if output_exists:
        output.mkdir()
    download = AsyncMock()
    extract = Mock()
    prepare = AsyncMock()
    monkeypatch.setattr(script, "SESSION_DOWNLOAD_DIR", tmp_path)
    monkeypatch.setattr(script, "download_session_pair", download)
    monkeypatch.setattr(script, "extract_session_frames", extract)
    monkeypatch.setattr(position_preparation, "prepare_position_dataset", prepare)

    result = asyncio.run(script.prepare_sessions(
        None, [str(folder) if use_path else session_id.upper()],
    ))

    assert result["frames"] == 0
    assert result["sessions"] == result["annotation_directories"] == []
    assert len(result["skipped_sessions"]) == 1
    download.assert_not_awaited()
    extract.assert_not_called()
    prepare.assert_not_awaited()
    assert list(folder.iterdir()) == []
    assert output.exists() == output_exists


def test_mixed_selection_only_downloads_and_processes_new_sessions(downloadable_dataset, capsys):
    data = downloadable_dataset
    existing = data.downloads / data.folders[0].name
    existing.mkdir(parents=True)

    assert script.main(data.args) == 0

    result = json.loads(capsys.readouterr().out)
    assert result["frames"] == 3
    assert [session["id"] for session in result["sessions"]] == [data.folders[1].name]
    assert result["skipped_sessions"] == [data.folders[0].name]
    assert data.requests.count("racing-session/download/init") == 1
    assert f"racing-session/{data.folders[0].name}/videos" not in data.requests
    assert result["annotation_directories"] == [str(data.downloads / data.folders[1].name / "frames")]
    assert list(existing.iterdir()) == []


def test_rerun_preserves_images_annotations_and_output_without_processing(downloadable_dataset, capsys):
    data = downloadable_dataset
    assert script.main(data.args) == 0
    capsys.readouterr()
    records = load_position_manifest(data.downloads)
    annotation = records["a" * 24][0]["image"].with_suffix(".json")
    write_json(annotation, {"imagePath": records["a" * 24][0]["image"].name, "shapes": []})
    before = {path: path.read_bytes() for path in data.downloads.rglob("*") if path.is_file()}
    data.requests.clear()
    data.backend.getCompleteActiveModelData.reset_mock()
    assert script.main(data.args) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["frames"] == 0
    assert result["skipped_sessions"] == [folder.name for folder in data.folders]
    assert not data.requests
    data.backend.getCompleteActiveModelData.assert_not_awaited()
    assert not data.output.exists()
    assert {path: path.read_bytes() for path in data.downloads.rglob("*") if path.is_file()} == before


def test_preparation_preserves_existing_index_directory(downloadable_dataset, capsys):
    data = downloadable_dataset
    data.output.mkdir(parents=True)
    marker = data.output / "notes.txt"
    marker.write_text("Existing notes")
    index = data.output / "positions.json"
    write_json(index, {"sessions": []})
    before = index.read_bytes()

    assert script.main(data.args) == 0

    result = json.loads(capsys.readouterr().out)
    assert result["frames"] == 6
    assert set(load_position_manifest(data.downloads)) == {folder.name for folder in data.folders}
    assert marker.read_text() == "Existing notes"
    assert index.read_bytes() == before


def test_later_run_keeps_all_sessions_discoverable_without_an_index(downloadable_dataset, capsys):
    data = downloadable_dataset
    first, second = [folder.name for folder in data.folders]
    assert script.main(["--sessions", first]) == 0
    capsys.readouterr()
    records = load_position_manifest(data.downloads)
    image = records[first][0]["image"]
    write_json(image.with_suffix(".json"), {"imagePath": image.name, "shapes": []})
    before = {path: path.read_bytes() for path in (data.downloads / first).rglob("*") if path.is_file()}

    assert script.main(data.args) == 0

    result = json.loads(capsys.readouterr().out)
    assert result["frames"] == 3
    assert result["skipped_sessions"] == [first]
    assert not data.output.exists()
    discovered = load_position_manifest(data.downloads)
    assert set(discovered) == {first, second}
    assert sum(len(frames) for frames in discovered.values()) == 6
    assert {path: path.read_bytes() for path in (data.downloads / first).rglob("*") if path.is_file()} == before


@pytest.mark.parametrize("options", [
    ["--fps", "0"], ["--fps", "nan"], ["--max-delta-ms", "-1"], ["--max-delta-ms", "inf"],
    ["--video-offset", f"{'c' * 24}=0"],
])
def test_invalid_cli_options_fail_before_download(downloadable_dataset, options, capsys):
    data = downloadable_dataset
    assert script.main([*data.args, *options]) == 1
    assert "Preparation failed" in capsys.readouterr().err
    assert not data.requests
    assert not data.downloads.exists()
    assert not data.output.exists()


@pytest.mark.parametrize("options", [["--fps", "2"], ["--max-delta-ms", "200"],
                                        ["--video-offset", f"{'a' * 24}=0.25"]])
def test_existing_sessions_are_skipped_even_with_new_alignment_options(dataset, monkeypatch, options, capsys):
    monkeypatch.setattr(script, "SESSION_DOWNLOAD_DIR", dataset.folders[0].parent)
    assert script.main(["--sessions", dataset.folders[0].name, dataset.folders[1].name, *options]) == 0
    assert json.loads(capsys.readouterr().out)["skipped_sessions"] == [folder.name for folder in dataset.folders]
    assert not dataset.output.exists()


def test_existing_sessions_are_skipped_without_validating_frames(dataset, capsys):
    manifest_path = dataset.folders[0] / "frames/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["session"]["sessionId"] = "c" * 24
    write_json(manifest_path, manifest)
    assert script.main(["--sessions", str(dataset.folders[0]), str(dataset.folders[1])]) == 0
    assert json.loads(capsys.readouterr().out)["frames"] == 0
    assert not dataset.output.exists()


def test_duplicate_selection_fails_before_download(downloadable_dataset, capsys):
    data = downloadable_dataset
    assert script.main(["--sessions", data.folders[0].name, data.folders[0].name.upper()]) == 1
    assert "must appear once" in capsys.readouterr().err
    assert not data.requests


def test_completed_downloads_are_skipped_on_retry_after_failure(downloadable_dataset, capsys):
    data = downloadable_dataset
    data.backend.getCompleteActiveModelData.side_effect = RuntimeError("Reference model missing")
    assert script.main(data.args) == 1
    assert "Reference model missing" in capsys.readouterr().err
    assert not data.output.exists()
    assert len(list(data.downloads.rglob("*.jpg"))) == 6
    data.requests.clear()
    data.backend.getCompleteActiveModelData.reset_mock()
    data.backend.getCompleteActiveModelData.side_effect = None
    assert script.main(data.args) == 0
    assert json.loads(capsys.readouterr().out)["frames"] == 0
    assert not data.requests
    data.backend.getCompleteActiveModelData.assert_not_awaited()
    assert not data.output.exists()


@pytest.mark.parametrize("status", [403, 404, 500])
def test_download_failure_stops_before_extraction_or_references(downloadable_dataset, capsys, status):
    data = downloadable_dataset
    data.backend.call_backend_function.side_effect = None
    data.backend.call_backend_function.return_value = {"error": f"Rejected {status}"}
    assert script.main(data.args) == 1
    assert f"Rejected {status}" in capsys.readouterr().err
    data.backend.getCompleteActiveModelData.assert_not_awaited()
    assert not data.output.exists()
    assert not data.downloads.exists()


@pytest.mark.parametrize("options", [["--fps", "2"], ["--sessions"]])
def test_cli_requires_sessions(options):
    with pytest.raises(SystemExit) as error:
        script.main(options)
    assert error.value.code == 2


def test_guided_preparation_skips_existing_local_sessions(dataset, monkeypatch, capsys):
    from app.integrations.backend import client

    monkeypatch.setattr(client, "backend_service", dataset.backend)
    monkeypatch.setattr(script, "SESSION_DOWNLOAD_DIR", dataset.folders[0].parent)
    dataset.responses["racing-session/with-videos"] = {"list": [
        {"sessionId": folder.name, "created_date": datetime.now().astimezone().isoformat()}
        for folder in dataset.folders
    ]}
    answers = iter([
        "6", "1", "2", "",  # Pick both listed sessions.
        "", "", "", "",  # Default sampling/alignment, no offsets, start.
    ])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))
    assert script.main([]) == 0
    assert not dataset.output.exists()
    dataset.backend.getCompleteActiveModelData.assert_not_awaited()
    dataset.backend.call_backend_function.assert_awaited_once_with("racing-session/with-videos")
    output = capsys.readouterr().out
    assert "Preparation summary" in output
    assert '"frames": 0' in output


def test_guided_preparation_reprompts_invalid_answers_before_downloading(downloadable_dataset, monkeypatch, capsys):
    data = downloadable_dataset
    train, val = [folder.name for folder in data.folders]
    answers = iter([
        "bad", "6", "", "99", "c" * 24, train.upper(), train, val, "",  # Invalid and duplicate IDs.
        "0", "nan", "2", "-1", "inf", "100",  # Sampling and tolerance.
        "invalid", f"{'c' * 24}=0", f"{train}=0.25", f"{train}=0.5", "",
        "",  # Start.
    ])

    def answer(prompt):
        assert "directory" not in prompt.lower()
        assert all(endpoint == "racing-session/with-videos" for endpoint in data.requests)
        assert not data.downloads.exists()  # Only metadata until all settings are chosen.
        return next(answers)

    monkeypatch.setattr("builtins.input", answer)
    assert script.main([]) == 0
    assert not data.output.exists()
    records = load_position_manifest(data.downloads)
    assert len(records["a" * 24]) == 1
    assert len(records["b" * 24]) == 2
    output = capsys.readouterr().out
    assert "Each session must appear once" in output
    assert "Choose a new directory" not in output
    assert "already entered" in output


def test_guided_image_cancellation_leaves_no_downloads_or_output(downloadable_dataset, monkeypatch):
    data = downloadable_dataset
    answers = iter([
        "1", "", "", "", "2",
    ])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))
    assert script.main([]) == 0
    assert data.requests == ["racing-session/with-videos"]
    assert not data.downloads.exists()
    assert not data.output.exists()


def test_guided_date_ranges_download_only_selected_sessions(downloadable_dataset, monkeypatch, capsys):
    data = downloadable_dataset
    request = data.backend.call_backend_function.side_effect

    async def with_old_session(endpoint, *args, **kwargs):
        response = await request(endpoint, *args, **kwargs)
        if endpoint == "racing-session/with-videos":
            response["list"].append({
                "sessionId": "c" * 24,
                "created_date": (datetime.now().astimezone() - timedelta(days=30)).isoformat(),
            })
        return response

    data.backend.call_backend_function.side_effect = with_old_session
    answers = iter(["last 2 days", "", "", "", "1"])

    def answer(_):
        assert all(endpoint == "racing-session/with-videos" for endpoint in data.requests)
        return next(answers)

    monkeypatch.setattr("builtins.input", answer)
    assert script.main([]) == 0
    assert data.requests[0] == "racing-session/with-videos"
    assert data.requests.count("racing-session/download/init") == 2
    assert {folder.name for folder in data.downloads.iterdir()} == {folder.name for folder in data.folders}
    assert "c" * 24 in capsys.readouterr().out  # Older sessions are listed, but not downloaded.


@pytest.mark.parametrize("response, message", [
    ({"list": []}, "No sessions with attached videos"),
    ({"error": "Denied"}, "Denied"),
    ({"unexpected": []}, "Invalid video session listing"),
])
def test_guided_listing_failure_stops_before_download(downloadable_dataset, capsys, response, message):
    data = downloadable_dataset
    data.backend.call_backend_function.side_effect = None
    data.backend.call_backend_function.return_value = response
    assert script.main([]) == 1
    assert message in capsys.readouterr().err
    data.backend.call_backend_function.assert_awaited_once_with("racing-session/with-videos")
    data.backend.getCompleteActiveModelData.assert_not_awaited()
    assert not data.downloads.exists()


@pytest.mark.parametrize("choice, expected", [
    ("Today", "ab"), ("Yesterday", "cd"), ("Last 2 days", "abcd"),
    ("Last 3 days", "abcde"), ("Last week", "abcdef"), ("Pick a session", "1"),
    ("All sessions with videos", "abcdef12"), ("7", "abcdef12"),
])
def test_date_selection_uses_local_calendar_boundaries_including_dst(monkeypatch, capsys, choice, expected):
    class FixedDatetime(datetime):
        @classmethod
        def now(cls):
            return cls(2026, 11, 2, 12)

    backend = SimpleNamespace(call_backend_function=AsyncMock(return_value={"list": [
        {"sessionId": key * 24, "created_date": created}
        for key, created in [
            ("a", "2026-11-02T17:00:00Z"),  # Today.
            ("b", "2026-11-02T05:00:00Z"),  # Today at local midnight.
            ("c", "2026-11-02T04:59:59Z"),  # Yesterday at 23:59:59 EST.
            ("d", "2026-11-01T04:00:00Z"),  # Yesterday at midnight EDT (25-hour day).
            ("e", "2026-11-01T03:59:59Z"),  # Two calendar days ago.
            ("f", "2026-10-27T04:00:00Z"),  # Six calendar days ago.
            ("1", "2026-10-27T03:59:59Z"),  # Outside the last seven days.
            ("2", "2026-11-03T05:00:00Z"),  # Tomorrow must not match a recent range.
        ]
    ]}))
    answers = iter([choice, "7", ""])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))
    try:
        with monkeypatch.context() as local:
            local.setenv("TZ", "America/Toronto")
            time.tzset()
            local.setattr(script, "datetime", FixedDatetime)
            sessions = asyncio.run(script._list_video_sessions(backend))
            selected = set()
            assert script._prompt_sessions("Sessions", sessions, selected) == [key * 24 for key in expected]
            assert selected == {key * 24 for key in expected}
    finally:
        time.tzset()
    output = capsys.readouterr().out
    assert "2026-11-01 23:59:59 EST" in output
    assert "2026-11-01 00:00:00 EDT" in output
    backend.call_backend_function.assert_awaited_once_with("racing-session/with-videos")


@pytest.mark.parametrize("choice", ["last week", "All sessions with videos", "7"])
def test_guided_range_can_select_all_sessions_without_splitting(downloadable_dataset, monkeypatch, capsys, choice):
    data = downloadable_dataset
    answers = iter([
        choice,  # Select every available session for preparation.
        "", "", "", "1",
    ])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))
    assert script.main([]) == 0
    assert "splits will be assigned when training starts" in capsys.readouterr().out
    records = load_position_manifest(data.downloads)
    assert len(records["a" * 24]) == len(records["b" * 24]) == 3


@pytest.mark.parametrize("interruption", [EOFError, KeyboardInterrupt])
def test_guided_preparation_handles_closed_input_and_interrupts(downloadable_dataset, monkeypatch, capsys, interruption):
    def interrupted(_):
        raise interruption

    monkeypatch.setattr("builtins.input", interrupted)
    assert script.main([]) == 130
    assert "Preparation cancelled" in capsys.readouterr().err
