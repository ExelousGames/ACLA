"""Save position targets beside extracted images for recursive training discovery."""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd

from app.top_laps.shared import bucket_key_from_dataframe, deserialize_top_lap_store
from .position_data import PositionCurve, load_position_manifest, reference_inputs


logger = logging.getLogger(__name__)


def signed_targets(references, driver_position):
    driver = np.asarray(driver_position, dtype=np.float64)
    if driver.shape != (3,) or not np.isfinite(driver).all():
        raise ValueError("Driver position must be a finite XYZ triple (training targets only).")
    # Subtract before converting to float32 so small raw-coordinate differences survive.
    return (references - driver).astype(np.float32)


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def _local_path(folder, name):
    path = (folder / name).resolve()
    if not path.is_relative_to(folder.resolve()):
        raise ValueError(f"Frame bundle path escapes its folder: {name}")
    return path


def _array(value):
    return json.loads(value) if isinstance(value, str) else value


def driver_position(row):
    """Trim raw arrays to active cars, then read the matching player's XYZ."""
    active_cars = _number(row.get("Graphics_active_cars"), "Graphics_active_cars")
    if active_cars < 0 or not active_cars.is_integer():
        raise ValueError("Graphics_active_cars must be a nonnegative integer")
    active_cars = int(active_cars)
    if active_cars == 0:
        return None

    coordinates = _array(row.get("Graphics_car_coordinates"))
    if not isinstance(coordinates, list):
        raise ValueError("Graphics_car_coordinates must be an array")
    coordinates = coordinates[:active_cars]
    ids = _array(row.get("Graphics_car_id"))
    if not isinstance(ids, list):
        return None
    ids = ids[:active_cars]
    player = _number(row.get("Graphics_player_car_id"), "Graphics_player_car_id")
    if player < 0 or not player.is_integer() or ids.count(player) != 1:
        return None
    slot = ids.index(player)
    if slot >= len(coordinates) or not isinstance(coordinates[slot], dict):
        raise ValueError("Graphics_car_coordinates is missing the player's XYZ coordinates")
    return [_number(coordinates[slot].get(axis), f"Player {axis}") for axis in "xyz"]


def _prepare_session(extraction: dict) -> dict | None:
    folder = Path(extraction["directory"]).expanduser().resolve()
    session_id = extraction["session"]["sessionId"]
    source = extraction["session"].get("game_recorded_from")
    games = {"acc": "acc", "iracing": "iracing", "iracing_live": "iracing", "iracing_recorded": "iracing"}
    if source not in games:
        raise ValueError(f"Session {session_id} has unsupported game_recorded_from: {source!r}")
    frames, rows = [], {}
    video = extraction["videos"][0]
    video_folder = _local_path(folder, video["output_folder"])
    pairs = extraction["frames"]
    if len(pairs) != video["matched_frames"]:
        raise ValueError(f"Extracted frame count does not match manifest: {video_folder}")
    if not pairs:
        logger.warning(
            f"Skipping session {session_id}: No frame/telemetry pairs were provided. "
            f"Run frame extraction before preparing position targets. Extraction folder: {video_folder}"
        )
        return None
    for pair in pairs:
        try:
            if pair["session_id"] != session_id or pair["video_index"] != video["index"]:
                raise ValueError("Frame belongs to another session or video")
            row = pair["telemetry"]
            driver_xyz = driver_position(row)
            if driver_xyz is None:
                continue
            if source == "acc":
                # Align raw ACC coordinates with the reference/map axes.
                driver_xyz[1], driver_xyz[2] = driver_xyz[2], driver_xyz[1]
            for field in ("Static_track", "Static_car_model"):
                if not isinstance(row.get(field), str) or not row[field].strip():
                    raise ValueError(f"{field} is required to select an exact reference lap")
            grip = _number(row.get("Graphics_track_grip_status"), "Graphics_track_grip_status")
            if not 0 <= grip <= 6 or not grip.is_integer():
                raise ValueError("Graphics_track_grip_status must be an integer from 0 to 6")
            position = _number(row.get("Graphics_normalized_car_position"), "Graphics_normalized_car_position")
            if not 0 <= position <= 1:
                raise ValueError("Graphics_normalized_car_position must be between 0 and 1")
            frames.append({
                "image": _local_path(video_folder, pair["image"]),
                "normalized_position": position,
                "driver_position": driver_xyz,
            })
            # Frame sampling must not reweight a telemetry row's grip.
            rows[pair["telemetry_row_index"]] = row
        except (ValueError, KeyError, TypeError) as exc:
            raise ValueError(f"{video_folder}, frame {pair.get('frame_index')}: {exc}") from exc
    if not frames:
        first = pairs[0]
        active_cars = first["telemetry"].get("Graphics_active_cars")
        player = first["telemetry"].get("Graphics_player_car_id")
        logger.warning(
            f"Skipping session {session_id}: No usable position-training frames: all {len(pairs)} extracted frames "
            "were skipped because the player's car could not be identified in the aligned telemetry. "
            f"First rejected frame {first['frame_index']} (telemetry row {first['telemetry_row_index']}): "
            f"Graphics_active_cars={active_cars!r}; Graphics_player_car_id={player!r}. "
            "Graphics_player_car_id must be a nonnegative integer appearing exactly once within the first "
            "Graphics_active_cars slots of Graphics_car_id, with corresponding Graphics_car_coordinates. "
            f"Extracted images: {video_folder}"
        )
        return None
    telemetry = pd.DataFrame(rows.values())
    if any(telemetry[field].nunique() != 1 for field in ("Static_track", "Static_car_model")):
        raise ValueError(f"Session {session_id} contains multiple tracks or cars")
    return {
        "id": session_id, "game": games[source],
        "reference_bucket": list(bucket_key_from_dataframe(telemetry, game=source)),
        "frames": frames,
    }


def _track_key(value):
    return value.strip().lower().replace(" ", "_").replace("-", "_")


async def _request(backend, endpoint):
    response = await backend.call_backend_function(endpoint, timeout_seconds=300.0)
    if isinstance(response, dict) and response.get("error"):
        raise RuntimeError(str(response["error"]))
    return response


async def prepare_position_dataset(backend, extractions: list[dict]) -> dict:
    """Prepare frame matches, skipping sessions without usable frames, a reference lap, or a map."""
    if not extractions:
        raise ValueError("Provide at least one session")
    ids = [extraction["session"]["sessionId"] for extraction in extractions]
    if len(set(ids)) != len(ids):
        raise ValueError("Each session must appear once")
    sessions, skipped = [], []
    for extraction in extractions:
        session = _prepare_session(extraction)
        if session is None:
            skipped.append(extraction["session"]["sessionId"])
        else:
            sessions.append(session)
    if not sessions:
        return {"frames": 0, "sessions": [], "skipped_sessions": skipped}

    model = await backend.getCompleteActiveModelData("top_lap_reference")
    store = deserialize_top_lap_store(model.modelData)
    prepared_sessions = []
    maps, listings = {}, {}
    for session in sessions:
        bucket = tuple(session["reference_bucket"])
        if bucket not in store.entries:
            available = [key for key in store.entries if key[:3] == bucket[:3]]
            logger.warning(
                "Skipping session %s: No exact reference lap for %s. Available buckets for this game/track/car: %s",
                session["id"], bucket, available,
            )
            skipped.append(session["id"])
            continue
        reference_lap = PositionCurve.from_top_lap(store.entries[bucket])

        game, track = session["game"], _track_key(session["reference_bucket"][1])
        if game not in listings:
            listings[game] = (await _request(backend, f"circuit-map/list?game={game}"))["list"]
        # The list is newest-first. Match only the source track key.
        matches = [item for item in listings[game] if _track_key(item.get("source_track_key") or "") == track]
        if not matches:
            logger.warning("Skipping session %s: No circuit map for %s/%s", session["id"], game, track)
            skipped.append(session["id"])
            continue
        map_id = matches[0]["id"]
        if map_id not in maps:
            detail = await _request(backend, f"circuit-map/{map_id}")
            if detail["game"] != game or _track_key(detail.get("source_track_key") or "") != track:
                raise ValueError(f"Circuit map {map_id} does not match {game}/{track}")
            maps[map_id] = PositionCurve.from_points(detail["samples"]["middle_line"])
        prepared_frames = []
        for frame in session["frames"]:
            references = reference_inputs(frame["normalized_position"], reference_lap, maps[map_id])
            prepared_frames.append({
                "image": frame["image"],
                "references": references.tolist(),
                "target": signed_targets(references, frame["driver_position"]).tolist(),
            })
        session["frames"] = prepared_frames
        prepared_sessions.append(session)

    if not prepared_sessions:
        return {"frames": 0, "sessions": [], "skipped_sessions": skipped}

    created = []
    try:
        for session in prepared_sessions:
            for frame in session["frames"]:
                image = frame["image"]
                position_path = image.with_suffix(".position.json")
                with position_path.open("x", encoding="utf-8") as target:
                    created.append(position_path)
                    json.dump({
                        "session_id": session["id"], "image": image.name,
                        "references": frame["references"], "target": frame["target"],
                    }, target, indent=2, allow_nan=False)
                    target.write("\n")
        # Validate the saved position files and images through folder discovery.
        for folder in sorted({path.parent for path in created}):
            load_position_manifest(folder)
    except Exception:
        for path in created:
            path.unlink()
        raise
    return {
        "frames": len(created),
        "sessions": [{key: value for key, value in session.items() if key != "frames"} for session in prepared_sessions],
        "skipped_sessions": skipped,
    }
