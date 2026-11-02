"""Prepare image/world XYZ and track context with car-local offset targets.

Run without arguments for step-by-step prompts.
Downloads are saved under session_downloads/session_downloaded_by_prepare_position_dataset.
The main menu also copies saved Labelme annotations, their images, and available
.position.json files into session_downloads/session image segmentation samples.
Existing image folders get numbered siblings, such as video_000_1 and video_000_2.
Telemetry is used during preparation; the trainer receives only prepared inputs
and targets in a .position.json file beside each image, discovered recursively
during training. Heading/pitch/roll rotate reference-minus-driver offsets into
local XYZ (right, forward, up), using Rz(heading) Rx(pitch) Ry(roll)
directly in the corrected world axes, then negating ACC's left-positive local X.
Each position file also saves orientation: [heading, pitch, roll] in radians
from the aligned telemetry row, as required by the trainer. Driver world XYZ
is retained only for the annotation map and is not a model input.
Local targets currently support ACC only; iRacing sessions are skipped because
their recorded position and heading fields use different world frames.
Images are randomly split within each game/map when training starts, with 20% for validation.
Reference lines and centerline maps are downloaded from the backend.
Game and map names come from the selected centerline map's game and circuit_name.
"""

from __future__ import annotations

import asyncio
from bisect import bisect_left
from datetime import datetime, timedelta
import json
import math
from pathlib import Path
import random
import re
import shutil
import sys
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.storage.session_download import SESSION_DOWNLOAD_DIR, download_session_pair


TIME_FIELD = "Graphics_analyst_session_time"


def move_annotated_frames() -> dict:
    """Copy annotated frames locally, preserving originals, paths, and position data."""
    destination = SESSION_DOWNLOAD_DIR.parent / "session image segmentation samples"
    output_directories = {}
    duplicated_frames = 0
    copies, sessions = [], []
    for folder in sorted(SESSION_DOWNLOAD_DIR.glob("*")):
        if not re.fullmatch(r"[0-9a-fA-F]{24}", folder.name) or not folder.is_dir():
            continue
        count = 0
        for annotation_path in sorted((folder / "frames").rglob("*.json")):
            if annotation_path.name.endswith(".position.json"):
                continue
            annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
            if not isinstance(annotation, dict) or not {"imagePath", "shapes"} <= annotation.keys():
                continue
            filename = annotation["imagePath"]
            if not isinstance(filename, str) or not filename or not isinstance(annotation["shapes"], list):
                raise ValueError(f"Invalid Labelme annotation: {annotation_path}")
            image = annotation_path.parent / filename
            if (Path(filename).is_absolute() or image.parent.resolve() != annotation_path.parent.resolve()
                    or image.stem != annotation_path.stem):
                raise ValueError(f"Annotation must reference a matching image in the same folder: {annotation_path}")
            if not image.is_file():
                raise FileNotFoundError(f"Annotated image is missing: {image}")
            files = [image, annotation_path]
            position = image.with_suffix(".position.json")
            if position.is_file():
                files.append(position)
            directory = destination / image.parent.relative_to(SESSION_DOWNLOAD_DIR)
            if directory not in output_directories:
                target_directory = directory
                number = 1
                while (target_directory.exists() or target_directory.is_symlink()
                       or target_directory in output_directories.values()):
                    target_directory = directory.with_name(f"{directory.name}_{number}")
                    number += 1
                output_directories[directory] = target_directory
            target_directory = output_directories[directory]
            copies.append([(source, target_directory / source.name) for source in files])
            count += 1
            duplicated_frames += int(target_directory != directory)
        if count:
            sessions.append({"id": folder.name, "frames": count})

    for pair in copies:
        pair[0][1].parent.mkdir(parents=True, exist_ok=True)
        copied = []
        try:
            for source, target in pair:
                copied.append(target)
                shutil.copy2(source, target)
        except OSError:
            # Keep an image, annotation, and optional position file together on failure.
            for target in reversed(copied):
                target.unlink(missing_ok=True)
            raise
    return {
        "frames": len(copies), "sessions": sessions, "directory": str(destination),
        "duplicated_frames": duplicated_frames,
        "duplicated_directories": [
            str(target) for directory, target in output_directories.items() if target != directory
        ],
    }


def _finite_number(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def _validate_frame_interval(interval: tuple[float, float] | None) -> None:
    if interval is None:
        return
    minimum, maximum = (_finite_number(value, "Frame interval") for value in interval)
    if minimum <= 0 or maximum < minimum:
        raise ValueError("Frame interval must be positive, with the minimum no greater than the maximum")


def _bundle_file(directory: Path, filename: str) -> Path:
    if not isinstance(filename, str) or Path(filename).name != filename:
        raise ValueError("Manifest files must be filenames inside the downloaded session")
    path = directory / filename
    if not path.is_file():
        raise FileNotFoundError(f"Downloaded file is missing: {path}")
    return path


def extract_session_frames(
    session_folder: Path,
    output_folder: Path | None = None,
    *,
    frame_interval: tuple[float, float] | None = None,
    video_offset: float | None = None,
    max_delta_ms: float = 100.0,
) -> dict:
    """Use video presentation times and nearest analyst-time rows, without interpolation.

    The offset is seconds from the first telemetry row to the session video's
    start. By default, the first video frame aligns with the first telemetry row.
    Frame/telemetry matches are returned in memory for position preparation.
    Frame intervals are (minimum, maximum) seconds; equal bounds are fixed.
    Each next sample uses the first frame at or after a uniformly drawn interval.
    """
    _validate_frame_interval(frame_interval)
    if _finite_number(max_delta_ms, "max_delta_ms") < 0:
        raise ValueError("max_delta_ms must be nonnegative")
    session_folder = session_folder.expanduser().resolve()
    destination = (output_folder or session_folder / "frames").expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"Output already exists: {destination}")

    manifest = json.loads((session_folder / "manifest.json").read_text(encoding="utf-8"))
    if not manifest["videos"]:
        raise ValueError("Session manifest has no video")
    video = manifest["videos"][0]
    video_path = _bundle_file(session_folder, video["file"])
    offset = 0.0 if video_offset is None else _finite_number(video_offset, "Video offset")

    rows = json.loads(_bundle_file(session_folder, manifest["telemetry_file"]).read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError("Telemetry must be a nonempty JSON array")
    times = []
    for index, row in enumerate(rows):
        value = row.get(TIME_FIELD) if isinstance(row, dict) else None
        times.append(_finite_number(value, f"Telemetry row {index} {TIME_FIELD}"))
    if any(right < left for left, right in zip(times, times[1:])):
        raise ValueError(f"{TIME_FIELD} moves backwards; use a session with a continuous analyst clock")
    # Match on relative milliseconds to keep the alignment tolerance in one unit.
    relative_times = [(value - times[0]) * 1000 for value in times]

    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required: python -m pip install opencv-python") from exc

    summary = {
        "session": manifest["session"],
        "source_manifest": str(session_folder / "manifest.json"),
        "telemetry_time_field": TIME_FIELD,
        "first_telemetry_time_seconds": times[0],
        "frame_interval_seconds": frame_interval,
        "max_delta_ms": max_delta_ms,
    }
    pairs = []
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with TemporaryDirectory(prefix=f".{destination.name}-", dir=destination.parent) as temporary:
            staging = Path(temporary)
            folder = staging / f"video_{video['index']:03d}"
            folder.mkdir()
            capture = cv2.VideoCapture(str(video_path))
            frame_index = saved = skipped = 0
            previous_ms = -1.0
            first_frame_ms = None
            next_sample_ms = 0.0
            try:
                if not capture.isOpened():
                    raise OSError(f"Cannot open video: {video_path}")
                while True:
                    success, frame = capture.read()
                    if not success:
                        break
                    presentation_ms = capture.get(cv2.CAP_PROP_POS_MSEC)
                    if not math.isfinite(presentation_ms) or presentation_ms < 0 or presentation_ms <= previous_ms:
                        raise ValueError(f"Video has missing or non-increasing frame timestamps: {video_path}")
                    previous_ms = presentation_ms
                    if first_frame_ms is None:
                        first_frame_ms = presentation_ms
                    video_ms = presentation_ms - first_frame_ms
                    source_index = frame_index
                    frame_index += 1
                    if frame_interval is not None:
                        if video_ms + 1e-6 < next_sample_ms:
                            continue
                        next_sample_ms = video_ms + random.uniform(*frame_interval) * 1000
                    target_ms = offset * 1000 + video_ms
                    if target_ms < relative_times[0] or target_ms > relative_times[-1]:
                        skipped += 1
                        continue
                    insertion = bisect_left(relative_times, target_ms)
                    candidates = [i for i in (insertion - 1, insertion) if 0 <= i < len(rows)]
                    row_index = min(candidates, key=lambda i: abs(relative_times[i] - target_ms))
                    delta_ms = relative_times[row_index] - target_ms
                    if abs(delta_ms) > max_delta_ms:
                        skipped += 1
                        continue

                    image_path = folder / f"frame_{source_index:06d}.jpg"
                    if not cv2.imwrite(str(image_path), frame):
                        raise OSError(f"Cannot write image: {image_path}")
                    pair = {
                        "image": image_path.name,
                        "session_id": manifest["session"]["sessionId"],
                        "video_index": video["index"],
                        "video_file": video["file"],
                        "frame_index": source_index,
                        "video_time_ms": video_ms,
                        "video_offset_seconds": offset,
                        "telemetry_row_index": row_index,
                        "telemetry_time_seconds": times[row_index],
                        "alignment_delta_ms": delta_ms,
                        "telemetry": rows[row_index],
                    }
                    pairs.append(pair)
                    saved += 1
                if frame_index == 0:
                    raise OSError(f"No frames could be read from: {video_path}")
            finally:
                capture.release()
            summary["videos"] = [{
                **video,
                "output_folder": folder.name,
                "offset_seconds": offset,
                "offset_source": "explicit" if video_offset is not None else "first_frame_alignment",
                "first_frame_time_ms": first_frame_ms,
                "frames_read": frame_index,
                "matched_frames": saved,
                "unmatched_frames": skipped,
            }]
            if not saved:
                raise ValueError("No frames matched telemetry; check the video offset and maximum time delta")
            (staging / "manifest.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
            if destination.exists():
                raise FileExistsError(f"Output already exists: {destination}")
            staging.rename(destination)
    except cv2.error as exc:
        raise OSError(str(exc)) from exc
    return {"directory": str(destination), **summary, "frames": pairs}


def _video_offset(value: str) -> tuple[str, float]:
    try:
        session_id, seconds_text = value.split("=", 1)
        seconds = float(seconds_text)
        if not re.fullmatch(r"[0-9a-fA-F]{24}", session_id) or not math.isfinite(seconds):
            raise ValueError
        return session_id.lower(), seconds
    except ValueError as exc:
        raise ValueError("Use SESSION_ID=SECONDS") from exc


async def prepare_sessions(
    backend, selected_sessions: list[str], *,
    frame_interval: tuple[float, float] | None = None, video_offsets: list[tuple[str, float]] | None = None,
    max_delta_ms: float | None = None,
) -> dict:
    """Skip existing sessions; download new videos and create image/position JSON pairs."""
    if not selected_sessions:
        raise ValueError("Provide at least one session")
    _validate_frame_interval(frame_interval)
    if max_delta_ms is not None and _finite_number(max_delta_ms, "max_delta_ms") < 0:
        raise ValueError("max_delta_ms must be nonnegative")

    sessions, ids, skipped = [], set(), []
    for value in selected_sessions:
        is_id = bool(re.fullmatch(r"[0-9a-fA-F]{24}", value))
        folder = (SESSION_DOWNLOAD_DIR / value.lower() if is_id else Path(value)).expanduser().resolve()
        session_id = value.lower() if is_id else str(folder)
        if session_id in ids:
            raise ValueError("Each session must appear once")
        ids.add(session_id)
        if folder.exists():
            skipped.append(session_id)
            continue
        if not is_id:
            raise FileNotFoundError(f"Session folder not found: {folder}")
        sessions.append((session_id, folder))
    offsets = {}
    for session_id, seconds in video_offsets or ():
        if session_id not in ids:
            raise ValueError(f"Video offset refers to an unselected session: {session_id}")
        if session_id in offsets:
            raise ValueError(f"Duplicate video offset: {session_id}")
        offsets[session_id] = seconds

    if not sessions:
        return {"frames": 0, "sessions": [], "annotation_directories": [], "skipped_sessions": skipped}

    from training.image_segmentation.position_preparation import prepare_position_dataset

    extractions = []
    for session_id, folder in sessions:
        await download_session_pair(backend, session_id, SESSION_DOWNLOAD_DIR)
        extractions.append(extract_session_frames(
            folder, frame_interval=frame_interval, video_offset=offsets.get(session_id),
            max_delta_ms=100.0 if max_delta_ms is None else max_delta_ms,
        ))

    result = await prepare_position_dataset(backend, extractions)
    prepared_ids = {session["id"] for session in result["sessions"]}
    result["annotation_directories"] = [str(folder / "frames") for session_id, folder in sessions if session_id in prepared_ids]
    result["skipped_sessions"] = skipped + result["skipped_sessions"]
    return result


def _prompt_choice(label: str, options: list[str]) -> int:
    print(label)
    for index, option in enumerate(options, 1):
        print(f"  {index}. {option}")
    while True:
        value = (input("Choose an option by number or name [1]: ").strip() or "1").casefold()
        for index, option in enumerate(options, 1):
            if value in (str(index), option.casefold()):
                return index
        print(f"Enter a number from 1 to {len(options)}, or an option name.")


def _prompt_frame_interval() -> tuple[float, float] | None:
    number = r"[+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
    while True:
        value = input("Seconds between frames (number or min - max) [Enter: all frames]: ").strip()
        if not value:
            return None
        try:
            match = re.fullmatch(rf"({number})(?:\s*-\s*({number}))?", value)
            if match is None:
                raise ValueError
            minimum, maximum = match.groups()
            interval = (float(minimum), float(maximum or minimum))
            _validate_frame_interval(interval)
            return interval
        except ValueError:
            print("Enter a positive, finite number or range such as 0.5 - 1.5 (min <= max), or press Enter for all frames.")


def _prompt_number(label: str) -> float | None:
    while True:
        value = input(label).strip()
        if not value:
            return None
        try:
            number = _finite_number(float(value), "Value")
            if number < 0:
                raise ValueError
            return number
        except ValueError:
            print("Enter a finite nonnegative number, or press Enter for the default.")


async def _list_video_sessions(backend) -> list[tuple[str, datetime]]:
    response = await backend.call_backend_function("racing-session/with-videos")
    if isinstance(response, dict) and response.get("error"):
        raise RuntimeError(response["error"])
    if not isinstance(response, dict) or not isinstance(response.get("list"), list):
        raise ValueError("Invalid video session listing from backend")
    sessions = []
    for session in response["list"]:
        session_id = session["sessionId"]
        if not re.fullmatch(r"[0-9a-fA-F]{24}", session_id):
            raise ValueError("Invalid session ID in video session listing")
        created = datetime.fromisoformat(session["created_date"].replace("Z", "+00:00")).astimezone()
        sessions.append((session_id.lower(), created))
    if not sessions:
        raise ValueError("No sessions with attached videos are available for the configured backend account")
    return sessions


def _prompt_sessions(label: str, sessions: list[tuple[str, datetime]], selected: set[str]) -> list[str]:
    print(label)
    choices = {}
    for index, (session_id, created) in enumerate(sessions, 1):
        choices[str(index)] = session_id
        status = " (already selected)" if session_id in selected else ""
        print(f"  {index}. {created:%Y-%m-%d %H:%M:%S %Z} | {session_id}{status}")
    today = datetime.now().astimezone().date()
    print(f"Dates use local time; today is {today}. Last 2/3 days and last week (7 days) include today.")
    while True:
        period = _prompt_choice("Which days do you want to download?", [
            "Today", "Yesterday", "Last 2 days", "Last 3 days", "Last week", "Pick a session",
            "All sessions with videos",
        ])
        if period == 7:
            matches = [session_id for session_id, _ in sessions if session_id not in selected]
            selected.update(matches)
            print(f"Selected all {len(matches)} available video session(s): {', '.join(matches)}")
            return matches
        if period == 6:
            break
        end = today - timedelta(days=1) if period == 2 else today
        start = end - timedelta(days={1: 0, 2: 0, 3: 1, 4: 2, 5: 6}[period])
        matches = [session_id for session_id, created in sessions
                   if session_id not in selected and start <= created.date() <= end]
        if not matches:
            print(f"No unselected video sessions from {start} to {end}. Choose another range or pick a session.")
            continue
        selected.update(matches)
        print(f"Selected {len(matches)} session(s) from {start} to {end}: {', '.join(matches)}")
        return matches

    print("Choose a listed session number or ID.")
    print("Add one session at a time. Press Enter when the selection is complete.")
    picked = []
    while True:
        value = input("Session: ").strip()
        if not value:
            if picked:
                return picked
            print("Select at least one session.")
            continue
        session_id = choices.get(value, value.lower())
        if session_id not in choices.values():
            print("Choose a session from the video session listing.")
            continue
        if session_id in selected:
            print("Each session must appear once.")
            continue
        selected.add(session_id)
        picked.append(session_id)
        print(f"Added {session_id}.")


async def _interactive_options() -> dict | None:
    from training.integrations.backend import backend_service

    print("Position dataset preparation. Press Enter to accept defaults; Ctrl+C to cancel.\n")
    sessions = await _list_video_sessions(backend_service)
    selected = set()
    selected_sessions = _prompt_sessions("Step 1: Sessions with videos", sessions, selected)
    print("Sessions whose local folders already exist will be skipped.")
    print("\nStep 2: Frame sampling")
    frame_interval = _prompt_frame_interval()
    print("\nStep 3: Telemetry alignment")
    max_delta_ms = _prompt_number("Maximum time difference in ms [Enter: 100]: ")
    print("\nStep 4: Optional video start offsets")
    print("Default: align each session's video with its first telemetry row.")
    print("Enter SESSION_ID=SECONDS, one per line, or press Enter to continue.")
    offsets = {}
    while True:
        value = input("Video offset: ").strip()
        if not value:
            break
        try:
            session_id, seconds = _video_offset(value)
            if session_id not in selected:
                raise ValueError("Choose a session selected for preparation")
            if session_id in offsets:
                raise ValueError("An offset for that session was already entered")
        except ValueError as exc:
            print(f"Invalid offset: {exc}")
            continue
        offsets[session_id] = seconds

    print("\nPreparation summary")
    print(f"Sessions: {', '.join(selected_sessions)}")
    print("Training and validation splits will be assigned when training starts.")
    if frame_interval is None:
        print("Frame interval: all frames")
    elif frame_interval[0] == frame_interval[1]:
        print(f"Frame interval: {frame_interval[0]:g} seconds")
    else:
        print(f"Frame interval: random {frame_interval[0]:g} - {frame_interval[1]:g} seconds")
    print(f"Maximum time difference: {max_delta_ms if max_delta_ms is not None else '100'} ms")
    for session_id, seconds in offsets.items():
        print(f"Video offset: {session_id}={seconds} seconds")
    if _prompt_choice("Start preparation?", ["Start", "Cancel"]) == 2:
        return None
    return {
        "selected_sessions": selected_sessions, "frame_interval": frame_interval,
        "video_offsets": list(offsets.items()), "max_delta_ms": max_delta_ms,
    }


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv:
        print("Run without command-line options to use the interactive menu.", file=sys.stderr)
        return 2

    async def run() -> int:
        choice = _prompt_choice("Position dataset main menu", [
            "Download sessions", "Copy annotated images", "Exit",
        ])
        if choice == 3:
            return 0
        if choice == 2:
            print(json.dumps(move_annotated_frames(), indent=2))
            return 0
        options = await _interactive_options()
        if options is None:
            print("Preparation cancelled.")
            return 0

        from training.integrations.backend import backend_service

        result = await prepare_sessions(backend_service, **options)
        print(json.dumps(result, indent=2))
        return 0

    try:
        return asyncio.run(run())
    except (EOFError, KeyboardInterrupt):
        print("\nPreparation cancelled.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"Preparation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
