"""Save complete backend telemetry/video bundles in local training storage."""

from __future__ import annotations

import errno
import json
import logging
import re
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)
SESSION_DOWNLOAD_DIR = Path(__file__).resolve().parents[2] / "storage" / "session_downloads"


class _Video(BaseModel):
    index: int = Field(ge=0)
    name: str
    size: int = Field(gt=0)
    mimeType: Literal["video/mp4"]
    startedAt: int = Field(ge=0)


class _Session(BaseModel):
    model_config = ConfigDict(extra="allow")

    sessionId: str
    chunkCount: int = Field(gt=0)
    dataPoints: int = Field(gt=0)


class _Download(BaseModel):
    downloadId: str = Field(min_length=1)
    sessionMetadata: list[_Session]


async def _request(backend_service, endpoint: str, method: str = "GET", data: dict | None = None):
    result = await backend_service.call_backend_function(endpoint, method, data, timeout_seconds=300.0)
    if isinstance(result, dict) and result.get("error"):
        raise RuntimeError(str(result["error"]))
    return result


async def download_session_pair(backend_service, session_id: str, storage_dir: Path) -> dict:
    """Reuse backend authentication; publish only after every file succeeds."""
    if not re.fullmatch(r"[0-9a-fA-F]{24}", session_id):
        raise ValueError("Invalid session ID")
    session_id = session_id.lower()
    destination = storage_dir / session_id
    if destination.exists():
        raise FileExistsError(f"Session download already exists: {destination}")

    # The backend lists the session's single recording in an array.
    videos = await _request(backend_service, f"racing-session/{session_id}/videos")
    if not videos:
        raise FileNotFoundError("Session has no attached MP4 video")
    video = _Video.model_validate(videos[0])

    initialization = await _request(
        backend_service, "racing-session/download/init", "POST", {"sessionId": session_id},
    )
    download_id = initialization.get("downloadId") if isinstance(initialization, dict) else None
    try:
        download = _Download.model_validate(initialization)
        if not download.sessionMetadata:
            raise FileNotFoundError("Session telemetry was not found")
        if len(download.sessionMetadata) != 1 or download.sessionMetadata[0].sessionId != session_id:
            raise ValueError("Backend returned telemetry for a different session")
        session = download.sessionMetadata[0]

        storage_dir.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix=f".{session_id}-", dir=storage_dir) as temporary:
            staging = Path(temporary)
            count = 0
            # Keep at most one backend telemetry chunk in memory.
            with (staging / "telemetry.json").open("w", encoding="utf-8") as output:
                output.write("[")
                for index in range(session.chunkCount):
                    rows = await _request(backend_service, "racing-session/download/chunk", "POST", {
                        "downloadId": download.downloadId,
                        "sessionId": session_id,
                        "chunkIndex": index,
                    })
                    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
                        raise ValueError("Backend telemetry chunk must contain a list of records")
                    for row in rows:
                        if count:
                            output.write(",")
                        json.dump(row, output, allow_nan=False)
                        count += 1
                output.write("]\n")
            if count != session.dataPoints:
                raise ValueError("Downloaded telemetry count does not match session metadata")

            # Backend filenames are metadata, never local filesystem paths.
            filename = f"video_{video.index:03d}.mp4"
            size = 0
            with (staging / filename).open("wb") as output:
                async for chunk in backend_service.iter_session_video(session_id, video.index):
                    size += len(chunk)
                    output.write(chunk)
            if size != video.size:
                raise ValueError("Downloaded video size does not match metadata")

            manifest = {
                "session": session.model_dump(),
                "telemetry_file": "telemetry.json",
                "videos": [{**video.model_dump(), "file": filename}],
            }
            (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
            # A single rename makes telemetry and the video visible together.
            if destination.exists():
                raise FileExistsError(f"Session download already exists: {destination}")
            try:
                staging.rename(destination)
            except OSError as exc:
                if exc.errno in (errno.EEXIST, errno.ENOTEMPTY):
                    raise FileExistsError(f"Session download already exists: {destination}") from exc
                raise

        return {
            "session_id": session_id,
            "directory": str(destination),
            "telemetry_file": str(destination / "telemetry.json"),
            "video_files": [str(destination / filename)],
            "manifest_file": str(destination / "manifest.json"),
        }
    finally:
        if download_id:
            try:
                await _request(
                    backend_service, "racing-session/download/cleanup", "POST", {"downloadId": download_id},
                )
            except Exception:
                # The backend also expires these handles; cleanup cannot undo a saved pair.
                logger.warning("Could not release backend download handle for session %s", session_id)
