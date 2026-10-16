from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from app.integrations.backend import client as backend_client
from training.storage.session_download import download_session_pair


SESSION_ID = "0123456789abcdef01234567"
VIDEO_PATH = f"/racing-session/{SESSION_ID}/videos"
CHUNK_PATH = "/racing-session/download/chunk"
CLEANUP_PATH = "/racing-session/download/cleanup"
ROWS = [{"time": 100, "speed": 50}, {"time": 200, "speed": 75}]


class VideoStream(httpx.AsyncByteStream):
    def __init__(self, content, error=None):
        self.content = content
        self.error = error

    async def __aiter__(self):
        yield self.content[:3]
        if self.error:
            raise self.error("Transfer interrupted")
        yield self.content[3:]


@pytest.fixture
def backend(monkeypatch, tmp_path):
    backend = SimpleNamespace(
        requests=[], statuses={}, stream_error=None,
        chunks=[[ROWS[0]], [ROWS[1]]],
        metadata=[{
            "sessionId": SESSION_ID, "chunkCount": 2, "dataPoints": 2,
            "session_name": "Practice", "map": "Spa", "car_name": "GT3",
        }],
        videos=[
            {"index": 0, "name": "../../outside.mp4", "size": 8, "mimeType": "video/mp4", "startedAt": 100},
        ],
    )

    def handle(request):
        backend.requests.append(request)
        path = request.url.path
        if path == "/userinfo/auth/login":
            assert json.loads(request.content) == {"email": "owner@example.com", "password": "password"}
            return httpx.Response(200, json={"access_token": "service-token"})
        assert request.headers["Authorization"] == "Bearer service-token"
        status = backend.statuses.get(path)
        if status:
            return httpx.Response(status, json={"message": "Rejected"})
        if path == VIDEO_PATH:
            return httpx.Response(200, json=backend.videos)
        if path == "/racing-session/download/init":
            assert json.loads(request.content) == {"sessionId": SESSION_ID}
            return httpx.Response(200, json={"downloadId": "download-id", "sessionMetadata": backend.metadata})
        if path == CLEANUP_PATH:
            assert json.loads(request.content) == {"downloadId": "download-id"}
            return httpx.Response(200, json={})
        # Neither telemetry nor a partially downloaded video may be published.
        assert not (tmp_path / SESSION_ID).exists()
        if path == CHUNK_PATH:
            body = json.loads(request.content)
            assert body["sessionId"] == SESSION_ID
            assert body["downloadId"] == "download-id"
            return httpx.Response(200, json=backend.chunks[body["chunkIndex"]])
        if path.startswith(VIDEO_PATH + "/"):
            return httpx.Response(200, stream=VideoStream(b"mp4-data", backend.stream_error))
        raise AssertionError(f"Unexpected backend request: {request.method} {path}")

    client_type = httpx.AsyncClient
    monkeypatch.setattr(
        backend_client.httpx, "AsyncClient",
        lambda **kwargs: client_type(transport=httpx.MockTransport(handle), **kwargs),
    )
    monkeypatch.setattr(backend_client, "settings", SimpleNamespace(
        backend_server_ip="http://backend", backend_proxy_port="7001",
        ai_service_username="owner@example.com", ai_service_password="password",
    ))
    service = backend_client.BackendService()
    return service, backend, tmp_path


def download(service, root, session_id=SESSION_ID):
    return asyncio.run(download_session_pair(service, session_id, root))


def test_download_publishes_telemetry_and_session_video_together(backend):
    service, backend, root = backend
    result = download(service, root)

    saved = root / SESSION_ID
    assert result["session_id"] == SESSION_ID
    assert result["directory"] == str(saved)
    assert json.loads(Path(result["telemetry_file"]).read_text()) == ROWS
    assert result["video_files"] == [str(saved / "video_000.mp4")]
    assert Path(result["video_files"][0]).read_bytes() == b"mp4-data"
    manifest = json.loads(Path(result["manifest_file"]).read_text())
    assert manifest["session"] == backend.metadata[0]
    assert manifest["telemetry_file"] == "telemetry.json"
    assert manifest["videos"] == [{**backend.videos[0], "file": "video_000.mp4"}]
    assert sorted(path.name for path in saved.iterdir()) == [
        "manifest.json", "telemetry.json", "video_000.mp4",
    ]
    assert list(root.iterdir()) == [saved]
    assert backend.requests[-1].url.path == CLEANUP_PATH
    assert sum(request.url.path == "/userinfo/auth/login" for request in backend.requests) == 1
    assert all(request.url.host == "backend" and request.url.port == 7001 for request in backend.requests)


def test_existing_pair_is_preserved_without_backend_requests(backend):
    service, backend, root = backend
    download(service, root)
    original = {path.name: path.read_bytes() for path in (root / SESSION_ID).iterdir()}
    backend.requests.clear()
    with pytest.raises(FileExistsError):
        download(service, root, SESSION_ID.upper())
    assert not backend.requests
    assert {path.name: path.read_bytes() for path in (root / SESSION_ID).iterdir()} == original


def test_backend_credentials_are_required(backend):
    service, backend, root = backend
    service.username = None
    with pytest.raises(RuntimeError, match="Failed to establish backend connection"):
        download(service, root)
    assert not backend.requests
    assert not list(root.iterdir())


@pytest.mark.parametrize("session_id", ["not-an-id", "abc", "z" * 24])
def test_invalid_session_id_is_rejected_before_backend_access(backend, session_id):
    service, backend, root = backend
    with pytest.raises(ValueError, match="Invalid session ID"):
        download(service, root, session_id)
    assert not backend.requests


def test_session_without_video_does_not_download_telemetry(backend):
    service, backend, root = backend
    backend.videos = []
    with pytest.raises(FileNotFoundError, match="no attached MP4 video"):
        download(service, root)
    assert [request.url.path for request in backend.requests] == ["/userinfo/auth/login", VIDEO_PATH]
    assert not list(root.iterdir())


@pytest.mark.parametrize("path,status", [
    (VIDEO_PATH, 401), (VIDEO_PATH, 403), (VIDEO_PATH, 404),
    (CHUNK_PATH, 500), (VIDEO_PATH + "/0", 404), (VIDEO_PATH + "/0", 500),
])
def test_backend_failure_never_leaves_an_unpaired_download(backend, path, status):
    service, backend, root = backend
    backend.statuses[path] = status
    with pytest.raises((RuntimeError, httpx.HTTPStatusError)):
        download(service, root)
    assert not list(root.iterdir())
    if path != VIDEO_PATH:
        assert backend.requests[-1].url.path == CLEANUP_PATH


@pytest.mark.parametrize("error", [httpx.ReadTimeout, httpx.ReadError])
def test_interrupted_video_removes_telemetry_and_partial_files(backend, error):
    service, backend, root = backend
    backend.stream_error = error
    with pytest.raises(error):
        download(service, root)
    assert not list(root.iterdir())
    assert backend.requests[-1].url.path == CLEANUP_PATH


@pytest.mark.parametrize("fault", ["count", "video-size", "wrong-session", "invalid-chunk", "invalid-json", "zero-chunks"])
def test_incomplete_or_mismatched_data_is_not_published(backend, fault):
    service, backend, root = backend
    if fault == "count":
        backend.metadata[0]["dataPoints"] = 3
    elif fault == "video-size":
        backend.videos[0]["size"] = 100
    elif fault == "wrong-session":
        backend.metadata[0]["sessionId"] = "f" * 24
    elif fault == "invalid-chunk":
        backend.chunks[1] = {"error": "Missing telemetry"}
    elif fault == "invalid-json":
        backend.chunks[1] = [42]
    else:
        backend.metadata[0]["chunkCount"] = 0
    with pytest.raises((ValueError, RuntimeError)):
        download(service, root)
    assert not list(root.iterdir())
    assert backend.requests[-1].url.path == CLEANUP_PATH


def test_cleanup_failure_does_not_undo_completed_pair(backend):
    service, backend, root = backend
    backend.statuses[CLEANUP_PATH] = 500
    download(service, root)
    assert (root / SESSION_ID / "telemetry.json").is_file()
    assert (root / SESSION_ID / "video_000.mp4").is_file()
