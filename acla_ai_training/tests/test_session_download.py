from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient

from scripts import download_session
from training.api import session_download
from training.api.app import app


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
def api(monkeypatch, tmp_path):
    backend = SimpleNamespace(
        requests=[], statuses={}, stream_error=None,
        chunks=[[ROWS[0]], [ROWS[1]]],
        metadata=[{
            "sessionId": SESSION_ID, "chunkCount": 2, "dataPoints": 2,
            "session_name": "Practice", "map": "Spa", "car_name": "GT3",
        }],
        videos=[
            {"index": 0, "name": "../../outside.mp4", "size": 8, "mimeType": "video/mp4", "startedAt": 100},
            {"index": 2, "name": "second.mp4", "size": 8, "mimeType": "video/mp4", "startedAt": 200},
        ],
    )

    def handle(request):
        backend.requests.append(request)
        assert request.headers["Authorization"] == "Bearer owner-token"
        path = request.url.path
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
        session_download.httpx, "AsyncClient",
        lambda **kwargs: client_type(transport=httpx.MockTransport(handle), **kwargs),
    )
    app.dependency_overrides[session_download.get_backend_url] = lambda: "http://backend:7001/"
    app.dependency_overrides[session_download.get_download_directory] = lambda: tmp_path
    try:
        with TestClient(app) as client:
            yield client, backend, tmp_path
    finally:
        app.dependency_overrides.pop(session_download.get_backend_url, None)
        app.dependency_overrides.pop(session_download.get_download_directory, None)


def download(client, session_id=SESSION_ID):
    return client.post(
        f"/racing-session/{session_id}/download",
        headers={"Authorization": "Bearer owner-token"},
    )


def test_download_publishes_telemetry_and_all_videos_together(api):
    client, backend, root = api
    response = download(client)

    assert response.status_code == 201, response.text
    saved = root / SESSION_ID
    result = response.json()
    assert result["session_id"] == SESSION_ID
    assert result["directory"] == str(saved)
    assert json.loads(Path(result["telemetry_file"]).read_text()) == ROWS
    assert result["video_files"] == [str(saved / "video_000.mp4"), str(saved / "video_002.mp4")]
    assert all(Path(path).read_bytes() == b"mp4-data" for path in result["video_files"])
    manifest = json.loads(Path(result["manifest_file"]).read_text())
    assert manifest["session"] == backend.metadata[0]
    assert manifest["telemetry_file"] == "telemetry.json"
    assert manifest["videos"] == [
        {**video, "file": filename}
        for video, filename in zip(backend.videos, ["video_000.mp4", "video_002.mp4"])
    ]
    assert sorted(path.name for path in saved.iterdir()) == [
        "manifest.json", "telemetry.json", "video_000.mp4", "video_002.mp4",
    ]
    assert list(root.iterdir()) == [saved]
    assert backend.requests[-1].url.path == CLEANUP_PATH


def test_existing_pair_is_preserved_without_backend_requests(api):
    client, backend, root = api
    assert download(client).status_code == 201
    original = {path.name: path.read_bytes() for path in (root / SESSION_ID).iterdir()}
    backend.requests.clear()
    assert download(client, SESSION_ID.upper()).status_code == 409
    assert not backend.requests
    assert {path.name: path.read_bytes() for path in (root / SESSION_ID).iterdir()} == original


def test_owner_token_is_required(api):
    client, backend, root = api
    response = client.post(f"/racing-session/{SESSION_ID}/download")
    assert response.status_code == 401
    assert not backend.requests
    assert not list(root.iterdir())


@pytest.mark.parametrize("session_id", ["not-an-id", "abc", "z" * 24])
def test_invalid_session_id_is_rejected_before_backend_access(api, session_id):
    client, backend, _ = api
    assert download(client, session_id).status_code == 422
    assert not backend.requests


def test_session_without_video_does_not_download_telemetry(api):
    client, backend, root = api
    backend.videos = []
    assert download(client).status_code == 404
    assert len(backend.requests) == 1
    assert not list(root.iterdir())


@pytest.mark.parametrize("path,status,expected", [
    (VIDEO_PATH, 401, 401), (VIDEO_PATH, 403, 403), (VIDEO_PATH, 404, 404),
    (CHUNK_PATH, 500, 502), (VIDEO_PATH + "/2", 404, 404),
    (VIDEO_PATH + "/2", 500, 502),
])
def test_backend_failure_never_leaves_an_unpaired_download(api, path, status, expected):
    client, backend, root = api
    backend.statuses[path] = status
    assert download(client).status_code == expected
    assert not list(root.iterdir())
    if path != VIDEO_PATH:
        assert backend.requests[-1].url.path == CLEANUP_PATH


@pytest.mark.parametrize("error,status", [(httpx.ReadTimeout, 504), (httpx.ReadError, 502)])
def test_interrupted_video_removes_telemetry_and_partial_files(api, error, status):
    client, backend, root = api
    backend.stream_error = error
    assert download(client).status_code == status
    assert not list(root.iterdir())
    assert backend.requests[-1].url.path == CLEANUP_PATH


@pytest.mark.parametrize("fault", ["count", "video-size", "wrong-session", "invalid-chunk", "invalid-json", "zero-chunks"])
def test_incomplete_or_mismatched_data_is_not_published(api, fault):
    client, backend, root = api
    if fault == "count":
        backend.metadata[0]["dataPoints"] = 3
    elif fault == "video-size":
        backend.videos[1]["size"] = 100
    elif fault == "wrong-session":
        backend.metadata[0]["sessionId"] = "f" * 24
    elif fault == "invalid-chunk":
        backend.chunks[1] = {"error": "Missing telemetry"}
    elif fault == "invalid-json":
        backend.chunks[1] = [42]
    else:
        backend.metadata[0]["chunkCount"] = 0
    assert download(client).status_code == 502
    assert not list(root.iterdir())
    assert backend.requests[-1].url.path == CLEANUP_PATH


def test_cleanup_failure_does_not_undo_completed_pair(api):
    client, backend, root = api
    backend.statuses[CLEANUP_PATH] = 500
    assert download(client).status_code == 201
    assert (root / SESSION_ID / "telemetry.json").is_file()
    assert (root / SESSION_ID / "video_002.mp4").is_file()


@pytest.mark.parametrize("status,exit_code", [(201, 0), (404, 1), (409, 1), (502, 1)])
def test_script_calls_local_api_with_owner_token(monkeypatch, capsys, status, exit_code):
    monkeypatch.setenv("BACKEND_ACCESS_TOKEN", "owner-token")
    requests = []

    def handle(request):
        requests.append(request)
        return httpx.Response(status, json={"directory": "/app/storage/session_downloads/session"})

    client_type = httpx.Client
    monkeypatch.setattr(
        download_session.httpx, "Client",
        lambda **kwargs: client_type(transport=httpx.MockTransport(handle), **kwargs),
    )
    assert download_session.main([SESSION_ID, "--api-url", "http://localhost:8002/"]) == exit_code
    assert str(requests[0].url) == f"http://localhost:8002/racing-session/{SESSION_ID}/download"
    assert requests[0].headers["Authorization"] == "Bearer owner-token"
    output = capsys.readouterr()
    assert "owner-token" not in output.out + output.err
    assert bool(output.err) == bool(exit_code)


def test_script_requires_owner_token(monkeypatch):
    monkeypatch.delenv("BACKEND_ACCESS_TOKEN", raising=False)
    with pytest.raises(SystemExit) as error:
        download_session.main([SESSION_ID])
    assert error.value.code == 2
