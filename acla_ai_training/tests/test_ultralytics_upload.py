from __future__ import annotations

import json
from email import policy
from email.parser import BytesParser
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi.testclient import TestClient

from training import model_publication
from training.api import ultralytics
from training.api.app import app


METADATA = {
    "name": "track-segments",
    "task": "segment",
    "classNames": ["track", "curb", "grass", "car", "other"],
    "metadata": {"baseModel": "yolo11n-seg.pt", "epochs": 50, "metrics": {"map50": 0.92}},
}
RECORD = {"_id": "model-id", "framework": "ultralytics", **METADATA}


@pytest.fixture
def api(monkeypatch):
    backend = SimpleNamespace(
        base_url="http://backend", base_port="7001", token="service-token",
        ensure_connection=AsyncMock(return_value=True),
        establish_connection=AsyncMock(return_value=True),
        requests=[], responses=[201],
    )
    backend.get_auth_headers = lambda: {"Authorization": f"Bearer {backend.token}"}

    def handle(request):
        message = BytesParser(policy=policy.default).parsebytes(
            f"Content-Type: {request.headers['content-type']}\r\n\r\n".encode() + request.content
        )
        parts = {
            part.get_param("name", header="content-disposition"): part
            for part in message.iter_parts()
        }
        backend.requests.append((request, parts))
        result = backend.responses.pop(0)
        if isinstance(result, type) and issubclass(result, httpx.RequestError):
            raise result("Backend unavailable", request=request)
        return httpx.Response(result, json=RECORD if result == 201 else {"message": "Rejected"})

    client_type = httpx.AsyncClient
    monkeypatch.setattr(
        model_publication.httpx, "AsyncClient",
        lambda **kwargs: client_type(transport=httpx.MockTransport(handle), **kwargs),
    )
    app.dependency_overrides[ultralytics.get_backend_service] = lambda: backend
    try:
        with TestClient(app) as client:
            yield client, backend
    finally:
        app.dependency_overrides.pop(ultralytics.get_backend_service, None)


def upload(client, *, content=b"checkpoint", filename="best.pt", metadata=None):
    return client.post(
        "/models/ultralytics/upload",
        data={"metadata": json.dumps(METADATA) if metadata is None else metadata},
        files={"file": (filename, content, "application/octet-stream")},
    )


def test_upload_forwards_binary_metadata_and_service_authentication(api):
    client, backend = api
    # Exceed the spool threshold to exercise an on-disk multipart upload.
    weights = bytes(range(256)) * 5000
    response = upload(client, content=weights)

    assert response.status_code == 201
    assert response.json() == RECORD
    backend.ensure_connection.assert_awaited_once()
    request, parts = backend.requests[0]
    assert str(request.url) == "http://backend:7001/ai-model/ultralytics"
    assert request.headers["Authorization"] == "Bearer service-token"
    assert set(parts) == {"file", "metadata"}
    assert parts["file"].get_filename() == "best.pt"
    assert parts["file"].get_payload(decode=True) == weights
    assert json.loads(parts["metadata"].get_payload(decode=True)) == METADATA


def test_expired_authentication_refreshes_and_resends_complete_checkpoint(api):
    client, backend = api
    backend.responses = [401, 201]

    async def refresh():
        backend.token = "renewed-token"
        return True

    backend.establish_connection.side_effect = refresh
    response = upload(client)

    assert response.status_code == 201
    backend.establish_connection.assert_awaited_once()
    assert [request.headers["Authorization"] for request, _ in backend.requests] == [
        "Bearer service-token", "Bearer renewed-token",
    ]
    assert all(parts["file"].get_payload(decode=True) == b"checkpoint" for _, parts in backend.requests)


@pytest.mark.parametrize("filename,content", [("best.json", b"weights"), ("best.pt", b"")])
def test_invalid_checkpoint_is_rejected_without_backend_call(api, filename, content):
    client, backend = api
    assert upload(client, filename=filename, content=content).status_code == 400
    backend.ensure_connection.assert_not_awaited()
    assert not backend.requests


@pytest.mark.parametrize("metadata", [
    "not-json", "[]", "{}",
    json.dumps({**METADATA, "name": "   "}),
    json.dumps({**METADATA, "name": "x" * 201}),
    json.dumps({**METADATA, "task": "unknown"}),
    json.dumps({**METADATA, "classNames": []}),
    json.dumps({**METADATA, "classNames": ["track", "track"]}),
    json.dumps({**METADATA, "classNames": ["track", "  "]}),
    json.dumps({**METADATA, "classNames": {"0": "track"}}),
    json.dumps({**METADATA, "metadata": []}),
    json.dumps({**METADATA, "metadata": {"map50": float("nan")}}),
])
def test_invalid_metadata_is_rejected_before_publication(api, metadata):
    client, backend = api
    assert upload(client, metadata=metadata).status_code == 422
    backend.ensure_connection.assert_not_awaited()
    assert not backend.requests


@pytest.mark.parametrize("limit", ["MAX_MODEL_BYTES", "MAX_METADATA_BYTES"])
def test_oversized_upload_is_rejected_before_publication(api, monkeypatch, limit):
    client, backend = api
    monkeypatch.setattr(ultralytics, limit, 4)
    assert upload(client).status_code == 413
    backend.ensure_connection.assert_not_awaited()


@pytest.mark.parametrize("missing", ["file", "metadata"])
def test_file_and_metadata_are_required(api, missing):
    client, backend = api
    response = client.post(
        "/models/ultralytics/upload",
        data={} if missing == "metadata" else {"metadata": json.dumps(METADATA)},
        files={} if missing == "file" else {"file": ("best.pt", b"checkpoint")},
    )
    assert response.status_code == 422
    assert not backend.requests


@pytest.mark.parametrize("status", [400, 413, 422, 500])
def test_backend_rejection_is_reported_without_duplicate_upload(api, status):
    client, backend = api
    backend.responses = [status]
    response = upload(client)
    assert response.status_code == (502 if status == 500 else status)
    assert str(status) in response.json()["detail"]
    assert len(backend.requests) == 1


@pytest.mark.parametrize("error,status", [(httpx.ReadTimeout, 504), (httpx.ConnectError, 502)])
def test_transport_failures_are_reported_without_retry(api, error, status):
    client, backend = api
    backend.responses = [error]
    assert upload(client).status_code == status
    assert len(backend.requests) == 1


def test_missing_backend_authentication_returns_service_unavailable(api):
    client, backend = api
    backend.ensure_connection.return_value = False
    assert upload(client).status_code == 503
    assert not backend.requests


@pytest.mark.parametrize("refresh_success,status", [(False, 503), (True, 502)])
def test_failed_authentication_refresh_does_not_loop(api, refresh_success, status):
    client, backend = api
    backend.responses = [401, 401]
    backend.establish_connection.return_value = refresh_success
    assert upload(client).status_code == status
    backend.establish_connection.assert_awaited_once()
    assert len(backend.requests) == (2 if refresh_success else 1)
