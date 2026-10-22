import asyncio
import base64
import json

import httpx
import pytest

from training.integrations import backend


def test_backend_settings_read_existing_env_names_and_dotenv(tmp_path, monkeypatch):
    for name in ("BACKEND_SERVER_IP", "BACKEND_PROXY_PORT", "AI_SERVICE_USERNAME", "AI_SERVICE_PASSWORD"):
        monkeypatch.delenv(name, raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text(
        "BACKEND_SERVER_IP=http://backend\nBACKEND_PROXY_PORT=7001\n"
        "AI_SERVICE_USERNAME=owner@example.com\nAI_SERVICE_PASSWORD=password\n"
        "UNRELATED_SETTING=ignored\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("BACKEND_PROXY_PORT", "7002")
    settings = backend.BackendSettings(_env_file=env_file)
    assert settings.backend_server_ip == "http://backend"
    assert settings.backend_proxy_port == "7002"
    assert settings.ai_service_username == "owner@example.com"
    assert settings.ai_service_password == "password"


@pytest.mark.parametrize("failed_chunk", [False, True])
def test_authenticated_model_download_assembles_backend_chunks(tmp_path, monkeypatch, failed_chunk):
    payload = {"top_lap_store": {"acc|track|car|grip2": "encoded-reference"}}
    encoded = json.dumps(payload).encode("utf-8")
    chunks = [encoded[:17], encoded[17:]]
    requests = []

    def handle(request):
        path = request.url.path
        requests.append(path)
        if path == "/userinfo/auth/login":
            assert json.loads(request.content) == {"email": "owner@example.com", "password": "password"}
            return httpx.Response(200, json={"access_token": "owner-token"})
        assert request.headers["Authorization"] == "Bearer owner-token"
        if path == "/ai-model/active/top_lap_reference/prepare-chunked":
            return httpx.Response(200, json={
                "success": True,
                "data": {"modelType": "top_lap_reference", "isActive": True, "metadata": {"laps": 1}},
                "chunking": {"sessionId": "model-download", "totalChunks": 2},
            })
        assert path.startswith("/ai-model/active/chunked-data/model-download/")
        index = int(path.rsplit("/", 1)[1])
        if failed_chunk and index == 1:
            return httpx.Response(200, json={"success": False, "message": "Session expired"})
        return httpx.Response(200, json={
            "success": True, "data": base64.b64encode(chunks[index]).decode("ascii"),
        })

    client_type = httpx.AsyncClient
    monkeypatch.setattr(backend.httpx, "AsyncClient", lambda **kwargs: client_type(
        transport=httpx.MockTransport(handle), **kwargs,
    ))
    monkeypatch.setattr(backend, "settings", backend.BackendSettings(
        _env_file=None, backend_server_ip="http://backend", backend_proxy_port="7001",
        ai_service_username="owner@example.com", ai_service_password="password",
    ))
    monkeypatch.setattr(backend.tempfile, "tempdir", str(tmp_path))
    service = backend.BackendService()
    if failed_chunk:
        with pytest.raises(Exception, match="Session expired"):
            asyncio.run(service.getCompleteActiveModelData("top_lap_reference"))
    else:
        result = asyncio.run(service.getCompleteActiveModelData("top_lap_reference"))
        assert result.modelData == payload
        assert result.modelType == "top_lap_reference"
        assert result.isActive is True
        assert result.metadata == {"laps": 1}
    assert requests == [
        "/userinfo/auth/login",
        "/ai-model/active/top_lap_reference/prepare-chunked",
        "/ai-model/active/chunked-data/model-download/0",
        "/ai-model/active/chunked-data/model-download/1",
    ]
    assert not list(tmp_path.glob("model_chunks_*"))
