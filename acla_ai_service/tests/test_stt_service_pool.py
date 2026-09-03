from __future__ import annotations

import asyncio
import importlib
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.voice import stt_service_pool


class _FakeWorker:
    def __init__(self, worker_id: int) -> None:
        self.worker_id = worker_id
        self.closed = False

    async def close(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_pool_preloads_bounded_multicore_workers_and_reuses_them(
    monkeypatch,
):
    cpu_threads = []
    workers = []

    def fake_build_worker(thread_count):
        worker = _FakeWorker(len(workers))
        workers.append(worker)
        cpu_threads.append(thread_count)
        return worker

    monkeypatch.setattr(stt_service_pool, "_build_worker", fake_build_worker)
    pool = stt_service_pool.VoiceSTTServicePool(cpu_count=8)

    await pool.start()

    assert pool.size == 2
    assert cpu_threads == [4, 4]

    first_lease = pool.lease()
    second_lease = pool.lease()
    first = await first_lease.__aenter__()
    second = await second_lease.__aenter__()

    async def take_next_worker():
        async with pool.lease() as worker:
            return worker

    waiting = asyncio.create_task(take_next_worker())
    await asyncio.sleep(0)
    assert not waiting.done()

    await first_lease.__aexit__(None, None, None)
    assert await waiting is first
    await second_lease.__aexit__(None, None, None)

    await pool.close()

    assert all(worker.closed for worker in workers)


@pytest.mark.asyncio
async def test_session_stt_processor_leases_model_only_while_transcribing(
    monkeypatch,
):
    stt_module = ModuleType("pipecat.services.whisper.stt")
    pool = stt_service_pool.VoiceSTTServicePool(cpu_count=1)
    shared_model = object()
    worker = stt_service_pool.VoiceSTTWorker(whisper_model=shared_model)
    available = asyncio.Queue(maxsize=1)
    available.put_nowait(worker)
    pool._available = available
    pool._workers = [worker]
    observed_models = []

    class FakeWhisperSTTService:
        def __init__(self, *, model):
            self.requested_model = model
            self._load()

        def _load(self):
            raise AssertionError("a session processor must not load a model")

        async def run_stt(self, audio):
            observed_models.append(self._model)
            assert pool._available.empty()
            if audio == b"fail":
                raise RuntimeError("transcription failed")
            yield audio

    stt_module.WhisperSTTService = FakeWhisperSTTService
    monkeypatch.setitem(sys.modules, "pipecat", ModuleType("pipecat"))
    monkeypatch.setitem(sys.modules, "pipecat.services", ModuleType("pipecat.services"))
    monkeypatch.setitem(
        sys.modules,
        "pipecat.services.whisper",
        ModuleType("pipecat.services.whisper"),
    )
    monkeypatch.setitem(sys.modules, "pipecat.services.whisper.stt", stt_module)

    processor = pool.create_session_stt()
    assert pool._available.qsize() == 1
    frames = [frame async for frame in processor.run_stt(b"audio")]

    assert frames == [b"audio"]
    assert observed_models == [shared_model]
    assert processor._model is None
    assert pool._available.qsize() == 1
    assert processor.requested_model == "large-v3-turbo"

    with pytest.raises(RuntimeError, match="transcription failed"):
        [frame async for frame in processor.run_stt(b"fail")]

    assert processor._model is None
    assert pool._available.qsize() == 1


@pytest.mark.asyncio
async def test_application_lifespan_starts_and_closes_stt_pool(monkeypatch):
    startup = importlib.import_module("app.startup.app")
    pool = SimpleNamespace(
        size=2,
        start=AsyncMock(),
        close=AsyncMock(),
    )
    monkeypatch.setattr(startup, "get_voice_stt_service_pool", lambda: pool)
    monkeypatch.setattr(
        startup,
        "resolve_chat_llm_config",
        lambda: SimpleNamespace(provider="openai", base_url=None, model="test"),
    )
    monkeypatch.setattr(
        startup,
        "check_llama_server",
        AsyncMock(return_value=SimpleNamespace(
            reachable=False,
            base_url="http://llama",
            error="offline",
        )),
    )
    monkeypatch.setattr(
        startup.backend_service,
        "establish_connection",
        AsyncMock(return_value=True),
    )
    monkeypatch.setattr(startup, "hydrate_chatbot_models", AsyncMock(return_value={}))

    async with startup.lifespan(SimpleNamespace()):
        pool.start.assert_awaited_once_with()
        pool.close.assert_not_awaited()

    pool.close.assert_awaited_once_with()
