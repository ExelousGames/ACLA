"""Startup-owned pool of reusable Whisper inference workers."""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, AsyncIterator, Optional


LOGGER = logging.getLogger(__name__)

WHISPER_MODEL = "large-v3-turbo"
_MAX_STT_WORKERS = 2


@dataclass
class VoiceSTTWorker:
    """One preloaded Whisper model that handles one inference at a time."""

    whisper_model: Any

    async def close(self) -> None:
        self.whisper_model = None


class VoiceSTTServicePool:
    """Bounded pool of startup-loaded STT workers.

    Each WebSocket owns its lightweight Pipecat processor and mutable audio
    buffers. A worker is leased only while that processor runs Whisper for one
    completed speech segment, then immediately returned to the pool.
    """

    def __init__(self, cpu_count: Optional[int] = None) -> None:
        visible_cpus = cpu_count if cpu_count is not None else os.cpu_count()
        self._cpu_count = max(1, visible_cpus or 1)
        self._size = min(_MAX_STT_WORKERS, self._cpu_count)
        self._cpu_threads_per_worker = max(1, self._cpu_count // self._size)
        self._available: Optional[asyncio.Queue[VoiceSTTWorker]] = None
        self._workers: list[VoiceSTTWorker] = []

    @property
    def size(self) -> int:
        return self._size

    @property
    def is_ready(self) -> bool:
        return self._available is not None

    async def start(self) -> None:
        """Construct every heavyweight worker before the app accepts traffic."""
        if self._available is not None:
            return

        LOGGER.info(
            "Loading %d reusable Whisper STT worker(s), %d CPU thread(s) each",
            self._size,
            self._cpu_threads_per_worker,
        )
        workers = []
        for _ in range(self._size):
            worker = await asyncio.to_thread(
                _build_worker,
                self._cpu_threads_per_worker,
            )
            workers.append(worker)

        available: asyncio.Queue[VoiceSTTWorker] = asyncio.Queue(
            maxsize=self._size,
        )
        for worker in workers:
            available.put_nowait(worker)

        self._workers = workers
        self._available = available
        LOGGER.info("Whisper STT worker pool ready")

    @asynccontextmanager
    async def lease(self) -> AsyncIterator[VoiceSTTWorker]:
        """Lease one Whisper model for a single transcription."""
        available = self._available
        if available is None:
            raise RuntimeError("Whisper STT worker pool has not been started")

        worker = await available.get()
        try:
            yield worker
        finally:
            if self._available is available:
                available.put_nowait(worker)

    def create_session_stt(self) -> Any:
        """Create a session-local STT processor that leases per inference."""
        from pipecat.services.whisper.stt import WhisperSTTService

        worker_pool = self

        class PooledWhisperSTTService(WhisperSTTService):
            def _load(self) -> None:
                # Models are loaded once by VoiceSTTServicePool.start().
                self._model = None

            async def run_stt(self, audio: bytes):
                async with worker_pool.lease() as worker:
                    self._model = worker.whisper_model
                    try:
                        async for frame in super().run_stt(audio):
                            yield frame
                    finally:
                        self._model = None

        return PooledWhisperSTTService(model=WHISPER_MODEL)

    async def close(self) -> None:
        """Release references to pooled models during application shutdown."""
        workers = self._workers
        self._available = None
        self._workers = []
        for worker in workers:
            await worker.close()


def _build_worker(cpu_threads: int) -> VoiceSTTWorker:
    """Build one worker off the event loop; model construction is blocking."""
    from faster_whisper import WhisperModel

    return VoiceSTTWorker(
        whisper_model=WhisperModel(
            WHISPER_MODEL,
            device="auto",
            compute_type="default",
            cpu_threads=cpu_threads,
            num_workers=1,
        ),
    )


_pool = VoiceSTTServicePool()


def get_voice_stt_service_pool() -> VoiceSTTServicePool:
    return _pool
