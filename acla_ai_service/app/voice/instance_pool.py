"""An elastic object pool for inference engines owned by one application loop."""

from __future__ import annotations

import asyncio
from typing import Awaitable, Callable, Generic, TypeVar

T = TypeVar("T")
R = TypeVar("R")


class InstancePool(Generic[T]):
    """Keep engines alive across requests and lend each to one operation at a time.

    The first operation loads the minimum population. When all engines are
    busy, another is created and retained. Factories and operations must move
    blocking model work to a thread. Pool-owned tasks keep that work leased
    even when the requesting session is cancelled.
    """

    def __init__(self, factory: Callable[[], Awaitable[T]], *, min_size: int = 10):
        if min_size < 1:
            raise ValueError("min_size must be at least 1")
        self._factory = factory
        self._min_size = min_size
        self._instances: list[T] = []
        self._available: list[T] = []
        self._initialize_lock = asyncio.Lock()
        self._jobs: set[asyncio.Task] = set()
        self._closed = False
        self._close_task: asyncio.Task | None = None

    @property
    def stats(self) -> dict[str, int]:
        return {
            "minimum": self._min_size,
            "total": len(self._instances),
            "available": len(self._available),
            "in_use": len(self._instances) - len(self._available),
        }

    async def _initialize(self) -> None:
        async with self._initialize_lock:
            while len(self._instances) < self._min_size:
                instance = await self._factory()
                self._instances.append(instance)
                self._available.append(instance)

    async def run(self, operation: Callable[[T], Awaitable[R]]) -> R:
        """Run with an exclusive engine, returning it after work actually ends."""
        if self._closed:
            raise RuntimeError("Instance pool is closed")
        job = asyncio.create_task(self._run(operation))
        self._jobs.add(job)
        job.add_done_callback(self._job_done)
        return await asyncio.shield(job)

    def _job_done(self, job: asyncio.Task) -> None:
        self._jobs.discard(job)
        # Retrieve failures even if the requesting session has disconnected.
        if not job.cancelled():
            job.exception()

    async def _run(self, operation: Callable[[T], Awaitable[R]]) -> R:
        await self._initialize()
        # No await between checking availability and taking an instance:
        # another coroutine on this application loop cannot take it too.
        if self._available:
            instance = self._available.pop()
        else:
            instance = await self._factory()
            self._instances.append(instance)
        try:
            return await operation(instance)
        finally:
            self._available.append(instance)

    async def aclose(self) -> None:
        """Reject new work, drain native inference, then drop model references."""
        if self._close_task is None:
            self._closed = True
            self._close_task = asyncio.create_task(self._drain())
        await asyncio.shield(self._close_task)

    async def _drain(self) -> None:
        if self._jobs:
            await asyncio.gather(*self._jobs, return_exceptions=True)
        self._available.clear()
        self._instances.clear()
