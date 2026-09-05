from __future__ import annotations

import asyncio
import threading
from unittest.mock import AsyncMock

import pytest

from app.voice.instance_pool import InstancePool


@pytest.mark.asyncio
async def test_minimum_is_loaded_once_and_instances_are_reused():
    factory = AsyncMock(side_effect=object)
    pool = InstancePool(factory)
    assert pool.stats == {"minimum": 10, "total": 0, "available": 0, "in_use": 0}

    first = await pool.run(AsyncMock(side_effect=lambda instance: instance))
    second = await pool.run(AsyncMock(side_effect=lambda instance: instance))

    assert first is second
    assert factory.await_count == 10
    assert pool.stats == {"minimum": 10, "total": 10, "available": 10, "in_use": 0}
    await pool.aclose()
    assert pool.stats["total"] == 0


@pytest.mark.asyncio
async def test_concurrent_sessions_get_exclusive_engines_and_grow_past_ten():
    factory = AsyncMock(side_effect=object)
    pool = InstancePool(factory)
    engines = []
    all_started = asyncio.Event()
    finish = asyncio.Event()

    async def operation(engine):
        engines.append(engine)
        if len(engines) == 11:
            all_started.set()
        await finish.wait()
        return engine

    sessions = [asyncio.create_task(pool.run(operation)) for _ in range(11)]
    try:
        await asyncio.wait_for(all_started.wait(), timeout=2)
        assert len({id(engine) for engine in engines}) == 11
        assert factory.await_count == 11
        assert pool.stats["in_use"] == 11
    finally:
        finish.set()
        await asyncio.gather(*sessions)
        await pool.aclose()


@pytest.mark.asyncio
async def test_operation_failure_returns_engine_for_next_request():
    pool = InstancePool(AsyncMock(side_effect=object))
    failed_engine = None

    async def fail(engine):
        nonlocal failed_engine
        failed_engine = engine
        raise ValueError("inference failed")

    with pytest.raises(ValueError, match="inference failed"):
        await pool.run(fail)

    reused = await pool.run(AsyncMock(side_effect=lambda engine: engine))
    assert reused is failed_engine
    assert pool.stats["available"] == 10
    await pool.aclose()


@pytest.mark.asyncio
async def test_partial_initialization_can_be_retried_without_losing_engines():
    attempts = 0

    async def factory():
        nonlocal attempts
        attempts += 1
        if attempts == 3:
            raise RuntimeError("model load failed")
        return object()

    pool = InstancePool(factory)
    operation = AsyncMock()
    with pytest.raises(RuntimeError, match="model load failed"):
        await pool.run(operation)
    operation.assert_not_awaited()
    assert pool.stats["total"] == 2

    await pool.run(operation)
    assert attempts == 11
    assert pool.stats["available"] == 10
    await pool.aclose()


@pytest.mark.asyncio
async def test_growth_failure_does_not_steal_busy_instance():
    factory = AsyncMock(side_effect=[object(), RuntimeError("out of memory"), object()])
    pool = InstancePool(factory, min_size=1)
    started = asyncio.Event()
    finish = asyncio.Event()

    async def hold(engine):
        started.set()
        await finish.wait()

    session = asyncio.create_task(pool.run(hold))
    try:
        await asyncio.wait_for(started.wait(), timeout=2)
        with pytest.raises(RuntimeError, match="out of memory"):
            await pool.run(AsyncMock())
        assert pool.stats["in_use"] == 1
        await pool.run(AsyncMock())
        assert pool.stats["total"] == 2
        assert pool.stats["available"] == 1
    finally:
        finish.set()
        await session
        await pool.aclose()


@pytest.mark.asyncio
async def test_cancelled_session_keeps_native_thread_leased_until_it_finishes():
    pool = InstancePool(AsyncMock(side_effect=object), min_size=1)
    started = asyncio.Event()
    finish = threading.Event()
    loop = asyncio.get_running_loop()
    busy_engine = None

    def native_inference(engine):
        nonlocal busy_engine
        busy_engine = engine
        loop.call_soon_threadsafe(started.set)
        assert finish.wait(timeout=5)

    session = asyncio.create_task(pool.run(
        lambda engine: asyncio.to_thread(native_inference, engine),
    ))
    try:
        await asyncio.wait_for(started.wait(), timeout=2)
        session.cancel()
        with pytest.raises(asyncio.CancelledError):
            await session

        assert pool.stats["in_use"] == 1
        other_engine = await pool.run(AsyncMock(side_effect=lambda engine: engine))
        assert other_engine is not busy_engine
        assert pool.stats["total"] == 2

        closing = asyncio.create_task(pool.aclose())
        await asyncio.sleep(0)
        assert not closing.done()
        with pytest.raises(RuntimeError, match="closed"):
            await pool.run(AsyncMock())
    finally:
        finish.set()
        await pool.aclose()
    await closing
    assert pool.stats["total"] == 0


@pytest.mark.asyncio
async def test_cancelled_first_request_does_not_cancel_model_loading():
    started = asyncio.Event()
    finish = threading.Event()
    loop = asyncio.get_running_loop()
    created = []

    def load_model():
        loop.call_soon_threadsafe(started.set)
        assert finish.wait(timeout=5)
        model = object()
        created.append(model)
        return model

    pool = InstancePool(lambda: asyncio.to_thread(load_model), min_size=1)
    session = asyncio.create_task(pool.run(AsyncMock()))
    try:
        await asyncio.wait_for(started.wait(), timeout=2)
        session.cancel()
        with pytest.raises(asyncio.CancelledError):
            await session
        assert not created
    finally:
        finish.set()
        await pool.aclose()
    assert len(created) == 1
    assert pool.stats["total"] == 0
