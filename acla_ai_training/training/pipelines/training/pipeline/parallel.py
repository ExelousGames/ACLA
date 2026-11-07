"""Bounded, ordered process execution for telemetry preparation chunks."""

import asyncio
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from itertools import islice
from multiprocessing import get_context


async def map_chunks_in_processes(
    worker, chunks, *, max_workers: int, initializer=None, initargs=(),
):
    """Keep at most one chunk per worker in flight and preserve input order."""
    if max_workers < 1:
        raise ValueError("Workers must be at least 1")

    if max_workers == 1:
        if initializer is not None:
            initializer(*initargs)
        for chunk in chunks:
            yield worker(chunk)
        return

    # Spawn avoids inheriting backend connections or the parent's running loop.
    with ProcessPoolExecutor(
        max_workers=max_workers,
        mp_context=get_context("spawn"),
        initializer=initializer,
        initargs=initargs,
    ) as executor:
        chunks = iter(chunks)
        pending = deque(
            executor.submit(worker, chunk)
            for chunk in islice(chunks, max_workers)
        )
        try:
            while pending:
                yield await asyncio.wrap_future(pending.popleft())
                for chunk in islice(chunks, 1):
                    pending.append(executor.submit(worker, chunk))
        finally:
            for future in pending:
                future.cancel()
