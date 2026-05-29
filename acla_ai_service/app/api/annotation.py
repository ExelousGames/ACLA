"""Annotation pipeline HTTP endpoints.

Two routes:

  - ``POST /annotation/run``         (blocking) — added in Step 13
  - ``POST /annotation/run/stream``  (SSE)      — added in PR #5

Both replace the in-process ``from app.local_annotation_agent.workflow import
run_annotation`` import that the Streamlit researcher UI uses today.
The streaming variant surfaces the agent's progress / VLM-token /
step-event callbacks live so callers can render incremental output.

Telemetry is referenced by ``(cache_key, session_id)`` — the AI service
loads from its own Lance-backed telemetry store so requests stay small.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from dataclasses import asdict, is_dataclass
from typing import Any, AsyncIterator, Dict, List, Literal, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.local_annotation_agent.workflow import (
    AnnotationPipelineConfig,
    AnnotationResult,
    LapAnnotationResult,
    run_annotation,
)
from app.storage import get_shared_telemetry_store

LOGGER = logging.getLogger(__name__)

router = APIRouter(prefix="/annotation", tags=["annotation"])


Flow = Literal["detailed", "lap"]
Backend = Literal["local", "claude"]


class _ConfigBody(BaseModel):
    """Config knobs forwarded to AnnotationPipelineConfig.

    Mirrors the dataclass fields without their callback machinery — the
    HTTP entrypoint omits streaming callbacks (deferred to SSE).
    """

    backend: Backend = "local"
    max_new_tokens: int = 1500
    temperature: float = 0.7

    # local VLM (llama-server)
    gguf_path: Optional[str] = None
    mmproj_path: Optional[str] = None
    context_size: int = 32768
    n_gpu_layers: int = -1
    hf_repo: str = "Qwen/Qwen2.5-VL-72B-Instruct"
    quantization_type: str = "Q4_K_M"

    # claude
    claude_model: str = "claude-sonnet-4-6"
    claude_use_thinking: bool = False


class _AnnotationRunRequest(BaseModel):
    """Body for `POST /annotation/run`.

    The dataframe is referenced by `(cache_key, session_id)` so the AI
    service loads it from its own telemetry store — clients don't have to
    serialise telemetry payloads.
    """

    flow: Flow
    cache_key: str = Field(..., description="Telemetry store cache key for the dataset")
    session_id: str = Field(..., description="Chunk / session ID within the cache key")
    session_label: str = Field("", description="Forwarded to AgentResponse.session_id for audit")
    config: Optional[_ConfigBody] = None

    # detailed-flow inputs
    start_index: Optional[int] = None
    end_index: Optional[int] = None
    parent_main_labels: Optional[List[str]] = None
    existing_children: Optional[List[Dict[str, Any]]] = None

    # lap-flow inputs
    lap_start: Optional[int] = None
    lap_end: Optional[int] = None
    section_id: Optional[str] = None
    section_start: Optional[int] = None
    section_end: Optional[int] = None
    circuit_id: Optional[str] = None
    existing_section_annotations: Optional[List[Dict[str, Any]]] = None


def _result_to_dict(result: Any) -> Dict[str, Any]:
    """AnnotationResult / LapAnnotationResult are dataclasses; convert to dict
    without forcing callers to learn each one's field layout."""
    if is_dataclass(result):
        return asdict(result)
    if isinstance(result, dict):
        return result
    raise TypeError(f"Unsupported annotation result type: {type(result).__name__}")


def _load_dataframe(cache_key: str, session_id: str):
    """Pull the DataFrame for ``(cache_key, session_id)`` from the telemetry store."""
    store = get_shared_telemetry_store()
    chunk = store.get_chunk(cache_key, session_id)
    if not chunk:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No telemetry found for cache_key={cache_key!r} "
                f"session_id={session_id!r}"
            ),
        )

    raw = chunk
    if isinstance(chunk, dict) and "data" in chunk:
        raw = chunk["data"]
    if not isinstance(raw, list):
        raise HTTPException(
            status_code=422,
            detail=f"Telemetry chunk is not a list of records (got {type(raw).__name__})",
        )

    # Deferred to avoid importing pandas at module load.
    import pandas as pd
    return pd.DataFrame(raw)


@router.post("/run")
async def annotation_run(req: _AnnotationRunRequest) -> Dict[str, Any]:
    """Run one annotation pass.

    Replaces the in-process `run_annotation(...)` call the Streamlit UI
    makes today. Streaming progress is NOT surfaced here — clients that
    need per-step VLM tokens should wait for `/annotation/run/stream`.
    """
    df = _load_dataframe(req.cache_key, req.session_id)

    config_body = req.config or _ConfigBody()
    config = AnnotationPipelineConfig(
        backend=config_body.backend,
        max_new_tokens=config_body.max_new_tokens,
        temperature=config_body.temperature,
        gguf_path=config_body.gguf_path,
        mmproj_path=config_body.mmproj_path,
        context_size=config_body.context_size,
        n_gpu_layers=config_body.n_gpu_layers,
        hf_repo=config_body.hf_repo,
        quantization_type=config_body.quantization_type,
        claude_model=config_body.claude_model,
        claude_use_thinking=config_body.claude_use_thinking,
    )

    try:
        result = run_annotation(
            flow=req.flow,
            df=df,
            config=config,
            session_id=req.session_label,
            # detailed-flow inputs (run_annotation validates which set is required)
            start_index=req.start_index,
            end_index=req.end_index,
            parent_main_labels=req.parent_main_labels,
            existing_children=req.existing_children,
            # lap-flow inputs
            lap_start=req.lap_start,
            lap_end=req.lap_end,
            section_id=req.section_id,
            section_start=req.section_start,
            section_end=req.section_end,
            circuit_id=req.circuit_id,
            existing_section_annotations=req.existing_section_annotations,
        )
    except ValueError as exc:
        # run_annotation raises ValueError for missing required kwargs.
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        LOGGER.exception("Annotation run failed (flow=%s)", req.flow)
        raise HTTPException(
            status_code=500,
            detail=f"Annotation failed: {type(exc).__name__}: {exc}",
        ) from exc

    return {
        "flow": req.flow,
        "backend": config.backend,
        "result": _result_to_dict(result),
    }


# ─── Streaming variant ───────────────────────────────────────────────────────
#
# `run_annotation` is synchronous (the agent loop blocks). Its progress
# callbacks (`progress_callback`, `vlm_stream_callback`, `vlm_prompt_callback`,
# `vlm_reasoning_callback`, `step_event_callback`) fire from inside that
# blocking call.
#
# To turn those into SSE events:
#   1. Spawn a worker thread that calls `run_annotation` with the callbacks.
#   2. Each callback pushes a formatted SSE string into an asyncio.Queue,
#      crossing the thread→loop boundary via `loop.call_soon_threadsafe`.
#   3. The route's async generator drains the queue and yields each event
#      back to FastAPI's StreamingResponse.
#   4. A None sentinel pushed in the worker's `finally` closes the stream.
#
# Callers wanting cancellation should drop the HTTP connection; the worker
# thread can't be force-stopped, but its events become orphans and the
# server-side garbage collector eventually frees them. For short-lived
# annotation runs (seconds to a few minutes) that's an acceptable trade.


def _sse(event_type: str, **payload: Any) -> str:
    """Format one SSE frame. Matches the shape used by /naturallanguagequery/stream."""
    body = {"type": event_type, **payload}
    return f"data: {json.dumps(body, ensure_ascii=False, default=str)}\n\n"


@router.post(
    "/run/stream",
    responses={
        200: {
            "content": {"text/event-stream": {}},
            "description": (
                "Server-Sent Events stream. Event types: progress, vlm_prompt, "
                "vlm_stream, vlm_reasoning, step_event, done, error."
            ),
        }
    },
)
async def annotation_run_stream(req: _AnnotationRunRequest) -> StreamingResponse:
    """Streaming variant of `/annotation/run`.

    Emits the same final result as the blocking endpoint, plus live events
    as the agent executes. Useful for the Streamlit UI's live VLM-token
    display (was driven by in-process callbacks pre-refactor).

    Event payloads:
      progress     {"node": str, "detail": str}
      vlm_prompt   {"prompt": str, "stage": dict}
      vlm_stream   {"chunk": str}            ← user-visible VLM tokens
      vlm_reasoning{"chunk": str}            ← thinking blocks (claude only)
      step_event   {"summary": str, "stage": dict}
      done         {"flow": "detailed"|"lap", "backend": str, "result": dict}
      error        {"message": str, "error_type": str}
    """
    df = _load_dataframe(req.cache_key, req.session_id)

    config_body = req.config or _ConfigBody()
    config = AnnotationPipelineConfig(
        backend=config_body.backend,
        max_new_tokens=config_body.max_new_tokens,
        temperature=config_body.temperature,
        gguf_path=config_body.gguf_path,
        mmproj_path=config_body.mmproj_path,
        context_size=config_body.context_size,
        n_gpu_layers=config_body.n_gpu_layers,
        hf_repo=config_body.hf_repo,
        quantization_type=config_body.quantization_type,
        claude_model=config_body.claude_model,
        claude_use_thinking=config_body.claude_use_thinking,
    )

    queue: asyncio.Queue[Optional[str]] = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def push(frame: Optional[str]) -> None:
        """Thread-safe enqueue. The None sentinel closes the stream."""
        loop.call_soon_threadsafe(queue.put_nowait, frame)

    def on_progress(node: str, detail: str) -> None:
        push(_sse("progress", node=node, detail=detail))

    def on_vlm_prompt(prompt: str, stage: Dict[str, Any]) -> None:
        push(_sse("vlm_prompt", prompt=prompt, stage=stage))

    def on_vlm_stream(chunk: str) -> None:
        push(_sse("vlm_stream", chunk=chunk))

    def on_vlm_reasoning(chunk: str) -> None:
        push(_sse("vlm_reasoning", chunk=chunk))

    def on_step_event(summary: str, stage: Dict[str, Any]) -> None:
        push(_sse("step_event", summary=summary, stage=stage))

    def runner() -> None:
        try:
            result = run_annotation(
                flow=req.flow,
                df=df,
                config=config,
                session_id=req.session_label,
                progress_callback=on_progress,
                vlm_prompt_callback=on_vlm_prompt,
                vlm_stream_callback=on_vlm_stream,
                vlm_reasoning_callback=on_vlm_reasoning,
                step_event_callback=on_step_event,
                start_index=req.start_index,
                end_index=req.end_index,
                parent_main_labels=req.parent_main_labels,
                existing_children=req.existing_children,
                lap_start=req.lap_start,
                lap_end=req.lap_end,
                section_id=req.section_id,
                section_start=req.section_start,
                section_end=req.section_end,
                circuit_id=req.circuit_id,
                existing_section_annotations=req.existing_section_annotations,
            )
            push(_sse(
                "done",
                flow=req.flow,
                backend=config.backend,
                result=_result_to_dict(result),
            ))
        except ValueError as exc:
            # run_annotation's own arg-validation errors
            push(_sse("error", message=str(exc), error_type="ValueError"))
        except Exception as exc:
            LOGGER.exception("Streaming annotation run failed (flow=%s)", req.flow)
            push(_sse(
                "error",
                message=f"{type(exc).__name__}: {exc}",
                error_type=type(exc).__name__,
            ))
        finally:
            push(None)

    threading.Thread(target=runner, daemon=True, name=f"annotation-{req.flow}").start()

    async def event_source() -> AsyncIterator[str]:
        while True:
            frame = await queue.get()
            if frame is None:
                return
            yield frame

    return StreamingResponse(
        event_source(),
        media_type="text/event-stream",
        headers={
            # Disable any reverse-proxy buffering so events flush immediately.
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


__all__ = ["router"]
