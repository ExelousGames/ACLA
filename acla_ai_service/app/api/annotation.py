"""Annotation pipeline HTTP endpoints.

Two routes:

  - ``POST /annotation/run``         (blocking) — added in Step 13
  - ``POST /annotation/run/stream``  (SSE)      — added in PR #5

Both replace the in-process ``from app.local_annotation_agent.workflow import
run_annotation`` import that the Streamlit researcher UI uses today.
The streaming variant surfaces the agent's progress / VLM-token /
step-event callbacks live so callers can render incremental output.

Telemetry is supplied directly by the request body. Annotation tools must
only inspect the incoming segment/lap records, not reload a broader session
from shared storage.
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

LOGGER = logging.getLogger(__name__)

router = APIRouter(prefix="/annotation", tags=["annotation"])


Flow = Literal["detailed", "lap"]
class _ConfigBody(BaseModel):
    """Provider-neutral config forwarded to AnnotationPipelineConfig."""

    provider_id: str = "local_vlm"
    model: str = ""
    max_new_tokens: int = 1500
    temperature: float = 0.7
    max_iterations: int = 3
    provider_options: Dict[str, Any] = Field(default_factory=dict)


class _AnnotationRunRequest(BaseModel):
    """Body for `POST /annotation/run`.

    The caller must provide exactly the telemetry records the agent may
    inspect. There is intentionally no cache/session fallback here.
    """

    class Config:
        extra = "forbid"

    flow: Flow
    telemetry_data: List[Dict[str, Any]] = Field(
        ...,
        min_items=1,
        description="Telemetry records for the segment/lap this annotation run may inspect.",
    )
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
    revision_start: Optional[int] = None
    revision_end: Optional[int] = None
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


def _dataframe_from_records(records: List[Dict[str, Any]], origin_start: int = 0):
    """Build a DataFrame from only the request's allowed telemetry records.

    The annotation tools address rows by absolute iloc. Keep those absolute
    coordinates on the dataframe index without manufacturing empty prefix rows.
    """
    # Deferred to avoid importing pandas at module load.
    import pandas as pd
    df = pd.DataFrame(records)
    origin = max(0, int(origin_start or 0))
    if origin == 0:
        return df
    df.index = range(origin, origin + len(df))
    return df


def _telemetry_origin(req: _AnnotationRunRequest) -> int:
    if req.flow == "detailed" and req.start_index is not None:
        return int(req.start_index)
    if req.flow == "lap":
        for value in (req.revision_start, req.section_start, req.lap_start):
            if value is not None:
                return int(value)
    return 0


@router.post("/run")
async def annotation_run(req: _AnnotationRunRequest) -> Dict[str, Any]:
    """Run one annotation pass.

    Replaces the in-process `run_annotation(...)` call the Streamlit UI
    makes today. Streaming progress is NOT surfaced here — clients that
    need per-step VLM tokens should wait for `/annotation/run/stream`.
    """
    df = _dataframe_from_records(req.telemetry_data, _telemetry_origin(req))

    config_body = req.config or _ConfigBody()
    config = AnnotationPipelineConfig(
        provider_id=config_body.provider_id,
        model=config_body.model,
        max_new_tokens=config_body.max_new_tokens,
        temperature=config_body.temperature,
        max_iterations=config_body.max_iterations,
        provider_options=dict(config_body.provider_options),
    )

    try:
        result = run_annotation(
            flow=req.flow,
            df=df,
            config=config,
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
            revision_start=req.revision_start,
            revision_end=req.revision_end,
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
        "provider_id": config.provider_id,
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
      done         {"flow": "detailed"|"lap", "provider_id": str, "result": dict}
      error        {"message": str, "error_type": str}
    """
    df = _dataframe_from_records(req.telemetry_data, _telemetry_origin(req))

    config_body = req.config or _ConfigBody()
    config = AnnotationPipelineConfig(
        provider_id=config_body.provider_id,
        model=config_body.model,
        max_new_tokens=config_body.max_new_tokens,
        temperature=config_body.temperature,
        max_iterations=config_body.max_iterations,
        provider_options=dict(config_body.provider_options),
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
                revision_start=req.revision_start,
                revision_end=req.revision_end,
                circuit_id=req.circuit_id,
                existing_section_annotations=req.existing_section_annotations,
            )
            push(_sse(
                "done",
                flow=req.flow,
                provider_id=config.provider_id,
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
