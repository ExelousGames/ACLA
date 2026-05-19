"""Annotation pipeline HTTP endpoint.

`POST /annotation/run` is the inbound boundary for kicking off an
annotation run from outside the service. It replaces the in-process
`from app.pipelines.annotation import run_annotation` import that the
Streamlit researcher UI uses today — Step 13 of the hexagonal refactor.

The endpoint loads the requested DataFrame from the AI service's own
Zarr telemetry store (the same store the UI uses today), so callers
only need to send a `(cache_key, session_id)` reference rather than
serialising several MB of telemetry per request.

Streaming progress callbacks (the `progress_callback`, `vlm_stream_callback`,
etc. the UI relies on for live VLM token rendering) are NOT yet exposed
here; they require Server-Sent Events. A follow-up endpoint at
`/annotation/run/stream` will add that surface.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, List, Literal, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.pipelines.annotation import (
    AnnotationPipelineConfig,
    AnnotationResult,
    LapAnnotationResult,
    run_annotation,
)
from app.storage.zarr import get_shared_zarr_store

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
    service loads it from its own Zarr store — clients don't have to
    serialise telemetry payloads.
    """

    flow: Flow
    cache_key: str = Field(..., description="Zarr cache key (group path) for the dataset")
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
    """Pull the DataFrame for ``(cache_key, session_id)`` from the Zarr store."""
    store = get_shared_zarr_store()
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


__all__ = ["router"]
