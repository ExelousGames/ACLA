"""
Public annotation pipeline entry — the only function the UI calls.

    from app.local_annotation_agent.workflow import (
        run_annotation,
        AnnotationPipelineConfig,
        AnnotationResult,
        LapAnnotationResult,
    )

    config = AnnotationPipelineConfig(backend="local", ...)

    result = run_annotation(
        flow="detailed",                # "detailed" or "lap"
        df=df, range_=(start, end),
        config=config,
        callbacks=callbacks,
        # flow-specific kwargs:
        parent_main_labels=[...], existing_children=[...],   # detailed
        # OR
        section_id=..., section_start=..., section_end=...,  # lap
        circuit_id=..., existing_section_annotations=[...],
    )

Internally:
    1. Picks the flow module (annotation.flows.detailed / .lap).
    2. ``flow.build_request(...)`` translates domain intent into AgentRequest.
    3. ``run_agent(request)`` dispatches to the local or claude runner.
    4. ``flow.parse(response, ...)`` decodes raw text into a typed result.

Each layer has one job. The agent box never sees racing types; the flows
never see runners; the UI never sees the box.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

# Side-effect imports — register annotation-domain agents, tools, and
# structured-attachment formatters with the agent box. Order matters:
# these must run before any caller invokes run_annotation(). Was previously
# triggered by importing the `annotation` package's __init__; now hoisted
# here since pipelines/annotation/ doubles as the public entry.
from app.local_annotation_agent.workflow import formatters  # noqa: F401
from app.local_annotation_agent.workflow import agents      # noqa: F401
from app.local_annotation_agent.workflow import tools       # noqa: F401

from app.local_annotation_agent import AgentRequest, AgentResponse, run_agent
from app.shared.contracts import (
    AgentCallbacks,
    BackendConfig,
    NoopCallbacks,
)
from app.local_annotation_agent.workflow.flows import detailed as detailed_flow
from app.local_annotation_agent.workflow.flows import lap as lap_flow
from app.local_annotation_agent.workflow.followup import run_claude_followup
from app.local_annotation_agent.workflow.results import (
    AnnotationResult,
    LapAnnotationResult,
)

LOGGER = logging.getLogger(__name__)


Flow = Literal["detailed", "lap"]
Backend = Literal["local", "claude"]


# ---------------------------------------------------------------------------
# Backwards-compatible config — wraps BackendConfig + retains old field names
# ---------------------------------------------------------------------------


@dataclass
class AnnotationPipelineConfig:
    """Config the UI passes. Mirrors the old shape so callers don't break.

    Translates to ``BackendConfig`` internally before reaching the agent.
    Backend-irrelevant generation knobs sit at the top level; backend-
    specific knobs are grouped by backend.
    """

    backend: Backend = "local"

    max_iterations: int = 3  # kept for API parity; not honoured by the new runners
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

    def to_backend_config(self) -> BackendConfig:
        return BackendConfig(
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            gguf_path=self.gguf_path,
            mmproj_path=self.mmproj_path,
            hf_repo=self.hf_repo,
            quantization_type=self.quantization_type,
            context_size=self.context_size,
            n_gpu_layers=self.n_gpu_layers,
            claude_model=self.claude_model,
            claude_use_thinking=self.claude_use_thinking,
        )


# ---------------------------------------------------------------------------
# Callback adapter — UI passes loose callables; AgentCallbacks expects a struct
# ---------------------------------------------------------------------------


@dataclass
class _CallbackBag:
    """In-process AgentCallbacks built from the UI's loose kwargs."""
    progress: Optional[Callable[[str, str], None]] = None
    vlm_prompt: Optional[Callable[[str, Dict[str, Any]], None]] = None
    vlm_stream: Optional[Callable[[str], None]] = None
    vlm_reasoning: Optional[Callable[[str], None]] = None
    step_event: Optional[Callable[[str, Dict[str, Any]], None]] = None


def _bag_from_kwargs(
    progress_callback,
    vlm_prompt_callback,
    vlm_stream_callback,
    vlm_reasoning_callback,
    step_event_callback,
) -> AgentCallbacks:
    return _CallbackBag(
        progress=progress_callback,
        vlm_prompt=vlm_prompt_callback,
        vlm_stream=vlm_stream_callback,
        vlm_reasoning=vlm_reasoning_callback,
        step_event=step_event_callback,
    )


# ---------------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------------


def run_annotation(
    *,
    flow: Flow,
    df,
    config: Optional[AnnotationPipelineConfig] = None,
    progress_callback: Optional[Callable] = None,
    vlm_prompt_callback: Optional[Callable] = None,
    vlm_stream_callback: Optional[Callable] = None,
    vlm_reasoning_callback: Optional[Callable] = None,
    step_event_callback: Optional[Callable] = None,
    session_id: str = "",
    # detailed-flow inputs
    start_index: Optional[int] = None,
    end_index: Optional[int] = None,
    parent_main_labels: Optional[List[str]] = None,
    existing_children: Optional[List[dict]] = None,
    # lap-flow inputs
    lap_start: Optional[int] = None,
    lap_end: Optional[int] = None,
    section_id: Optional[str] = None,
    section_start: Optional[int] = None,
    section_end: Optional[int] = None,
    circuit_id: Optional[str] = None,
    existing_section_annotations: Optional[List[dict]] = None,
) -> Union[AnnotationResult, LapAnnotationResult]:
    """Run one annotation across the chosen flow and backend.

    The dispatch is data: pick the flow module, build the request, run
    the agent, parse the response. No conditional logic about backends
    here — that's resolved inside the agent runner.
    """
    config = config or AnnotationPipelineConfig()
    callbacks = _bag_from_kwargs(
        progress_callback,
        vlm_prompt_callback,
        vlm_stream_callback,
        vlm_reasoning_callback,
        step_event_callback,
    )
    backend_config = config.to_backend_config()

    if flow == "detailed":
        return _run_detailed(
            backend=config.backend,
            df=df,
            parent_start=_require(start_index, "start_index"),
            parent_end=_require(end_index, "end_index"),
            parent_main_labels=list(parent_main_labels or []),
            existing_children=list(existing_children or []),
            backend_config=backend_config,
            callbacks=callbacks,
            session_id=session_id,
        )
    if flow == "lap":
        return _run_lap(
            backend=config.backend,
            df=df,
            lap_start=_require(lap_start, "lap_start"),
            lap_end=_require(lap_end, "lap_end"),
            section_id=_require(section_id, "section_id"),
            section_start=_require(section_start, "section_start"),
            section_end=_require(section_end, "section_end"),
            circuit_id=_require(circuit_id, "circuit_id"),
            existing_section_annotations=list(existing_section_annotations or []),
            backend_config=backend_config,
            callbacks=callbacks,
            session_id=session_id,
        )
    raise ValueError(f"unknown flow {flow!r}; expected 'detailed' or 'lap'")


def _run_detailed(
    *,
    backend: Backend,
    df,
    parent_start: int,
    parent_end: int,
    parent_main_labels: List[str],
    existing_children: List[dict],
    backend_config: BackendConfig,
    callbacks: AgentCallbacks,
    session_id: str,
) -> AnnotationResult:
    request = detailed_flow.build_request(
        backend=backend,
        df=df,
        parent_start=parent_start,
        parent_end=parent_end,
        parent_main_labels=parent_main_labels,
        existing_children=existing_children,
        config=backend_config,
        callbacks=callbacks,
        session_id=session_id,
    )
    response = run_agent(request)
    return detailed_flow.parse(
        response,
        backend=backend,
        parent_start=parent_start,
        parent_end=parent_end,
    )


def _run_lap(
    *,
    backend: Backend,
    df,
    lap_start: int,
    lap_end: int,
    section_id: str,
    section_start: int,
    section_end: int,
    circuit_id: str,
    existing_section_annotations: List[dict],
    backend_config: BackendConfig,
    callbacks: AgentCallbacks,
    session_id: str,
) -> LapAnnotationResult:
    request = lap_flow.build_request(
        backend=backend,
        df=df,
        lap_start=lap_start,
        lap_end=lap_end,
        section_id=section_id,
        section_start=section_start,
        section_end=section_end,
        circuit_id=circuit_id,
        existing_section_annotations=existing_section_annotations,
        config=backend_config,
        callbacks=callbacks,
        session_id=session_id,
    )
    response = run_agent(request)
    # The LLM picks the circuit + circuit_section labels itself (via the
    # get_circuit_id / locate_circuit_section tools), so result.label_ids
    # already carries them — no deterministic post-merge here.
    return lap_flow.parse(
        response,
        backend=backend,
        lap_start=lap_start,
        lap_end=lap_end,
        section_id=section_id,
        section_start=section_start,
        section_end=section_end,
    )


def _require(value, name: str):
    if value is None:
        raise ValueError(f"run_annotation: required argument '{name}' is missing")
    return value


__all__ = [
    "AnnotationPipelineConfig",
    "AnnotationResult",
    "LapAnnotationResult",
    "run_annotation",
    "run_claude_followup",
]
