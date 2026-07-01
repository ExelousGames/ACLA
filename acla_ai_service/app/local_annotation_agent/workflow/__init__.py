"""
Public annotation pipeline entry — the only function the UI calls.

    from app.local_annotation_agent.workflow import (
        run_annotation,
        AnnotationPipelineConfig,
        AnnotationResult,
        LapAnnotationResult,
    )

    config = AnnotationPipelineConfig(provider_id="claude_cli", ...)

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
    3. ``run_agent(request)`` dispatches through the annotation provider registry.
    4. ``flow.parse(response, ...)`` decodes raw text into a typed result.

Each layer has one job. The agent box never sees racing types; the flows
never see runners; the UI never sees the box.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Literal, Optional, Union

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
    NoopCallbacks,
    ProviderConfig,
)
from app.annotation_providers.registry import get_annotation_provider
from app.local_annotation_agent.workflow.flows import detailed as detailed_flow
from app.local_annotation_agent.workflow.flows import lap as lap_flow
from app.local_annotation_agent.workflow.followup import run_claude_followup
from app.local_annotation_agent.workflow.results import (
    AnnotationResult,
    LapAnnotationResult,
)

LOGGER = logging.getLogger(__name__)


Flow = Literal["detailed", "lap"]
@dataclass
class AnnotationPipelineConfig:
    """Provider-neutral config the annotation UI/API passes."""

    provider_id: str = "claude_cli"
    model: str = ""
    max_iterations: int = 3
    max_new_tokens: int = 1500
    temperature: float = 0.7
    provider_options: Dict[str, Any] = field(default_factory=dict)

    def to_provider_config(self) -> ProviderConfig:
        provider = get_annotation_provider(self.provider_id)
        model = self.model or provider.default_model_id()
        options = dict(provider.option_defaults())
        options.update(self.provider_options or {})
        options.setdefault("max_turns", int(self.max_iterations) * 10)
        return ProviderConfig(
            provider_id=self.provider_id,
            model=model,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            provider_options=options,
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


def _bounded_df(df, start: int, end: int):
    """Return a DataFrame exposing only [start, end) telemetry rows.

    Tools use absolute iloc coordinates, so preserve those coordinates on the
    dataframe index while exposing only the requested working window.
    """
    s = int(start)
    e = int(end)
    if e <= s:
        return df.iloc[0:0].copy()
    return df.loc[(df.index >= s) & (df.index < e)].copy()


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
    section_split_basis: Optional[str] = None,
    opponent_interaction: Optional[dict] = None,
    existing_section_annotations: Optional[List[dict]] = None,
) -> Union[AnnotationResult, LapAnnotationResult]:
    """Run one annotation across the chosen flow and annotation provider.

    The dispatch is data: pick the flow module, build the request, run
    the agent, parse the response. No provider-specific logic
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
    provider = get_annotation_provider(config.provider_id)
    provider_config = config.to_provider_config()

    if flow == "detailed":
        parent_start = _require(start_index, "start_index")
        parent_end = _require(end_index, "end_index")
        return _run_detailed(
            provider_id=config.provider_id,
            prompt_mode=provider.prompt_mode,
            df=_bounded_df(df, parent_start, parent_end),
            parent_start=parent_start,
            parent_end=parent_end,
            parent_main_labels=list(parent_main_labels or []),
            existing_children=list(existing_children or []),
            provider_config=provider_config,
            callbacks=callbacks,
            session_id=session_id,
        )
    if flow == "lap":
        required_lap_start = _require(lap_start, "lap_start")
        required_lap_end = _require(lap_end, "lap_end")
        required_section_start = _require(section_start, "section_start")
        required_section_end = _require(section_end, "section_end")
        return _run_lap(
            provider_id=config.provider_id,
            prompt_mode=provider.prompt_mode,
            df=_bounded_df(df, required_section_start, required_section_end),
            lap_start=required_lap_start,
            lap_end=required_lap_end,
            section_id=_require(section_id, "section_id"),
            section_start=required_section_start,
            section_end=required_section_end,
            circuit_id=_require(circuit_id, "circuit_id"),
            section_split_basis=section_split_basis,
            opponent_interaction=opponent_interaction,
            existing_section_annotations=list(existing_section_annotations or []),
            provider_config=provider_config,
            callbacks=callbacks,
            session_id=session_id,
        )
    raise ValueError(f"unknown flow {flow!r}; expected 'detailed' or 'lap'")


def _run_detailed(
    *,
    provider_id: str,
    prompt_mode: str,
    df,
    parent_start: int,
    parent_end: int,
    parent_main_labels: List[str],
    existing_children: List[dict],
    provider_config: ProviderConfig,
    callbacks: AgentCallbacks,
    session_id: str,
) -> AnnotationResult:
    request = detailed_flow.build_request(
        provider_id=provider_id,
        prompt_mode=prompt_mode,
        df=df,
        parent_start=parent_start,
        parent_end=parent_end,
        parent_main_labels=parent_main_labels,
        existing_children=existing_children,
        config=provider_config,
        callbacks=callbacks,
        session_id=session_id,
    )
    response = run_agent(request)
    return detailed_flow.parse(
        response,
        prompt_mode=prompt_mode,
        parent_start=parent_start,
        parent_end=parent_end,
    )


def _run_lap(
    *,
    provider_id: str,
    prompt_mode: str,
    df,
    lap_start: int,
    lap_end: int,
    section_id: str,
    section_start: int,
    section_end: int,
    circuit_id: str,
    section_split_basis: Optional[str],
    opponent_interaction: Optional[dict],
    existing_section_annotations: List[dict],
    provider_config: ProviderConfig,
    callbacks: AgentCallbacks,
    session_id: str,
) -> LapAnnotationResult:
    request = lap_flow.build_request(
        provider_id=provider_id,
        prompt_mode=prompt_mode,
        df=df,
        lap_start=lap_start,
        lap_end=lap_end,
        section_id=section_id,
        section_start=section_start,
        section_end=section_end,
        circuit_id=circuit_id,
        section_split_basis=section_split_basis,
        opponent_interaction=opponent_interaction,
        existing_section_annotations=existing_section_annotations,
        config=provider_config,
        callbacks=callbacks,
        session_id=session_id,
    )
    response = run_agent(request)
    # The LLM picks the circuit + circuit_section labels itself (via the
    # get_circuit_id / locate_circuit_section tools), so result.label_ids
    # already carries them — no deterministic post-merge here.
    return lap_flow.parse(
        response,
        prompt_mode=prompt_mode,
        lap_start=lap_start,
        lap_end=lap_end,
        section_id=section_id,
        section_start=section_start,
        section_end=section_end,
        circuit_id=circuit_id,
        section_split_basis=section_split_basis,
        opponent_interaction=opponent_interaction,
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
